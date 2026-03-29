#!/usr/bin/env python3
"""Offline local CPU training for copy-move detection/segmentation.

Uses only local synthetic data and caps CPU usage by thread count.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def set_limits(threads: int) -> None:
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(threads)
    cv2.setNumThreads(max(1, threads))
    torch.set_num_threads(max(1, threads))
    torch.set_num_interop_threads(max(1, min(4, threads)))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_bg(h: int, w: int) -> np.ndarray:
    coarse = np.random.rand(24, 24).astype(np.float32)
    img = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_CUBIC)
    img = cv2.GaussianBlur(img, (0, 0), 1.0)
    for _ in range(random.randint(22, 40)):
        cy = random.randint(0, h - 1)
        cx = random.randint(0, w - 1)
        r = random.randint(4, 15)
        amp = random.uniform(-0.18, 0.2)
        yy, xx = np.ogrid[:h, :w]
        d2 = (yy - cy) ** 2 + (xx - cx) ** 2
        blob = np.exp(-d2 / max(1.0, float(2 * r * r)))
        img += amp * blob.astype(np.float32)
    img += 0.02 * np.random.randn(h, w).astype(np.float32)
    return np.clip(img, 0.0, 1.0)


def inject_hard_negative(img: np.ndarray) -> np.ndarray:
    h, w = img.shape
    out = img.copy()
    for _ in range(random.randint(2, 4)):
        ph = random.randint(18, 42)
        pw = random.randint(18, 42)
        y1 = random.randint(0, h - ph)
        x1 = random.randint(0, w - pw)
        y2 = random.randint(0, h - ph)
        x2 = random.randint(0, w - pw)

        yy, xx = np.ogrid[:ph, :pw]
        cy = random.uniform(ph * 0.35, ph * 0.65)
        cx = random.uniform(pw * 0.35, pw * 0.65)
        r1 = random.uniform(min(ph, pw) * 0.15, min(ph, pw) * 0.35)
        r2 = random.uniform(min(ph, pw) * 0.15, min(ph, pw) * 0.35)

        blob1 = np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * r1 * r1))).astype(
            np.float32
        )
        blob2 = np.exp(
            -(
                ((yy - (cy + random.uniform(-2.0, 2.0))) ** 2)
                + ((xx - (cx + random.uniform(-2.0, 2.0))) ** 2)
            )
            / (2.0 * r2 * r2)
        ).astype(np.float32)

        amp1 = random.uniform(-0.17, 0.17)
        amp2 = amp1 + random.uniform(-0.025, 0.025)

        out[y1 : y1 + ph, x1 : x1 + pw] = np.clip(
            out[y1 : y1 + ph, x1 : x1 + pw] + amp1 * blob1, 0.0, 1.0
        )
        out[y2 : y2 + ph, x2 : x2 + pw] = np.clip(
            out[y2 : y2 + ph, x2 : x2 + pw] + amp2 * blob2, 0.0, 1.0
        )
    return out


def forge(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h, w = img.shape
    ph = random.randint(18, 48)
    pw = random.randint(18, 48)
    y = random.randint(0, h - ph)
    x = random.randint(0, w - pw)

    for _ in range(120):
        dy = random.randint(0, h - ph)
        dx = random.randint(0, w - pw)
        if abs(dy - y) + abs(dx - x) > min(h, w) // 3:
            break

    p = img[y : y + ph, x : x + pw].copy()
    gain = random.uniform(0.96, 1.04)
    bias = random.uniform(-0.025, 0.025)
    p = np.clip(gain * p + bias, 0.0, 1.0)
    if random.random() < 0.35:
        p = cv2.GaussianBlur(p, (0, 0), random.uniform(0.2, 0.6))

    out = img.copy()
    a = random.uniform(0.88, 1.0)
    out[dy : dy + ph, dx : dx + pw] = np.clip(
        a * p + (1.0 - a) * out[dy : dy + ph, dx : dx + pw], 0.0, 1.0
    )
    m = np.zeros((h, w), dtype=np.uint8)
    m[y : y + ph, x : x + pw] = 1
    m[dy : dy + ph, dx : dx + pw] = 1
    return out, m


def generate_dataset(
    img_dir: str,
    mask_dir: str,
    n: int,
    size: int,
    forged_ratio: float,
    hard_negative_ratio: float,
    seed: int,
) -> None:
    set_seed(seed)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    ids = list(range(1, n + 1))
    random.shuffle(ids)
    forged = set(ids[: int(round(n * forged_ratio))])

    for i in range(1, n + 1):
        img = make_bg(size, size)
        mask = np.zeros((size, size), dtype=np.uint8)
        if i in forged:
            img, mask = forge(img)
        else:
            if random.random() < hard_negative_ratio:
                img = inject_hard_negative(img)
        cv2.imwrite(os.path.join(img_dir, f"{i}.png"), (img * 255).astype(np.uint8))
        np.save(os.path.join(mask_dir, f"{i}.npy"), mask)


class PairDataset(Dataset):
    def __init__(
        self, ids: List[str], img_dir: str, mask_dir: str, size: int, train: bool
    ):
        self.ids = ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.size = size
        self.train = train

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        sid = self.ids[idx]
        g = cv2.imread(os.path.join(self.img_dir, sid + ".png"), cv2.IMREAD_GRAYSCALE)
        m = np.load(os.path.join(self.mask_dir, sid + ".npy")).astype(np.uint8)

        if g.shape[0] != self.size or g.shape[1] != self.size:
            g = cv2.resize(g, (self.size, self.size), interpolation=cv2.INTER_AREA)
            m = cv2.resize(m, (self.size, self.size), interpolation=cv2.INTER_NEAREST)

        if self.train:
            if random.random() < 0.5:
                g = np.fliplr(g).copy()
                m = np.fliplr(m).copy()
            if random.random() < 0.5:
                g = np.flipud(g).copy()
                m = np.flipud(m).copy()
            if random.random() < 0.3:
                k = random.choice([3, 5])
                g = cv2.GaussianBlur(g, (k, k), 0)
            if random.random() < 0.25:
                noise = np.random.normal(0, 3.5, g.shape).astype(np.float32)
                g = np.clip(g.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        x = torch.from_numpy(g.astype(np.float32) / 255.0).unsqueeze(0)
        y = torch.from_numpy((m > 0).astype(np.float32)).unsqueeze(0)
        c = torch.tensor([1.0 if y.sum() > 0 else 0.0], dtype=torch.float32)
        return x, y, c


class CachedDataset(Dataset):
    def __init__(self, base: Dataset):
        self.cache = [base[i] for i in range(len(base))]

    def __len__(self) -> int:
        return len(self.cache)

    def __getitem__(self, idx: int):
        return self.cache[idx]


class AuxClsDataset(Dataset):
    def __init__(self, image_paths: List[str], labels: List[int], size: int):
        self.image_paths = image_paths
        self.labels = labels
        self.size = size

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        p = self.image_paths[idx]
        y = self.labels[idx]
        g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if g is None:
            g = np.zeros((self.size, self.size), dtype=np.uint8)
        if g.shape[0] != self.size or g.shape[1] != self.size:
            g = cv2.resize(g, (self.size, self.size), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(g.astype(np.float32) / 255.0).unsqueeze(0)
        cls = torch.tensor([float(y)], dtype=torch.float32)
        dummy_mask = torch.zeros((1, self.size, self.size), dtype=torch.float32)
        return x, dummy_mask, cls


class ConvBlock(nn.Module):
    def __init__(self, ci: int, co: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ci, co, 3, padding=1),
            nn.BatchNorm2d(co),
            nn.ReLU(inplace=True),
            nn.Conv2d(co, co, 3, padding=1),
            nn.BatchNorm2d(co),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TinyUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.e1 = ConvBlock(1, 24)
        self.e2 = ConvBlock(24, 48)
        self.e3 = ConvBlock(48, 96)
        self.pool = nn.MaxPool2d(2)
        self.b = ConvBlock(96, 128)
        self.u3 = nn.ConvTranspose2d(128, 96, 2, stride=2)
        self.d3 = ConvBlock(192, 96)
        self.u2 = nn.ConvTranspose2d(96, 48, 2, stride=2)
        self.d2 = ConvBlock(96, 48)
        self.u1 = nn.ConvTranspose2d(48, 24, 2, stride=2)
        self.d1 = ConvBlock(48, 24)
        self.mask_head = nn.Conv2d(24, 1, 1)
        self.aux_mask_head = nn.Conv2d(48, 1, 1)
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, 1)
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        b = self.b(self.pool(e3))
        cls = self.cls_head(b)
        d3 = self.d3(torch.cat([self.u3(b), e3], dim=1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], dim=1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], dim=1))
        mask = self.mask_head(d1)
        aux = self.aux_mask_head(d2)
        aux = F.interpolate(
            aux, size=mask.shape[-2:], mode="bilinear", align_corners=False
        )
        return mask, aux, cls


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    p = torch.sigmoid(logits)
    num = 2.0 * (p * target).sum(dim=(1, 2, 3)) + 1e-6
    den = (p + target).sum(dim=(1, 2, 3)) + 1e-6
    return 1.0 - (num / den).mean()


def mask_f1_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = (y_true > 0).astype(np.uint8)
    y_pred = (y_pred > 0).astype(np.uint8)
    tp = int((y_true * y_pred).sum())
    fp = int(((1 - y_true) * y_pred).sum())
    fn = int((y_true * (1 - y_pred)).sum())
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0
    if tp == 0:
        return 0.0
    return 2.0 * tp / float(2 * tp + fp + fn)


def calibrate_thresholds(
    model: nn.Module,
    va_dl: DataLoader,
    device: torch.device,
    tta_modes: List[int],
) -> Tuple[float, float, float, float]:
    m_true = []
    m_prob = []
    c_true = []
    c_prob = []

    with torch.no_grad():
        for x, y, c in va_dl:
            x = x.to(device)
            y = y.to(device)
            c = c.to(device)

            mlog_acc = None
            cacc = 0.0
            for m in tta_modes:
                xx = _apply_flip(x, m)
                mm, _aux, cc = model(xx)
                mm = _apply_flip(mm, m)
                mlog_acc = mm if mlog_acc is None else (mlog_acc + mm)
                cacc += torch.sigmoid(cc)

            mlog = mlog_acc / float(len(tta_modes))
            cpr = cacc / float(len(tta_modes))
            mpr = torch.sigmoid(mlog)

            m_true.append(y.cpu().numpy())
            m_prob.append(mpr.cpu().numpy())
            c_true.append(c.cpu().numpy())
            c_prob.append(cpr.cpu().numpy())

    y_true = np.concatenate(m_true, axis=0)
    y_prob = np.concatenate(m_prob, axis=0)
    c_t = np.concatenate(c_true, axis=0).reshape(-1)
    c_p = np.concatenate(c_prob, axis=0).reshape(-1)

    mask_ths = np.linspace(0.35, 0.75, 17)
    cls_ths = np.linspace(0.30, 0.70, 17)
    best = (-1.0, 0.5, 0.5, 0.0)

    for mt in mask_ths:
        pm = (y_prob > mt).astype(np.uint8)
        area = pm.reshape(pm.shape[0], -1).mean(axis=1)
        for ct in cls_ths:
            c_pred = (c_p >= ct).astype(np.uint8)
            # small area suppresses false positives
            gated = ((c_pred == 1) & (area >= 0.002)).astype(np.uint8)

            f1s = []
            for i in range(pm.shape[0]):
                pred_mask = pm[i, 0] if gated[i] == 1 else np.zeros_like(pm[i, 0])
                f1s.append(mask_f1_np(y_true[i, 0], pred_mask))
            mean_f1 = float(np.mean(f1s))

            if mean_f1 > best[0]:
                best = (
                    mean_f1,
                    float(mt),
                    float(ct),
                    float(np.mean(area[gated == 1]) if gated.sum() > 0 else 0.0),
                )

    return best[0], best[1], best[2], best[3]


@dataclass
class TrainCfg:
    img_size: int = 128
    batch_size: int = 12
    epochs: int = 16
    lr: float = 1e-3
    workers: int = 4
    threads: int = 8
    seed: int = 42
    patience: int = 6
    min_delta: float = 0.001
    cached: bool = True


def collect_ids(img_dir: str, mask_dir: str) -> List[str]:
    ids = []
    for n in sorted(os.listdir(img_dir)):
        if not n.endswith(".png"):
            continue
        sid = os.path.splitext(n)[0]
        if os.path.exists(os.path.join(mask_dir, sid + ".npy")):
            ids.append(sid)
    return ids


def collect_aux_hf_data(size: int) -> Tuple[List[str], List[int]]:
    root = os.path.join("hf_data", "lo1206__tampered_image")
    out_paths: List[str] = []
    out_labels: List[int] = []
    if not os.path.isdir(root):
        return out_paths, out_labels

    for split in ["train", "test"]:
        d = os.path.join(root, split)
        if not os.path.isdir(d):
            continue
        for n in sorted(os.listdir(d)):
            if not n.lower().endswith(".jpg"):
                continue
            p = os.path.join(d, n)
            nl = n.lower()
            label = 0 if nl == "id.jpg" else 1
            out_paths.append(p)
            out_labels.append(label)
    return out_paths, out_labels


def reinforcement_finetune_cls(
    model: nn.Module,
    cfg: TrainCfg,
    aux_paths: List[str],
    aux_labels: List[int],
) -> None:
    if not aux_paths:
        return

    ds = AuxClsDataset(aux_paths, aux_labels, cfg.img_size)
    dl = DataLoader(ds, batch_size=max(8, cfg.batch_size), shuffle=True, num_workers=0)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr * 0.4, weight_decay=1e-4)

    for ep in range(3):
        losses = []
        for x, _dummy, c in dl:
            mlog, auxlog, clog = model(x)
            p = torch.sigmoid(mlog).detach()
            uncertainty = torch.mean(4.0 * p * (1.0 - p), dim=(1, 2, 3), keepdim=True)
            reward = 1.0 + 0.5 * uncertainty
            bce = F.binary_cross_entropy_with_logits(clog, c, reduction="none")
            loss_cls = (bce * reward).mean()
            # keep decoder stable
            reg = 0.02 * (torch.mean(torch.abs(mlog)) + torch.mean(torch.abs(auxlog)))
            loss = loss_cls + reg
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        print(f"rl_finetune_epoch={ep + 1} loss={float(np.mean(losses)):.4f}")


def train_local(img_dir: str, mask_dir: str, cfg: TrainCfg, model_out: str) -> None:
    set_limits(cfg.threads)
    set_seed(cfg.seed)

    ids = collect_ids(img_dir, mask_dir)
    rng = np.random.default_rng(cfg.seed)
    idx = np.arange(len(ids))
    rng.shuffle(idx)
    ids = [ids[i] for i in idx]
    split = int(0.8 * len(ids))
    tr_ids = ids[:split]
    va_ids = ids[split:]

    tr_ds = PairDataset(tr_ids, img_dir, mask_dir, size=cfg.img_size, train=True)
    va_ds = PairDataset(va_ids, img_dir, mask_dir, size=cfg.img_size, train=False)
    if cfg.cached:
        tr_ds = CachedDataset(tr_ds)
        va_ds = CachedDataset(va_ds)
    tr_dl = DataLoader(
        tr_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=(0 if cfg.cached else cfg.workers),
    )
    va_dl = DataLoader(
        va_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=(0 if cfg.cached else cfg.workers),
    )

    device = torch.device("cpu")
    model = TinyUNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(2, cfg.epochs), eta_min=cfg.lr * 0.1
    )

    best = {"val_cls_acc": 0.0, "val_mask_f1": 0.0, "epoch": -1}
    best_state = None
    best_score = -1e9
    wait = 0

    best_mask_th = 0.5
    best_cls_th = 0.5
    best_area_hint = 0.0

    for ep in range(1, cfg.epochs + 1):
        model.train()
        tr_loss = 0.0
        for x, y, c in tr_dl:
            x = x.to(device)
            y = y.to(device)
            c = c.to(device)
            mlog, auxlog, clog = model(x)
            l_mask = F.binary_cross_entropy_with_logits(mlog, y) + dice_loss(mlog, y)
            l_aux = F.binary_cross_entropy_with_logits(auxlog, y) + dice_loss(auxlog, y)
            l_cls = F.binary_cross_entropy_with_logits(clog, c)
            loss = 0.65 * l_mask + 0.20 * l_aux + 0.15 * l_cls
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tr_loss += float(loss.item())

        model.eval()
        cls_true = []
        cls_pred = []
        f1s = []
        with torch.no_grad():
            for x, y, c in va_dl:
                x = x.to(device)
                y = y.to(device)
                c = c.to(device)
                mlog, _aux, clog = model(x)
                mp = (torch.sigmoid(mlog) > 0.5).float()
                cp = (torch.sigmoid(clog) > 0.5).float()
                cls_true.extend(c.cpu().numpy().reshape(-1).tolist())
                cls_pred.extend(cp.cpu().numpy().reshape(-1).tolist())
                y_np = y.cpu().numpy()
                p_np = mp.cpu().numpy()
                for i in range(y_np.shape[0]):
                    f1s.append(mask_f1_np(y_np[i, 0], p_np[i, 0]))

        cls_true_arr = np.asarray(cls_true)
        cls_pred_arr = np.asarray(cls_pred)
        cls_acc = float(np.mean(cls_true_arr == cls_pred_arr))
        mf1 = float(np.mean(f1s)) if f1s else 0.0
        tr_loss /= max(1, len(tr_dl))
        lr_now = opt.param_groups[0]["lr"]
        print(
            f"epoch={ep:02d} lr={lr_now:.6f} train_loss={tr_loss:.4f} "
            f"val_cls_acc={cls_acc:.4f} val_mask_f1={mf1:.4f}"
        )

        score = mf1 + 0.25 * cls_acc
        if score > (best_score + cfg.min_delta):
            best = {"val_cls_acc": cls_acc, "val_mask_f1": mf1, "epoch": ep}
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            best_score = score
            wait = 0

            calib_f1, calib_mt, calib_ct, area_hint = calibrate_thresholds(
                model, va_dl, device, tta_modes=[0, 1, 2, 3]
            )
            best_mask_th = calib_mt
            best_cls_th = calib_ct
            best_area_hint = area_hint
            print(
                f"calib_best_f1={calib_f1:.4f} calib_mask_th={calib_mt:.3f} "
                f"calib_cls_th={calib_ct:.3f}"
            )
        else:
            wait += 1

        scheduler.step()
        if wait >= cfg.patience:
            print(f"early_stopping_triggered epoch={ep} patience={cfg.patience}")
            break

    if best_state is None:
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    aux_paths, aux_labels = collect_aux_hf_data(cfg.img_size)
    if aux_paths:
        print(f"aux_hf_samples={len(aux_paths)} starting_rl_finetune")
        reinforcement_finetune_cls(model, cfg, aux_paths, aux_labels)
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    ckpt = {
        "state_dict": best_state,
        "best": best,
        "cfg": asdict(cfg),
        "img_size": cfg.img_size,
        "mask_threshold": best_mask_th,
        "cls_threshold": best_cls_th,
        "area_hint": best_area_hint,
    }
    torch.save(ckpt, model_out)
    print(f"threads_used={cfg.threads}")
    print(
        f"best_epoch={best['epoch']} best_val_cls_acc={best['val_cls_acc']:.6f} best_val_mask_f1={best['val_mask_f1']:.6f}"
    )
    print(f"saved_model={model_out}")


def rle_encode(mask: np.ndarray) -> str:
    flat = mask.reshape(-1).astype(np.uint8)
    idx = np.where(flat > 0)[0]
    if idx.size == 0:
        return "authentic"
    runs: List[int] = []
    s = int(idx[0])
    p = s
    l = 1
    for cur in idx[1:]:
        cur = int(cur)
        if cur == p + 1:
            l += 1
        else:
            runs.extend([s, l])
            s = cur
            l = 1
        p = cur
    runs.extend([s, l])
    return str(runs)


def _apply_flip(x: torch.Tensor, mode: int) -> torch.Tensor:
    if mode == 0:
        return x
    if mode == 1:
        return torch.flip(x, dims=[3])
    if mode == 2:
        return torch.flip(x, dims=[2])
    if mode == 3:
        return torch.flip(x, dims=[2, 3])
    raise ValueError(mode)


def predict_csv(
    model_path: str, input_dir: str, output_csv: str, threads: int, tta: int
) -> None:
    set_limits(threads)
    ck = torch.load(model_path, map_location="cpu")
    model = TinyUNet().cpu()
    model.load_state_dict(ck["state_dict"])
    model.eval()
    img_size = int(ck.get("img_size", 128))
    mask_th = float(ck.get("mask_threshold", 0.5))
    cls_th = float(ck.get("cls_threshold", 0.5))

    files = [p for p in sorted(os.listdir(input_dir)) if p.lower().endswith(".png")]
    rows = []
    auth = 0
    forg = 0
    with torch.no_grad():
        for n in files:
            sid = os.path.splitext(n)[0]
            g = cv2.imread(os.path.join(input_dir, n), cv2.IMREAD_GRAYSCALE)
            h, w = g.shape
            gr = cv2.resize(g, (img_size, img_size), interpolation=cv2.INTER_AREA)
            x = (
                torch.from_numpy(gr.astype(np.float32) / 255.0)
                .unsqueeze(0)
                .unsqueeze(0)
            )
            modes = [0]
            if tta >= 2:
                modes = [0, 1]
            if tta >= 4:
                modes = [0, 1, 2, 3]

            mlog_acc = None
            clog_acc = 0.0
            for m in modes:
                xx = _apply_flip(x, m)
                mm, _aux, cc = model(xx)
                mm = _apply_flip(mm, m)
                mlog_acc = mm if mlog_acc is None else (mlog_acc + mm)
                clog_acc += float(torch.sigmoid(cc).item())

            mlog_acc = mlog_acc / float(len(modes))
            cls_prob = clog_acc / float(len(modes))
            pm = (torch.sigmoid(mlog_acc)[0, 0].numpy() > mask_th).astype(np.uint8)
            if pm.shape != (h, w):
                pm = cv2.resize(pm, (w, h), interpolation=cv2.INTER_NEAREST)
                pm = (pm > 0).astype(np.uint8)

            if cls_prob < cls_th or pm.sum() == 0:
                ann = "authentic"
                auth += 1
            else:
                ann = rle_encode(pm)
                forg += 1
            rows.append((sid, ann))

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        import csv

        wr = csv.writer(f)
        wr.writerow(["case_id", "annotation"])
        wr.writerows(rows)
    print(f"threads_used={threads}")
    print(f"predicted={len(rows)} authentic={auth} forged={forg}")
    print(f"saved={output_csv}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)

    g = sub.add_parser("generate")
    g.add_argument("--img-dir", default="deep_mock/images")
    g.add_argument("--mask-dir", default="deep_mock/masks")
    g.add_argument("--n", type=int, default=360)
    g.add_argument("--size", type=int, default=128)
    g.add_argument("--forged-ratio", type=float, default=0.56)
    g.add_argument("--hard-negative-ratio", type=float, default=0.35)
    g.add_argument("--seed", type=int, default=42)

    t = sub.add_parser("train")
    t.add_argument("--img-dir", default="deep_mock/images")
    t.add_argument("--mask-dir", default="deep_mock/masks")
    t.add_argument("--model-out", default="deep_local_model.pt")
    t.add_argument("--img-size", type=int, default=128)
    t.add_argument("--batch-size", type=int, default=12)
    t.add_argument("--epochs", type=int, default=16)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--workers", type=int, default=4)
    t.add_argument("--threads", type=int, default=8)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--patience", type=int, default=6)
    t.add_argument("--min-delta", type=float, default=0.001)
    t.add_argument("--cached", action="store_true")

    p = sub.add_parser("predict")
    p.add_argument("--model", default="deep_local_model.pt")
    p.add_argument("--input-dir", default="deep_mock/images")
    p.add_argument("--output-csv", default="submission.csv")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--tta", type=int, default=4, choices=[1, 2, 4])

    args = ap.parse_args()

    if args.mode == "generate":
        generate_dataset(
            args.img_dir,
            args.mask_dir,
            args.n,
            args.size,
            args.forged_ratio,
            args.hard_negative_ratio,
            args.seed,
        )
        print(f"generated n={args.n} at {args.img_dir} with masks {args.mask_dir}")
    elif args.mode == "train":
        cfg = TrainCfg(
            img_size=args.img_size,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            workers=args.workers,
            threads=args.threads,
            seed=args.seed,
            patience=args.patience,
            min_delta=args.min_delta,
            cached=args.cached,
        )
        train_local(args.img_dir, args.mask_dir, cfg, args.model_out)
    elif args.mode == "predict":
        predict_csv(args.model, args.input_dir, args.output_csv, args.threads, args.tta)
    else:
        raise ValueError(args.mode)


if __name__ == "__main__":
    main()
