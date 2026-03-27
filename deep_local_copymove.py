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
from typing import List, Tuple

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
    img_dir: str, mask_dir: str, n: int, size: int, forged_ratio: float, seed: int
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
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, 1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        b = self.b(self.pool(e3))
        cls = self.cls_head(b)
        d3 = self.d3(torch.cat([self.u3(b), e3], dim=1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], dim=1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], dim=1))
        mask = self.mask_head(d1)
        return mask, cls


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


@dataclass
class TrainCfg:
    img_size: int = 128
    batch_size: int = 12
    epochs: int = 16
    lr: float = 1e-3
    workers: int = 4
    threads: int = 8
    seed: int = 42


def collect_ids(img_dir: str, mask_dir: str) -> List[str]:
    ids = []
    for n in sorted(os.listdir(img_dir)):
        if not n.endswith(".png"):
            continue
        sid = os.path.splitext(n)[0]
        if os.path.exists(os.path.join(mask_dir, sid + ".npy")):
            ids.append(sid)
    return ids


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
    tr_dl = DataLoader(
        tr_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.workers
    )
    va_dl = DataLoader(
        va_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.workers
    )

    device = torch.device("cpu")
    model = TinyUNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)

    best = {"val_cls_acc": 0.0, "val_mask_f1": 0.0, "epoch": -1}
    best_state = None

    for ep in range(1, cfg.epochs + 1):
        model.train()
        tr_loss = 0.0
        for x, y, c in tr_dl:
            x = x.to(device)
            y = y.to(device)
            c = c.to(device)
            mlog, clog = model(x)
            l_mask = F.binary_cross_entropy_with_logits(mlog, y) + dice_loss(mlog, y)
            l_cls = F.binary_cross_entropy_with_logits(clog, c)
            loss = 0.8 * l_mask + 0.2 * l_cls
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
                mlog, clog = model(x)
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
        print(
            f"epoch={ep:02d} train_loss={tr_loss:.4f} val_cls_acc={cls_acc:.4f} val_mask_f1={mf1:.4f}"
        )

        if (mf1 + 0.3 * cls_acc) > (best["val_mask_f1"] + 0.3 * best["val_cls_acc"]):
            best = {"val_cls_acc": cls_acc, "val_mask_f1": mf1, "epoch": ep}
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state is None:
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    ckpt = {
        "state_dict": best_state,
        "best": best,
        "cfg": asdict(cfg),
        "img_size": cfg.img_size,
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


def predict_csv(model_path: str, input_dir: str, output_csv: str, threads: int) -> None:
    set_limits(threads)
    ck = torch.load(model_path, map_location="cpu")
    model = TinyUNet().cpu()
    model.load_state_dict(ck["state_dict"])
    model.eval()
    img_size = int(ck.get("img_size", 128))

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
            mlog, clog = model(x)
            cls_prob = float(torch.sigmoid(clog).item())
            pm = (torch.sigmoid(mlog)[0, 0].numpy() > 0.5).astype(np.uint8)
            if pm.shape != (h, w):
                pm = cv2.resize(pm, (w, h), interpolation=cv2.INTER_NEAREST)
                pm = (pm > 0).astype(np.uint8)

            if cls_prob < 0.5 or pm.sum() == 0:
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

    p = sub.add_parser("predict")
    p.add_argument("--model", default="deep_local_model.pt")
    p.add_argument("--input-dir", default="deep_mock/images")
    p.add_argument("--output-csv", default="submission.csv")
    p.add_argument("--threads", type=int, default=8)

    args = ap.parse_args()

    if args.mode == "generate":
        generate_dataset(
            args.img_dir, args.mask_dir, args.n, args.size, args.forged_ratio, args.seed
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
        )
        train_local(args.img_dir, args.mask_dir, cfg, args.model_out)
    elif args.mode == "predict":
        predict_csv(args.model, args.input_dir, args.output_csv, args.threads)
    else:
        raise ValueError(args.mode)


if __name__ == "__main__":
    main()
