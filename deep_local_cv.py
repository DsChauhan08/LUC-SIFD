#!/usr/bin/env python3
"""5-fold CV for deep_local_copymove model (CPU, capped threads)."""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from deep_local_copymove import PairDataset, TinyUNet, mask_f1_np, set_limits, set_seed


def collect_ids(img_dir: str, mask_dir: str):
    ids = []
    for n in sorted(os.listdir(img_dir)):
        if n.endswith(".png"):
            sid = n[:-4]
            if os.path.exists(os.path.join(mask_dir, sid + ".npy")):
                ids.append(sid)
    return ids


def apply_flip(x: torch.Tensor, mode: int) -> torch.Tensor:
    if mode == 0:
        return x
    if mode == 1:
        return torch.flip(x, dims=[3])
    if mode == 2:
        return torch.flip(x, dims=[2])
    if mode == 3:
        return torch.flip(x, dims=[2, 3])
    raise ValueError(mode)


class CachedPairDataset(Dataset):
    def __init__(self, base: PairDataset):
        self.base = base
        self.cache = [base[i] for i in range(len(base))]

    def __len__(self) -> int:
        return len(self.cache)

    def __getitem__(self, idx: int):
        return self.cache[idx]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-dir", default="deep_mock/images")
    ap.add_argument("--mask-dir", default="deep_mock/masks")
    ap.add_argument("--img-size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tta", type=int, default=4, choices=[1, 2, 4])
    ap.add_argument("--cached", action="store_true")
    args = ap.parse_args()

    set_limits(args.threads)
    set_seed(args.seed)

    ids = np.array(collect_ids(args.img_dir, args.mask_dir))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(ids)
    folds = np.array_split(ids, 5)

    fold_acc = []
    fold_f1 = []

    for fi in range(5):
        va = set(folds[fi].tolist())
        tr_ids = [s for s in ids.tolist() if s not in va]
        va_ids = list(va)

        tr_ds = PairDataset(tr_ids, args.img_dir, args.mask_dir, args.img_size, True)
        va_ds = PairDataset(va_ids, args.img_dir, args.mask_dir, args.img_size, False)
        if args.cached:
            tr_ds = CachedPairDataset(tr_ds)
            va_ds = CachedPairDataset(va_ds)
        tr_dl = DataLoader(
            tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=0
        )
        va_dl = DataLoader(
            va_ds, batch_size=args.batch_size, shuffle=False, num_workers=0
        )

        model = TinyUNet().cpu()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

        for _ep in range(args.epochs):
            model.train()
            for x, y, c in tr_dl:
                mlog, auxlog, clog = model(x)
                bce = F.binary_cross_entropy_with_logits(mlog, y)
                p = torch.sigmoid(mlog)
                dice = (
                    1.0
                    - (
                        (2 * (p * y).sum((1, 2, 3)) + 1e-6)
                        / ((p + y).sum((1, 2, 3)) + 1e-6)
                    ).mean()
                )
                bce_aux = F.binary_cross_entropy_with_logits(auxlog, y)
                p_aux = torch.sigmoid(auxlog)
                dice_aux = (
                    1.0
                    - (
                        (2 * (p_aux * y).sum((1, 2, 3)) + 1e-6)
                        / ((p_aux + y).sum((1, 2, 3)) + 1e-6)
                    ).mean()
                )
                cls = F.binary_cross_entropy_with_logits(clog, c)
                loss = 0.65 * (bce + dice) + 0.20 * (bce_aux + dice_aux) + 0.15 * cls
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

        model.eval()
        c_true = []
        c_pred = []
        s_f1 = []
        with torch.no_grad():
            for x, y, c in va_dl:
                modes = [0]
                if args.tta >= 2:
                    modes = [0, 1]
                if args.tta >= 4:
                    modes = [0, 1, 2, 3]

                mlog_acc = None
                cprob = 0.0
                for m in modes:
                    xx = apply_flip(x, m)
                    ml, _aux, cl = model(xx)
                    ml = apply_flip(ml, m)
                    mlog_acc = ml if mlog_acc is None else (mlog_acc + ml)
                    cprob += torch.sigmoid(cl)

                mlog = mlog_acc / float(len(modes))
                clog = cprob / float(len(modes))
                pm = (torch.sigmoid(mlog) > 0.5).float().numpy()
                pc = (clog > 0.5).float().numpy().reshape(-1)
                yy = y.numpy()
                cc = c.numpy().reshape(-1)
                c_true.extend(cc.tolist())
                c_pred.extend(pc.tolist())
                for i in range(pm.shape[0]):
                    s_f1.append(mask_f1_np(yy[i, 0], pm[i, 0]))

        acc = float(np.mean(np.asarray(c_true) == np.asarray(c_pred)))
        mf1 = float(np.mean(s_f1))
        fold_acc.append(acc)
        fold_f1.append(mf1)
        print(f"fold{fi + 1}_cls_acc={acc:.4f} fold{fi + 1}_mask_f1={mf1:.4f}")

    print(f"cv_cls_acc_mean={float(np.mean(fold_acc)):.6f}")
    print(f"cv_mask_f1_mean={float(np.mean(fold_f1)):.6f}")
    print(f"threads_used={args.threads}")


if __name__ == "__main__":
    main()
