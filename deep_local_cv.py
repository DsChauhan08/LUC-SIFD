#!/usr/bin/env python3
"""5-fold CV for deep_local_copymove model (CPU, capped threads)."""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-dir", default="deep_mock/images")
    ap.add_argument("--mask-dir", default="deep_mock/masks")
    ap.add_argument("--img-size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
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
                mlog, clog = model(x)
                bce = F.binary_cross_entropy_with_logits(mlog, y)
                p = torch.sigmoid(mlog)
                dice = (
                    1.0
                    - (
                        (2 * (p * y).sum((1, 2, 3)) + 1e-6)
                        / ((p + y).sum((1, 2, 3)) + 1e-6)
                    ).mean()
                )
                cls = F.binary_cross_entropy_with_logits(clog, c)
                loss = 0.8 * (bce + dice) + 0.2 * cls
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

        model.eval()
        c_true = []
        c_pred = []
        s_f1 = []
        with torch.no_grad():
            for x, y, c in va_dl:
                mlog, clog = model(x)
                pm = (torch.sigmoid(mlog) > 0.5).float().numpy()
                pc = (torch.sigmoid(clog) > 0.5).float().numpy().reshape(-1)
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
