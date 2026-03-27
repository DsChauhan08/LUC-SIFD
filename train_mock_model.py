#!/usr/bin/env python3
"""Train a lightweight copy-move detector on synthetic data and evaluate."""

from __future__ import annotations

import argparse
import os
from typing import List, Tuple

import cv2
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def set_thread_limits(n_threads: int) -> None:
    os.environ["OMP_NUM_THREADS"] = str(n_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n_threads)
    os.environ["MKL_NUM_THREADS"] = str(n_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(n_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n_threads)
    cv2.setNumThreads(max(1, n_threads))


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
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


def list_ids(img_dir: str, mask_dir: str) -> List[str]:
    out = []
    for n in sorted(os.listdir(img_dir)):
        if not n.endswith(".png"):
            continue
        sid = os.path.splitext(n)[0]
        if os.path.exists(os.path.join(mask_dir, sid + ".npy")):
            out.append(sid)
    return out


def load_gray(path: str, max_side: int = 256) -> np.ndarray:
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if g is None:
        raise FileNotFoundError(path)
    h, w = g.shape
    if max(h, w) > max_side:
        s = max_side / float(max(h, w))
        g = cv2.resize(
            g,
            (max(1, int(round(w * s))), max(1, int(round(h * s)))),
            interpolation=cv2.INTER_AREA,
        )
    return g.astype(np.float32) / 255.0


def extract_features(gray: np.ndarray) -> np.ndarray:
    g = (gray * 255).astype(np.uint8)
    h, w = g.shape

    # global stats
    mean = float(gray.mean())
    std = float(gray.std())
    p95 = float(np.quantile(gray, 0.95))
    p05 = float(np.quantile(gray, 0.05))

    # gradients
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gm = np.sqrt(gx * gx + gy * gy)
    gmean = float(gm.mean())
    gstd = float(gm.std())

    # template matching cues
    def tm_stats(patch: int, samples: int = 40) -> Tuple[float, float]:
        best_corr = -1.0
        best_sad = 1.0
        rng = np.random.default_rng(123 + patch)
        for _ in range(samples):
            y = int(rng.integers(0, h - patch))
            x = int(rng.integers(0, w - patch))
            tpl = g[y : y + patch, x : x + patch]
            res = cv2.matchTemplate(g, tpl, cv2.TM_CCOEFF_NORMED)
            y0 = max(0, y - patch)
            y1 = min(res.shape[0], y + patch)
            x0 = max(0, x - patch)
            x1 = min(res.shape[1], x + patch)
            res[y0:y1, x0:x1] = -1
            _mn, mx, _mnl, loc = cv2.minMaxLoc(res)
            if mx > best_corr:
                best_corr = float(mx)
            yy, xx = loc[1], loc[0]
            p2 = g[yy : yy + patch, xx : xx + patch]
            sad = float(
                np.mean(np.abs(tpl.astype(np.float32) - p2.astype(np.float32))) / 255.0
            )
            if sad < best_sad:
                best_sad = sad
        return best_corr, best_sad

    c12, s12 = tm_stats(12)
    c20, s20 = tm_stats(20)

    # ORB cue
    orb = cv2.ORB_create(nfeatures=1200)
    kps, des = orb.detectAndCompute(g, None)
    nkp = len(kps) if kps is not None else 0
    good = 0
    if des is not None and nkp > 5:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        knn = bf.knnMatch(des, des, k=3)
        for mset in knn:
            if len(mset) < 2:
                continue
            m1, m2 = mset[0], mset[1]
            if m1.queryIdx == m1.trainIdx:
                continue
            if m1.distance < 0.8 * m2.distance:
                p1 = kps[m1.queryIdx].pt
                p2 = kps[m1.trainIdx].pt
                if (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 > 12 * 12:
                    good += 1

    return np.array(
        [
            mean,
            std,
            p05,
            p95,
            gmean,
            gstd,
            c12,
            s12,
            c20,
            s20,
            float(nkp),
            float(good),
        ],
        dtype=np.float32,
    )


def make_dataset(img_dir: str, mask_dir: str, max_side: int = 256):
    ids = list_ids(img_dir, mask_dir)
    X = []
    y = []
    gt_masks = {}
    shapes = {}
    for sid in ids:
        gray = load_gray(os.path.join(img_dir, sid + ".png"), max_side=max_side)
        X.append(extract_features(gray))
        m = np.load(os.path.join(mask_dir, sid + ".npy")).astype(np.uint8)
        if m.shape != gray.shape:
            m = cv2.resize(
                m, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST
            )
            m = (m > 0).astype(np.uint8)
        y.append(1 if m.sum() > 0 else 0)
        gt_masks[sid] = m
        shapes[sid] = gray.shape
    return ids, np.stack(X), np.array(y, dtype=np.uint8), gt_masks, shapes


def segment_from_template(
    gray: np.ndarray, score: float, patch: int = 24
) -> np.ndarray:
    g = (gray * 255).astype(np.uint8)
    h, w = g.shape
    out = np.zeros((h, w), dtype=np.uint8)
    n_trials = 60 if score > 0.8 else 25
    rng = np.random.default_rng(77)
    for _ in range(n_trials):
        y = int(rng.integers(0, h - patch))
        x = int(rng.integers(0, w - patch))
        tpl = g[y : y + patch, x : x + patch]
        res = cv2.matchTemplate(g, tpl, cv2.TM_CCOEFF_NORMED)
        y0 = max(0, y - patch)
        y1 = min(res.shape[0], y + patch)
        x0 = max(0, x - patch)
        x1 = min(res.shape[1], x + patch)
        res[y0:y1, x0:x1] = -1
        _mn, mx, _mnl, loc = cv2.minMaxLoc(res)
        if mx < 0.93:
            continue
        yy, xx = loc[1], loc[0]
        p2 = g[yy : yy + patch, xx : xx + patch]
        sad = float(
            np.mean(np.abs(tpl.astype(np.float32) - p2.astype(np.float32))) / 255.0
        )
        if sad > 0.03:
            continue
        out[y : y + patch, x : x + patch] = 1
        out[yy : yy + patch, xx : xx + patch] = 1

    if out.sum() == 0:
        return out
    k = np.ones((3, 3), np.uint8)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k, iterations=1)
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, k, iterations=1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-dir", default="mock_easy/images")
    ap.add_argument("--mask-dir", default="mock_easy/masks")
    ap.add_argument("--max-side", type=int, default=256)
    ap.add_argument("--model-out", default="mock_model.pkl")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    set_thread_limits(args.threads)

    ids, X, y, gt_masks, _shapes = make_dataset(
        args.img_dir, args.mask_dir, args.max_side
    )
    n = len(ids)
    split = int(0.8 * n)
    train_ids = ids[:split]
    val_ids = ids[split:]
    Xtr, ytr = X[:split], y[:split]
    Xva, yva = X[split:], y[split:]

    clf = RandomForestClassifier(
        n_estimators=350,
        max_depth=14,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=args.threads,
    )
    clf.fit(Xtr, ytr)
    pva = clf.predict_proba(Xva)[:, 1]
    yhat = (pva >= 0.5).astype(np.uint8)
    acc = accuracy_score(yva, yhat)

    f1s = []
    for sid, prob, cls in zip(val_ids, pva, yhat):
        gray = load_gray(os.path.join(args.img_dir, sid + ".png"), args.max_side)
        if cls == 0:
            pm = np.zeros_like(gray, dtype=np.uint8)
        else:
            pm = segment_from_template(gray, score=float(prob), patch=24)
        f1s.append(f1_score(gt_masks[sid], pm))

    mean_f1 = float(np.mean(f1s))
    print(f"val_cls_acc={acc:.6f}")
    print(f"val_mask_mean_f1={mean_f1:.6f}")
    print(f"threads_used={args.threads}")

    joblib.dump(clf, args.model_out)
    print(f"saved model to {args.model_out}")


if __name__ == "__main__":
    main()
