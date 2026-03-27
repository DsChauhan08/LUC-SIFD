#!/usr/bin/env python3
"""ORB + RANSAC copy-move detector for quick CPU experiments."""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import List, Tuple

import cv2
import numpy as np


def rle_encode(mask: np.ndarray) -> str:
    flat = np.asarray(mask, dtype=np.uint8).reshape(-1)
    idx = np.where(flat > 0)[0]
    if idx.size == 0:
        return "authentic"
    runs: List[int] = []
    start = int(idx[0])
    prev = int(idx[0])
    length = 1
    for cur in idx[1:]:
        cur = int(cur)
        if cur == prev + 1:
            length += 1
        else:
            runs.extend([start, length])
            start = cur
            length = 1
        prev = cur
    runs.extend([start, length])
    return str(runs)


def rle_decode(annotation: str, shape: Tuple[int, int]) -> np.ndarray:
    h, w = shape
    if annotation == "authentic":
        return np.zeros((h, w), dtype=np.uint8)
    runs = json.loads(annotation)
    flat = np.zeros(h * w, dtype=np.uint8)
    for i in range(0, len(runs), 2):
        start = int(runs[i])
        length = int(runs[i + 1])
        flat[start : start + length] = 1
    return flat.reshape(h, w)


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


def list_images(input_dir: str) -> List[str]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return [
        os.path.join(input_dir, n)
        for n in sorted(os.listdir(input_dir))
        if os.path.splitext(n.lower())[1] in exts
        and os.path.isfile(os.path.join(input_dir, n))
    ]


def case_id(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def load_gray(path: str, max_side: int = 512) -> np.ndarray:
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
    return g


def detect_mask(
    gray: np.ndarray,
    nfeatures: int = 1500,
    ratio: float = 0.78,
    min_shift: float = 14.0,
    patch: int = 12,
    min_match: int = 16,
) -> np.ndarray:
    orb = cv2.ORB_create(nfeatures=nfeatures)
    kps, des = orb.detectAndCompute(gray, None)
    h, w = gray.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    if des is None or len(kps) < 8:
        return mask

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = bf.knnMatch(des, des, k=3)
    good = []
    for mset in knn:
        if len(mset) < 3:
            continue
        m1, m2, _m3 = mset[0], mset[1], mset[2]
        if m1.queryIdx == m1.trainIdx:
            continue
        if m1.distance < ratio * m2.distance:
            p1 = kps[m1.queryIdx].pt
            p2 = kps[m1.trainIdx].pt
            if (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 >= min_shift * min_shift:
                good.append((m1.queryIdx, m1.trainIdx))

    if len(good) < min_match:
        return mask

    src = np.float32([kps[i].pt for i, _ in good]).reshape(-1, 1, 2)
    dst = np.float32([kps[j].pt for _, j in good]).reshape(-1, 1, 2)
    _H, inliers = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=2.5
    )
    if inliers is None:
        return mask

    inliers = inliers.ravel().astype(bool)
    if inliers.sum() < max(10, min_match // 2):
        return mask

    for (i, j), keep in zip(good, inliers):
        if not keep:
            continue
        for idx in (i, j):
            x, y = kps[idx].pt
            x = int(round(x))
            y = int(round(y))
            y0 = max(0, y - patch)
            y1 = min(h, y + patch + 1)
            x0 = max(0, x - patch)
            x1 = min(w, x + patch + 1)
            mask[y0:y1, x0:x1] = 1

    if mask.sum() == 0:
        return mask

    k = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    return mask.astype(np.uint8)


def run_mock_eval(args: argparse.Namespace) -> None:
    ims = list_images(args.input_dir)
    scores = []
    auth = 0
    forg = 0
    for p in ims:
        cid = case_id(p)
        gt_path = os.path.join(args.mask_dir, cid + ".npy")
        if not os.path.exists(gt_path):
            continue
        g = load_gray(p, max_side=args.max_side)
        pm = detect_mask(
            g,
            nfeatures=args.nfeatures,
            ratio=args.ratio,
            min_shift=args.min_shift,
            patch=args.patch,
            min_match=args.min_match,
        )
        area = float(pm.sum()) / float(pm.size)
        ann = "authentic" if area < args.auth_area else rle_encode(pm)
        pred = rle_decode(ann, g.shape)
        gt = np.load(gt_path).astype(np.uint8)
        if gt.shape != g.shape:
            gt = cv2.resize(
                gt, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST
            )
            gt = (gt > 0).astype(np.uint8)
        scores.append(f1_score(gt, pred))
        if gt.sum() == 0:
            auth += 1
        else:
            forg += 1
    print(f"mock_eval_n={len(scores)} forged={forg} authentic={auth}")
    print(f"mock_mean_f1={float(np.mean(scores)):.6f}")


def run_predict(args: argparse.Namespace) -> None:
    ims = list_images(args.input_dir)
    rows = []
    auth = 0
    forg = 0
    for p in ims:
        g = load_gray(p, max_side=args.max_side)
        pm = detect_mask(
            g,
            nfeatures=args.nfeatures,
            ratio=args.ratio,
            min_shift=args.min_shift,
            patch=args.patch,
            min_match=args.min_match,
        )
        area = float(pm.sum()) / float(pm.size)
        ann = "authentic" if area < args.auth_area else rle_encode(pm)
        if ann == "authentic":
            auth += 1
        else:
            forg += 1
        rows.append((case_id(p), ann))

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["case_id", "annotation"])
        wr.writerows(rows)
    print(f"predicted {len(rows)} images -> {args.output_csv}")
    print(f"pred_counts authentic={auth} forged={forg}")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="mode", required=True)

    def add_common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--max-side", type=int, default=512)
        sp.add_argument("--nfeatures", type=int, default=1500)
        sp.add_argument("--ratio", type=float, default=0.78)
        sp.add_argument("--min-shift", type=float, default=14.0)
        sp.add_argument("--patch", type=int, default=12)
        sp.add_argument("--min-match", type=int, default=16)
        sp.add_argument("--auth-area", type=float, default=0.004)

    p1 = sub.add_parser("predict")
    p1.add_argument("--input-dir", required=True)
    p1.add_argument("--output-csv", default="submission.csv")
    add_common(p1)

    p2 = sub.add_parser("mock-eval")
    p2.add_argument("--input-dir", required=True)
    p2.add_argument("--mask-dir", required=True)
    add_common(p2)
    return p


def main() -> None:
    args = parser().parse_args()
    if args.mode == "predict":
        run_predict(args)
    else:
        run_mock_eval(args)


if __name__ == "__main__":
    main()
