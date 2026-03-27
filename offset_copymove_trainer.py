#!/usr/bin/env python3
"""Local CPU-only copy-move pipeline with 8-core training/evaluation.

This script is fully offline and supports:
- Synthetic hard-but-solvable dataset generation
- Parameter training (grid search) for segmentation F1
- Evaluation and prediction CSV export
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np


def set_thread_limits(n_threads: int) -> None:
    os.environ["OMP_NUM_THREADS"] = str(n_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n_threads)
    os.environ["MKL_NUM_THREADS"] = str(n_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(n_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n_threads)
    cv2.setNumThreads(max(1, n_threads))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


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
        s = int(runs[i])
        l = int(runs[i + 1])
        flat[s : s + l] = 1
    return flat.reshape(h, w)


def list_images(input_dir: str) -> List[str]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    out = []
    for n in sorted(os.listdir(input_dir)):
        p = os.path.join(input_dir, n)
        if os.path.isfile(p) and os.path.splitext(n.lower())[1] in exts:
            out.append(p)
    return out


def case_id(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def load_gray(path: str, max_side: int) -> np.ndarray:
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


def make_background(h: int, w: int) -> np.ndarray:
    coarse = np.random.rand(28, 28).astype(np.float32)
    bg = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_CUBIC)
    bg = cv2.GaussianBlur(bg, (0, 0), 1.2)

    for _ in range(random.randint(28, 48)):
        cy = random.randint(0, h - 1)
        cx = random.randint(0, w - 1)
        r = random.randint(4, 18)
        amp = random.uniform(-0.18, 0.18)
        yy, xx = np.ogrid[:h, :w]
        d2 = (yy - cy) ** 2 + (xx - cx) ** 2
        blob = np.exp(-d2 / max(1.0, float(2 * r * r)))
        bg += amp * blob.astype(np.float32)

    # thin structures, similar to microscopy/chart artifacts
    for _ in range(random.randint(8, 20)):
        x1, y1 = random.randint(0, w - 1), random.randint(0, h - 1)
        x2, y2 = random.randint(0, w - 1), random.randint(0, h - 1)
        c = random.uniform(0.2, 0.8)
        t = random.randint(1, 2)
        cv2.line(bg, (x1, y1), (x2, y2), c, thickness=t)

    bg += 0.025 * np.random.randn(h, w).astype(np.float32)
    return np.clip(bg, 0.0, 1.0)


def random_patch(h: int, w: int) -> Tuple[int, int, int, int]:
    ph = random.randint(28, 74)
    pw = random.randint(28, 74)
    y = random.randint(0, h - ph)
    x = random.randint(0, w - pw)
    return y, x, ph, pw


def far_destination(
    h: int, w: int, ph: int, pw: int, y: int, x: int
) -> Tuple[int, int]:
    for _ in range(300):
        dy = random.randint(0, h - ph)
        dx = random.randint(0, w - pw)
        if abs(dy - y) + abs(dx - x) > min(h, w) // 4:
            return dy, dx
    return max(0, h - ph), max(0, w - pw)


def forge_copy_move(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h, w = img.shape
    y, x, ph, pw = random_patch(h, w)
    dy, dx = far_destination(h, w, ph, pw, y, x)
    patch = img[y : y + ph, x : x + pw].copy()

    # hard but solvable perturbations (no large rotations)
    gain = random.uniform(0.95, 1.05)
    bias = random.uniform(-0.03, 0.03)
    patch = np.clip(gain * patch + bias, 0.0, 1.0)
    if random.random() < 0.3:
        patch = cv2.GaussianBlur(patch, (0, 0), random.uniform(0.2, 0.8))

    alpha = random.uniform(0.86, 1.0)
    out = img.copy()
    out[dy : dy + ph, dx : dx + pw] = np.clip(
        alpha * patch + (1.0 - alpha) * out[dy : dy + ph, dx : dx + pw],
        0.0,
        1.0,
    )

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y : y + ph, x : x + pw] = 1
    mask[dy : dy + ph, dx : dx + pw] = 1
    return out, mask


def generate_dataset(
    out_img_dir: str,
    out_mask_dir: str,
    n: int,
    forged_ratio: float,
    size: int,
) -> None:
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)

    ids = list(range(1, n + 1))
    random.shuffle(ids)
    k = int(round(n * forged_ratio))
    forged_ids = set(ids[:k])

    for i in range(1, n + 1):
        img = make_background(size, size)
        mask = np.zeros((size, size), dtype=np.uint8)
        if i in forged_ids:
            img, mask = forge_copy_move(img)
        img_u8 = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(out_img_dir, f"{i}.png"), img_u8)
        np.save(os.path.join(out_mask_dir, f"{i}.npy"), mask)


@dataclass
class Params:
    max_side: int = 128
    offset_step: int = 4
    max_offset_ratio: float = 0.38
    min_offset: int = 14
    offset_top: int = 6
    diff_threshold: float = 0.048
    vote_threshold: int = 1
    min_component_area: int = 40
    max_component_area_ratio: float = 0.45
    auth_area_threshold: float = 0.0035
    blur_before: float = 0.4


def find_best_offsets(gray: np.ndarray, p: Params) -> List[Tuple[float, int, int]]:
    h, w = gray.shape
    g = gray
    if p.blur_before > 0:
        g = cv2.GaussianBlur(g, (0, 0), p.blur_before)

    max_off = max(2, int(min(h, w) * p.max_offset_ratio))
    cands: List[Tuple[float, int, int]] = []

    for dy in range(-max_off, max_off + 1, p.offset_step):
        for dx in range(-max_off, max_off + 1, p.offset_step):
            if dy == 0 and dx == 0:
                continue
            if math.hypot(float(dy), float(dx)) < p.min_offset:
                continue

            y0a = max(0, dy)
            y1a = h + min(0, dy)
            x0a = max(0, dx)
            x1a = w + min(0, dx)
            y0b = max(0, -dy)
            y1b = h - max(0, dy)
            x0b = max(0, -dx)
            x1b = w - max(0, dx)
            if y1a <= y0a or x1a <= x0a:
                continue

            a = g[y0a:y1a, x0a:x1a]
            b = g[y0b:y1b, x0b:x1b]
            d = np.abs(a - b)

            # emphasize compact low-diff structure, not only mean
            q = float(np.quantile(d, 0.12))
            s = float(np.mean(d < (q + 1e-6)))
            score = q - 0.02 * s
            cands.append((score, dy, dx))

    cands.sort(key=lambda t: t[0])
    return cands[: p.offset_top]


def build_mask(gray: np.ndarray, p: Params) -> np.ndarray:
    h, w = gray.shape
    votes = np.zeros((h, w), dtype=np.uint8)
    offsets = find_best_offsets(gray, p)
    if not offsets:
        return np.zeros((h, w), dtype=np.uint8)

    g = gray
    if p.blur_before > 0:
        g = cv2.GaussianBlur(g, (0, 0), p.blur_before)

    for _score, dy, dx in offsets:
        y0a = max(0, dy)
        y1a = h + min(0, dy)
        x0a = max(0, dx)
        x1a = w + min(0, dx)
        y0b = max(0, -dy)
        y1b = h - max(0, dy)
        x0b = max(0, -dx)
        x1b = w - max(0, dx)

        a = g[y0a:y1a, x0a:x1a]
        b = g[y0b:y1b, x0b:x1b]
        sim = (np.abs(a - b) <= p.diff_threshold).astype(np.uint8)

        tmp = np.zeros((h, w), dtype=np.uint8)
        tmp[y0a:y1a, x0a:x1a] = np.maximum(tmp[y0a:y1a, x0a:x1a], sim)
        tmp[y0b:y1b, x0b:x1b] = np.maximum(tmp[y0b:y1b, x0b:x1b], sim)
        votes = np.clip(votes + tmp, 0, 255)

    mask = (votes >= p.vote_threshold).astype(np.uint8)

    k = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)

    # component filtering
    nlab, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)
    max_area = int(p.max_component_area_ratio * h * w)
    for i in range(1, nlab):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if a < p.min_component_area:
            continue
        if a > max_area:
            continue
        out[lab == i] = 1
    return out


def predict_annotation(gray: np.ndarray, p: Params) -> str:
    mask = build_mask(gray, p)
    ar = float(mask.sum()) / float(mask.size)
    if ar < p.auth_area_threshold:
        return "authentic"
    return rle_encode(mask)


def evaluate_set(files: Sequence[str], mask_dir: str, p: Params) -> Tuple[float, float]:
    f1s = []
    cls_true = []
    cls_pred = []
    for fp in files:
        cid = case_id(fp)
        mp = os.path.join(mask_dir, cid + ".npy")
        if not os.path.exists(mp):
            continue
        g = load_gray(fp, p.max_side)
        gt = np.load(mp).astype(np.uint8)
        if gt.shape != g.shape:
            gt = cv2.resize(
                gt, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST
            )
            gt = (gt > 0).astype(np.uint8)
        ann = predict_annotation(g, p)
        pm = rle_decode(ann, g.shape)
        f1s.append(f1_score(gt, pm))
        cls_true.append(1 if gt.sum() > 0 else 0)
        cls_pred.append(0 if ann == "authentic" else 1)

    if not f1s:
        return 0.0, 0.0
    cls_acc = float(np.mean(np.asarray(cls_true) == np.asarray(cls_pred)))
    return float(np.mean(f1s)), cls_acc


def train_params(train_files: Sequence[str], mask_dir: str, n_jobs: int) -> Params:
    base = Params()

    grid = []
    for diff_t in (0.040, 0.048, 0.056):
        for vote_t in (1, 2, 3):
            for min_a in (30, 50, 80):
                for auth_t in (0.0025, 0.0055):
                    for topk in (5, 8):
                        grid.append(
                            Params(
                                max_side=base.max_side,
                                offset_step=4,
                                max_offset_ratio=0.38,
                                min_offset=base.min_offset,
                                offset_top=topk,
                                diff_threshold=diff_t,
                                vote_threshold=vote_t,
                                min_component_area=min_a,
                                max_component_area_ratio=base.max_component_area_ratio,
                                auth_area_threshold=auth_t,
                                blur_before=base.blur_before,
                            )
                        )

    # fast search subset first, then full-train rerank top candidates
    subset = list(train_files)
    if len(subset) > 140:
        rng = np.random.default_rng(123)
        idx = rng.choice(len(subset), size=140, replace=False)
        subset = [subset[i] for i in idx]

    results = []
    for pp in grid:
        mf1, cacc = evaluate_set(subset, mask_dir, pp)
        results.append((mf1, cacc, pp))
    results.sort(key=lambda t: (t[0], t[1]), reverse=True)
    top = results[:6]
    reranked = []
    for _, _, pp in top:
        mf1, cacc = evaluate_set(train_files, mask_dir, pp)
        reranked.append((mf1, cacc, pp))
    reranked.sort(key=lambda t: (t[0], t[1]), reverse=True)
    best = reranked[0]
    print(f"train_search_best_f1={best[0]:.6f} cls_acc={best[1]:.6f}")
    return best[2]


def cmd_generate(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    generate_dataset(
        args.out_img_dir, args.out_mask_dir, args.n, args.forged_ratio, args.size
    )
    print(
        f"generated n={args.n} images at {args.out_img_dir} and masks at {args.out_mask_dir}"
    )


def cmd_train(args: argparse.Namespace) -> None:
    set_thread_limits(args.threads)
    set_seed(args.seed)

    files = list_images(args.input_dir)
    n = len(files)
    if n < 10:
        raise ValueError("Need at least 10 images for train/val split")
    idx = np.arange(n)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(idx)
    split = int(0.8 * n)
    tr = [files[i] for i in idx[:split]]
    va = [files[i] for i in idx[split:]]

    best = train_params(tr, args.mask_dir, n_jobs=args.threads)
    tr_f1, tr_acc = evaluate_set(tr, args.mask_dir, best)
    va_f1, va_acc = evaluate_set(va, args.mask_dir, best)

    with open(args.params_out, "w", encoding="utf-8") as f:
        json.dump(asdict(best), f, indent=2)

    print(f"threads_used={args.threads}")
    print(f"train_mean_f1={tr_f1:.6f} train_cls_acc={tr_acc:.6f}")
    print(f"val_mean_f1={va_f1:.6f} val_cls_acc={va_acc:.6f}")
    print(f"saved_params={args.params_out}")


def load_params(path: str) -> Params:
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return Params(**d)


def cmd_eval(args: argparse.Namespace) -> None:
    set_thread_limits(args.threads)
    p = load_params(args.params)
    files = list_images(args.input_dir)
    mf1, cacc = evaluate_set(files, args.mask_dir, p)
    print(f"threads_used={args.threads}")
    print(f"eval_mean_f1={mf1:.6f}")
    print(f"eval_cls_acc={cacc:.6f}")


def cmd_predict(args: argparse.Namespace) -> None:
    set_thread_limits(args.threads)
    p = load_params(args.params)
    files = list_images(args.input_dir)

    rows = []
    auth = 0
    forg = 0
    for fp in files:
        g = load_gray(fp, p.max_side)
        ann = predict_annotation(g, p)
        rows.append((case_id(fp), ann))
        if ann == "authentic":
            auth += 1
        else:
            forg += 1

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["case_id", "annotation"])
        wr.writerows(rows)

    print(f"threads_used={args.threads}")
    print(f"predicted={len(rows)} authentic={auth} forged={forg}")
    print(f"saved={args.output_csv}")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="mode", required=True)

    g = sub.add_parser("generate")
    g.add_argument("--out-img-dir", default="mock_hard/images")
    g.add_argument("--out-mask-dir", default="mock_hard/masks")
    g.add_argument("--n", type=int, default=520)
    g.add_argument("--forged-ratio", type=float, default=0.56)
    g.add_argument("--size", type=int, default=256)
    g.add_argument("--seed", type=int, default=42)

    t = sub.add_parser("train")
    t.add_argument("--input-dir", default="mock_hard/images")
    t.add_argument("--mask-dir", default="mock_hard/masks")
    t.add_argument("--params-out", default="best_params.json")
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--threads", type=int, default=8)

    e = sub.add_parser("eval")
    e.add_argument("--input-dir", default="mock_hard/images")
    e.add_argument("--mask-dir", default="mock_hard/masks")
    e.add_argument("--params", default="best_params.json")
    e.add_argument("--threads", type=int, default=8)

    pr = sub.add_parser("predict")
    pr.add_argument("--input-dir", default="mock_hard/images")
    pr.add_argument("--params", default="best_params.json")
    pr.add_argument("--output-csv", default="submission.csv")
    pr.add_argument("--threads", type=int, default=8)

    return p


def main() -> None:
    args = parser().parse_args()
    if args.mode == "generate":
        cmd_generate(args)
    elif args.mode == "train":
        cmd_train(args)
    elif args.mode == "eval":
        cmd_eval(args)
    elif args.mode == "predict":
        cmd_predict(args)
    else:
        raise ValueError(args.mode)


if __name__ == "__main__":
    main()
