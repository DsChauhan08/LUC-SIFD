#!/usr/bin/env python3
"""
CPU-first copy-move forgery detector for biomedical images.

This script provides:
1) Heuristic copy-move segmentation from self-similarity of image patches.
2) Authentic vs forged decision using calibrated suspicious-area threshold.
3) Kaggle-ready submission writing with strict RLE format.
4) Local mock evaluation support when ground-truth masks are available.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_gray(path: str, max_side: int) -> np.ndarray:
    img = Image.open(path).convert("L")
    w, h = img.size
    scale = 1.0
    if max(w, h) > max_side:
        scale = max_side / float(max(w, h))
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        img = img.resize((nw, nh), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def integral_image(arr: np.ndarray) -> np.ndarray:
    return np.pad(arr.cumsum(0).cumsum(1), ((1, 0), (1, 0)), mode="constant")


def rect_sum(ii: np.ndarray, y0: int, x0: int, y1: int, x1: int) -> float:
    return float(ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0])


def patch_stats(
    img: np.ndarray, patch: int, stride: int
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = img.shape
    ys = list(range(0, h - patch + 1, stride))
    xs = list(range(0, w - patch + 1, stride))
    n = len(ys) * len(xs)
    pos = np.zeros((n, 2), dtype=np.int32)
    feat = np.zeros((n, 9), dtype=np.float32)

    gx = np.zeros_like(img)
    gy = np.zeros_like(img)
    gx[:, 1:-1] = img[:, 2:] - img[:, :-2]
    gy[1:-1, :] = img[2:, :] - img[:-2, :]
    mag = np.sqrt(gx * gx + gy * gy)

    ii = integral_image(img)
    ii2 = integral_image(img * img)
    iim = integral_image(mag)

    k = 0
    for y in ys:
        y1 = y + patch
        for x in xs:
            x1 = x + patch
            s = rect_sum(ii, y, x, y1, x1)
            s2 = rect_sum(ii2, y, x, y1, x1)
            sm = rect_sum(iim, y, x, y1, x1)
            area = float(patch * patch)
            mean = s / area
            var = max(0.0, s2 / area - mean * mean)

            p = img[y:y1, x:x1]
            q1 = p[: patch // 2, : patch // 2].mean()
            q2 = p[: patch // 2, patch // 2 :].mean()
            q3 = p[patch // 2 :, : patch // 2].mean()
            q4 = p[patch // 2 :, patch // 2 :].mean()

            vx = float(np.abs(p[:, 1:] - p[:, :-1]).mean()) if patch > 1 else 0.0
            vy = float(np.abs(p[1:, :] - p[:-1, :]).mean()) if patch > 1 else 0.0

            pos[k] = (y, x)
            feat[k] = np.array(
                [mean, var, sm / area, q1, q2, q3, q4, vx, vy], dtype=np.float32
            )
            k += 1

    feat_mean = feat.mean(axis=0, keepdims=True)
    feat_std = feat.std(axis=0, keepdims=True) + 1e-6
    feat = (feat - feat_mean) / feat_std
    return pos, feat


def quantize_descriptors(feat: np.ndarray, bins: int) -> np.ndarray:
    f = feat[:, :4]
    lo = f.min(axis=0, keepdims=True)
    hi = f.max(axis=0, keepdims=True)
    sc = (f - lo) / np.maximum(hi - lo, 1e-6)
    q = np.clip((sc * bins).astype(np.int32), 0, bins - 1)
    return q


def topk_matches(
    pos: np.ndarray,
    feat: np.ndarray,
    patch: int,
    k: int,
    min_offset: float,
    sim_threshold: float,
) -> List[Tuple[int, int, float]]:
    f = feat
    n = f.shape[0]
    sims = f @ f.T
    norms = np.linalg.norm(f, axis=1, keepdims=True)
    denom = norms @ norms.T
    sims = sims / np.maximum(denom, 1e-8)
    np.fill_diagonal(sims, -1.0)

    matches: List[Tuple[int, int, float]] = []
    min_off = float(min_offset * patch)
    for i in range(n):
        row = sims[i]
        idx = np.argpartition(-row, min(k, n - 1) - 1)[: min(k, n - 1)]
        yi, xi = pos[i]
        for j in idx:
            s = float(row[j])
            if s < sim_threshold:
                continue
            yj, xj = pos[j]
            d = math.hypot(float(yi - yj), float(xi - xj))
            if d < min_off:
                continue
            a, b = (i, j) if i < j else (j, i)
            matches.append((a, b, s))

    dedup = {}
    for a, b, s in matches:
        key = (a, b)
        if key not in dedup or s > dedup[key]:
            dedup[key] = s
    out = [(a, b, s) for (a, b), s in dedup.items()]
    out.sort(key=lambda t: t[2], reverse=True)
    return out


def draw_patch(mask: np.ndarray, y: int, x: int, patch: int) -> None:
    h, w = mask.shape
    y1 = min(h, y + patch)
    x1 = min(w, x + patch)
    mask[y:y1, x:x1] = 1


def binary_dilate(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    out = mask.astype(np.uint8)
    for _ in range(iterations):
        p = np.pad(out, 1, mode="constant")
        nb = [
            p[1:-1, 1:-1],
            p[:-2, 1:-1],
            p[2:, 1:-1],
            p[1:-1, :-2],
            p[1:-1, 2:],
            p[:-2, :-2],
            p[:-2, 2:],
            p[2:, :-2],
            p[2:, 2:],
        ]
        out = np.maximum.reduce(nb)
    return out


def binary_erode(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    out = mask.astype(np.uint8)
    for _ in range(iterations):
        p = np.pad(out, 1, mode="constant")
        nb = [
            p[1:-1, 1:-1],
            p[:-2, 1:-1],
            p[2:, 1:-1],
            p[1:-1, :-2],
            p[1:-1, 2:],
            p[:-2, :-2],
            p[:-2, 2:],
            p[2:, :-2],
            p[2:, 2:],
        ]
        out = np.minimum.reduce(nb)
    return out


def binary_open(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    return binary_dilate(binary_erode(mask, iterations), iterations)


def binary_close(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    return binary_erode(binary_dilate(mask, iterations), iterations)


def connected_components(mask: np.ndarray) -> List[np.ndarray]:
    h, w = mask.shape
    vis = np.zeros((h, w), dtype=np.uint8)
    comps: List[np.ndarray] = []

    for y in range(h):
        for x in range(w):
            if mask[y, x] == 0 or vis[y, x] == 1:
                continue
            stack = [(y, x)]
            vis[y, x] = 1
            pts = []
            while stack:
                cy, cx = stack.pop()
                pts.append((cy, cx))
                for ny in (cy - 1, cy, cy + 1):
                    if ny < 0 or ny >= h:
                        continue
                    for nx in (cx - 1, cx, cx + 1):
                        if nx < 0 or nx >= w:
                            continue
                        if vis[ny, nx] == 0 and mask[ny, nx] == 1:
                            vis[ny, nx] = 1
                            stack.append((ny, nx))
            comp = np.zeros((h, w), dtype=np.uint8)
            ys, xs = zip(*pts)
            comp[np.array(ys), np.array(xs)] = 1
            comps.append(comp)
    return comps


def filter_components(
    mask: np.ndarray, min_area: int, max_area_ratio: float, max_keep_components: int
) -> np.ndarray:
    h, w = mask.shape
    max_area = int(max_area_ratio * h * w)
    kept = []
    for comp in connected_components(mask):
        a = int(comp.sum())
        if a < min_area:
            continue
        if a > max_area:
            continue
        kept.append((a, comp))
    kept.sort(key=lambda t: t[0], reverse=True)
    if max_keep_components > 0:
        kept = kept[:max_keep_components]
    out = np.zeros_like(mask, dtype=np.uint8)
    for _a, comp in kept:
        out = np.maximum(out, comp)
    return out


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


@dataclass
class DetectorConfig:
    max_side: int = 512
    patch: int = 12
    stride: int = 6
    topk: int = 8
    min_offset_patches: float = 2.0
    sim_threshold: float = 0.90
    min_component_area: int = 48
    max_component_area_ratio: float = 0.35
    auth_area_ratio_threshold: float = 0.007
    morph_iterations: int = 1
    enable_offset_branch: bool = True
    offset_step: int = 8
    offset_max_ratio: float = 0.35
    offset_top: int = 8
    pixel_diff_threshold: float = 0.035
    offset_vote_threshold: int = 1
    enable_bucket_branch: bool = True
    descriptor_size: int = 4
    descriptor_bins: int = 16
    bucket_max_per_key: int = 20
    bucket_pair_limit: int = 1200
    patch_diff_threshold: float = 0.06
    max_keep_components: int = 10


class CopyMoveDetector:
    def __init__(self, cfg: DetectorConfig) -> None:
        self.cfg = cfg

    def _bucket_branch(
        self, gray: np.ndarray, pos: np.ndarray, feat: np.ndarray
    ) -> np.ndarray:
        cfg = self.cfg
        h, w = gray.shape
        if len(pos) < 2:
            return np.zeros((h, w), dtype=np.uint8)

        q = quantize_descriptors(feat, cfg.descriptor_bins)
        buckets = {}
        for i, key_v in enumerate(q):
            key = tuple(int(x) for x in key_v[: cfg.descriptor_size])
            arr = buckets.setdefault(key, [])
            if len(arr) < cfg.bucket_max_per_key:
                arr.append(i)

        cand_pairs: List[Tuple[int, int]] = []
        for idxs in buckets.values():
            m = len(idxs)
            if m < 2:
                continue
            for a in range(m):
                ia = idxs[a]
                ya, xa = pos[ia]
                for b in range(a + 1, m):
                    ib = idxs[b]
                    yb, xb = pos[ib]
                    d = math.hypot(float(ya - yb), float(xa - xb))
                    if d < cfg.patch * cfg.min_offset_patches:
                        continue
                    cand_pairs.append((ia, ib))
                    if len(cand_pairs) >= cfg.bucket_pair_limit:
                        break
                if len(cand_pairs) >= cfg.bucket_pair_limit:
                    break
            if len(cand_pairs) >= cfg.bucket_pair_limit:
                break

        mask = np.zeros((h, w), dtype=np.uint8)
        p = cfg.patch
        for ia, ib in cand_pairs:
            ya, xa = pos[ia]
            yb, xb = pos[ib]
            pa = gray[ya : ya + p, xa : xa + p]
            pb = gray[yb : yb + p, xb : xb + p]
            if pa.shape != (p, p) or pb.shape != (p, p):
                continue
            mad = float(np.mean(np.abs(pa - pb)))
            if mad <= cfg.patch_diff_threshold:
                draw_patch(mask, int(ya), int(xa), p)
                draw_patch(mask, int(yb), int(xb), p)

        return mask

    def _offset_branch(self, gray: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        h, w = gray.shape
        max_off = max(2, int(min(h, w) * cfg.offset_max_ratio))
        cands: List[Tuple[float, int, int]] = []

        for dy in range(-max_off, max_off + 1, cfg.offset_step):
            for dx in range(-max_off, max_off + 1, cfg.offset_step):
                if dy == 0 and dx == 0:
                    continue
                if (
                    math.hypot(float(dy), float(dx))
                    < cfg.patch * cfg.min_offset_patches
                ):
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

                a = gray[y0a:y1a, x0a:x1a]
                b = gray[y0b:y1b, x0b:x1b]
                mad = float(np.mean(np.abs(a - b)))
                cands.append((mad, dy, dx))

        if not cands:
            return np.zeros_like(gray, dtype=np.uint8)

        cands.sort(key=lambda t: t[0])
        cands = cands[: cfg.offset_top]

        votes = np.zeros_like(gray, dtype=np.uint8)
        for _mad, dy, dx in cands:
            y0a = max(0, dy)
            y1a = h + min(0, dy)
            x0a = max(0, dx)
            x1a = w + min(0, dx)
            y0b = max(0, -dy)
            y1b = h - max(0, dy)
            x0b = max(0, -dx)
            x1b = w - max(0, dx)
            a = gray[y0a:y1a, x0a:x1a]
            b = gray[y0b:y1b, x0b:x1b]
            sim = (np.abs(a - b) <= cfg.pixel_diff_threshold).astype(np.uint8)

            tmp = np.zeros_like(gray, dtype=np.uint8)
            tmp[y0a:y1a, x0a:x1a] = np.maximum(tmp[y0a:y1a, x0a:x1a], sim)
            tmp[y0b:y1b, x0b:x1b] = np.maximum(tmp[y0b:y1b, x0b:x1b], sim)
            votes = np.clip(votes + tmp, 0, 255)

        return (votes >= cfg.offset_vote_threshold).astype(np.uint8)

    def predict_mask(self, gray: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        pos, feat = patch_stats(gray, cfg.patch, cfg.stride)
        mask = np.zeros_like(gray, dtype=np.uint8)
        if len(pos) >= 2:
            matches = topk_matches(
                pos,
                feat,
                patch=cfg.patch,
                k=cfg.topk,
                min_offset=cfg.min_offset_patches,
                sim_threshold=cfg.sim_threshold,
            )
            for a, b, _s in matches:
                ya, xa = pos[a]
                yb, xb = pos[b]
                draw_patch(mask, int(ya), int(xa), cfg.patch)
                draw_patch(mask, int(yb), int(xb), cfg.patch)

        if cfg.enable_offset_branch:
            mask = np.maximum(mask, self._offset_branch(gray))

        if cfg.enable_bucket_branch:
            mask = np.maximum(mask, self._bucket_branch(gray, pos, feat))

        if cfg.morph_iterations > 0:
            mask = binary_close(mask, cfg.morph_iterations)
            mask = binary_open(mask, cfg.morph_iterations)

        mask = filter_components(
            mask,
            cfg.min_component_area,
            cfg.max_component_area_ratio,
            cfg.max_keep_components,
        )
        return mask

    def predict_annotation(self, gray: np.ndarray) -> str:
        mask = self.predict_mask(gray)
        area_ratio = float(mask.sum()) / float(mask.size)
        if area_ratio < self.cfg.auth_area_ratio_threshold:
            return "authentic"
        return rle_encode(mask)


def list_images(input_dir: str) -> List[str]:
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    out = []
    for name in sorted(os.listdir(input_dir)):
        p = os.path.join(input_dir, name)
        if not os.path.isfile(p):
            continue
        if os.path.splitext(name.lower())[1] in exts:
            out.append(p)
    return out


def case_id_from_name(path: str) -> str:
    base = os.path.basename(path)
    stem = os.path.splitext(base)[0]
    if stem.isdigit():
        return stem
    return stem


def run_predict(args: argparse.Namespace) -> None:
    cfg = DetectorConfig(
        max_side=args.max_side,
        patch=args.patch,
        stride=args.stride,
        topk=args.topk,
        min_offset_patches=args.min_offset_patches,
        sim_threshold=args.sim_threshold,
        min_component_area=args.min_component_area,
        max_component_area_ratio=args.max_component_area_ratio,
        auth_area_ratio_threshold=args.auth_area_ratio_threshold,
        morph_iterations=args.morph_iterations,
        enable_offset_branch=not args.disable_offset_branch,
        offset_step=args.offset_step,
        offset_max_ratio=args.offset_max_ratio,
        offset_top=args.offset_top,
        pixel_diff_threshold=args.pixel_diff_threshold,
        offset_vote_threshold=args.offset_vote_threshold,
        enable_bucket_branch=not args.disable_bucket_branch,
        descriptor_size=args.descriptor_size,
        descriptor_bins=args.descriptor_bins,
        bucket_max_per_key=args.bucket_max_per_key,
        bucket_pair_limit=args.bucket_pair_limit,
        patch_diff_threshold=args.patch_diff_threshold,
        max_keep_components=args.max_keep_components,
    )
    det = CopyMoveDetector(cfg)
    files = list_images(args.input_dir)
    t0 = time.time()
    rows: List[Tuple[str, str]] = []
    authentic_n = 0
    forged_n = 0
    for p in files:
        g = load_gray(p, max_side=cfg.max_side)
        ann = det.predict_annotation(g)
        if ann == "authentic":
            authentic_n += 1
        else:
            forged_n += 1
        rows.append((case_id_from_name(p), ann))
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["case_id", "annotation"])
        for cid, ann in rows:
            wr.writerow([cid, ann])
    dt = time.time() - t0
    print(f"predicted {len(rows)} images in {dt:.2f}s -> {args.output_csv}")
    print(f"pred_counts authentic={authentic_n} forged={forged_n}")


def run_mock_eval(args: argparse.Namespace) -> None:
    cfg = DetectorConfig(
        max_side=args.max_side,
        patch=args.patch,
        stride=args.stride,
        topk=args.topk,
        min_offset_patches=args.min_offset_patches,
        sim_threshold=args.sim_threshold,
        min_component_area=args.min_component_area,
        max_component_area_ratio=args.max_component_area_ratio,
        auth_area_ratio_threshold=args.auth_area_ratio_threshold,
        morph_iterations=args.morph_iterations,
        enable_offset_branch=not args.disable_offset_branch,
        offset_step=args.offset_step,
        offset_max_ratio=args.offset_max_ratio,
        offset_top=args.offset_top,
        pixel_diff_threshold=args.pixel_diff_threshold,
        offset_vote_threshold=args.offset_vote_threshold,
        enable_bucket_branch=not args.disable_bucket_branch,
        descriptor_size=args.descriptor_size,
        descriptor_bins=args.descriptor_bins,
        bucket_max_per_key=args.bucket_max_per_key,
        bucket_pair_limit=args.bucket_pair_limit,
        patch_diff_threshold=args.patch_diff_threshold,
        max_keep_components=args.max_keep_components,
    )
    det = CopyMoveDetector(cfg)
    files = list_images(args.input_dir)
    scores: List[float] = []
    forged = 0
    authentic = 0
    for p in files:
        base = case_id_from_name(p)
        mask_path = os.path.join(args.mask_dir, base + ".npy")
        if not os.path.exists(mask_path):
            continue
        g = load_gray(p, max_side=cfg.max_side)
        pred_ann = det.predict_annotation(g)
        gt = np.load(mask_path).astype(np.uint8)
        if gt.shape != g.shape:
            gt_img = Image.fromarray((gt * 255).astype(np.uint8))
            gt_img = gt_img.resize((g.shape[1], g.shape[0]), Image.NEAREST)
            gt = (np.asarray(gt_img) > 0).astype(np.uint8)
        pred = rle_decode(pred_ann, g.shape)
        scores.append(f1_score(gt, pred))
        if gt.sum() == 0:
            authentic += 1
        else:
            forged += 1
    if not scores:
        print("no matched image/mask pairs found")
        return
    print(f"mock_eval_n={len(scores)} forged={forged} authentic={authentic}")
    print(f"mock_mean_f1={float(np.mean(scores)):.6f}")
    print(f"mock_std_f1={float(np.std(scores)):.6f}")


def default_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="mode", required=True)

    def add_common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--max-side", type=int, default=512)
        sp.add_argument("--patch", type=int, default=12)
        sp.add_argument("--stride", type=int, default=6)
        sp.add_argument("--topk", type=int, default=8)
        sp.add_argument("--min-offset-patches", type=float, default=2.0)
        sp.add_argument("--sim-threshold", type=float, default=0.90)
        sp.add_argument("--min-component-area", type=int, default=48)
        sp.add_argument("--max-component-area-ratio", type=float, default=0.35)
        sp.add_argument("--auth-area-ratio-threshold", type=float, default=0.007)
        sp.add_argument("--morph-iterations", type=int, default=1)
        sp.add_argument("--disable-offset-branch", action="store_true")
        sp.add_argument("--offset-step", type=int, default=8)
        sp.add_argument("--offset-max-ratio", type=float, default=0.35)
        sp.add_argument("--offset-top", type=int, default=8)
        sp.add_argument("--pixel-diff-threshold", type=float, default=0.035)
        sp.add_argument("--offset-vote-threshold", type=int, default=1)
        sp.add_argument("--disable-bucket-branch", action="store_true")
        sp.add_argument("--descriptor-size", type=int, default=4)
        sp.add_argument("--descriptor-bins", type=int, default=16)
        sp.add_argument("--bucket-max-per-key", type=int, default=20)
        sp.add_argument("--bucket-pair-limit", type=int, default=1200)
        sp.add_argument("--patch-diff-threshold", type=float, default=0.06)
        sp.add_argument("--max-keep-components", type=int, default=10)

    sp1 = sub.add_parser("predict")
    sp1.add_argument("--input-dir", required=True)
    sp1.add_argument("--output-csv", default="submission.csv")
    add_common(sp1)

    sp2 = sub.add_parser("mock-eval")
    sp2.add_argument("--input-dir", required=True)
    sp2.add_argument("--mask-dir", required=True)
    add_common(sp2)
    return p


def main() -> None:
    parser = default_parser()
    args = parser.parse_args()
    set_seed(42)
    if args.mode == "predict":
        run_predict(args)
    elif args.mode == "mock-eval":
        run_mock_eval(args)
    else:
        raise ValueError(args.mode)


if __name__ == "__main__":
    main()
