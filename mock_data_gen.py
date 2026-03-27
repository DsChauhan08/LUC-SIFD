#!/usr/bin/env python3
"""Generate a small synthetic biomedical-like dataset for local pipeline tests."""

from __future__ import annotations

import argparse
import os
import random
from typing import Tuple

import numpy as np
from PIL import Image, ImageFilter


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def smooth_noise(h: int, w: int) -> np.ndarray:
    a = np.random.normal(0.5, 0.15, size=(h, w)).astype(np.float32)
    a = np.clip(a, 0.0, 1.0)
    img = Image.fromarray((a * 255).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(radius=1.2)
    )
    return np.asarray(img, dtype=np.float32) / 255.0


def add_spots(img: np.ndarray, n_spots: int) -> np.ndarray:
    h, w = img.shape
    out = img.copy()
    for _ in range(n_spots):
        cy = random.randint(0, h - 1)
        cx = random.randint(0, w - 1)
        r = random.randint(4, 16)
        yy, xx = np.ogrid[:h, :w]
        d2 = (yy - cy) ** 2 + (xx - cx) ** 2
        blob = np.exp(-d2 / (2.0 * (r**2)))
        amp = random.uniform(-0.22, 0.28)
        out += amp * blob.astype(np.float32)
    return np.clip(out, 0.0, 1.0)


def make_background(h: int, w: int) -> np.ndarray:
    x = np.linspace(0.0, 1.0, w, dtype=np.float32)
    y = np.linspace(0.0, 1.0, h, dtype=np.float32)
    gx, gy = np.meshgrid(x, y)
    grad = 0.35 * gx + 0.2 * gy
    noise = smooth_noise(h, w)
    out = 0.45 * noise + 0.55 * grad
    out = add_spots(out, n_spots=random.randint(20, 45))
    return np.clip(out, 0.0, 1.0)


def random_patch_bounds(
    h: int, w: int, min_sz: int = 24, max_sz: int = 72
) -> Tuple[int, int, int, int]:
    ph = random.randint(min_sz, max_sz)
    pw = random.randint(min_sz, max_sz)
    y = random.randint(0, h - ph)
    x = random.randint(0, w - pw)
    return y, x, ph, pw


def place_patch_non_overlap(
    h: int, w: int, ph: int, pw: int, sy: int, sx: int
) -> Tuple[int, int]:
    for _ in range(200):
        dy = random.randint(0, h - ph)
        dx = random.randint(0, w - pw)
        if abs(dy - sy) + abs(dx - sx) > min(h, w) // 5:
            return dy, dx
    return max(0, h - ph), max(0, w - pw)


def forge_copy_move(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h, w = img.shape
    y, x, ph, pw = random_patch_bounds(h, w)
    dy, dx = place_patch_non_overlap(h, w, ph, pw, y, x)
    out = img.copy()
    patch = out[y : y + ph, x : x + pw].copy()
    if random.random() < 0.4:
        patch = np.flipud(patch)
    if random.random() < 0.4:
        patch = np.fliplr(patch)
    alpha = random.uniform(0.86, 1.0)
    out[dy : dy + ph, dx : dx + pw] = np.clip(
        alpha * patch + (1.0 - alpha) * out[dy : dy + ph, dx : dx + pw], 0.0, 1.0
    )
    m = np.zeros((h, w), dtype=np.uint8)
    m[y : y + ph, x : x + pw] = 1
    m[dy : dy + ph, dx : dx + pw] = 1
    return out, m


def generate(
    out_img_dir: str, out_mask_dir: str, n: int, forged_ratio: float, size: int
) -> None:
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)
    forged_target = int(round(n * forged_ratio))
    ids = list(range(1, n + 1))
    random.shuffle(ids)
    forged_set = set(ids[:forged_target])

    for i in range(1, n + 1):
        img = make_background(size, size)
        mask = np.zeros((size, size), dtype=np.uint8)
        if i in forged_set:
            img, mask = forge_copy_move(img)

        ip = os.path.join(out_img_dir, f"{i}.png")
        mp = os.path.join(out_mask_dir, f"{i}.npy")
        Image.fromarray((img * 255).astype(np.uint8)).save(ip)
        np.save(mp, mask)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-img-dir", default="mock/images")
    ap.add_argument("--out-mask-dir", default="mock/masks")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--forged-ratio", type=float, default=0.5)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    generate(args.out_img_dir, args.out_mask_dir, args.n, args.forged_ratio, args.size)
    print(
        f"generated n={args.n} images at {args.out_img_dir} and masks at {args.out_mask_dir}"
    )


if __name__ == "__main__":
    main()
