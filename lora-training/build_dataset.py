#!/usr/bin/env python3
"""Build a style LoRA training set from folders of frames: no near-duplicates, no dark or gray frames.

The script reads each input folder (also its subfolders), drops the frames that are too small, too dark, or too
gray, drops each frame that is nearly the same as a frame it kept (a perceptual difference hash), and crops black
letterbox bars. It takes at most --per-folder frames from each folder, in a random order fixed by --seed, so that
each source gets its share. The kept frames go to the output folder as JPEG files with numbered names.

Examples:
    python -I build_dataset.py frames/a frames/b --out dataset
    python -I build_dataset.py frames/a frames/b --out dataset --per-folder 100 --min-distance 32
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
HASH_SIDE = 16  # the difference hash has HASH_SIDE x HASH_SIDE bits
STATS_SIZE = (160, 90)  # the frame size for the brightness and saturation check
DEFAULT_MIN_DISTANCE = 40  # bits (of 256) that two kept frames must differ in
DEFAULT_MIN_BRIGHTNESS = 0.38  # mean HSV value of a usable frame, 0 to 1
DEFAULT_MIN_SATURATION = 0.22  # mean HSV saturation of a usable frame, 0 to 1
DEFAULT_MIN_WIDTH = 960  # pixels
BAR_BRIGHTNESS = 0.06  # a row darker than this is a letterbox bar
BAR_TOLERANCE = 2  # rows: a smaller bar is not cropped
JPEG_QUALITY = 95
DEFAULT_SEED = 7


def dhash(im: Image.Image) -> np.ndarray:
    """The difference hash: one bit for each pair of neighbor pixels of a small gray copy."""
    g = np.asarray(im.convert("L").resize((HASH_SIDE + 1, HASH_SIDE)), dtype=np.int16)
    return (g[:, 1:] > g[:, :-1]).flatten()


def brightness_saturation(im: Image.Image) -> tuple[float, float]:
    hsv = np.asarray(im.convert("RGB").resize(STATS_SIZE).convert("HSV"), dtype=np.float32) / 255
    return float(hsv[..., 2].mean()), float(hsv[..., 1].mean())


def crop_bars(im: Image.Image) -> Image.Image:
    """Remove black letterbox bars at the top and the bottom."""
    rows = np.where(np.asarray(im.convert("L"), dtype=np.float32).mean(axis=1) / 255 > BAR_BRIGHTNESS)[0]
    if len(rows) == 0:
        return im
    top, bottom = int(rows[0]), int(rows[-1]) + 1
    if top > BAR_TOLERANCE or bottom < im.height - BAR_TOLERANCE:
        return im.crop((0, top, im.width, bottom))
    return im


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", nargs="+", type=Path, help="folders of source frames")
    ap.add_argument("--out", required=True, type=Path, help="output folder (must be empty or not exist)")
    ap.add_argument("--per-folder", type=int, default=0, help="at most this many frames from each folder (0: all)")
    ap.add_argument("--min-distance", type=int, default=DEFAULT_MIN_DISTANCE,
                    help=f"near-duplicate limit in hash bits of {HASH_SIDE * HASH_SIDE}")
    ap.add_argument("--min-brightness", type=float, default=DEFAULT_MIN_BRIGHTNESS)
    ap.add_argument("--min-saturation", type=float, default=DEFAULT_MIN_SATURATION)
    ap.add_argument("--min-width", type=int, default=DEFAULT_MIN_WIDTH)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED, help="order of the frames in each folder")
    a = ap.parse_args()

    if a.out.exists() and any(a.out.iterdir()):
        print(f"{a.out} is not empty", file=sys.stderr)
        return 1
    a.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    kept: list[np.ndarray] = []
    for folder in a.folders:
        files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
        rng.shuffle(files)
        added = 0
        for path in files:
            if a.per_folder and added >= a.per_folder:
                break
            try:
                im = crop_bars(Image.open(path).convert("RGB"))
            except OSError:
                continue
            if im.width < a.min_width:
                continue
            value, saturation = brightness_saturation(im)
            if value < a.min_brightness or saturation < a.min_saturation:
                continue
            h = dhash(im)
            if any(int((h != k).sum()) < a.min_distance for k in kept):
                continue
            kept.append(h)
            added += 1
            im.save(a.out / f"frame_{len(kept):05d}.jpg", quality=JPEG_QUALITY)
        print(f"{folder}: {added} of {len(files)} kept")
    print(f"total {len(kept)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
