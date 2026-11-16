#!/usr/bin/env python3
"""Make the example sheet of comfyui/README.md.

Column 1 has the source photo image-styles/_baseline.png in each row. The next columns
have one style each. Row 1 has the ChatGPT example images of image-styles/. Each
next row has one seed of this workflow, made from the source photo.

The script calls style_transfer.py for each seed, then puts the images into
one JPEG. It needs Pillow (the Python environment of ComfyUI has it).

Examples:
    python comfyui/make_example_sheet.py --work sheet-work --sheet example-sheet.jpg
    python comfyui/make_example_sheet.py --work sheet-work --sheet example-sheet.jpg --seeds 1 2 3 --server http://127.0.0.1:8188
    python comfyui/make_example_sheet.py --work sheet-work --sheet example-sheet.jpg --skip-generation
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
STYLES_DIR = HERE.parent / "image-styles"
BASELINE = STYLES_DIR / "_baseline.png"
TRANSFER_SCRIPT = HERE / "style_transfer.py"
EXAMPLE_SUFFIX = ".example.png"

DEFAULT_SEEDS = [1, 2, 3, 4, 5]
DEFAULT_SERVER = "http://127.0.0.1:8188"
TILE_WIDTH = 260
TILE_HEIGHT = 390
GAP = 6
LABEL_HEIGHT = 26
FONT_SIZE = 16
FONT_FILES = ["arial.ttf", "DejaVuSans.ttf"]
JPEG_QUALITY = 85
BACKGROUND = "white"
TEXT_COLOR = "black"
MISSING_COLOR = "red"


def styles() -> list[str]:
    return sorted(p.stem for p in STYLES_DIR.glob("*.json"))


def generate(seeds: list[int], work: Path, server: str) -> None:
    for seed in seeds:
        cmd = [sys.executable, str(TRANSFER_SCRIPT), str(BASELINE), "--styles", str(STYLES_DIR),
               "--output", str(work / f"seed{seed}"), "--seed", str(seed), "--server", server]
        print(f"seed {seed}: generating", flush=True)
        subprocess.run(cmd, check=True)


def load_font():
    for name in FONT_FILES:
        try:
            return ImageFont.truetype(name, FONT_SIZE)
        except OSError:
            continue
    return ImageFont.load_default()


def fit(path: Path) -> Image.Image:
    image = Image.open(path).convert("RGB")
    scale = min(TILE_WIDTH / image.width, TILE_HEIGHT / image.height)
    return image.resize((round(image.width * scale), round(image.height * scale)), Image.LANCZOS)


def short(text: str, font, draw: ImageDraw.ImageDraw) -> str:
    while draw.textlength(text, font=font) > TILE_WIDTH and len(text) > 1:
        text = text[:-2] + "…"
    return text


def build_sheet(seeds: list[int], work: Path, sheet: Path, row_label: str) -> None:
    """Column 1 has the baseline photo in each row. The next columns have the styles."""
    names = ["Baseline (source)"] + styles()
    rows = [("ChatGPT examples (image-styles/)", [BASELINE] + [STYLES_DIR / f"{n}{EXAMPLE_SUFFIX}" for n in names[1:]])]
    rows += [(f"{row_label}, seed {s}", [BASELINE] + [work / f"seed{s}" / f"{BASELINE.stem}__{n}.png" for n in names[1:]])
             for s in seeds]

    width = GAP + len(names) * (TILE_WIDTH + GAP)
    row_height = LABEL_HEIGHT + TILE_HEIGHT + GAP
    height = LABEL_HEIGHT + len(rows) * row_height
    canvas = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    font = load_font()

    for col, name in enumerate(names):
        draw.text((GAP + col * (TILE_WIDTH + GAP), GAP), short(name, font, draw), fill=TEXT_COLOR, font=font)
    for row, (label, paths) in enumerate(rows):
        top = LABEL_HEIGHT + row * row_height
        draw.text((GAP, top + GAP // 2), label, fill=TEXT_COLOR, font=font)
        for col, path in enumerate(paths):
            left = GAP + col * (TILE_WIDTH + GAP)
            if path.exists():
                canvas.paste(fit(path), (left, top + LABEL_HEIGHT))
            else:
                draw.text((left, top + LABEL_HEIGHT), "missing", fill=MISSING_COLOR, font=font)

    sheet.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(sheet, quality=JPEG_QUALITY)
    print(f"sheet: {sheet} ({canvas.width}x{canvas.height})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True, help="folder for the generated images")
    ap.add_argument("--sheet", required=True, help="path of the output JPEG")
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS, help="one row for each seed")
    ap.add_argument("--server", default=DEFAULT_SERVER, help="ComfyUI address")
    ap.add_argument("--row-label", default="Qwen Image 2.1 Turbo + enhancer", help="label of the generated rows")
    ap.add_argument("--skip-generation", action="store_true", help="use the images that are already in --work")
    args = ap.parse_args()

    work = Path(args.work)
    if not args.skip_generation:
        generate(args.seeds, work, args.server)
    build_sheet(args.seeds, work, Path(args.sheet), args.row_label)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
