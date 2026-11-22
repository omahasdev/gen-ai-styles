#!/usr/bin/env python3
"""Make the example matrix of comfyui/README.md: the ChatGPT examples and the two workflows, for all styles.

Column 1 has the source photo image-styles/_baseline.png. The next columns have one style each, in alphabetical
order. The rows are:

1. The ChatGPT example images of image-styles/
2. Enhancer (1-enhancer/): one row for each --enhancer-seeds value
3. Storyteller (2-storyteller/, story mode): one row for each --story-seeds value

The script makes no image. It takes the images that the example sheet tool of each workflow made:
1-enhancer/make_example_sheet.py puts them in <work>/seed<N>/, and 2-storyteller/make_example_sheet.py puts them
in one folder. It needs Pillow (the Python environment of ComfyUI has it).

Example:
    python comfyui/1-enhancer/make_example_sheet.py --work work/1 --sheet comfyui/1-enhancer/example-sheet.jpg
    python comfyui/2-storyteller/make_example_sheet.py --work work/2 --sheet comfyui/2-storyteller/example-sheet.jpg
    python comfyui/make_example_matrix.py --enhancer-work work/1 --story-work work/2 --sheet comfyui/example-matrix.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
STYLES_DIR = HERE.parent / "image-styles"
BASELINE = STYLES_DIR / "_baseline.png"
EXAMPLE_SUFFIX = ".example.png"

DEFAULT_ENHANCER_SEEDS = [1, 2, 3, 4, 5]  # all seeds of 1-enhancer/make_example_sheet.py
DEFAULT_STORY_SEEDS = [11, 12, 13]  # all story seeds of 2-storyteller/make_example_sheet.py
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


def enhancer_result(work: Path, seed: int, style_name: str) -> Path:
    """The Enhancer names a result <work>/seed<N>/<image>__<style>.png."""
    return work / f"seed{seed}" / f"{BASELINE.stem}__{style_name}.png"


def story_result(work: Path, seed: int, style_name: str) -> Path:
    """The Storyteller names a result '<work>/<image> (<style>, story-s<seed>) - <title>.png'."""
    matches = sorted(work.glob(f"{BASELINE.stem} ({style_name}, story-s{seed})*.png"))
    return matches[0] if matches else work / f"{BASELINE.stem} ({style_name}, story-s{seed}).png"


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


def build_sheet(enhancer_seeds: list[int], story_seeds: list[int], enhancer_work: Path, story_work: Path,
                sheet: Path) -> None:
    names = styles()
    rows = [("ChatGPT examples (image-styles/)", [STYLES_DIR / f"{n}{EXAMPLE_SUFFIX}" for n in names])]
    rows += [(f"Enhancer, seed {s}", [enhancer_result(enhancer_work, s, n) for n in names])
             for s in enhancer_seeds]
    rows += [(f"Storyteller, story mode, seed {s}", [story_result(story_work, s, n) for n in names]) for s in story_seeds]
    columns = ["Source photo"] + names

    width = GAP + len(columns) * (TILE_WIDTH + GAP)
    row_height = LABEL_HEIGHT + TILE_HEIGHT + GAP
    canvas = Image.new("RGB", (width, LABEL_HEIGHT + len(rows) * row_height), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    font = load_font()
    for col, name in enumerate(columns):
        draw.text((GAP + col * (TILE_WIDTH + GAP), GAP), short(name, font, draw), fill=TEXT_COLOR, font=font)
    for row, (label, paths) in enumerate(rows):
        top = LABEL_HEIGHT + row * row_height
        draw.text((GAP, top + GAP // 2), label, fill=TEXT_COLOR, font=font)
        for col, path in enumerate([BASELINE] + paths):
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
    ap.add_argument("--enhancer-work", required=True, help="the --work folder of 1-enhancer/make_example_sheet.py")
    ap.add_argument("--story-work", required=True, help="the --work folder of 2-storyteller/make_example_sheet.py")
    ap.add_argument("--sheet", required=True, help="path of the output JPEG")
    ap.add_argument("--enhancer-seeds", type=int, nargs="*", default=DEFAULT_ENHANCER_SEEDS)
    ap.add_argument("--story-seeds", type=int, nargs="*", default=DEFAULT_STORY_SEEDS)
    args = ap.parse_args()
    build_sheet(args.enhancer_seeds, args.story_seeds, Path(args.enhancer_work), Path(args.story_work),
                Path(args.sheet))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
