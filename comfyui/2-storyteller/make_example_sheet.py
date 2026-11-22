#!/usr/bin/env python3
"""Make the example sheet of comfyui/2-storyteller/README.md.

Column 1 has the source photo image-styles/_baseline.png in each row. The next
columns have one style each. The rows are:

1. The ChatGPT example images of image-styles/
2. One row for each adapt seed: adapt mode, a new LLM seed for each row
3. One row for each story seed: story mode, a new story card for each row

Each --variant adds one column directly after the --variant-style column: that style with extra options of
style_transfer.py, for example a LoRA. The variant results go into a subfolder of --work. In a variant column,
the ChatGPT row shows the example of --variant-style. A run skips each result that is already in --work.

The script calls style_transfer.py for each seed, then puts the images into
one JPEG. It needs Pillow (the Python environment of ComfyUI has it).

By default (--layout flat), all generations go into the --work folder, and
their names tell them apart: <image>__<style>__<mode>-s<seed>[__<story slug>].png.
With --layout folders, each mode and seed gets its own subfolder.

Examples:
    python comfyui/2-storyteller/make_example_sheet.py --work sheet-work --sheet example-sheet.jpg
    python comfyui/2-storyteller/make_example_sheet.py --work sheet-work --sheet example-sheet.jpg --adapt-seeds 1 2 --story-seeds 5 6
    python comfyui/2-storyteller/make_example_sheet.py --work sheet-work --sheet example-sheet.jpg --skip-generation
    python comfyui/2-storyteller/make_example_sheet.py --work sheet-work --sheet example-sheet.jpg \
        --variant "Anime + generic LoRA=--no-loras --extra-lora Nick_2anime_QwenImage2.1.safetensors:0.8:2anime" \
        --variant "Anime + own LoRA=--no-loras --extra-lora 'my_lora.safetensors:1.0:mystyle style'"
"""

from __future__ import annotations

import argparse
import re
import shlex
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
STYLES_DIR = HERE.parents[1] / "image-styles"
BASELINE = STYLES_DIR / "_baseline.png"
TRANSFER_SCRIPT = HERE / "style_transfer.py"
EXAMPLE_SUFFIX = ".example.png"

DEFAULT_ADAPT_SEEDS = [1, 2, 3]
DEFAULT_STORY_SEEDS = [11, 12, 13]
DEFAULT_SERVER = "http://127.0.0.1:8188"
DEFAULT_VARIANT_STYLE = "Anime Isekai Comedy"
MODE_LABELS = {"adapt": "Adapt (no story)", "story": "Story"}
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


LAYOUTS = ("flat", "folders")


def run_dir(work: Path, mode: str, seed: int, layout: str) -> Path:
    """flat: all results in one folder, told apart by their names. folders: one folder for each run."""
    return work if layout == "flat" else work / f"{mode}-seed{seed}"


def find_result(work: Path, mode: str, seed: int, style_name: str, layout: str) -> Path:
    """The result file of one style and run. Names are <baseline>__<style>__<mode>-s<seed>[__<slug>].png."""
    folder = run_dir(work, mode, seed, layout)
    # "<image> (<style>, <mode>-s<seed>) - <title>.png", or without a title ("--no-title")
    matches = sorted(folder.glob(f"{BASELINE.stem} ({style_name}, {mode}-s{seed}) - *.png"))
    matches = matches or sorted(folder.glob(f"{BASELINE.stem} ({style_name}, {mode}-s{seed}).png"))
    return matches[0] if matches else folder / f"{BASELINE.stem} ({style_name}, {mode}-s{seed}).png"


def parse_variant(text: str) -> tuple[str, list[str]]:
    """A variant column: "LABEL=OPTIONS", the options of style_transfer.py in shell syntax."""
    label, _, options = text.partition("=")
    if not label.strip():
        raise argparse.ArgumentTypeError(f"a variant needs LABEL=OPTIONS: {text}")
    return label.strip(), shlex.split(options)


def slug(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")


def generate(mode: str, seeds: list[int], work: Path, server: str, layout: str) -> None:
    for seed in seeds:
        if all(find_result(work, mode, seed, n, layout).exists() for n in styles()):
            print(f"{mode} seed {seed}: exists, skipped", flush=True)
            continue
        cmd = [sys.executable, str(TRANSFER_SCRIPT), str(BASELINE), "--styles", str(STYLES_DIR),
               "--output", str(run_dir(work, mode, seed, layout)), "--seed", str(seed), "--server", server,
               "--mode", mode]  # --seed sets the image, prompt, and story seeds: each row gets a new prompt
        print(f"{mode} seed {seed}: generating", flush=True)
        subprocess.run(cmd, check=True)


def generate_variant(style: str, label: str, options: list[str], groups, work: Path, server: str) -> None:
    folder = work / slug(label)
    for mode, seeds in groups:
        for seed in seeds:
            if find_result(folder, mode, seed, style, "flat").exists():
                print(f"{label}, {mode} seed {seed}: exists, skipped", flush=True)
                continue
            print(f"{label}, {mode} seed {seed}: generating", flush=True)
            subprocess.run([sys.executable, str(TRANSFER_SCRIPT), str(BASELINE),
                            "--styles", str(STYLES_DIR / f"{style}.json"), "--output", str(folder),
                            "--seed", str(seed), "--server", server, "--mode", mode, *options], check=True)


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


def build_sheet(groups: list[tuple[str, list[int]]], work: Path, sheet: Path, model_label: str, layout: str,
                variant_style: str = DEFAULT_VARIANT_STYLE, variants=()) -> None:
    """Column 1 has the baseline photo in each row. The next columns have the styles. The variant columns come
    directly after the column of --variant-style."""
    # one column: (header, style, folder of the results, layout of the folder)
    columns = []
    for n in styles():
        columns.append((n, n, work, layout))
        if n == variant_style:
            columns += [(label, n, work / slug(label), "flat") for label, _ in variants]
    names = ["Baseline (source)"] + [header for header, _, _, _ in columns]
    rows = [("ChatGPT examples (image-styles/)",
             [BASELINE] + [STYLES_DIR / f"{style}{EXAMPLE_SUFFIX}" for _, style, _, _ in columns])]
    for mode, seeds in groups:
        for s in seeds:
            label = f"{model_label}, {MODE_LABELS[mode]}, seed {s}"
            rows.append((label, [BASELINE] + [find_result(folder, mode, s, style, lay)
                                              for _, style, folder, lay in columns]))

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
    ap.add_argument("--adapt-seeds", type=int, nargs="*", default=DEFAULT_ADAPT_SEEDS,
                    help="one adapt-mode row for each seed (the seed is also the LLM seed)")
    ap.add_argument("--story-seeds", type=int, nargs="*", default=DEFAULT_STORY_SEEDS,
                    help="one story-mode row for each seed")
    ap.add_argument("--server", default=DEFAULT_SERVER, help="ComfyUI address")
    ap.add_argument("--model-label", default="Qwen Image 2.1 Turbo", help="label of the generated rows")
    ap.add_argument("--skip-generation", action="store_true", help="use the images that are already in --work")
    ap.add_argument("--layout", choices=LAYOUTS, default="flat",
                    help="flat: all results in --work, told apart by their file names (default). "
                         "folders: one subfolder for each mode and seed")
    ap.add_argument("--variant", action="append", type=parse_variant, default=[], metavar="LABEL=OPTIONS",
                    help="one more column: --variant-style with extra options of style_transfer.py, for example a LoRA")
    ap.add_argument("--variant-style", default=DEFAULT_VARIANT_STYLE, help="the style of the variant columns")
    args = ap.parse_args()

    work = Path(args.work)
    groups = [("adapt", args.adapt_seeds), ("story", args.story_seeds)]
    if not args.skip_generation:
        for mode, seeds in groups:
            generate(mode, seeds, work, args.server, args.layout)
        for label, options in args.variant:
            generate_variant(args.variant_style, label, options, groups, work, args.server)
    build_sheet(groups, work, Path(args.sheet), args.model_label, args.layout, args.variant_style, args.variant)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
