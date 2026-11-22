#!/usr/bin/env python3
"""Caption the frames of a training set with a local vision model (an OpenAI-compatible server, for example
LM Studio).

Each caption starts with the trigger words, then tells what the frame shows (people, poses, expressions, clothes,
place, light, shot) in plain words. The caption names no work, studio, or character, and no medium: the LoRA must
learn the drawing manner under the trigger, not a name. The caption goes into a .txt file next to the image.

Examples:
    python -I caption.py dataset --trigger "mystyle style"
    python -I caption.py dataset --trigger "mystyle style" --model qwen/qwen3-vl-8b --force
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import re
import sys
import urllib.request
from pathlib import Path

from PIL import Image

DEFAULT_SERVER = "http://127.0.0.1:1234/v1/chat/completions"
DEFAULT_MODEL = "qwen/qwen3-vl-8b"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
SEND_LONG_SIDE = 1024  # pixels: the frame size sent to the vision model
JPEG_QUALITY = 90
MAX_TOKENS = 300
TEMPERATURE = 0.2
TIMEOUT_S = 300
PROMPT = (
    "Describe this picture in one paragraph of 40 to 70 words for an image training caption. "
    "Tell what it shows: each person (hair color and style, eye color, clothes, pose, facial expression), "
    "the place and the background, the light, and the shot (close-up, medium shot, wide shot). "
    "Use plain visual words. Do not name the picture, a series, a studio, or a character, and do not say "
    "'anime', 'cartoon', 'screenshot', 'illustration', or 'drawing'. No text in the picture. Start directly "
    "with the content."
)
MEDIUM_WORDS = re.compile(r"\b(anime|cartoon|screenshot|illustration|drawing|animated)\b", re.I)
THINK_BLOCK = re.compile(r"<think>.*?</think>", re.S)


def ask(server: str, model: str, path: Path) -> str:
    im = Image.open(path).convert("RGB")
    im.thumbnail((SEND_LONG_SIDE, SEND_LONG_SIDE))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=JPEG_QUALITY)
    data = base64.b64encode(buf.getvalue()).decode()
    body = {"model": model, "max_tokens": MAX_TOKENS, "temperature": TEMPERATURE,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{data}"}}]}]}
    req = urllib.request.Request(server, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
        text = json.load(r)["choices"][0]["message"]["content"]
    text = MEDIUM_WORDS.sub("", " ".join(THINK_BLOCK.sub("", text).split()))
    return re.sub(r" +", " ", text).strip()


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path, help="folder of training images")
    ap.add_argument("--trigger", required=True, help='trigger words at the start of each caption, for example "mystyle style"')
    ap.add_argument("--server", default=DEFAULT_SERVER, help="chat completions address of the vision model")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--force", action="store_true", help="write a caption also when the .txt file exists")
    a = ap.parse_args()
    images = sorted(p for p in a.folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    for i, path in enumerate(images, 1):
        out = path.with_suffix(".txt")
        if out.exists() and not a.force:
            continue
        caption = f"{a.trigger}. {ask(a.server, a.model, path)}"
        out.write_text(caption, encoding="utf-8")
        print(f"{i}/{len(images)} {path.name}: {caption[:120]}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
