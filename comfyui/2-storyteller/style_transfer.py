#!/usr/bin/env python3
"""Redraw images in the styles of image-styles/*.json with a running ComfyUI server.

The script uses only the Python standard library. It reads the API graph
style-transfer.api.json, which is next to this file, and for each pair of
(image, style) it:

1. Uploads the image to the server (one time for each image).
2. Makes the prompt with one of three modes:
   - direct: the style JSON goes straight to the image model.
   - adapt: an LLM adapts the style JSON to this image and writes the prompt.
   - story: the storyteller adds a scene concept, and the LLM writes a prompt
     for a new picture of the subject.
3. Puts the job in the server queue.

It puts all jobs in the queue first, so that the GPU does not wait between
jobs. Then it downloads each result as "<image> (<style>, <mode>-s<seed>) - <title>.png".

Examples:
    python style_transfer.py photo.jpg --output out
    python style_transfer.py photo.jpg --mode direct --seed 1 --output out
    python style_transfer.py photos/ --styles "../../image-styles/Art Nouveau Watercolor.json" --output out
    python style_transfer.py photo.jpg --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import mimetypes
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_GRAPH = HERE / "style-transfer.api.json"
DEFAULT_STYLES = HERE.parents[1] / "image-styles"
DEFAULT_SERVER = "http://127.0.0.1:8188"  # ComfyUI default address
STORYTELLER_DIR = HERE.parents[1] / "storyteller"
DEFAULT_STORY_CONFIG = STORYTELLER_DIR / "storyteller.json"

sys.path.insert(0, str(STORYTELLER_DIR))
import storyteller  # noqa: E402  (the module is in storyteller/ of the repository root)
sys.path.insert(0, str(HERE))  # python -I does not add the script folder
import style_schema  # noqa: E402  (next to this file)
import prompt_hygiene  # noqa: E402  (next to this file)
import graph_check  # noqa: E402  (next to this file)
import lora_routing  # noqa: E402  (next to this file)
import source_cleanup  # noqa: E402  (next to this file)

DEFAULT_LORAS = HERE / "loras.json"  # the local LoRA registry; absent: no LoRA
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
STYLE_SKIP_PREFIX = "_"  # files such as _baseline.png are not styles

# Node ids in style-transfer.api.json that the script changes.
NODE_IMAGE = "4"
NODE_ENCODE = "5"
NODE_CACHE = "6"
NODE_GUIDER = "7"
NODE_NOISE = "10"
NODE_SAMPLE = "11"
NODE_SAVE = "13"

POLL_INTERVAL_S = 1.0
HTTP_TIMEOUT_S = 60
JOB_TIMEOUT_S = 7200  # a job waits for every job queued before it; with thinking each takes up to ~10 min
SEED_BITS = 48  # RandomNoise accepts large seeds, keep them readable

PLACEHOLDER_RE = re.compile(r"\{([A-Z_]+)\}")
SOURCE_TAG = "<image1>"  # Qwen Image 2.1 refers to reference image 1 with this tag

# Prompt enhancer (adapt and story modes): the official Qwen 3.5 9B image-to-image prompt
# enhancer of Qwen Image 2.1. The sampling values are the values of its release.
ENHANCER_MODEL = "qwen3.5_9b_qwen_image_2.1_pe_i2i.int8_convrot.safetensors"
ENHANCER_SYSTEM_PROMPT = HERE / "style-enhancer.system.md"
ENHANCER_MAX_TOKENS = 2000  # without thinking: room for ~1,200 words, a hard stop against a runaway prompt
ENHANCER_MAX_TOKENS_THINKING = 16384  # with thinking: the reasoning block often needs more than 8,000 tokens
ENHANCER_SAMPLING = {"temperature": 1.0, "top_k": 20, "top_p": 0.95, "min_p": 0.0,
                     "repetition_penalty": 1.0, "presence_penalty": 0.0}
NODE_ENH_CLIP, NODE_ENH_SYSTEM, NODE_ENH_GENERATE, NODE_ENH_SHOW = "100", "101", "102", "103"
OUTLINE_INDENT = "  "

# Title of each result, for the file name (story and adapt modes): like a ChatGPT image title.
NODE_DECODE = "12"
NODE_TITLE_GENERATE, NODE_TITLE_SHOW = "104", "105"
TITLE_SYSTEM_PROMPT = "You name pictures. You answer with the title only."
TITLE_REQUEST = ("Write a title for this picture in {language}. The title is a short noun phrase of 3 to 6 words, "
                 "not a sentence and with no verb. Name the main subject and the place or the story element. "
                 "Examples of the form: Cat in a vintage dining room; Cat and a miniature tea party; "
                 "Cat, coffee, and TV static; Striped cat in a waiting room. "
                 "No quotation marks, no art style name, no period at the end.")
TITLE_MAX_TOKENS = 40
TITLE_MAX_CHARS = 60
DEFAULT_TITLE_LANGUAGE = "English"


# --------------------------------------------------------------------------- prompt


def load_style(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def style_placeholders(gen: dict) -> dict[str, str]:
    """Map each placeholder key of generation_instructions to its token, for example subject -> SUBJECT."""
    result = {}
    for key, value in gen.items():
        if key.endswith("_placeholder") and isinstance(value, str):
            match = PLACEHOLDER_RE.fullmatch(value.strip())
            if match:
                result[key[: -len("_placeholder")]] = match.group(1)
    return result


def fill_placeholders(node, fills: dict[str, str], tokens: dict[str, str]):
    """Replace each {TOKEN} in the strings of a JSON tree with the value that the user gave."""
    if isinstance(node, dict):
        return {k: fill_placeholders(v, fills, tokens) for k, v in node.items()}
    if isinstance(node, list):
        return [fill_placeholders(v, fills, tokens) for v in node]
    if isinstance(node, str):
        def repl(match):
            token = match.group(1)
            key = tokens.get(token, token.lower())
            return fills.get(key) or fills.get(token.lower()) or match.group(0)
        return PLACEHOLDER_RE.sub(repl, node)
    return node


def without_negatives(node):
    """The style without its "no" rules: every list under NEGATIVE_LIST_KEYS and negative_prompt."""
    if isinstance(node, dict):
        return {k: without_negatives(v) for k, v in node.items()
                if not style_schema.is_negative_key(k)}
    if isinstance(node, list):
        return [without_negatives(x) for x in node]
    return node


def style_spec(style: dict, fills: dict[str, str], rules_positive: str) -> dict:
    """The style with the placeholders filled; without its "no" rules when --rules-positive none."""
    gen = style["generation_instructions"]
    tokens = {token: key for key, token in style_placeholders(gen).items()}
    spec = fill_placeholders(style, fills, tokens)
    return without_negatives(spec) if rules_positive == "none" else spec


def build_direct_prompt(style: dict, fills: dict[str, str], rules_positive: str, fmt: str = "json") -> str:
    """Direct mode: give the image model the style definition as structured JSON.

    The text encoder of Qwen Image 2.1 is the Qwen3-VL 8B language model, and
    the tokenizer does not truncate. Thus the model reads all 16 sections with
    their labels. --rules-positive sets what the positive prompt does with the
    "no" rules: none (removed), plain (they stay in the JSON), explained (they
    stay, and the header says that they must not appear).
    """
    spec = style_spec(style, fills, rules_positive)
    unfilled = sorted({m for m in PLACEHOLDER_RE.findall(json.dumps(spec, ensure_ascii=False))})
    name = style.get("style_name", "the target style")
    extra = [k for k in NEGATIVE_LIST_KEYS[NEGATIVE_LIST_KEYS.index("do_not_reproduce"):]
             if f'"{k}"' in json.dumps(spec)]
    extra_text = "".join(f', every \"{k}\" list' for k in extra)
    negatives = ("" if rules_positive != "explained" else
                 f"Treat \"negative_prompt\", every \"avoid\" list{extra_text}, and \"explicitly_not_the_style\" "
                 f"as things that must not appear in the image. ")
    header = (
        f'Transform {SOURCE_TAG} completely into the art style "{name}". '
        f"The JSON below is the full specification of this style. Obey every section. "
        f"Sections near the top of generation_instructions.priority_order (if present) "
        f"win when two rules conflict. {negatives}"
        f"Keep the identity and the pose of the subject of {SOURCE_TAG}, and apply the style "
        f"to the whole image, including the background."
    )
    if unfilled:
        header += (" These placeholders have no value, so derive them from "
                   f"{SOURCE_TAG}: " + ", ".join("{" + t + "}" for t in unfilled) + ".")
    if fmt == "outline":
        # Qwen-Image draws double-quoted strings as text: give the style without any quotes
        header = header.replace("The JSON below", "The outline below").replace('"', "")
        return f"{header}\n\nStyle specification:\n" + "\n".join(outline(spec))
    body = json.dumps(spec, ensure_ascii=False, separators=(",", ":"))
    return f"{header}\n\nStyle specification (JSON):\n{body}"


def outline(node, depth: int = 0) -> list[str]:
    """Render a JSON tree as an indented outline without quotes.

    The keys, the nesting, and the order stay the same as in the JSON. The
    prompt enhancer treats text in double quotes as text to paint into the
    image, so the outline has no quotes.
    """
    pad = OUTLINE_INDENT * depth
    lines = []
    if isinstance(node, dict):
        for key, value in node.items():
            if isinstance(value, (dict, list)):
                lines.append(f"{pad}{key}:")
                lines += outline(value, depth + 1)
            else:
                lines.append(f"{pad}{key}: {scalar(value)}")
    elif isinstance(node, list):
        for item in node:
            if isinstance(item, (dict, list)):
                lines.append(f"{pad}-")
                lines += outline(item, depth + 1)
            else:
                lines.append(f"{pad}- {scalar(item)}")
    else:
        lines.append(f"{pad}{scalar(node)}")
    return lines


def scalar(value) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value).replace('"', "")


def build_enhancer_request(style: dict, fills: dict[str, str], rules_positive: str) -> str:
    """Adapt mode: the user turn of the prompt enhancer, the task and the style as an outline.

    The enhancer turns the "no" rules into positive statements (system prompt), so --rules-positive plain and
    explained are the same here. With --rules-positive none, the rules are not in the outline.
    """
    spec = style_spec(style, fills, rules_positive)
    return ("Redraw the input image in the art style that this specification defines.\n\n"
            "Style specification:\n" + "\n".join(outline(spec)))


# Reference size from the style's rendering, for every prompt mode. It reads only generic schema fields
# and medium words, never a style name, so a new style file gets its size by the same formula.
HAND_MEDIUM_FIELDS = ("line_art_prominence", "painterly_rendering")
HAND_MEDIUM_WORDS = ("watercolor", "ink", "gouache", "pencil", "illustration", "illustrated", "cartoon",
                     "drawn", "painted", "painting", "linework", "line art", "comic", "anime")
PHOTO_MEDIUM_WORDS = ("photograph", "photographic", "film still", "cinematic", "photoreal", "live-action")
WORD_DISTANCE_HAND = 0.8    # a style that names a hand-made medium in words, without numbers
WORD_DISTANCE_PHOTO = 0.15  # a style that names a photographic medium in words
WORD_DISTANCE_UNKNOWN = 0.5
REFERENCE_MIN_FRACTION = 0.4  # a fully drawn style: 40 % of the canvas side (about 500 px on 1024 x 1536)
REFERENCE_STEP = 32


def photo_distance(style: dict) -> tuple[float, str]:
    """How far the style's medium is from a photograph, 0 (photo) to 1 (fully drawn), and why."""
    rendering = style.get("rendering") if isinstance(style.get("rendering"), dict) else {}
    realism = rendering.get("realism_level")
    hand = [rendering[k] for k in HAND_MEDIUM_FIELDS if isinstance(rendering.get(k), (int, float))]
    if isinstance(realism, (int, float)) or hand:
        parts = ([1 - realism] if isinstance(realism, (int, float)) else []) + hand
        return round(min(1.0, max(0.0, max(parts))), 2), f"rendering fields (1 - realism_level, {', '.join(HAND_MEDIUM_FIELDS)})"
    text = " ".join(str(rendering.get(k, "")) for k in ("medium", "technique", "visual_finish")).lower()
    if any(w in text for w in PHOTO_MEDIUM_WORDS):
        return WORD_DISTANCE_PHOTO, "photographic medium words in rendering"
    if any(w in text for w in HAND_MEDIUM_WORDS):
        return WORD_DISTANCE_HAND, "hand-made medium words in rendering"
    return WORD_DISTANCE_UNKNOWN, "no rendering information"


def style_reference_size(style: dict, canvas_side: int) -> dict:
    """The reference size for this style on this canvas: small for a drawn medium, the full canvas for a photo.

    canvas_side is the side of a square with the canvas area (1254 for 1024 x 1536). A fully drawn style gets
    REFERENCE_MIN_FRACTION of it, a photographic style all of it.
    """
    distance, why = photo_distance(style)
    fraction = 1 - distance * (1 - REFERENCE_MIN_FRACTION)
    size = int(round(canvas_side * fraction / REFERENCE_STEP) * REFERENCE_STEP)
    return {"photo_distance": distance, "photo_distance_source": why, "reference_size": size}


# Output canvas: the three sizes of ChatGPT images.
CANVASES = {"portrait": (1024, 1536), "landscape": (1536, 1024), "square": (1024, 1024)}
CANVAS_AUTO = "auto"


def source_canvas(path: Path) -> str:
    """The canvas whose aspect ratio is nearest to the source image (in log space, so 2:3 and 3:2 are symmetric)."""
    size = image_size(path)
    if not size:
        return "square"
    ratio = size[0] / size[1]
    return min(CANVASES, key=lambda name: abs(math.log(ratio * CANVASES[name][1] / CANVASES[name][0])))


CANVAS_SOURCE = "source"  # the exact aspect ratio of the source image, with the area of a ChatGPT image
CANVAS_AREA = 1024 * 1536
# The widest aspect ratio that the Qwen-Image model card recommends (16:9, 1664 x 928). A wider or taller
# source gets a canvas at this limit.
CANVAS_MAX_RATIO = 16 / 9


RATIO_PREFIX = "ratio:"  # a canvas of a fixed aspect ratio W:H at CANVAS_AREA, for example "ratio:3:4"
RATIO_RE = re.compile(r"\b(\d{1,2})\s*:\s*(\d{1,2})\b")
RATIO_PATHS = ("aspect_ratio", "preferred_aspect_ratio")
RATIO_FLEXIBLE_WORDS = ("source", "flexible", "preserve")


def style_fixed_ratio(style: dict) -> tuple[int, int] | None:
    """The one aspect ratio that the style fixes (for example "3:4 portrait"), or None: no ratio, several ratios, or
    a ratio that defers to the source or is flexible."""
    composition = style.get("composition") if isinstance(style.get("composition"), dict) else {}
    text = " ".join(str(composition.get(k, "")) for k in RATIO_PATHS).lower()
    if any(w in text for w in RATIO_FLEXIBLE_WORDS):
        return None
    found = RATIO_RE.findall(text)
    return (int(found[0][0]), int(found[0][1])) if len(found) == 1 else None


def story_canvas(card: str, path: Path) -> str:
    """The canvas of the Format line of a story card; the source aspect ratio if the line asks for it or is missing."""
    value = storyteller.card_value(card, "Format").lower()
    for name in CANVASES:
        if value.startswith(name):
            return name
    return CANVAS_SOURCE


def canvas_size(canvas: str, path: Path) -> tuple[int, int]:
    """Width and height of the canvas; for "source", the source aspect ratio at CANVAS_AREA, multiples of 32."""
    if canvas in CANVASES:
        return CANVASES[canvas]
    if canvas.startswith(RATIO_PREFIX):
        w, h = (int(x) for x in canvas[len(RATIO_PREFIX):].split(":"))
        size = (w, h)
    else:
        size = image_size(path) or (1, 1)
    ratio = min(CANVAS_MAX_RATIO, max(1 / CANVAS_MAX_RATIO, size[0] / size[1]))
    width = round(math.sqrt(CANVAS_AREA * ratio) / LATENT_MULTIPLE) * LATENT_MULTIPLE
    height = round(math.sqrt(CANVAS_AREA / ratio) / LATENT_MULTIPLE) * LATENT_MULTIPLE
    return max(LATENT_MULTIPLE, width), max(LATENT_MULTIPLE, height)


LATENT_MULTIPLE = 32
NODE_OUTPUT_LATENT = "14"


EXIF_ORIENTATION_TAG = 0x0112
EXIF_ROTATED = (5, 6, 7, 8)  # orientations that turn the image by 90 degrees: width and height swap


def jpeg_orientation(segment: bytes) -> int:
    """The EXIF orientation of a JPEG APP1 segment (after the length bytes), or 1."""
    if segment[:6] != b"Exif\x00\x00":
        return 1
    tiff = segment[6:]
    order = {b"II": "little", b"MM": "big"}.get(tiff[:2])
    if not order:
        return 1
    offset = int.from_bytes(tiff[4:8], order)
    count = int.from_bytes(tiff[offset:offset + 2], order)
    for n in range(count):
        entry = tiff[offset + 2 + 12 * n: offset + 14 + 12 * n]
        if len(entry) == 12 and int.from_bytes(entry[:2], order) == EXIF_ORIENTATION_TAG:
            return int.from_bytes(entry[8:10], order)
    return 1


def image_size(path: Path) -> tuple[int, int] | None:
    """Width and height of a PNG, JPEG, WebP, or BMP file as displayed, read from its header (standard library
    only). A JPEG with an EXIF orientation that turns it by 90 degrees gives the turned size."""
    data = path.read_bytes()
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")
    if data[:2] == b"\xff\xd8":
        i, orientation = 2, 1
        while i + 9 < len(data):
            if data[i] != 0xFF:
                i += 1
                continue
            marker, length = data[i + 1], int.from_bytes(data[i + 2:i + 4], "big")
            if marker == 0xE1:
                orientation = jpeg_orientation(data[i + 4:i + 2 + length])
            if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
                width, height = int.from_bytes(data[i + 7:i + 9], "big"), int.from_bytes(data[i + 5:i + 7], "big")
                return (height, width) if orientation in EXIF_ROTATED else (width, height)
            i += 2 + length
        return None
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        chunk = data[12:16]
        if chunk == b"VP8 ":
            return int.from_bytes(data[26:28], "little") & 0x3FFF, int.from_bytes(data[28:30], "little") & 0x3FFF
        if chunk == b"VP8L":
            bits = int.from_bytes(data[21:25], "little")
            return (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
        if chunk == b"VP8X":
            return int.from_bytes(data[24:27], "little") + 1, int.from_bytes(data[27:30], "little") + 1
        return None
    if data[:2] == b"BM":
        return int.from_bytes(data[18:22], "little", signed=True), abs(int.from_bytes(data[22:26], "little", signed=True))
    return None


def seed_tag(mode: str, seeds: dict[str, int]) -> str:
    """<mode>-s<N> when every seed of the mode is the same, else <mode>-i<image>-p<prompt>-c<story card>."""
    used = [("i", seeds["image"])]
    if mode in ("adapt", "story"):
        used.append(("p", seeds["prompt"]))
    if mode == "story":
        used.append(("c", seeds["story"]))
    if len({value for _, value in used}) == 1:
        return f"{mode}-s{used[0][1]}"
    return mode + "".join(f"-{letter}{value}" for letter, value in used)


def result_name(image_path: Path, style_path: Path, tags: list[str]) -> str:
    """<image> (<style>, <mode and seeds>[, <option tags>]): the title or the story slug follows after " - "."""
    return f"{image_path.stem} ({style_path.stem}, {', '.join(tags)})"


RETRY_NO_THINKING_TAG = "retry-nothink"  # the job ran again without thinking (the written prompt was empty)
RETRY_THINKING_TAG = "retry-think"  # the job ran again with thinking (the written prompt had a leak)


def with_tag(stem: str, tag: str) -> str:
    """A result_name stem with one more tag before its closing parenthesis."""
    return f"{stem[:-1]}, {tag})" if stem.endswith(")") else f"{stem} ({tag})"


NEGATIVE_LIST_KEYS = style_schema.NEGATIVE_LIST_KEYS


def build_negative(style: dict) -> str:
    """All "no" rules of the style for the negative prompt: negative_prompt, every avoid list,
    and explicitly_not_the_style. The image model uses it only when cfg is more than 1."""
    items = [s.strip() for s in style["generation_instructions"].get("negative_prompt", "").split(",") if s.strip()]

    def walk(node, key=""):
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, k)
        elif isinstance(node, list) and key in NEGATIVE_LIST_KEYS:
            items.extend(x.strip() for x in node if isinstance(x, str) and x.strip())

    walk(style)
    seen, unique = set(), []
    for item in items:
        if item.lower() not in seen:
            seen.add(item.lower())
            unique.append(item)
    return ", ".join(unique)


def build_story_request(style: dict, fills: dict[str, str], story_seed: int, config: dict, rules_positive: str) -> str:
    """Story mode: the storyteller adds a scene concept for the story seed in front of the style outline."""
    spec = style_spec(style, fills, rules_positive)
    return storyteller.build_request(spec, config, story_seed, "\n".join(outline(spec)))


WORD_TARGET_RE = re.compile(r"Write \d+ to \d+ words\.")


def set_word_target(system_prompt: str, words: str | None) -> str:
    """Replace the length rule of the enhancer system prompt, for example words="800-1200"."""
    if not words:
        return system_prompt
    low, _, high = words.partition("-")
    return WORD_TARGET_RE.sub(f"Write {int(low)} to {int(high or low)} words.", system_prompt)


def add_enhancer(graph: dict, request: str, system_prompt: str, seed: int, thinking: bool,
                 max_tokens: int = ENHANCER_MAX_TOKENS, identity: bool | str = False,
                 identity_request: str | None = None) -> None:
    """Insert the prompt enhancer: it reads the image and the full style, and writes the edit prompt."""
    graph[NODE_ENH_CLIP] = {"class_type": "CLIPLoader", "_meta": {"title": "Prompt enhancer model"},
                            "inputs": {"clip_name": ENHANCER_MODEL, "type": "qwen_image", "device": "default"}}
    graph[NODE_ENH_SYSTEM] = {"class_type": "PrimitiveStringMultiline", "_meta": {"title": "Enhancer system prompt"},
                              "inputs": {"value": system_prompt}}
    sampling = {f"sampling_mode.{k}": v for k, v in ENHANCER_SAMPLING.items()}
    graph[NODE_ENH_GENERATE] = {"class_type": "TextGenerate", "_meta": {"title": "Write the edit prompt"},
                                "inputs": {"clip": [NODE_ENH_CLIP, 0], "image": [NODE_IMAGE, 0],
                                           "system_prompt": [NODE_ENH_SYSTEM, 0], "prompt": request,
                                           "max_length": max_tokens, "sampling_mode": "on",
                                           **sampling, "sampling_mode.seed": seed,
                                           "thinking": thinking, "use_default_template": True, "mtp": "auto"}}
    source = [NODE_ENH_GENERATE, 0]
    for n, (pattern, replace) in enumerate(prompt_hygiene.WORD_RULES):
        node = str(NODE_WRITE_CLEAN_FIRST + n)
        graph[node] = {"class_type": "RegexReplace", "_meta": {"title": "Clean the written prompt"},
                       "inputs": {"string": source, "regex_pattern": pattern, "replace": replace,
                                  "case_insensitive": True, "multiline": False, "dotall": True, "count": 0}}
        source = [node, 0]
    graph[NODE_ENH_SHOW] = {"class_type": "PreviewAny", "_meta": {"title": "Written prompt"},
                            "inputs": {"source": source}}
    graph[NODE_ENCODE]["inputs"]["prompt"] = source
    if identity:
        add_identity_notes(graph, identity if isinstance(identity, str) else None, identity_request)
        notes = graph[NODE_ID_SHOW]["inputs"]["source"]
        for n, (pattern, replace) in enumerate(WRITER_NOTES_CLEANUP):
            node = str(NODE_WRITER_NOTES_CLEAN_FIRST + n)
            graph[node] = {"class_type": "RegexReplace", "_meta": {"title": "Identity facts for the writer"},
                           "inputs": {"string": notes, "regex_pattern": pattern, "replace": replace,
                                      "case_insensitive": True, "multiline": False, "dotall": True, "count": 0}}
            notes = [node, 0]
        graph[NODE_WRITER_NOTES] = {"class_type": "StringConcatenate", "_meta": {"title": "Request + identity facts"},
                                    "inputs": {"string_a": request, "string_b": notes,
                                               "delimiter": WRITER_NOTES_INTRO}}
        graph[NODE_ENH_GENERATE]["inputs"]["prompt"] = [NODE_WRITER_NOTES, 0]


# card74 (2026-11-21): the writer did not see the identity notes, and it named colors and items of its own: 'warm
# auburn brown' for dark brown hair (2 of 3), 'goggles' for a clear visor, 'deep terracotta' for a dark corset. The
# notes come after the written prompt and weigh less. The writer now gets the cleaned notes after its request.
# card75: the hair and corset colors held 3 of 3, but the writer also copied the face list of a creature word for
# word, and the creature became an old man in 3 of 3 seeds. The writer takes only the colors and the item nouns.
NODE_WRITER_NOTES = "143"
WRITER_NOTES_INTRO = ("\n\nThe identity facts of the input, from a second look at it. Take from them only the color of "
                      "the hair, the skin, and each garment, and the noun of each worn item. The request decides the "
                      "faces, the heads, and all else:\n")
# card76: the writer copied the face list of the notes also against the narrowed intro ('large bulbous nose, a wrinkled
# chin and jaw, small ears', 'open mouth with visible teeth'), the creature became a scowling old man in 3 of 3 seeds,
# and 'one gloved hand' gave a glove. The writer gets the notes without the clauses of the face, the head, the skin,
# and the hands. The encoder still gets the full notes.
WRITER_NOTES_CLEANUP = (
    (r"[,;]?\s*[^,;:.]*\b(?:eyes?|nose|mouth|teeth|lips|brows?|chin|jaw|ears?|expression|grimac\w*|scowl\w*|"
     r"smil\w*|face|head|skin|hands?|wrinkle\w*)\b[^,;.]*", ""),
)
NODE_WRITER_NOTES_CLEAN_FIRST = 400


def writer_notes(notes: str) -> str:
    """The identity notes as the writer gets them, after WRITER_NOTES_CLEANUP (for tests)."""
    for pattern, replace in WRITER_NOTES_CLEANUP:
        notes = re.sub(pattern, replace, notes, flags=re.IGNORECASE | re.DOTALL)
    return notes


# Identity notes (2026-11-20). The long written prompt (600-800 words) drops identity facts in some seeds: the
# glasses of the tunnel man (2 of 5 runs), the hair color of the rider, the buckles of a corset. A softened source
# (--source-soften) cannot correct this, because the image model sees the people blurred. A second, short call of
# the same model lists only these facts, and the pipeline appends the list to the written prompt.
NODE_ID_GENERATE, NODE_ID_SHOW, NODE_ID_JOIN = "109", "110", "111"
NODE_ID_IMAGE = "120"  # the people crop for the identity notes
# The RegexReplace nodes of IDENTITY_CLEANUP get the ids from here on, in a range of their own: from 112, the ninth
# rule got id 120, the id of the people crop, and the server rejected the graph (card38).
NODE_ID_CLEAN_FIRST = 200
NODE_WRITE_CLEAN_FIRST = 300  # the RegexReplace nodes of prompt_hygiene.WORD_RULES on the written prompt
IDENTITY_MAX_TOKENS = 200
# card79 (2026-11-22): the notes called a charcoal coat 'a dark leather coat' with no hue word, the writer filled in
# 'dark warm brown' or 'deep warm browns', and the image model drew a brown coat in 2 of 3 seeds; the seed whose
# writer wrote 'cool dark neutrals' kept the charcoal. A dark garment gets its hue word in the notes. card80: a list
# of example hue words in that request ('black, charcoal, dark gray, dark brown') made the notes call a dark red
# velvet jacket 'dark brown' and a gray-brown corset 'dark gray' in 3 of 3 seeds, so the request names no hue.
# card81: 'never the word dark alone' still made the notes call that jacket 'dark brown' (the jacket was near-black
# brown in 2 of 3 seeds); the request only asks for the hue of a dark garment too. card82: that wording made the notes
# call a gray-gold pinafore 'pale sage-green' and the image model drew it green in 3 of 3 seeds (as in card72). Each
# request for a hue makes the notes invent hues, so the request asks for the color only, and
# prompt_hygiene.DARK_HUE_RULES turn a bare 'dark' garment into a near-black one.
# card72 (2026-11-21): a request for 'the exact hue (also a mixed hue)' turned a gray-gold patterned dress into
# 'sage-green brocade' and the image model drew it green in 3 of 3 seeds; a request to name no fabric texture did not
# remove 'quilted'. The request asks for the color only, and prompt_hygiene.FABRIC_TEXTURE_RULES remove the texture.
IDENTITY_SYSTEM_PROMPT = "You list the visible identity facts of the subjects of an image, in plain visual words."
IDENTITY_REQUEST = (
    "An artist redraws this image without seeing it. List the identity facts of the main subjects that the artist "
    "must keep. For each main person, one sentence: the position (left, right, center), the sex, the age in words, "
    "the hair color and style, the facial hair (a beard, a mustache, or light stubble on the jaw), the glasses, a "
    "hat, earrings or a helmet, each garment with its color, each of its layers from the neck down, and its visible "
    "parts (buckles, straps, lacing, a chain, a print), and the facial expression in strong words for the brows, the eyes, and the mouth (for example lowered "
    "angry brows and a tight mouth, or a wide open laugh). For a main animal, one sentence: the "
    "species, the fur colors, and the pattern. For a main "
    "creature (a monster, an alien, a puppet), one sentence that names, in this order, the outline of its head and "
    "where the head is widest, its brow, its eyes with their size and place, its nose, its mouth and lips, its chin and "
    "jaw, its ears, and its hair, each with its shape and size compared with a human face (for example smaller, wider, "
    "flatter, or rounder) and the hair with its color and where it grows, the skin color and texture, and the body "
    "build; name the shape of an ear only when the ear is clearly visible. Then one short sentence for the place behind them: its main surfaces "
    "with their exact material, their surface pattern, and their color family, as the photo shows them (for example "
    "a smooth pale yellow painted wall); a plain "
    "surface stays plain, without joints or blocks that the photo does not show. Do "
    "not describe the style or the mood. "
    "Name only the items that are present; a small item (a watch, a ring, a bracelet) only when its face, "
    "stone, or metal is clearly visible. Write at most 100 words, one time. Start with 'Identity to keep:'.")
# card18 (2026-11-20): without sampling, the model sometimes writes the list two times and adds a word count, and
# it denied items ('no glasses'), and the image model then drew glasses on a woman who had none. A ban of the
# words 'no' and 'without' in the request made the model reason about the ban inside its answer (card19), so the
# request only asks for present items, and the server cleans the list with these regular expressions (Python re,
# case-insensitive, dot matches newlines), in order.
# card62-card67 (user, several times): a small figure fixed to a prop (a gnome taped to a handlebar) came back as a
# clean, cute doll with a big face, because the scene reading and the identity notes called it 'a doll head with
# blond hair'; the writer copies those words, also against its own rule. A figure that the text fixes to an object
# becomes a rough bundle of cloth, and its hair becomes a tuft of that color. card70: the notes said 'a small <kind>
# figure with red hat and blue clothing attached at the left end', and one seed drew a cute figure with a face, so
# '<word> figure' counts too.
FIXED_FIGURE_CLEANUP = (
    (r"\b(?:an? |the )?(?:small |tiny |little )?(?:[a-z]+-haired )?(?:doll|figurine|toy figure|toy)(?: head| figure)?\b"
     r"(?=[^.;]{0,80}\b(?:strapped|tied|taped|fixed|mounted|attached|lashed|wrapped|bound|lying on|(?:at|on) the "
     r"(?:left|right) (?:end|grip))\b)", "a small rough cloth bundle"),
    (r"\b(?:an? |the )?(?:small|tiny|little) (?:[a-z]+ )?figure\b"
     r"(?=[^.;]{0,80}\b(?:strapped|tied|taped|fixed|mounted|attached|lashed|bound|(?:at|on) the (?:left|right) "
     r"(?:end|grip))\b)", "a small rough cloth bundle"),
    (r"(rough cloth bundle) with ([a-z -]+?) hair\b", r"\1 with a \2 tuft"),
)
IDENTITY_CLEANUP = (
    (r"(Identity to keep:.*?)\s*Identity to keep:.*", r"\1"),  # only the first list
    (r"\n\s*\n.*", ""),                                         # only the first paragraph (a self-check follows)
    (r"\s*\((?:note|about|approximately|\d).*", ""),           # a word count or a note, and all after it
    (r"\s*\b\d+ words\.?\s*$", ""),                            # a word count at the end
    (r"\s*\bWrite at most\b.*", ""),                           # card49: an echo of the request tail
    # card23: a drawing needs a visible line for a fact. 'Rimless glasses' (thin frames read as none) and 'light
    # stubble' came back as no glasses and a clean jaw.
    (r"\b(?:rimless|frameless)\b", "thin dark-framed"),
    (r"\b(?:only )?(?:light|faint|slight) stubble\b", "a visible short dark stubble beard"),
    (r"(?:^|(?<=[.:]))\s*(?:no|none)\b[^.]*\.", ""),           # a sentence that denies an item
    (r"[,;]\s*(?:no|without)\b[^,;.]*", ""),                   # a denied item in a list
    # card38: 'Behind her, a blurred brown-gray woodland backdrop' at the end of the prompt gave a photo bokeh
    # background: the image model draws the blur of the photo when the words name it.
    (r"\b(?:blurred|blurry|out-of-focus|out of focus|soft-focus|defocused|bokeh)\s*(?:,\s*)?", ""),
) + (FIXED_FIGURE_CLEANUP + prompt_hygiene.FABRIC_TEXTURE_RULES + prompt_hygiene.DARK_HUE_RULES
     + prompt_hygiene.SKIN_HUE_RULES + prompt_hygiene.MIXED_GREEN_RULES)


def clean_identity(text: str) -> str:
    """The identity list after IDENTITY_CLEANUP, as the server makes it (for tests and stored prompts)."""
    for pattern, replace in IDENTITY_CLEANUP:
        text = re.sub(pattern, replace, text, flags=re.IGNORECASE | re.DOTALL)
    return text


def add_identity_notes(graph: dict, crop: str | None = None, request: str | None = None) -> None:
    """The writer model lists the identity facts of the sharp source (or of the people crop, the server name in
    crop); the encoder gets the written prompt plus the list. The written prompt (NODE_ENH_SHOW) stays as written;
    NODE_ID_SHOW holds the list."""
    image = [NODE_IMAGE, 0]
    if crop:
        graph[NODE_ID_IMAGE] = {"class_type": "LoadImage", "_meta": {"title": "The people of the source, cropped"},
                                "inputs": {"image": crop}}
        image = [NODE_ID_IMAGE, 0]
    graph[NODE_ID_GENERATE] = {"class_type": "TextGenerate", "_meta": {"title": "List the identity facts"},
                               "inputs": {"clip": [NODE_ENH_CLIP, 0], "image": image,
                                          "system_prompt": IDENTITY_SYSTEM_PROMPT, "prompt": request or IDENTITY_REQUEST,
                                          "max_length": IDENTITY_MAX_TOKENS, "sampling_mode": "off",
                                          "thinking": False, "use_default_template": True, "mtp": "auto"}}
    source = [NODE_ID_GENERATE, 0]
    for n, (pattern, replace) in enumerate(IDENTITY_CLEANUP):
        node = str(NODE_ID_CLEAN_FIRST + n)
        graph[node] = {"class_type": "RegexReplace", "_meta": {"title": "Clean the identity facts"},
                       "inputs": {"string": source, "regex_pattern": pattern, "replace": replace,
                                  "case_insensitive": True, "multiline": False, "dotall": True, "count": 0}}
        source = [node, 0]
    graph[NODE_ID_SHOW] = {"class_type": "PreviewAny", "_meta": {"title": "Identity facts"},
                           "inputs": {"source": source}}
    if NODE_ENCODE in graph:
        graph[NODE_ID_JOIN] = {"class_type": "StringConcatenate", "_meta": {"title": "Written prompt + identity"},
                               "inputs": {"string_a": graph[NODE_ENCODE]["inputs"]["prompt"], "string_b": source,
                                          "delimiter": "\n\n"}}
        graph[NODE_ENCODE]["inputs"]["prompt"] = [NODE_ID_JOIN, 0]


def with_identity(written: str, entry: dict) -> str:
    """The written prompt plus the identity list of the job, as the encoder got it."""
    notes = "".join(entry.get("outputs", {}).get(NODE_ID_SHOW, {}).get("text", [])).strip()
    return f"{written.strip()}\n\n{notes}" if notes and written.strip() else written


def add_titler(graph: dict, language: str) -> None:
    """After the decode, the enhancer model looks at the finished image and writes a short title for the file name."""
    graph[NODE_TITLE_GENERATE] = {"class_type": "TextGenerate", "_meta": {"title": "Title of the picture"},
                                  "inputs": {"clip": [NODE_ENH_CLIP, 0], "image": [NODE_DECODE, 0],
                                             "system_prompt": TITLE_SYSTEM_PROMPT,
                                             "prompt": TITLE_REQUEST.format(language=language),
                                             "max_length": TITLE_MAX_TOKENS, "sampling_mode": "off",
                                             "thinking": False, "use_default_template": True, "mtp": "auto"}}
    graph[NODE_TITLE_SHOW] = {"class_type": "PreviewAny", "_meta": {"title": "Title"},
                              "inputs": {"source": [NODE_TITLE_GENERATE, 0]}}


NODE_NEG_REQUEST, NODE_NEG_GENERATE, NODE_NEG_SHOW = "106", "107", "108"
NODE_NAG = "15"
NODE_SOFT_IMAGE = "16"  # the blurred source for the image model (--source-soften)
NAG_NODE = "QwenImage21NAG"  # custom node, see custom_nodes/qwen-image21-nag in the README
NAG_DEFAULT_ALPHA = 0.5  # defaults of the NAG paper and of the core ComfyUI NAGuidance node
NAG_DEFAULT_TAU = 1.5
NEGATIVE_MAX_TOKENS = 600
NEGATIVE_SYSTEM_PROMPT = "You write negative prompts for an image generation model."
NEGATIVE_REQUEST = (
    "Write the negative prompt for the picture that the positive prompt below describes. From the style "
    "rules below, choose the faults that are most likely to appear in this picture, and add other visual "
    "faults that would break this style in this picture. Write 20 to 40 short visual noun phrases on one "
    "line, separated by commas. Write only the things that must not appear: no sentences, no negation "
    "words, no explanations.\n\nStyle rules of what must not appear:\n{rules}\n\n{story}Positive prompt:\n")
NEGATIVE_STORY_PART = (
    "Story card of the picture: the scene that the positive prompt was written from. Also name the faults "
    "that this scene invites, for example wrong objects, wrong light, or a wrong mood for this setting and "
    "moment.\n{card}\n\n")


def add_negative_writer(graph: dict, rules: str, story_card: str = "") -> None:
    """After the enhancer writes the positive prompt, the same model writes a negative prompt for this picture.

    With a story card (--negative-source story), the writer also knows the scene concept of the story.
    """
    story = NEGATIVE_STORY_PART.format(card=story_card) if story_card else ""
    graph[NODE_NEG_REQUEST] = {"class_type": "StringConcatenate", "_meta": {"title": "Negative request"},
                               "inputs": {"string_a": NEGATIVE_REQUEST.format(rules=rules, story=story),
                                          "string_b": [NODE_ENH_GENERATE, 0], "delimiter": ""}}
    graph[NODE_NEG_GENERATE] = {"class_type": "TextGenerate", "_meta": {"title": "Write the negative prompt"},
                                "inputs": {"clip": [NODE_ENH_CLIP, 0], "image": [NODE_IMAGE, 0],
                                           "system_prompt": NEGATIVE_SYSTEM_PROMPT, "prompt": [NODE_NEG_REQUEST, 0],
                                           "max_length": NEGATIVE_MAX_TOKENS, "sampling_mode": "off",
                                           "thinking": False, "use_default_template": True, "mtp": "auto"}}
    graph[NODE_NEG_SHOW] = {"class_type": "PreviewAny", "_meta": {"title": "Written negative prompt"},
                            "inputs": {"source": [NODE_NEG_GENERATE, 0]}}
    graph[NODE_ENCODE]["inputs"]["negative_prompt"] = [NODE_NEG_GENERATE, 0]


def clean_title(text: str) -> str:
    """One line of the title, safe as a Windows, macOS, and Linux file name."""
    line = next((x for x in text.strip().splitlines() if x.strip()), "")
    line = re.sub(r'[<>:"/\\|?*\x00-\x1f«»“”„]', "", line).strip(" .'")
    line = re.sub(r"^(a|an|the)\s+", "", line, flags=re.IGNORECASE)  # "Cat in ...", as ChatGPT names
    line = line[:1].upper() + line[1:]
    return line[:TITLE_MAX_CHARS].rstrip(" ,.")


# --------------------------------------------------------------------------- server


class ComfyClient:
    def __init__(self, server: str):
        self.base = server.rstrip("/")
        self.client_id = uuid.uuid4().hex

    def _request(self, method: str, path: str, data: bytes | None = None, headers: dict | None = None) -> bytes:
        req = urllib.request.Request(self.base + path, data=data, method=method, headers=headers or {})
        try:
            with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_S) as resp:
                return resp.read()
        except urllib.error.HTTPError as err:
            body = err.read().decode("utf-8", "replace")
            raise RuntimeError(f"{method} {path} -> HTTP {err.code}: {body}") from err
        except urllib.error.URLError as err:
            raise SystemExit(f"cannot reach ComfyUI at {self.base} ({err.reason}). "
                             "Start ComfyUI, or give its address with --server.") from err

    def get_json(self, path: str):
        return json.loads(self._request("GET", path))

    def post_json(self, path: str, payload: dict):
        data = json.dumps(payload).encode("utf-8")
        return json.loads(self._request("POST", path, data, {"Content-Type": "application/json"}))

    def upload_image(self, path: Path) -> str:
        """Upload under a content-hash name, so that a second run reuses the same file."""
        content = path.read_bytes()
        name = hashlib.sha256(content).hexdigest()[:16] + path.suffix.lower()
        boundary = uuid.uuid4().hex
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        body = b"".join([
            f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="{name}"\r\n'
            f"Content-Type: {mime}\r\n\r\n".encode(),
            content,
            f'\r\n--{boundary}\r\nContent-Disposition: form-data; name="overwrite"\r\n\r\ntrue'
            f"\r\n--{boundary}--\r\n".encode(),
        ])
        reply = json.loads(self._request("POST", "/upload/image", body,
                                         {"Content-Type": f"multipart/form-data; boundary={boundary}"}))
        sub = reply.get("subfolder") or ""
        return f"{sub}/{reply['name']}" if sub else reply["name"]

    def queue(self, graph: dict, front: bool = False) -> str:
        """Queue a graph; front=True puts it before the waiting jobs (a retry must not wait for the whole batch)."""
        body = {"prompt": graph, "client_id": self.client_id}
        if front:
            body["front"] = True
        reply = self.post_json("/prompt", body)
        if reply.get("node_errors"):
            raise RuntimeError(f"graph rejected: {json.dumps(reply['node_errors'])}")
        return reply["prompt_id"]

    def wait(self, prompt_id: str) -> dict:
        deadline = time.monotonic() + JOB_TIMEOUT_S
        while time.monotonic() < deadline:
            hist = self.get_json(f"/history/{prompt_id}")
            if prompt_id in hist:
                entry = hist[prompt_id]
                status = entry.get("status", {})
                if status.get("completed") or status.get("status_str") == "error":
                    return entry
            time.sleep(POLL_INTERVAL_S)
        raise TimeoutError(f"job {prompt_id} did not finish in {JOB_TIMEOUT_S} s")

    def download(self, image: dict) -> bytes:
        query = urllib.parse.urlencode({k: image.get(k, "") for k in ("filename", "subfolder", "type")})
        return self._request("GET", f"/view?{query}")


def execution_seconds(entry: dict) -> float | None:
    """Server-side time of the job, from the execution_start and execution_success messages."""
    stamps = {name: data.get("timestamp") for name, data in entry.get("status", {}).get("messages", [])}
    start, end = stamps.get("execution_start"), stamps.get("execution_success")
    return (end - start) / 1000 if start and end else None


# --------------------------------------------------------------------------- main


def collect(paths: list[str], suffixes: set[str]) -> list[Path]:
    found = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            found += sorted(x for x in p.iterdir()
                            if x.suffix.lower() in suffixes and not x.name.startswith(STYLE_SKIP_PREFIX)
                            and ".example." not in x.name)
        elif p.is_file():
            found.append(p)
        else:
            sys.exit(f"not found: {raw}")
    return found


def parse_fills(args) -> dict[str, str]:
    fills = {k: v for k, v in {"subject": args.subject, "context": args.context,
                                "composition": args.composition, "palette": args.palette}.items() if v}
    for item in args.fill:
        key, sep, value = item.partition("=")
        if not sep:
            sys.exit(f"--fill must be KEY=VALUE, got: {item}")
        fills[key.strip().lower()] = value
    return fills


MODES = ("direct", "adapt", "story")
RULES_POSITIVE_CHOICES = ("none", "plain", "explained")
RULES_NEGATIVE_CHOICES = ("off", "on")
NEGATIVE_SOURCES = ("style", "written", "story")
GUIDANCES = ("cfg", "nag")
DEFAULT_RULES_POSITIVE = "explained"
NEGATIVE_DEFAULT_CFG = 2.0  # the sampler reads the negative prompt only at cfg more than 1
NAG_DEFAULT_SCALE = 1.5  # measured on Qwen Image 2.1 Turbo: 2 and more break the image into collages and text
REFERENCE_AUTO = "auto"
REFERENCE_CANVAS = "canvas"
CANVAS_CHOICES = (CANVAS_SOURCE, CANVAS_AUTO) + tuple(CANVASES)
CANVAS_TEXT = {CANVAS_SOURCE: "the aspect ratio of the source image", "portrait": "portrait, taller than wide (2:3)",
               "landscape": "landscape, wider than tall (3:2)", "square": "square (1:1)"}


def reference_size_arg(value: str):
    if value in (REFERENCE_AUTO, REFERENCE_CANVAS):
        return value
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"use {REFERENCE_AUTO}, {REFERENCE_CANVAS}, or a number of pixels, "
                                         f"not {value!r}")


def soften_arg(value: str):
    if value == REFERENCE_AUTO:
        return value
    try:
        fraction = float(value)
    except ValueError:
        fraction = -1.0
    if not 0 <= fraction < 1:
        raise argparse.ArgumentTypeError(f"use {REFERENCE_AUTO} or a part of the long side from 0 to 1, not {value!r}")
    return fraction


def soften_of(style: dict, setting) -> float:
    """The source blur of this style: the --source-soften number, or from the style's distance to a photograph."""
    if setting == REFERENCE_AUTO:
        return source_cleanup.soften_fraction(photo_distance(style)[0])
    return float(setting)


# card56 (2026-11-21): a style that turns a figure into a counted set of large flat shapes ('three to eight large
# shapes') drew the hunter pair near-photographic with the full identity notes (stubble, studs, lacing, embroidery),
# and as a flat sign with the notes off, at the same seed; a stronger source blur (0.06) changed nothing. But
# without the notes the hat shape and the dress hue changed. For such a style, the notes keep only the large shapes
# and their colors. card57: without a creature sentence, the creature of the desk became a bald old man.
SHAPE_BUDGET_RE = re.compile(r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+) to "
                             r"(?:\w+) (?:large |flat |simple )*shapes\b", re.IGNORECASE)
IDENTITY_SIGNATURE_REQUEST = (
    "An artist redraws this image as a few large flat shapes, without seeing it. List the signature facts of the "
    "main subjects that the artist must keep. For each main person, one short sentence: the position (left, right, "
    "center), the sex, the hair color and the outline of the hair, the headwear with its shape, and each large "
    "garment with its color. For a main animal, one short sentence: the species and the fur colors. For a main "
    "creature (a monster, an alien, a puppet), one short sentence: the outline of its head and its ears compared "
    "with a human head, its hair color, and its skin color. Then one short "
    "sentence for the color of the place behind them. Name only large shapes and their colors. Write at most 50 "
    "words, one time. Start with 'Identity to keep:'.")


def shape_budget(style: dict) -> bool:
    """Whether the detail budget of the style limits a figure to a counted set of shapes."""
    budget = style.get("source_adaptation", {}).get("detail_budget", {})
    return any(SHAPE_BUDGET_RE.search(str(v)) for v in (budget.values() if isinstance(budget, dict) else [budget]))


# card57 (2026-11-21): a style that prints a face in one or two inks ('the face as a ... duotone cut-out') got faces
# in the full color of the photo, also when the written prompt said duotone: the image model keeps the colors of the
# source pixels. For such a style, the heads in the copy for the image model are gray.
GRAY_FACE_RE = re.compile(r"\bface\b[^.]*\b(?:duotone|monochrome|grayscale|greyscale|black-and-white)\b|"
                          r"\b(?:duotone|monochrome|grayscale|greyscale|black-and-white)\b[^.]*\bface\b", re.IGNORECASE)


def gray_face(style: dict) -> bool:
    """Whether the subject design of the style prints a face in one or two inks."""
    return bool(GRAY_FACE_RE.search(json.dumps(style.get("subject_design", {}), ensure_ascii=False)))


def identity_request_of(style: dict) -> str:
    """The request of the identity notes for this style: the signature list for a shape-budget style."""
    return IDENTITY_SIGNATURE_REQUEST if shape_budget(style) else IDENTITY_REQUEST


def identity_on(style: dict, args) -> bool:
    """Whether the jobs of this style get the identity notes (--identity-notes)."""
    if args.identity_notes == "auto":
        return soften_of(style, args.source_soften) > 0
    return args.identity_notes == "on"


def random_seed() -> int:
    return uuid.uuid4().int >> (128 - SEED_BITS)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+", help="image files or folders")
    ap.add_argument("--styles", nargs="+", default=[str(DEFAULT_STYLES)], help="style JSON files or folders")

    g = ap.add_argument_group("1. mode and seeds")
    g.add_argument("--mode", choices=MODES, default="story",
                   help="who writes the positive prompt. direct: nobody, the style JSON goes to the image model "
                        "(approximately 15 s; good for a style that is mainly a medium and a frame). adapt: the "
                        "LLM writes it, for the style's own generic scene. story: the LLM writes it, with an "
                        "enrichment from the storyteller: concept, moment, story details (default)")
    g.add_argument("--direct-format", choices=("json", "outline"), default="json",
                   help="direct: the style as compact JSON (default) or as an outline without quotes (the image "
                        "model can draw quoted strings as text)")
    g.add_argument("--image-seed", type=int, help="noise of the image model: a new image from the same prompt")
    g.add_argument("--prompt-seed", type=int, help="adapt and story: sampling of the LLM, different details of the written prompt")
    g.add_argument("--story-seed", type=int, help="story: the draw of the story card (concept, setting, camera)")
    g.add_argument("--seed", type=int,
                   help="sets each of the three seeds above that is not given; a seed that stays unset is random")

    g = ap.add_argument_group("2. story (storyteller)")
    g.add_argument("--story-config", default=str(DEFAULT_STORY_CONFIG),
                   help="storyteller config: lines, pools, probabilities (see storyteller/README.md)")
    g.add_argument("--story-line", nargs="+", default=[], metavar="LABEL=P",
                   help="the probability of a story card line, for example Homage=0.8 Moment=0")

    g = ap.add_argument_group("3. canvas and reference")
    g.add_argument("--canvas", choices=CANVAS_CHOICES, default=CANVAS_SOURCE,
                   help="the output size. source (default): the exact aspect ratio of the source image, with the "
                        "area of a ChatGPT image (1024 x 1536). auto: a ChatGPT size; in story mode the Format line "
                        "of the story card chooses it (filtered by the style's composition text), in direct and "
                        "adapt the size nearest to the source. portrait 1024 x 1536, landscape 1536 x 1024, "
                        "square 1024 x 1024")
    g.add_argument("--reference-size", type=reference_size_arg, default=REFERENCE_AUTO, metavar="N|auto|canvas",
                   help="the source goes to the image model at about N x N pixels: small enough that the "
                        "subject is redrawn in the style, large enough to keep its identity. auto (default): from "
                        f"{REFERENCE_MIN_FRACTION * 100:.0f} %% of the canvas side for a fully drawn style to the full side "
                        "for a photographic one, from the style's rendering fields. canvas: the full canvas side, "
                        "like ChatGPT. 0: the encoder default")

    g = ap.add_argument_group('4. "no" rules and negative prompt')
    g.add_argument("--rules-positive", choices=RULES_POSITIVE_CHOICES, default=DEFAULT_RULES_POSITIVE,
                   help="the style's no rules (avoid lists, negative_prompt, explicitly_not_the_style, "
                        "do_not_reproduce, do_not) in the positive prompt. none: removed. plain: they stay as plain data. "
                        "explained (default): they stay, and the prompt says that they must not appear. In adapt "
                        "and story, the LLM turns the rules that it gets into positive statements, so plain is "
                        "the same as explained")
    g.add_argument("--rules-negative", choices=RULES_NEGATIVE_CHOICES, default="off",
                   help="on: the no rules also go to the negative prompt (see --negative-source and --guidance). "
                        "off (default): the negative prompt stays empty")
    g.add_argument("--negative-source", choices=NEGATIVE_SOURCES, default="style",
                   help="with --rules-negative on. style (default): the raw rules. written: adapt and story "
                        "only, the LLM writes a negative prompt for this picture from the rules and its positive "
                        "prompt. story: story mode only, as written, and the LLM also gets the story card")
    g.add_argument("--guidance", choices=GUIDANCES, default="cfg",
                   help="how the negative prompt is applied. cfg (default): classifier-free guidance, scale "
                        f"{NEGATIVE_DEFAULT_CFG:g} by default when there is a negative prompt. nag: Normalized "
                        f"Attention Guidance at cfg 1, scale {NAG_DEFAULT_SCALE:g} by default; needs the "
                        "QwenImage21NAG custom node. Both take approximately 2x sampling time")
    g.add_argument("--guidance-scale", type=float,
                   help="the cfg value or the NAG scale. Without a negative prompt, cfg 1 is the Turbo default")
    g.add_argument("--nag-alpha", type=float, default=NAG_DEFAULT_ALPHA, help="NAG blend, 0 to 1")
    g.add_argument("--nag-tau", type=float, default=NAG_DEFAULT_TAU, help="NAG norm limit")

    g = ap.add_argument_group("5. LLM (adapt and story)")
    g.add_argument("--prompt-thinking", action=argparse.BooleanOptionalAction, default=True,
                   help="the LLM reasons before it writes (default on: it obeys the style rules much better; "
                        "approximately 4x slower). An empty result reruns the job without thinking")
    g.add_argument("--prompt-system", default=str(ENHANCER_SYSTEM_PROMPT), metavar="FILE",
                   help="the system prompt of the LLM (default: style-enhancer.system.md next to this script)")
    g.add_argument("--prompt-words", metavar="MIN-MAX",
                   help="length of the written prompt in words, for example 500-800 (default 800-1200)")
    g.add_argument("--prompt-max-tokens", type=int,
                   help=f"token limit of the LLM output (default {ENHANCER_MAX_TOKENS}, with thinking "
                        f"{ENHANCER_MAX_TOKENS_THINKING})")
    g.add_argument("--title", action=argparse.BooleanOptionalAction, default=True,
                   help="name each result with a short title of the finished picture, like ChatGPT does "
                        "(approximately 3 s more for each image)")
    g.add_argument("--title-language", default=DEFAULT_TITLE_LANGUAGE,
                   help="language of the title, for example English or Ukrainian")

    g = ap.add_argument_group("6. placeholders")
    g.add_argument("--subject", help="value for {SUBJECT}")
    g.add_argument("--context", help="value for {CONTEXT}")
    g.add_argument("--composition", help="value for {COMPOSITION}")
    g.add_argument("--palette", help="value for {PALETTE}")
    g.add_argument("--fill", action="append", default=[], metavar="KEY=VALUE",
                   help="value for a style-specific placeholder, for example surreal_event=...")

    g = ap.add_argument_group("7. server and output")
    g.add_argument("--output", default="style-transfer-output", help="folder for the results")
    g.add_argument("--server", default=DEFAULT_SERVER, help="ComfyUI address")
    g.add_argument("--graph", default=str(DEFAULT_GRAPH), help="API graph JSON")
    g.add_argument("--loras", default=str(DEFAULT_LORAS), metavar="FILE",
                   help="the LoRA registry: each style gets the generic LoRAs whose words match it (see "
                        "lora_routing.py); a LoRA of one concrete work never routes")
    g.add_argument("--no-loras", action="store_true", help="load no LoRA, also when the registry has a match")
    g.add_argument("--extra-lora", action="append", default=[], metavar="FILE[:STRENGTH[:TRIGGER]]",
                   help="a LoRA chosen by hand for every job of this run, beside the automatic routing (any scope); "
                        "for example a private style LoRA for a preview. Can be given more than one time")
    g.add_argument("--source-cleanup", action=argparse.BooleanOptionalAction, default=True,
                   help="fill a fault of the photo at the frame edge (an out-of-focus object in front of the lens) "
                        "before the image model sees it (default on, see source_cleanup.py)")
    g.add_argument("--source-soften", type=soften_arg, default=REFERENCE_AUTO, metavar="auto|FRACTION",
                   help="blur the copy of the source that the image model gets, so that the drawing is free of the "
                        "photo's contours (the writer still sees the sharp source). Only the people are blurred "
                        "(a person mask, source_cleanup.py): the scene keeps its core details and its light (a blur "
                        "of the whole frame drops them, user 2026-11-20). auto (default): from the style's "
                        f"distance to a photograph, up to {source_cleanup.SOFTEN_MAX * 100:.1f} %% of the long side for a fully "
                        "drawn style, none for a photographic one. A number: that part of the long side for every "
                        "style. 0: off")
    g.add_argument("--identity-notes", choices=["auto", "on", "off"], default="auto",
                   help="a second, short writer call lists the identity facts of the source (hair color, glasses, "
                        "garments with their colors and parts, fur colors), and the list goes after the written "
                        "prompt. auto: on when the source is softened for this style (the image model then cannot "
                        "correct a fact that the written prompt drops)")
    g.add_argument("--place-check", choices=["auto", "off"], default="auto",
                   help="story mode: before the writer runs, the writer model answers PLACE or PLAIN for the "
                        "background of each source; for a place, the story card leaves out its setting lines, so "
                        "that the place is adapted, not replaced. off: the writer decides alone")
    g.add_argument("--scene-reading", choices=["auto", "off"], default="auto",
                   help="story mode: before the writer runs, the writer model reads each source as one moment of a "
                        "story in the tone of the style (what happens, what each person feels and why, how they "
                        "relate, the acting, and the story props with their parts); the story card gets this "
                        "reading as a line. off: the card only asks to keep the emotion of the photo")
    g.add_argument("--style-refs", metavar="FILE",
                   help="a local registry of style reference images (its \"references\" list, same matching as the "
                        "LoRAs): each style gets the matching images as more reference images of the encoder")
    g.add_argument("--cache-device", choices=["auto", "gpu", "cpu", "off"], help="KV cache device")
    g.add_argument("--cache-dtype", choices=["default", "int8", "int4"], help="KV cache precision")
    g.add_argument("--dry-run", action="store_true", help="print the prompts and do not contact the server")
    g.add_argument("--server-copy", action=argparse.BooleanOptionalAction, default=True,
                   help="the server also saves each result in its own output folder (default on). "
                        "--no-server-copy: a temporary preview only, for experiments")
    g.add_argument("--check-graph", action="store_true",
                   help="check every graph of the run against the node list of the server, and queue nothing")
    g.add_argument("--object-info", metavar="FILE",
                   help="with --check-graph: a saved /object_info JSON file instead of the server")
    g.add_argument("--manifest", action=argparse.BooleanOptionalAction, default=True,
                   help="write <result>.json next to each result: the inputs, seeds, options, hashes, and models")

    g = ap.add_argument_group("8. prompt store, two-phase run, and image seeds")
    g.add_argument("--prompt-store", metavar="FOLDER",
                   help="adapt and story: run in two phases. First the LLM writes the prompts of all jobs, "
                        "the script cleans and checks them, and keeps them in FOLDER (<key>.txt and <key>.json, "
                        "written once, reused for the same inputs). Then the image model renders all jobs from "
                        "the stored prompts. The LLM loads once, and a prompt can be reused with new image seeds. "
                        "The results get no LLM title (the name has the story slug)")
    g.add_argument("--write-only", action="store_true",
                   help="with --prompt-store: only write and store the prompts, render nothing")
    g.add_argument("--written-prompt", metavar="FILE",
                   help="adapt and story: render from this prompt text file for every job, without the LLM")
    g.add_argument("--leak-retry", action=argparse.BooleanOptionalAction, default=True,
                   help="without thinking: if the written prompt contains the LLM's reasoning, run the job "
                        "again with thinking (default on; use --no-leak-retry in experiments)")
    g.add_argument("--warm-up", action=argparse.BooleanOptionalAction, default=True,
                   help="with --prompt-store: one short writer job before phase 1, because the first writer "
                        "job after a model load is slow (default on)")
    g.add_argument("--image-seeds", type=int, default=1, metavar="N",
                   help="N images for each job, with the image seeds S, S+1, ... (default 1). The written "
                        "prompt is the same for all N")
    return ap


def main() -> int:
    # Titles and style names can contain any character; a Windows console uses a legacy code page.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    ap = build_parser()
    args = ap.parse_args()
    uses_llm = args.mode in ("adapt", "story")
    has_negative = args.rules_negative == "on"
    if args.negative_source == "written" and not uses_llm:
        ap.error("--negative-source written needs the LLM: use it with --mode adapt or --mode story")
    if args.negative_source == "story" and args.mode != "story":
        ap.error("--negative-source story needs a story card: use it with --mode story")
    if args.guidance == "nag" and not has_negative:
        ap.error("--guidance nag needs a negative prompt: use --rules-negative on")
    scale = args.guidance_scale
    if scale is None and has_negative:
        scale = NAG_DEFAULT_SCALE if args.guidance == "nag" else NEGATIVE_DEFAULT_CFG
    if has_negative and args.guidance == "cfg" and scale <= 1:
        print("note: at cfg 1 the sampler ignores the negative prompt; use --guidance nag or a scale more than 1")

    # Option tags for the file name: only what differs from the plain default run.
    tags = []
    if args.rules_positive != DEFAULT_RULES_POSITIVE:
        tags.append(f"pos-{args.rules_positive}")
    if has_negative:
        tags.append(f"neg-{args.negative_source}")
    if scale is not None:
        tags.append(f"{args.guidance}{scale:g}")
    if uses_llm and not args.prompt_thinking:
        tags.append("no-think")
    if uses_llm and args.prompt_words:
        tags.append(f"words{args.prompt_words}")
    if args.reference_size != REFERENCE_AUTO:
        tags.append(f"ref{args.reference_size}")
    if args.canvas != CANVAS_SOURCE:
        tags.append(f"canvas-{args.canvas}")
    if args.mode == "direct" and args.direct_format != "json":
        tags.append(f"fmt-{args.direct_format}")

    images = collect(args.images, IMAGE_SUFFIXES)
    styles = collect(args.styles, {".json"})
    if not images or not styles:
        sys.exit("no images or no styles found")
    fills = parse_fills(args)
    template = json.loads(Path(args.graph).read_text(encoding="utf-8"))
    lora_registry = [] if args.no_loras or not args.loras else lora_routing.load_registry(Path(args.loras))
    model_node = next((nid for nid, node in template.items() if node.get("class_type") == "UNETLoader"), None)
    model_name = template[model_node]["inputs"].get("unet_name", "") if model_node else ""

    extra_loras = [lora_routing.parse_extra(spec) for spec in args.extra_lora]

    def loras_of(style: dict) -> list[dict]:
        return (lora_routing.select(style, lora_registry, model_name) + extra_loras) if model_node else []

    ref_registry = lora_routing.load_references(Path(args.style_refs)) if args.style_refs else []
    ref_uploads: dict[Path, str] = {}

    def refs_of(style: dict) -> list[dict]:
        return lora_routing.select(style, ref_registry, model_name) if model_node else []

    source_fill: dict[Path, float] = {}  # source path -> the part of the frame that the source cleanup filled

    def upload_ref(path: Path) -> str:
        if path not in ref_uploads:
            ref_uploads[path] = client.upload_image(path) if not args.check_graph else path.name
        return ref_uploads[path]
    story_config = {}
    if args.mode == "story":
        try:
            story_config = storyteller.set_probabilities(storyteller.load_config(args.story_config), args.story_line)
        except ValueError as err:
            ap.error(str(err))
        if args.canvas != CANVAS_AUTO:  # a fixed canvas: the Format line of the card names it, it draws nothing
            for line in story_config.get("lines", []):
                if line.get("label") == "Format":
                    line.pop("pool", None)
                    line["text"] = CANVAS_TEXT[args.canvas]
    place_of: dict[Path, bool | None] = {}  # source -> the input shows a place of its own (--place-check)
    surfaces_of: dict[Path, str | None] = {}  # source -> the main surfaces of its place
    if (args.mode == "story" and args.place_check == "auto" and not args.check_graph and not args.written_prompt):
        probe = ComfyClient(args.server)
        for image_path in images:
            place_of[image_path] = source_shows_place(probe, image_path)
            print(f"place  {image_path.name}: {PLACE_WORDS.get(place_of[image_path], 'unknown')}")
            if place_of[image_path]:
                surfaces_of[image_path] = source_place_surfaces(probe, image_path)
                print(f"place  {image_path.name}: surfaces {surfaces_of[image_path]}")

    scene_of: dict[tuple[Path, Path], str | None] = {}  # (source, style) -> the scene reading (--scene-reading)
    reads_scene = (args.mode == "story" and args.scene_reading == "auto" and not args.check_graph
                   and not args.written_prompt)
    scene_probe = ComfyClient(args.server) if reads_scene else None

    def config_of(image_path: Path, style_path: Path, style: dict) -> dict:
        """The story config of a source: without the setting lines when the source shows a place, and with the
        scene reading of the source for this style."""
        config = with_colors_line(without_setting_lines(story_config, surfaces_of.get(image_path))
                                  if place_of.get(image_path) else story_config)
        if not reads_scene:
            return config
        if (image_path, style_path) not in scene_of:
            scene_of[image_path, style_path] = read_scene(scene_probe, image_path, style, args)
            print(f"scene  {image_path.name} / {style_path.stem}: {scene_of[image_path, style_path] or 'no reading'}")
        return with_scene_line(config, scene_of[image_path, style_path])

    if args.prompt_max_tokens is None:
        args.prompt_max_tokens = ENHANCER_MAX_TOKENS_THINKING if args.prompt_thinking else ENHANCER_MAX_TOKENS

    def pick(value):
        if value is not None:
            return value
        return args.seed if args.seed is not None else random_seed()

    jobs = []
    style_path_of: dict[int, Path] = {}  # id of a loaded style -> its file, for card_of
    for style_path in styles:
        style = load_style(style_path)
        style_path_of[id(style)] = style_path
        for image_path in images:
            seeds = {"image": pick(args.image_seed), "prompt": pick(args.prompt_seed), "story": pick(args.story_seed)}
            canvas = args.canvas
            fixed = style_fixed_ratio(style)
            if fixed and canvas in (CANVAS_SOURCE, CANVAS_AUTO):
                canvas = f"{RATIO_PREFIX}{fixed[0]}:{fixed[1]}"  # the style fixes its format: it wins over the source
            if args.mode == "story":
                prompt = build_story_request(style, fills, seeds["story"], config_of(image_path, style_path, style),
                                             args.rules_positive)
                if canvas == CANVAS_AUTO:
                    card = storyteller.story_card(style_spec(style, fills, args.rules_positive),
                                                  config_of(image_path, style_path, style),
                                                  seeds["story"])
                    canvas = story_canvas(card, image_path)
            else:
                prompt = (build_enhancer_request(style, fills, args.rules_positive) if args.mode == "adapt"
                          else build_direct_prompt(style, fills, args.rules_positive, args.direct_format))
                if canvas == CANVAS_AUTO:
                    canvas = source_canvas(image_path)
            width, height = canvas_size(canvas, image_path)
            side = round(math.sqrt(width * height))
            if args.reference_size == REFERENCE_AUTO:
                reference = style_reference_size(style, side)["reference_size"]
            elif args.reference_size == REFERENCE_CANVAS:
                reference = round(side / REFERENCE_STEP) * REFERENCE_STEP
            else:
                reference = args.reference_size
            jobs.append((image_path, style_path, style, prompt, seeds, (width, height), reference))

    if args.dry_run:
        for image_path, style_path, style, prompt, seeds, (width, height), reference in jobs:
            negative = ""
            if has_negative:
                negative = (f"negative: {build_negative(style)}\n" if args.negative_source == "style"
                            else f"negative: written by the LLM after the positive prompt "
                                 f"({'with' if args.negative_source == 'story' else 'without'} the story card)\n")
            print(f"=== {result_name(image_path, style_path, [seed_tag(args.mode, seeds)] + tags)}\n"
                  f"canvas: {width} x {height}, reference: {reference} "
                  f"({json.dumps(style_reference_size(style, round(math.sqrt(width * height))))})\n"
                  f"{negative}{prompt}\n")
        return 0

    system_prompt = Path(args.prompt_system).read_text(encoding="utf-8") if uses_llm else ""
    system_prompt = set_word_target(system_prompt, args.prompt_words)
    writes_negative = has_negative and args.negative_source in ("written", "story")
    two_phase = uses_llm and bool(args.prompt_store or args.written_prompt)
    if args.write_only and not args.prompt_store:
        ap.error("--write-only needs --prompt-store")
    if args.written_prompt and writes_negative:
        ap.error("--written-prompt has no written negative: use --negative-source style or --prompt-store")
    if args.image_seeds < 1:
        ap.error("--image-seeds must be 1 or more")
    if args.image_seeds > 1:
        jobs = [(i, s, st, p, {**seeds, "image": seeds["image"] + k}, size, ref)
                for i, s, st, p, seeds, size, ref in jobs for k in range(args.image_seeds)]
    uploaded: dict[Path, str] = {}
    creature_boxes: dict[Path, tuple | None] = {}  # the head box of a main creature of each source
    softened: dict[tuple[Path, float, bool], str] = {}  # (source, blur, gray heads) -> the server name of the copy
    crops: dict[Path, str] = {}  # source -> the server name of its people crop, for the identity notes
    if args.check_graph:
        info = (graph_check.load_object_info(args.object_info) if args.object_info
                else graph_check.fetch_object_info(args.server))
        uploaded.update((job[0], job[0].name) for job in jobs)
    else:
        client = ComfyClient(args.server)
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        for image_path in dict.fromkeys(job[0] for job in jobs):
            uploaded[image_path] = upload_source(client, image_path, args, source_fill)
        for image_path, _, style, *_ in jobs:
            fraction = soften_of(style, args.source_soften)
            gray = gray_face(style)
            if fraction and (image_path, fraction, gray) not in softened:
                if image_path not in creature_boxes:  # the question only for a source with people (else no blur)
                    creature_boxes[image_path] = (creature_head_box(client, image_path)
                                                  if source_cleanup.has_people(image_path) else None)
                    print(f"{image_path.name}: creature head box {creature_boxes[image_path]}")
                boxes = (creature_boxes[image_path],) if creature_boxes[image_path] else ()
                softened[(image_path, fraction, gray)] = upload_softened(client, image_path, args, fraction, gray,
                                                                         boxes)
            if identity_on(style, args) and image_path not in crops:
                crops[image_path] = upload_people_crop(client, image_path, args)
        manifest_base = run_manifest(args, client, template, system_prompt, story_config) if args.manifest else {}

    def card_of(style: dict, seeds: dict, image_path: Path) -> str:
        """The story card that the writer got (card59: the negative writer and the manifest got the raw card, with
        the Setting lines of a source that shows a place and without the Colors line)."""
        return storyteller.story_card(style_spec(style, fills, args.rules_positive),
                                      config_of(image_path, style_path_of[id(style)], style), seeds["story"])

    def names(image_path, style_path, style, seeds, extra_tags=()):
        stem = result_name(image_path, style_path, [seed_tag(args.mode, seeds)] + tags + list(extra_tags))
        slug = storyteller.story_slug(style, story_config, seeds["story"]) if args.mode == "story" else ""
        return stem, (f"{stem} - {slug}" if slug else stem)

    def image_graph(image_path, style, request, seeds, size, reference, prompt_text=None, negative_text=None):
        """The graph of one job. With prompt_text, the stored prompt goes to the encoder and no LLM runs."""
        graph = json.loads(json.dumps(template))
        graph[NODE_IMAGE]["inputs"]["image"] = uploaded[image_path]
        enc = graph[NODE_ENCODE]["inputs"]
        if uses_llm and prompt_text is None:
            add_enhancer(graph, request, system_prompt, seeds["prompt"], args.prompt_thinking, args.prompt_max_tokens,
                         identity_on(style, args) and (crops.get(image_path) or True), identity_request_of(style))
            if args.title:
                add_titler(graph, args.title_language)
        else:
            enc["prompt"] = request if prompt_text is None else prompt_text
        if reference:
            # A small pixel reference keeps the silhouette, markings, and colors, but too little surface
            # texture to copy. The output has its own canvas latent.
            enc["resolution"] = reference
        soft = softened.get((image_path, soften_of(style, args.source_soften), gray_face(style)))
        if soft:
            # The image model gets the blurred copy; the writer and the titler keep the sharp source (NODE_IMAGE).
            graph[NODE_SOFT_IMAGE] = {"class_type": "LoadImage", "_meta": {"title": "Source, blurred for the image model"},
                                      "inputs": {"image": soft}}
            enc["images.image_1"] = [NODE_SOFT_IMAGE, 0]
        width, height = size
        graph[NODE_OUTPUT_LATENT] = {"class_type": "EmptyLatentImage", "_meta": {"title": "Output canvas"},
                                     "inputs": {"width": width, "height": height, "batch_size": 1}}
        graph[NODE_SAMPLE]["inputs"]["latent_image"] = [NODE_OUTPUT_LATENT, 0]
        if has_negative:
            if writes_negative and negative_text is not None:
                enc["negative_prompt"] = negative_text
            elif args.negative_source == "story":
                add_negative_writer(graph, build_negative(style), card_of(style, seeds, image_path))
            elif args.negative_source == "written":
                add_negative_writer(graph, build_negative(style))
            else:
                enc["negative_prompt"] = build_negative(style)
        if args.guidance == "nag" and has_negative:
            # NAG applies the negative prompt inside attention, so cfg stays 1 and Turbo does not overshoot.
            graph[NODE_NAG] = {"class_type": NAG_NODE, "_meta": {"title": "Normalized Attention Guidance"},
                               "inputs": {"model": [NODE_CACHE, 0], "nag_scale": scale,
                                          "nag_alpha": args.nag_alpha, "nag_tau": args.nag_tau}}
            graph[NODE_GUIDER]["inputs"]["model"] = [NODE_NAG, 0]
        elif scale is not None:
            graph[NODE_GUIDER]["inputs"]["cfg"] = scale
        if args.cache_device:
            graph[NODE_CACHE]["inputs"]["device"] = args.cache_device
        if args.cache_dtype:
            graph[NODE_CACHE]["inputs"]["dtype"] = args.cache_dtype
        graph[NODE_NOISE]["inputs"]["noise_seed"] = seeds["image"]
        if not args.server_copy:
            graph[NODE_SAVE] = {"class_type": "PreviewImage", "_meta": {"title": "Result (temporary)"},
                                "inputs": {"images": [NODE_DECODE, 0]}}
        lora_routing.apply(graph, loras_of(style), model_node, NODE_ENCODE)
        lora_routing.apply_references(graph, refs_of(style), NODE_ENCODE, upload_ref)
        return graph

    def save_result(entry, name, image_path, style_path, style, seeds, size, reference, extra) -> str | None:
        """Download the result image, write the prompt files and the manifest; return the file name."""
        target = None
        for node_out in entry.get("outputs", {}).values():
            for image in node_out.get("images", []):
                target = out_dir / f"{name}.png"
                if target.exists():  # the same title twice: keep both
                    target = out_dir / f"{name} {uuid.uuid4().hex[:4]}.png"
                target.write_bytes(client.download(image))
        if target is None:
            return None
        if extra.get("written_prompt"):
            target.with_suffix(".prompt.txt").write_text(extra["written_prompt"], encoding="utf-8")
        if extra.get("written_negative"):
            target.with_suffix(".negative.txt").write_text(extra["written_negative"], encoding="utf-8")
        if args.manifest:
            width, height = size
            card = card_of(style, seeds, image_path) if args.mode == "story" else ""
            manifest = {**manifest_base, "result": target.name, "source": image_path.name,
                        "source_sha256": sha256_file(image_path), "source_cleanup": source_fill.get(image_path, 0.0),
                        "source_soften": soften_of(style, args.source_soften),
                        "identity_notes": identity_on(style, args) if uses_llm else None,
                        "source_place": place_of.get(image_path),
                        "scene_reading": scene_of.get((image_path, style_path)),
                        "style": style_path.stem,
                        "style_sha256": sha256_file(style_path), "mode": args.mode, "seeds": seeds,
                        "canvas": [width, height], "reference_size": reference,
                        "reference_source": args.reference_size, "story_card": card,
                        "story_lines": args.story_line, "placeholders": fills,
                        "rules_positive": args.rules_positive, "rules_negative": args.rules_negative,
                        "negative_source": args.negative_source if has_negative else None,
                        "guidance": args.guidance if has_negative or scale is not None else "cfg",
                        "guidance_scale": scale if scale is not None else 1.0,
                        "loras": lora_routing.describe(loras_of(style)),
                        "style_references": [Path(p).name for e in refs_of(style) for p in e.get("images", [])],
                        "server_seconds": execution_seconds(entry), **extra}
            for key in ("written_prompt", "written_negative"):
                if manifest.get(key):
                    manifest[key + "_sha256"] = prompt_hygiene.sha256_text(manifest.pop(key))
            target.with_suffix(".json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False),
                                                   encoding="utf-8")
        return target.name

    if args.check_graph:
        problems = 0
        for image_path, style_path, style, prompt, seeds, size, reference in jobs:
            graphs = {"render": image_graph(image_path, style, prompt, seeds, size, reference,
                                            prompt_text="check" if two_phase else None, negative_text="check")}
            if two_phase and not args.written_prompt:
                negative = ((build_negative(style), card_of(style, seeds, image_path) if args.negative_source == "story" else "")
                            if writes_negative else None)
                image_node = {**template[NODE_IMAGE], "inputs": {**template[NODE_IMAGE]["inputs"],
                                                                "image": image_path.name}}
                graphs["write"] = write_graph(image_node, prompt, system_prompt, seeds["prompt"], args.prompt_thinking,
                                              args.prompt_max_tokens, negative, identity_on(style, args),
                                              identity_request_of(style))
            for phase, graph in graphs.items():
                found = graph_check.check(graph, info)
                problems += len(found)
                for line in found:
                    print(f"PROBLEM {image_path.name} x {style_path.stem} ({phase}): {line}")
        print(f"checked {len(jobs)} job(s): {problems} problem(s)")
        return 1 if problems else 0

    if two_phase:
        return run_two_phase(args, jobs, client, system_prompt, writes_negative, image_graph, names, save_result,
                             card_of, build_negative)

    queued = []
    for image_path, style_path, style, prompt, seeds, size, reference in jobs:
        graph = image_graph(image_path, style, prompt, seeds, size, reference)
        stem, fallback = names(image_path, style_path, style, seeds)
        graph[NODE_SAVE]["inputs"]["filename_prefix"] = "style-transfer/" + re.sub(r"[^\w ,()-]", "_", stem)
        if not args.server_copy:
            graph[NODE_SAVE]["inputs"].pop("filename_prefix")
        queued.append((client.queue(graph), image_path, style_path, style, stem, fallback, graph, seeds, size,
                       reference))
    print(f"queued {len(queued)} job(s): {len(images)} image(s) x {len(styles)} style(s)")

    failures = 0
    for prompt_id, image_path, style_path, style, stem, fallback, graph, seeds, size, reference in queued:
        entry = client.wait(prompt_id)
        written_text = "".join(entry.get("outputs", {}).get(NODE_ENH_SHOW, {}).get("text", [])).strip()
        thinking_on = graph.get(NODE_ENH_GENERATE, {}).get("inputs", {}).get("thinking")
        retries = []
        failed = entry.get("status", {}).get("status_str") == "error"
        # card81 (2026-11-22): with thinking, the writer once collapsed into 20,000 characters of quotes and articles
        # ('" " the " a, " the'), and the image model drew a realistic picture; a collapse gets the same retry.
        collapse = prompt_hygiene.degenerate_reasons(written_text)
        if NODE_ENH_GENERATE in graph and (not written_text or collapse) and thinking_on and not failed:
            # The reasoning used the whole token budget and no prompt was written; the image would
            # be an unchanged copy of the source. Run the job again without thinking.
            print(f"retry  {image_path.name} x {style_path.stem}: "
                  f"{'; '.join(collapse) or 'empty written prompt'} with thinking, running again without thinking")
            graph[NODE_ENH_GENERATE]["inputs"]["thinking"] = False
            graph[NODE_ENH_GENERATE]["inputs"]["max_length"] = ENHANCER_MAX_TOKENS
            entry = client.wait(client.queue(graph, front=True))
            retries.append(RETRY_NO_THINKING_TAG)
        elif (NODE_ENH_GENERATE in graph and not thinking_on and args.leak_retry and not failed
              and prompt_hygiene.leak_reasons(written_text)):
            # Without thinking, the writer sometimes puts its own reasoning into the prompt; the image then
            # drifts to realism. Run the job again with thinking, which keeps the reasoning out.
            print(f"retry  {image_path.name} x {style_path.stem}: reasoning in the written prompt "
                  f"({'; '.join(prompt_hygiene.leak_reasons(written_text))}), running again with thinking")
            graph[NODE_ENH_GENERATE]["inputs"]["thinking"] = True
            graph[NODE_ENH_GENERATE]["inputs"]["max_length"] = ENHANCER_MAX_TOKENS_THINKING
            entry = client.wait(client.queue(graph, front=True))
            retries.append(RETRY_THINKING_TAG)
        for tag in retries:
            retried = with_tag(stem, tag)
            stem, fallback = retried, fallback.replace(stem, retried, 1)
        if entry.get("status", {}).get("status_str") == "error":
            failures += 1
            print(f"FAILED {image_path.name} x {style_path.stem}: {json.dumps(entry['status'].get('messages'))[:500]}")
            continue
        title = clean_title("".join(entry.get("outputs", {}).get(NODE_TITLE_SHOW, {}).get("text", [])))
        name = f"{stem} - {title}" if title else fallback
        written = with_identity("".join(entry.get("outputs", {}).get(NODE_ENH_SHOW, {}).get("text", [])), entry)
        written_negative = "".join(entry.get("outputs", {}).get(NODE_NEG_SHOW, {}).get("text", []))
        extra = {"thinking_requested": args.prompt_thinking if uses_llm else None,
                 "thinking_used": graph.get(NODE_ENH_GENERATE, {}).get("inputs", {}).get("thinking"),
                 "retries": retries, "phases": "one graph", "title": title,
                 "written_prompt": written, "written_negative": written_negative}
        target = save_result(entry, name, image_path, style_path, style, seeds, size, reference, extra)
        if target:
            secs = execution_seconds(entry)
            timing = f"{secs:.1f} s" if secs is not None else "n/a"
            print(f"done   {target}  reference={reference}  server time {timing}")
    return 1 if failures else 0


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_manifest(args, client, template: dict, system_prompt: str, story_config: dict) -> dict:
    """The part of the manifest that is the same for all results of this run."""
    try:
        system_stats = client.get_json("/system_stats").get("system", {})
    except Exception:  # noqa: BLE001  (a missing version must not stop the run)
        system_stats = {}
    models = {node.get("class_type"): {k: v for k, v in node.get("inputs", {}).items() if k.endswith("_name")}
              for node in template.values() if any(k.endswith("_name") for k in node.get("inputs", {}))}
    return {"code_sha256": sha256_file(Path(__file__)), "graph_sha256": sha256_file(Path(args.graph)),
            "system_prompt_sha256": prompt_hygiene.sha256_text(system_prompt) if system_prompt else None,
            "story_config_sha256": (prompt_hygiene.sha256_text(json.dumps(story_config, sort_keys=True))
                                    if story_config else None),
            "word_target": args.prompt_words, "models": models,
            "writer_model": ENHANCER_MODEL if args.mode in ("adapt", "story") else None,
            "comfyui_version": system_stats.get("comfyui_version"),
            "cleanup_version": prompt_hygiene.CLEANUP_VERSION}


def write_graph(image_graph_node: dict, request: str, system_prompt: str, seed: int, thinking: bool,
                max_tokens: int, negative: tuple[str, str] | None, identity: bool | str = False,
                identity_request: str | None = None) -> dict:
    """An LLM-only graph: the source image, the prompt writer, (if asked) the negative writer and the identity
    notes."""
    graph = {NODE_IMAGE: image_graph_node, NODE_ENCODE: {"inputs": {}}}
    add_enhancer(graph, request, system_prompt, seed, thinking, max_tokens, identity, identity_request)
    if negative is not None:
        add_negative_writer(graph, *negative)
    del graph[NODE_ENCODE]  # no image is rendered in the write phase
    return graph


WARM_UP_REQUEST = "Describe the input image in one sentence."
WARM_UP_MAX_TOKENS = 32


def warm_up(client, image_node: dict) -> None:
    """One short writer job before phase 1. After a fresh load, the first writer job decodes at a fraction of
    the normal speed (part of the weights stays in RAM); the next writer job is fast. The short job takes that
    slow first place, so that no long prompt runs in it."""
    graph = {NODE_IMAGE: image_node, NODE_ENCODE: {"inputs": {}}}
    add_enhancer(graph, WARM_UP_REQUEST, "", 0, False, WARM_UP_MAX_TOKENS)
    del graph[NODE_ENCODE]
    started = time.monotonic()
    client.wait(client.queue(graph))
    print(f"warm-up of the prompt writer: {time.monotonic() - started:.0f} s")


def store_prompt(store: Path, key: str, text: str, meta: dict) -> None:
    """Write a prompt into the store once: an existing key is never overwritten (the writer is not
    reproducible, so the stored text is the reproducible artifact)."""
    store.mkdir(parents=True, exist_ok=True)
    try:
        with (store / f"{key}.txt").open("x", encoding="utf-8") as f:
            f.write(text)
    except FileExistsError:
        return
    (store / f"{key}.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")


def upload_source(client, path: Path, args, source_fill: dict) -> str:
    """Upload the source image, cleaned of a fault at the frame edge when the cleanup finds one."""
    send = path
    if args.source_cleanup:
        send, source_fill[path] = source_cleanup.clean(path, Path(args.output) / ".source-clean")
        if source_fill[path]:
            print(f"source {path.name}: filled a fault at the frame edge ({source_fill[path]:.1%} of the frame)")
    return client.upload_image(send)


# The place check (2026-11-20). The story card offers a setting "only when the input shows no place", and the
# writer decides. With a 600-800-word prompt the writer misjudged a stone wall as no place (card25: the hunter pair
# in a meadow in 1 of 2 seeds; card9 and card12 the same). A short question on the source decides instead, before
# the card is drawn.
PLACE_SETTING_LABELS = ("Setting", "Spot", "Place details", "Time of day")  # card42: a lamp-lit evening added lamps
PLACE_MAX_TOKENS = 8
PLACE_REQUEST = (
    "Look at the background behind the main subjects of this image. Answer PLACE if it shows a place or a surface "
    "that a viewer can name, sharp or softly out of focus: a room, a hall, a street, a tunnel, a wall, a fence, a "
    "curtain, a floor, a landscape, a forest. Answer PLAIN only for an even, featureless backdrop or a blur without "
    "any surface that a viewer can name. Answer with exactly one of the two words PLACE or PLAIN.")
PLACE_WORDS = {True: "a place of its own, the story card leaves out its setting lines", False: "plain"}
NODE_PLACE_IMAGE, NODE_PLACE_GENERATE, NODE_PLACE_SHOW = "130", "131", "132"


def source_shows_place(client, path: Path) -> bool | None:
    """PLACE (True) or PLAIN (False) from the writer model for the background of the source; None on a failure."""
    graph = {NODE_PLACE_IMAGE: {"class_type": "LoadImage", "_meta": {"title": "Source"},
                                "inputs": {"image": client.upload_image(path)}},
             NODE_ENH_CLIP: {"class_type": "CLIPLoader", "_meta": {"title": "Prompt enhancer model"},
                             "inputs": {"clip_name": ENHANCER_MODEL, "type": "qwen_image", "device": "default"}},
             NODE_PLACE_GENERATE: {"class_type": "TextGenerate", "_meta": {"title": "Place or plain"},
                                   "inputs": {"clip": [NODE_ENH_CLIP, 0], "image": [NODE_PLACE_IMAGE, 0],
                                              "system_prompt": IDENTITY_SYSTEM_PROMPT, "prompt": PLACE_REQUEST,
                                              "max_length": PLACE_MAX_TOKENS, "sampling_mode": "off",
                                              "thinking": False, "use_default_template": True, "mtp": "auto"}},
             NODE_PLACE_SHOW: {"class_type": "PreviewAny", "_meta": {"title": "Place answer"},
                               "inputs": {"source": [NODE_PLACE_GENERATE, 0]}}}
    entry = client.wait(client.queue(graph))
    answer = "".join(entry.get("outputs", {}).get(NODE_PLACE_SHOW, {}).get("text", [])).upper()
    if not answer.strip():
        return None
    return "PLAIN" not in answer  # the model sometimes names the place ("WALL") instead of the word PLACE


# The surfaces of the place (card69, 2026-11-21). The identity notes read the hunter wall right ('a plain gray plaster
# wall with faint vertical streaks'), but the writer wrote its own 'gray stone wall' with 'stone joints, mortar
# seams' and 'warm ochre' in 1 of 3 seeds, and the image model drew stone blocks on a warm wall. The notes come after
# the written prompt; a line of the story card weighs more. A short question names the surfaces before the card is
# drawn, and the Place line carries them.
SURFACES_MAX_TOKENS = 60
SURFACES_MAX_WORDS = 30
SURFACES_REQUEST = (
    "Name the main surfaces of the place behind the main subjects of this image in one short sentence: each surface "
    "with its exact material, its surface pattern, and its color family and lightness, as the photo shows them (for "
    "example a smooth pale yellow painted wall). A "
    "plain surface stays plain, without joints or blocks that the photo does not show. Do not describe the people, "
    "the style, the light, or the mood. Write at most 25 words.")
NODE_SURFACES_IMAGE, NODE_SURFACES_GENERATE, NODE_SURFACES_SHOW = "136", "137", "138"


def clean_surfaces(answer: str) -> str | None:
    """The first sentence of the surfaces answer, without blur words; None when it is empty or too long."""
    text = clean_identity(answer.strip().split("\n")[0]).strip()
    text = re.split(r"(?<=[.!?])\s", text)[0].strip().rstrip(".")
    if not text or len(text.split()) > SURFACES_MAX_WORDS:
        return None
    return text[:1].lower() + text[1:]


def source_place_surfaces(client, path: Path) -> str | None:
    """The main surfaces of the place of the source in plain visual words; None on a failure."""
    graph = {NODE_SURFACES_IMAGE: {"class_type": "LoadImage", "_meta": {"title": "Source"},
                                   "inputs": {"image": client.upload_image(path)}},
             NODE_ENH_CLIP: {"class_type": "CLIPLoader", "_meta": {"title": "Prompt enhancer model"},
                             "inputs": {"clip_name": ENHANCER_MODEL, "type": "qwen_image", "device": "default"}},
             NODE_SURFACES_GENERATE: {"class_type": "TextGenerate", "_meta": {"title": "Place surfaces"},
                                      "inputs": {"clip": [NODE_ENH_CLIP, 0], "image": [NODE_SURFACES_IMAGE, 0],
                                                 "system_prompt": IDENTITY_SYSTEM_PROMPT, "prompt": SURFACES_REQUEST,
                                                 "max_length": SURFACES_MAX_TOKENS, "sampling_mode": "off",
                                                 "thinking": False, "use_default_template": True, "mtp": "auto"}},
             NODE_SURFACES_SHOW: {"class_type": "PreviewAny", "_meta": {"title": "Surfaces answer"},
                                  "inputs": {"source": [NODE_SURFACES_GENERATE, 0]}}}
    entry = client.wait(client.queue(graph))
    return clean_surfaces("".join(entry.get("outputs", {}).get(NODE_SURFACES_SHOW, {}).get("text", [])))


# The head of a creature (card68 A/B, 2026-11-21). Any blur of a non-human head (a monster) erases its build: the
# image model drew a human old man in 3 of 3 seeds, also with strong head words in the prompt. The face detectors do
# not find such a head, so a short question gives its box, and the source stays sharp there
# (source_cleanup.soften). Only an answer of exactly four numbers counts; any other answer means no creature.
CREATURE_MAX_TOKENS = 24
CREATURE_SCALE = 1000         # the box numbers are parts of the width and the height on this scale
CREATURE_REQUEST = (
    "Look at the main characters of this image. If one of them has a non-human head with a face (a monster, an alien, "
    "a troll-like creature, a puppet, a mask), give the box around that head, with its hair and ears, as four integers "
    "x1 y1 x2 y2 on a scale from 0 to 1000 of the image width and height. A person, an animal, a toy, or a prop is "
    "not such a character; then answer NONE. Answer with the four integers or the word NONE only.")
CREATURE_BOX = re.compile(r"^[\s\[(]*(\d+)[\s,]+(\d+)[\s,]+(\d+)[\s,]+(\d+)[\s\]).]*$")
NODE_CREATURE_IMAGE, NODE_CREATURE_GENERATE, NODE_CREATURE_SHOW = "133", "134", "135"


def parse_creature_box(answer: str) -> tuple[float, float, float, float] | None:
    """The box (left, top, right, bottom as parts of 1) of an answer of exactly four numbers; None otherwise."""
    match = CREATURE_BOX.match(answer.strip())
    if not match:
        return None
    left, top, right, bottom = (int(v) / CREATURE_SCALE for v in match.groups())
    if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
        return None
    return left, top, right, bottom


def creature_head_box(client, path: Path) -> tuple[float, float, float, float] | None:
    """The head box of a main creature of the source from the writer model; None without one or on a failure."""
    graph = {NODE_CREATURE_IMAGE: {"class_type": "LoadImage", "_meta": {"title": "Source"},
                                   "inputs": {"image": client.upload_image(path)}},
             NODE_ENH_CLIP: {"class_type": "CLIPLoader", "_meta": {"title": "Prompt enhancer model"},
                             "inputs": {"clip_name": ENHANCER_MODEL, "type": "qwen_image", "device": "default"}},
             NODE_CREATURE_GENERATE: {"class_type": "TextGenerate", "_meta": {"title": "Creature head box"},
                                      "inputs": {"clip": [NODE_ENH_CLIP, 0], "image": [NODE_CREATURE_IMAGE, 0],
                                                 "system_prompt": IDENTITY_SYSTEM_PROMPT, "prompt": CREATURE_REQUEST,
                                                 "max_length": CREATURE_MAX_TOKENS, "sampling_mode": "off",
                                                 "thinking": False, "use_default_template": True, "mtp": "auto"}},
             NODE_CREATURE_SHOW: {"class_type": "PreviewAny", "_meta": {"title": "Creature answer"},
                                  "inputs": {"source": [NODE_CREATURE_GENERATE, 0]}}}
    entry = client.wait(client.queue(graph))
    return parse_creature_box("".join(entry.get("outputs", {}).get(NODE_CREATURE_SHOW, {}).get("text", [])))


# Scene reading (2026-11-20). The story card only asked to keep the emotion of the photo one step stronger, so the
# writer copied the faces of the photo. User: ChatGPT reads the context, the emotions, and the tone "two levels
# higher" (the hunter pair as a tense face-off, the unicorn rider as a wild comic charge), and our pipeline lost or
# invented story props. A separate call of the writer model reads the enlarged sharp source as one moment of a story
# in five labeled lines. Checks (card38): a numbered request was ignored (the model only described the picture); the
# style's mood words in the request leaked into the reading (a tense face-off became a "loving comedic duo", a grumpy
# creature "manic delight"), so the request has no style text: the writer adds the tone of the style. A face can mean
# more than one thing, so the request asks to decide from the context. Thinking mode used all its tokens to think and
# gave no answer, so it is off.
SCENE_MAX_TOKENS = 400
SCENE_SIDE = 1024             # the long side of the enlarged source for the reading (the sources are 210-417 px high)
SCENE_LABEL = "Scene reading"
SCENE_FRAME = ("how a director reads the input image; act it out in the means of the style, stronger and clearer than "
               "the photo but the same feeling, only clearer (never a stronger neighbor feeling, such as anger for unease), "
               "and keep each story prop with its parts at its size in the input; a figure or a doll "
               "fixed to a prop stays a small, rough part of that prop, never a separate standing figure: ")
SCENE_SYSTEM_PROMPT = "You read an image as one moment of a story, as the director of an animated picture would."
SCENE_REQUEST = "\n".join([
    "This photo will become one frame of an animated picture. Read the photo itself; do not give it a mood of its own.",
    "Read it as a director reads a moment of a story: not what things look like, but what happens and what people feel.",
    "A face can mean more than one thing: a wide-open mouth can be fear, rage, or wild joy, and narrowed eyes can be "
    "anger, focus, or suspicion. Decide from the whole context: the costume, the props, the place, the pose, and what "
    "the people are doing.",
    "When the people pose for the camera (a portrait, a poster, a group photo), the feeling is the attitude that each "
    "one shows to the viewer (calm, cool, confident, proud, playful), not a quarrel between them; a person who looks "
    "at the camera does not look at the other people, and a composed face is never fury.",
    "Name a creature by its visible size, face shape, and skin, never by a guess of its species or kind, because "
    "the artist draws the guessed kind.",
    "Answer in exactly these five lines, one or two sentences each:",
    "Moment: what happens, what led to it, and what each person is trying to do.",
    "Feelings: for each main person or creature, from left to right, the feeling in strong words, why, and how they "
    "relate to the others.",
    "Tone: two or three words.",
    "Acting: for each main person or creature, the brows, the eyelids (heavy and half-closed, or wide), the eyes "
    "(open), the mouth, the head, the hands, and the body line that show this same feeling a step stronger, never a "
    "different feeling; read the lids first: upper lids that droop over the top of the eye read tired, bored, or "
    "sad, lids narrowed under brows pulled down read anger or suspicion, and lids opened wide read surprise, fear, "
    "or joy; the permanent lines of a "
    "face (wrinkles, scars, a heavy brow ridge) are its shape, not an expression; add an emotion symbol (a sweat drop, a tear streak, an anger mark, "
    "a blush) only when it fits.",
    "Props: each object that tells the story, by its visible shapes, materials, and colors, and how it is held; never a "
    "guess of what a homemade object stands for (a cardboard box with a painted face, not a robot); a small bundle of "
    "cloth or foam stays a bundle of shapes and colors, never a figurine or a doll.",
    "Do not describe the clothes, the hair, or the place. Write nothing after the five lines.",
])
SCENE_CLEANUP = (
    (r"\n\s*(?:Wait|Let me|Note|Check|Hmm|Actually)\b.*", ""),  # a self-check, and all after it
    (r"(Props:[^\n]*)\n\s*\n.*", r"\1"),                       # nothing after the Props line
    (r"\s*\((?:note|about|approximately|\d+ words).*", ""),     # a word count or a note, and all after it
    (r"\s*\b\d+ words\.?\s*$", ""),                            # a word count at the end
    (r"\beyes (?:squeezed|screwed|pressed) (?:shut|closed)\b", "eyes narrowed but open"),
    (r"\b(?:blurred|blurry|out-of-focus|out of focus|soft-focus|defocused|bokeh)\s*(?:,\s*)?", ""),
    # card46: a reading listed denied symbols ('no tears, no anger marks, no blush'); the image model draws every
    # named item, also an item after 'no', so a denied item goes, with the list of denials it is in
    (r"[;,]?\s*\b(?:no|without) (?:a |an )?(?:sweat drops?|tears?|tear streaks?|anger marks?|blush(?:es)?|emotion "
     r"symbols?)\b(?:,?\s*(?:and |or )?(?:no )?(?:a |an )?(?:sweat drops?|tears?|tear streaks?|anger marks?|"
     r"blush(?:es)?|emotion symbols?)\b)*", ""),
    (r",?\s*\bwith no\b[^.;]*", ""),                            # a denied list ('with no anger, no fear')
) + FIXED_FIGURE_CLEANUP + (
    (r"\s+", " "),
)
NODE_SCENE_IMAGE, NODE_SCENE_GENERATE, NODE_SCENE_SHOW = "140", "141", "142"


def clean_scene(text: str) -> str:
    """The scene reading after SCENE_CLEANUP; a reading written as a JSON object becomes the labeled lines."""
    try:
        fields = json.loads(text.strip().removeprefix("```json").strip("`\n "))
        if isinstance(fields, dict):
            text = "\n".join(f"{label}: {value}" for label, value in fields.items())
    except ValueError:
        pass
    for pattern, replace in SCENE_CLEANUP:
        text = re.sub(pattern, replace, text, flags=re.IGNORECASE | re.DOTALL)
    return text.strip()


def read_scene(client, path: Path, style: dict, args) -> str | None:
    """The writer model's reading of the source as one moment of a story (five labeled lines); None on a
    failure or an empty answer."""
    request = SCENE_REQUEST
    big = source_cleanup.enlarged(path, Path(args.output) / ".source-scene", SCENE_SIDE)
    graph = {NODE_SCENE_IMAGE: {"class_type": "LoadImage", "_meta": {"title": "Source for the scene reading"},
                                "inputs": {"image": client.upload_image(big or path)}},
             NODE_ENH_CLIP: {"class_type": "CLIPLoader", "_meta": {"title": "Prompt enhancer model"},
                             "inputs": {"clip_name": ENHANCER_MODEL, "type": "qwen_image", "device": "default"}},
             NODE_SCENE_GENERATE: {"class_type": "TextGenerate", "_meta": {"title": "Scene reading"},
                                   "inputs": {"clip": [NODE_ENH_CLIP, 0], "image": [NODE_SCENE_IMAGE, 0],
                                              "system_prompt": SCENE_SYSTEM_PROMPT, "prompt": request,
                                              "max_length": SCENE_MAX_TOKENS, "sampling_mode": "off",
                                              "thinking": False, "use_default_template": True, "mtp": "auto"}},
             NODE_SCENE_SHOW: {"class_type": "PreviewAny", "_meta": {"title": "Scene reading"},
                               "inputs": {"source": [NODE_SCENE_GENERATE, 0]}}}
    entry = client.wait(client.queue(graph))
    text = clean_scene("".join(entry.get("outputs", {}).get(NODE_SCENE_SHOW, {}).get("text", [])))
    return text or None


# card47: the palette rule of the writer's system prompt says that each element keeps its hue family, but the writer
# still wrote 'keeping the woman's hair as deep warm brown' for black hair under a warm palette. A card line weighs
# more, so every story card gets this line.
COLORS_LABEL = "Colors"
# card51: 'the palette color nearest to it' made a red jacket terracotta (the nearest red of a warm palette) and gave
# brown hair the cool ink of a four-ink style, so a hue without its own palette color keeps the input color.
COLORS_LINE = ("each person, animal, and creature keeps the hue of the input for the hair, the fur, the skin, and each "
               "garment: use a palette color of the same hue, and where the palette has none, keep the input's own "
               "color; a hue never moves to a neighbor hue of the palette (a yellow scarf never turns orange, a violet "
               "dress never turns blue), and a very dark or very light part keeps its value and its hue (a black part stays a "
               "neutral or cool black, never a brown-black, because a warm palette turns black cloth brown); the place keeps the color temperature of each surface of the input (a "
               "cool surface stays cool, a warm one warm), also under a warm or cool light; a light pool or a sun patch that "
               "the style asks for lies on the ground as a lighter tint of the ground's own color, never on a cool wall; name the color of each "
               "surface and each skin by the hue and the lightness that the input shows, never next to a list of the "
               "palette colors")
# card65: the writer copied the example hue words of this line ('a pale sage green') to a tan creature skin, and the
# image model drew a green creature. The line gives no hue words now.
# card63: the writer wrote a black uniform as 'deep warm brown-black', and the image model drew a brown coat.
# card61: the writer wrote a cool wall 'within an earthy palette of warm browns, cream, terracotta, ochre', and the
# image model painted the wall warm beige, although the same sentence said 'cool'.
# card50: the identity notes named the 'layered shoulder cape' of a coat, but the writer wrote 'the coat as one broad
# dark mass with two fold lines', against the detail rule of its system prompt.
GARMENTS_LABEL = "Garments"
# card52: in a four-ink style, the layers became more bands, and the bands took the cool ink (a red jacket in teal).
# card53: a style that turns a figure into a sign of a few large shapes (its detail budget) drew the people as full
# realistic figures, and this line asked for more parts, so the line yields to the shape count of the style.
GARMENTS_LINE = ("each garment keeps its layers and visible parts of the input as separate clear shapes, as many as "
                 "the detail budget of the style allows, in the means of the style, each shape in the hue of that part "
                 "in the input; the sentence of each person names each layer by its shape and place (for example a "
                 "short hood over the shoulders of a jacket, a wide belt over a tunic), never only 'flat panels'; "
                 "a dark garment is never a shapeless mass, and a style that turns a figure into a few "
                 "large flat shapes keeps that reduction; when the style shows more of a figure than the input does (the "
                 "legs and the feet of a full-length figure), the hidden part continues the garments of the input in "
                 "the hues of those garments, never a new garment in a palette color")
# card64: the Art Nouveau Arch writer extended a waist-up woman to full length with 'a long skirt of dusty rose and
# terracotta' that the input does not show.


def with_colors_line(config: dict) -> dict:
    """The story config with the Colors and Garments lines after the Concept line."""
    lines = list(config.get("lines", []))
    at = next((i + 1 for i, x in enumerate(lines) if x.get("label") == "Concept"), 0)
    lines[at:at] = [{"label": COLORS_LABEL, "text": COLORS_LINE, "probability": 1.0},
                    {"label": GARMENTS_LABEL, "text": GARMENTS_LINE, "probability": 1.0}]
    return {**config, "lines": lines}


def with_scene_line(config: dict, reading: str | None) -> dict:
    """The story config with the scene reading as a line after the Concept line (unchanged without a reading)."""
    if not reading:
        return config
    lines = list(config.get("lines", []))
    at = next((i + 1 for i, x in enumerate(lines) if x.get("label") == "Concept"), 0)
    lines.insert(at, {"label": SCENE_LABEL, "text": SCENE_FRAME + reading, "probability": 1.0})
    return {**config, "lines": lines}


# card43: the writer filled the background template of a style ('simplified architecture', 'a pale blue distance')
# with shelves and a window in front of a plank wall, against its system rule. A line of the story card weighs more,
# because the writer gets it as the request, so a place source gets this line in the place of the setting lines.
PLACE_LABEL = "Place"
PLACE_LINE = ("the place of the input image, redrawn as it is: the same surfaces with their material, pattern, and "
              "colors; add no furniture, shelf, window, door, lamp, sign, or prop that the input does not show, and "
              "no sky or far distance where the input shows a wall; only an element that a mandatory feature or a "
              "background rule of the style names by itself (an ornament, a surreal contradiction, furniture that "
              "the style says it always builds) can join this place, and then it keeps the colors and the light of "
              "the input place; a rule about the look of a background (its painting, its light, its depth) never "
              "adds an object, and a sentence of the place never names an object that is not in the input")
# card68: with 'a furnished room that the style always builds around a plain backdrop', the Anime writer furnished
# the hunter wall (a coat rack with a cloak on hooks, a lantern) in 1 of 3 seeds, although no Anime rule asks for
# furniture; the exception now needs a style rule that names the element.
# card64: with 'only a mandatory feature', the Vintage Cozy writer kept the bare stone wall, although its background
# rules say 'Always keep a furnished cozy environment' and 'If the source background is plain ... build a cozy
# furnished interior'.


def without_setting_lines(config: dict, surfaces: str | None = None) -> dict:
    """The story config for a source with a place of its own: without the lines that build a setting (for a source
    without a place), and with a Place line after the Concept line that keeps the place as it is."""
    lines = [x for x in config.get("lines", []) if x.get("label") not in PLACE_SETTING_LABELS]
    at = next((i + 1 for i, x in enumerate(lines) if x.get("label") == "Concept"), 0)
    text = PLACE_LINE.replace("the place of the input image,", f"the place of the input image ({surfaces}),", 1) \
        if surfaces else PLACE_LINE
    lines.insert(at, {"label": PLACE_LABEL, "text": text, "probability": 1.0})
    return {**config, "lines": lines}


def upload_people_crop(client, path: Path, args) -> str | None:
    """Upload the crop of the people of the source for the identity notes; None for a source without people."""
    crop = source_cleanup.people_crop(path, Path(args.output) / ".source-people")
    return client.upload_image(crop) if crop else None


def upload_softened(client, path: Path, args, fraction: float, gray_heads: bool = False,
                    sharp_boxes: tuple = ()) -> str:
    """Upload the blurred copy of the (cleaned) source that the image model gets; sharp_boxes stay sharp."""
    send = source_cleanup.clean(path, Path(args.output) / ".source-clean")[0] if args.source_cleanup else path
    return client.upload_image(source_cleanup.soften(send, Path(args.output) / ".source-soft", fraction,
                                                    gray_heads, sharp_boxes))


def run_two_phase(args, jobs, client, system_prompt, writes_negative, image_graph, names, save_result, card_of,
                  negative_rules) -> int:
    """Phase 1 writes, cleans, checks, and stores the prompts of all jobs. Phase 2 renders all jobs from them."""
    store = Path(args.prompt_store) if args.prompt_store else None
    template_image = json.loads(Path(args.graph).read_text(encoding="utf-8"))[NODE_IMAGE]
    written: dict[tuple, dict] = {}
    failures = 0
    warmed_up = not args.warm_up
    for image_path, style_path, style, request, seeds, size, reference in jobs:
        job_key = (image_path, style_path, request, seeds["prompt"], seeds["story"])
        if job_key in written:
            continue  # another image seed of the same prompt
        if args.written_prompt:
            text = prompt_hygiene.clean(Path(args.written_prompt).read_text(encoding="utf-8"))
            written[job_key] = {"key": None, "text": text, "negative": None, "retries": [],
                                "thinking_used": None, "source_file": Path(args.written_prompt).name}
            continue
        negative = None
        if writes_negative:
            negative = (negative_rules(style), card_of(style, seeds, image_path) if args.negative_source == "story" else "")
        key = prompt_hygiene.prompt_key(
            writer_model=ENHANCER_MODEL, request=request,
            system_prompt=system_prompt + (identity_request_of(style) if identity_on(style, args) else ""),
            source_sha256=sha256_file(image_path), seed=seeds["prompt"], thinking=args.prompt_thinking,
            max_tokens=args.prompt_max_tokens, sampling=ENHANCER_SAMPLING,
            negative=json.dumps([args.negative_source, NEGATIVE_SYSTEM_PROMPT, NEGATIVE_REQUEST, negative])
            if negative else "")
        if (store / f"{key}.txt").exists() and (store / f"{key}.json").exists():
            meta = json.loads((store / f"{key}.json").read_text(encoding="utf-8"))
            meta["text"] = (store / f"{key}.txt").read_text(encoding="utf-8")
            print(f"stored {image_path.name} x {style_path.stem}: prompt {key[:12]} from the store")
            written[job_key] = meta
            continue
        node = json.loads(json.dumps(template_image))
        node["inputs"]["image"] = upload_source(client, image_path, args, {})
        crop = upload_people_crop(client, image_path, args) if identity_on(style, args) else None
        thinking, max_tokens, retries = args.prompt_thinking, args.prompt_max_tokens, []

        def run(front=False):
            graph = write_graph(node, request, system_prompt, seeds["prompt"], thinking, max_tokens, negative,
                                identity_on(style, args) and (crop or True), identity_request_of(style))
            entry = client.wait(client.queue(graph, front=front))
            raw = "".join(entry.get("outputs", {}).get(NODE_ENH_SHOW, {}).get("text", []))
            neg = "".join(entry.get("outputs", {}).get(NODE_NEG_SHOW, {}).get("text", [])).strip()
            return entry, raw, neg

        if not warmed_up:
            warm_up(client, node)
            warmed_up = True
        started = time.monotonic()
        entry, raw, neg = run()
        if entry.get("status", {}).get("status_str") == "error":
            failures += 1
            print(f"FAILED {image_path.name} x {style_path.stem}: the write job failed or was interrupted")
            continue
        words = prompt_hygiene.audit(prompt_hygiene.clean(raw))["words"]
        collapse = prompt_hygiene.degenerate_reasons(prompt_hygiene.clean(raw))  # card81: see the one-graph path
        if (words < prompt_hygiene.MIN_WORDS or collapse) and thinking:
            # The reasoning used the token budget: the answer is empty or cut. Write again without thinking.
            print(f"retry  {image_path.name} x {style_path.stem}: {words} words written with thinking "
                  f"({'; '.join(collapse) or 'budget used up'}), writing again without thinking")
            thinking, max_tokens = False, ENHANCER_MAX_TOKENS
            entry, raw, neg = run(front=True)
            retries.append(RETRY_NO_THINKING_TAG)
        text = prompt_hygiene.clean(raw)
        reasons = prompt_hygiene.leak_reasons(text)
        if reasons and not thinking and args.leak_retry:
            print(f"retry  {image_path.name} x {style_path.stem}: reasoning in the written prompt "
                  f"({'; '.join(reasons)}), writing again with thinking")
            thinking, max_tokens = True, ENHANCER_MAX_TOKENS_THINKING
            entry, raw, neg = run(front=True)
            retries.append(RETRY_THINKING_TAG)
            text = prompt_hygiene.clean(raw)
            reasons = prompt_hygiene.leak_reasons(text)
        if entry.get("status", {}).get("status_str") == "error" or not text:
            failures += 1
            print(f"FAILED {image_path.name} x {style_path.stem}: no written prompt")
            continue
        text = with_identity(text, entry)
        meta = {"key": key, "raw": raw, "negative": neg or None, "negative_sees": "the raw written prompt",
                "thinking_requested": args.prompt_thinking, "thinking_used": thinking, "retries": retries,
                "leak_reasons": reasons, "audit": prompt_hygiene.audit(text),
                "seconds": round(time.monotonic() - started, 1), "server_seconds": execution_seconds(entry),
                "source": image_path.name, "style": style_path.stem, "mode": args.mode, "seeds": seeds,
                "cleanup_version": prompt_hygiene.CLEANUP_VERSION}
        store_prompt(store, key, text, meta)
        print(f"wrote  {image_path.name} x {style_path.stem}: prompt {key[:12]}, {meta['audit']['words']} words"
              f"{', leak: ' + '; '.join(reasons) if reasons else ''}")
        written[job_key] = {**meta, "text": text}
    if args.write_only:
        print(f"wrote {len(written)} prompt(s) into {store}")
        return 1 if failures else 0

    queued = []
    for image_path, style_path, style, request, seeds, size, reference in jobs:
        meta = written.get((image_path, style_path, request, seeds["prompt"], seeds["story"]))
        if meta is None:
            continue
        graph = image_graph(image_path, style, request, seeds, size, reference, prompt_text=meta["text"],
                            negative_text=meta.get("negative"))
        stem, name = names(image_path, style_path, style, seeds, meta.get("retries", []))
        if args.server_copy:
            graph[NODE_SAVE]["inputs"]["filename_prefix"] = "style-transfer/" + re.sub(r"[^\w ,()-]", "_", stem)
        queued.append((client.queue(graph), image_path, style_path, style, name, seeds, size, reference, meta))
    print(f"queued {len(queued)} render job(s) from written prompts")
    for prompt_id, image_path, style_path, style, name, seeds, size, reference, meta in queued:
        entry = client.wait(prompt_id)
        if entry.get("status", {}).get("status_str") == "error":
            failures += 1
            print(f"FAILED {image_path.name} x {style_path.stem}: {json.dumps(entry['status'].get('messages'))[:500]}")
            continue
        extra = {"thinking_requested": meta.get("thinking_requested"), "thinking_used": meta.get("thinking_used"),
                 "retries": meta.get("retries", []), "phases": "two phases", "prompt_key": meta.get("key"),
                 "written_prompt_file": meta.get("source_file"),
                 "leak_reasons": meta.get("leak_reasons", []), "title": None,
                 "written_prompt": meta["text"], "written_negative": meta.get("negative") or ""}
        target = save_result(entry, name, image_path, style_path, style, seeds, size, reference, extra)
        if target:
            secs = execution_seconds(entry)
            print(f"done   {target}  reference={reference}  server time "
                  f"{f'{secs:.1f} s' if secs is not None else 'n/a'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
