#!/usr/bin/env python3
"""Redraw images in the styles of image-styles/*.json with a running ComfyUI server.

The script uses only the Python standard library. It reads the API graph
style-transfer.api.json, which is next to this file, and for each pair of
(image, style) it:

1. Uploads the image to the server (one time for each image).
2. Composes an edit prompt from the generation_instructions of the style.
3. Puts the job in the server queue.

It puts all jobs in the queue first, so that the GPU does not wait between
jobs. Then it downloads each result as "<image>__<style>.png".

Examples:
    python style_transfer.py photo.jpg --output out
    python style_transfer.py photos/ --styles "../image-styles/Art Nouveau Watercolor.json" --output out
    python style_transfer.py photo.jpg --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
DEFAULT_STYLES = HERE.parent / "image-styles"
DEFAULT_SERVER = "http://127.0.0.1:8188"  # ComfyUI default address
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
STYLE_SKIP_PREFIX = "_"  # files such as _baseline.png are not styles

# Node ids in style-transfer.api.json that the script changes.
NODE_IMAGE = "4"
NODE_ENCODE = "5"
NODE_CACHE = "6"
NODE_GUIDER = "7"
NODE_NOISE = "10"
NODE_SAVE = "13"

POLL_INTERVAL_S = 1.0
HTTP_TIMEOUT_S = 60
JOB_TIMEOUT_S = 1800
SEED_BITS = 48  # RandomNoise accepts large seeds, keep them readable

PLACEHOLDER_RE = re.compile(r"\{([A-Z_]+)\}")
SOURCE_TAG = "<image1>"  # Qwen Image 2.1 refers to reference image 1 with this tag

# Prompt enhancer (enhance mode): the official Qwen 3.5 9B image-to-image prompt
# enhancer of Qwen Image 2.1. The sampling values are the values of its release.
ENHANCER_MODEL = "qwen3.5_9b_qwen_image_2.1_pe_i2i.int8_convrot.safetensors"
ENHANCER_SYSTEM_PROMPT = HERE / "style-enhancer.system.md"
ENHANCER_MAX_TOKENS = 4096  # thinking block + written prompt
ENHANCER_SAMPLING = {"temperature": 1.0, "top_k": 20, "top_p": 0.95, "min_p": 0.0,
                     "repetition_penalty": 1.0, "presence_penalty": 0.0}
ENHANCER_SEED = 0  # fixed, so that ComfyUI reuses the written prompt for the same image and style
NODE_ENH_CLIP, NODE_ENH_SYSTEM, NODE_ENH_GENERATE, NODE_ENH_SHOW = "100", "101", "102", "103"
OUTLINE_INDENT = "  "


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


def as_sentence_list(items: list[str]) -> str:
    return " ".join(s.strip().rstrip(".") + "." for s in items if s.strip())


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


def build_full_prompt(style: dict, fills: dict[str, str]) -> str:
    """Give the model the complete style definition as structured JSON.

    The text encoder of Qwen Image 2.1 is the Qwen3-VL 8B language model, and
    the tokenizer does not truncate. Thus the model reads all 16 sections with
    their labels: rules under "avoid", "negative_prompt", and
    "explicitly_not_the_style" keep their meaning in the JSON structure.
    """
    gen = style["generation_instructions"]
    tokens = {token: key for key, token in style_placeholders(gen).items()}
    spec = fill_placeholders(style, fills, tokens)
    unfilled = sorted({m for m in PLACEHOLDER_RE.findall(json.dumps(spec, ensure_ascii=False))})
    name = style.get("style_name", "the target style")
    header = (
        f'Transform {SOURCE_TAG} completely into the art style "{name}". '
        f"The JSON below is the full specification of this style. Obey every section. "
        f"Sections near the top of generation_instructions.priority_order (if present) "
        f"win when two rules conflict. Treat \"negative_prompt\", every \"avoid\" list, and "
        f"\"explicitly_not_the_style\" as things that must not appear in the image. "
        f"Keep the identity and the pose of the subject of {SOURCE_TAG}, and apply the style "
        f"to the whole image, including the background."
    )
    if unfilled:
        header += (" These placeholders have no value, so derive them from "
                   f"{SOURCE_TAG}: " + ", ".join("{" + t + "}" for t in unfilled) + ".")
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


def build_enhancer_request(style: dict, fills: dict[str, str]) -> str:
    """The user turn of the prompt enhancer: the task and the complete style as an outline."""
    gen = style["generation_instructions"]
    tokens = {token: key for key, token in style_placeholders(gen).items()}
    spec = fill_placeholders(style, fills, tokens)
    return ("Redraw the input image in the art style that this specification defines.\n\n"
            "Style specification:\n" + "\n".join(outline(spec)))


def add_enhancer(graph: dict, request: str, system_prompt: str, seed: int, thinking: bool) -> None:
    """Insert the prompt enhancer: it reads the image and the full style, and writes the edit prompt."""
    graph[NODE_ENH_CLIP] = {"class_type": "CLIPLoader", "_meta": {"title": "Prompt enhancer model"},
                            "inputs": {"clip_name": ENHANCER_MODEL, "type": "qwen_image", "device": "default"}}
    graph[NODE_ENH_SYSTEM] = {"class_type": "PrimitiveStringMultiline", "_meta": {"title": "Enhancer system prompt"},
                              "inputs": {"value": system_prompt}}
    sampling = {f"sampling_mode.{k}": v for k, v in ENHANCER_SAMPLING.items()}
    graph[NODE_ENH_GENERATE] = {"class_type": "TextGenerate", "_meta": {"title": "Write the edit prompt"},
                                "inputs": {"clip": [NODE_ENH_CLIP, 0], "image": [NODE_IMAGE, 0],
                                           "system_prompt": [NODE_ENH_SYSTEM, 0], "prompt": request,
                                           "max_length": ENHANCER_MAX_TOKENS, "sampling_mode": "on",
                                           **sampling, "sampling_mode.seed": seed,
                                           "thinking": thinking, "use_default_template": True, "mtp": "auto"}}
    graph[NODE_ENH_SHOW] = {"class_type": "PreviewAny", "_meta": {"title": "Written prompt"},
                            "inputs": {"source": [NODE_ENH_GENERATE, 0]}}
    graph[NODE_ENCODE]["inputs"]["prompt"] = [NODE_ENH_GENERATE, 0]


def build_prompt(style: dict, fills: dict[str, str], with_avoid: bool) -> str:
    """Compose a short edit instruction for one style (compact mode).

    At cfg 1 the model ignores the negative prompt. Thus the "avoid" part goes
    into the positive prompt as plain statements when with_avoid is true.
    """
    gen = style["generation_instructions"]
    name = style.get("style_name", "the target style")
    principle = style.get("source_adaptation", {}).get("principle", "")
    parts = [
        f'Transform {SOURCE_TAG} completely into the style "{name}". '
        f"Keep the identity and the pose of the subject of {SOURCE_TAG}, "
        f"and apply the style to the whole image, including the background. {principle}".strip(),
        gen["style_prompt"].strip(),
    ]

    mandatory = style.get("style_fidelity", {}).get("mandatory_features", [])
    if mandatory:
        parts.append("Mandatory style features: " + as_sentence_list(mandatory))

    for key, token in style_placeholders(gen).items():
        value = fills.get(key) or fills.get(token.lower())
        if value:
            parts.append(f"{key.replace('_', ' ').capitalize()}: {value.strip()}")

    if with_avoid:
        avoid = [s.strip() for s in gen.get("negative_prompt", "").split(",") if s.strip()]
        if avoid:
            parts.append("The image must not contain: " + ", ".join(avoid) + ".")

    prompt = "\n\n".join(parts)
    # A placeholder token that stays in the text gives the model no information.
    return PLACEHOLDER_RE.sub("", prompt)


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

    def queue(self, graph: dict) -> str:
        reply = self.post_json("/prompt", {"prompt": graph, "client_id": self.client_id})
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+", help="image files or folders")
    ap.add_argument("--styles", nargs="+", default=[str(DEFAULT_STYLES)], help="style JSON files or folders")
    ap.add_argument("--output", default="style-transfer-output", help="folder for the results")
    ap.add_argument("--server", default=DEFAULT_SERVER, help="ComfyUI address")
    ap.add_argument("--graph", default=str(DEFAULT_GRAPH), help="API graph JSON")
    ap.add_argument("--seed", type=int, help="fixed seed; a random seed for each job if not set")
    ap.add_argument("--resolution", type=int, help="the source is resized to approximately N x N pixels with the same aspect ratio, "
                         "multiple of 32 (graph default 1024, model maximum 2048)")
    ap.add_argument("--cfg", type=float, help="guidance; more than 1 uses the negative prompt and takes approximately 2x time")
    ap.add_argument("--cache-device", choices=["auto", "gpu", "cpu", "off"], help="KV cache device")
    ap.add_argument("--cache-dtype", choices=["default", "int8", "int4"], help="KV cache precision")
    ap.add_argument("--prompt-mode", choices=["enhance", "full", "compact"], default="enhance",
                    help="enhance: the prompt enhancer model reads the image and the complete style, and writes "
                         "the edit prompt (default). full: send the complete style JSON to the edit model. "
                         "compact: send style_prompt, mandatory features, and the avoid list only")
    ap.add_argument("--enhancer-seed", type=int, default=ENHANCER_SEED,
                    help="seed of the prompt enhancer; keep it fixed to reuse a written prompt")
    ap.add_argument("--enhancer-thinking", action=argparse.BooleanOptionalAction, default=False,
                    help="let the enhancer reason before it writes (approximately 8x slower)")
    ap.add_argument("--no-avoid", action="store_true",
                    help="compact mode: do not put the negative list into the positive prompt")
    ap.add_argument("--subject", help="value for {SUBJECT}")
    ap.add_argument("--context", help="value for {CONTEXT}")
    ap.add_argument("--composition", help="value for {COMPOSITION}")
    ap.add_argument("--palette", help="value for {PALETTE}")
    ap.add_argument("--fill", action="append", default=[], metavar="KEY=VALUE",
                    help="value for a style-specific placeholder, for example surreal_event=...")
    ap.add_argument("--dry-run", action="store_true", help="print the prompts and do not contact the server")
    args = ap.parse_args()

    images = collect(args.images, IMAGE_SUFFIXES)
    styles = collect(args.styles, {".json"})
    if not images or not styles:
        sys.exit("no images or no styles found")
    fills = parse_fills(args)
    template = json.loads(Path(args.graph).read_text(encoding="utf-8"))

    jobs = []
    for style_path in styles:
        style = load_style(style_path)
        if args.prompt_mode == "enhance":
            prompt = build_enhancer_request(style, fills)
        elif args.prompt_mode == "full":
            prompt = build_full_prompt(style, fills)
        else:
            prompt = build_prompt(style, fills, with_avoid=not args.no_avoid)
        for image_path in images:
            jobs.append((image_path, style_path, style, prompt))

    if args.dry_run:
        for image_path, style_path, _, prompt in jobs:
            print(f"=== {image_path.name} x {style_path.stem}\n{prompt}\n")
        return 0

    system_prompt = ENHANCER_SYSTEM_PROMPT.read_text(encoding="utf-8") if args.prompt_mode == "enhance" else ""
    client = ComfyClient(args.server)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    uploaded: dict[Path, str] = {}
    queued = []
    for image_path, style_path, style, prompt in jobs:
        if image_path not in uploaded:
            uploaded[image_path] = client.upload_image(image_path)
        graph = json.loads(json.dumps(template))
        graph[NODE_IMAGE]["inputs"]["image"] = uploaded[image_path]
        enc = graph[NODE_ENCODE]["inputs"]
        if args.prompt_mode == "enhance":
            add_enhancer(graph, prompt, system_prompt, args.enhancer_seed, args.enhancer_thinking)
        else:
            enc["prompt"] = prompt
        if args.resolution:
            enc["resolution"] = args.resolution
        if args.cfg is not None:
            graph[NODE_GUIDER]["inputs"]["cfg"] = args.cfg
        if graph[NODE_GUIDER]["inputs"]["cfg"] > 1:
            enc["negative_prompt"] = style["generation_instructions"].get("negative_prompt", "")
        if args.cache_device:
            graph[NODE_CACHE]["inputs"]["device"] = args.cache_device
        if args.cache_dtype:
            graph[NODE_CACHE]["inputs"]["dtype"] = args.cache_dtype
        seed = args.seed if args.seed is not None else uuid.uuid4().int >> (128 - SEED_BITS)
        graph[NODE_NOISE]["inputs"]["noise_seed"] = seed
        graph[NODE_SAVE]["inputs"]["filename_prefix"] = f"style-transfer/{image_path.stem}__{style_path.stem}"
        queued.append((client.queue(graph), image_path, style_path, seed))
    print(f"queued {len(queued)} job(s): {len(images)} image(s) x {len(styles)} style(s)")

    failures = 0
    for prompt_id, image_path, style_path, seed in queued:
        entry = client.wait(prompt_id)
        if entry.get("status", {}).get("status_str") == "error":
            failures += 1
            print(f"FAILED {image_path.name} x {style_path.stem}: {json.dumps(entry['status'].get('messages'))[:500]}")
            continue
        stem = f"{image_path.stem}__{style_path.stem}"
        written = entry.get("outputs", {}).get(NODE_ENH_SHOW, {}).get("text")
        if written:
            (out_dir / f"{stem}.prompt.txt").write_text("".join(written), encoding="utf-8")
        for node_out in entry.get("outputs", {}).values():
            for image in node_out.get("images", []):
                target = out_dir / f"{stem}.png"
                target.write_bytes(client.download(image))
                secs = execution_seconds(entry)
                timing = f"{secs:.1f} s" if secs is not None else "n/a"
                print(f"done   {target.name}  seed={seed}  server time {timing}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
