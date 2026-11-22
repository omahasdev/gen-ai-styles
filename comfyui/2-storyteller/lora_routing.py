"""LoRA routing: the backstage chooses the LoRAs of a picture from the words of its style.

A style file never names a LoRA. The local registry (loras.json) lists the installed LoRAs. Each entry says
for which diffusion model it is, and which words of a style call for it ("when": dotted key paths of the
style and a regular expression, as in the storyteller modes). For each style, the routing selects the entries
whose model matches the diffusion model of the graph and whose pattern matches the style text. The graph gets
one LoRA loader (model only) for each selected entry, between the diffusion model and its first user, and the
prompt gets the trigger words of the selected entries in front.

Only generic LoRAs route: an entry must have "scope": "generic", which means that the LoRA gives a broad kind of
look (for example photo to TV anime redraw), never the look of one concrete work, series, or style. An entry
without it is never selected. A LoRA trained on one work is not a part of this mechanism.

Registry format (all keys of an entry except "file" and "when" are optional):
    {"loras": [{"file": "<name in models/loras>", "model": "<part of the diffusion model file name>", "scope": "generic",
                "strength": 0.8, "trigger": "<trigger words>", "note": "<what it does>",
                "when": {"paths": ["style_definition.primary_identity", "rendering.medium"],
                         "pattern": "<regular expression, case-insensitive>"}}]}
"""
import itertools
import json
import re
from pathlib import Path

DEFAULT_STRENGTH = 1.0
LORA_NODE = "LoraLoaderModelOnly"
TRIGGER_NODE = "StringConcatenate"
# node ids of the added loaders: the first free ids from 900. style_transfer.py adds node blocks at 100, 200 (identity
# notes), 300 (prompt hygiene) and 400 (writer notes), so the loaders must not start in one of these blocks.
FIRST_LORA_NODE_ID = 900
TRIGGER_NODE_ID = "199"
GENERIC = "generic"  # the only scope that routes


def load_registry(path: Path | None) -> list[dict]:
    """The entries of the registry, or none when the file is absent."""
    if path is None or not Path(path).exists():
        return []
    return json.loads(Path(path).read_text(encoding="utf-8")).get("loras", [])


def _value(style: dict, dotted: str):
    node = style
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def style_text(style: dict, paths: list[str]) -> str:
    """The text of the style at these key paths, as one string."""
    parts = []
    for p in paths:
        v = _value(style, p)
        if v is not None:
            parts.append(v if isinstance(v, str) else json.dumps(v, ensure_ascii=False))
    return "\n".join(parts)


def select(style: dict, registry: list[dict], model_name: str) -> list[dict]:
    """The registry entries for this style and this diffusion model, in registry order."""
    chosen = []
    for entry in registry:
        if entry.get("scope") != GENERIC:
            continue  # a LoRA of one concrete work or style never routes automatically
        if entry.get("autoroute") is False:
            continue  # kept in the registry for --extra-lora, but a check showed that it makes the result worse
        if entry.get("model") and entry["model"].lower() not in model_name.lower():
            continue
        when = entry.get("when", {})
        if re.search(when.get("pattern", r"(?!)"), style_text(style, when.get("paths", [])), re.IGNORECASE | re.DOTALL):
            chosen.append(entry)
    return chosen


def apply(graph: dict, entries: list[dict], model_node: str, encode_node: str) -> None:
    """Insert one loader for each entry after the model node, and put the trigger words in front of the prompt."""
    if not entries:
        return
    users = [(nid, key) for nid, node in graph.items() for key, val in node.get("inputs", {}).items()
             if val == [model_node, 0]]
    previous = [model_node, 0]
    free_ids = (str(i) for i in itertools.count(FIRST_LORA_NODE_ID) if str(i) not in graph)
    for entry in entries:
        nid = next(free_ids)
        graph[nid] = {"class_type": LORA_NODE, "_meta": {"title": f"LoRA {entry['file']}"},
                      "inputs": {"model": previous, "lora_name": entry["file"],
                                 "strength_model": entry.get("strength", DEFAULT_STRENGTH)}}
        previous = [nid, 0]
    for nid, key in users:
        graph[nid]["inputs"][key] = previous
    triggers = " ".join(e["trigger"] for e in entries if e.get("trigger"))
    if not triggers:
        return
    enc = graph[encode_node]["inputs"]
    if isinstance(enc.get("prompt"), list):  # the prompt comes from the writer node of the same graph
        graph[TRIGGER_NODE_ID] = {"class_type": TRIGGER_NODE, "_meta": {"title": "LoRA trigger words"},
                                  "inputs": {"string_a": triggers + ". ", "string_b": enc["prompt"], "delimiter": ""}}
        enc["prompt"] = [TRIGGER_NODE_ID, 0]
    else:
        enc["prompt"] = f"{triggers}. {enc.get('prompt', '')}"


FIRST_REFERENCE_NODE_ID = 220  # node ids of the added reference image loaders
REFERENCE_INPUT = "images.image_{n}"  # the reference inputs of the Qwen Image 2.1 encoder
REFERENCE_TAG = "<image{n}>"
DEFAULT_REFERENCE_NOTE = ("{tags} show only the drawing style: take from them the line, the color, the eyes and faces, "
                          "and the shading, never their characters, objects, places, or composition.")


def load_references(path: Path | None) -> list[dict]:
    """The style reference entries of the registry: local images that show the look of a style kind."""
    if path is None or not Path(path).exists():
        return []
    return json.loads(Path(path).read_text(encoding="utf-8")).get("references", [])


def apply_references(graph: dict, entries: list[dict], encode_node: str, upload) -> list[str]:
    """Add the images of the entries as more reference images of the encoder, after the ones it has, and tell the
    prompt which tags show the style. upload(path) returns the server name of a local image. Returns the paths."""
    enc = graph[encode_node]["inputs"]
    used = [int(m.group(1)) for k in enc for m in [re.fullmatch(r"images\.image_(\d+)", k)] if m]
    n = max(used, default=0)
    paths, tags = [], []
    for entry in entries:
        for path in entry.get("images", []):
            n += 1
            nid = next(str(i) for i in itertools.count(FIRST_REFERENCE_NODE_ID) if str(i) not in graph)
            graph[nid] = {"class_type": "LoadImage", "_meta": {"title": f"Style reference {Path(path).name}"},
                          "inputs": {"image": upload(Path(path))}}
            enc[REFERENCE_INPUT.format(n=n)] = [nid, 0]
            paths.append(str(path))
            tags.append(REFERENCE_TAG.format(n=n))
    if not tags:
        return paths
    note = (entries[0].get("note") or DEFAULT_REFERENCE_NOTE).format(tags=" and ".join(tags))
    if isinstance(enc.get("prompt"), list):
        # A second call (the identity lines beside the style references) gets its own note node.
        nid = next(f"{TRIGGER_NODE_ID}r{i or ''}" for i in itertools.count() if f"{TRIGGER_NODE_ID}r{i or ''}" not in graph)
        graph[nid] = {"class_type": TRIGGER_NODE, "_meta": {"title": "Reference note"},
                      "inputs": {"string_a": enc["prompt"], "string_b": " " + note, "delimiter": ""}}
        enc["prompt"] = [nid, 0]
    else:
        enc["prompt"] = f"{enc.get('prompt', '')} {note}"
    return paths


def parse_extra(spec: str) -> dict:
    """An entry for a LoRA chosen by hand: "file[:strength[:trigger]]". It applies to every job of the run."""
    file, _, rest = spec.partition(":")
    strength, _, trigger = rest.partition(":")
    entry = {"file": file, "strength": float(strength) if strength else DEFAULT_STRENGTH, "manual": True}
    if trigger:
        entry["trigger"] = trigger
    return entry


def describe(entries: list[dict]) -> list[dict]:
    """The manifest record of the selected entries."""
    return [{"file": e["file"], "strength": e.get("strength", DEFAULT_STRENGTH), "trigger": e.get("trigger"),
             **({"manual": True} if e.get("manual") else {})} for e in entries]
