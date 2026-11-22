"""The schema of a style file: the section names, the "no"-rule keys, and a validator (standard library only).

Usage: python style_schema.py <style.json or folder> [...]
Prints the problems (the file breaks the schema) and the warnings (the file is valid, but a tool gets less
from it) of each file. The exit code is the number of files with problems.
"""
import json
import sys
from pathlib import Path

# The 16 top-level keys of a style file, in this order (AGENTS.md of the repository).
TOP_KEYS = ("style_name", "style_definition", "rendering", "composition", "framing", "linework", "color_system",
            "shading_and_lighting", "subject_design", "botanical_design", "background_design",
            "ornamental_language", "texture", "style_fidelity", "source_adaptation", "generation_instructions")

# A string list under one of these keys is a "no" rule of the style, in every section.
NEGATIVE_LIST_KEYS = ("avoid", "explicitly_not_the_style", "do_not_reproduce", "do_not")
# A comma-separated string of "no" rules.
NEGATIVE_TEXT_KEYS = ("negative_prompt",)
NEGATIVE_KEYS = NEGATIVE_LIST_KEYS + NEGATIVE_TEXT_KEYS

FIDELITY_KEYS = ("mandatory_features", "optional_features", "explicitly_not_the_style")
SOURCE_ADAPTATION_KEYS = ("principle", "subject_independence", "detail_budget", "complex_source_handling",
                          "complexity_rule")
GENERATION_KEYS = ("style_prompt", "negative_prompt")  # the placeholder keys were removed: no style text used them
GENERATION_OPTIONAL_KEYS = ("priority_order", "reference_image_usage")
STRENGTH_RANGE = (0, 1)  # a number in a style file is a strength


def is_negative_key(key: str) -> bool:
    return key in NEGATIVE_KEYS


def validate(style) -> tuple[list[str], list[str]]:
    """(problems, warnings) of one parsed style file."""
    problems, warnings = [], []
    if not isinstance(style, dict):
        return ["the file is not a JSON object"], []
    keys = tuple(style)
    if keys != TOP_KEYS:
        missing = [k for k in TOP_KEYS if k not in style]
        extra = [k for k in keys if k not in TOP_KEYS]
        problems.append(f"top-level keys differ: missing {missing}, extra {extra}"
                        if missing or extra else "top-level keys are in the wrong order")
    if not isinstance(style.get("style_name"), str):
        problems.append("style_name is not a string")
    for key in TOP_KEYS[1:]:
        if key in style and not isinstance(style[key], dict):
            problems.append(f"{key} is not an object")

    def walk(node, path):
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, f"{path}.{k}" if path else k)
        elif isinstance(node, list):
            if path.rsplit(".", 1)[-1] in NEGATIVE_LIST_KEYS and not all(isinstance(x, str) for x in node):
                problems.append(f"{path}: a 'no' list must hold strings only")
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")
        elif isinstance(node, (int, float)) and not isinstance(node, bool):
            if not STRENGTH_RANGE[0] <= node <= STRENGTH_RANGE[1]:
                problems.append(f"{path}: number {node} is not in {STRENGTH_RANGE[0]}-{STRENGTH_RANGE[1]}")
    walk(style, "")

    fidelity = style.get("style_fidelity", {})
    for key in FIDELITY_KEYS:
        if not (isinstance(fidelity, dict) and isinstance(fidelity.get(key), list)):
            problems.append(f"style_fidelity.{key} is not a string list")
    adaptation = style.get("source_adaptation", {})
    for key in SOURCE_ADAPTATION_KEYS:
        if not (isinstance(adaptation, dict) and key in adaptation):
            problems.append(f"source_adaptation.{key} is missing")
    if isinstance(adaptation, dict) and not isinstance(adaptation.get("detail_budget", {}), dict):
        problems.append("source_adaptation.detail_budget is not an object")
    generation = style.get("generation_instructions", {})
    if isinstance(generation, dict):
        gkeys = tuple(generation)
        if gkeys[:len(GENERATION_KEYS)] != GENERATION_KEYS:
            problems.append(f"generation_instructions must start with {list(GENERATION_KEYS)}")
        for key in gkeys[len(GENERATION_KEYS):]:
            if key not in GENERATION_OPTIONAL_KEYS and not key.endswith("_placeholder"):
                problems.append(f"generation_instructions.{key}: not an allowed key")
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "storyteller"))
        import storyteller
        settings, _ = storyteller.style_lists(style, storyteller.load_config().get("style_sources", {}))
        if not settings:
            warnings.append("the storyteller finds no setting list: story cards use the setting fallback")
    except (ImportError, OSError, ValueError) as err:
        warnings.append(f"storyteller check skipped: {err}")
    return problems, warnings


def main() -> int:
    paths = []
    for arg in sys.argv[1:]:
        p = Path(arg)
        paths += sorted(p.glob("*.json")) if p.is_dir() else [p]
    bad = 0
    for path in paths:
        try:
            style = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as err:
            problems, warnings = [f"cannot read: {err}"], []
        else:
            problems, warnings = validate(style)
        bad += bool(problems)
        print(f"{'FAIL' if problems else 'ok  '} {path.name}")
        for line in problems:
            print(f"     problem: {line}")
        for line in warnings:
            print(f"     warning: {line}")
    return bad


if __name__ == "__main__":
    raise SystemExit(main())
