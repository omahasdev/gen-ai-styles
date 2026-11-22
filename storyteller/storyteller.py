#!/usr/bin/env python3
"""Storyteller: a configurable scene concept for each image.

The storyteller sits between a style file and a prompt writer (a person, an
LLM, or a prompt enhancer). It does not change the style. For each seed it
draws a scene concept, the "story card", from the story library of the config
and from the lists of the style. A new seed gives a new concept.

The storyteller knows no style. All style-dependent material comes from the
style file through generic key paths and words of the config.

The config has three kinds of parts that combine freely:
- lines and libraries: what a card can say (concept, setting, moment, objects, ...)
- rules: named clauses, written once and inserted with {rule:NAME} into any line or
  note, each gated by style tags and modes like a library item
- recipes: named card structures; each seed draws one recipe, which sets the line
  probabilities, the budgets (the maximum number of lines of one kind, for example
  objects), and an optional approach note

See README.md in this folder for the config format.

Examples:
    python storyteller/storyteller.py "image-styles/<style>.json" --seed 3
    python storyteller/storyteller.py "image-styles/<style>.json" --seed 1 2 3 --config my-story.json
    python storyteller/storyteller.py "image-styles/<style>.json" --seed 7 --request
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).resolve().parent / "storyteller.json"
SEED_SALT = "story"  # a string seed spreads small integer seeds better
PLACEHOLDER_RE = re.compile(r"\{(lib|style|card):([a-z_]+)(?:\*(\d+)-(\d+))?\}")
RULE_RE = re.compile(r"\{rule:([a-z_]+)\}")
SPACES_RE = re.compile(r" {2,}")
SLUG_MAX_LENGTH = 40


def load_config(path: Path | str = DEFAULT_CONFIG) -> dict:
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def set_probabilities(config: dict, settings: list[str]) -> dict:
    """Apply LABEL=PROBABILITY overrides, for example ["Homage=0.8", "Look=0"]. Returns a changed copy.

    An override has priority over the probability that a recipe gives the line."""
    config = json.loads(json.dumps(config))
    by_label = {line["label"].lower(): line for line in config.get("lines", [])}
    forced = config.setdefault("forced_probabilities", {})
    for item in settings:
        label, sep, value = item.partition("=")
        line = by_label.get(label.strip().lower())
        if not sep or line is None:
            raise ValueError(f"unknown story line or bad format: {item!r} (use LABEL=PROBABILITY)")
        line["probability"] = float(value)
        forced[line["label"]] = float(value)
    return config


def set_recipe(config: dict, name: str | None) -> dict:
    """Use only the named recipe for every seed. Returns a changed copy."""
    if not name:
        return config
    config = json.loads(json.dumps(config))
    recipes = [r for r in config.get("recipes", []) if r["name"].lower() == name.lower()]
    if not recipes:
        known = ", ".join(r["name"] for r in config.get("recipes", []))
        raise ValueError(f"unknown recipe: {name!r} (known: {known})")
    config["recipes"] = recipes
    return config


# ---------------------------------------------------------------- style reading

def _negation_re(sources: dict) -> re.Pattern | None:
    words = sources.get("negation_words", [])
    return re.compile(r"\b(" + "|".join(re.escape(w) for w in words) + r")\b", re.IGNORECASE) if words else None


def _quiet_section(style: dict, section: str, sources: dict) -> bool:
    """True if the top-level section has an importance that makes it a weak story source, for example "Low"."""
    node = style.get(section)
    if not isinstance(node, dict) or not isinstance(node.get("importance"), str):
        return False
    pattern = sources.get("quiet_importance")
    return bool(pattern and re.search(pattern, node["importance"], re.IGNORECASE))


def _style_list_items(style: dict, sources: dict):
    """Yield (dotted key, items) for each string list of the style that is not a "no" rule or a quiet section."""
    exclude_keys = sources.get("exclude_keys", [])
    negation = _negation_re(sources)

    def walk(node, path):
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, path + [key])
        elif isinstance(node, list) and node and all(isinstance(x, str) for x in node):
            key = ".".join(path)
            if any(x in key for x in exclude_keys) or _quiet_section(style, path[0], sources):
                return
            items = [x.strip() for x in node if x.strip() and not (negation and negation.search(x))
                     and "{" not in x]
            if items:
                found.append((key, items))

    found: list[tuple[str, list[str]]] = []
    walk({k: v for k, v in style.items() if k not in sources.get("skip_sections", [])}, [])
    return found


def style_lists(style: dict, sources: dict) -> tuple[list[str], list[str]]:
    """Collect the string lists of the style that can seed a scene: (settings, details)."""
    settings, details = [], []
    for key, items in _style_list_items(style, sources):
        if any(k in key for k in sources.get("setting_keys", [])):
            settings.extend(items)
        elif any(k in key for k in sources.get("detail_keys", [])):
            details.extend(items)
    return settings, details


def style_material(style: dict, sources: dict) -> dict[str, list[str]]:
    """All lists that the card can draw from the style: settings, details, and the "material" groups of the config."""
    settings, details = style_lists(style, sources)
    material = {"style_settings": settings, "style_details": details}
    lists = _style_list_items(style, sources)
    for name, keys in sources.get("material", {}).items():
        material[name] = [x for key, items in lists if any(k in key for k in keys) for x in items]
    # the style's own descriptions of the kinds of objects of a place, for example its furniture
    place = []
    negation = _negation_re(sources)
    for section, keys in sources.get("place_keys", {}).items():
        node = style.get(section)
        if isinstance(node, dict):
            for k in keys:
                value = node.get(k)
                if not isinstance(value, str):
                    continue
                first = value.split(";")[0].strip().rstrip(".")  # the positive part, before a qualifying clause
                if first and not (negation and negation.search(first)):
                    place.append(first)
    material["place_keys"] = place
    return material


def style_has(style: dict, dotted_path: str, unless_value: str | None = None) -> bool:
    """True if the style defines a value at this path, for example "composition.camera".

    With unless_value (a regular expression), a value whose text matches it counts as not defined,
    for example an outer background of type "None".
    """
    node = style
    for part in dotted_path.split("."):
        if not isinstance(node, dict) or part not in node:
            return False
        node = node[part]
    if node in (None, "", [], {}):
        return False
    if unless_value:
        text = node.get("type", "") if isinstance(node, dict) else node
        if re.search(unless_value, json.dumps(text, ensure_ascii=False), re.IGNORECASE):
            return False
    return True


def style_text(style: dict, paths: list[str], exclude_keys: list[str] | None = None) -> str:
    """The lowercase text of the style values at these dotted paths, without keys that contain exclude_keys."""
    exclude_keys = exclude_keys or []

    def clean(node):
        if isinstance(node, dict):
            return {k: clean(v) for k, v in node.items() if not any(x in k for x in exclude_keys)}
        return node

    parts = []
    for path in paths:
        node = style
        for part in path.split("."):
            node = node.get(part) if isinstance(node, dict) else None
        if node not in (None, "", [], {}):
            parts.append(json.dumps(clean(node), ensure_ascii=False))
    return " ".join(parts).lower()


def style_mode(style: dict, config: dict) -> str:
    """The scene mode of the style: the first rule of config["modes"] whose pattern matches the style text."""
    exclude = config.get("style_sources", {}).get("exclude_keys", [])
    for name, mode in config.get("modes", {}).items():
        rule = mode.get("when")
        if not rule:
            return name
        text = style_text(style, rule.get("paths", []), exclude)
        if re.search(rule["pattern"], text, re.IGNORECASE):
            return name
    return "scene"


def style_tags(style: dict, config: dict) -> set[str]:
    """The tags of config["style_tags"] whose words occur in the positive style text."""
    exclude = config.get("style_sources", {}).get("exclude_keys", [])
    tags = set()
    for tag, rule in config.get("style_tags", {}).items():
        text = style_text(style, rule.get("paths", []), exclude)
        if any(w in text for w in rule.get("words", [])):
            tags.add(tag)
    return tags


def primary_tags(style: dict, config: dict) -> set[str]:
    """The tags whose words occur most often in the positive style text: the primary character of the style.

    A style can have several tags (an uncanny style that mentions "comforting familiarity" is also cozy); the
    primary tag keeps the draws close to what the style is mostly about."""
    exclude = config.get("style_sources", {}).get("exclude_keys", [])
    hits = {}
    for tag, rule in config.get("style_tags", {}).items():
        text = style_text(style, rule.get("paths", []), exclude)
        hits[tag] = sum(text.count(w) for w in rule.get("words", []))
    top = max(hits.values(), default=0)
    ratio = config.get("draw", {}).get("primary_ratio", 1.0)  # a tag almost as strong as the top one is primary too
    return {tag for tag, n in hits.items() if n and n >= top * ratio}


def place_kind(text: str, config: dict) -> str | None:
    """"indoor" or "outdoor" for a setting text, from the words of config["place_words"]; None if unclear."""
    text = text.lower()
    hits = {kind for kind, words in config.get("place_words", {}).items()
            if any(re.search(rf"\b{re.escape(w)}", text) for w in words)}
    return hits.pop() if len(hits) == 1 else None


def filter_pool(style: dict, line: dict) -> tuple[list[str] | str, list[float] | None]:
    """Apply the "style_filter" of a pool line: the style's own words limit or weight the pool.

    The first word of each pool item is its key, for example "portrait". If the style text contains a
    "source_words" phrase, the result is "source_text" (a string, not a pool). If the style names exactly one
    explicit ratio such as 3:4 (and no "flexible_words"), the result is "ratio_text" with that ratio. Keys named by the style
    limit the pool; with a "flexible_words" word, they only get "preference_boost" times the weight.
    """
    pool, weights = line["pool"], line.get("weights") or [1] * len(line["pool"])
    rule = line.get("style_filter")
    if not rule:
        return pool, weights
    text = style_text(style, rule.get("paths", []))
    if any(w in text for w in rule.get("source_words", [])):
        return rule["source_text"], None
    ratios = re.findall(r"\b\d{1,2}\s*:\s*\d{1,2}\b", text)
    if rule.get("ratio_text") and len(ratios) == 1 and not any(w in text for w in rule.get("flexible_words", [])):
        return rule["ratio_text"].format(ratio=ratios[0].replace(" ", "")), None
    named = {key for key, words in rule.get("match", {}).items() if any(w in text for w in words)}
    if not named:
        return pool, weights
    keys = [re.split(r"\W", item, maxsplit=1)[0] for item in pool]
    if any(w in text for w in rule.get("flexible_words", [])):
        boost = rule.get("preference_boost", 1)
        return pool, [w * boost if k in named else w for k, w in zip(keys, weights)]
    kept = [(item, w) for item, k, w in zip(pool, keys, weights) if k in named]
    return [item for item, _ in kept], [w for _, w in kept]


def card_value(card: str, label: str) -> str:
    """The value of one line of a story card, or an empty string."""
    for line in card.splitlines():
        if line.startswith(f"{label}: "):
            return line[len(label) + 2:]
    return ""


# ---------------------------------------------------------------- drawing

class _Context:
    """Everything that the draws of one card need."""

    def __init__(self, style: dict, config: dict, seed: int):
        self.style, self.config, self.seed = style, config, seed
        sources = config.get("style_sources", {})
        self.material = style_material(style, sources)
        self.mode = style_mode(style, config)
        self.tags = style_tags(style, config)
        self.primary = primary_tags(style, config)
        self.caps = set(config.get("modes", {}).get(self.mode, {}).get("caps", []))
        self.style_key = str(style.get("style_name", ""))
        self.boost = config.get("draw", {}).get("boost_factor", 1)
        self.drawn: dict[str, str] = {}  # the core value of each drawn line, by label
        self.recipe = self._draw_recipe()
        self.budgets = {**config.get("budgets", {}), **self.recipe.get("budgets", {})}
        self.used: dict[str, int] = {}  # the number of drawn lines of each kind

    def _draw_recipe(self) -> dict:
        recipes = [{**r, "text": r["name"]} for r in self.config.get("recipes", [])]
        picked = walk_draw(self, "recipe", recipes)
        return picked[0] if picked else {}

    def probability(self, line: dict) -> float:
        """The probability of a line: an override of the command line, else the recipe, else the line."""
        label = line["label"]
        for source in (self.config.get("forced_probabilities", {}), self.recipe.get("lines", {})):
            if label in source:
                return float(source[label])
        return line.get("probability", 1.0)

    def within_budget(self, kind: str | None) -> bool:
        return not kind or kind not in self.budgets or self.used.get(kind, 0) < self.budgets[kind]

    def spend(self, kind: str | None) -> None:
        if kind:
            self.used[kind] = self.used.get(kind, 0) + 1

    def rules(self, text: str) -> str:
        """Insert the named rules ({rule:NAME}); a rule that does not fit the style or the mode becomes empty."""
        def repl(match):
            rule = self.config.get("rules", {}).get(match.group(1))
            if rule is None:
                raise KeyError(f"unknown rule: {match.group(1)}")
            rule = _item(rule)
            fits = self.allowed(rule) and (not rule.get("modes") or self.mode in rule["modes"])
            return self.rules(rule["text"]) if fits else ""
        return SPACES_RE.sub(" ", RULE_RE.sub(repl, text)).strip()

    def allowed(self, item: dict) -> bool:
        if not set(item.get("needs", [])) <= self.caps:
            return False
        if item.get("needs_any") and not set(item["needs_any"]) & self.caps:
            return False
        if item.get("only") and not set(item["only"]) & self.tags:
            return False
        return not set(item.get("unless", [])) & self.tags

    def weight(self, item: dict) -> int:
        weight = item.get("weight", 1)
        if set(item.get("boost", [])) & self.tags:
            weight *= self.boost
        return weight


def _item(raw) -> dict:
    return {"text": raw} if isinstance(raw, str) else raw


def walk_draw(ctx: _Context, key: str, items: list, count: int = 1, step: int = 1) -> list[dict]:
    """Draw count distinct items, deterministic for the seed, with few repeats across consecutive seeds.

    The items get a fixed order for each style and key (a shuffle), and each item appears as often as its
    weight in a deck, round-robin. Seed n starts at deck position n * step, so a block of consecutive seeds
    walks through the deck and sees every allowed item before one repeats. Items that do not fit the style
    or the card (needs, only, unless) are skipped.
    """
    pool = [_item(x) for x in items]
    pool = [x for x in pool if ctx.allowed(x)]
    if not pool or count < 1:
        return []
    random.Random(f"{SEED_SALT}|{ctx.style_key}|{key}").shuffle(pool)
    rounds = max(ctx.weight(x) for x in pool)
    deck = [x for r in range(rounds) for x in pool if ctx.weight(x) > r]
    start = (ctx.seed * step) % len(deck)
    chosen, seen = [], set()
    for offset in range(len(deck)):
        item = deck[(start + offset) % len(deck)]
        if item["text"] in seen:
            continue
        seen.add(item["text"])
        chosen.append(item)
        if len(chosen) == count:
            break
    return chosen


def _join(values: list[str]) -> str:
    if len(values) <= 2:
        return " and ".join(values)
    separator = "; " if any("," in v for v in values) else ", "  # items with commas need a stronger separator
    return separator.join(values[:-1]) + separator.rstrip() + " and " + values[-1]


def _fill(ctx: _Context, label: str, template: str, rng: random.Random) -> tuple[str, str]:
    """Fill the {lib:NAME}, {style:NAME}, and {card:LABEL} placeholders of a template. Returns (text, first value)."""
    first = []

    def repl(match):
        kind, name, low, high = match.group(1), match.group(2), match.group(3), match.group(4)
        if kind == "card":
            return ctx.drawn.get(name, "")
        low, high = (int(low), int(high)) if low else (1, 1)
        count = rng.randint(low, high)
        items = (ctx.config.get("library", {}).get(name, []) if kind == "lib"
                 else ctx.material.get(name, []))
        picked = []
        for index in range(count):  # each item has its own order, so the groups vary across seeds
            remaining = [x for x in items if _item(x)["text"] not in picked]
            picked += [x["text"] for x in walk_draw(ctx, f"{label}/{name}#{index}", remaining)]
        if not picked:
            raise LookupError(name)
        first.append(picked[0])
        return _join(picked)

    text = PLACEHOLDER_RE.sub(repl, template)
    return text, (first[0] if first else text)


def _line_value(ctx: _Context, line: dict, rng: random.Random) -> tuple[str, str] | None:
    """The (value, core) of one line, or None if the line is dropped. Core is the drawn item without the frame text."""
    style, label = ctx.style, line["label"]
    override = line.get("if_style_has")
    if override and any(style_has(style, path, override.get("unless_value")) for path in override.get("paths", [])):
        text = override.get("text")
        return (text, text) if text else None
    if "variants" in line:  # weighted alternative templates, for example a blended or a dominant homage
        variants = [v for v in line["variants"] if ctx.allowed(v)]
        if not variants:
            return None
        variant = rng.choices(variants, weights=[v.get("weight", 1) for v in variants])[0]
        try:
            return _fill(ctx, label, variant["template"], rng)
        except LookupError:  # the style has no material for a placeholder
            return _fill(ctx, label, line["fallback_template"], rng) if line.get("fallback_template") else None
    if "pool" in line:
        pool, weights = filter_pool(style, line)  # optional "weights" make some items more frequent
        if isinstance(pool, str):
            return pool, pool  # the style decides this line, for example "the aspect ratio of the source image"
        if not pool:
            return None
        deck = [{"text": p, "weight": int(w)} for p, w in zip(pool, weights)]
        value = walk_draw(ctx, label, deck)[0]["text"]
        return value, value
    if "source" in line or "library" in line:
        items = ctx.material.get(line["source"], []) if "source" in line else []
        from_style = bool(items)
        if not items and "library" in line:
            items = ctx.config.get("library", {}).get(line["library"], [])
        key = label
        share = line.get("primary_share", 0)
        primary = [x for x in items if set(_item(x).get("only", [])) & ctx.primary]
        if primary and rng.random() < share:  # mostly the primary character of the style, sometimes its other tags
            items, key = primary, f"{label}/primary"
        picked = walk_draw(ctx, key, items)
        if not picked:
            fallback = line.get("fallback")
            return (fallback, fallback) if fallback else None
        core = picked[0]["text"]
        prefix = line.get("item_prefix", "") if from_style or "library" in line else ""
        return prefix + core, core
    if "template" in line:
        try:
            return _fill(ctx, label, line["template"], rng)
        except LookupError:  # the style has no material for a placeholder
            return _fill(ctx, label, line["fallback_template"], rng) if line.get("fallback_template") else None
    text = line.get("text")
    return (text, text) if text else None


def _mode_line(line: dict, mode: str) -> dict | None:
    """The line for this scene mode: dropped if the mode is not in "modes", changed by "mode_overrides"."""
    if line.get("modes") and mode not in line["modes"]:
        return None
    changes = line.get("mode_overrides", {}).get(mode)
    if not changes:
        return line
    line = {k: v for k, v in line.items() if k not in ("pool", "source", "library", "template", "text", "variants")
            or k in changes}
    line.update(changes)
    return line


def draw_card(style: dict, config: dict, seed: int) -> list[tuple[str, str, str]]:
    """Draw the lines of one card: a list of (label, full value, core value)."""
    ctx = _Context(style, config, seed)
    result = []
    for line in config.get("lines", []):
        line = _mode_line(line, ctx.mode)
        if line is None:
            continue
        if line.get("only") and not set(line["only"]) & ctx.tags:
            continue  # for example the surprise line only for a style with the uncanny tag
        if line.get("requires") and not ctx.material.get(line["requires"]):
            continue  # for example a shot type only for a style that lists its shot types
        if line.get("unless_material") and ctx.material.get(line["unless_material"]):
            continue
        if line.get("unless_line") and line["unless_line"] in {label for label, _, _ in result}:
            continue  # for example no generic camera line when the card has a shot type of the style
        rng = random.Random(f"{SEED_SALT}-{seed}-{line['label']}")
        if rng.random() >= ctx.probability(line):
            continue
        if line["label"] not in config.get("forced_probabilities", {}) and not ctx.within_budget(line.get("kind")):
            continue  # the recipe allows no more lines of this kind, for example objects
        if "slots" in line:  # a group of open slots: draw "count" of them, each becomes its own line
            remaining = list(line["slots"])
            for index in range(line.get("count", 1)):  # each slot has its own order, so the pairs vary
                picked = walk_draw(ctx, f"{line['label']}#{index}", remaining)
                if not picked:
                    break
                slot = picked[0]
                remaining = [x for x in remaining if x["label"] != slot["label"]]
                if not ctx.within_budget(slot.get("kind")):
                    continue
                ctx.spend(slot.get("kind"))
                result.append((slot["label"], ctx.rules(f"{line.get('prefix', '')}{slot['text']}"), slot["text"]))
            continue
        drawn = _line_value(ctx, line, rng)
        if not drawn:
            continue
        value, core = drawn
        ctx.drawn[line["label"].lower().replace(" ", "_")] = core
        if line.get("adds_kind"):  # the setting decides indoor or outdoor for the lines after it
            library = map(_item, config.get("library", {}).get(line.get("library", ""), []))
            kind = next((x.get("kind") for x in library if x["text"] == core), None) or place_kind(core, config)
            if kind:
                ctx.caps.add(kind)
        ctx.spend(line.get("kind"))
        result.append((line["label"], ctx.rules(f"{line.get('prefix', '')}{value}{line.get('suffix', '')}"), core))
    note = ctx.recipe.get("note")
    if note:  # the approach of the recipe, as the last line of the card
        result.append((config.get("recipe_label", "Approach"), ctx.rules(note), ctx.recipe["name"]))
    return result


def story_card(style: dict, config: dict, seed: int) -> str:
    """Draw one story card.

    A line with "if_style_has" gives the style control of that aspect: when the style defines one
    of the listed paths, the line uses the replacement "text", or the line is dropped if there is no text.
    """
    return "\n".join(f"{label}: {value}" for label, value, _ in draw_card(style, config, seed))


def story_slug(style: dict, config: dict, seed: int) -> str:
    """A short file-name slug for the story of this seed: the core of the first line marked "slug"."""
    slug_labels = [x["label"] for x in config.get("lines", []) if x.get("slug")]
    lines = {label: core for label, _, core in draw_card(style, config, seed)}
    for label in slug_labels:
        if lines.get(label):
            words = re.sub(r"[^a-z0-9]+", "-", lines[label].lower()).strip("-")
            return words[:SLUG_MAX_LENGTH].rstrip("-")
    return ""


def build_request(style: dict, config: dict, seed: int, spec_text: str) -> str:
    """A complete request for a prompt writer: intro, precedence rule, story card, and the style text."""
    card = story_card(style, config, seed)
    slot_labels = {s["label"] for x in config.get("lines", []) for s in x.get("slots", [])}
    has_slot = any(card_value(card, label) for label in slot_labels)
    notes = (config.get("request_intro"), config.get("precedence_note"),
             config.get("open_slot_note") if has_slot else None)
    ctx = _Context(style, config, seed)
    intro = " ".join(ctx.rules(x) for x in notes if x)
    return (f"{intro}\n\n"
            f"{config['card_heading']}\n{card}\n\n"
            f"{config['spec_heading']}\n{spec_text}")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):  # style texts hold non-ASCII letters; a redirected Windows console is not UTF-8
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("style", help="style JSON file")
    ap.add_argument("--seed", type=int, nargs="+", default=[0], help="one card for each seed")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG), help="storyteller config")
    ap.add_argument("--request", action="store_true",
                    help="print the complete request (intro, story card, full style JSON) to paste into "
                         "ChatGPT or another image tool, together with the source image")
    ap.add_argument("--set", nargs="+", default=[], metavar="LABEL=PROBABILITY",
                    help="change the probability of story lines, for example Homage=0.8 Mood=0")
    ap.add_argument("--recipe", help='use only this recipe of the config for every seed, for example "quiet moment"')
    args = ap.parse_args()

    try:
        config = set_recipe(set_probabilities(load_config(args.config), args.set), args.recipe)
    except ValueError as err:
        ap.error(str(err))
    with open(args.style, encoding="utf-8") as f:
        style = json.load(f)
    spec_text = json.dumps(style, ensure_ascii=False, indent=2)
    for seed in args.seed:
        body = build_request(style, config, seed, spec_text) if args.request else story_card(style, config, seed)
        header = f"=== seed {seed}\n" if len(args.seed) > 1 or not args.request else ""
        print(f"{header}{body}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
