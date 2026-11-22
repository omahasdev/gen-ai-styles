"""Hygiene of a written prompt (standard library only): cleanup, leak check, audit, and the prompt key.

- clean(): with thinking, the LLM sometimes prints its reasoning and "</think>" before the answer; only the
  text after the last "</think>" is kept. The LLM sometimes wraps the prompt in JSON ({"prompt": "..."}) or in a code fence. The image
  model must get the plain text. JSON is unwrapped with json.loads, so escapes such as \\" and \\n are decoded.
- clean() also keeps only the last draft when the writer printed several drafts one after another (the opening
  sentence comes again): the drafts make a prompt of thousands of words that the image model cannot use.
- leak_reasons(): without thinking, the writer sometimes puts its own reasoning into the prompt (planning
  phrases, quoted checklist words, a draft of many lines). A leak makes the image more realistic.
- audit(): numbers that the prompt store keeps for each prompt.
- prompt_key(): a stable key of everything that decides the written prompt, for the prompt store.
"""
from __future__ import annotations

import hashlib
import json
import re

CLEANUP_VERSION = 16
THINK_END = "</think>"

FENCE_RE = re.compile(r"^\s*```[a-zA-Z]*\s*\n(.*?)\n\s*```\s*$", re.DOTALL)
PROMPT_FIELDS = ("prompt", "instruction", "edit_instruction", "text", "positive_prompt", "description")

JSON_PREFIX_RE = re.compile(r'^\s*[\[{]\s*(?:"[a-z_]+"\s*:\s*)?"')
MIN_WORDS = 150  # a shorter cleaned prompt was cut or failed; the system prompt asks for 800 to 1200 words
MAX_LINES = 6  # the instruction is one paragraph; more non-empty lines is a draft
PLANNING_RE = re.compile(r"\b(let me|word count|i need to|i will now|i'll (now|add|expand|write)|rewrite (it )?as|"
                         r"need \d+\s*-\s*\d+ words|final check|now expand|draft:)", re.IGNORECASE)
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
REPEAT_MIN_WORDS = 8  # a repeated sentence of at least this many words is a second draft, not a style phrase
CHECKLIST_RE = re.compile(r'"(no|not|never|without|avoid|free of|instead of)"', re.IGNORECASE)
# card68: the writer copied the style palette ('ochre, sky blue and muted foliage green') into a prompt of a closed
# wooden room, and the image model drew a window with a blue sky in 2 of 3 seeds. A color name that names an object
# invites that object, so the color keeps only its hue and lightness.
# The rules are pattern strings (case-insensitive), because the server graph runs them too (RegexReplace nodes).
OBJECT_COLOR_RULES = (
    (r"\bsky[- ]blue(?=s?\b)", "light blue"),   # card70: a plural ('foliage greens') keeps its s
    (r"\b(?:ocean|sea)[- ]blue(?=s?\b)", "deep blue"),
    (r"\bsea[- ]green(?=s?\b)", "blue-green"),
    (r"\b(?:foliage|leaf|grass|forest|moss)[- ]green(?=s?\b)", "green"),
    (r"\bbrick[- ]red(?=s?\b)", "red-brown"),
)
# card71/72: the notes named a fine knit top 'orange quilted', and the image model drew a grid of lines over the
# sleeves in 6 of 6 seeds, also when the writer was asked for a flat fill. A fabric texture word goes, and the garment
# keeps its color and its noun.
FABRIC_TEXTURE_RULES = (
    (r"\b(?:quilted|ribbed|(?:rib|cable|waffle)[- ]knit|knitted|knit)\s+(?=[a-z])", ""),
)
# card73: the Colors line says that a light pool never lies on a cool wall, but the writer wrote 'a soft warm light
# pool grazing the upper part of the wall', and the image model painted a cool gray wall beige. The light on a wall
# keeps no hue word.
WALL_LIGHT_RULES = (
    # card74: 'a warm key light from the upper front so both faces and the wall share one light logic'
    # card75: 'a soft warm light pool on the plaster behind the woman'
    (r"\b(?:warm|golden|amber) (?=(?:[a-z]+ ){0,2}(?:light|glow|sun)[^.;]{0,60}\b(?:walls?|plaster|backdrop)\b)", ""),
)
# card75: the notes called a black coat 'dark navy', the writer copied it, and the image model drew a mid-blue coat
# in 3 of 3 seeds. A dark navy garment of a photo is near black. card76: "near-black navy" still gave a mid blue.
# card78-card80: the notes said 'a dark leather coat' with no hue word, the writer filled in 'dark warm brown', and
# the image model drew a brown coat for a charcoal one in 2 of 3 seeds. A bare 'dark' garment is a near-black one.
HUE_WORD = (r"(?:black|white|gr[ae]y|charcoal|brown|red|maroon|burgundy|crimson|orange|yellow|gold|olive|green|teal|"
            r"blue|navy|purple|violet|plum|pink|tan|beige|cream|ochre|rust|silver|warm|cool|near-black)")
GARMENT_WORD = (r"(?:coat|overcoat|jacket|cloak|cape|hat|cap|hood|trousers|pants|jeans|boots|shirt|blouse|sweater|"
                r"dress|skirt|vest|waistcoat|corset|gloves|scarf|robe|suit|uniform)")
DARK_HUE_RULES = (
    (r"\b(?:dark|deep) navy(?:[- ]blue)?\b", "near-black"),
    (rf"\bdark (?!{HUE_WORD}\b)(?=(?:[a-z-]+ ){{0,2}}{GARMENT_WORD}s?\b)", "near-black "),
)
# card82: the writer wrote 'one clawed hand planted on the bench' for a creature whose notes said 'gloved hand', and
# the image model drew long dark claws in 2 of 3 seeds. A claw word on a hand goes; the hand of the input stays.
CLAW_RULES = (
    (r"\b(?:clawed|claw-tipped|taloned)\s+(?=(?:[a-z-]+ )?(?:hands?|fingers?|paws?)\b)", ""),
)
# card75: the writer prompt is not cleaned of a figure on a prop, and the writer wrote 'the small rough cloth-bundle
# <kind> figure ... strapped sideways to the bar', '<kind>-like cloth bundle', and 'its head lolling sideways with wide
# blank eyes'. The image model drew a cute doll with a face in 2 of 3 seeds. A bundle on a prop has no face.
FIXED_BUNDLE_RULES = (
    (r"\b(?:small|tiny|little) (?:[a-z-]+ ){0,3}figure\b(?=[^.;]{0,120}\b(?:strapped|tied|taped|fixed|mounted|"
     r"attached|lashed|bound|at the (?:left|right) (?:end|grip))\b)", "small rough cloth bundle"),
    # card81: 'the taped doll figure with blond hair and a red hood strapped to the left grip' gave a cute doll with
    # a face in 3 of 3 seeds.
    (r"\b(?:doll|figurine)(?: head| figure)?\b(?=[^.;]{0,120}\b(?:strapped|tied|taped|fixed|mounted|attached|lashed|"
     r"bound|(?:at|on) the (?:left|right) (?:end|grip))\b)", "rough cloth bundle"),
    (r"(rough cloth bundle) with ([a-z -]+?) hair\b", r"\1 with a \2 tuft"),
    (r"\b[a-z]+-like (?=(?:[a-z]+ ){0,2}bundle\b)", ""),
    (r"(\bbundle\b[^.;]*?),\s*(?:with |and )?its (?:[a-z-]+ ){0,2}(?:head|face|eyes|mouth)\b[^,.;]*", r"\1"),
)
# card77: the writer copied the whole style palette as one list ('a palette of warm browns, cream, terracotta, ochre,
# light blue and muted green'), and the image model painted a cool gray wall beige (hunter 603) and added a blue sky
# with hills to a closed woodland (unicorn 603). A list of colors for the whole picture has no element to keep; the
# colors that the writer gives to each element stay.
PALETTE_WORD = r"(?!(?:and|or|across|the|a|an|in|on|under|with|for|to|over|at|from|of|that|so|as)\b)[a-z-]+"
PALETTE_ITEM = rf"(?:{PALETTE_WORD} ){{0,2}}{PALETTE_WORD}"
PALETTE_LIST_RULES = (
    # card85: 'a medium saturation broadcast range of warm browns, cream, terracotta, ochre, light blue, and muted
    # green' kept the list, and the image model drew an ochre creature face (creature 602).
    (rf"\b(palette|range|scheme)\s+of\s+{PALETTE_ITEM}(?:,\s*{PALETTE_ITEM})+(?:,?\s+(?:and|or)\s+{PALETTE_ITEM})?", r"\1"),
)
# card74 ('a pale blue distance across the top') and card80 ('a pale blue distance plane'): the writer gave a closed
# woodland a blue far plane, and the image model drew a strip of blue sky. A far plane keeps no blue hue word; the
# image model takes the color of the distance from the input.
BLUE_DISTANCE_RULES = (
    (r"\b(?:(?:pale|light|soft|cool|hazy|misty)[- ])?blue(?:[- ]gr[ae]y)?\s+(?=(?:[a-z-]+ ){0,3}"
     r"(?:distance|horizon|haze|far plane|ground|floor)\b)", ""),  # card83: 'a pale blue ground plane' became water
)
# card84: the notes called the pale olive-beige skin of a creature 'a wrinkled tan-yellow face', the writer made it
# 'tan skin as flat warm ochre-tan fill', and the image model painted a saturated yellow face in 3 of 3 seeds. A hue
# word that drifts to yellow goes from skin; the image model takes the skin color from the input.
SKIN_DRIFT_WORD = r"(?:warm|cool|yellow(?:ish)?|ochre|gold(?:en)?|orange|olive|green(?:ish)?|tan|amber)"
SKIN_WORD = r"(?:skin|face|complexion)"
# card85: the writer also wrote 'face simplified into broad flat tan color islands', 'skin as a flat tan color mass',
# and 'skin stays flat warm tones'. A flat fill of skin may come some words after the skin word.
SKIN_FILL_SPAN = 40
SKIN_FILL_WORD = r"(?:fill|colou?r|tones?|mass|islands?)"
SKIN_HUE_RULES = (
    (rf"\b(?:{SKIN_DRIFT_WORD}[- ])+(?={SKIN_WORD}\b)", ""),
    (rf"(\b{SKIN_WORD}\b[^.;,]{{0,{SKIN_FILL_SPAN}}}?\bflat )(?:{SKIN_DRIFT_WORD}[- ])+(?={SKIN_FILL_WORD}\b)",
     r"\1"),
)
# card72 and card82 ('pale sage-green' for a gray-gold dress in the notes) and card85 ('muted gray-green with gold
# motifs for the pinafore' in the writer text, against 'silver-gold' in the notes): a muted mixed green is the hue that
# the models invent for a gray fabric, and the image model drew the dress green each time. A muted mixed green is gray.
MIXED_GREEN_RULES = (
    (r"\b(?:(?:muted|pale|soft|dusty)\s+)?(?:gr[ae]y-green|green-gr[ae]y|sage-green)\b", "gray"),
)
WORD_RULES = (OBJECT_COLOR_RULES + FABRIC_TEXTURE_RULES + WALL_LIGHT_RULES + DARK_HUE_RULES + FIXED_BUNDLE_RULES
              + PALETTE_LIST_RULES + BLUE_DISTANCE_RULES + CLAW_RULES + SKIN_HUE_RULES
              + MIXED_GREEN_RULES)
WORD_PATTERNS = tuple((re.compile(pattern, re.IGNORECASE), word) for pattern, word in WORD_RULES)


def clean(raw: str) -> str:
    """The plain prompt text: without a code fence, a JSON wrapper, and outer white space."""
    text = raw.strip()
    if THINK_END in text:
        # The writer sometimes prints its reasoning, the closing tag, and then the answer: keep the answer only.
        text = text.rpartition(THINK_END)[2].strip()
    fence = FENCE_RE.match(text)
    if fence:
        text = fence.group(1).strip()
    if text[:1] in "{[":
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            # A wrapper cut at the token limit: remove its opening {"key": " so that only prose remains.
            return JSON_PREFIX_RE.sub("", text).strip()
        text = unwrap(value) or text
    text = last_draft(text.strip())
    for pattern, word in WORD_PATTERNS:
        text = pattern.sub(word, text)
    return text


def last_draft(text: str) -> str:
    """The last draft, when the text repeats its opening sentence (several drafts one after another)."""
    sentences = SENTENCE_RE.split(text)
    first = sentences[0].strip() if sentences else ""
    if len(first.split()) < REPEAT_MIN_WORDS:
        return text
    start = text.rfind(first)
    return text[start:].strip() if start > 0 else text


def repeated_sentences(text: str) -> list[str]:
    """The long sentences that occur more than one time."""
    seen, repeated = set(), []
    for sentence in (s.strip() for s in SENTENCE_RE.split(text)):
        if len(sentence.split()) < REPEAT_MIN_WORDS:
            continue
        if sentence in seen and sentence not in repeated:
            repeated.append(sentence)
        seen.add(sentence)
    return repeated


def unwrap(value) -> str:
    """The prompt string inside a parsed JSON value: a known field, or else the longest string."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return max((unwrap(v) for v in value), key=len, default="")
    if isinstance(value, dict):
        for key in PROMPT_FIELDS:
            if isinstance(value.get(key), str):
                return value[key]
        return max((unwrap(v) for v in value.values()), key=len, default="")
    return ""


def leak_reasons(text: str) -> list[str]:
    """Why the text looks like the writer's reasoning and not a finished prompt (empty: no leak)."""
    reasons = []
    planning = PLANNING_RE.search(text)
    if planning:
        reasons.append(f"planning phrase: {planning.group(0)!r}")
    checklist = CHECKLIST_RE.search(text)
    if checklist:
        reasons.append(f"checklist word: {checklist.group(0)}")
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) > MAX_LINES:
        reasons.append(f"{len(lines)} lines")
    repeated = repeated_sentences(text)
    if repeated:
        reasons.append(f"{len(repeated)} repeated sentence(s), first: {repeated[0][:60]!r}")
    reasons += degenerate_reasons(text)
    return reasons


DEGENERATE_MIN_WORDS = 50       # shorter texts are too short to judge their word statistics
DEGENERATE_UNIQUE_RATIO = 0.15  # a real prompt of 500 words has approximately 0.4-0.6 distinct words
DEGENERATE_TOP_SHARE = 0.2      # the most frequent word of a real prompt ("the") is approximately 0.08
DEGENERATE_QUOTE_RATIO = 0.2    # double quotes per word; quoted checklist words of a leak reach approximately 0.09


def degenerate_reasons(text: str) -> list[str]:
    """Signs of a sampling collapse: a text of repeated function words and stray quotes instead of prose."""
    words = re.findall(r"[a-z']+", text.lower())
    if len(words) < DEGENERATE_MIN_WORDS:
        return []
    reasons = []
    unique = len(set(words)) / len(words)
    if unique < DEGENERATE_UNIQUE_RATIO:
        reasons.append(f"degenerate text: {unique:.2f} distinct words per word")
    top = max(words.count(w) for w in set(words)) / len(words)
    if top > DEGENERATE_TOP_SHARE:
        reasons.append(f"degenerate text: one word is {top:.0%} of the text")
    quotes = text.count('"') / len(words)
    if quotes > DEGENERATE_QUOTE_RATIO:
        reasons.append(f"degenerate text: {quotes:.2f} double quotes per word")
    return reasons


def audit(text: str) -> dict:
    """Numbers of the prompt for the store: words, non-empty lines, double quotes."""
    return {"words": len(text.split()), "lines": sum(1 for line in text.splitlines() if line.strip()),
            "double_quotes": text.count('"')}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def prompt_key(*, writer_model: str, system_prompt: str, request: str, source_sha256: str, seed: int,
               thinking: bool, max_tokens: int, sampling: dict, negative: str = "") -> str:
    """SHA-256 of everything that decides the written prompt (and the written negative, if any)."""
    parts = {"writer_model": writer_model, "system_prompt": system_prompt, "request": request,
             "source_sha256": source_sha256, "seed": seed, "thinking": thinking, "max_tokens": max_tokens,
             "sampling": sampling, "negative": negative, "cleanup_version": CLEANUP_VERSION}
    return sha256_text(json.dumps(parts, sort_keys=True, ensure_ascii=False))
