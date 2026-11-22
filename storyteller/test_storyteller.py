"""Tests of the storyteller. Standard library only, CPU only, no model.

Run from the folder of this file: python -I -m unittest discover -s . -p "test_storyteller.py"
The tests find the style files in an "image-styles" folder of a parent directory, or in the folder that the
environment variable STORYTELLER_STYLES names.
"""
import importlib.util
import json
import os
import re
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _find_module() -> Path:
    for folder in (HERE, HERE.parent / "storyteller"):
        if (folder / "storyteller.py").exists():
            return folder / "storyteller.py"
    raise FileNotFoundError("storyteller.py")


def _find_styles() -> Path:
    if os.environ.get("STORYTELLER_STYLES"):
        return Path(os.environ["STORYTELLER_STYLES"])
    for parent in HERE.parents:
        if (parent / "image-styles").is_dir():
            return parent / "image-styles"
    raise FileNotFoundError("image-styles (set STORYTELLER_STYLES)")


MODULE_PATH = _find_module()
_spec = importlib.util.spec_from_file_location("storyteller_under_test", MODULE_PATH)
st = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(st)

STYLES = sorted(_find_styles().glob("*.json"))
BLOCK = 10  # the size of a block of consecutive seeds in the diversity tests
SEEDS = range(1, BLOCK + 1)
LONG_SEEDS = range(1, 41)
NEGATIVE_KEYS = ("avoid", "explicitly_not_the_style", "do_not_reproduce", "do_not")
FORMAT_STARTS = ("portrait", "landscape", "square")
CANVAS_TEXT = "portrait, taller than wide (2:3)"  # the Format text that style_transfer.py sets for a fixed canvas
MIN_DISTINCT_MOMENTS = 8  # of BLOCK cards
MANY_SEEDS = 200
MIN_LANDING_GROUPS = 20  # distinct groups of homage landing places in MANY_SEEDS cards
ROOM_WORDS = re.compile(r"(room|rooms|interior|furniture|table|chair|shelf|shelves|lamp|kettle|tools|box|cupboard|sofa)")
TIME_WORDS = re.compile(r"(morning|evening|night|dawn|dusk|sunset|sunrise|noon|afternoon|midday|moon|moonlit|low sun)")
SEASON_WORDS = re.compile(r"(snow|snowy|blossom|autumn|winter|spring|summer|harvest)")
# Capitalized word groups that the docs and the code may contain: tool and section names, not works or persons.
ALLOWED_CAPITALIZED: set[str] = set()
PROPER_NAME_RE = re.compile(r"\b[A-Z][a-z]+\b(?: (?:of |the |and )?[A-Z][a-z]+\b)+")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def without_negatives(node):
    """The style without its "no" lists, as style_transfer.py makes it with --rules-positive none."""
    if isinstance(node, dict):
        return {k: without_negatives(v) for k, v in node.items() if k not in NEGATIVE_KEYS + ("negative_prompt",)}
    if isinstance(node, list):
        return [without_negatives(x) for x in node]
    return node


def negative_items(node, key=""):
    if isinstance(node, dict):
        for k, v in node.items():
            yield from negative_items(v, k)
    elif isinstance(node, list) and key in NEGATIVE_KEYS:
        yield from (x for x in node if isinstance(x, str))


def lines(card: str) -> dict:
    return dict(line.split(": ", 1) for line in card.splitlines())


def cores(style: dict, config: dict, label: str, seeds=SEEDS) -> list[str]:
    return [core for seed in seeds for lab, _, core in st.draw_card(style, config, seed) if lab == label]


def config_texts(config: dict):
    """Every text of the library and of the lines of the config."""
    def walk(node):
        if isinstance(node, str):
            yield node
        elif isinstance(node, dict):
            for v in node.values():
                yield from walk(v)
        elif isinstance(node, list):
            for v in node:
                yield from walk(v)
    yield from walk(config.get("library", {}))
    yield from walk(config.get("lines", []))
    for key in ("request_intro", "precedence_note", "open_slot_note"):
        yield config.get(key, "")


class StorytellerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = st.load_config(MODULE_PATH.parent / "storyteller.json")
        cls.styles = {p.stem: load(p) for p in STYLES}
        if not cls.styles:
            raise unittest.SkipTest("no style files found")

    def test_determinism(self):
        for name, style in self.styles.items():
            for seed in SEEDS:
                first = st.story_card(style, self.config, seed)
                again = st.story_card(json.loads(json.dumps(style)), st.load_config(MODULE_PATH.parent / "storyteller.json"), seed)
                self.assertEqual(first, again, f"{name} seed {seed}")

    def test_seeds_give_different_cards(self):
        for name, style in self.styles.items():
            cards = {st.story_card(style, self.config, seed) for seed in SEEDS}
            self.assertEqual(len(cards), BLOCK, name)

    def test_no_style_names_in_code_or_config(self):
        texts = [MODULE_PATH.read_text(encoding="utf-8").lower(),
                 (MODULE_PATH.parent / "storyteller.json").read_text(encoding="utf-8").lower(),
                 (MODULE_PATH.parent / "README.md").read_text(encoding="utf-8").lower()]
        for style in self.styles.values():
            name = style["style_name"].lower()
            for text in texts:
                self.assertFalse(name in text, f"style name in code or config: {name}")

    def test_no_proper_names_in_config(self):
        # A work, an author, a brand, or a person has a capitalized multi-word name. The config has none.
        for text in config_texts(self.config):
            self.assertIsNone(PROPER_NAME_RE.search(text), text)
        for item in (x if isinstance(x, str) else x["text"] for pool in self.config["library"].values() for x in pool):
            self.assertFalse(item[:1].isupper(), item)

    def test_no_negative_rules_leak(self):
        for name, style in self.styles.items():
            bad = {x.lower() for x in negative_items(style) if len(x) > 3}
            for seed in LONG_SEEDS:
                for label, value, core in st.draw_card(style, self.config, seed):
                    self.assertNotIn(core.lower(), bad, f"{name} seed {seed} {label}")

    def test_style_items_have_no_negation(self):
        sources = self.config["style_sources"]
        negation = re.compile(r"\b(no|not|never|avoid|without)\b", re.IGNORECASE)
        for name, style in self.styles.items():
            for group, items in st.style_material(style, sources).items():
                for item in items:
                    self.assertIsNone(negation.search(item), f"{name} {group}: {item}")

    def test_camera_and_format_defer_to_style(self):
        override = next(x for x in self.config["lines"] if x["label"] == "Camera")["if_style_has"]["text"]
        for name, style in self.styles.items():
            comp = style.get("composition", {})
            ratio = " ".join(str(comp.get(k, "")) for k in ("aspect_ratio", "preferred_aspect_ratio")).lower()
            for seed in SEEDS:
                card = lines(st.story_card(style, self.config, seed))
                if "camera" in comp and "Camera" in card:
                    self.assertEqual(card["Camera"], override, name)
                fmt = card.get("Format", "")
                if "preserve" in ratio and "source" in ratio:
                    self.assertEqual(fmt, "the aspect ratio of the source image", name)
                elif re.fullmatch(r"\D*\d{1,2}:\d{1,2}\D*", ratio) and "flexible" not in ratio:
                    self.assertTrue(fmt.startswith(re.search(r"\d{1,2}:\d{1,2}", ratio).group()), name)
                else:
                    self.assertTrue(fmt.startswith(FORMAT_STARTS), f"{name}: {fmt}")

    def test_camera_pool_for_style_without_composition(self):
        style = {"style_name": "plain test style", "composition": {}}
        pool = next(x for x in self.config["lines"] if x["label"] == "Camera")["pool"]
        values = {lines(st.story_card(style, self.config, seed))["Camera"] for seed in SEEDS}
        self.assertTrue(values <= set(pool))
        self.assertGreater(len(values), 1)

    def test_settings_do_not_repeat_in_a_block(self):
        for name, style in self.styles.items():
            settings = cores(style, self.config, "Setting")
            self.assertEqual(len(settings), BLOCK, name)
            style_settings = st.style_lists(style, self.config["style_sources"])[0]
            expected = min(BLOCK, len(style_settings)) if style_settings else BLOCK
            self.assertGreaterEqual(len(set(settings)), expected, f"{name}: {settings}")

    def test_moments_vary_in_a_block(self):
        config = st.set_probabilities(self.config, ["Moment=1"])  # an opt-in line: off in the default card
        for name, style in self.styles.items():
            moments = cores(style, config, "Moment") or cores(style, config, "Story mark")
            limit = min(MIN_DISTINCT_MOMENTS, len(moments))
            self.assertGreaterEqual(len(set(moments)), limit, f"{name}: {moments}")

    def test_default_card_builds_on_image(self):
        # The input image is the plot (user, 2026-11-18): the default card invents no plot, event, story object,
        # moment, acting, or open slot; only an uncanny style gets one surprise, as a change of what the image shows.
        invented = {"Plot", "Event", "Story object", "Moment", "Staging", "Acting", "Story homage"}
        slot_line = next(x for x in self.config["lines"] if x["label"] == "Open slots")
        invented |= {s["label"] for s in slot_line["slots"]}
        for name, style in self.styles.items():
            uncanny = "uncanny" in st.style_tags(style, self.config)
            for seed in LONG_SEEDS:
                card = lines(st.story_card(style, self.config, seed))
                self.assertFalse(invented & set(card), f"{name} seed {seed}: {invented & set(card)}")
                if not uncanny:
                    self.assertNotIn("Surprise", card, f"{name} seed {seed}")
                elif st.style_mode(style, self.config) != "graphic":
                    self.assertIn("Surprise", card, f"{name} seed {seed}")

    def test_only_anime_gets_the_cast_design(self):
        # Each main character gets a unique design, the crowd a simple one (user, 2026-11-19): only for anime.
        hits = set()
        for name, style in self.styles.items():
            for seed in LONG_SEEDS:
                if "Cast design" in lines(st.story_card(style, self.config, seed)):
                    hits.add(name)
        self.assertEqual(hits, {n for n, s in self.styles.items() if "anime" in st.style_tags(s, self.config)})
        self.assertEqual(len(hits), 1, hits)

    def test_graphic_style_gets_no_scene_lines(self):
        for name, style in self.styles.items():
            if st.style_mode(style, self.config) != "graphic":
                continue
            for seed in LONG_SEEDS:
                card = lines(st.story_card(style, self.config, seed))
                for label in ("Time of day", "Moment", "Event", "Spot", "Story object", "Outside the frame"):
                    self.assertNotIn(label, card, f"{name} seed {seed}")

    def test_modes_are_found_from_style_text(self):
        modes = {st.style_mode(style, self.config) for style in self.styles.values()}
        self.assertIn("scene", modes)

    def test_tagged_items_only_for_matching_styles(self):
        slot = "Wrong detail"
        for name, style in self.styles.items():
            if "uncanny" in st.style_tags(style, self.config):
                continue
            for seed in LONG_SEEDS:
                self.assertNotIn(slot, lines(st.story_card(style, self.config, seed)), name)

    def test_open_slots_and_homage(self):
        slot_line = next(x for x in self.config["lines"] if x["label"] == "Open slots")
        slot_labels = {s["label"] for s in slot_line["slots"]}
        config = st.set_probabilities(self.config, ["Open slots=1"])  # an opt-in line: off in the default card
        for name, style in self.styles.items():
            homage = 0
            for seed in LONG_SEEDS:
                card = lines(st.story_card(style, config, seed))
                slots = [label for label in card if label in slot_labels]
                self.assertGreaterEqual(len(slots), 1, f"{name} seed {seed}")
                self.assertLessEqual(len(slots), slot_line["count"])
                if "Homage" in card:
                    homage += 1
                    self.assertIn("Leave out characters, names, logos, and text", card["Homage"])
            self.assertGreater(homage, 0, name)

    def test_request_has_slot_note_only_with_slots(self):
        style = next(iter(self.styles.values()))
        request = st.build_request(style, st.set_probabilities(self.config, ["Open slots=1"]), 1, "{}")
        note = st._Context(style, self.config, 1).rules(self.config["open_slot_note"])  # with its rules inserted
        self.assertIn(note, request)
        no_slots = st.set_probabilities(self.config, ["Open slots=0"])
        self.assertNotIn(note, st.build_request(style, no_slots, 1, "{}"))

    def test_style_transfer_call_sequence(self):
        """The calls of comfyui/2-storyteller/style_transfer.py in story mode, with a fixed and an automatic canvas."""
        for fixed_canvas in (False, True):
            config = st.set_probabilities(st.load_config(MODULE_PATH.parent / "storyteller.json"), ["Homage=1"])
            if fixed_canvas:
                for line in config["lines"]:
                    if line["label"] == "Format":
                        line.pop("pool", None)
                        line["text"] = CANVAS_TEXT
            for name, style in self.styles.items():
                for spec in (style, without_negatives(style)):
                    for seed in SEEDS:
                        card = st.story_card(spec, config, seed)
                        fmt = st.card_value(card, "Format")
                        self.assertTrue(fmt, name)
                        if fixed_canvas:
                            self.assertEqual(fmt, CANVAS_TEXT)
                        self.assertTrue(st.card_value(card, "Homage"))
                        request = st.build_request(spec, config, seed, "outline")
                        self.assertIn(card, request)
                        slug = st.story_slug(style, config, seed)
                        self.assertTrue(slug and len(slug) <= st.SLUG_MAX_LENGTH, name)
                        self.assertEqual(slug, st.story_slug(spec, config, seed), name)

    def test_decorative_cards_have_no_room_words(self):
        # A decorative style drew a room when the card named furniture or interior objects (GOAL.md, R8/R9).
        for name, style in self.styles.items():
            if st.style_mode(style, self.config) != "decorative":
                continue
            for seed in LONG_SEEDS:
                card = st.story_card(style, self.config, seed).lower()
                self.assertIsNone(ROOM_WORDS.search(card), f"{name} seed {seed}: {ROOM_WORDS.search(card)}")

    def test_scene_items_have_no_time_or_season(self):
        # The Time of day line sets the time: a setting, a concept, or a moment must not set another one.
        for pool in ("tensions", "relationships", "settings", "spots", "story_objects", "moments", "events", "staging"):
            for item in self.config["library"][pool]:
                text = item if isinstance(item, str) else item["text"]
                self.assertIsNone(TIME_WORDS.search(text), f"{pool}: {text}")
        for item in self.config["library"]["settings"] + self.config["library"]["spots"]:
            text = item if isinstance(item, str) else item["text"]
            self.assertIsNone(SEASON_WORDS.search(text), text)

    def test_homage_landing_groups_vary(self):
        config = st.set_probabilities(self.config, ["Homage=1"])
        for name, style in self.styles.items():
            groups = set()
            for seed in range(1, MANY_SEEDS + 1):
                value = st.card_value(st.story_card(style, config, seed), "Homage")
                groups.add(value.split("these places: ", 1)[1].split(".")[0])
            ctx = st._Context(style, config, 1)
            allowed = [x for x in config["library"]["landings"] if ctx.allowed(st._item(x))]
            # with only a few allowed places (a graphic style), each place and each pair in both orders can occur
            expected = min(MIN_LANDING_GROUPS, len(allowed) * len(allowed))
            self.assertGreaterEqual(len(groups), expected, name)

    def test_rules_are_inserted_and_gated(self):
        """A {rule:NAME} becomes its text; a rule for scene styles is empty in a graphic card; no placeholder stays."""
        config = json.loads(json.dumps(self.config))
        config["rules"]["scene_only"] = {"text": "SCENE RULE.", "needs_any": ["scene", "decor"]}
        config["request_intro"] += " {rule:scene_only}"
        for name, style in self.styles.items():
            request = st.build_request(style, config, 1, "{}")
            self.assertNotIn("{rule:", request, name)
            scene = st.style_mode(style, config) != "graphic"
            self.assertEqual("SCENE RULE." in request, scene, name)
        config["request_intro"] += " {rule:missing}"
        with self.assertRaises(KeyError):
            st.build_request(next(iter(self.styles.values())), config, 1, "{}")

    def test_recipes_vary_and_can_be_forced(self):
        style = next(s for s in self.styles.values() if st.style_mode(s, self.config) == "scene")
        names = {st._Context(style, self.config, seed).recipe.get("name") for seed in range(1, MANY_SEEDS + 1)}
        # a recipe with weight 0 is opt-in: only --recipe uses it
        self.assertEqual(names, {r["name"] for r in self.config["recipes"] if r.get("weight", 1) > 0})
        quiet = st.set_recipe(self.config, "quiet moment")
        for seed in SEEDS:
            card = lines(st.story_card(style, quiet, seed))
            self.assertNotIn("Event", card)
            self.assertIn("Approach", card)
        with self.assertRaises(ValueError):
            st.set_recipe(self.config, "no such recipe")
        labels = {x["label"] for x in self.config["lines"]}
        for recipe in self.config["recipes"]:  # a recipe changes the story lines, never the frame of the card
            self.assertLessEqual(set(recipe.get("lines", {})), labels, recipe["name"])
            self.assertFalse(set(recipe.get("lines", {})) & {"Concept", "Setting", "Camera", "Shot", "Format", "Time of day"},
                             recipe["name"])

    def test_budgets_limit_object_lines(self):
        kinds = {x["label"]: x.get("kind") for x in self.config["lines"]}
        kinds.update({s["label"]: s.get("kind") for x in self.config["lines"] for s in x.get("slots", [])})
        for name, style in self.styles.items():
            for seed in range(1, MANY_SEEDS + 1):
                ctx = st._Context(style, self.config, seed)
                card = lines(st.story_card(style, self.config, seed))
                objects = sum(1 for label in card if kinds.get(label) == "object")
                self.assertLessEqual(objects, ctx.budgets.get("object", objects), f"{name} seed {seed}")

    def test_homage_without_influences(self):
        style = {"style_name": "plain test style", "composition": {}}
        config = st.set_probabilities(self.config, ["Homage=1", "Story homage=1"])
        card = lines(st.story_card(style, config, 1))
        self.assertIn("Homage", card)
        self.assertIn("Story homage", card)

    def test_shot_line_defers_to_style(self):
        shot = next(x for x in self.config["lines"] if x["label"] == "Shot")
        override = next(x for x in self.config["lines"] if x["label"] == "Camera")["if_style_has"]["text"]
        for name, style in self.styles.items():
            if not st.style_material(style, self.config["style_sources"])["shots"]:
                continue
            for seed in LONG_SEEDS:
                card = lines(st.story_card(style, self.config, seed))
                self.assertNotEqual("Shot" in card, "Camera" in card, f"{name} seed {seed}")
                if "Shot" in card:
                    self.assertTrue(card["Shot"].startswith(shot["item_prefix"]))
                else:
                    self.assertEqual(card["Camera"], override)

    def test_modes_of_synthetic_styles(self):
        graphic = {"style_name": "synthetic sign", "composition": {"structure": "One flat sign on blank paper, no scenery"}}
        decorative = {"style_name": "synthetic ornament", "background_design": {"type": "Decorative ornamental field"}}
        scene = {"style_name": "synthetic scene", "background_design": {"style": "Painted everyday rooms and streets"}}
        self.assertEqual(st.style_mode(graphic, self.config), "graphic")
        self.assertEqual(st.style_mode(decorative, self.config), "decorative")
        self.assertEqual(st.style_mode(scene, self.config), "scene")
        for style in (graphic, decorative, scene):
            self.assertEqual(len({st.story_card(style, self.config, seed) for seed in SEEDS}), BLOCK)

    def test_no_proper_names_in_docs_and_code(self):
        for path in (MODULE_PATH, MODULE_PATH.parent / "README.md"):
            for match in PROPER_NAME_RE.finditer(path.read_text(encoding="utf-8")):
                self.assertIn(match.group(), ALLOWED_CAPITALIZED, f"{path.name}: {match.group()}")

    def test_set_probabilities(self):
        config = st.set_probabilities(self.config, ["Homage=0", "Mood=1"])
        style = next(iter(self.styles.values()))
        for seed in SEEDS:
            self.assertNotIn("Homage", lines(st.story_card(style, config, seed)))
        with self.assertRaises(ValueError):
            st.set_probabilities(self.config, ["Nope=1"])


if __name__ == "__main__":
    unittest.main()
