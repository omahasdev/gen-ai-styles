"""Tests of the ComfyUI workflow script and its modules. CPU only, no server: a stub client records the graphs.

Run from the repository root: python -I -m unittest discover tests
"""
import contextlib
import io
import json
import struct
import sys
import tempfile
import unittest
import zlib
from pathlib import Path

TOOL = Path(__file__).resolve().parents[1]
REPO = TOOL.parents[1]
sys.path.insert(0, str(TOOL))
import graph_check  # noqa: E402
import prompt_hygiene  # noqa: E402
import style_schema  # noqa: E402
import style_transfer as st  # noqa: E402

STYLES = REPO / "image-styles"
BASELINE = STYLES / "_baseline.png"
# Styles that the tests leave out until the owner adds them to the test set (none now).
EXCLUDED_STYLES: set[str] = set()
TEST_STYLE = STYLES / "Art Nouveau Character Arch.json"
LEAK_TEXT = "Let me write the prompt.\nWord count: 300.\nA cat."
IDENTITY_TEXT = "Identity to keep: a brown-gray tabby cat with a white chest."
# A varied prompt of more than MIN_WORDS distinct words: a repeated sentence would count as degenerate text.
CLEAN_TEXT = "A drawn tabby cat sits among lilies inside a pale arch. " + " ".join(
    a + b + c for a in "bcdfg" for b in "aeiou" for c in "lmnprst")


def test_styles():
    return sorted(p for p in STYLES.glob("*.json") if p.stem not in EXCLUDED_STYLES)


class StubClient:
    """Records the queued graphs. The writer answers LEAK_TEXT without thinking when leak=True."""
    leak = False
    wrapped = False
    place = "PLAIN"
    fail = False
    cut = False

    def __init__(self, server):
        self.graphs = []
        StubClient.last = self

    def upload_image(self, path):
        return Path(path).name

    def get_json(self, path):
        return {"system": {"comfyui_version": "stub"}}

    def queue(self, graph, front=False):
        self.graphs.append((json.loads(json.dumps(graph)), front))
        return str(len(self.graphs) - 1)

    def wait(self, prompt_id):
        graph, _ = self.graphs[int(prompt_id)]
        outputs = {}
        writer = graph.get(st.NODE_ENH_GENERATE)
        if writer:
            thinking = writer["inputs"]["thinking"]
            text = LEAK_TEXT if (StubClient.leak and not thinking) else CLEAN_TEXT
            if StubClient.cut and thinking:
                text = 'reasoning ' + st.prompt_hygiene.THINK_END + ' {' + chr(34) + 'instruction' + chr(34) + ': ' + chr(34) + 'A cat'
            if StubClient.wrapped:
                text = json.dumps({"prompt": text})
            outputs[st.NODE_ENH_SHOW] = {"text": [text]}
        if st.NODE_NEG_GENERATE in graph:
            outputs[st.NODE_NEG_SHOW] = {"text": ["blur, photo"]}
        if st.NODE_PLACE_GENERATE in graph:
            outputs[st.NODE_PLACE_SHOW] = {"text": [StubClient.place]}
        if st.NODE_ID_GENERATE in graph:
            outputs[st.NODE_ID_SHOW] = {"text": [IDENTITY_TEXT]}
        if st.NODE_SAVE in graph:
            outputs[st.NODE_SAVE] = {"images": [{"filename": "x.png", "subfolder": "", "type": "output"}]}
        if StubClient.fail and writer:
            return {"status": {"status_str": "error", "messages": []}, "outputs": {}}
        return {"status": {"status_str": "success"}, "outputs": outputs}

    def download(self, image):
        return b"png"


def run_script(args):
    """Run style_transfer.main() with a stub client; return (exit code, stub, printed text)."""
    st.ComfyClient = StubClient
    old = sys.argv
    sys.argv = ["style_transfer.py", *map(str, args)]
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            code = st.main()
    finally:
        sys.argv = old
    return code, StubClient.last if hasattr(StubClient, "last") else None, out.getvalue()


def png_header(width, height):
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    chunk = struct.pack(">I", len(ihdr)) + b"IHDR" + ihdr + struct.pack(">I", zlib.crc32(b"IHDR" + ihdr))
    return b"\x89PNG\r\n\x1a\n" + chunk


def jpeg_header(width, height, orientation=None):
    data = b"\xff\xd8"
    if orientation:
        tiff = b"II*\x00" + struct.pack("<I", 8) + struct.pack("<H", 1)
        tiff += struct.pack("<HHIHH", 0x0112, 3, 1, orientation, 0) + b"\x00\x00\x00\x00"
        app1 = b"Exif\x00\x00" + tiff
        data += b"\xff\xe1" + struct.pack(">H", len(app1) + 2) + app1
    sof = struct.pack(">BHHB", 8, height, width, 3) + b"\x01\x11\x00\x02\x11\x00\x03\x11\x00"
    return data + b"\xff\xc0" + struct.pack(">H", len(sof) + 2) + sof


class ImageSizeTest(unittest.TestCase):
    def check(self, data, expected):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x"
            path.write_bytes(data)
            self.assertEqual(st.image_size(path), expected)

    def test_formats(self):
        self.check(png_header(300, 200), (300, 200))
        self.check(jpeg_header(300, 200), (300, 200))
        self.check(jpeg_header(300, 200, orientation=6), (200, 300))
        self.check(jpeg_header(300, 200, orientation=3), (300, 200))
        bmp = b"BM" + b"\x00" * 16 + struct.pack("<ii", 300, -200)
        self.check(bmp, (300, 200))
        vp8x = b"RIFF\x00\x00\x00\x00WEBPVP8X" + b"\x00" * 8 + (299).to_bytes(3, "little") + (199).to_bytes(3, "little")
        self.check(vp8x, (300, 200))
        bits = (300 - 1) | ((200 - 1) << 14)
        vp8l = b"RIFF\x00\x00\x00\x00WEBPVP8L" + b"\x00" * 4 + b"\x2f" + bits.to_bytes(4, "little")
        self.check(vp8l, (300, 200))

    def test_canvas_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wide.png"
            path.write_bytes(png_header(4000, 1000))
            width, height = st.canvas_size(st.CANVAS_SOURCE, path)
            self.assertAlmostEqual(width / height, st.CANVAS_MAX_RATIO, delta=0.05)


class HygieneTest(unittest.TestCase):
    def test_clean(self):
        self.assertEqual(prompt_hygiene.clean(json.dumps({"prompt": 'A "drawn"\ncat.'})), 'A "drawn"\ncat.')
        self.assertEqual(prompt_hygiene.clean("```json\n{\"prompt\": \"A cat.\"}\n```"), "A cat.")
        self.assertEqual(prompt_hygiene.clean("  A cat.  "), "A cat.")
        self.assertEqual(prompt_hygiene.clean("{not json"), "{not json")
        leaked = 'Draft {"instruction": "x"}</user>\n</think>\n\n{"instruction": "A drawn cat."}'
        self.assertEqual(prompt_hygiene.clean(leaked), "A drawn cat.")

    def test_a_color_that_names_an_object_keeps_only_its_hue(self):
        text = "A coat in ochre, Sky Blue and muted foliage green, with brick-red trim."
        self.assertEqual(prompt_hygiene.clean(text), "A coat in ochre, light blue and muted green, with red-brown trim.")
        self.assertEqual(prompt_hygiene.clean("A blue sky over green grass."), "A blue sky over green grass.")
        self.assertEqual(prompt_hygiene.clean("Ochre and muted foliage greens."), "Ochre and muted greens.")

    def test_a_fabric_texture_word_goes_and_the_garment_stays(self):
        text = "A teal cable-knit scarf over a Ribbed gray vest and a knit cap."
        self.assertEqual(prompt_hygiene.clean(text), "A teal scarf over a gray vest and a cap.")
        wall = "Add a soft warm light pool grazing the top of the wall. A warm light on the floor."
        self.assertEqual(prompt_hygiene.clean(wall),
                         "Add a soft light pool grazing the top of the wall. A warm light on the floor.")
        notes = "Identity to keep: a man in a quilted brown jacket."
        self.assertEqual(st.clean_identity(notes), "Identity to keep: a man in a brown jacket.")

    def test_a_bundle_on_a_prop_gets_no_face_and_dark_navy_stays_dark(self):
        figure = "Draw the little felt-wrapped owl figure with its green scarf taped to the oar, the oar wet."
        self.assertEqual(prompt_hygiene.clean(figure),
                         "Draw the small rough cloth bundle with its green scarf taped to the oar, the oar wet.")
        self.assertEqual(prompt_hygiene.clean("Keep the owl-like felt bundle on the oar."),
                         "Keep the felt bundle on the oar.")
        face = "The felt bundle hangs from the oar, its round face tilted up with big eyes, the boat rocks."
        self.assertEqual(prompt_hygiene.clean(face), "The felt bundle hangs from the oar, the boat rocks.")
        self.assertEqual(prompt_hygiene.clean("A deep navy jacket. A golden glow on the stone backdrop."),
                         "A near-black jacket. A glow on the stone backdrop.")
        self.assertEqual(st.clean_identity("Identity to keep: a dark navy skirt."),
                         "Identity to keep: a near-black skirt.")

    def test_a_palette_list_goes_and_the_element_colors_stay(self):
        text = "A bright palette of lemon yellows, mint, coral and soft violet across the frame, the cap stays mint."
        self.assertEqual(prompt_hygiene.clean(text), "A bright palette across the frame, the cap stays mint.")
        self.assertEqual(prompt_hygiene.clean("Keep the palette of the input."), "Keep the palette of the input.")

    def test_a_doll_figure_on_a_prop_becomes_a_bundle(self):
        text = "An oar with a stitched doll figure with gray hair and a green cap lashed to the blade."
        self.assertEqual(prompt_hygiene.clean(text), "An oar with a stitched rough cloth bundle with a gray tuft and a "
                                                     "green cap lashed to the blade.")
        notes = "Identity to keep: a boy holding a kite with a felt doll figure with black hair on the right end."
        self.assertNotIn("doll", st.clean_identity(notes))
        self.assertTrue(prompt_hygiene.degenerate_reasons(' " the " a, " " the' * 20))

    def test_a_bare_dark_garment_is_near_black_and_a_hand_has_no_claws(self):
        text = "A dark wool overcoat, a dark green scarf, a darker sash, and a taloned hand on the rail."
        self.assertEqual(prompt_hygiene.clean(text), "A near-black wool overcoat, a dark green scarf, a darker sash, "
                                                     "and a hand on the rail.")

    def test_a_far_plane_keeps_no_blue(self):
        text = "Reeds in front, a misty blue horizon band behind the pier, and a blue boat."
        self.assertEqual(prompt_hygiene.clean(text), "Reeds in front, a horizon band behind the pier, and a blue boat.")
        self.assertEqual(prompt_hygiene.clean("a cool blue-grey stone floor under the stall"),
                         "a stone floor under the stall")

    def test_skin_keeps_no_hue_that_drifts_to_yellow(self):
        text = ("A goblin with a sallow greenish-gold complexion, a pale face, and dark brown skin on the arms; "
                "its knobby skin as flat golden-olive fill under a yellow lantern.")
        self.assertEqual(prompt_hygiene.clean(text),
                         "A goblin with a sallow complexion, a pale face, and dark brown skin on the arms; "
                         "its knobby skin as flat fill under a yellow lantern.")
        self.assertEqual(prompt_hygiene.clean("the troll's face drawn as wide flat amber-orange color shapes"),
                         "the troll's face drawn as wide flat color shapes")
        self.assertEqual(prompt_hygiene.clean("A muted gray-green apron over a sage-green tunic, a green hat, "
                                              "and a broadcast range of rust, teal, and deep plum: the cart"),
                         "A gray apron over a gray tunic, a green hat, and a broadcast range: the cart")
        notes = "Identity to keep: an ogre with a lumpy olive-tan face and a gold ring."
        self.assertEqual(st.clean_identity(notes), "Identity to keep: an ogre with a lumpy face and a gold ring.")

    def test_the_writer_gets_the_notes_without_the_face(self):
        notes = ("Identity to keep: a man, red short hair, thin lips pressed shut, wearing a green apron; "
                 "he holds a ladle in one hand. Behind him a tiled wall.")
        self.assertEqual(st.writer_notes(notes),
                         "Identity to keep: a man, red short hair, wearing a green apron. Behind him a tiled wall.")

    def test_repeated_drafts_keep_the_last_one(self):
        # The writer printed eight drafts one after another (5,766 words, 2026-11-19): keep the last draft only.
        opening = "Redraw the input photograph of a cat on a wooden floor as one frame of the style."
        draft1 = f"{opening} The fur is soft paint. Good."
        draft2 = f"{opening} The fur is flat cel paint with one hard shadow."
        text = f"{draft1} {draft2}"
        self.assertTrue(any("repeated" in r for r in prompt_hygiene.leak_reasons(text)))
        self.assertEqual(prompt_hygiene.clean(text), draft2)
        self.assertEqual(prompt_hygiene.leak_reasons(prompt_hygiene.clean(text)), [])
        self.assertEqual(prompt_hygiene.clean(draft2), draft2)  # one draft stays as it is

    def test_leak(self):
        self.assertTrue(prompt_hygiene.leak_reasons(LEAK_TEXT))
        self.assertTrue(prompt_hygiene.leak_reasons('Use "no" shadows.'))
        self.assertFalse(prompt_hygiene.leak_reasons(CLEAN_TEXT))
        collapse = ' '.join(['a the "the, a "'] * 60)
        self.assertTrue(prompt_hygiene.degenerate_reasons(collapse))
        self.assertTrue(prompt_hygiene.leak_reasons(collapse))

    def test_key(self):
        base = dict(writer_model="m", system_prompt="s", request="r", source_sha256="h", seed=1, thinking=False,
                    max_tokens=10, sampling={"t": 1})
        self.assertEqual(prompt_hygiene.prompt_key(**base), prompt_hygiene.prompt_key(**base))
        self.assertNotEqual(prompt_hygiene.prompt_key(**base), prompt_hygiene.prompt_key(**{**base, "seed": 2}))


class SchemaTest(unittest.TestCase):
    def test_styles_valid(self):
        for path in test_styles():
            problems, _ = style_schema.validate(json.loads(path.read_text(encoding="utf-8")))
            self.assertEqual(problems, [], path.name)

    def test_minimal_style(self):
        style = {key: {} for key in style_schema.TOP_KEYS}
        style["style_name"] = "Minimal"
        style["style_fidelity"] = {k: [] for k in style_schema.FIDELITY_KEYS}
        style["source_adaptation"] = {k: ({} if k == "detail_budget" else "x")
                                      for k in style_schema.SOURCE_ADAPTATION_KEYS}
        style["generation_instructions"] = {k: "x" for k in style_schema.GENERATION_KEYS}
        self.assertEqual(style_schema.validate(style)[0], [])
        style["rendering"] = {"realism_level": 2}
        self.assertTrue(style_schema.validate(style)[0])

    def test_no_rule_keys(self):
        for key in ("avoid", "explicitly_not_the_style", "do_not_reproduce", "do_not", "negative_prompt"):
            self.assertTrue(style_schema.is_negative_key(key), key)
        stripped = st.without_negatives({"a": {"do_not": ["x"], "keep": ["y"]}})
        self.assertEqual(stripped, {"a": {"keep": ["y"]}})


class GraphCheckTest(unittest.TestCase):
    INFO = {"Loader": {"input": {"required": {"name": [["a", "b"]]}}, "output": ["MODEL"]},
            "Use": {"input": {"required": {"model": ["MODEL"], "steps": ["INT", {}]}}, "output": ["IMAGE"]}}

    def test_good_graph(self):
        graph = {"1": {"class_type": "Loader", "inputs": {"name": "a"}},
                 "2": {"class_type": "Use", "inputs": {"model": ["1", 0], "steps": 8}}}
        self.assertEqual(graph_check.check(graph, self.INFO), [])

    def test_faults(self):
        graph = {"1": {"class_type": "Unknown", "inputs": {}},
                 "2": {"class_type": "Use", "inputs": {"model": ["9", 0]}},
                 "3": {"class_type": "Use", "inputs": {"model": ["2", 0], "steps": 8}}}
        problems = " ".join(graph_check.check(graph, self.INFO))
        self.assertIn("unknown class", problems)
        self.assertIn("missing required input steps", problems)
        self.assertIn("link to missing node", problems)
        self.assertIn("type IMAGE", problems)

    def test_outline_has_no_quotes(self):
        for path in test_styles():
            style = json.loads(path.read_text(encoding="utf-8"))
            prompt = st.build_direct_prompt(style, {}, "explained", "outline")
            self.assertNotIn('"', prompt, path.name)


class FixedRatioTest(unittest.TestCase):
    def test_style_ratio(self):
        self.assertEqual(st.style_fixed_ratio({"composition": {"aspect_ratio": "3:4 portrait"}}), (3, 4))
        self.assertIsNone(st.style_fixed_ratio({"composition": {"preferred_aspect_ratio": "2:3 or 3:5"}}))
        self.assertIsNone(st.style_fixed_ratio({"composition": {"aspect_ratio": "Preserve the source 4:3"}}))
        self.assertIsNone(st.style_fixed_ratio({"composition": {"aspect_ratio": "Flexible"}}))

    def test_canvas_from_ratio(self):
        width, height = st.canvas_size(st.RATIO_PREFIX + "3:4", BASELINE)
        self.assertAlmostEqual(width / height, 3 / 4, delta=0.02)


class NamesTest(unittest.TestCase):
    def test_tag(self):
        stem = st.result_name(Path("_baseline.png"), TEST_STYLE, ["story-s1"])
        self.assertEqual(st.with_tag(stem, "retry-think"), "_baseline (Art Nouveau Character Arch, story-s1, retry-think)")


class ScriptTest(unittest.TestCase):
    def setUp(self):
        StubClient.leak = StubClient.wrapped = StubClient.fail = StubClient.cut = False
        StubClient.place = "PLAIN"
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_the_scene_reading_follows_the_concept_line(self):
        config = {"lines": [{"label": "Concept", "text": "c"}, {"label": "Format", "text": "f"}]}
        lines = st.with_scene_line(config, "Scene: a standoff.")["lines"]
        self.assertEqual([x["label"] for x in lines], ["Concept", st.SCENE_LABEL, "Format"])
        self.assertTrue(lines[1]["text"].endswith("Scene: a standoff."))
        self.assertEqual(len(config["lines"]), 2)  # the shared config stays as it is
        self.assertIs(st.with_scene_line(config, None), config)

    def test_a_shape_budget_style_gets_the_signature_identity_request(self):
        counted = {"source_adaptation": {"detail_budget": {"figure": "Four to six large flat shapes and one detail"}}}
        free = {"source_adaptation": {"detail_budget": {"figure": "Preserve recognizable features."}}}
        self.assertIs(st.identity_request_of(counted), st.IDENTITY_SIGNATURE_REQUEST)
        self.assertIs(st.identity_request_of(free), st.IDENTITY_REQUEST)
        self.assertIs(st.identity_request_of({}), st.IDENTITY_REQUEST)

    def test_a_style_with_a_one_ink_face_gets_gray_heads(self):
        self.assertTrue(st.gray_face({"subject_design": {"people": "The face stays a monochrome print."}}))
        self.assertFalse(st.gray_face({"subject_design": {"face": "Large eyes. A duotone sky."}}))
        self.assertFalse(st.gray_face({}))

    def test_a_figure_fixed_to_a_prop_becomes_a_rough_bundle(self):
        scene = st.clean_scene("Props: a toy head with red hair and a green hat tied to the back of a chair.")
        self.assertIn("a small rough cloth bundle with a red tuft and a green hat tied", scene)
        self.assertNotIn("toy", scene)
        notes = st.clean_identity("Identity to keep: a small figurine in yellow clothing taped to the lamp post.")
        self.assertIn("a small rough cloth bundle in yellow clothing taped", notes)
        held = "Identity to keep: a boy holding a doll in his arms."
        self.assertEqual(st.clean_identity(held), held)
        notes = st.clean_identity("Identity to keep: a small elf figure with a green cap attached at the right end.")
        self.assertIn("a small rough cloth bundle with a green cap attached", notes)
        cloak = "Identity to keep: a tall figure wrapped in a gray cloak."
        self.assertEqual(st.clean_identity(cloak), cloak)

    def test_the_place_line_names_the_surfaces(self):
        self.assertEqual(st.clean_surfaces("A plain gray concrete wall with soft streaks.\nMore text."),
                         "a plain gray concrete wall with soft streaks")
        self.assertIsNone(st.clean_surfaces(""))
        self.assertIsNone(st.clean_surfaces(" ".join(["word"] * (st.SURFACES_MAX_WORDS + 1))))
        config = {"lines": [{"label": "Concept", "text": "c"}, {"label": "Setting", "text": "s"}]}
        place = st.without_setting_lines(config, "a plain gray concrete wall")["lines"][1]["text"]
        self.assertTrue(place.startswith("the place of the input image (a plain gray concrete wall), redrawn"))
        self.assertEqual(st.without_setting_lines(config)["lines"][1]["text"], st.PLACE_LINE)

    def test_only_four_numbers_give_a_creature_box(self):
        self.assertEqual(st.parse_creature_box("470 0 998 700"), (0.47, 0.0, 0.998, 0.7))
        self.assertEqual(st.parse_creature_box("[470, 0, 998, 700]"), (0.47, 0.0, 0.998, 0.7))
        self.assertIsNone(st.parse_creature_box("NONE"))
        self.assertIsNone(st.parse_creature_box("Looking at the image: two people are depicted."))
        self.assertIsNone(st.parse_creature_box("The man stands at 10 20 and the woman at 30 40."))
        self.assertIsNone(st.parse_creature_box("700 0 300 500"))  # left after right
        self.assertIsNone(st.parse_creature_box("0 0 1200 500"))  # outside the image

    def test_the_colors_line_follows_the_concept_line(self):
        config = {"lines": [{"label": "Concept", "text": "c"}, {"label": "Format", "text": "f"}]}
        lines = st.with_scene_line(st.with_colors_line(config), "Scene: a standoff.")["lines"]
        self.assertEqual([x["label"] for x in lines], ["Concept", st.SCENE_LABEL, st.COLORS_LABEL, st.GARMENTS_LABEL, "Format"])
        self.assertEqual(len(config["lines"]), 2)  # the shared config stays as it is

    def test_scene_cleanup_keeps_the_five_lines_only(self):
        raw = ("Moment: a wild charge.\n\nFeelings: wild joy, eyes squeezed shut.\n\nTone: comic.\n\n"
               "Acting: brows up.\n\nProps: a stick against a blurred wood.\n\nWait, the props must not name the place.")
        self.assertEqual(st.clean_scene(raw), "Moment: a wild charge. Feelings: wild joy, eyes narrowed but open. "
                                              "Tone: comic. Acting: brows up. Props: a stick against a wood.")
        self.assertEqual(st.clean_scene("Acting: a calm mouth, her body upright; no tears, no anger marks, no blush. "
                                        "Props: a hat."), "Acting: a calm mouth, her body upright. Props: a hat.")
        self.assertEqual(st.clean_scene("Acting: brows high, all muscles tense; no sweat drop, no tear, no anger mark."),
                         "Acting: brows high, all muscles tense.")
        self.assertEqual(st.clean_scene("Acting: a wide smile, with no anger, no fear, and no pose anywhere. Props: a cup."),
                         "Acting: a wide smile. Props: a cup.")
        self.assertEqual(st.clean_scene('{"Moment": "a walk.", "Tone": "quiet"}'), "Moment: a walk. Tone: quiet")

    def test_a_source_with_a_place_gets_no_setting_lines(self):
        def request(place):
            StubClient.place = place
            code, stub, _ = run_script(self.base("--mode", "story", "--story-seed", "601"))
            self.assertEqual(code, 0)
            manifest = json.loads(next(self.out.glob("*.json")).read_text(encoding="utf-8"))
            for f in self.out.glob("*"):
                f.unlink() if f.is_file() else None
            graph = stub.graphs[0][0]
            prompt = graph[st.NODE_ENH_GENERATE]["inputs"]["prompt"]
            if prompt == [st.NODE_WRITER_NOTES, 0]:  # the writer gets its request plus the identity facts
                prompt = graph[st.NODE_WRITER_NOTES]["inputs"]["string_a"]
            return prompt, manifest["source_place"]
        plain, seen = request("PLAIN")
        self.assertFalse(seen)
        self.assertIn("Setting:", plain)  # story seed 601 of the test style draws a setting
        placed, seen = request("PLACE")
        self.assertTrue(seen)
        self.assertNotIn("Setting:", placed)
        self.assertNotIn("Spot:", placed)
        self.assertIn("Place: " + st.PLACE_LINE, placed)
        self.assertNotIn("Place: ", plain)
        lines = st.without_setting_lines({"lines": [{"label": "Concept"}, {"label": "Setting"}, {"label": "Mood"}]})
        self.assertEqual([x["label"] for x in lines["lines"]], ["Concept", st.PLACE_LABEL, "Mood"])

    def base(self, *extra):
        return [BASELINE, "--styles", TEST_STYLE, "--seed", "1", "--output", self.out, *extra]

    def test_help(self):
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as done:
            sys.argv = ["style_transfer.py", "--help"]
            st.main()
        self.assertEqual(done.exception.code, 0)

    def test_one_graph_manifest(self):
        code, stub, _ = run_script(self.base("--mode", "adapt"))
        self.assertEqual(code, 0)
        self.assertEqual(len(stub.graphs), 1)
        manifest = json.loads(next(self.out.glob("*.json")).read_text(encoding="utf-8"))
        self.assertEqual(manifest["mode"], "adapt")
        self.assertEqual(manifest["phases"], "one graph")
        self.assertIn("written_prompt_sha256", manifest)

    def test_identity_notes_follow_the_written_prompt(self):
        code, stub, _ = run_script(self.base("--mode", "adapt", "--identity-notes", "on"))
        self.assertEqual(code, 0)
        graph = stub.graphs[0][0]
        self.assertEqual(graph[st.NODE_ENCODE]["inputs"]["prompt"], [st.NODE_ID_JOIN, 0])
        self.assertEqual(graph[st.NODE_ID_GENERATE]["inputs"]["image"], [st.NODE_IMAGE, 0])  # the sharp source
        manifest = json.loads(next(self.out.glob("*.json")).read_text(encoding="utf-8"))
        self.assertTrue(manifest["identity_notes"])
        self.assertEqual(manifest["written_prompt_sha256"],
                         prompt_hygiene.sha256_text(CLEAN_TEXT + "\n\n" + IDENTITY_TEXT))
        code, stub, _ = run_script(self.base("--mode", "adapt", "--source-soften", "0"))  # auto, no softening: off
        self.assertNotIn(st.NODE_ID_GENERATE, stub.graphs[0][0])

    def test_identity_cleanup_ids_clash_with_no_other_node(self):
        graph = json.loads(st.DEFAULT_GRAPH.read_text(encoding="utf-8"))
        fixed = set(graph) | {st.NODE_ID_GENERATE, st.NODE_ID_SHOW, st.NODE_ID_JOIN, st.NODE_ID_IMAGE,
                              st.NODE_ENH_CLIP, st.NODE_ENH_SYSTEM, st.NODE_ENH_GENERATE, st.NODE_ENH_SHOW,
                              st.NODE_PLACE_IMAGE, st.NODE_PLACE_GENERATE, st.NODE_PLACE_SHOW,
                              st.NODE_SCENE_IMAGE, st.NODE_SCENE_GENERATE, st.NODE_SCENE_SHOW}
        cleanup = {str(st.NODE_ID_CLEAN_FIRST + n) for n in range(len(st.IDENTITY_CLEANUP))}
        self.assertFalse(cleanup & fixed)

    def test_identity_cleanup_drops_denied_items_and_repeats(self):
        raw = ("Identity to keep: a man, no glasses, a green jacket. (26 words)\n"
               "Identity to keep: a man again. No hats.")
        self.assertEqual(st.clean_identity(raw), "Identity to keep: a man, a green jacket.")
        kept = "Identity to keep: a woman with red earrings and a north star print."
        self.assertEqual(st.clean_identity("Identity to keep: a man. Write at most 100 words, one time. Start with '"),
                         "Identity to keep: a man.")
        self.assertEqual(st.clean_identity(kept + " No animals present."), kept)
        leak = " (Note: the instruction says to avoid 'no glasses'. Let me rephrase."
        self.assertEqual(st.clean_identity(kept + leak), kept)  # reasoning in the answer goes
        self.assertEqual(st.clean_identity(kept + " 62 words."), kept)
        self.assertEqual(st.clean_identity("Identity to keep: rimless glasses, only light stubble on the jaw."),
                         "Identity to keep: thin dark-framed glasses, a visible short dark stubble beard on the jaw.")
        self.assertEqual(st.clean_identity("Identity to keep: a man. Behind her, a blurred brown-gray woodland."),
                         "Identity to keep: a man. Behind her, a brown-gray woodland.")

    def test_two_phase_store_and_reuse(self):
        store = self.out / "store"
        StubClient.wrapped = True
        code, stub, _ = run_script(self.base("--mode", "story", "--prompt-store", store, "--write-only", "--no-warm-up"))
        self.assertEqual(code, 0)
        self.assertEqual(len(stub.graphs), 1)
        write_graph = stub.graphs[0][0]
        self.assertNotIn(st.NODE_SAMPLE, write_graph)  # the write phase renders nothing
        stored = list(store.glob("*.txt"))
        self.assertEqual(len(stored), 1)
        noted = CLEAN_TEXT + "\n\n" + IDENTITY_TEXT  # the identity notes follow (softened by default)
        self.assertEqual(stored[0].read_text(encoding="utf-8"), noted)  # the JSON wrapper is gone
        code, stub, _ = run_script(self.base("--mode", "story", "--prompt-store", store, "--image-seeds", "2", "--no-warm-up",
                                             "--no-server-copy"))
        self.assertEqual(code, 0)
        self.assertEqual(len(stub.graphs), 2)  # no LLM job: the prompt comes from the store
        for graph, _ in stub.graphs:
            self.assertEqual(graph[st.NODE_ENCODE]["inputs"]["prompt"], noted)
            self.assertNotIn(st.NODE_ENH_GENERATE, graph)
            self.assertEqual(graph[st.NODE_SAVE]["class_type"], "PreviewImage")
        seeds = sorted(g[st.NODE_NOISE]["inputs"]["noise_seed"] for g, _ in stub.graphs)
        self.assertEqual(seeds, [1, 2])
        manifest = json.loads(sorted(self.out.glob("*.json"))[0].read_text(encoding="utf-8"))
        self.assertEqual(manifest["phases"], "two phases")
        self.assertTrue(manifest["prompt_key"])

    def test_leak_retry(self):
        StubClient.leak = True
        store = self.out / "store"
        code, stub, printed = run_script(self.base("--mode", "adapt", "--no-prompt-thinking", "--prompt-store", store,
                                                   "--no-warm-up"))
        self.assertEqual(code, 0)
        writes = [(g, front) for g, front in stub.graphs if st.NODE_ENH_GENERATE in g and st.NODE_SAMPLE not in g]
        self.assertEqual(len(writes), 2)
        self.assertTrue(writes[1][1])  # the retry goes to the front of the queue
        self.assertTrue(writes[1][0][st.NODE_ENH_GENERATE]["inputs"]["thinking"])
        self.assertTrue(any("retry-think" in p.name for p in self.out.glob("*.png")), printed)

    def test_no_leak_retry(self):
        StubClient.leak = True
        code, stub, _ = run_script(self.base("--mode", "adapt", "--no-prompt-thinking", "--no-leak-retry",
                                             "--prompt-store", self.out / "store", "--write-only",
                                             "--no-warm-up"))
        self.assertEqual(len(stub.graphs), 1)
        meta = json.loads(next((self.out / "store").glob("*.json")).read_text(encoding="utf-8"))
        self.assertTrue(meta["leak_reasons"])

    def test_warm_up_first(self):
        code, stub, _ = run_script(self.base("--mode", "adapt", "--prompt-store", self.out / "store", "--write-only"))
        self.assertEqual(len(stub.graphs), 2)
        first = stub.graphs[0][0][st.NODE_ENH_GENERATE]["inputs"]
        self.assertEqual(first["max_length"], st.WARM_UP_MAX_TOKENS)

    def test_cut_answer_retries_without_thinking(self):
        StubClient.cut = True
        code, stub, _ = run_script(self.base("--mode", "adapt", "--prompt-store", self.out / "store", "--write-only",
                                             "--no-warm-up"))
        self.assertEqual(code, 0)
        self.assertEqual(len(stub.graphs), 2)
        self.assertFalse(stub.graphs[1][0][st.NODE_ENH_GENERATE]["inputs"]["thinking"])
        meta = json.loads(next((self.out / "store").glob("*.json")).read_text(encoding="utf-8"))
        self.assertEqual(meta["retries"], [st.RETRY_NO_THINKING_TAG])

    def test_failed_write_has_no_retry(self):
        StubClient.fail = True
        code, stub, _ = run_script(self.base("--mode", "adapt", "--prompt-store", self.out / "store", "--write-only",
                                             "--no-warm-up"))
        self.assertEqual(code, 1)
        self.assertEqual(len(stub.graphs), 1)
        self.assertFalse(list((self.out / "store").glob("*.txt")))

    def test_store_write_once(self):
        store = self.out / "store"
        st.store_prompt(store, "k", "first", {"n": 1})
        st.store_prompt(store, "k", "second", {"n": 2})
        self.assertEqual((store / "k.txt").read_text(encoding="utf-8"), "first")


if __name__ == "__main__":
    unittest.main()
