"""LoRA routing: the registry chooses LoRAs from the words of a style; no style file names a LoRA."""
import json
import re
import sys
import unittest
from pathlib import Path

TOOL = Path(__file__).resolve().parents[1]
ROOT = TOOL.parents[1]
sys.path.insert(0, str(TOOL))
import lora_routing  # noqa: E402

REGISTRY = TOOL / "loras.json"
GRAPH = TOOL / "style-transfer.api.json"
STYLES = sorted((ROOT / "image-styles").glob("*.json"))
QWEN = "qwen_image_2.1_turbo_int8_convrot.safetensors"
ENTRY = {"file": "test.safetensors", "model": "qwen_image", "scope": "generic", "strength": 0.5, "trigger": "trig",
         "when": {"paths": ["rendering.medium"], "pattern": "pencil"}}


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


class LoraRoutingTest(unittest.TestCase):
    def test_selection_by_style_words_and_model(self):
        style = {"rendering": {"medium": "Soft pencil drawing"}}
        self.assertEqual(lora_routing.select(style, [ENTRY], QWEN), [ENTRY])
        self.assertEqual(lora_routing.select(style, [ENTRY], "flux-2-klein.safetensors"), [])
        self.assertEqual(lora_routing.select({"rendering": {"medium": "oil paint"}}, [ENTRY], QWEN), [])
        self.assertEqual(lora_routing.select({}, [ENTRY], QWEN), [])

    def test_only_generic_loras_route(self):
        # A LoRA of one concrete work never routes automatically (user, 2026-11-19); only a generic one does.
        style = {"rendering": {"medium": "Soft pencil drawing"}}
        for scope in (None, "work", "style"):
            entry = {k: v for k, v in ENTRY.items() if k != "scope"} | ({"scope": scope} if scope else {})
            self.assertEqual(lora_routing.select(style, [entry], QWEN), [], scope)

    def test_autoroute_false_keeps_an_entry_out_of_the_routing(self):
        style = {"rendering": {"medium": "Soft pencil drawing"}}
        self.assertEqual(lora_routing.select(style, [ENTRY | {"autoroute": False}], QWEN), [])

    def test_registry_entries_are_generic_and_never_key_on_a_style_name(self):
        styles = [load(p) for p in STYLES]
        for entry in lora_routing.load_registry(REGISTRY):
            self.assertEqual(entry.get("scope"), lora_routing.GENERIC, entry["file"])
            for s in styles:  # the pattern describes a kind of look, never the name of one style
                self.assertIsNone(re.search(entry["when"]["pattern"], s["style_name"], re.I), (entry["file"], s["style_name"]))

    def test_extra_lora_by_hand(self):
        # A LoRA chosen by hand for a run (a private preview), beside the automatic routing.
        self.assertEqual(lora_routing.parse_extra("x.safetensors"),
                         {"file": "x.safetensors", "strength": lora_routing.DEFAULT_STRENGTH, "manual": True})
        entry = lora_routing.parse_extra("x.safetensors:0.8:mystyle style")
        self.assertEqual((entry["strength"], entry["trigger"]), (0.8, "mystyle style"))
        self.assertTrue(lora_routing.describe([entry])[0]["manual"])

    def test_apply_inserts_loader_and_trigger(self):
        graph = load(GRAPH)
        users_before = [nid for nid, n in graph.items() if ["1", 0] in n.get("inputs", {}).values()]
        graph["5"]["inputs"]["prompt"] = "Redraw the photo."
        lora_routing.apply(graph, [ENTRY], "1", "5")
        loader = graph[str(lora_routing.FIRST_LORA_NODE_ID)]
        self.assertEqual(loader["inputs"]["model"], ["1", 0])
        self.assertEqual(loader["inputs"]["strength_model"], ENTRY["strength"])
        for nid in users_before:
            self.assertNotIn(["1", 0], graph[nid]["inputs"].values(), nid)
        self.assertTrue(graph["5"]["inputs"]["prompt"].startswith("trig. Redraw"))

    def test_apply_keeps_the_nodes_of_the_graph(self):
        # The identity notes of style_transfer.py add RegexReplace nodes from id 200: a loader must not replace one.
        graph = load(GRAPH)
        taken = str(lora_routing.FIRST_LORA_NODE_ID)
        graph[taken] = {"class_type": "RegexReplace", "inputs": {}}
        lora_routing.apply(graph, [ENTRY, dict(ENTRY, file="second.safetensors")], "1", "5")
        self.assertEqual(graph[taken]["class_type"], "RegexReplace")
        loaders = [n for n in graph.values() if n["class_type"] == lora_routing.LORA_NODE]
        self.assertEqual([n["inputs"]["lora_name"] for n in loaders], [ENTRY["file"], "second.safetensors"])

    def test_apply_with_a_linked_prompt_uses_a_concat_node(self):
        graph = load(GRAPH)
        graph["5"]["inputs"]["prompt"] = ["102", 0]
        lora_routing.apply(graph, [ENTRY], "1", "5")
        self.assertEqual(graph["5"]["inputs"]["prompt"], [lora_routing.TRIGGER_NODE_ID, 0])
        self.assertEqual(graph[lora_routing.TRIGGER_NODE_ID]["inputs"]["string_b"], ["102", 0])

    def test_no_entries_leave_the_graph_unchanged(self):
        graph = load(GRAPH)
        before = json.dumps(graph, sort_keys=True)
        lora_routing.apply(graph, [], "1", "5")
        self.assertEqual(json.dumps(graph, sort_keys=True), before)

    def test_references_follow_the_source_image(self):
        graph = load(GRAPH)
        graph["5"]["inputs"]["prompt"] = "Redraw <image1>."
        entry = {"images": ["a.jpg", "b.jpg"], "note": "{tags} show the style."}
        paths = lora_routing.apply_references(graph, [entry], "5", lambda p: "up-" + p.name)
        enc = graph["5"]["inputs"]
        self.assertEqual(paths, ["a.jpg", "b.jpg"])
        self.assertEqual(graph[enc["images.image_2"][0]]["inputs"]["image"], "up-a.jpg")
        self.assertIn("images.image_3", enc)
        self.assertTrue(enc["prompt"].endswith("<image2> and <image3> show the style."))
        self.assertEqual(enc["images.image_1"], ["4", 0])  # the source stays image 1

    def test_missing_registry_means_no_lora(self):
        self.assertEqual(lora_routing.load_registry(ROOT / "no-such-registry.json"), [])

    def test_registry_matches_only_the_intended_styles(self):
        # Every registry entry must reach at least one style and never every style: a pattern that matches all
        # styles is a word that every style uses, not a style trait.
        registry = lora_routing.load_registry(REGISTRY)
        styles = [load(p) for p in STYLES]
        for entry in registry:
            pattern_only = {k: v for k, v in entry.items() if k != "autoroute"}  # the pattern, also of a switched-off entry
            hits = [s["style_name"] for s in styles if lora_routing.select(s, [pattern_only], QWEN)]
            self.assertTrue(hits, entry["file"])
            self.assertLess(len(hits), len(styles), entry["file"])

    def test_no_style_file_names_a_lora(self):
        for p in STYLES:
            text = p.read_text(encoding="utf-8").lower()
            self.assertIsNone(re.search(r"\blora\b", text), p.name)
            self.assertNotIn(".safetensors", text, p.name)


if __name__ == "__main__":
    unittest.main()
