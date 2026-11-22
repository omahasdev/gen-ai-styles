"""Check an API graph against the node definitions of a running ComfyUI server, without running it.

check(graph, object_info) returns a list of problems:
- an unknown node class
- a required input that is missing (dynamic combos: the chosen option and its sub-inputs; autogrow: at least
  one item)
- a link to a missing node or to an output index that the node does not have
- a link whose output type does not match the input type ("*" and COMBO inputs accept anything)
Get the node definitions once with fetch_object_info(server) or load_object_info(file).
"""
import json
import urllib.request

ANY = "*"


def fetch_object_info(server: str) -> dict:
    """The node definitions of a running server (metadata only, no model runs)."""
    with urllib.request.urlopen(f"{server}/object_info", timeout=60) as r:
        return json.load(r)


def load_object_info(path: str) -> dict:
    """Node definitions saved from GET /object_info, so a check needs no server."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _input_type(spec):
    t = spec[0]
    return "COMBO" if isinstance(t, list) else t


def _is_link(value):
    return isinstance(value, list) and len(value) == 2 and isinstance(value[0], str) and isinstance(value[1], int)


def check(graph: dict, info: dict) -> list[str]:
    problems = []
    for nid, node in graph.items():
        cls = node.get("class_type")
        if cls not in info:
            problems.append(f"#{nid}: unknown class {cls}")
            continue
        spec = info[cls]["input"]
        inputs = node.get("inputs", {})
        expected = {}
        for name, s in spec.get("required", {}).items():
            t = _input_type(s)
            if t == "COMFY_DYNAMICCOMBO_V3":
                choice = inputs.get(name)
                options = {o["key"]: o["inputs"] for o in s[1]["options"]}
                if choice not in options:
                    problems.append(f"#{nid} {cls}: {name}={choice!r} is not one of {list(options)}")
                    continue
                for sub, subspec in options[choice].get("required", {}).items():
                    if f"{name}.{sub}" not in inputs:
                        problems.append(f"#{nid} {cls}: missing {name}.{sub}")
                    expected[f"{name}.{sub}"] = _input_type(subspec)
            elif t == "COMFY_AUTOGROW_V3":
                if not any(k.startswith(f"{name}.") for k in inputs):
                    problems.append(f"#{nid} {cls}: autogrow input {name} has no item")
            else:
                if name not in inputs:
                    problems.append(f"#{nid} {cls}: missing required input {name}")
                expected[name] = t
        for name, s in spec.get("optional", {}).items():
            t = _input_type(s)
            if t == "COMFY_AUTOGROW_V3":
                item = s[1]["template"]["input"]
                item_type = [_input_type(v) for part in item.values() for v in part.values()]
                for k in inputs:
                    if k.startswith(f"{name}."):
                        expected[k] = item_type[0] if item_type else ANY
            else:
                expected[name] = t
        for name, value in inputs.items():
            if not _is_link(value):
                continue
            src, slot = value
            if src not in graph:
                problems.append(f"#{nid} {cls}.{name}: link to missing node #{src}")
                continue
            src_cls = graph[src].get("class_type")
            outputs = info.get(src_cls, {}).get("output", [])
            if slot >= len(outputs):
                problems.append(f"#{nid} {cls}.{name}: #{src} {src_cls} has no output {slot}")
                continue
            want, have = expected.get(name, ANY), outputs[slot]
            have = "COMBO" if isinstance(have, list) else have
            if want not in (ANY, "COMBO") and have != ANY and want != have and have not in want.split(","):
                problems.append(f"#{nid} {cls}.{name}: type {have} from #{src} {src_cls}, expected {want}")
    return problems
