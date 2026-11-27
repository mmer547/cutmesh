"""The meshDict specification of cutmesh 1.0 is frozen (the manual's "meshDict compatibility"):
every key path of tests/data/meshdict_1.0.json must stay in the schema (cutmesh/schema.py)
with the same types (or more), the same allowed values (or more), and no new required key.
Keys may be added.  A top-level key leaves only through config.DEPRECATED_KEYS.

    python tests/test_meshdict_frozen.py --write    # write the snapshot (only at a major version)
"""
import json
import os
import sys
import unittest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

from cutmesh import config, schema  # noqa: E402

SNAPSHOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "meshdict_1.0.json")


def _types(node):
    t = node.get("type")
    return sorted(t) if isinstance(t, list) else ([t] if t else [])


def spec(s=None):
    """{key path: {"type": [...], "enum": [...]?, "required": [...]?}} of the meshDict schema"""
    s = s or schema.meshdict_schema()
    out = {}

    def walk(node, path):
        if not isinstance(node, dict):
            return
        e = {"type": _types(node)}
        if "enum" in node:
            e["enum"] = sorted(node["enum"], key=json.dumps)
        if node.get("required"):
            e["required"] = sorted(node["required"])
        out["/".join(path) or "."] = e
        for k, v in node.get("properties", {}).items():
            walk(v, path + [k])
        if isinstance(node.get("items"), dict):
            walk(node["items"], path + ["[]"])
        for key in ("oneOf", "anyOf", "allOf"):
            for i, alt in enumerate(node.get(key, [])):
                walk(alt, path + [f"{key}{i}"])
    walk(s, [])
    return out


class FrozenMeshDict(unittest.TestCase):
    def test_nothing_removed_or_narrowed(self):
        with open(SNAPSHOT, encoding="utf-8") as f:
            frozen = json.load(f)["keys"]
        now = spec()
        for path, e in frozen.items():
            top = path.split("/")[0]
            if path not in now:
                self.assertIn(top, config.DEPRECATED_KEYS,
                              f"{path} left the meshDict without going through config.DEPRECATED_KEYS")
                continue
            n = now[path]
            self.assertTrue(set(e["type"]) <= set(n["type"]), f"{path}: type {e['type']} -> {n['type']}")
            if "enum" in e:
                self.assertIn("enum", n, f"{path}: allowed values now unrestricted? {n}")
                missing = [v for v in e["enum"] if v not in n["enum"]]
                self.assertFalse(missing, f"{path}: allowed values removed: {missing}")
            new_req = set(n.get("required", [])) - set(e.get("required", []))
            self.assertFalse(new_req, f"{path}: newly required keys {sorted(new_req)}")

    def test_snapshot_is_of_the_current_major_version(self):
        from cutmesh import __version__
        with open(SNAPSHOT, encoding="utf-8") as f:
            v = json.load(f)["version"]
        self.assertEqual(v.split(".")[0], __version__.split(".")[0])


if __name__ == "__main__":
    if "--write" in sys.argv:
        from cutmesh import __version__
        with open(SNAPSHOT, "w", encoding="utf-8") as f:
            json.dump({"version": __version__, "keys": spec()}, f, indent=1, ensure_ascii=False)
            f.write("\n")
        print(f"wrote {SNAPSHOT}")
    else:
        unittest.main()
