"""The meshDict specification of cutmesh 1.0 is frozen (the manual's "meshDict compatibility"):
every key path of tests/data/meshdict_1.0.json must stay in the schema (cutmesh/schema.py)
with the same types (or more), the same allowed values (or more), bounds that are the same or
looser, no new restriction (an enum, a bound, a closed object) and no new required key.
Keys may be added and restrictions dropped.  A top-level key leaves only through
config.DEPRECATED_KEYS.  Paths: "a/b" a property, "a/[]" the items of an array, "a/{}" the
values of a map (additionalProperties), "a/{pattern}" those of patternProperties,
"a/oneOf0" an alternative.

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


def _bound(node, incl, excl):
    """the tighter of an inclusive and an exclusive bound -> [value, exclusive] or None"""
    b = [(node[incl], False)] if incl in node else []
    b += [(node[excl], True)] if excl in node else []
    if not b:
        return None
    lower = incl.startswith("min")
    return list(max(b) if lower else min(b, key=lambda x: (x[0], not x[1])))


def spec(s=None):
    """{key path: {"type": [...], "enum"?, "lower"?, "upper"?, "minItems"?, "maxItems"?,
    "minLength"?, "closed"?, "required"?}} of the meshDict schema"""
    s = s or schema.meshdict_schema()
    out = {}

    def walk(node, path):
        if not isinstance(node, dict):
            return
        e = {"type": _types(node)}
        if "enum" in node:
            e["enum"] = sorted(node["enum"], key=json.dumps)
        lo = _bound(node, "minimum", "exclusiveMinimum")
        hi = _bound(node, "maximum", "exclusiveMaximum")
        if lo:
            e["lower"] = lo
        if hi:
            e["upper"] = hi
        for k in ("minItems", "maxItems", "minLength"):
            if k in node:
                e[k] = node[k]
        if node.get("additionalProperties") is False:
            e["closed"] = True
        if node.get("required"):
            e["required"] = sorted(node["required"])
        out["/".join(path) or "."] = e
        for k, v in node.get("properties", {}).items():
            walk(v, path + [k])
        if isinstance(node.get("items"), dict):
            walk(node["items"], path + ["[]"])
        if isinstance(node.get("additionalProperties"), dict):
            walk(node["additionalProperties"], path + ["{}"])
        for pat, v in node.get("patternProperties", {}).items():
            walk(v, path + ["{" + pat + "}"])
        for key in ("oneOf", "anyOf", "allOf"):
            for i, alt in enumerate(node.get(key, [])):
                walk(alt, path + [f"{key}{i}"])
    walk(s, [])
    return out


def narrowed(old, new):
    """how the entry new restricts more than old (a list of reasons; empty: compatible)"""
    why = []
    if not set(old["type"]) <= set(new["type"]):
        why.append(f"type {old['type']} -> {new['type']}")
    if "enum" in new:
        if "enum" not in old:
            why.append(f"new allowed values {new['enum']}")
        else:
            missing = [v for v in old["enum"] if v not in new["enum"]]
            if missing:
                why.append(f"allowed values removed: {missing}")
    for k, looser in (("lower", lambda o, n: n[0] < o[0] or (n[0] == o[0] and (o[1] or not n[1]))),
                      ("upper", lambda o, n: n[0] > o[0] or (n[0] == o[0] and (o[1] or not n[1]))),
                      ("minItems", lambda o, n: n <= o), ("minLength", lambda o, n: n <= o),
                      ("maxItems", lambda o, n: n >= o)):
        if k in new and (k not in old or not looser(old[k], new[k])):
            why.append(f"{k} {old.get(k)} -> {new[k]}")
    if new.get("closed") and not old.get("closed"):
        why.append("no longer takes other keys")
    req = set(new.get("required", [])) - set(old.get("required", []))
    if req:
        why.append(f"newly required keys {sorted(req)}")
    return why


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
            why = narrowed(e, now[path])
            self.assertFalse(why, f"{path}: " + "; ".join(why))

    def test_the_check_sees_narrowing(self):
        e = {"type": ["integer"], "lower": [0, False], "maxItems": 3}
        self.assertFalse(narrowed(e, {"type": ["integer", "string"], "lower": [-1, False]}))
        self.assertFalse(narrowed(e, {"type": ["integer"], "lower": [0, False], "maxItems": 3}))
        for n in ({"type": ["string"]},
                  {"type": ["integer"], "enum": [1]},
                  {"type": ["integer"], "lower": [0, True]},
                  {"type": ["integer"], "lower": [1, False]},
                  {"type": ["integer"], "upper": [9, False]},
                  {"type": ["integer"], "maxItems": 2},
                  {"type": ["integer"], "closed": True},
                  {"type": ["integer"], "required": ["x"]}):
            self.assertTrue(narrowed(e, n), n)
        self.assertTrue(narrowed({"type": ["string"], "enum": ["a", "b"]}, {"type": ["string"], "enum": ["a"]}))
        self.assertFalse(narrowed({"type": ["string"], "enum": ["a"]}, {"type": ["string"]}))

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
