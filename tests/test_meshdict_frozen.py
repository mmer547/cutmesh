"""The meshDict specification of cutmesh 1.0 is frozen (the manual's "meshDict compatibility"):
every key path of tests/data/meshdict_1.0.json must stay in the schema (cutmesh/schema.py)
with the same types (or more), the same allowed values (or more), bounds that are the same or
looser, no new restriction (an enum, a bound, a closed object, any other validation
keyword) and no new required key.  Restrictions may be dropped.  A new sub-schema counts as
a restriction unless it surely only adds: a property of a closed object (a new key), a map
value schema on a closed object, another anyOf alternative.  So a new key in a map, a new
allOf branch, oneOf alternative or patternProperties entry fails (conservatively).  A top-level key leaves only through
config.DEPRECATED_KEYS.  Paths: "a/b" a property, "a/[]" the items of an array, "a/{}" the
values of a map (additionalProperties), "a/{pattern}" those of patternProperties,
"a/oneOf0" an alternative.

    python tests/test_meshdict_frozen.py --write    # write the snapshot (only at a major version)
"""
import json
import os
import re
import sys
import unittest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

from cutmesh import config, schema  # noqa: E402

SNAPSHOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "meshdict_1.0.json")


# validation keywords the comparison does not interpret: a new one, or a changed value, counts
# as a narrowing; dropping one is allowed
OTHER = ("const", "pattern", "format", "multipleOf", "maxLength", "uniqueItems", "minProperties",
         "maxProperties", "propertyNames", "dependentRequired", "dependentSchemas", "contains",
         "prefixItems", "not", "if", "then", "else", "$ref")


def _types(node):
    t = node.get("type")
    return sorted(t) if isinstance(t, list) else ([t] if t else [])


def _accepts(old, new):
    """do the JSON types new accept every value the types old accept?  [] = any type;
    "number" includes "integer"
    """
    if not new:
        return True
    if not old:
        return False
    return all(t in new or (t == "integer" and "number" in new) for t in old)


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
        other = {k: node[k] for k in OTHER if k in node}
        if other:
            e["other"] = other
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
    if not _accepts(old["type"], new["type"]):
        why.append(f"type {old['type'] or 'any'} -> {new['type'] or 'any'}")
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
    for k, v in new.get("other", {}).items():
        if k not in old.get("other", {}) or old["other"][k] != v:
            why.append(f"{k} {old.get('other', {}).get(k)} -> {v}")
    return why


COMBINATOR = re.compile(r"(allOf|anyOf|oneOf)\d+")


def problems(frozen, now):
    """the incompatibilities of the spec now with the frozen spec: [(path, reason)]"""
    out = []
    for path, e in frozen.items():
        if path in now:
            out += [(path, w) for w in narrowed(e, now[path])]
            continue
        if re.search(r"(^|/)allOf\d+(/|$)", path):
            continue                                  # a dropped allOf branch only broadens
        if path.split("/")[0] not in config.DEPRECATED_KEYS:
            out.append((path, "left the meshDict without going through config.DEPRECATED_KEYS"))
    for path in now:
        if path in frozen or path == ".":
            continue
        parent, _, last = path.rpartition("/")
        parent = parent or "."
        if parent not in frozen:
            continue                                  # under a new sub-schema: judged there
        closed = frozen[parent].get("closed", False)
        if not COMBINATOR.fullmatch(last) and last not in ("[]", "{}") and not last.startswith("{"):
            if not closed:
                out.append((path, "a new property of an open object restricts its values"))
        elif last == "{}":
            if not closed:
                out.append((path, "a new map value schema restricts the values"))
        elif last.startswith("anyOf"):
            prefix = "" if parent == "." else parent + "/"
            if not any(re.fullmatch(re.escape(prefix) + r"anyOf\d+", q) for q in frozen):
                out.append((path, "a first anyOf restricts the values"))
        else:                                         # allOf, oneOf, patternProperties, items
            out.append((path, f"a new {last} restricts the values"))
    return out


class FrozenMeshDict(unittest.TestCase):
    def test_nothing_removed_or_narrowed(self):
        with open(SNAPSHOT, encoding="utf-8") as f:
            frozen = json.load(f)["keys"]
        bad = problems(frozen, spec())
        self.assertFalse(bad, "\n".join(f"{p}: {w}" for p, w in bad))

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
        # types: number includes integer, no type accepts anything
        self.assertFalse(narrowed({"type": ["integer"]}, {"type": ["number"]}))
        self.assertFalse(narrowed({"type": ["integer"]}, {"type": []}))
        self.assertTrue(narrowed({"type": []}, {"type": ["string"]}))
        self.assertTrue(narrowed({"type": ["number"]}, {"type": ["integer"]}))
        self.assertTrue(narrowed({"type": ["string"]}, {"type": ["string"], "other": {"pattern": "^a"}}))

    def test_anyof_alternatives(self):
        old = {"type": "object", "additionalProperties": False,
               "properties": {"a": {"anyOf": [{"type": "string"}]}, "b": {"type": "string"}}}
        new = json.loads(json.dumps(old))
        new["properties"]["a"]["anyOf"].append({"type": "null"})
        self.assertFalse(problems(spec(old), spec(new)))
        new["properties"]["b"]["anyOf"] = [{"type": "string"}]
        self.assertTrue(problems(spec(old), spec(new)))

    def test_changes_to_the_schema(self):
        """whole-schema changes: (change, narrows?)"""
        def add_allof(s):
            s["properties"]["case"]["allOf"] = [{"enum": ["."]}]

        def add_oneof(s):
            s["properties"]["case"]["oneOf"] = [{"type": "string"}, {"type": "null"}]

        def more_alternatives(s):    # a zone defined by a cylinder now matches two branches
            s["properties"]["rotatingZones"]["items"]["oneOf"].append({"required": ["name"]})

        def map_pattern(s):          # {"rotor": 2} must now also be a string
            s["properties"]["surfaceRefinement"]["patternProperties"] = {"^rotor$": {"type": "string"}}

        def map_property(s):         # the same through properties
            s["properties"]["surfaceRefinement"]["properties"] = {"rotor": {"type": "string"}}

        def zone_key(s):             # a new key of a closed object
            s["properties"]["rotatingZones"]["items"]["properties"]["newKey"] = {"type": "number"}

        def pattern(s):
            s["properties"]["case"]["pattern"] = "^[a-z]+$"

        def comment_values(s):
            s["patternProperties"]["^(_|//)"]["type"] = "string"

        def map_values(s):
            s["properties"]["surfaceRefinement"]["additionalProperties"]["type"] = "string"

        def new_key(s):
            s["properties"]["newKey"] = {"type": "boolean", "default": False}

        def to_number(s):
            s["properties"]["surfaceRefinement"]["additionalProperties"]["type"] = "number"

        with open(SNAPSHOT, encoding="utf-8") as f:
            frozen = json.load(f)["keys"]
        for change, narrows in ((add_allof, True), (add_oneof, True), (pattern, True),
                                (comment_values, True), (map_values, True),
                                (more_alternatives, True), (map_pattern, True), (map_property, True),
                                (new_key, False), (zone_key, False), (to_number, False)):
            s = schema.meshdict_schema()
            change(s)
            self.assertEqual(bool(problems(frozen, spec(s))), narrows, change.__name__)

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
