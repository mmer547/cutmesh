"""The meshDict schema (cutmesh/schema.py) against the checks of config.validate_config:
the same keys, the same defaults and the same ranges; and the copies in docs/ up to date."""
import glob
import json
import os
import subprocess
import sys
import unittest

from cutmesh import config
from cutmesh.config import ConfigError, validate_config
from cutmesh.layers import DEFAULTS as LAYER_DEFAULTS
from cutmesh.mesher import DEFAULTS
from cutmesh.schema import meshdict_schema

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLE = os.path.join(ROOT, "examples", "stirred_tank")


def problems(**over):
    """the error lines validate_config gives for the defaults with `over` set"""
    cfg = dict(DEFAULTS)
    cfg.update(domain="tank.stl", baseCellSize=0.01, _base=EXAMPLE)
    cfg.update(over)
    try:
        validate_config(cfg)
    except ConfigError as e:
        return [ln.strip() for ln in str(e).splitlines()[1:]]
    return []


def bounds_cases(prop):
    """(value, accepted?) pairs at the bounds of a number / integer property"""
    integer = "integer" in (prop["type"] if isinstance(prop["type"], list) else [prop["type"]])
    out = []
    for kw, sign, closed in (("minimum", -1, True), ("exclusiveMinimum", -1, False),
                             ("maximum", 1, True), ("exclusiveMaximum", 1, False)):
        if kw not in prop:
            continue
        b = prop[kw]
        step = 1 if integer else max(abs(b), 1.0) * 1e-6
        out.append((b, closed))
        out.append((b + sign * step, False))
        out.append((b - sign * step, True))
    return out


class SchemaKeys(unittest.TestCase):
    def setUp(self):
        self.s = meshdict_schema()
        self.p = self.s["properties"]

    def test_top_level_keys(self):
        self.assertEqual(set(self.p), set(DEFAULTS) | config.OTHER_KEYS)

    def test_nested_keys(self):
        zone = self.p["rotatingZones"]["items"]["properties"]
        self.assertEqual(set(zone), config.ZONE_KEYS)
        self.assertEqual(set(zone["cylinder"]["properties"]), config.CYLINDER_KEYS)
        self.assertEqual(set(self.p["baseMesh"]["properties"]), config.BASEMESH_KEYS)
        self.assertEqual(set(self.p["addLayers"]["properties"]), set(LAYER_DEFAULTS))

    def test_defaults(self):
        for k, v in DEFAULTS.items():
            self.assertEqual(self.p[k]["default"], v, k)
        for k, v in LAYER_DEFAULTS.items():
            self.assertEqual(self.p["addLayers"]["properties"][k]["default"], v, k)

    def test_every_key_has_a_group_and_description(self):
        groups = {g["key"] for g in self.s["x-groups"]}
        for k, v in self.p.items():
            self.assertIn(v.get("x-group"), groups, k)
            self.assertTrue(v.get("description"), k)

    def test_examples_use_known_keys(self):
        for path in glob.glob(os.path.join(ROOT, "examples", "*", "meshDict*.json")):
            with open(path, encoding="utf-8") as f:
                cfg = json.load(f)
            for k in cfg:
                if not (k.startswith("_") or k.startswith("//")):
                    self.assertIn(k, self.p, f"{os.path.basename(path)}: {k}")


class SchemaRanges(unittest.TestCase):
    """values at the bounds of the schema are accepted / refused by validate_config alike"""

    def test_top_level_bounds(self):
        p = meshdict_schema()["properties"]
        checked = 0
        for k, prop in p.items():
            if k not in DEFAULTS:
                continue
            for v, ok in bounds_cases(prop):
                bad = [e for e in problems(**{k: v}) if e.startswith(k + " ")]
                self.assertEqual(not bad, ok, f"{k} = {v!r}: {bad}")
                checked += 1
        self.assertGreater(checked, 40)

    def test_base_cell_size(self):
        self.assertTrue(problems(baseCellSize=0))
        self.assertFalse(problems(baseCellSize=1e-6))

    def test_enums(self):
        p = meshdict_schema()["properties"]
        for k, prop in p.items():
            if "enum" not in prop:
                continue
            for v in prop["enum"]:
                self.assertEqual(problems(**{k: v}), [], f"{k} = {v!r}")
            self.assertTrue(problems(**{k: "notAValue"}), k)

    def test_booleans(self):
        p = meshdict_schema()["properties"]
        for k, prop in p.items():
            if prop["type"] == "boolean":
                self.assertEqual(problems(**{k: True}), [], k)
                self.assertTrue(problems(**{k: 1}), k)

    def test_null(self):
        """keys whose schema allows null are accepted as null (= off) by validate_config"""
        p = meshdict_schema()["properties"]
        nullable = [k for k, v in p.items() if "null" in v["type"] and k != "domain"]
        self.assertTrue({"addLayers", "baseMesh", "gapRefinement", "autoGapRefinement",
                         "featureLevel"} <= set(nullable))
        for k in nullable:
            self.assertEqual(problems(**{k: None}), [], k)
        zp = p["rotatingZones"]["items"]["properties"]
        cyl = {"point1": [0, 0, 0.06], "point2": [0, 0, 0.14], "radius": 0.07}
        for k, v in zp.items():
            if "null" in v["type"]:
                self.assertEqual(problems(rotatingZones=[{"name": "r", "cylinder": cyl, k: None}]), [], k)

    def test_zone_levels(self):
        zp = meshdict_schema()["properties"]["rotatingZones"]["items"]["properties"]
        cyl = {"point1": [0, 0, 0.06], "point2": [0, 0, 0.14], "radius": 0.07}
        for k in ("surfaceLevel", "volumeLevel"):
            for v, ok in bounds_cases(zp[k]):
                bad = problems(rotatingZones=[{"name": "r", "cylinder": cyl, k: v}])
                self.assertEqual(not bad, ok, f"{k} = {v}: {bad}")


class DocsCopies(unittest.TestCase):
    def test_docs_schema_up_to_date(self):
        r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "gen_meshdict_schema.py"),
                            "--check"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_schema_is_json(self):
        with open(os.path.join(ROOT, "docs", "meshdict.schema.json"), encoding="utf-8") as f:
            s = json.load(f)
        self.assertEqual(s["required"], ["domain", "baseCellSize"])


if __name__ == "__main__":
    unittest.main()
