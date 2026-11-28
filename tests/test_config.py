"""Input checks before meshing: unreadable STL files (geometry.read_stl), meshDict settings
(config.validate_config / check_names), overlapping rotating zones and the command line."""
import io
import json
import os
import struct
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

import numpy as np

from cutmesh.config import ConfigError, check_names, validate_config
from cutmesh.geometry import STLError, read_stl, write_stl_ascii, Cylinder
from cutmesh.mesher import load_config, check_zone_overlap
from cutmesh.__main__ import main as cli


def box():
    v = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], float)
    f = [[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
         [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]]
    return v[np.array(f)]


class TempDir(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.d = self._td.name

    def tearDown(self):
        self._td.cleanup()

    def write(self, name, data):
        p = os.path.join(self.d, name)
        with open(p, "wb" if isinstance(data, bytes) else "w") as f:
            f.write(data)
        return p


class TestReadSTL(TempDir):
    def assertSTLError(self, data, text):
        p = self.write("x.stl", data)
        with self.assertRaises(STLError) as cm:
            read_stl(p)
        self.assertIn(text, str(cm.exception))
        self.assertIn(p, str(cm.exception))

    def test_empty(self):
        self.assertSTLError(b"", "empty")
        self.assertSTLError("  \n", "empty")

    def test_short_binary(self):
        self.assertSTLError(b"\x01" * 40, "84-byte header")

    def test_truncated_binary(self):
        self.assertSTLError(b"\x00" * 80 + struct.pack("<I", 10) + b"\x00" * 100, "truncated")

    def test_binary_without_triangles(self):
        self.assertSTLError(b"\x00" * 80 + struct.pack("<I", 0), "no triangles")

    def test_ascii_without_triangles(self):
        self.assertSTLError("solid a\nendsolid a\n", "no triangles")

    def test_ascii_bad_vertex(self):
        self.assertSTLError("solid a\nfacet normal 0 0 1\nouter loop\nvertex 0 0\n", ":4: bad vertex")

    def test_ascii_vertex_count(self):
        self.assertSTLError("solid a\nvertex 0 0 0\nvertex 1 0 0\nendsolid a\n", "not a multiple of 3")

    def test_non_finite(self):
        t = box()
        t[0, 0, 0] = np.nan
        p = os.path.join(self.d, "nan.stl")
        write_stl_ascii(p, [("walls", t)])
        with self.assertRaises(STLError) as cm:
            read_stl(p)
        self.assertIn("non-finite", str(cm.exception))

    def test_good_files_still_read(self):
        p = os.path.join(self.d, "a.stl")
        write_stl_ascii(p, [("a", box()), ("b", box() + 2)])
        s = read_stl(p)
        self.assertEqual([n for n, _ in s], ["a", "b"])
        t = box().astype("<f4")
        rec = b"".join(b"\x00" * 12 + tri.tobytes() + b"\x00\x00" for tri in t)
        q = self.write("b.stl", b"solid bin" + b"\x00" * 71 + struct.pack("<I", len(t)) + rec)
        (name, tris), = read_stl(q)
        self.assertEqual(name, "bin")
        np.testing.assert_allclose(tris, box())


class SchemaAgreement(TempDir):
    """validate_config accepts what the JSON schema (cutmesh/schema.py, the specification)
    accepts, and rejects what it rejects: boundary and wrong-type values for every key of the
    top level and of the nested objects, each in an otherwise minimal meshDict.  The checker
    is stricter only where a value the schema allows cannot work (STRICTER)."""
    ZONE = {"name": "r", "cylinder": {"point1": [0, 0, 0.2], "point2": [0, 0, 0.4], "radius": 0.1}}
    PLACES = {
        "": lambda d: d,
        "addLayers": lambda d: d.setdefault("addLayers", {}),
        "gapRefinement": lambda d: d.setdefault("gapRefinement", {"level": 3, "thickness": 0.004}),
        "autoGapRefinement": lambda d: d.setdefault("autoGapRefinement", {}),
        "baseMesh": lambda d: d.setdefault("baseMesh", {"type": "cylindrical"}),
        "rotatingZones/[]": lambda d: d.setdefault("rotatingZones", [json.loads(json.dumps(SchemaAgreement.ZONE))])[0],
        "rotatingZones/[]/cylinder": lambda d: SchemaAgreement.PLACES["rotatingZones/[]"](d)["cylinder"],
        "rotatingZones/[]/wallProximity": lambda d: SchemaAgreement.PLACES["rotatingZones/[]"](d).setdefault(
            "wallProximity", {"distance": 0.01, "level": 3}),
        "refinementRegions/[]": lambda d: d.setdefault("refinementRegions", [
            {"cylinder": {"point1": [0, 0, 0], "point2": [0, 0, 0.5], "radius": 0.2}, "level": 2}])[0],
    }
    # (place/key, value) the checker rejects on purpose: no direction, an empty range, a
    # size ratio never met (cells halve), blocks without a core
    STRICTER = {("baseMesh/axis", "[0.0, 0.0, 0.0]"), ("rotatingZones/[]/axis", "[0.0, 0.0, 0.0]"),
                ("baseMesh/zRange", "[0.0, 0.0]"), ("baseMesh/zRange", "[1.0, 1.0]"),
                ("baseMesh/zRange", "[-1.0, -1.0]"), ("baseMesh/maxSizeRatio", "1"),
                ("baseMesh/maxSizeRatio", "1e-09"), ("baseMesh/blocks", "[]")}

    @staticmethod
    def values(p):
        t = p.get("type")
        ts = t if isinstance(t, list) else [t]
        out = [None, True, "x", [], {}, -1, 0, 1, 2.5, 1e9, -1e9]
        for k in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"):
            if k in p:
                out += [p[k], p[k] - 1e-9, p[k] + 1e-9, p[k] - 1, p[k] + 1]
        out += p.get("enum", [])
        if "integer" in ts:
            out += [3, 3.0, 3.5]
        if "array" in ts:
            n = p.get("minItems", 3)
            out += [[0.0] * n, [1.0] * n, [0.0] * (n + 1), [0.0] * max(n - 1, 0), [-1.0] * n, ["a"] * n]
        return out

    def test_schema_and_checker_agree(self):
        try:
            import jsonschema
        except ImportError:
            self.skipTest("jsonschema is not installed")
        from cutmesh.mesher import DEFAULTS
        from cutmesh.schema import meshdict_schema
        S = meshdict_schema("en")
        V = jsonschema.Draft202012Validator(S)
        write_stl_ascii(os.path.join(self.d, "box.stl"), [("walls", box())])
        base = {"domain": "box.stl", "baseCellSize": 0.1}

        def node(path):
            n = S
            for seg in [q for q in path.split("/") if q]:
                n = n["items"] if seg == "[]" else n["properties"][seg]
            return n

        bad = []
        for place, get in self.PLACES.items():
            for key, p in node(place).get("properties", {}).items():
                if p.get("type") == "object" and "properties" in p:
                    continue                         # its keys are checked at their own place
                for v in self.values(p):
                    if place == "" and key in ("domain", "case") and isinstance(v, str):
                        continue                     # file and directory names: tested elsewhere
                    d = json.loads(json.dumps(base))
                    get(d)[key] = v
                    s_ok = V.is_valid(d)
                    cfg = dict(DEFAULTS, **json.loads(json.dumps(d)), _base=self.d)
                    try:
                        validate_config(cfg, None)
                        c_ok = True
                    except ConfigError:
                        c_ok = False
                    name = (place + "/" if place else "") + key
                    if s_ok != c_ok and not (s_ok and (name, repr(v)) in self.STRICTER):
                        bad.append(f"{name} = {v!r}: schema {'accepts' if s_ok else 'rejects'}, "
                                   f"checker {'accepts' if c_ok else 'rejects'}")
        self.assertFalse(bad, "\n".join(bad))

    def test_integer_valued_floats_become_int(self):
        cfg = dict(__import__("cutmesh.mesher", fromlist=["DEFAULTS"]).DEFAULTS,
                   domain="box.stl", baseCellSize=0.1, padding=2.0, surfaceRefinement={"walls": 2.0},
                   addLayers={"nLayers": 3.0}, _base=self.d)
        write_stl_ascii(os.path.join(self.d, "box.stl"), [("walls", box())])
        validate_config(cfg, None)
        for v in (cfg["padding"], cfg["surfaceRefinement"]["walls"], cfg["addLayers"]["nLayers"]):
            self.assertIs(type(v), int)
        self.assertIs(type(cfg["baseCellSize"]), float)


class SizeCheck(TempDir):
    """settings asking for more cells than the machine can hold stop before meshing"""

    def test_check_size(self):
        from cutmesh.mesher import _check_size
        msgs = []
        _check_size(1e5, msgs.append, memory=(16 * 2 ** 30, 8 * 2 ** 30))          # fits
        self.assertEqual(msgs, [])
        _check_size(1e7, msgs.append, memory=(16 * 2 ** 30, 4 * 2 ** 30))          # more than is free
        self.assertTrue(msgs and msgs[0].startswith("WARNING: the settings ask for at least 1e+07 cells"))
        with self.assertRaises(ConfigError) as cm:
            _check_size(1e8, msgs.append, memory=(16 * 2 ** 30, 8 * 2 ** 30))       # more than the machine
        self.assertIn("this machine has 16 GB", str(cm.exception))
        with self.assertRaises(ConfigError):
            _check_size(1e11, msgs.append, memory=(None, None))                    # memory unknown
        msgs.clear()
        _check_size(1e6, msgs.append, memory=(2 * 2 ** 30, 0))                      # nothing left
        self.assertTrue(msgs and "0 GB is free now" in msgs[0], msgs)

    def test_cgroup_memory(self):
        """the tightest limit over the process's own cgroup and its ancestors, with that
        group's usage (cgroup v2 and v1; the group from /proc/self/cgroup)"""
        from cutmesh.mesher import _cgroup_memory
        none = os.path.join(self.d, "noproc")
        os.makedirs(os.path.join(self.d, "v2"))
        self.write("v2/memory.max", "2147483648\n")
        self.write("v2/memory.current", "1932735283\n")              # 1.8 GiB used
        self.assertEqual(_cgroup_memory(os.path.join(self.d, "v2"), none), (2147483648, 2147483648 - 1932735283))
        # a limit on an ancestor of the process's group, none at the root or on the group
        r = os.path.join(self.d, "tree")
        for d in ("", "a", "a/b"):
            os.makedirs(os.path.join(r, d), exist_ok=True)
            self.write(os.path.join("tree", d, "memory.max"), "max\n")
        self.write("tree/a/memory.max", "1073741824\n")
        self.write("tree/a/memory.current", "123\n")
        proc = self.write("cg2", "0::/a/b\n")
        self.assertEqual(_cgroup_memory(r, proc), (1073741824, 1073741824 - 123))
        # a parent filled by other processes leaves less than the child's own limit
        self.write("tree/a/b/memory.max", "536870912\n")
        self.write("tree/a/b/memory.current", "0\n")
        self.write("tree/a/memory.current", "1000000000\n")
        self.assertEqual(_cgroup_memory(r, proc), (536870912, 1073741824 - 1000000000))
        # v1: the memory controller's line
        os.makedirs(os.path.join(self.d, "v1", "memory", "x", "y"))
        self.write("v1/memory/memory.limit_in_bytes", "9223372036854771712\n")
        self.write("v1/memory/x/y/memory.limit_in_bytes", "536870912\n")
        self.write("v1/memory/x/y/memory.usage_in_bytes", "100\n")
        proc = self.write("cg1", "5:devices:/\n4:memory:/x/y\n0::/\n")
        self.assertEqual(_cgroup_memory(os.path.join(self.d, "v1"), proc), (536870912, 536870912 - 100))
        os.makedirs(os.path.join(self.d, "nolimit"))
        self.write("nolimit/memory.max", "max\n")
        self.assertEqual(_cgroup_memory(os.path.join(self.d, "nolimit"), none), (None, None))
        self.assertEqual(_cgroup_memory(os.path.join(self.d, "none"), none), (None, None))
        # _memory: the container's limit is the total, what it has left the free memory
        from unittest import mock
        import cutmesh.mesher as M
        with mock.patch.object(M, "_cgroup_memory", return_value=(2 * 2 ** 30, int(0.2 * 2 ** 30))):
            total, avail = M._memory()
        if total == 2 * 2 ** 30:                     # (a machine with less memory keeps its own)
            self.assertAlmostEqual(avail / 2 ** 30, 0.2, places=3)
            msgs = []
            M._check_size(1e6, msgs.append, memory=(total, avail))            # 1 GB: more than is left
            self.assertTrue(msgs)

    def test_extreme_level_stops_before_meshing(self):
        write_stl_ascii(os.path.join(self.d, "box.stl"), [("walls", box())])
        p = self.write("m.json", json.dumps({"domain": "box.stl", "baseCellSize": 0.1,
                                             "surfaceRefinement": {"default": 30}}))
        from cutmesh.mesher import generate
        with self.assertRaises(ConfigError) as cm:
            generate(load_config(p), log=lambda s: None)
        self.assertIn("Lower the refinement levels", str(cm.exception))

    def test_volume_only_inside_the_domain_and_once(self):
        """a refinement cylinder reaching far beyond the domain, or two over the same place,
        count only the fluid they cover, once"""
        from cutmesh.mesher import _min_cells
        from cutmesh.geometry import Surface
        write_stl_ascii(os.path.join(self.d, "box.stl"), [("walls", box())])
        cfg = load_config(self.write("m.json", json.dumps({"domain": "box.stl", "baseCellSize": 0.25})))
        dom = Surface(read_stl(os.path.join(self.d, "box.stl")), log=lambda s: None)
        lev = np.zeros(1)
        huge = Cylinder([0.5, 0.5, -1000], [0.5, 0.5, 1000], 1000)
        n0 = _min_cells(cfg, dom, [], [], 0.25, lev)
        n1 = _min_cells(cfg, dom, [], [(huge, 1)], 0.25, lev)
        self.assertLess(n1, 1e4)                                  # the box in level-1 cells: 512
        self.assertGreater(n1, n0)
        self.assertAlmostEqual(_min_cells(cfg, dom, [], [(huge, 1), (huge, 1)], 0.25, lev), n1)
        # a region smaller than a lattice cell, on a lattice point, counts its own volume only
        c = 0.5 + 0.5 / 32
        tiny = Cylinder([c, c, c - 1e-6], [c, c, c + 1e-6], 1e-6)
        self.assertLess(_min_cells(cfg, dom, [], [(tiny, 11)], 0.25, lev) - n0, 1)
        from cutmesh.mesher import _check_size
        _check_size(n1, lambda s: None, memory=(2 ** 30, 2 ** 30))  # no error

    def test_cylindrical_base_counts_only_refinement(self):
        """with a cylindrical base, baseCellSize is not the size of the base cells: a fine
        baseCellSize with nothing refined estimates nothing"""
        from cutmesh.mesher import _min_cells
        from cutmesh.geometry import Surface
        write_stl_ascii(os.path.join(self.d, "box.stl"), [("walls", box())])
        cfg = load_config(self.write("m.json", json.dumps({"domain": "box.stl", "baseCellSize": 0.002})))
        dom = Surface(read_stl(os.path.join(self.d, "box.stl")), log=lambda s: None)
        self.assertGreater(_min_cells(cfg, dom, [], [], 0.002, np.zeros(1)), 6e7)
        self.assertEqual(_min_cells(cfg, dom, [], [], 0.002, np.zeros(1), cartesian=False), 0)
        self.assertGreater(_min_cells(cfg, dom, [], [], 0.25, np.array([3]), cartesian=False), 1000)

    def test_lower_bound_of_the_examples(self):
        """the estimate stays below the cells the examples get (it is a lower bound)"""
        from cutmesh.mesher import _min_cells, build_zones
        from cutmesh.geometry import Surface
        ex = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples", "stirred_tank")
        cfg = load_config(os.path.join(ex, "meshDict.json"))
        dom = Surface(read_stl(os.path.join(ex, cfg["domain"])), log=lambda s: None)
        lev = np.array([int(cfg["surfaceRefinement"].get(n, cfg["surfaceRefinement"]["default"])) for n in dom.names])
        n = _min_cells(cfg, dom, build_zones(cfg), [], float(cfg["baseCellSize"]), lev)
        self.assertTrue(0.3 * 102429 < n < 102429, n)                   # the example has 102429 cells


class TestConfig(TempDir):
    def cfg(self, **kw):
        stl = os.path.join(self.d, "box.stl")
        write_stl_ascii(stl, [("walls", box())])
        md = {"domain": "box.stl", "baseCellSize": 0.1}
        md.update(kw)
        p = self.write("m.json", json.dumps(md))
        return load_config(p)

    def validate(self, cfg):
        lines = []
        validate_config(cfg, lines.append)
        return lines

    def errors(self, cfg):
        with self.assertRaises(ConfigError) as cm:
            validate_config(cfg, lambda s: None)
        return str(cm.exception)

    def test_minimal_ok(self):
        self.assertEqual(self.validate(self.cfg()), [])

    def test_unknown_key_warns_with_suggestion(self):
        lines = self.validate(self.cfg(surfaceRefinment={"default": 1}, _comment="x"))
        self.assertEqual(len(lines), 1)
        self.assertTrue(lines[0].startswith("WARNING: meshDict:"))
        self.assertIn("'surfaceRefinment'", lines[0])
        self.assertIn("did you mean 'surfaceRefinement'", lines[0])

    def test_unknown_nested_keys_warn(self):
        lines = self.validate(self.cfg(addLayers={"nLayer": 3},
                                       rotatingZones=[{"name": "r", "omega": 1, "omgea": 2,
                                                       "cylinder": {"point1": [0, 0, 0], "point2": [0, 0, 1],
                                                                    "radius": 0.2}}]))
        self.assertTrue(any("'nLayer'" in s and "nLayers" in s for s in lines))
        self.assertTrue(any("'omgea'" in s for s in lines))

    def test_required_and_missing_file(self):
        cfg = self.cfg()
        del cfg["baseCellSize"]
        cfg["domain"] = "nothere.stl"
        msg = self.errors(cfg)
        self.assertIn("baseCellSize is required", msg)
        self.assertIn("file not found", msg)
        self.assertIn("2 problem(s)", msg)

    def test_bad_values_are_all_reported(self):
        msg = self.errors(self.cfg(baseCellSize=0, bufferLayers=1.5, mergeRatio=2, checkLeaks="yes",
                                   hangingSnap="loose", surfaceRefinement={"default": -1},
                                   gapRefinement={"level": 2}))
        for s in ("baseCellSize", "bufferLayers", "mergeRatio", "checkLeaks", "hangingSnap",
                  "surfaceRefinement.default", "gapRefinement: thickness is required"):
            self.assertIn(s, msg)

    def test_rotating_zone_checks(self):
        cyl = {"point1": [0, 0, 0], "point2": [0, 0, 0], "radius": -1}
        msg = self.errors(self.cfg(rotatingZones=[{"name": "a", "cylinder": cyl},
                                                  {"name": "a", "stl": "none.stl"},
                                                  {"cylinder": cyl, "stl": "x.stl"}]))
        self.assertIn("point1 and point2 must differ", msg)
        self.assertIn("radius", msg)
        self.assertIn("used twice", msg)
        self.assertIn("none.stl", msg)
        self.assertIn("name is required", msg)
        self.assertIn("either cylinder or stl", msg)

    def test_cylindrical_base_checks(self):
        ok = {"type": "cylindrical"}
        self.assertEqual(self.validate(self.cfg(baseMesh=dict(ok, rOuter=1.0, cellSize=0.1))), [])
        manual = {"core": {"halfWidth": 0.1, "n": 2}, "blocks": [{"rOuter": 0.5, "nr": 2}]}
        self.assertEqual(self.validate(self.cfg(baseMesh=dict(ok, **manual))), [])
        msg = self.errors(self.cfg(baseMesh=dict(ok, core=manual["core"])))
        self.assertIn("both core and blocks, or neither", msg)
        msg = self.errors(self.cfg(baseMesh=dict(ok, rOuter=1.0, **manual)))
        self.assertIn("rOuter and cellSize are for the automatic layout", msg)
        msg = self.errors(self.cfg(baseMesh=dict(ok, rOuter=0, cellSize="x")))
        self.assertIn("baseMesh.rOuter", msg)
        self.assertIn("baseMesh.cellSize", msg)
        # each of these stopped with a traceback, ran without end or used up the memory
        for bm, text in (({"nz": 0}, "baseMesh.nz"), ({"nz": 2.5}, "baseMesh.nz must be an integer"),
                         ({"axis": [0, 0, 0]}, "baseMesh.axis must not be zero"),
                         ({"axis": [0, 1]}, "baseMesh.axis must have 3 items"),
                         ({"origin": "0 0 0"}, "baseMesh.origin must be a list"),
                         ({"dz": -0.01}, "baseMesh.dz"), ({"zRange": [0.3, 0.0]}, "low < high"),
                         ({"maxSizeRatio": 1.0}, "maxSizeRatio must be greater than 1"),
                         ({"core": {"halfWidth": 0.1, "n": 2}, "blocks": [{"radii": [0.2, 0.5], "nr": 1}]},
                          "baseMesh.blocks[0]: rOuter is required")):
            self.assertIn(text, self.errors(self.cfg(baseMesh=dict(ok, **bm))), bm)

    def test_snap_tolerance_outside_the_tested_range_warns(self):
        for t, warns in ((0.3, False), (0.2, False), (0.45, False), (0.1, True), (0.5, True), (0, True)):
            w = [s for s in self.validate(self.cfg(snapTolerance=t)) if "snapTolerance" in s]
            self.assertEqual(bool(w), warns, t)

    def test_layer_checks(self):
        self.assertEqual(self.validate(self.cfg(addLayers={})), [])
        self.assertEqual(self.validate(self.cfg(addLayers={"expansionRatio": 0.9, "maxSkewness": None})), [])
        for al, text in (({"expansionRatio": 0}, "addLayers.expansionRatio"),
                         ({"expansionRatio": -1}, "addLayers.expansionRatio"),
                         ({"finalLayerThickness": 0}, "addLayers.finalLayerThickness"),
                         ({"nIter": "x"}, "addLayers.nIter must be an integer"),
                         ({"relativeSizes": "no"}, "addLayers.relativeSizes must be true or false"),
                         ({"maxNonOrtho": 95}, "addLayers.maxNonOrtho")):
            self.assertIn(text, self.errors(self.cfg(addLayers=al)), al)

    def test_check_names(self):
        cfg = self.cfg(surfaceRefinement={"default": 1, "wall": 2}, patchTypes={"walls": "wall", "inlet": "patch"},
                       addLayers={"patches": ["walls", "wallz"]})
        lines = []
        check_names(cfg, ["walls"], lines.append)
        self.assertEqual(len(lines), 3)
        self.assertTrue(any("surfaceRefinement" in s and "'wall'" in s and "did you mean 'walls'" in s for s in lines))
        self.assertTrue(any("patchTypes" in s and "'inlet'" in s for s in lines))
        self.assertTrue(any("addLayers.patches" in s and "'wallz'" in s for s in lines))

    def test_bad_json(self):
        p = self.write("bad.json", "{\"domain\": ")
        with self.assertRaises(ConfigError):
            load_config(p)
        p = self.write("list.json", "[1, 2]")
        with self.assertRaises(ConfigError):
            load_config(p)


class TestZoneOverlap(unittest.TestCase):
    def zone(self, name, z0, z1, r=0.5, x=0.0):
        return dict(name=name, geom=Cylinder([x, 0, z0], [x, 0, z1], r))

    def test_overlap(self):
        with self.assertRaises(ConfigError) as cm:
            check_zone_overlap([self.zone("a", 0, 1), self.zone("b", 0.5, 1.5)], 0.05)
        self.assertIn("'a' and 'b' overlap", str(cm.exception))

    def test_contained(self):
        with self.assertRaises(ConfigError):
            check_zone_overlap([self.zone("a", 0, 1), self.zone("b", 0.2, 0.8, r=0.2)], 0.05)

    def test_coincident(self):
        # the same cylinder under two names: every surface sample lies on the other's
        # surface, the insides are shared
        with self.assertRaises(ConfigError) as cm:
            check_zone_overlap([self.zone("a", 0, 1), self.zone("b", 0, 1)], 0.05)
        self.assertIn("'a' and 'b' overlap", str(cm.exception))

    def test_touching_and_apart_ok(self):
        check_zone_overlap([self.zone("a", 0, 1), self.zone("b", 1, 2), self.zone("c", 0, 1, x=3)], 0.05)


class TestCLI(TempDir):
    def run_cli(self, *args):
        err = io.StringIO()
        with redirect_stderr(err):
            rc = cli(list(args))
        return rc, err.getvalue()

    def test_usage_errors(self):
        self.assertEqual(self.run_cli("m.json", "-case")[0], 2)
        rc, err = self.run_cli("m.json", "-nocheck")
        self.assertEqual(rc, 2)
        self.assertIn("unknown option '-nocheck'", err)
        self.assertEqual(self.run_cli("a.json", "b.json")[0], 2)
        self.assertEqual(self.run_cli("-noCheck")[0], 2)

    def test_config_error_is_reported_without_traceback(self):
        p = self.write("m.json", json.dumps({"domain": "none.stl", "baseCellSize": 0.1}))
        rc, err = self.run_cli(p)
        self.assertEqual(rc, 1)
        self.assertTrue(err.startswith("ERROR: meshDict"))
        self.assertIn("file not found", err)

    def test_case_that_is_a_file(self):
        write_stl_ascii(os.path.join(self.d, "box.stl"), [("walls", box())])
        f = self.write("notadir", "x")
        p = self.write("m.json", json.dumps({"domain": "box.stl", "baseCellSize": 0.5, "case": "notadir"}))
        rc, err = self.run_cli(p)
        self.assertEqual(rc, 1)
        self.assertIn("case must be a directory", err)
        p = self.write("m2.json", json.dumps({"domain": "box.stl", "baseCellSize": 0.5}))
        rc, err = self.run_cli(p, "-case", f)                 # the same through -case
        self.assertEqual(rc, 1)
        self.assertIn("must be a directory", err)
        # -case overrides the meshDict's case: a file there does not matter then
        out = os.path.join(self.d, "out")
        with redirect_stdout(io.StringIO()):
            rc, err = self.run_cli(self.write("m3.json", json.dumps(
                {"domain": "box.stl", "baseCellSize": 0.5, "case": "notadir"})), "-case", out, "-noCheck")
        self.assertEqual(rc, 0, err)
        self.assertTrue(os.path.isfile(os.path.join(out, "constant", "polyMesh", "owner")))

    def test_empty_stl_is_reported(self):
        self.write("e.stl", b"")
        p = self.write("m.json", json.dumps({"domain": "e.stl", "baseCellSize": 0.1}))
        rc, err = self.run_cli(p)
        self.assertEqual(rc, 1)
        self.assertIn("empty STL file", err)


if __name__ == "__main__":
    unittest.main()


class DeprecatedKeys(TempDir):
    """a renamed key keeps being read as its replacement, with a WARNING (config.DEPRECATED_KEYS)"""

    def test_renamed_and_retired(self):
        from unittest import mock
        from cutmesh import config
        write_stl_ascii(self.write("s.stl", ""), [("walls", box())])
        table = {"oldBuffer": ("bufferLayers", "0.9.0", "renamed"), "oldFlag": (None, "0.9.0", "")}
        with mock.patch.dict(config.DEPRECATED_KEYS, table):
            p = self.write("m.json", json.dumps({"domain": "s.stl", "baseCellSize": 0.1, "oldBuffer": 4,
                                                 "oldFlag": True}))
            cfg = load_config(p)
            self.assertEqual(cfg["bufferLayers"], 4)
            self.assertNotIn("oldBuffer", cfg)
            self.assertNotIn("oldFlag", cfg)
            lines = []
            validate_config(cfg, lines.append)
            self.assertTrue(any("'oldBuffer' is deprecated since 0.9.0: read as 'bufferLayers'" in s for s in lines), lines)
            self.assertTrue(any("'oldFlag' is deprecated since 0.9.0: it is ignored" in s for s in lines), lines)
            self.assertFalse(any("unknown key" in s for s in lines), lines)
            # the new key given as well wins
            p = self.write("m2.json", json.dumps({"domain": "s.stl", "baseCellSize": 0.1, "oldBuffer": 4,
                                                  "bufferLayers": 1}))
            self.assertEqual(load_config(p)["bufferLayers"], 1)

    def test_table_entries(self):
        # every entry names a known replacement (or none) and a version
        from cutmesh import config
        from cutmesh.mesher import DEFAULTS
        for old, (new, ver, note) in config.DEPRECATED_KEYS.items():
            self.assertNotIn(old, DEFAULTS)
            self.assertTrue(new is None or new in DEFAULTS or new in config.OTHER_KEYS, old)
            self.assertRegex(ver, r"^\d+\.\d+\.\d+$")
