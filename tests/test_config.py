"""Input checks before meshing: unreadable STL files (geometry.read_stl), meshDict settings
(config.validate_config / check_names), overlapping rotating zones and the command line."""
import io
import json
import os
import struct
import tempfile
import unittest
from contextlib import redirect_stderr

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

    def test_empty_stl_is_reported(self):
        self.write("e.stl", b"")
        p = self.write("m.json", json.dumps({"domain": "e.stl", "baseCellSize": 0.1}))
        rc, err = self.run_cli(p)
        self.assertEqual(rc, 1)
        self.assertIn("empty STL file", err)


if __name__ == "__main__":
    unittest.main()
