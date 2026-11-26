"""benchmark/mesh_hash.py and the mesh check in run_bench.compare (--same-mesh)."""
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "benchmark"))

import mesh_hash  # noqa: E402
import run_bench  # noqa: E402


def write_case(root, owner=b"0 1 2"):
    d = os.path.join(root, "constant", "polyMesh")
    os.makedirs(d)
    for name, data in (("owner", owner), ("points", b"(0 0 0)"), ("faces", b"3(0 1 2)")):
        with open(os.path.join(d, name), "wb") as f:
            f.write(data)
    return root


def result(name, h):
    return {"name": name, "status": "PASS", "meshHash": h, "quality": {}}


class MeshHash(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        t = self.tmp.name
        self.a = mesh_hash.mesh_hash(write_case(os.path.join(t, "a")))
        self.b = mesh_hash.mesh_hash(write_case(os.path.join(t, "b")))
        self.c = mesh_hash.mesh_hash(write_case(os.path.join(t, "c"), owner=b"0 1 3"))

    def tearDown(self):
        self.tmp.cleanup()

    def test_hash(self):
        self.assertEqual(self.a, self.b)
        self.assertEqual(mesh_hash.diff(self.a, self.c), ["owner"])
        # the polyMesh directory itself works too
        d = os.path.join(self.tmp.name, "a", "constant", "polyMesh")
        self.assertEqual(mesh_hash.mesh_hash(d), self.a)
        self.assertEqual(mesh_hash.main([os.path.join(self.tmp.name, "a"), os.path.join(self.tmp.name, "c")]), 1)

    def test_compare(self):
        base = {"platform": run_bench.platform_id(), "results": [result("x", self.a)]}
        same = run_bench.compare([result("x", self.b)], base, same_mesh=True)
        self.assertEqual(same, ["same   x: same mesh"])
        changed = run_bench.compare([result("x", self.c)], base, same_mesh=True)
        self.assertTrue(changed[0].startswith("WORSE") and "owner" in changed[0], changed)
        info = run_bench.compare([result("x", self.c)], base)
        self.assertTrue(info[0].startswith("mesh"), info)
        other = dict(base, platform="another machine")
        lines = run_bench.compare([result("x", self.c)], other, same_mesh=True)
        self.assertTrue(lines[0].startswith("note") and not any(s.startswith("WORSE") for s in lines), lines)


def gres(name="x", **kw):
    q = {"nCells": 1000, "nNonOrth70": 0, "nSkew4": 2, "nNegVol": 0}
    q.update(kw.pop("quality", {}))
    r = {"name": name, "status": "PASS", "quality": q,
         "featureEdges": {"over02": 0.010, "p95": 0.10}, "layers": {"full": 90.0}}
    r.update(kw)
    return r


class QualityGate(unittest.TestCase):
    def test_pass(self):
        o = gres()
        self.assertEqual(run_bench.gate(gres(quality={"nCells": 1029, "nSkew4": 1}), o), [])
        for n in (970, 1030):                  # exactly 3 %: inside (no rounding at the boundary)
            self.assertEqual(run_bench.gate(gres(quality={"nCells": n}), o), [], n)
        self.assertEqual(run_bench.gate(gres(layers={"full": 89.1},
                                             featureEdges={"over02": 0.014, "p95": 0.11}), o), [])

    def test_fail(self):
        o = gres()
        cases = [({"quality": {"nNonOrth70": 1}}, "non-orth"),
                 ({"quality": {"nSkew4": 3}}, "skewness"),
                 ({"quality": {"nNegVol": 1}}, "non-positive"),
                 ({"quality": {"nCells": 1031}}, "cells"),
                 ({"quality": {"nCells": 969}}, "cells"),
                 ({"layers": {"full": 89.0}}, "layer"),
                 ({"layers": None}, "layer"),
                 ({"featureEdges": {"over02": 0.016, "p95": 0.10}}, "over 0.2"),
                 ({"featureEdges": {"over02": 0.010, "p95": 0.13}}, "p95")]
        for kw, what in cases:
            g = run_bench.gate(gres(**kw), o)
            self.assertTrue(len(g) == 1 and what in g[0], (kw, g))

    def test_existing_non_orth(self):
        # a baseline that already has faces over 70 degrees: not more of them
        o = gres(quality={"nNonOrth70": 4})
        self.assertEqual(run_bench.gate(gres(quality={"nNonOrth70": 4}), o), [])
        self.assertEqual(len(run_bench.gate(gres(quality={"nNonOrth70": 5}), o)), 1)

    def test_no_mesh(self):
        # an input rejected in both runs (no quality report): the status decides
        err = {"name": "x", "status": "ERROR", "error": "STLError"}
        self.assertEqual(run_bench.gate(err, err), [])
        self.assertEqual(run_bench.gate(err, gres()), ["no quality report"])

    def test_not_applicable(self):
        # no layers / no feature edges in the case (None in both runs): those checks are skipped
        o = gres(layers=None, featureEdges=None)
        self.assertEqual(run_bench.gate(gres(layers=None, featureEdges=None), o), [])

    def test_not_measured(self):
        # a measure missing (a baseline from before the measures, a failed measurement): no pass
        o = gres()
        for drop in ("layers", "featureEdges"):
            for side in ("base", "run"):
                a, b = gres(), gres()
                del (a if side == "base" else b)[drop]
                g = run_bench.gate(b, a)
                self.assertTrue(len(g) == 1 and "not measured" in g[0], (drop, side, g))
        for side in ("base", "run"):
            a, b = gres(), gres()
            r = a if side == "base" else b
            del r["layers"], r["featureEdges"]
            r["metricsError"] = "ValueError: x"
            g = run_bench.gate(b, a)
            self.assertTrue(len(g) == 1 and "no measures" in g[0] and "ValueError" in g[0], (side, g))
        # a quality failure is still reported with them
        g = run_bench.gate(gres(quality={"nSkew4": 9}, metricsError="E"), o)
        self.assertEqual(len(g), 2, g)

    def test_compare(self):
        base = {"platform": "another machine", "results": [gres("a"), gres("b")]}
        lines = run_bench.compare([gres("a"), gres("b", quality={"nSkew4": 9}), gres("c")], base, use_gate=True)
        self.assertIn("gate   a: ok", lines)
        self.assertTrue(any(s.startswith("WORSE  b: gate: faces skewness > 4") for s in lines), lines)
        self.assertIn("WORSE  c: gate: not in the baseline", lines)

    def test_compare_skipped(self):
        # judged in the baseline but skipped here: not a pass; skipped in both: not judged
        skip = {"name": "a", "status": "SKIP", "notes": ["STL not found"]}
        base = {"platform": "another machine", "results": [gres("a")]}
        lines = run_bench.compare([skip], base, use_gate=True)
        self.assertIn("WORSE  a: gate: skipped in this run (baseline PASS)", lines)
        base = {"platform": "another machine", "results": [dict(skip)]}
        lines = run_bench.compare([skip], base, use_gate=True)
        self.assertFalse(any(s.startswith("WORSE") for s in lines), lines)
        # without --gate a skip is not a regression (as before)
        base = {"platform": "another machine", "results": [gres("a")]}
        self.assertFalse(any(s.startswith("WORSE") for s in run_bench.compare([skip], base)))


class MeshMetrics(unittest.TestCase):
    def test_feature_edges(self):
        import numpy as np
        import mesh_metrics
        # a unit square face split into two boundary quads: edge x = 0.5 is a mesh edge
        P = np.array([[0, 0, 0], [0.5, 0, 0], [1, 0, 0], [0, 1, 0], [0.5, 1, 0], [1, 1, 0]], float)
        faces = (np.array([4, 4]), np.array([0, 1, 4, 3, 1, 2, 5, 4]))
        on = np.array([[[0.5, 0.1, 0], [0.5, 0.9, 0]]])
        r = mesh_metrics.feature_edge_deviation(P, faces, 0, on, 0.1)
        self.assertLess(r["max"], 1e-12)
        off = on + [0.05, 0, 0]                    # half a cell (0.1) off the mesh edge
        r = mesh_metrics.feature_edge_deviation(P, faces, 0, off, 0.1)
        self.assertAlmostEqual(r["max"], 0.5)
        self.assertEqual(r["over02"], 1.0)

    def test_log(self):
        import mesh_metrics
        lines = ["base cell 0.0052, max level 3 (smallest cell 0.00065)",
                 "addLayers: 10 layer cells, total 20 cells; wall points with full thickness 93.5%,"
                 " reduced 5.0%, none 1.5% (1.0s)"]
        self.assertEqual(mesh_metrics.smallest_cell(lines), 0.00065)
        # a cylindrical base that lowers it: its value (the last one)
        cyl = lines[:1] + ["smallest cell of the cylindrical base 0.000412"] + lines[1:]
        self.assertEqual(mesh_metrics.smallest_cell(cyl), 0.000412)
        self.assertEqual(mesh_metrics.smallest_cell(["boundary faces: max distance 0.1 (1.54 x smallest cell)"]), None)
        self.assertEqual(mesh_metrics.layer_coverage(lines), {"full": 93.5, "reduced": 5.0, "none": 1.5})
        self.assertIsNone(mesh_metrics.layer_coverage(lines[:1]))


if __name__ == "__main__":
    unittest.main()
