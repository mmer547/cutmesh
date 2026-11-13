"""Prism layers at concave corners (layers.py, concaveSlide): in a closed cylinder the
flat ends meet the side wall at a concave edge.  The wall points of one wall next to the
edge, moving along their own normal, end up behind the corner point (which moves along
the bisector) and the layer cells between them invert; sliding them along their wall
keeps the layers at full thickness up to the corner.

python -m unittest discover -s tests
"""
import json
import os
import re
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from cutmesh import quality                                   # noqa: E402
from cutmesh.geometry import write_stl_ascii                  # noqa: E402
from cutmesh.layers import add_layers                         # noqa: E402
from cutmesh.mesher import load_config, generate              # noqa: E402


def cylinder_stl(path, R, z0, z1, n=96):
    """closed cylinder around the z axis, outward normals"""
    t = np.linspace(0, 2 * np.pi, n + 1)[:-1]
    c = np.stack([R * np.cos(t), R * np.sin(t)], 1)
    tris = []
    for i in range(n):
        a, b = c[i], c[(i + 1) % n]
        A0, B0, A1, B1 = np.r_[a, z0], np.r_[b, z0], np.r_[a, z1], np.r_[b, z1]
        tris += [[A0, B0, B1], [A0, B1, A1], [np.r_[0, 0, z0], B0, A0], [np.r_[0, 0, z1], A1, B1]]
    write_stl_ascii(path, [("walls", np.array(tris))])


class ConcaveCorner(unittest.TestCase):
    h = 0.01
    _mesh = None

    @classmethod
    def mesh(cls):
        if cls._mesh is None:
            with tempfile.TemporaryDirectory() as d:
                cylinder_stl(os.path.join(d, "s.stl"), 4.37 * cls.h, -3.13 * cls.h, 3.13 * cls.h)
                with open(os.path.join(d, "m.json"), "w") as f:
                    json.dump({"domain": "s.stl", "baseCellSize": cls.h,
                               "surfaceRefinement": {"default": 0}, "featureLevel": 0}, f)
                cls._mesh = generate(load_config(os.path.join(d, "m.json")), log=lambda s: None)
        return cls._mesh

    def layers(self, spec):
        msgs = []
        L = add_layers(self.mesh(), spec, log=msgs.append)
        full = float(re.search(r"full thickness ([0-9.]+)%", " ".join(msgs)).group(1)) / 100
        return L, full, msgs

    def test_full_thickness_up_to_the_corner(self):
        _, full_off, _ = self.layers({"concaveSlide": None})
        L, full, msgs = self.layers({})
        self.assertGreater(full, 0.9)
        self.assertGreater(full, full_off + 0.3)
        self.assertFalse(any("WARNING" in m for m in msgs))
        q = quality.check(L["points"], L["faces"], L["owner"], L["neighbour"], log=lambda s: None)
        self.assertIsNotNone(q)

    def test_corner_point_on_the_bisector(self):
        # the points on the edge move off both walls by the same thickness
        L, _, _ = self.layers({"nIter": 0, "nCollapseIter": 0})
        W, T, X = (L["layer_thickness"][k] for k in ("points", "T", "X"))
        D = np.asarray(L["points"])[W] - X
        R, Z = 4.37 * self.h, 3.13 * self.h
        r = np.hypot(X[:, 0], X[:, 1])
        edge = (np.abs(np.abs(X[:, 2]) - Z) < 1e-9) & (np.abs(r - R) < 1e-3 * self.h) & (T > 0)
        self.assertGreater(edge.sum(), 10)
        radial = -np.einsum("ij,ij->i", D[edge][:, :2], X[edge][:, :2]) / r[edge]
        axial = -np.sign(X[edge, 2]) * D[edge, 2]
        np.testing.assert_allclose(radial / T[edge], 1.0, atol=0.08)
        np.testing.assert_allclose(axial / T[edge], 1.0, atol=0.08)


def frustum_stl(path, r0, r1, z0, z1, n=96):
    """closed truncated cone around the z axis (radius r0 at z0, r1 at z1), outward normals"""
    t = np.linspace(0, 2 * np.pi, n + 1)[:-1]
    tris = []
    for i in range(n):
        a, b = t[i], t[(i + 1) % n]
        A0, B0 = np.r_[r0 * np.cos(a), r0 * np.sin(a), z0], np.r_[r0 * np.cos(b), r0 * np.sin(b), z0]
        A1, B1 = np.r_[r1 * np.cos(a), r1 * np.sin(a), z1], np.r_[r1 * np.cos(b), r1 * np.sin(b), z1]
        tris += [[A0, B0, B1], [A0, B1, A1], [np.r_[0, 0, z0], B0, A0], [np.r_[0, 0, z1], A1, B1]]
    write_stl_ascii(path, [("walls", np.array(tris))])


class CloseWallPoints(unittest.TestCase):
    """The bottom of a cyclone cone: along the concave edge between the cone and the flat
    bottom, cutting leaves wall points a few thousandths of a cell apart (a snapped octree
    vertex next to a cut point).  Unmerged, the zero-length edges between them flip the
    layer faces there and the layers collapse along the edge (mergeWallPoints)."""
    z0 = -0.0254
    _mesh = None

    @classmethod
    def mesh(cls):
        if cls._mesh is None:
            with tempfile.TemporaryDirectory() as d:
                frustum_stl(os.path.join(d, "s.stl"), 0.00381, 0.0127, cls.z0, 0.0)
                with open(os.path.join(d, "m.json"), "w") as f:
                    json.dump({"domain": "s.stl", "baseCellSize": 0.002, "surfaceRefinement": {"default": 1}}, f)
                cls._mesh = generate(load_config(os.path.join(d, "m.json")), log=lambda s: None)
        return cls._mesh

    def ring_cover(self, spec):
        msgs = []
        L = add_layers(self.mesh(), spec, log=msgs.append)
        X, T = L["layer_thickness"]["X"], L["layer_thickness"]["T"]
        ring = np.abs(X[:, 2] - self.z0) < 0.0015
        return L, (T[ring] > 0).mean(), msgs

    def test_layers_along_the_bottom_edge(self):
        _, cover_off, _ = self.ring_cover({"mergeWallPoints": 0})
        L, cover, msgs = self.ring_cover({})
        self.assertTrue(any("merged" in m for m in msgs))
        self.assertLess(cover_off, 0.9)
        self.assertEqual(cover, 1.0)
        q = quality.check(L["points"], L["faces"], L["owner"], L["neighbour"], log=lambda s: None)
        self.assertEqual(q["openCellEdges"], 0)
        self.assertLess(q["maxClosedness"], 1e-12)
        self.assertEqual(q["nNegVol"], 0)
        self.assertEqual(q["nBadPyramids"], 0)
        self.assertTrue(q["upperTriangular"])

    def test_merge_keeps_the_mesh(self):
        # cells, patches and volume unchanged; no unused points
        from cutmesh.layers import _merge_close_wall_points
        M = self.mesh()
        out, nm, nf = _merge_close_wall_points(M, ["walls"], 0.1, lambda s: None)
        self.assertGreater(nm, 0)
        nC = lambda R: int(max(np.max(R["owner"]), np.max(R["neighbour"]))) + 1
        self.assertEqual(nC(out), nC(M))
        self.assertEqual(sum(p["nFaces"] for p in out["patches"]) + len(out["neighbour"]), len(out["faces"][0]))
        self.assertEqual(len(np.unique(out["faces"][1])), len(out["points"]))
        q0 = quality.check(M["points"], M["faces"], M["owner"], M["neighbour"], log=lambda s: None)
        q1 = quality.check(out["points"], out["faces"], out["owner"], out["neighbour"], log=lambda s: None)
        self.assertAlmostEqual(q1["totalVol"] / q0["totalVol"], 1.0, places=5)
        self.assertEqual(q1["openCellEdges"], 0)


class MergeKeepsCellsValid(unittest.TestCase):
    def test_thin_cell_not_collapsed(self):
        # a thin box (1 x 1 x 0.05) with every face on the wall: its vertical edges are
        # shorter than 0.1 x the longest wall edge, but merging them would flatten the cell
        from cutmesh.layers import _merge_close_wall_points
        t = 0.05
        P = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                      [0, 0, t], [1, 0, t], [1, 1, t], [0, 1, t]], float)
        F = [[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
             [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
        M = dict(points=P, faces=(np.array([len(f) for f in F]), np.array(sum(F, []))),
                 owner=np.zeros(len(F), np.int64), neighbour=np.zeros(0, np.int64),
                 patches=[dict(name="wall", type="wall", startFace=0, nFaces=len(F))])
        msgs = []
        out, nm, nf = _merge_close_wall_points(M, ["wall"], 0.1, msgs.append)
        self.assertEqual((nm, nf), (0, 0))
        self.assertIs(out, M)
        self.assertTrue(any("left as they are" in m for m in msgs))
        q = quality.check(out["points"], out["faces"], out["owner"], out["neighbour"], log=lambda s: None)
        self.assertAlmostEqual(q["totalVol"], t)


if __name__ == "__main__":
    unittest.main()
