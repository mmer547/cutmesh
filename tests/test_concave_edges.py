"""Concave feature edges (the solid juts into the fluid) between the vertex lines: the
vertex in the solid wedge is snapped onto the edge from up to concaveSnapMax x h, so the
edge is kept instead of chamfered (mesher.snap_features).

python -m unittest discover -s tests
"""
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cutmesh import quality                                   # noqa: E402
from cutmesh.geometry import Surface                          # noqa: E402
from cutmesh.mesher import load_config, generate              # noqa: E402
from test_thin_gaps import voxel_stl, _write_json             # noqa: E402


class ConcaveEdge(unittest.TestCase):
    h = 0.01

    def generate(self, G, extra):
        h, v = self.h, self.h / 20
        with tempfile.TemporaryDirectory() as d:
            voxel_stl(os.path.join(d, "s.stl"), G, v)
            _write_json(os.path.join(d, "m.json"), {"domain": "s.stl", "baseCellSize": h, "featureLevel": 0,
                                                    "surfaceRefinement": {"default": 0}, **extra})
            R = generate(load_config(os.path.join(d, "m.json")), log=lambda s: None)
        P = R["points"]
        q = quality.check(P, R["faces"], R["owner"], R["neighbour"], log=lambda s: None)
        self.assertEqual(q["nNegVol"], 0)
        self.assertEqual(q["nBadPyramids"], 0)
        self.assertEqual(q["openCellEdges"], 0)
        return P, q["totalVol"] / (G.sum() * v ** 3)

    def mesh(self, corner, extra):
        """L-shaped channel: solid block [0, corner]^2 x [0, 3h]"""
        h, v = self.h, self.h / 20
        G = np.ones((120, 120, 60), bool)
        k = int(round(corner / v))
        G[:k, :k, :] = False
        P, vr = self.generate(G, extra)
        c = k * v
        on_edge = int((np.hypot(P[:, 0] - c, P[:, 1] - c) < 1e-6 * h).sum())
        return on_edge, vr

    def test_thin_post_kept(self):
        # a 0.8 h square post around a vertex line: its solid vertices lie 0.57 h from the
        # post's (re-entrant) edges, but snapping them there would remove the post
        G = np.ones((120, 120, 60), bool)
        G[52:68, 52:68, :] = False
        _, vr = self.generate(G, {})
        _, vr0 = self.generate(G, {"concaveSnapMax": 0.5})
        self.assertAlmostEqual(vr, vr0, delta=1e-9)
        self.assertLess(vr, 1.015)               # 1.018: the post gone entirely

    def test_corner_kept(self):
        # nearest (solid) vertex 0.57 h from the edge: beyond the usual 0.5 h snap
        n, vr = self.mesh(2.4 * self.h, {})
        self.assertGreaterEqual(n, 2)
        n0, vr0 = self.mesh(2.4 * self.h, {"concaveSnapMax": 0.5})
        self.assertEqual(n0, 0)
        self.assertLess(abs(vr - 1), abs(vr0 - 1))

    def test_flattened_face_rejected(self):
        # the solid vertex would land on the diagonal of the next cell's faces: not snapped
        n, vr = self.mesh(2.5 * self.h, {})
        self.assertEqual(n, 0)
        self.assertAlmostEqual(vr, 1.0, delta=0.005)


def _box(lo, hi):
    """Outward-wound triangles of an axis-aligned box"""
    x0, y0, z0 = lo; x1, y1, z1 = hi
    P = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], float)
    q = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (2, 3, 7, 6), (1, 2, 6, 5), (0, 4, 7, 3)]
    return np.array([[P[a], P[b], P[c]] for a, b, c, d in q for a, b, c in ((a, b, c), (a, c, d))])


class ConcaveFeatureEdges(unittest.TestCase):
    def l_shape(self):
        """L-shaped prism: unit square minus [0, 0.5]^2, extruded over z in [0, 1]"""
        xy = [(0.5, 0), (1, 0), (1, 1), (0, 1), (0, 0.5), (0.5, 0.5)]
        n = len(xy)
        T = []
        for i in range(n):
            a, b = xy[i], xy[(i + 1) % n]
            T += [[(*a, 0), (*b, 0), (*b, 1)], [(*a, 0), (*b, 1), (*a, 1)]]
        for z, rev in ((0, True), (1, False)):   # caps: two convex quads
            for quad in ([0, 1, 2, 5], [5, 2, 3, 4]):
                p = [(*xy[k], z) for k in quad]
                tri = [[p[0], p[1], p[2]], [p[0], p[2], p[3]]]
                T += [t[::-1] for t in tri] if rev else tri
        return np.array(T, float)

    def test_winding_independent(self):
        T = self.l_shape()
        for tris in (T, T[:, ::-1]):
            seg = Surface([("s", tris)]).feature_edges(30.0, concave_only=True)
            self.assertEqual(len(seg), 1)        # only the re-entrant vertical edge
            self.assertTrue(np.allclose(seg[0][:, :2], 0.5))
            # all 18 sharp edges (6 vertical + 12 cap edges) either way
            self.assertEqual(len(Surface([("s", tris)]).feature_edges(30.0)), 18)

    def test_nested_shell(self):
        # fluid between two boxes: the inner box's edges are re-entrant for the fluid
        T = np.concatenate([_box((0, 0, 0), (3, 3, 3)), _box((1, 1, 1), (2, 2, 2))])
        seg = Surface([("s", T)]).feature_edges(30.0, concave_only=True)
        self.assertEqual(len(seg), 12)
        self.assertTrue(((seg >= 1 - 1e-12) & (seg <= 2 + 1e-12)).all())

    def test_close_nested_shell(self):
        # inner box 0.1 off the outer one: the probes at the outer (convex) edges would
        # land in the inner solid without stopping at its wall
        T = np.concatenate([_box((0, 0, 0), (3, 3, 3)), _box((0.1, 0.1, 0.1), (2.9, 2.9, 2.9))])
        for tris in (T, T[:, ::-1]):
            seg = Surface([("s", tris)]).feature_edges(30.0, concave_only=True)
            self.assertEqual(len(seg), 12)
            self.assertTrue(((seg >= 0.1 - 1e-12) & (seg <= 2.9 + 1e-12)).all())

    def test_open_edges_skipped(self):
        tri = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], float)
        self.assertEqual(len(Surface([("s", tri)]).feature_edges(30.0, concave_only=True)), 0)
        self.assertEqual(len(Surface([("s", tri)]).feature_edges(30.0)), 3)


if __name__ == "__main__":
    unittest.main()
