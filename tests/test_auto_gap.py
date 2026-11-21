"""autoGapRefinement wall samples (cutmesh/mesher.py _auto_gap_samples).

python -m unittest discover -s tests
"""
import os
import sys
import unittest
from unittest import mock

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "examples", "stirred_tank"))

from cutmesh.geometry import Surface                       # noqa: E402
from cutmesh.mesher import _auto_gap_samples, _interior_samples   # noqa: E402
import make_geometry as tank                               # noqa: E402


def box(lo, hi, name="walls"):
    """closed box, outward normals"""
    x0, y0, z0 = lo; x1, y1, z1 = hi
    V = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], float)
    F = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4),
         (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7)]
    return [(name, V[np.array(F)])]


def dist_to_edges(P, T):
    d = np.full(len(P), np.inf)
    for a, b in ((0, 1), (1, 2), (2, 0)):
        A, B = T[:, a], T[:, b]
        AB = B - A
        t = np.clip(np.einsum("ij,ij->i", P - A, AB) / np.einsum("ij,ij->i", AB, AB), 0, 1)
        d = np.minimum(d, np.linalg.norm(P - (A + t[:, None] * AB), axis=1))
    return d


class InteriorSamples(unittest.TestCase):
    def test_inside_and_dense_enough(self):
        T = np.array([[[0, 0, 0], [0.15, 0, 0], [0.15, 0.0118, 0]],      # sliver (tank bottom fan)
                      [[0, 0, 0], [1, 0, 0], [2, 0.01, 0]],              # obtuse needle
                      [[0, 0, 0], [1, 0, 0], [0, 1, 0]]], float)
        s = 0.01
        P, ti = _interior_samples(T, s)
        self.assertTrue(np.all(dist_to_edges(P, T[ti]) > 1e-6))         # no point on an edge
        from scipy.spatial import cKDTree
        rng = np.random.default_rng(0)
        for k in range(len(T)):
            q = rng.random((20000, 2)); q = q[q.sum(1) < 1]
            X = T[k, 0] + q[:, :1] * (T[k, 1] - T[k, 0]) + q[:, 1:] * (T[k, 2] - T[k, 0])
            self.assertLess(cKDTree(P[ti == k]).query(X)[0].max(), s)  # every point near a sample


class AutoGapSamples(unittest.TestCase):
    def test_tank_rims_are_not_gaps(self):
        # the plain tank has no narrow gap; the old code took the bottom and lid rims for
        # zero-width gaps (a sample on the rim edge measured along the faceted side wall)
        # and refined them to maxLevel (238 samples here)
        dom = Surface(tank.build())
        h0 = 0.0125
        dom.build_search(2 * h0 / 2 ** 3)
        rl = np.array([{"impeller": 2, "shaft": 2}.get(n, 1) for n in dom.names])
        groups, nun = _auto_gap_samples(dom, h0, rl, 2.0, 3)
        self.assertEqual(groups, [])
        self.assertEqual(nun, 0)

    def test_closed_box_has_no_gap(self):
        dom = Surface(box((0, 0, 0), (1, 0.6, 0.4)))
        dom.build_search(0.01)
        groups, nun = _auto_gap_samples(dom, 0.1, np.array([1]), 3.0, 4)
        self.assertEqual(groups, [])
        self.assertEqual(nun, 0)

    def test_same_spacing_reuses_widths_in_fine_sample_order(self):
        near = box((0, 0, 0), (1, 1, 1))[0][1]
        walls = np.concatenate([near, near + 4])
        dom = Surface([("low", walls[::2]), ("high", walls[1::2])])
        h0 = 0.5
        high_probes = []

        def widths(dom, p, ti, lw, h0, nacross, eps):
            high_probes.append(p[lw == 2])
            return np.where(p[:, 0] > 2, np.inf, np.where(p[:, 1] < 0.5, h0 / 8, h0 / 4))

        with mock.patch("cutmesh.mesher._gap_widths", widths):
            groups, nun = _auto_gap_samples(dom, h0, np.array([1, 2]), 2.0, 3)
        expected, _ = _interior_samples(dom.tris[dom.tris[:, 0, 0] < 2], h0 / 8)
        self.assertEqual(len(groups), 1)
        lv, pts, n = groups[0]
        self.assertEqual(lv, 3)
        self.assertEqual(n, nun)
        self.assertEqual(nun, (expected[:, 1] < 0.5).sum())
        np.testing.assert_array_equal(pts, expected)
        coarse, _ = _interior_samples(dom.tris[dom.region == 1], h0 / 8)
        np.testing.assert_array_equal(np.concatenate(high_probes), coarse)

    def test_stator_gap(self):
        # stirred tank with a fixed ring 0.5 mm outside the blade corners: only the ring's
        # inner face and the blade tips are refined, to the level with 2 cells in 0.5 mm
        solids = tank.build() + tank.stator_ring(0.5e-3)
        dom = Surface(solids)
        h0 = 0.0125
        dom.build_search(2 * h0 / 2 ** 6)
        lvl = {"impeller": 3, "shaft": 2, "stator": 3, "statorInner": 3}
        rl = np.array([lvl.get(n, 1) for n in dom.names])
        groups, nun = _auto_gap_samples(dom, h0, rl, 2.0, 6)
        self.assertEqual(nun, 0)
        lev = {L: p for L, p, _ in groups}
        self.assertIn(6, lev)                       # 2 cells of level 6 (0.195 mm) fit in 0.5 mm
        p = np.concatenate([p for p in lev.values()])
        self.assertLess(len(p), 500000)             # not the whole STL at the finest spacing
        r = np.hypot(p[:, 0], p[:, 1])
        near_gap = (r > tank.R_BLADE - 1e-4) & (r < tank.R_TIP + 0.5e-3 + 1e-4)
        self.assertGreater(near_gap.mean(), 0.95, np.round(np.percentile(r, [0, 5, 50, 95, 100]), 4))
        self.assertTrue(np.all(p[:, 2] > tank.Z0 - tank.STATOR_DZ - 1e-3))   # nothing at the bottom
        self.assertTrue(np.all(p[:, 2] < tank.Z1 + tank.STATOR_DZ + 1e-3))   # ... or the lid


if __name__ == "__main__":
    unittest.main()
