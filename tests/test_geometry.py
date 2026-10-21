"""python -m unittest discover -s tests   (from the repository root)"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from cutmesh import quality                       # noqa: E402
from cutmesh.cutcell import _chain_loops, _simple_loops   # noqa: E402


def polygon_centroid_2d(xy):
    """exact area centroid of a simple polygon (shoelace)"""
    x, y = xy[:, 0], xy[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cr = x * y1 - x1 * y
    a = cr.sum() / 2
    return np.array([((x + x1) * cr).sum(), ((y + y1) * cr).sum()]) / (6 * a), a


class FaceGeometry(unittest.TestCase):
    def check(self, xy, z=0.3):
        P = np.c_[xy, np.full(len(xy), z)]
        Cf, Sf = quality.face_geometry(P, quality.FlatFaces([list(range(len(xy)))]))
        c, a = polygon_centroid_2d(xy)
        np.testing.assert_allclose(Cf[0], [c[0], c[1], z], atol=1e-12)
        np.testing.assert_allclose(Sf[0], [0, 0, a], atol=1e-12)

    def test_convex(self):
        self.check(np.array([[0, 0], [2, 0], [2, 1], [0, 1.0]]))

    def test_l_shape(self):
        self.check(np.array([[0, 0], [3, 0], [3, 1], [1, 1], [1, 3], [0, 3.0]]))

    def test_fan_centre_outside(self):
        # thin U shape: the vertex average lies outside the polygon, so some fan
        # triangles fold back and overlap (counted twice with unsigned weights)
        self.check(np.array([[0, 0], [4, 0], [4, 4], [3.8, 4], [3.8, 0.2],
                             [0.2, 0.2], [0.2, 4], [0, 4.0]]))


class CapLoops(unittest.TestCase):
    def test_figure_eight_is_split(self):
        # two triangles touching at vertex 0
        segs = [(0, 1), (1, 2), (2, 0), (0, 3), (3, 4), (4, 0)]
        loops = _chain_loops(segs)
        self.assertEqual(sorted(sorted(l) for l in loops), [[0, 1, 2], [0, 3, 4]])
        for l in loops:
            self.assertEqual(len(l), len(set(l)))

    def test_simple_loops(self):
        self.assertEqual(_simple_loops([0, 1, 2, 0, 3, 4]), [[0, 1, 2], [0, 3, 4]])
        self.assertEqual(_simple_loops([5, 6, 7, 8]), [[5, 6, 7, 8]])
        # nested repeat, and a degenerate back-and-forth part that is dropped
        self.assertEqual(_simple_loops([0, 1, 2, 3, 1, 4, 5]), [[1, 2, 3], [0, 1, 4, 5]])
        self.assertEqual(_simple_loops([0, 1, 2, 1, 3]), [[0, 1, 3]])


if __name__ == "__main__":
    unittest.main()
