"""python -m unittest discover -s tests   (from the repository root)"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from cutmesh.geometry import Surface               # noqa: E402
from cutmesh.mesher import _thin_gap, merge_coplanar_wall_faces   # noqa: E402


def merge(P, faces):
    """all faces boundary faces of cell 0 in one wall patch"""
    nn = np.array([len(q) for q in faces], np.int64)
    vv = np.array([v for q in faces for v in q], np.int64)
    owner = np.zeros(len(faces), np.int64)
    patches = [dict(name="wall", type="wall", startFace=0, nFaces=len(faces))]
    nn, vv, _, _, nj = merge_coplanar_wall_faces(np.asarray(P, float), nn, vv, owner, 0, patches)
    return nn, nj


def grid_quads(nx, ny, keep, z=None):
    """unit squares (i, j) of an nx x ny grid for (i, j) in keep; z(x, y) bends it"""
    P = [(x, y, 0.0 if z is None else z(x, y)) for y in range(ny + 1) for x in range(nx + 1)]
    v = lambda i, j: j * (nx + 1) + i
    return P, [[v(i, j), v(i + 1, j), v(i + 1, j + 1), v(i, j + 1)] for i, j in keep]


class MergeWallFaces(unittest.TestCase):
    def test_flat_rectangle_joins(self):
        P, F = grid_quads(3, 1, [(0, 0), (1, 0), (2, 0)])
        nn, nj = merge(P, F)
        self.assertEqual(nj, 2)
        self.assertEqual(len(nn), 1)

    def test_u_shape_kept(self):
        # the joined U has its vertex mean on the notch edge: fan triangles not positive
        P, F = grid_quads(3, 2, [(0, 0), (1, 0), (2, 0), (0, 1), (2, 1)])
        nn, nj = merge(P, F)
        self.assertEqual(nj, 0)
        self.assertEqual(len(nn), 5)

    def test_curved_strip_kept(self):
        # neighbours 4 deg apart (each pair passes a 5 deg test), the ends 8 deg apart
        ang = np.radians([0.0, 4.0, 8.0])
        xs = [0.0]
        zs = [0.0]
        for a in ang:
            xs.append(xs[-1] + np.cos(a)); zs.append(zs[-1] + np.sin(a))
        P = [(x, y, z) for y in (0.0, 1.0) for x, z in zip(xs, zs)]
        F = [[i, i + 1, i + 5, i + 4] for i in range(3)]
        nn, nj = merge(P, F)
        self.assertEqual(len(nn), 3)


def box(lo, hi):
    """closed box surface, outward normals"""
    x0, y0, z0 = lo; x1, y1, z1 = hi
    c = np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)], float)
    q = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    return np.array([[c[a], c[b], c[d]] for a, b, cc, d in q] + [[c[b], c[cc], c[d]] for a, b, cc, d in q])


def flip(t):
    return t[:, [0, 2, 1]]


class ThinGap(unittest.TestCase):
    def test_thin_plate_across_gap_not_skipped(self):
        # fluid box with a 0.2 thick solid plate 0.4 off the bottom wall (thickness 8):
        # 1-unit fixed steps from the bottom wall land in the fluid above the plate
        outer = box((-5, -5, 0), (5, 5, 20))
        plate = flip(box((-4, -4, 0.4), (4, 4, 0.6)))
        dom = Surface([("w", np.concatenate([outer, plate]))])
        dom.build_search(0.05)
        p = np.array([[0.0, 0.0, 0.0], [4.5, 4.5, 0.0]])
        into = np.array([[0, 0, 1.0], [0, 0, 1.0]])
        thin, nun = _thin_gap(dom, p, into, 8.0, 0.01)
        self.assertEqual(nun, 0)
        self.assertEqual(thin.tolist(), [True, False])


if __name__ == "__main__":
    unittest.main()
