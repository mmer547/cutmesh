"""python -m unittest discover -s tests   (from the repository root)"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from cutmesh.geometry import Surface               # noqa: E402
from cutmesh.mesher import (_thin_gap, _thin_gap_py, merge_coplanar_wall_faces,   # noqa: E402
                            _gap_refinement_samples, _wall_into, _wall_skewness)


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

    def test_slightly_folded_pair_joins(self):
        # two pieces of a curved wall (a cone cut by the grid) meet at a 2 deg fold: never
        # exactly coplanar, but one face; kept apart, the small piece's centre lies far off
        # the cell centre's normal and the layers collapse there
        for deg, joined in ((2.0, True), (4.5, True), (6.0, False)):
            t = np.radians(deg)
            P = [(-1, 0, 0), (0, 0, 0), (np.cos(t), 0, np.sin(t)),
                 (-1, 1, 0), (0, 1, 0), (np.cos(t), 1, np.sin(t))]
            nn, nj = merge(P, [[0, 1, 4, 3], [1, 2, 5, 4]])
            self.assertEqual(nj, int(joined), deg)
            self.assertEqual(len(nn), 1 if joined else 2, deg)

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

    def test_rust_matches_python(self):
        # the per-sample Rust walk gives the masks and counts of the NumPy passes
        outer = box((-5, -5, 0), (5, 5, 20))
        plate = flip(box((-4, -4, 0.4), (4, 4, 0.6)))
        slot = flip(box((-3, -0.05, 5), (3, 0.05, 15)))
        dom = Surface([("w", np.concatenate([outer, plate, slot]))])
        dom.build_search(0.05)
        if dom._rs is None or not hasattr(dom._rs, "thin_gap"):
            self.skipTest("Rust extension not built")
        p, ti = dom.samples(0.3)
        T = dom.tris[ti]
        nrm = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
        nrm /= np.linalg.norm(nrm, axis=1)[:, None]
        into = np.where(dom.inside(p - 0.01 * nrm)[:, None], -nrm, nrm)
        for tg, steps in ((8.0, 512), (0.5, 512), (12.0, 3)):
            a = _thin_gap_py(dom, p, into, tg, 0.01, steps)
            b = _thin_gap(dom, p, into, tg, 0.01, steps)
            np.testing.assert_array_equal(a[0], b[0])
            self.assertEqual(a[1], b[1])
            self.assertGreater(int(a[0].sum()), 0)

    def test_gap_refinement_first_look(self):
        # the first look (every 2 tg) and the fine samples of the triangles it finds give
        # the thin samples of sampling every wall finely
        outer = box((-5, -5, 0), (5, 5, 20))
        plate = flip(box((-4, -4, 0.4), (4, 4, 0.6)))
        slot = flip(box((-3, -0.05, 5), (3, 0.05, 15)))
        dom = Surface([("w", np.concatenate([outer, plate, slot]))])
        dom.build_search(0.05)
        sp, tg = 0.1, 0.5
        p, ti = dom.samples(sp, ordered=False)
        thin, _ = _thin_gap(dom, p, _wall_into(dom, p, ti, 0.02 * sp), tg, 0.02 * sp)
        q, nun, ncand = _gap_refinement_samples(dom, sp, tg)
        self.assertEqual(nun, 0)
        self.assertLess(ncand, len(dom.tris))
        self.assertGreater(len(q), 0)
        key = lambda a: a[np.lexsort(a.T[::-1])]               # noqa: E731
        np.testing.assert_array_equal(key(q), key(p[thin]))
        # no first look when its spacing (2 tg) is under two samples: every wall finely
        q2, _, nc2 = _gap_refinement_samples(dom, sp, 0.08)
        self.assertIsNone(nc2)


class WallSkewness(unittest.TestCase):
    def test_matches_quality(self):
        # a thin slab cell 1 x 1 x 0.1 with its top face split into a small and a large
        # part: the boundary skewness of quality.face_skewness (checkMesh's formula)
        from cutmesh import quality
        P = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                      [0, 0, .1], [1, 0, .1], [1, 1, .1], [0, 1, .1], [.9, 0, .1], [.9, 1, .1]], float)
        faces = [[0, 3, 2, 1], [4, 8, 9, 7], [8, 5, 6, 9], [0, 1, 5, 8, 4], [3, 7, 9, 6, 2],
                 [0, 4, 7, 3], [1, 2, 6, 5]]
        ff = quality.FlatFaces(faces)
        Cf, Sf = quality.face_geometry(P, ff)
        own = np.zeros(len(faces), np.int64)
        nei = np.zeros(0, np.int64)
        C, _ = quality.cell_geometry(1, own, nei, Cf, Sf)
        ref = quality.face_skewness(P, ff, Cf, Sf, C, own, nei)
        got = _wall_skewness(P, [(i, f) for i, f in enumerate(faces)], Cf, Sf, C[0])
        np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-12)
        self.assertGreater(got[2], 4.0)                  # the narrow strip far off the centre


if __name__ == "__main__":
    unittest.main()
