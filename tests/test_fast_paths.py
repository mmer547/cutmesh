"""Faster code paths that must give the same answers as the plain ones: the open cell
edge count of the quality check, duplicate-free surface sampling, sorted-query search,
the Rust octree stages (2:1 balance, polyhedral cells, leaf validity) and the Rust cut
kernel of the domain cut (with the feature hooks).

python -m unittest discover -s tests
"""
import contextlib
import copy
import io
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cutmesh import geometry, mesher, octree, quality          # noqa: E402
from test_polymesh_io import hex_grid                           # noqa: E402


def open_edges_brute(faces, owner, neighbour):
    """directed edges (cell, a, b) whose reverse (cell, b, a) is missing, by sets"""
    nI = len(neighbour)
    edges = []
    for f, p in enumerate(faces):
        for i in range(len(p)):
            a, b = p[i], p[(i + 1) % len(p)]
            edges.append((owner[f], a, b))
            if f < nI:
                edges.append((neighbour[f], b, a))
    have = set(edges)
    return sum(1 for c, a, b in edges if (c, b, a) not in have)


class OpenCellEdges(unittest.TestCase):
    def run_check(self, P, faces, own, nei):
        with contextlib.redirect_stdout(io.StringIO()):
            return quality.check(P, faces, own, nei, log=lambda *a: None)

    def test_closed_and_broken(self):
        P, (sizes, flat), own, nei, _, _, _ = hex_grid(3)
        ptr = np.concatenate([[0], np.cumsum(sizes)])
        faces = [flat[ptr[i]:ptr[i + 1]].tolist() for i in range(len(sizes))]
        self.assertEqual(self.run_check(P, faces, own, nei)["openCellEdges"], 0)
        # drop a vertex from a few faces (internal and boundary): their cells get open edges
        rng = np.random.default_rng(3)
        for f in rng.choice(len(faces), 6, replace=False).tolist():
            faces[f] = faces[f][1:]
        got = self.run_check(P, faces, own, nei)["openCellEdges"]
        self.assertGreater(got, 0)
        self.assertEqual(got, open_edges_brute(faces, own, nei))


class SurfaceSamples(unittest.TestCase):
    def test_same_points_without_repeats(self):
        rng = np.random.default_rng(0)
        tris = rng.uniform(-1, 1, (40, 3, 3))
        tris[5] = tris[4][[1, 2, 0]]                  # a repeated triangle stays two inputs
        for sp in (0.05, 0.2, 5.0):
            p, ti = geometry._sample_triangles(tris, sp)
            T, par = geometry.subdivide_triangles(tris, sp)
            q = np.concatenate([T.reshape(-1, 3), T.mean(1)])
            qi = np.concatenate([np.repeat(par, 3), par])

            def rows(x, i):
                k = np.empty((len(x), 4), np.int64)
                k[:, :3] = np.ascontiguousarray(x).view(np.int64)
                k[:, 3] = i
                return {tuple(r) for r in k.tolist()}
            a, b = rows(p, ti), rows(q, qi)
            self.assertEqual(a, b)                    # the same (point, triangle) set
            self.assertEqual(len(a), len(p))          # and each once


class SortedSearch(unittest.TestCase):
    def test_matches_numpy(self):
        rng = np.random.default_rng(1)
        a = np.unique(rng.integers(0, 10 ** 9, 50000))
        for n in (10, 100000):
            v = rng.integers(-5, 10 ** 9 + 5, (n, 2))
            np.testing.assert_array_equal(geometry.searchsorted(a, v), np.searchsorted(a, v))


def random_octree(n, L, periodic, seed, frac=0.15):
    rng = np.random.default_rng(seed)
    oc = octree.Octree(np.zeros(3), 1.0, n, L, periodic)
    for lv in range(L):
        oc.split((oc.lev == lv) & (rng.random(len(oc.lev)) < frac))
    return oc


CASES = [((3, 4, 5), 3, (False, False, False), 0), ((4, 3, 6), 3, (False, True, False), 1),
         ((5, 5, 5), 4, (True, False, True), 2), ((8, 7, 6), 4, (False, False, False), 3)]


@unittest.skipUnless(geometry.rust_available() and hasattr(geometry._core, "build_cells"),
                     "Rust extension not built")
class RustOctree(unittest.TestCase):
    """the Rust balance / build_cells / leaf test give exactly the arrays of the NumPy code"""

    def test_balance(self):
        for args in CASES:
            a = random_octree(*args)
            b = copy.deepcopy(a)
            la, lb = [], []
            a._balance_py(la.append)
            b.balance(lb.append)
            np.testing.assert_array_equal(a.ijk, b.ijk)
            np.testing.assert_array_equal(a.lev, b.lev)
            self.assertEqual(la, lb)

    def test_build_cells(self):
        for args in CASES:
            oc = random_octree(*args)
            oc.balance(lambda s: None)
            A, B = oc._build_cells_py(), oc.build_cells()
            self.assertEqual(sorted(A), sorted(B))
            for k in A:
                np.testing.assert_array_equal(np.asarray(A[k]), np.asarray(B[k]), err_msg=k)

    def test_leaf_bad(self):
        rng = np.random.default_rng(5)
        for args in CASES:
            oc = random_octree(*args)
            oc.balance(lambda s: None)
            M = oc.build_cells()
            leaves = np.arange(len(oc.lev))
            for amp in (0.05, 0.3, 0.6):
                P = M["points"] + rng.normal(0, amp / 2 ** oc.Lmax, M["points"].shape)
                fa = rng.random(len(P)) < 0.3
                for flat, ang in ((None, 2.0), (fa, 2.0), (fa, -1.0)):
                    a = mesher._leaf_bad_py(P, leaves, M["fptr"], M["fverts"], M["cell_fptr"],
                                            M["cell_size"], flat, ang)
                    b = mesher._leaf_bad(P, leaves, M["fptr"], M["fverts"], M["cell_fptr"],
                                         M["cell_size"], flat, ang)
                    np.testing.assert_array_equal(a, b)


class NumpyOrder(unittest.TestCase):
    """the Rust ports repeat NumPy's summation order; fails if NumPy changes it"""

    def test_einsum_rows(self):
        rng = np.random.default_rng(2)
        a = rng.normal(size=(50000, 3)) * rng.uniform(0, 10, (50000, 1))
        b = rng.normal(size=(50000, 3))
        p = a * b
        np.testing.assert_array_equal(np.einsum("ij,ij->i", a, b), (p[:, 0] + p[:, 2]) + p[:, 1])

    def test_norm_rows(self):
        rng = np.random.default_rng(3)
        a = rng.normal(size=(50000, 3)) * rng.uniform(0, 10, (50000, 1))
        s = a * a
        np.testing.assert_array_equal(np.linalg.norm(a, axis=1), np.sqrt((s[:, 0] + s[:, 1]) + s[:, 2]))


@unittest.skipUnless(geometry.rust_available() and hasattr(geometry._core, "CutKernel"),
                     "Rust extension not built")
class RustCutKernel(unittest.TestCase):
    """meshes with the cuts in the Rust kernel equal those of the Python cut"""

    def same(self, d, cfg):
        p = os.path.join(d, "m.json")
        with open(p, "w") as f:
            json.dump(cfg, f)
        A = mesher.generate(mesher.load_config(p), log=lambda s: None)
        with mock.patch.object(mesher, "_rust_cut_ok", lambda ctx, hk: False):
            B = mesher.generate(mesher.load_config(p), log=lambda s: None)
        np.testing.assert_array_equal(A["points"], B["points"])
        for k in ("owner", "neighbour"):
            np.testing.assert_array_equal(np.asarray(A[k]), np.asarray(B[k]))
        fa, fb = A["faces"], B["faces"]
        self.assertEqual(len(fa), len(fb))
        self.assertTrue(all(list(x) == list(y) for x, y in zip(fa, fb)))

    def test_turned_cube(self):
        # convex feature edges: chord points, split caps, the exact feature cut
        from test_exact_cut import turned_cube
        with tempfile.TemporaryDirectory() as d:
            turned_cube(os.path.join(d, "cube.stl"))
            for exact in (True, False):
                self.same(d, {"domain": "cube.stl", "baseCellSize": 0.15, "featureLevel": 1,
                              "surfaceRefinement": {"default": 1}, "exactFeatureCut": exact})

    def test_concave_channel(self):
        # a turned L-shaped channel: the concave edge (pinch points), and a zone
        with tempfile.TemporaryDirectory() as d:
            geometry.write_stl_ascii(os.path.join(d, "s.stl"), [("walls", l_prism())])
            self.same(d, {"domain": "s.stl", "baseCellSize": 0.008, "featureLevel": 2,
                          "surfaceRefinement": {"default": 1}, "exactFeatureCut": False,
                          "rotatingZones": [{"name": "z", "cylinder": {"point1": [0.06, 0.06, 0.0],
                                             "point2": [0.06, 0.06, 0.02], "radius": 0.015},
                                             "surfaceLevel": 1, "omega": 1.0}]})


def l_prism(c=0.037, L=0.1, H=0.03, ang=(7.0, 5.0, 11.0)):
    """L-shaped channel [0, L]^2 minus [0, c]^2 times [0, H], turned: outward triangles"""
    P = [(c, 0), (L, 0), (L, L), (c, L), (0, L), (0, c), (c, c)]
    top = [[(L, 0), (L, L), (c, L)], [(L, 0), (c, L), (c, c)], [(L, 0), (c, c), (c, 0)],
           [(0, c), (c, c), (c, L)], [(0, c), (c, L), (0, L)]]
    T = []
    for t in top:
        T.append([(x, y, H) for x, y in t])
        T.append([(x, y, 0.0) for x, y in t[::-1]])
    for i in range(len(P)):
        (x0, y0), (x1, y1) = P[i], P[(i + 1) % len(P)]
        T.append([(x0, y0, 0.0), (x1, y1, 0.0), (x1, y1, H)])
        T.append([(x0, y0, 0.0), (x1, y1, H), (x0, y0, H)])
    a, b, g = np.radians(ang)
    Rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    Ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    Rz = np.array([[np.cos(g), -np.sin(g), 0], [np.sin(g), np.cos(g), 0], [0, 0, 1]])
    return np.array(T, float) @ (Rz @ Ry @ Rx).T


if __name__ == "__main__":
    unittest.main()
