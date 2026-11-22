"""Faster code paths that must give the same answers as the plain ones: the open cell
edge count of the quality check, duplicate-free surface sampling, sorted-query search,
the Rust octree stages (2:1 balance, polyhedral cells, leaf validity) and the Rust cut
kernel of the domain cut (with the feature hooks).

python -m unittest discover -s tests
"""
import contextlib
import copy
import io
import itertools
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


@unittest.skipUnless(hasattr(geometry._core, "quality_metrics"), "Rust quality checks not built")
class RustQuality(unittest.TestCase):
    def compare_metrics(self, P, faces, own, nei):
        ff = quality.FlatFaces(faces)
        cf, sf = quality.face_geometry(P, ff)
        n_cells = int(max(own.max(), nei.max() if len(nei) else 0)) + 1
        c, _ = quality.cell_geometry(n_cells, own, nei, cf, sf)
        area = np.linalg.norm(sf, axis=1)
        # Both paths start from the same centres/areas, just as in the old Rust build.
        ref = quality._quality_metrics_py(P, ff, own, nei, cf, sf, c, area)
        got = geometry._core.quality_metrics(P, ff.ptr, ff.v, own, nei, cf, sf, c, area)
        for name, a, b in zip(("closed", "open", "duplicates", "cosine", "skew", "pyramids", "tets"),
                              got, ref):
            np.testing.assert_array_equal(a, b, err_msg=name)
        return got

    def test_warped_grid_and_boundary_only(self):
        rng = np.random.default_rng(17)
        for n in (1, 5):
            P, faces, own, nei, *_ = hex_grid(n)
            P += rng.normal(0, 0.35, P.shape)
            with self.subTest(n=n):
                self.compare_metrics(P, faces, own, nei)

    def test_broken_and_duplicate_edges(self):
        P, (_, verts), own, nei, *_ = hex_grid(2)
        faces = verts.reshape(-1, 4).tolist()
        faces[0].reverse()                       # internal face, both cell sides
        faces[1].pop()
        faces[-1].append(faces[-1][-1])           # a zero-length edge
        faces.append(faces[-1][:])
        got = self.compare_metrics(P, faces, np.append(own, own[-1]), nei)
        for i in (1, 2, 5, 6):
            self.assertGreater(got[i], 0)

    def test_empty_face_accepted(self):
        # face_geometry accepts a face without vertices, so the report must not fail on it.
        P, faces, own, nei, *_ = hex_grid(2)
        faces = [list(f) for f in quality.FlatFaces(faces).v.reshape(-1, 4)]
        faces.append([])
        res = quality.check(P, faces, np.append(own, own[-1]), nei, log=None)
        self.assertEqual(res["nFaces"], len(faces))
        self.assertEqual(res["openCellEdges"], 0)

    def test_report_without_extension(self):
        P, faces, own, nei, *_ = hex_grid(2)
        ff = quality.FlatFaces(faces)
        fast = quality.check(P, ff, own, nei, log=None)
        with mock.patch.object(quality, "_core", None):
            slow = quality.check(P, ff, own, nei, log=None)
        self.assertEqual(fast.keys(), slow.keys())
        for key in fast:
            if key != "_ff":
                np.testing.assert_array_equal(fast[key], slow[key], err_msg=key)
        self.assertIs(fast["_ff"], ff)


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

    def test_balance_after_split(self):
        # balanced, then a few leaves split: checking only the new leaves gives the
        # splits of the full check
        rng = np.random.default_rng(11)
        nsplit = 0
        for args in CASES:
            a = random_octree(*args)
            a.balance(lambda s: None)
            for _ in range(3):
                mark = (rng.random(len(a.lev)) < 0.02) & (a.lev < a.Lmax)
                a.split(mark)
                b = copy.deepcopy(a)
                self.assertGreaterEqual(a._new_from, 0)
                la, lb = [], []
                a.balance(la.append)                   # only the new leaves checked
                b._new_from = -1
                b.balance(lb.append)                   # every leaf checked
                np.testing.assert_array_equal(a.ijk, b.ijk)
                np.testing.assert_array_equal(a.lev, b.lev)
                self.assertEqual(la, lb)
                nsplit += len(la)
        self.assertGreater(nsplit, 0)

    @unittest.skipUnless(hasattr(geometry._core, "finer_face_neighbour"), "Rust extension not built")
    def test_finer_nearby(self):
        # the face-neighbour lookup answers "a finer leaf centre within 0.95 h" as the
        # KD-tree does, on balanced trees
        from scipy.spatial import cKDTree
        nfine = 0
        for args in CASES:
            oc = random_octree(*args)
            if oc.periodic.any():
                continue
            oc.balance(lambda s: None)
            c, hl = oc.centers(), oc.leaf_sizes()
            cand = np.nonzero(oc.lev < oc.Lmax)[0]
            ref = np.zeros(len(cand), bool)
            for lv in np.unique(oc.lev[cand]).tolist():
                m_ = oc.lev[cand] == lv
                fl = np.nonzero(oc.lev > lv)[0]
                if len(fl):
                    hc = float(hl[cand[m_]][0])
                    dd, _ = cKDTree(c[fl]).query(c[cand[m_]], distance_upper_bound=0.95 * hc)
                    ref[m_] = dd < 0.95 * hc
            np.testing.assert_array_equal(mesher._finer_nearby(oc, c, hl, cand), ref)
            nfine += int(ref.sum())
        self.assertGreater(nfine, 0)

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
    """the Rust ports repeat NumPy's summation order and CPython's set order; fails if
    NumPy / CPython (or the CPU's FMA path) changes them"""

    def test_matmul_rows(self):
        # (n, 1, 3) @ (n, 3, 1) adds the products as fma(a2, b2, fma(a1, b1, a0 b0));
        # checked here through exact rational arithmetic
        from fractions import Fraction as F
        rng = np.random.default_rng(4)
        d = rng.normal(size=(2000, 1, 3)) * rng.uniform(0, 10, (2000, 1, 1))
        s = rng.normal(size=(2000, 1, 3))
        m = (d @ s.transpose(0, 2, 1))[:, 0, 0]
        a, b = d[:, 0], s[:, 0]

        def fma(x, y, z):
            return float(F(x) * F(y) + F(z))
        ref = [fma(a[i, 2], b[i, 2], fma(a[i, 1], b[i, 1], a[i, 0] * b[i, 0])) for i in range(len(a))]
        np.testing.assert_array_equal(m, ref)

    @unittest.skipUnless(geometry.rust_available() and hasattr(geometry._core, "pyset_order"),
                         "Rust extension not built")
    def test_set_order(self):
        rng = np.random.default_rng(5)
        for _ in range(3000):
            keys = rng.integers(0, int(rng.choice([16, 1000, 10 ** 7])), int(rng.integers(1, 40))).tolist()
            s = set()
            for k in keys:
                s.add(k)
            self.assertEqual(list(s), geometry._core.pyset_order(keys))

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

    def test_matmul_shapes(self):
        # (m, 3) @ (3, k) (as X @ n[pl].T in featclip): per element fma(a2,b2, fma(a1,b1,
        # a0 b0)) when m, k >= 2 or m = k = 1, else fma(a2,b2, fma(a0,b0, a1 b1))
        # (featclip.rs mm)
        from fractions import Fraction as F

        def fma(x, y, z):
            return float(F(x) * F(y) + F(z))
        rng = np.random.default_rng(13)
        N = rng.normal(size=(40, 3))
        N /= np.linalg.norm(N, axis=1)[:, None]
        for m in (1, 2, 3, 8, 72, 130):
            for k in (1, 2, 3, 7):
                X = rng.normal(size=(m, 3)) * rng.uniform(0, 10, (m, 1))
                pl = rng.choice(40, k, replace=False)
                Y = X @ N[pl].T
                chain = (m >= 2 and k >= 2) or (m == 1 and k == 1)
                for i in range(m):
                    for j in range(k):
                        a, b = X[i], N[pl][j]
                        ref = (fma(a[2], b[2], fma(a[1], b[1], a[0] * b[0])) if chain
                               else fma(a[2], b[2], fma(a[0], b[0], a[1] * b[1])))
                        self.assertEqual(Y[i, j], ref, (m, k))

    def test_vector_products(self):
        # 1-D np.dot (and np.linalg.norm = its square root) is fma(a2,b2, fma(a1,b1, a0 b0));
        # a row of an (m, 3) @ (3,) product with m >= 2 is fma(a2,b2, fma(a0,b0, a1 b1))
        # (wallmerge.rs)
        from fractions import Fraction as F

        def fma(x, y, z):
            return float(F(x) * F(y) + F(z))
        rng = np.random.default_rng(6)
        for _ in range(400):
            a = rng.normal(size=3) * rng.uniform(0, 10)
            b = rng.normal(size=3)
            self.assertEqual(float(np.dot(a, b)), fma(a[2], b[2], fma(a[1], b[1], a[0] * b[0])))
            self.assertEqual(float(np.linalg.norm(a)), float(np.sqrt(fma(a[2], a[2], fma(a[1], a[1], a[0] * a[0])))))
            m = int(rng.integers(2, 40))
            M = rng.normal(size=(m, 3)) * rng.uniform(0, 10, (m, 1))
            ref = [fma(r[2], b[2], fma(r[0], b[0], r[1] * b[1])) for r in M]
            self.assertEqual((M @ b).tolist(), ref)


@unittest.skipUnless(geometry.rust_available() and hasattr(geometry._core, "CutKernel"),
                     "Rust extension not built")
class RustCutKernel(unittest.TestCase):
    """meshes with the cuts in the Rust kernel and the Rust merging equal those of the
    Python code"""

    def same(self, d, cfg):
        p = os.path.join(d, "m.json")
        with open(p, "w") as f:
            json.dump(cfg, f)
        A = mesher.generate(mesher.load_config(p), log=lambda s: None)

        real = mesher._core

        class Hidden:                          # the extension without the stage-A kernels
            def __getattr__(self, k):
                if k in ("merge_wall_faces", "contact_candidates", "surface_contact_samples",
                         "split_samples", "cut_batch_csr", "face_sign_counts", "vertex_min",
                         "leaf_any", "leaf_vertices", "finer_face_neighbour", "leaf_min", "leaf_mean", "feature_hook", "piece_geometry", "pair_faces", "first_two",
                         "pieces_volume_bad", "PlaneClipRs"):
                    raise AttributeError(k)
                return getattr(real, k)
        with mock.patch.object(mesher, "_rust_cut_ok", lambda ctx, hk: False), \
                mock.patch.object(mesher, "_rust_merge_ok", lambda: False), \
                mock.patch.object(mesher, "_core", Hidden()):
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


@unittest.skipUnless(geometry.rust_available() and hasattr(geometry._core.TriMesh, "through_wall"),
                     "Rust extension not built")
class RustLeakCheck(unittest.TestCase):
    def test_bypass_lines(self):
        # a plate thinner than a cell, vanished without fillThinWalls: lines through it
        from test_thin_gaps import voxel_stl
        from cutmesh.diagnose import through_wall
        h, v = 0.01, 0.0005
        G = np.ones((120, 120, 180), bool)
        G[:, 41:47, 0:120] = False
        with tempfile.TemporaryDirectory() as d:
            voxel_stl(os.path.join(d, "w.stl"), G, v)
            with open(os.path.join(d, "m.json"), "w") as f:
                json.dump({"domain": "w.stl", "baseCellSize": h, "surfaceRefinement": {"default": 0},
                           "featureLevel": 0, "fillThinWalls": 0, "checkLeaks": False}, f)
            R = mesher.generate(mesher.load_config(os.path.join(d, "m.json")), log=lambda s: None)
            dom = geometry.Surface(geometry.read_stl(os.path.join(d, "w.stl")))
        n1, X1 = through_wall(dom, R["points"], R["faces"], R["owner"], R["neighbour"])

        class NoKernel:                        # the Rust queries, without the fused kernel
            def __init__(self, rs):
                self.rs = rs

            def __getattr__(self, k):
                if k in ("through_wall", "through_wall_cells"):
                    raise AttributeError(k)
                return getattr(self.rs, k)
        dom._rs = NoKernel(dom._rs)
        n0, X0 = through_wall(dom, R["points"], R["faces"], R["owner"], R["neighbour"])
        self.assertGreater(n0, 0)
        self.assertEqual(n0, n1)
        np.testing.assert_array_equal(X0, X1)


@unittest.skipUnless(geometry.rust_available() and hasattr(geometry._core, "merge_wall_faces"),
                     "Rust extension not built")
class RustWallMerge(unittest.TestCase):
    """merge_coplanar_wall_faces in Rust gives the faces of the Python version"""

    def faces(self, seed):
        # a gently curved wall (folds of a few degrees around the 5 degree limit) split
        # into quads and triangles; runs of faces belong to one cell, some loops are
        # broken or reversed
        rng = np.random.default_rng(seed)
        n = 14
        x, y = np.meshgrid(np.arange(n + 1.0), np.arange(n + 1.0), indexing="ij")
        z = 0.03 * rng.uniform(0.5, 2.0) * (x - n / 2) ** 2 / n + rng.normal(0, 0.002, x.shape)
        P = np.stack([x, y, z], -1).reshape(-1, 3) * 0.01
        vid = np.arange((n + 1) ** 2).reshape(n + 1, n + 1)
        polys = []
        for i in range(n):
            for j in range(n):
                a, b, c, d = vid[i, j], vid[i + 1, j], vid[i + 1, j + 1], vid[i, j + 1]
                r = rng.uniform()
                if r < 0.3:
                    polys += [[a, b, c], [a, c, d]]
                elif r < 0.35:
                    polys += [[a, b, c, d][::-1]]
                else:
                    polys += [[a, b, c, d]]
        nI = 5
        internal = [[0, 1, 2]] * nI
        own, cell, k = [], 0, 0
        while k < len(polys):
            m = int(rng.integers(1, 9))
            own += [cell] * min(m, len(polys) - k)
            k += m
            cell += 1
        allp = internal + polys
        nn = np.array([len(q) for q in allp], np.int64)
        vv = np.array([v for q in allp for v in q], np.int64)
        owner = np.array(list(range(nI)) + own, np.int64)
        h = len(polys) // 2
        patches = [dict(name="a", type="wall", startFace=nI, nFaces=h),
                   dict(name="b", type="wall", startFace=nI + h, nFaces=len(polys) - h)]
        return P, nn, vv, owner, nI, patches

    def test_same_faces(self):
        joined = 0
        for seed in range(25):
            args = self.faces(seed)
            for ang in (5.0, 2.0, 20.0):
                a = mesher.merge_coplanar_wall_faces(*args, max_angle=ang)
                b = mesher._merge_coplanar_wall_faces_py(*args, max_angle=ang)
                for x, y in zip(a[:3], b[:3]):
                    np.testing.assert_array_equal(x, y)
                self.assertEqual(a[3], b[3])
                self.assertEqual(a[4], b[4])
                joined += a[4]
        self.assertGreater(joined, 100)


@unittest.skipUnless(geometry.rust_available() and hasattr(geometry._core, "cut_batch_csr"),
                     "Rust extension not built")
class RustCutBatch(unittest.TestCase):
    """the parallel cut (cut_batch_csr) numbers vertices and faces as the sequential one"""

    def test_same_as_sequential(self):
        n = 9
        vid = np.arange((n + 1) ** 3).reshape(n + 1, n + 1, n + 1)
        g = np.stack(np.meshgrid(*[np.arange(n + 1.0)] * 3, indexing="ij"), -1).reshape(-1, 3) / n
        fkey, cells = {}, []

        def face(q):
            k = tuple(sorted(q))
            return fkey.setdefault(k, len(fkey))
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    v = vid[i:i + 2, j:j + 2, k:k + 2]
                    quads = [[v[0, 0, 0], v[0, 0, 1], v[0, 1, 1], v[0, 1, 0]], [v[1, 0, 0], v[1, 1, 0], v[1, 1, 1], v[1, 0, 1]],
                             [v[0, 0, 0], v[1, 0, 0], v[1, 0, 1], v[0, 0, 1]], [v[0, 1, 0], v[0, 1, 1], v[1, 1, 1], v[1, 1, 0]],
                             [v[0, 0, 0], v[0, 1, 0], v[1, 1, 0], v[1, 0, 0]], [v[0, 0, 1], v[1, 0, 1], v[1, 1, 1], v[0, 1, 1]]]
                    cells.append([(face(q), [int(x) for x in q]) for q in quads])
        for r, snap in ((0.37, 0.0), (0.41, 0.03)):
            phi = np.linalg.norm(g - 0.5, axis=1) - r
            phi[np.abs(phi) < snap] = 0.0              # vertices on the surface
            for shared in (True, False):
                nid = len(fkey)
                P = mesher._Pieces.from_lists([[c, 0, 0] for c in cells])
                (it, side, cf, ck, cv, cvv), np1, par1, nid1 = geometry._core.cut_batch_csr(
                    g, phi, nid, *P.csr(), shared, {})
                res, np2, par2, nid2 = geometry._core.cut_batch(g, phi, nid, cells, shared, {})
                self.assertEqual((nid1, par1), (nid2, par2))
                np.testing.assert_array_equal(np1, np2)
                flat = [(k, s, comp) for k, (negc, posc) in enumerate(res)
                        for s, cs in ((0, negc), (1, posc)) for comp in cs]
                A = mesher._Pieces(it, side, cf, ck, cv, cvv)
                self.assertEqual(len(flat), len(A))
                for i, (k, s, comp) in enumerate(flat):
                    self.assertEqual((A.leaf[i], A.zone[i]), (k, s))
                    self.assertEqual(A.faces(i), [(f, list(p)) for f, p in comp])
                self.assertGreater(len(np1), 10)


@unittest.skipUnless(geometry.rust_available() and hasattr(geometry._core, "leaf_vertices"),
                     "Rust extension not built")
class RustFaceOps(unittest.TestCase):
    """the per-face / per-leaf reductions in Rust give the NumPy answers of the code
    they replace in mesher.generate"""

    def faces(self, seed):
        rng = np.random.default_rng(seed)
        nv, nleaf = 500, 120
        fn = rng.integers(3, 7, 900)                       # face sizes
        fptr = np.concatenate([[0], np.cumsum(fn)]).astype(np.int64)
        fverts = rng.integers(0, nv, int(fptr[-1])).astype(np.int64)
        cut = np.sort(rng.choice(np.arange(1, len(fn)), nleaf - 1, replace=False))
        cptr = np.concatenate([[0], cut, [len(fn)]]).astype(np.int64)   # faces by leaf
        return rng, nv, fn, fptr, fverts, cptr

    def test_face_signs_and_vertex_min(self):
        for seed in range(5):
            rng, nv, fn, fptr, fverts, cptr = self.faces(seed)
            phi = rng.choice([-1.0, 0.0, 0.0, 2.0, np.nan], nv) * rng.random(nv)
            got = geometry._core.face_sign_counts(phi, fptr, fverts)
            pv = phi[fverts]
            for g, m in zip(got, (pv < 0, pv > 0, pv == 0)):
                np.testing.assert_array_equal(g, np.add.reduceat(m.astype(np.int32), fptr[:-1]))
            val = rng.random(len(fn))
            val[rng.random(len(fn)) < 0.05] = np.nan
            ref = np.full(nv + 3, np.inf)                  # vertices in no face stay inf
            np.minimum.at(ref, fverts, np.repeat(val, fn))
            np.testing.assert_array_equal(geometry._core.vertex_min(nv + 3, fptr, fverts, val), ref)

    @unittest.skipUnless(hasattr(geometry._core, "pieces_volume_bad"), "Rust extension not built")
    def test_pieces_volume_bad(self):
        # mesher._piece_volume / _piece_bad of the clipped cells' pieces, the points read
        # from a _PosList (its array, its array block and its list of added rows)
        rng = np.random.default_rng(5)
        nbad = 0
        for t in range(40):
            pos = mesher._PosList(rng.random((30, 3)))
            pos.extend(rng.random((10, 3)))
            pos.extend([list(x) for x in rng.random((10, 3))])
            pos.append([float(x) for x in rng.random(3)])
            comps = []
            for _ in range(int(rng.integers(1, 4))):
                comps.append([(int(f), [int(v) for v in rng.choice(len(pos), int(rng.integers(3, 7)), replace=False)])
                              for f in range(int(rng.integers(4, 9)))])
            if t % 5 == 0:                          # a real cell: a unit cube's faces
                q = [[0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1], [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]]
                base = len(pos)
                for x in itertools.product((0.0, 1.0), repeat=3):
                    pos.append([x[2], x[1], x[0]])
                comps.append([(f, [base + v for v in p]) for f, p in enumerate(q)])
            h = float(rng.random()) + 0.1
            got = geometry._core.pieces_volume_bad(pos.a, pos.b, pos.added, comps, h)
            ref = [(mesher._piece_volume(pos, c), mesher._piece_bad(pos, c, h)) for c in comps]
            self.assertEqual(got, ref)
            nbad += sum(b for _, b in got)
        self.assertGreater(nbad, 0)
        with self.assertRaises(ValueError):
            geometry._core.pieces_volume_bad(pos.a, pos.b, pos.added, [[(0, [0, 1, len(pos)])]], 1.0)

    def test_piece_geometry(self):
        # volumes and "bad" flags of pieces as the NumPy code of the merging stage
        rng = np.random.default_rng(21)
        P = rng.normal(size=(400, 3))
        pieces = []
        for _ in range(300):
            nf = int(rng.integers(1, 8))
            pieces.append([list(rng.choice(400, int(rng.integers(3, 7)), replace=False)) for _ in range(nf)])
        fptr = np.concatenate([[0], np.cumsum([len(p) for p in pieces])]).astype(np.int64)
        vn = np.array([len(f) for p in pieces for f in p], np.int64)
        vptr = np.concatenate([[0], np.cumsum(vn)]).astype(np.int64)
        verts = np.array([v for p in pieces for f in p for v in f], np.int64)
        hp = rng.uniform(0.1, 2.0, len(pieces))
        nP = len(pieces)
        py_cell = np.repeat(np.arange(nP), np.diff(fptr))
        Cf, Sf = quality.face_geometry(P, quality.FlatFaces((vn, verts)))
        cnt = np.bincount(py_cell, minlength=nP)
        est = np.stack([np.bincount(py_cell, Cf[:, k], nP) for k in range(3)], 1) / np.maximum(cnt, 1)[:, None]
        pv = np.einsum("ij,ij->i", Sf, Cf - est[py_cell])
        vol = np.bincount(py_cell, pv, nP) / 3.0
        w = pv[:, None] * (0.75 * Cf + 0.25 * est[py_cell])
        cc = np.stack([np.bincount(py_cell, w[:, k], nP) for k in range(3)], 1)
        cc /= np.where(np.abs(vol) > 0, 3.0 * vol, 1.0)[:, None]
        selfpyr = np.einsum("ij,ij->i", Sf, Cf - cc[py_cell])
        bad = np.bincount(py_cell, (selfpyr < 0.05 * np.linalg.norm(Sf, axis=1) * hp[py_cell]).astype(int), nP) > 0
        v2, b2 = geometry._core.piece_geometry(P, fptr, vptr, verts, hp)
        np.testing.assert_array_equal(v2, vol)
        np.testing.assert_array_equal(b2, bad)
        self.assertTrue(bad.any() and not bad.all())

    def test_pair_faces(self):
        rng = np.random.default_rng(22)
        for n, nkey in ((0, 1), (50, 30), (20000, 9000)):
            key = rng.integers(0, nkey, n).astype(np.int64)
            cell = rng.integers(-1, n // 3 + 2, n).astype(np.int64)
            logs = []
            ref = mesher._pair_faces_py(key, cell, logs.append)
            got = geometry._core.pair_faces(key, cell)
            for a, b in zip(ref, got[:5]):
                np.testing.assert_array_equal(a, b)
            self.assertEqual(bool(logs), got[5] > 0)

    def test_first_two(self):
        rng = np.random.default_rng(23)
        for n, m in ((5, 0), (50, 200), (3000, 9000)):
            kk0 = rng.integers(0, n, m).astype(np.int64)
            kc0 = rng.integers(0, 10 ** 6, m).astype(np.int64)
            kb1 = np.full(n, -1, np.int64); kb2 = np.full(n, -1, np.int64)
            o_ = np.argsort(kk0, kind="stable"); kk = kk0[o_]; kc = kc0[o_]
            first = np.ones(len(kk), bool); first[1:] = kk[1:] != kk[:-1]
            kb1[kk[first]] = kc[first]
            kb2[kk[~first]] = kc[~first]
            a, b = geometry._core.first_two(kk0, kc0, n)
            np.testing.assert_array_equal(a, kb1)
            np.testing.assert_array_equal(b, kb2)

    def test_leaf_any_and_vertices(self):
        for seed in range(5):
            rng, nv, fn, fptr, fverts, cptr = self.faces(seed)
            vptr = fptr[cptr]
            f_cell = np.repeat(np.arange(len(cptr) - 1), np.diff(cptr))
            for p in (0.01, 0.2, 0.9):
                m = rng.random(nv) < p
                np.testing.assert_array_equal(geometry._core.leaf_any(m, vptr, fverts),
                                              np.logical_or.reduceat(m[fverts], vptr[:-1]))
                lost = rng.random(len(cptr) - 1) < p
                v = fverts[lost[np.repeat(f_cell, fn)]]
                np.testing.assert_array_equal(geometry._core.leaf_vertices(lost, vptr, fverts), geometry.unique(v))
                np.testing.assert_array_equal(geometry._core.leaf_vertices(lost, vptr, fverts, m),
                                              geometry.unique(v[m[v]]))
            x = rng.standard_normal(nv)
            x[rng.random(nv) < 0.01] = np.nan
            np.testing.assert_array_equal(geometry._core.leaf_min(x, vptr, fverts),
                                          np.minimum.reduceat(np.minimum.reduceat(x[fverts], fptr[:-1]), cptr[:-1]))
            P = rng.normal(size=(nv, 3)) * 10.0
            lv = np.sort(rng.choice(len(cptr) - 1, 40, replace=False)).astype(np.int64)
            occ = np.concatenate([np.arange(vptr[c], vptr[c + 1]) for c in lv])
            own = np.repeat(np.arange(len(lv)), vptr[lv + 1] - vptr[lv])
            X = P[fverts[occ]]
            ref = np.stack([np.bincount(own, X[:, k], len(lv)) for k in range(3)], 1)
            ref /= np.bincount(own, minlength=len(lv))[:, None]
            np.testing.assert_array_equal(geometry._core.leaf_mean(P, lv, vptr, fverts), ref)


@unittest.skipUnless(geometry.rust_available() and hasattr(geometry._core, "PlaneClipRs"),
                     "Rust extension not built")
class RustPlaneClip(unittest.TestCase):
    """featclip.PlaneClip in Rust (attach_rust) answers as the Python code"""

    def test_same_as_python(self):
        from test_exact_cut import turned_cube
        from cutmesh.featclip import PlaneClip
        with tempfile.TemporaryDirectory() as d:
            turned_cube(os.path.join(d, "c.stl"))
            dom = geometry.Surface(geometry.read_stl(os.path.join(d, "c.stl")))
        feat = geometry.FeatureSet(dom.feature_edges(30.0), L=dom.L)
        rng = np.random.default_rng(12)
        h = 0.11
        g = np.arange(-1.0, 1.0 + 1e-9, h) + 0.013
        n = len(g)
        pts = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
        phi = dom.sdf(pts)
        snap = np.abs(phi) < 0.25 * h                   # vertices on the walls, as snapped
        cp, _, _ = dom.closest(pts[snap])
        pts[snap] = cp
        phi[snap] = 0.0
        A = PlaneClip(dom, pts, phi, h, feat)
        B = PlaneClip(dom, pts, phi, h, feat)
        A.zero_edge = B.zero_edge = {}
        B.attach_rust(geometry._core)
        self.assertIsNotNone(B._rs)
        vid = np.arange(n ** 3).reshape(n, n, n)
        faces, cells = {}, []
        for i in range(n - 1):
            for j in range(n - 1):
                for k in range(n - 1):
                    v = vid[i:i + 2, j:j + 2, k:k + 2]
                    if np.abs(phi[v.ravel()]).min() > 1.5 * h:
                        continue
                    quads = [[v[0, 0, 0], v[0, 0, 1], v[0, 1, 1], v[0, 1, 0]], [v[1, 0, 0], v[1, 1, 0], v[1, 1, 1], v[1, 0, 1]],
                             [v[0, 0, 0], v[1, 0, 0], v[1, 0, 1], v[0, 0, 1]], [v[0, 1, 0], v[0, 1, 1], v[1, 1, 1], v[1, 1, 0]],
                             [v[0, 0, 0], v[0, 1, 0], v[1, 1, 0], v[1, 0, 0]], [v[0, 0, 1], v[1, 0, 1], v[1, 1, 1], v[0, 1, 1]]]
                    fl = [(faces.setdefault(tuple(sorted(int(x) for x in q)), len(faces)), [int(x) for x in q])
                          for q in quads]
                    cells.append(fl)
        ncut = 0
        for fl in cells:
            for fid, poly in fl:
                for p in (poly, poly[::-1]):          # the other cell sees it reversed
                    ra, pa = A.clip_face(fid, p)
                    rb, pb = B.clip_face(fid, p)
                    self.assertEqual(ra, rb)
                    np.testing.assert_array_equal(np.reshape(pa, (-1, 3)), np.reshape(pb, (-1, 3)))
                    self.assertEqual(A.ls_compatible(fid, p), B.ls_compatible(fid, p))
                    ncut += any(r[0] != "v" for r in ra)
                self.assertEqual(list(A.face_planes.get(fid, ())), list(B.face_planes.get(fid, ())))
        self.assertGreater(ncut, 50)
        Ps = [pts[np.unique([v for _, p in fl for v in p])] for fl in cells]
        rs = [1.5 * float(np.linalg.norm(P - P.mean(0), axis=1).max()) for P in Ps]
        self.assertEqual(A.convex_many(Ps, rs), B.convex_many(Ps, rs))
        for P, r in list(zip(Ps, rs))[::7]:
            ca, pa = A.convex_at(P, r)
            cb, pb = B.convex_at(P, r)
            self.assertEqual((ca, list(pa)), (cb, list(pb)))
        self.assertTrue(any(B.convex_many(Ps, rs)) and not all(B.convex_many(Ps, rs)))
        for _ in range(200):
            tris = np.sort(rng.choice(len(dom.tris), int(rng.integers(1, 9)), replace=False))
            pl = A.planes(tris)
            self.assertEqual(pl, B.planes(tris))
            np.testing.assert_array_equal(A.groups(pl), B.groups(pl))

    @staticmethod
    def _grid_cells(T, h, shift):
        """a regular grid around the STL triangles T, its vertices near the walls snapped
        onto them, and the hex cells near the walls as [(face id, polygon)]"""
        from cutmesh.featclip import PlaneClip
        with tempfile.TemporaryDirectory() as d:
            geometry.write_stl_ascii(os.path.join(d, "s.stl"), [("walls", T)])
            dom = geometry.Surface(geometry.read_stl(os.path.join(d, "s.stl")))
        feat = geometry.FeatureSet(dom.feature_edges(30.0), L=dom.L)
        g = np.arange(-1.0, 1.0 + 1e-9, h) + shift
        n = len(g)
        pts = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
        phi = dom.sdf(pts)
        snap = np.abs(phi) < 0.25 * h
        cp, _, _ = dom.closest(pts[snap])
        pts[snap] = cp
        phi[snap] = 0.0
        vid = np.arange(n ** 3).reshape(n, n, n)
        faces, cells = {}, []
        for i in range(n - 1):
            for j in range(n - 1):
                for k in range(n - 1):
                    v = vid[i:i + 2, j:j + 2, k:k + 2]
                    if np.abs(phi[v.ravel()]).min() > 1.5 * h:
                        continue
                    quads = [[v[0, 0, 0], v[0, 0, 1], v[0, 1, 1], v[0, 1, 0]], [v[1, 0, 0], v[1, 1, 0], v[1, 1, 1], v[1, 0, 1]],
                             [v[0, 0, 0], v[1, 0, 0], v[1, 0, 1], v[0, 0, 1]], [v[0, 1, 0], v[0, 1, 1], v[1, 1, 1], v[1, 1, 0]],
                             [v[0, 0, 0], v[0, 1, 0], v[1, 1, 0], v[1, 0, 0]], [v[0, 0, 1], v[1, 0, 1], v[1, 1, 1], v[0, 1, 1]]]
                    cells.append([(faces.setdefault(tuple(sorted(int(x) for x in q)), len(faces)), [int(x) for x in q])
                                  for q in quads])

        def clip():
            pc = PlaneClip(dom, pts, phi, h, feat)
            pc.zero_edge = {}
            return pc
        return clip, pts, cells, len(faces)

    @staticmethod
    def _l_prism():
        """an L-shaped prism, turned: convex edges and a concave one"""
        L = [(-0.7, -0.7), (0.7, -0.7), (0.7, 0.0), (0.0, 0.0), (0.0, 0.7), (-0.7, 0.7)]
        z0, z1 = -0.6, 0.6
        T = []
        for i in range(6):
            (x0, y0), (x1, y1) = L[i], L[(i + 1) % 6]
            T += [[(x0, y0, z0), (x1, y1, z0), (x1, y1, z1)], [(x0, y0, z0), (x1, y1, z1), (x0, y0, z1)]]
        for i in (4, 5, 0, 1):                     # fan from the reflex corner (0, 0)
            (x0, y0), (x1, y1) = L[i], L[(i + 1) % 6]
            T += [[(0.0, 0.0, z1), (x0, y0, z1), (x1, y1, z1)], [(0.0, 0.0, z0), (x1, y1, z0), (x0, y0, z0)]]
        a, b, c = np.radians([19.0, 27.0, 11.0])
        Rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
        Ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
        Rz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
        return np.array(T, float) @ (Rz @ Ry @ Rx).T

    @unittest.skipUnless(hasattr(geometry._core, "PlaneClipRs") and hasattr(geometry._core.PlaneClipRs, "select_cells"),
                         "Rust extension not built")
    def test_select_cells(self):
        """mesher.select_clip_cells in Rust (PlaneClipRs.select_cells): the same cells,
        mixed faces, drops and shared faces, and the clip caches as the Python code
        leaves them"""
        from test_exact_cut import turned_cube
        with tempfile.TemporaryDirectory() as d:
            cube = turned_cube(os.path.join(d, "c.stl"))
        seen = [0, 0, 0]
        for T, h in ((cube, 0.11), (self._l_prism(), 0.1)):
            clip, pts, cells, nfaces = self._grid_cells(T, h, 0.013)
            A, B = clip(), clip()
            B.attach_rust(geometry._core)
            cptr = mesher._ptr([len(fl) for fl in cells])
            f_key = np.array([fid for fl in cells for fid, _ in fl], np.int64)
            fptr = mesher._ptr([len(p) for fl in cells for _, p in fl])
            fverts = np.array([v for fl in cells for _, p in fl for v in p], np.int64)
            f_cell = np.repeat(np.arange(len(cells)), np.diff(cptr))
            cof = mesher._faces_cells(f_key, f_cell, nfaces)
            self.assertIsNotNone(getattr(cof, "k12", None))
            cand = [c for c in range(len(cells)) if c % 3 != 1] + [5, 0]
            special = set(range(0, len(cells), 17))
            E, mixed, nback = mesher.select_clip_cells(A, cand, lambda c: cells[c], cof, special, pts)
            E2, mixed2, nback2, plain2 = B._rs.select_cells(cand, sorted(special), cptr, fptr, fverts, f_key, *cof.k12)
            self.assertEqual((sorted(E), sorted(mixed), nback), (E2, mixed2, nback2))
            self.assertEqual(sorted(set(fid for c in E for fid, _ in cells[c] if any(o not in E for o in cof(fid)))),
                             plain2)
            seen = [seen[0] + len(E), seen[1] + len(mixed), seen[2] + nback]
            for fl in cells:
                for fid, poly in fl:
                    ra, pa = A.clip_face(fid, poly)
                    rb, pb = B.clip_face(fid, poly)
                    self.assertEqual(ra, rb)
                    np.testing.assert_array_equal(np.reshape(pa, (-1, 3)), np.reshape(pb, (-1, 3)))
        self.assertTrue(all(seen), seen)


@unittest.skipUnless(geometry.rust_available() and hasattr(geometry._core, "feature_hook"),
                     "Rust extension not built")
class RustFeatureHook(unittest.TestCase):
    """the cap hook in Rust (fhook.rs) gives the polygons and new points of the Python
    hook, on the loops the meshing of two shapes hands it"""

    def capture(self, d, cfg):
        calls = []
        orig = mesher.make_feature_hook

        def capturing(feat, h_of_cell, *a, **k):
            res = orig(feat, h_of_cell, *a, **k)
            if res is None:
                return res
            hook, st = res

            def wrapped(ctx, loop):
                calls.append((feat, h_of_cell, st["cell"], list(loop), [list(map(float, ctx.pos[v])) for v in loop],
                              [v in ctx.onfeat for v in loop]))
                return hook(ctx, loop)
            wrapped.feat, wrapped.reach = hook.feat, hook.reach
            return wrapped, st
        p = os.path.join(d, "m.json")
        with open(p, "w") as f:
            json.dump(cfg, f)
        with mock.patch.object(mesher, "make_feature_hook", capturing):
            mesher.generate(mesher.load_config(p), log=lambda s: None)
        return calls

    def run_hook(self, feat, h_of_cell, cell, loop, P, flags, rust):
        real = mesher._core

        class Hidden:
            def __getattr__(self, k):
                if k == "feature_hook":
                    raise AttributeError(k)
                return getattr(real, k)
        base = 10 ** 6
        pos = {v: x for v, x in zip(loop, P)}

        class Pos:
            def __init__(self):
                self.added = []

            def __getitem__(self, v):
                return pos[v] if v < base else self.added[v - base]

            def __len__(self):
                return base + len(self.added)

            def append(self, x):
                self.added.append([float(c) for c in x])
        ctx = mesher._HookCtx(Pos(), [], mesher._OnFeat(v for v, f in zip(loop, flags) if f))
        with mock.patch.object(mesher, "_core", real if rust else Hidden()):
            hook, st = mesher.make_feature_hook(feat, h_of_cell)
            st["cell"] = cell
            polys = hook(ctx, loop)
        return [list(q) for q in polys], ctx.pos.added, list(ctx.onfeat.new), bool(st.get("did")), st["n"]

    def test_same_as_python(self):
        from test_exact_cut import turned_cube
        calls = []
        with tempfile.TemporaryDirectory() as d:
            turned_cube(os.path.join(d, "cube.stl"))
            for exact in (True, False):
                calls += self.capture(d, {"domain": "cube.stl", "baseCellSize": 0.15, "featureLevel": 1,
                                          "surfaceRefinement": {"default": 1}, "exactFeatureCut": exact})
            geometry.write_stl_ascii(os.path.join(d, "s.stl"), [("walls", l_prism())])
            calls += self.capture(d, {"domain": "s.stl", "baseCellSize": 0.008, "featureLevel": 2,
                                      "surfaceRefinement": {"default": 1}, "exactFeatureCut": False})
        changed = 0
        for c in calls:
            a = self.run_hook(*c, rust=False)
            b = self.run_hook(*c, rust=True)
            self.assertEqual(a, b)
            changed += a[3]
        self.assertGreater(len(calls), 50)
        self.assertGreater(changed, 10)


@unittest.skipUnless(geometry.rust_available() and hasattr(geometry._core, "argsort_stable"),
                     "Rust extension not built")
class RustArgsort(unittest.TestCase):
    def test_stable(self):
        rng = np.random.default_rng(7)
        for n, hi in ((0, 1), (1, 5), (1000, 3), (300000, 1000), (300000, 2 ** 40)):
            a = rng.integers(-hi, hi, n).astype(np.int64)
            np.testing.assert_array_equal(mesher._argsort_stable(a), np.argsort(a, kind="stable"))
            np.testing.assert_array_equal(geometry._core.argsort_stable(a), np.argsort(a, kind="stable"))


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
