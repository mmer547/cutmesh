"""Rust extension (cutmesh._core) against the Python geometry code.

python -m unittest discover -s tests   (skipped when the extension is not built)
"""
import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from cutmesh import geometry as G     # noqa: E402

STL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples", "stirred_tank", "tank.stl")


@unittest.skipUnless(G.rust_available(), "Rust extension not built (python tools/build_rust.py)")
class RustMatchesPython(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dom = G.Surface(G.read_stl(STL))
        cls.dom.build_search(0.002)
        rng = np.random.default_rng(0)
        lo, hi = cls.dom.bbmin, cls.dom.bbmax
        P = lo - 0.05 * (hi - lo) + rng.random((20000, 3)) * 1.1 * (hi - lo)
        t = cls.dom.tris[rng.integers(0, len(cls.dom.tris), 20000)]
        w = rng.dirichlet([1, 1, 1], 20000)
        Q = np.einsum("nk,nkj->nj", w, t) + rng.normal(0, 1e-3, (20000, 3))
        cls.X = np.concatenate([P, Q])

    def python_surface(self):
        d = G.Surface.__new__(G.Surface)
        d.__dict__.update(self.dom.__dict__)
        d._rs = None
        d._nb = None
        return d

    def test_closest(self):
        _, d1, _ = self.python_surface().closest(self.X)
        cp2, d2, t2 = self.dom.closest(self.X)
        np.testing.assert_allclose(d2, d1, rtol=0, atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(cp2 - self.X, axis=1), d2, atol=1e-12)
        self.assertTrue(((t2 >= 0) & (t2 < len(self.dom.tris))).all())

    def test_inside_identical(self):
        s1 = self.python_surface().inside(self.X)
        s2 = self.dom.inside(self.X)
        self.assertEqual(int((s1 != s2).sum()), 0)

    @unittest.skipUnless(hasattr(getattr(G._core, "TriMesh", None), "inside_bvh"), "Rust extension not built")
    def test_inside_grid_matches_bvh(self):
        # the (y, z) grid of BVH leaves tests the leaves the traversal tests: the same
        # parity, also for probes on the triangles' vertices, edges and box borders
        rng = np.random.default_rng(5)
        T = self.dom.tris
        V = T.reshape(-1, 3)
        mids = 0.5 * (T[:, 0] + T[:, 1])
        Y = np.concatenate([self.X, V, mids, V + rng.normal(0, 1e-12, V.shape),
                            np.stack([V[:, 0] - 1.0, V[:, 1], V[:, 2]], 1)])
        for k in range(3):
            Z = Y.copy()
            Z[:, k] += rng.choice([-1.0, 1.0], len(Z)) * rng.random(len(Z)) * 1e-9
            np.testing.assert_array_equal(self.dom._rs.inside(Z), self.dom._rs.inside_bvh(Z))
        far = np.array([[1e9, 0, 0], [0, 1e9, 0], [0, 0, -1e9], [np.nan, 0, 0]])
        np.testing.assert_array_equal(self.dom._rs.inside(far), self.dom._rs.inside_bvh(far))
        self.assertTrue(self.dom._rs.inside_grid())
        # solids in a row along x overlap in (y, z): no grid (its registrations would
        # grow with the square of the triangles), the BVH answers
        V = np.array([[x, y, z] for z in (0, 1) for y in (0, 1) for x in (0, 1)], float)
        F = np.array([(0, 2, 1), (1, 2, 3), (4, 5, 6), (5, 7, 6), (0, 1, 4), (1, 5, 4),
                      (2, 6, 3), (3, 6, 7), (0, 4, 2), (2, 4, 6), (1, 3, 5), (3, 7, 5)])
        T = np.concatenate([V[F] + [2.0 * i, 0, 0] for i in range(3000)])
        m = G._core.TriMesh(np.ascontiguousarray(T))
        self.assertFalse(m.inside_grid())
        q = rng.random((5000, 3)) * [6000, 1, 1]
        np.testing.assert_array_equal(m.inside(q), m.inside_bvh(q))
        self.assertTrue(m.inside(np.array([[0.5, 0.5, 0.5], [3000.5, 0.5, 0.5]])).all())

    @unittest.skipUnless(hasattr(G._core, "unique_points"), "Rust extension not built")
    def test_unique_points(self):
        # the KD-tree points of the refinement sources: the distinct rows (any order)
        from scipy.spatial import cKDTree
        from cutmesh import mesher
        rng = np.random.default_rng(6)
        p = np.round(rng.random((20000, 3)), 3)
        p = np.concatenate([p, p[:5000], -0.0 * p[:10]])
        a = G._core.unique_points(p)
        b = mesher._unique_points(p)
        self.assertEqual(len(a), len(b))
        np.testing.assert_array_equal(np.unique(a.view(np.uint64), axis=0), np.unique(b.view(np.uint64), axis=0))
        q = rng.random((3000, 3))
        k = mesher._points_kd(p)
        np.testing.assert_array_equal(k.query(q, distance_upper_bound=0.01)[0],
                                      cKDTree(b).query(q, distance_upper_bound=0.01)[0])

    @unittest.skipUnless(hasattr(getattr(G._core, "TriMesh", None), "samples_by_triangle"), "Rust extension not built")
    def test_samples_by_triangle(self):
        # samples(ordered=False): the points and triangle ids of samples(), as a multiset
        for spacing, regions in ((0.004, None), (0.0015, [0]), (0.01, [1])):
            a, ia = self.dom.samples(spacing, regions)
            b, ib = self.dom.samples(spacing, regions, ordered=False)
            ka = np.concatenate([a.view(np.int64), ia[:, None]], 1)
            kb = np.concatenate([b.view(np.int64), ib[:, None]], 1)
            self.assertGreater(len(ka), 1000)
            np.testing.assert_array_equal(ka[np.lexsort(ka.T[::-1])], kb[np.lexsort(kb.T[::-1])])

    def test_sdf(self):
        np.testing.assert_allclose(self.dom.sdf(self.X), self.python_surface().sdf(self.X), atol=1e-12)

    @unittest.skipUnless(hasattr(getattr(G._core, "TriMesh", None), "centroid_search"), "Rust centroid index not built")
    def test_release_search(self):
        # the centroid index let go after the snapping (memory) is built again if asked
        d = G.Surface(G.read_stl(STL))
        d.build_search(0.002)
        pts = self.X[::7]
        thr = np.full(len(pts), 0.004)
        before = d.approx_below(pts, thr)
        d.release_search()
        self.assertIsNone(d._centroid_search)
        np.testing.assert_array_equal(d.approx_below(pts, thr), before)
        self.assertIsNotNone(d._centroid_search)

    @unittest.skipUnless(hasattr(getattr(G._core, "TriMesh", None), "centroid_search"), "Rust centroid index not built")
    def test_centroid_thresholds(self):
        ref = self.python_surface()
        pts = self.X[::11]
        dist = ref.approx_distance(pts)
        self.assertEqual(self.dom._rmax, ref._rmax)
        for threshold in (0.0, 0.001, np.inf, np.nextafter(dist, 0), dist, np.nextafter(dist, np.inf)):
            np.testing.assert_array_equal(self.dom.approx_below(pts, threshold), dist < threshold)
        self.assertEqual(self.dom.approx_below(pts[:0], 0.01).shape, (0,))

    @unittest.skipUnless(hasattr(getattr(G._core, "TriMesh", None), "centroid_search"), "Rust centroid index not built")
    def test_rebuild_centroid_index(self):
        # A single triangle with tied longest edges, at nonzero world coordinates.
        t = np.array([[[5., 6., 7.], [6., 7., 7.], [6., 6., 8.]]])
        s = G.Surface([("wall", t)])
        for spacing in (0.24, 2.0):
            s.build_search(spacing)
            pieces, _ = G.subdivide_triangles(t, spacing)
            centres = pieces.mean(1)
            self.assertEqual(s._rmax, np.linalg.norm(pieces - centres[:, None], axis=2).max())
            np.testing.assert_array_equal(s.approx_below(centres, 1e-12), True)
            np.testing.assert_array_equal(s.approx_below(centres, 0.0), False)
            # Requesting distances after a rebuild must use this spacing as well.
            np.testing.assert_array_equal(s.approx_distance(centres), 0.0)

    @unittest.skipUnless(hasattr(getattr(G._core, "TriMesh", None), "samples"), "Rust sampling not built")
    def test_ordered_surface_samples(self):
        ref = self.python_surface()
        for spacing, regions in ((0.01, None), (0.007, [1, 3]), (1.0, [0]), (0.01, [])):
            with self.subTest(spacing=spacing, regions=regions):
                pts, ti = self.dom.samples(spacing, regions)
                want, parents = ref.samples(spacing, regions)
                np.testing.assert_array_equal(pts.view(np.uint64), want.view(np.uint64))
                np.testing.assert_array_equal(ti, parents)

    @unittest.skipUnless(hasattr(getattr(G._core, "TriMesh", None), "gap_widths"), "Rust gap widths not built")
    def test_batched_gap_widths(self):
        from cutmesh.mesher import _gap_widths
        rs = self.dom._rs
        previous = SimpleNamespace(inside=rs.inside, segment_hits=rs.segment_hits)
        ref = self.python_surface()
        p = self.dom.tris.mean(1)
        ti = np.arange(len(p))
        limits_level = ti % 3
        for eps in (0.0001, 0.1 * self.dom.inside_shift):
            got = _gap_widths(self.dom, p, ti, limits_level, 0.2, 2.0, eps)
            with mock.patch.object(self.dom, "_rs", previous):
                separate = _gap_widths(self.dom, p, ti, limits_level, 0.2, 2.0, eps)
            np.testing.assert_array_equal(got, separate)
            py = _gap_widths(ref, p, ti, limits_level, 0.2, 2.0, eps)
            np.testing.assert_allclose(got, py, rtol=0, atol=1e-12)
            self.assertTrue(np.isinf(got).any() and np.isfinite(got).any())

    def test_feature_segments(self):
        seg = self.dom.feature_edges(30.0)
        fr = G.FeatureSet(seg, L=self.dom.L)
        fp = G.FeatureSet(seg, L=self.dom.L)
        fp._rs = None
        c1, d1 = fp.closest(self.X)
        c2, d2 = fr.closest(self.X)
        np.testing.assert_allclose(d2, d1, atol=1e-12)
        np.testing.assert_allclose(c2, c1, atol=1e-9)


@unittest.skipUnless(G.rust_available(), "Rust extension not built (python tools/build_rust.py)")
class RustCutKernel(unittest.TestCase):
    def test_cut_batch_matches_cutcontext(self):
        from cutmesh.cutcell import CutContext, split_components
        # two unit cubes side by side (x in [0, 1] and [1, 2]), cut by a slanted plane
        xs, P = [0.0, 1.0, 2.0], []
        for x in xs:
            for y in (0.0, 1.0):
                for z in (0.0, 1.0):
                    P.append([x, y, z])
        P = np.array(P)
        vid = {tuple(p): i for i, p in enumerate(P.tolist())}
        v = lambda x, y, z: vid[(float(x), float(y), float(z))]

        def cube(x0, fid0):
            x1 = x0 + 1
            q = [[v(x0, 0, 0), v(x0, 0, 1), v(x0, 1, 1), v(x0, 1, 0)], [v(x1, 0, 0), v(x1, 1, 0), v(x1, 1, 1), v(x1, 0, 1)],
                 [v(x0, 0, 0), v(x1, 0, 0), v(x1, 0, 1), v(x0, 0, 1)], [v(x0, 1, 0), v(x0, 1, 1), v(x1, 1, 1), v(x1, 1, 0)],
                 [v(x0, 0, 0), v(x0, 1, 0), v(x1, 1, 0), v(x1, 0, 0)], [v(x0, 0, 1), v(x1, 0, 1), v(x1, 1, 1), v(x0, 1, 1)]]
            return [(fid0 + k, f) for k, f in enumerate(q)]
        c0 = cube(0, 0)
        c1 = cube(1, 6)
        c1[0] = (1, c0[1][1][::-1])        # shared face x = 1, same id, reversed
        phi = P[:, 0] * 0.4 + P[:, 2] - 0.8
        phi[v(1, 0, 0)] = 0.0              # a vertex on the surface
        for shared in (True, False):
            ctx = CutContext(P.tolist(), phi.tolist(), 100)
            ref = []
            for c in (c0, c1):
                neg, pos, _ = ctx.cut(c, shared)
                ref.append((split_components(neg), split_components(pos) if shared else []))
            res, newp, par, nid = G._core.cut_batch(P, phi, 100, [c0, c1], shared, {})
            norm = lambda r: [[[(int(f), list(map(int, p))) for f, p in comp] for comp in side] for side in r]
            self.assertEqual([norm(r) for r in res], [norm(r) for r in ref])
            np.testing.assert_allclose(newp, np.array(ctx.pos[len(P):]).reshape(-1, 3), atol=0)
            self.assertEqual([tuple(x) for x in par], [tuple(x) for x in ctx.new_parents])
            self.assertEqual(nid, ctx.next_id)


@unittest.skipUnless(G.rust_available(), "Rust extension not built (python tools/build_rust.py)")
class RustChordHook(unittest.TestCase):
    def test_pierce_matches_python(self):
        from cutmesh import mesher
        # unit square face in z = 0, cut along the chord from (1, 0.2) to (0.2, 1); the
        # kept vertex is (0, 0); a feature edge crosses the face at (0.8, 0.8)
        seg = np.array([[[0.8, 0.8, -1.0], [0.8, 0.8, 1.0]], [[5, 5, -1.0], [5, 5, 1.0]]])
        fr = G.FeatureSet(seg, L=10.0)
        fp = G.FeatureSet(seg, L=10.0)
        fp._rs = None

        class Ctx:
            pos = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.2, 0.0], [0.2, 1.0, 0.0]]
            phi = [-1.0, 1.0, 1.0, 1.0, 0.0, 0.0]
        poly = [0, 1, 2, 3]
        xr = mesher.make_chord_hook(fr)(Ctx, poly, 4, 5)
        xp = mesher.make_chord_hook(fp)(Ctx, poly, 4, 5)
        np.testing.assert_allclose(xr, [0.8, 0.8, 0.0], atol=1e-12)
        np.testing.assert_allclose(xr, xp, atol=1e-12)
        # the kept side bulging towards the feature point: no bend (non-convex)
        Ctx.phi = [1.0, 1.0, -1.0, 1.0, 0.0, 0.0]
        self.assertIsNone(mesher.make_chord_hook(fr)(Ctx, poly, 4, 5))
        self.assertIsNone(mesher.make_chord_hook(fp)(Ctx, poly, 4, 5))


@unittest.skipUnless(G.rust_available(), "Rust extension not built (python tools/build_rust.py)")
class RustFaceCellGeometry(unittest.TestCase):
    def setUp(self):
        from cutmesh import quality
        self.q = quality
        rng = np.random.default_rng(3)
        self.P = rng.random((400, 3))
        # random (also non-convex, non-planar) polygons of 3..9 vertices
        self.faces = [list(rng.choice(400, rng.integers(3, 10), replace=False)) for _ in range(500)]

    def python(self, fn, *a):
        saved = self.q._core
        self.q._core = None
        try:
            return fn(*a)
        finally:
            self.q._core = saved

    def test_face_geometry(self):
        ff = self.q.FlatFaces(self.faces)
        c1, s1 = self.python(self.q.face_geometry, self.P, ff)
        c2, s2 = self.q.face_geometry(self.P, ff)
        np.testing.assert_allclose(c2, c1, atol=1e-12)
        np.testing.assert_allclose(s2, s1, atol=1e-12)

    def test_cell_geometry(self):
        # a unit cube split into two cells by an internal face, as a polyMesh
        P = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
                      [0.5, 0, 0], [0.5, 1, 0], [0.5, 1, 1], [0.5, 0, 1.0]])
        faces = [[8, 9, 10, 11],                                  # internal, owner 0 -> 1
                 [0, 4, 7, 3], [0, 8, 11, 4], [3, 7, 10, 9], [0, 3, 9, 8], [4, 11, 10, 7],
                 [1, 2, 6, 5], [8, 1, 5, 11], [9, 10, 6, 2], [8, 9, 2, 1], [11, 5, 6, 10]]
        own = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]); nei = np.array([1])
        ff = self.q.FlatFaces(faces)
        Cf, Sf = self.q.face_geometry(P, ff)
        c1, v1 = self.python(self.q.cell_geometry, 2, own, nei, Cf, Sf)
        c2, v2 = self.q.cell_geometry(2, own, nei, Cf, Sf)
        np.testing.assert_allclose(v2, [0.5, 0.5], atol=1e-12)
        np.testing.assert_allclose(c2, [[0.25, 0.5, 0.5], [0.75, 0.5, 0.5]], atol=1e-12)
        np.testing.assert_allclose(v2, v1, atol=1e-14)
        np.testing.assert_allclose(c2, c1, atol=1e-14)

    def test_poly_cell(self):
        P = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1.0]])
        cube = [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [2, 3, 7, 6], [1, 2, 6, 5], [0, 4, 7, 3]]
        cf, sf, V, cc = G._core.poly_cell(P, cube)
        self.assertAlmostEqual(V, 1.0, places=12)
        np.testing.assert_allclose(cc, [0.5, 0.5, 0.5], atol=1e-12)
        c1, s1 = self.python(self.q.face_geometry, P, self.q.FlatFaces(cube))
        np.testing.assert_allclose(cf, c1, atol=1e-12)
        np.testing.assert_allclose(sf, s1, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
