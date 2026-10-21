"""Rust extension (cutmesh._core) against the Python geometry code.

python -m unittest discover -s tests   (skipped when the extension is not built)
"""
import os
import sys
import unittest

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

    def test_sdf(self):
        np.testing.assert_allclose(self.dom.sdf(self.X), self.python_surface().sdf(self.X), atol=1e-12)

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
