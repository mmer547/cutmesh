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


if __name__ == "__main__":
    unittest.main()
