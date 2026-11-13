"""Rust kernels of the boundary layers (jacobi_band, cells_bad) against the NumPy code
in layers.py, and add_layers with and without them.

python -m unittest discover -s tests   (skipped when the extension is not built)
"""
import os
import sys
import tempfile
import unittest

import numpy as np
from scipy import sparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cutmesh import layers as L                               # noqa: E402
from cutmesh.mesher import load_config, generate              # noqa: E402
from test_thin_gaps import voxel_stl, _write_json             # noqa: E402

_mesh = None


def box_mesh():
    """small cut-cell mesh of a box with a step (walls only)"""
    global _mesh
    if _mesh is None:
        h = 0.01
        G = np.zeros((40, 40, 50), bool)
        G[:, :, :25] = True
        G[5:35, 5:30, 25:50] = True
        with tempfile.TemporaryDirectory() as d:
            voxel_stl(os.path.join(d, "s.stl"), G, h / 10)
            _write_json(os.path.join(d, "m.json"), {"domain": "s.stl", "baseCellSize": h,
                                                    "surfaceRefinement": {"default": 1}})
            _mesh = generate(load_config(os.path.join(d, "m.json")), log=lambda s: None)
    return _mesh


@unittest.skipUnless(L._jacobi_band is not None and L._cells_bad_rs is not None,
                     "Rust extension not built (python tools/build_rust.py)")
class RustLayerKernels(unittest.TestCase):
    def test_jacobi_band(self):
        rng = np.random.default_rng(1)
        n = 3000
        A = sparse.random(n, n, density=0.003, random_state=2, format="csr")
        A = (A + A.T).tocsr()
        deg = np.asarray(A.sum(1)).ravel() + rng.random(n) + 0.5
        c = rng.normal(size=(n, 3))
        rows = np.sort(rng.choice(n, 200, replace=False)).astype(np.int64)
        M = np.repeat(np.eye(3)[None], len(rows), 0)
        nrm = rng.normal(size=(len(rows), 3))
        nrm /= np.linalg.norm(nrm, axis=1)[:, None]
        M -= np.einsum("ni,nj->nij", nrm, nrm)
        x = np.zeros((n, 3))
        for _ in range(40):
            x = (A @ x + c) / deg[:, None]
            x[rows] = np.einsum("nij,nj->ni", M, x[rows])
        y = L._jacobi_band(A.indptr.astype(np.int64), A.indices.astype(np.int64), A.data, c, deg,
                           rows, M, 40)
        np.testing.assert_allclose(y, x, rtol=1e-13, atol=1e-13 * np.abs(x).max())

    def _distorted(self):
        R = box_mesh()
        rng = np.random.default_rng(3)
        P = np.array(R["points"], float)
        P += rng.normal(0, 0.0015, P.shape)             # some cells become invalid (h = 0.01)
        nn, vv = (np.asarray(a, np.int64) for a in R["faces"])
        own, nei = np.asarray(R["owner"], np.int64), np.asarray(R["neighbour"], np.int64)
        nC = int(own.max()) + 1
        return P, nn, vv, own, nei, nC, rng

    def _both(self, *args, **kw):
        rs = L._cells_bad(*args, **kw)
        keep = L._cells_bad_rs
        try:
            L._cells_bad_rs = None
            py = L._cells_bad(*args, **kw)
        finally:
            L._cells_bad_rs = keep
        return rs, py

    def test_cells_bad(self):
        P, nn, vv, own, nei, nC, rng = self._distorted()
        nI, nF = len(nei), len(nn)
        max_no = rng.uniform(20, 70, nI)
        fs = np.sort(rng.choice(nF, nF // 2, replace=False))
        max_sk = (fs, rng.uniform(0.5, 4, len(fs)))
        fixed = (rng.random(nC) < 0.1, rng.normal(size=(nC, 3)) * 1e-3 + P.mean(0))
        for kw in ({}, dict(max_no=45.0), dict(max_no=max_no, max_sk=max_sk),
                   dict(max_no=max_no, max_sk=max_sk, fixed=fixed)):
            rs, py = self._both(P, nn, vv, own, nei, nC, **kw)
            self.assertGreater(int(py.sum()), 0)            # the test sees invalid cells
            self.assertEqual(int((rs != py).sum()), 0, kw)

    def test_cells_bad_per_face_neighbours(self):
        """neighbour per face (-1 for boundary faces), faces in any order"""
        P, nn, vv, own, nei, nC, rng = self._distorted()
        nI, nF = len(nei), len(nn)
        ref = L._cells_bad(P, nn, vv, own, nei, nC, 50.0, (np.arange(nF), np.full(nF, 3.0)))
        o = rng.permutation(nF)
        ptr = np.concatenate([[0], np.cumsum(nn)])
        vv2 = np.concatenate([vv[ptr[f]:ptr[f + 1]] for f in o])
        nei2 = np.concatenate([nei, np.full(nF - nI, -1)])[o]
        rs, py = self._both(P, nn[o], vv2, own[o], nei2, nC, np.full(nF, 50.0),
                            (np.arange(nF), np.full(nF, 3.0)))
        self.assertEqual(int((rs != ref).sum()), 0)
        self.assertEqual(int((py != ref).sum()), 0)

    def test_add_layers_same_without_kernels(self):
        R = box_mesh()
        a = L.add_layers(R, {}, log=lambda s: None)
        keep = L._jacobi_band, L._cells_bad_rs
        try:
            L._jacobi_band = L._cells_bad_rs = None
            b = L.add_layers(R, {}, log=lambda s: None)
        finally:
            L._jacobi_band, L._cells_bad_rs = keep
        np.testing.assert_allclose(a["points"], b["points"], rtol=0, atol=1e-12)
        for k in ("owner", "neighbour"):
            np.testing.assert_array_equal(a[k], b[k])
        np.testing.assert_array_equal(a["faces"][1], b["faces"][1])
        np.testing.assert_array_equal(a["layer_thickness"]["T"], b["layer_thickness"]["T"])


if __name__ == "__main__":
    unittest.main()
