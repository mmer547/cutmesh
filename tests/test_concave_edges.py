"""Concave feature edges (the solid juts into the fluid) between the vertex lines: the
vertex in the solid wedge is snapped onto the edge from up to concaveSnapMax x h, so the
edge is kept instead of chamfered (mesher.snap_features).

python -m unittest discover -s tests
"""
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from cutmesh import quality                                   # noqa: E402
from cutmesh.mesher import load_config, generate              # noqa: E402
from test_thin_gaps import voxel_stl, _write_json             # noqa: E402


class ConcaveEdge(unittest.TestCase):
    h = 0.01

    def mesh(self, corner, extra):
        """L-shaped channel: solid block [0, corner]^2 x [0, 3h]"""
        h, v = self.h, self.h / 20
        G = np.ones((120, 120, 60), bool)
        k = int(round(corner / v))
        G[:k, :k, :] = False
        with tempfile.TemporaryDirectory() as d:
            voxel_stl(os.path.join(d, "s.stl"), G, v)
            _write_json(os.path.join(d, "m.json"), {"domain": "s.stl", "baseCellSize": h, "featureLevel": 0,
                                                    "surfaceRefinement": {"default": 0}, **extra})
            R = generate(load_config(os.path.join(d, "m.json")), log=lambda s: None)
        P = R["points"]
        q = quality.check(P, R["faces"], R["owner"], R["neighbour"], log=lambda s: None)
        self.assertEqual(q["nNegVol"], 0)
        self.assertEqual(q["nBadPyramids"], 0)
        self.assertEqual(q["openCellEdges"], 0)
        c = k * v
        on_edge = int((np.hypot(P[:, 0] - c, P[:, 1] - c) < 1e-6 * h).sum())
        return on_edge, q["totalVol"] / (G.sum() * v ** 3)

    def test_corner_kept(self):
        # nearest (solid) vertex 0.57 h from the edge: beyond the usual 0.5 h snap
        n, vr = self.mesh(2.4 * self.h, {})
        self.assertGreaterEqual(n, 2)
        n0, vr0 = self.mesh(2.4 * self.h, {"concaveSnapMax": 0.5})
        self.assertEqual(n0, 0)
        self.assertLess(abs(vr - 1), abs(vr0 - 1))

    def test_flattened_face_rejected(self):
        # the solid vertex would land on the diagonal of the next cell's faces: not snapped
        n, vr = self.mesh(2.5 * self.h, {})
        self.assertEqual(n, 0)
        self.assertAlmostEqual(vr, 1.0, delta=0.005)


if __name__ == "__main__":
    unittest.main()
