"""Cut cells keeping little fluid next to finer cells (mesher thinCutCells): their centre
sits on the wall, and the faces to the finer cells are strongly non-orthogonal.  They
are split so the level transition moves off the wall.

python -m unittest discover -s tests
"""
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cutmesh import quality                                   # noqa: E402
from cutmesh.mesher import load_config, generate, _fluid_fraction   # noqa: E402
from test_thin_gaps import _write_json                        # noqa: E402
import test_concave_edges                                     # noqa: E402


class FluidFraction(unittest.TestCase):
    class Plane:
        """fluid: x < 0.3"""
        def sdf(self, p):
            return np.atleast_2d(p)[:, 0] - 0.3

    def test_plane(self):
        c = np.array([[0.0, 0, 0], [0.0, 0, 0], [5.0, 0, 0]])
        h = np.array([1.0, 0.2, 1.0])
        fr, cut = _fluid_fraction(self.Plane(), c, h)
        self.assertAlmostEqual(fr[0], 0.8, delta=1 / 6)
        self.assertEqual(fr[1], 1.0)                 # all fluid, not cut
        self.assertEqual(fr[2], 0.0)
        self.assertEqual(cut.tolist(), [True, False, False])


class ThinCutCells(unittest.TestCase):
    """Rotor-stator gap of benchmark2 (0.7 mm, rotating-zone interface inside it): the
    coarse cells against the rotor wall are cut to thin slabs next to the finer gap cells."""

    def mesh(self, extra):
        g = test_concave_edges.SlotMouths()
        with tempfile.TemporaryDirectory() as d:
            g.stl(os.path.join(d, "s.stl"))
            _write_json(os.path.join(d, "m.json"), {
                "domain": "s.stl", "baseCellSize": 0.005, "surfaceRefinement": {"default": 1, "rotor": 2},
                "featureLevel": 2, "gapRefinement": {"thickness": 0.001, "level": 3},
                "rotatingZones": [{"name": "mrf", "cylinder": {"point1": [0, 0, -0.001], "point2": [0, 0, 0.0035],
                                                              "radius": 0.0303625},
                                   "surfaceLevel": 3, "volumeLevel": 2, "omega": 0,
                                   "wallProximity": {"distance": 0.0005, "level": 4, "bufferLayers": 1}}],
                **extra})
            R = generate(load_config(os.path.join(d, "m.json")), log=lambda s: None)
        q = quality.check(R["points"], R["faces"], R["owner"], R["neighbour"], log=lambda s: None)
        self.assertEqual(q["nNegVol"], 0)
        self.assertEqual(q["nBadPyramids"], 0)
        self.assertEqual(q["openCellEdges"], 0)
        return q

    def test_fewer_non_orthogonal_faces(self):
        q0 = self.mesh({"thinCutCells": 0})
        q = self.mesh({})
        n0, n = (q0["_nonorth"] > 60).sum(), (q["_nonorth"] > 60).sum()
        self.assertGreater(n0, 100)
        self.assertLess(n, 0.3 * n0)
        self.assertLess(q["nCells"], 1.1 * q0["nCells"])


if __name__ == "__main__":
    unittest.main()
