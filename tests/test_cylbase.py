"""Cylindrical base mesh (cutmesh/cylbase.py) on a closed cylinder.

python -m unittest discover -s tests
"""
import json
import math
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from cutmesh import quality                                   # noqa: E402
from cutmesh.geometry import write_stl_ascii                  # noqa: E402
from cutmesh.mesher import load_config, generate              # noqa: E402


def cylinder_stl(path, r, h, n=48):
    """closed cylinder around the z axis (outward normals), solids wall/bottom/top"""
    th = np.linspace(0, 2 * math.pi, n + 1)[:-1]
    ring = np.c_[r * np.cos(th), r * np.sin(th)]
    wall, bot, top = [], [], []
    for i in range(n):
        a, b = ring[i], ring[(i + 1) % n]
        A0, B0, A1, B1 = [*a, 0], [*b, 0], [*a, h], [*b, h]
        wall += [[A0, B0, B1], [A0, B1, A1]]
        bot.append([[0, 0, 0], B0, A0])
        top.append([[0, 0, h], A1, B1])
    write_stl_ascii(path, [("wall", np.array(wall, float)), ("bottom", np.array(bot, float)),
                           ("top", np.array(top, float))])
    return 0.5 * n * r * r * math.sin(2 * math.pi / n) * h       # exact volume of the polygon prism


def _write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f)


class CylindricalBase(unittest.TestCase):
    def mesh(self, refine):
        with tempfile.TemporaryDirectory() as d:
            stl = os.path.join(d, "cyl.stl")
            vol = cylinder_stl(stl, 0.1, 0.1)
            cfg_path = os.path.join(d, "meshDict.json")
            cfg = {"domain": "cyl.stl", "baseCellSize": 0.025,
                   "surfaceRefinement": {"default": 1 if refine else 0}, "featureLevel": 0,
                   "baseMesh": {"type": "cylindrical", "nz": 6,
                                "core": {"halfWidth": 0.03, "n": 4},
                                # 16 -> 40 around: not a multiple
                                "blocks": [{"rOuter": 0.06, "nr": 2}, {"rOuter": 0.11, "nr": 3, "nTheta": 40}]}}
            _write_json(cfg_path, cfg)
            R = generate(load_config(cfg_path), log=lambda s: None)
        q = quality.check(R["points"], R["faces"], R["owner"], R["neighbour"], log=lambda s: None)
        return q, vol

    def check(self, q, vol, vol_tol):
        self.assertTrue(q["upperTriangular"])
        self.assertEqual(q["openCellEdges"], 0)
        self.assertLess(q["maxClosedness"], 1e-12)
        self.assertEqual(q["nNegVol"], 0)
        self.assertEqual(q["nBadPyramids"], 0)
        self.assertLess(q["maxNonOrth"], 70)
        self.assertAlmostEqual(q["totalVol"] / vol, 1.0, delta=vol_tol)

    def test_blocks_without_refinement(self):
        # 40 cells around cut the 48-gon wall with chords: about -0.4 % of volume
        self.check(*self.mesh(False), vol_tol=1e-2)

    def test_blocks_with_refinement(self):
        self.check(*self.mesh(True), vol_tol=1e-3)


if __name__ == "__main__":
    unittest.main()
