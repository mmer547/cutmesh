"""Exact cut at convex feature edges (exactFeatureCut, cutmesh/featclip.py) on a cube
turned so that no edge runs along the octree axes.

python -m unittest discover -s tests
"""
import json
import os
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))

from scipy.spatial import cKDTree                                 # noqa: E402
from cutmesh.geometry import Surface, write_stl_ascii, sample_segments   # noqa: E402
from cutmesh.mesher import run                                    # noqa: E402
from edge_check import edge_samples                               # noqa: E402


def turned_cube(path):
    V = np.array([[x, y, z] for z in (-0.5, 0.5) for y in (-0.5, 0.5) for x in (-0.5, 0.5)])
    F = [(0, 2, 1), (1, 2, 3), (4, 5, 6), (5, 7, 6), (0, 1, 4), (1, 5, 4),
         (2, 6, 3), (3, 6, 7), (0, 4, 2), (2, 4, 6), (1, 3, 5), (3, 7, 5)]
    a, b, c = np.radians([23.0, 31.0, 17.0])
    Rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    Ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    Rz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
    T = V[np.array(F)] @ (Rz @ Ry @ Rx).T
    # outward normals
    n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    flip = (n * T.mean(1)).sum(1) < 0
    T[flip] = T[flip][:, ::-1]
    write_stl_ascii(path, [("walls", T)])
    return T


def edge_deviation(R, tris, hmin):
    """distance from points along the cube's edges to the nearest mesh wall edge / hmin"""
    dom = Surface([("walls", tris)])
    seg = dom.feature_edges(30.0)
    fs = sample_segments(seg, hmin / 8)
    nn, vv = R["faces"]
    nI = len(R["neighbour"])
    be = edge_samples(R["points"], nn, vv, range(nI, len(nn)), hmin / 16)
    d, _ = cKDTree(be).query(fs)
    return d / hmin


class ExactFeatureCut(unittest.TestCase):
    def mesh(self, d, exact):
        p = os.path.join(d, f"m_{exact}.json")
        with open(p, "w") as f:
            json.dump({"domain": "cube.stl", "baseCellSize": 0.15, "surfaceRefinement": {"default": 1},
                       "exactFeatureCut": exact, "case": f"c_{exact}"}, f)
        logs = []
        R = run(p, log=logs.append, check=True)
        return R, logs

    def test_turned_cube_edges(self):
        with tempfile.TemporaryDirectory() as d:
            T = turned_cube(os.path.join(d, "cube.stl"))
            R1, logs = self.mesh(d, True)
            R0, _ = self.mesh(d, False)
        hmin = 0.15 / 2
        d1, d0 = edge_deviation(R1, T, hmin), edge_deviation(R0, T, hmin)
        self.assertTrue(any(s.startswith("exact feature cut:") for s in logs))
        self.assertGreater(d0.max(), 0.3)                # notches along the edges without it
        self.assertLess(d1.max(), 0.1)                   # the edges are captured with it
        self.assertLess(np.mean(d1 > 0.05), 0.01)
        for R in (R0, R1):
            q = R["quality"]
            self.assertEqual(q["nNegVol"], 0)
            self.assertEqual(q["nBadPyramids"], 0)
            self.assertEqual(q["openCellEdges"], 0)
        # at this coarse resolution (12 cells across) one clipped face reaches skewness 4.4;
        # at 24 cells across (benchmark cube_rotated) the faces above 4 go from 3 to 0
        self.assertLess(R1["quality"]["maxSkewness"], 5.0)
        self.assertAlmostEqual(R1["quality"]["totalVol"], 1.0, delta=1e-4)


if __name__ == "__main__":
    unittest.main()
