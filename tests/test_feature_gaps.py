"""Convex feature edges not aligned with the octree (featureGapSnap, mesher.fill_feature_gaps):
a cube tilted against the grid.  Where the vertices on an edge are more than half a cell
apart the cut chamfers the edge between them; snapping one more vertex into each such gap
keeps the edge.  Off by default.

python -m unittest discover -s tests
"""
import json
import os
import sys
import tempfile
import unittest

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from cutmesh import quality                                            # noqa: E402
from cutmesh.geometry import Surface, read_stl, sample_segments, write_stl_ascii  # noqa: E402
from cutmesh.mesher import load_config, generate                       # noqa: E402


def tilted_cube(path):
    v = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], float) - 0.5
    f = [[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
         [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]]

    def rot(axis, deg):
        a = np.asarray(axis, float) / np.linalg.norm(axis)
        t = np.radians(deg)
        K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
        return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K
    v = v @ rot([0, 0, 1], 30).T @ rot([1, 1, 0], 20).T
    write_stl_ascii(path, [("walls", v[np.array(f)])])


class FeatureGaps(unittest.TestCase):
    h0 = 0.12
    _res = {}

    @classmethod
    def mesh(cls, gap):
        if gap not in cls._res:
            with tempfile.TemporaryDirectory() as d:
                tilted_cube(os.path.join(d, "c.stl"))
                cfg = {"domain": "c.stl", "baseCellSize": cls.h0, "surfaceRefinement": {"default": 2}}
                if gap is not None:
                    cfg["featureGapSnap"] = gap
                with open(os.path.join(d, "m.json"), "w") as f:
                    json.dump(cfg, f)
                logs = []
                c = load_config(os.path.join(d, "m.json"))
                R = generate(c, log=logs.append)
                dom = Surface(read_stl(os.path.join(d, "c.stl")))
            seg = dom.feature_edges(30.0)
            cls._res[gap] = (R, logs, seg)
        return cls._res[gap]

    def edge_off(self, gap):
        """fraction of points along the cube edges farther than 0.2 hmin from a wall edge"""
        R, _, seg = self.mesh(gap)
        h = self.h0 / 4
        P = R["points"]; nn, vv = R["faces"]; nI = len(R["neighbour"])
        ptr = np.concatenate([[0], np.cumsum(nn)])
        e = []
        for f in range(nI, len(nn)):
            p = vv[ptr[f]:ptr[f + 1]]
            e.append(np.stack([p, np.roll(p, -1)], 1))
        e = np.unique(np.sort(np.concatenate(e), 1), axis=0)
        be = sample_segments(P[e], h / 16)
        d, _ = cKDTree(be).query(sample_segments(seg, h / 4))
        return float((d > 0.2 * h).mean())

    def test_off_by_default(self):
        _, logs, _ = self.mesh(None)
        self.assertFalse(any("feature gaps" in s for s in logs))

    def test_sharper_edges_when_on(self):
        off, on = self.edge_off(None), self.edge_off(0.75)
        self.assertLess(on, 0.6 * off, f"edge points off the mesh edges: {off:.3f} -> {on:.3f}")

    def test_mesh_stays_valid_and_logged(self):
        R, logs, _ = self.mesh(0.75)
        line = [s for s in logs if s.startswith("feature gaps:")]
        self.assertEqual(len(line), 1)
        self.assertIn("got a vertex snapped onto the edge", line[0])
        q = quality.check(R["points"], R["faces"], R["owner"], R["neighbour"], log=None)
        self.assertEqual(q["nNegVol"], 0)
        self.assertEqual(q["nBadPyramids"], 0)
        self.assertEqual(q["openCellEdges"], 0)
        self.assertTrue(q["upperTriangular"])


if __name__ == "__main__":
    unittest.main()
