"""Fluid gaps thinner than a cell must stay open, and solid walls thinner than a cell must
not vanish (fillThinGaps / fillThinWalls, the snap guard in mesher.py, diagnose.py).

Two chambers joined by a slot thinner than the cell size:
- between: the slot lies between two vertex planes, so no cell corner is in the fluid
  and the cells along the slot would be dropped (fixed by splitting them)
- centred: a vertex plane runs through the middle of the slot, close enough to both
  walls to be snapped, which would drop the cells too (fixed by the snap guard)

python -m unittest discover -s tests
"""
import json
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from cutmesh import quality                                   # noqa: E402
from cutmesh.geometry import write_stl_ascii                  # noqa: E402
from cutmesh.mesher import load_config, generate              # noqa: E402
from cutmesh.geometry import Surface, read_stl                # noqa: E402
from cutmesh.diagnose import through_wall                     # noqa: E402


def voxel_stl(path, filled, v):
    """outward boundary of the filled voxels as triangles"""
    F = np.pad(filled, 1)
    tris = []
    for ax in range(3):
        a, b = [d for d in range(3) if d != ax]
        for sgn in (-1, 1):
            face = F & ~np.roll(F, -sgn, axis=ax)
            for i, j, k in np.argwhere(face[1:-1, 1:-1, 1:-1]):
                p = np.array([i, j, k], float)
                if sgn > 0:
                    p[ax] += 1
                q = [p.copy() for _ in range(4)]
                q[1][a] += 1; q[2][a] += 1; q[2][b] += 1; q[3][b] += 1
                if (sgn > 0) != ((a, b) in ((0, 1), (1, 2), (2, 0))):
                    q = q[::-1]
                Q = np.array(q) * v
                tris += [[Q[0], Q[1], Q[2]], [Q[0], Q[2], Q[3]]]
    write_stl_ascii(path, [("walls", np.array(tris))])


def components(own, nei, nC):
    """number of face-connected groups of cells"""
    parent = np.arange(nC)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for a, b in zip(own[:len(nei)].tolist(), nei.tolist()):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    return len({find(i) for i in range(nC)})


def _write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f)


class ThinGaps(unittest.TestCase):
    h = 0.01

    def mesh(self, y_centre, thickness, fill, extra=None):
        h, v = self.h, self.h / 10
        G = np.zeros((60, 60, 90), bool)
        G[:, :, 0:30] = True                    # chamber z in [0, 3h]
        G[:, :, 60:90] = True                   # chamber z in [6h, 9h]
        y0, y1 = int(round((y_centre - thickness / 2) / v)), int(round((y_centre + thickness / 2) / v))
        G[10:50, y0:y1, 30:60] = True           # slot
        with tempfile.TemporaryDirectory() as d:
            voxel_stl(os.path.join(d, "s.stl"), G, v)
            _write_json(os.path.join(d, "m.json"), {"domain": "s.stl", "baseCellSize": h, "surfaceRefinement": {"default": 0},
                       "featureLevel": 0, "fillThinGaps": fill, **(extra or {})})
            R = generate(load_config(os.path.join(d, "m.json")), log=lambda s: None)
        own, nei = R["owner"], R["neighbour"]
        nC = int(max(own.max(), nei.max())) + 1
        q = quality.check(R["points"], R["faces"], own, nei, log=lambda s: None)
        self.assertEqual(q["nNegVol"], 0)
        self.assertEqual(q["nBadPyramids"], 0)
        self.assertEqual(q["openCellEdges"], 0)
        return components(own, nei, nC), q["totalVol"] / (G.sum() * v ** 3)

    def test_slot_between_vertex_planes(self):
        ncomp, vr = self.mesh(2.5 * self.h, 0.4 * self.h, fill=2)
        self.assertEqual(ncomp, 1)
        self.assertAlmostEqual(vr, 1.0, delta=0.01)
        ncomp, _ = self.mesh(2.5 * self.h, 0.4 * self.h, fill=0)
        self.assertEqual(ncomp, 2)              # the slot is lost without the splitting

    def test_auto_gap_refinement(self):
        # a slot a quarter of a cell wide: closed without, three cells across with it
        auto = {"autoGapRefinement": {"cellsAcross": 3, "maxLevel": 4}}
        ncomp, vr = self.mesh(2.5 * self.h, 0.25 * self.h, fill=2, extra=auto)
        self.assertEqual(ncomp, 1)
        self.assertAlmostEqual(vr, 1.0, delta=0.002)
        ncomp, _ = self.mesh(2.5 * self.h, 0.25 * self.h, fill=2)
        self.assertEqual(ncomp, 2)

    def test_isolated_region_removed(self):
        # a small cube of fluid apart from the chamber: removed below minRegionVolume
        h, v = self.h, self.h / 10
        G = np.zeros((60, 60, 90), bool)
        G[:, :, 0:30] = True
        G[24:30, 24:30, 50:56] = True                         # 0.6h cube, 0.15 % of the volume
        for frac, ncomp in ((1e-2, 1), (0, 2)):
            with tempfile.TemporaryDirectory() as d:
                voxel_stl(os.path.join(d, "s.stl"), G, v)
                _write_json(os.path.join(d, "m.json"), {"domain": "s.stl", "baseCellSize": h, "featureLevel": 0,
                           "surfaceRefinement": {"default": 0}, "minRegionVolume": frac})
                R = generate(load_config(os.path.join(d, "m.json")), log=lambda s: None)
            own, nei = R["owner"], R["neighbour"]
            nC = int(max(own.max(), nei.max())) + 1
            self.assertEqual(components(own, nei, nC), ncomp)
            q = quality.check(R["points"], R["faces"], own, nei, log=lambda s: None)
            self.assertEqual(q["openCellEdges"], 0)
            self.assertEqual(q["nNegVol"], 0)

    def test_slot_centred_on_vertex_plane(self):
        ncomp, vr = self.mesh(3.0 * self.h, 0.5 * self.h, fill=0)
        self.assertEqual(ncomp, 1)
        self.assertAlmostEqual(vr, 1.0, delta=0.01)


class ThinWalls(unittest.TestCase):
    """U-shaped channel around a solid plate thinner than a cell: without care the plate
    vanishes and the fluid on both sides joins through it (a bypass)."""
    h = 0.01

    def mesh(self, thickness, y_centre, fill):
        h, v = self.h, self.h / 20
        G = np.ones((120, 120, 180), bool)                   # 6h x 6h x 9h of fluid
        y0, y1 = int(round((y_centre - thickness / 2) / v)), int(round((y_centre + thickness / 2) / v))
        G[:, y0:y1, 0:120] = False                           # plate, open above z = 6h
        with tempfile.TemporaryDirectory() as d:
            voxel_stl(os.path.join(d, "w.stl"), G, v)
            _write_json(os.path.join(d, "m.json"), {"domain": "w.stl", "baseCellSize": h, "surfaceRefinement": {"default": 0},
                       "featureLevel": 0, "fillThinWalls": fill, "checkLeaks": False})
            R = generate(load_config(os.path.join(d, "m.json")), log=lambda s: None)
            dom = Surface(read_stl(os.path.join(d, "w.stl")))
        q = quality.check(R["points"], R["faces"], R["owner"], R["neighbour"], log=lambda s: None)
        self.assertEqual(q["nNegVol"], 0)
        self.assertEqual(q["nBadPyramids"], 0)
        n, _ = through_wall(dom, R["points"], R["faces"], R["owner"], R["neighbour"])
        return n, q["totalVol"] / (G.sum() * v ** 3)

    def test_plate_off_the_vertex_planes(self):
        n, vr = self.mesh(0.3 * self.h, 2.2 * self.h, fill=3)
        self.assertEqual(n, 0)
        self.assertAlmostEqual(vr, 1.0, delta=0.002)
        n, vr = self.mesh(0.3 * self.h, 2.2 * self.h, fill=0)
        self.assertGreater(n, 0)                              # the leak check sees the bypass
        self.assertGreater(vr, 1.01)

    def test_plate_thinner_than_the_samples(self):
        n, vr = self.mesh(0.15 * self.h, 2.4 * self.h, fill=3)
        self.assertEqual(n, 0)
        self.assertAlmostEqual(vr, 1.0, delta=0.002)


if __name__ == "__main__":
    unittest.main()
