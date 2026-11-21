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
from cutmesh.cylbase import auto_layout, coaxial_radii, AUTO_DR, _dyadic  # noqa: E402


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


MANUAL = {"core": {"halfWidth": 0.03, "n": 4},
          # 16 -> 40 around: not a multiple
          "blocks": [{"rOuter": 0.06, "nr": 2}, {"rOuter": 0.11, "nr": 3, "nTheta": 40}]}


class AutoLayout(unittest.TestCase):
    @staticmethod
    def nodes(core, blocks):
        """radial nodes from the O-grid circle out, block by block (ri, nodes, nTheta)"""
        out, ri = [], blocks[0]["rOuter"]
        for b in blocks[1:]:
            rs = np.array(b.get("radii", np.linspace(ri, b["rOuter"], b["nr"] + 1)))
            out.append((ri, rs, b["nTheta"]))
            ri = b["rOuter"]
        return out

    def check_sizes(self, h, r_out, core, blocks):
        self.assertEqual(blocks[0]["rOuter"], 2 * core["halfWidth"])
        self.assertAlmostEqual(blocks[-1]["rOuter"], r_out)
        for ri, rs, nth in self.nodes(core, blocks):
            self.assertAlmostEqual(rs[0], ri)
            self.assertEqual(nth % 4, 0)
            dr = np.diff(rs)
            # no edge longer than about 1.2 h (refinement would add a level), none much shorter
            self.assertTrue(np.all(dr <= AUTO_DR * h * (1 + 1e-9)) and np.all(dr > 0.3 * h))
            mid = math.pi * (rs[:-1] + rs[1:]) / nth            # around, per layer
            self.assertTrue(np.all(mid <= 1.3 * h) and np.all(mid > 0.65 * h), (mid / h, h))

    def test_cells_point_around_and_are_about_h(self):
        for h, r_out in [(0.025, 0.11), (0.0125, 0.1625), (0.001, 0.2)]:
            core, blocks = auto_layout(h, r_out)
            self.assertEqual(core, {"halfWidth": 0.8 * h, "n": 2})
            self.check_sizes(h, r_out, core, blocks)

    def test_cylinders_a_third_into_a_layer(self):
        # stirred tank: shaft, hub, blade tips, rotating zone, tank wall
        h, avoid = 0.0125, [0.008, 0.015, 0.0500899, 0.07, 0.15]
        r_out = 0.15 + 2 * h / 3
        core, blocks = auto_layout(h, r_out, avoid)
        self.check_sizes(h, r_out, core, blocks)
        a = core["halfWidth"]
        # the shaft inside the core: a third of a cell from the sides at every level
        self.assertGreaterEqual(_dyadic((a - 0.008) / a), 0.25)
        nodes = np.unique(np.concatenate([rs for _, rs, _ in self.nodes(core, blocks)]))
        for x in (0.0500899, 0.07, 0.15):
            i = int(np.searchsorted(nodes, x))
            t = (x - nodes[i - 1]) / (nodes[i] - nodes[i - 1])
            self.assertAlmostEqual(min(t, 1 - t), 1 / 3, places=6)
        core, _ = auto_layout(h, r_out, [0.07])             # nothing near the core: 0.8 h
        self.assertAlmostEqual(core["halfWidth"], 0.8 * h)

    def test_coaxial_radii(self):
        th = np.linspace(0, 2 * math.pi, 49)[:-1]
        ring = lambda r, t: (r * np.cos(t), r * np.sin(t))
        x1, y1 = ring(0.008, th)                           # a full circle
        x2, y2 = ring(0.05, np.array([0, 0.1, 1.5, 1.6, 3.1, 3.2, 4.7, 4.8]))   # 4 blade tips
        x3, y3 = ring(0.03, th[:24])                       # half a circle: not a cylinder
        x4, y4 = np.array([0.1, 0.0]), np.array([0.0, 0.1])
        r = coaxial_radii(np.r_[x1, x2, x3, x4], np.r_[y1, y2, y3, y4], 1e-5)
        self.assertEqual(sorted(round(v, 6) for v in r), [0.008, 0.05])

    def test_small_radius(self):
        core, blocks = auto_layout(0.01, 0.015)
        self.assertEqual(core["halfWidth"], 0.0075)
        self.assertEqual(blocks, [{"rOuter": 0.015, "nr": 1}])
        # just outside the O-grid circle: no thin ring, the O-grid ends at r_outer
        for r_out, avoid in [(1.60001, [0.9333433333333334]), (1.65, []), (2.05, [])]:
            core, blocks = auto_layout(1.0, r_out, avoid)
            self.assertEqual(len(blocks), 1)
            self.assertEqual(blocks[0]["rOuter"], r_out)
            self.assertGreater(r_out, core["halfWidth"] * math.sqrt(2) * 1.01)
            self.assertLessEqual((r_out - core["halfWidth"]) / blocks[0]["nr"], AUTO_DR)
        _, blocks = auto_layout(1.0, 2.2)                  # room for one more ring
        self.assertEqual(len(blocks), 2)


class CylindricalBase(unittest.TestCase):
    def mesh(self, refine, layout=MANUAL):
        with tempfile.TemporaryDirectory() as d:
            stl = os.path.join(d, "cyl.stl")
            vol = cylinder_stl(stl, 0.1, 0.1)
            cfg_path = os.path.join(d, "meshDict.json")
            cfg = {"domain": "cyl.stl", "baseCellSize": 0.025,
                   "surfaceRefinement": {"default": 1 if refine else 0}, "featureLevel": 0,
                   "baseMesh": dict({"type": "cylindrical", "nz": 6}, **layout)}
            _write_json(cfg_path, cfg)
            lines = []
            R = generate(load_config(cfg_path), log=lines.append)
        self.log = lines
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

    def test_radii_checked(self):
        for radii, msg in [([0.01, 0.03, 0.05, 0.07], "from 0.06"),          # not the block's ends
                           ([0.06, 0.09, 0.08, 0.11], "must increase"),
                           ([0.06, 0.08, 0.11], "nr + 1 = 4")]:
            layout = {"core": MANUAL["core"],
                      "blocks": [MANUAL["blocks"][0], {"rOuter": 0.11, "nr": 3, "nTheta": 40, "radii": radii}]}
            with self.assertRaises(ValueError) as cm:
                self.mesh(False, layout=layout)
            self.assertIn(msg, str(cm.exception))

    def test_automatic_layout(self):
        # no core / blocks: 2 x 2 core and polar blocks out to the STL radius + 2/3 cell.
        # The wall is cut by cells at other angles than the 48-gon's corners: the chords
        # between snapped vertices lose about 0.2 % of the volume
        self.check(*self.mesh(True, layout={}), vol_tol=3e-3)
        auto = [s for s in self.log if "automatic layout" in s]
        self.assertEqual(len(auto), 1)
        self.assertIn('"n": 2', auto[0])


if __name__ == "__main__":
    unittest.main()
