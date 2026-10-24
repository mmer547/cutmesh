"""Concave feature edges (the solid juts into the fluid) between the vertex lines: the
vertex in the solid wedge is snapped onto the edge from up to concaveSnapMax x h, so the
edge is kept instead of chamfered (mesher.snap_features).  At a level transition, the
nearest wall vertex slides along its wall onto the edge instead.

python -m unittest discover -s tests
"""
import os
import sys
import tempfile
import unittest

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cutmesh import quality                                   # noqa: E402
from cutmesh.geometry import Surface, FeatureSet, write_stl_ascii, sample_segments   # noqa: E402
from cutmesh.mesher import load_config, generate, make_pinch_hook   # noqa: E402
from cutmesh.cutcell import CutContext, split_components     # noqa: E402
from test_thin_gaps import voxel_stl, _write_json             # noqa: E402


class ConcaveEdge(unittest.TestCase):
    h = 0.01

    def generate(self, G, extra):
        h, v = self.h, self.h / 20
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
        return P, q["totalVol"] / (G.sum() * v ** 3)

    def mesh(self, corner, extra):
        """L-shaped channel: solid block [0, corner]^2 x [0, 3h]"""
        h, v = self.h, self.h / 20
        G = np.ones((120, 120, 60), bool)
        k = int(round(corner / v))
        G[:k, :k, :] = False
        P, vr = self.generate(G, extra)
        c = k * v
        on_edge = int((np.hypot(P[:, 0] - c, P[:, 1] - c) < 1e-6 * h).sum())
        return on_edge, vr

    def test_thin_post_kept(self):
        # a 0.8 h square post around a vertex line: its solid vertices lie 0.57 h from the
        # post's (re-entrant) edges, but snapping them there would remove the post
        G = np.ones((120, 120, 60), bool)
        G[52:68, 52:68, :] = False
        _, vr = self.generate(G, {})
        _, vr0 = self.generate(G, {"concaveSnapMax": 0.5})
        self.assertAlmostEqual(vr, vr0, delta=1e-9)
        self.assertLess(vr, 1.015)               # 1.018: the post gone entirely

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


def _box(lo, hi):
    """Outward-wound triangles of an axis-aligned box"""
    x0, y0, z0 = lo; x1, y1, z1 = hi
    P = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], float)
    q = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (2, 3, 7, 6), (1, 2, 6, 5), (0, 4, 7, 3)]
    return np.array([[P[a], P[b], P[c]] for a, b, c, d in q for a, b, c in ((a, b, c), (a, c, d))])


class ConcaveFeatureEdges(unittest.TestCase):
    def l_shape(self):
        """L-shaped prism: unit square minus [0, 0.5]^2, extruded over z in [0, 1]"""
        xy = [(0.5, 0), (1, 0), (1, 1), (0, 1), (0, 0.5), (0.5, 0.5)]
        n = len(xy)
        T = []
        for i in range(n):
            a, b = xy[i], xy[(i + 1) % n]
            T += [[(*a, 0), (*b, 0), (*b, 1)], [(*a, 0), (*b, 1), (*a, 1)]]
        for z, rev in ((0, True), (1, False)):   # caps: two convex quads
            for quad in ([0, 1, 2, 5], [5, 2, 3, 4]):
                p = [(*xy[k], z) for k in quad]
                tri = [[p[0], p[1], p[2]], [p[0], p[2], p[3]]]
                T += [t[::-1] for t in tri] if rev else tri
        return np.array(T, float)

    def test_winding_independent(self):
        T = self.l_shape()
        for tris in (T, T[:, ::-1]):
            seg = Surface([("s", tris)]).feature_edges(30.0, concave_only=True)
            self.assertEqual(len(seg), 1)        # only the re-entrant vertical edge
            self.assertTrue(np.allclose(seg[0][:, :2], 0.5))
            # all 18 sharp edges (6 vertical + 12 cap edges) either way
            self.assertEqual(len(Surface([("s", tris)]).feature_edges(30.0)), 18)

    def test_nested_shell(self):
        # fluid between two boxes: the inner box's edges are re-entrant for the fluid
        T = np.concatenate([_box((0, 0, 0), (3, 3, 3)), _box((1, 1, 1), (2, 2, 2))])
        seg = Surface([("s", T)]).feature_edges(30.0, concave_only=True)
        self.assertEqual(len(seg), 12)
        self.assertTrue(((seg >= 1 - 1e-12) & (seg <= 2 + 1e-12)).all())

    def test_close_nested_shell(self):
        # inner box close to the outer one: the probes at the outer (convex) edges would
        # land in the inner solid without stopping at its wall.  With the Rust extension
        # and with the sampled fallback of segment_hits (crossings within the first sample
        # interval of 1/64)
        for c in (0.1, 0.001):
            T = np.concatenate([_box((0, 0, 0), (3, 3, 3)), _box((c, c, c), (3 - c, 3 - c, 3 - c))])
            for tris in (T, T[:, ::-1]):
                for rust in (True, False):
                    s = Surface([("s", tris)])
                    if not rust:
                        if s._rs is None:
                            continue
                        s._rs = None
                        s._build_ray_bins()
                    seg = s.feature_edges(30.0, concave_only=True)
                    self.assertEqual(len(seg), 12, (c, rust))
                    self.assertTrue(((seg >= c - 1e-12) & (seg <= 3 - c + 1e-12)).all())

    def test_open_edges_skipped(self):
        tri = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], float)
        self.assertEqual(len(Surface([("s", tri)]).feature_edges(30.0, concave_only=True)), 0)
        self.assertEqual(len(Surface([("s", tri)]).feature_edges(30.0)), 3)


class SlotMouths(unittest.TestCase):
    """Narrow rotor-stator gap with 8 slots in the stator (benchmark2): the slot mouths are
    concave edges at the transition from the fine gap cells to the coarser slot cells.
    Away from the grid axes the solid tip lies in a fine cell whose vertices are all fluid
    or on the walls (no solid vertex to snap)."""
    mm = 1e-3
    Rr, Rb, w, depth, H = 29.92e-3, 30.62e-3, 4e-3, 6e-3, 2.5e-3

    def stl(self, path):
        slots = np.radians(np.arange(8) * 45.0)
        da = np.arcsin(self.w / 2 / self.Rb)
        ang = np.linspace(-np.pi, np.pi, 720, endpoint=False)
        for s in slots:
            ang = ang[np.abs(np.angle(np.exp(1j * (ang - s)))) > da + 1e-6]
        ang = np.sort(np.mod(np.concatenate([ang, slots - da, slots + da]) + np.pi, 2 * np.pi) - np.pi)
        inner = np.column_stack([self.Rr * np.cos(ang), self.Rr * np.sin(ang)])
        outer = np.column_stack([self.Rb * np.cos(ang), self.Rb * np.sin(ang)])
        mouth = {int(np.argmin(np.abs(np.angle(np.exp(1j * (ang - (s - da))))))): s for s in slots}
        caps, rotor, wall = [], [], []
        for i in range(len(ang)):
            j = (i + 1) % len(ang)
            caps.append([inner[i], inner[j], outer[j], outer[i]])
            rotor.append((inner[j], inner[i]))
            if i in mouth:
                u = self.depth * np.array([np.cos(mouth[i]), np.sin(mouth[i])])
                a, b = outer[i], outer[j]
                caps.append([a, b, b + u, a + u])
                wall += [(a, a + u), (a + u, b + u), (b + u, b)]
            else:
                wall.append((outer[i], outer[j]))
        H = self.H

        def sides(E):
            return [t for a, b in E for t in ([[*a, 0], [*b, H], [*b, 0]], [[*a, 0], [*a, H], [*b, H]])]
        tc = []
        for q in caps:          # counter-clockwise in xy
            for k in range(1, len(q) - 1):
                a, b, c = q[0], q[k], q[k + 1]
                tc += [[[*a, 0], [*c, 0], [*b, 0]], [[*a, H], [*b, H], [*c, H]]]
        write_stl_ascii(path, [("rotor", np.array(sides(rotor))), ("wall", np.array(sides(wall) + tc))])
        return np.array([[self.Rb * np.cos(s + g * da), self.Rb * np.sin(s + g * da)] for s in slots for g in (-1, 1)])

    def mesh(self, extra):
        with tempfile.TemporaryDirectory() as d:
            E = self.stl(os.path.join(d, "s.stl"))
            _write_json(os.path.join(d, "m.json"), {
                "domain": "s.stl", "baseCellSize": 0.005, "surfaceRefinement": {"default": 1, "rotor": 2},
                "featureLevel": 2, "gapRefinement": {"thickness": 0.001, "level": 3},
                "rotatingZones": [{"name": "mrf", "cylinder": {"point1": [0, 0, -0.001], "point2": [0, 0, 0.0035],
                                                              "radius": 0.0303625},
                                   "surfaceLevel": 3, "volumeLevel": 2, "omega": 0,
                                   "wallProximity": {"distance": 0.0005, "level": 4, "bufferLayers": 1}}],
                **extra})
            R = generate(load_config(os.path.join(d, "m.json")), log=lambda s: None)
        P, (nn, vv), nI = R["points"], R["faces"], len(R["neighbour"])
        q = quality.check(P, R["faces"], R["owner"], R["neighbour"], log=lambda s: None)
        self.assertEqual(q["nNegVol"], 0)
        self.assertEqual(q["nBadPyramids"], 0)
        self.assertEqual(q["openCellEdges"], 0)
        ptr = np.concatenate([[0], np.cumsum(nn)])
        bf = [vv[ptr[f]:ptr[f + 1]] for f in range(nI, len(nn))]
        e = np.concatenate([np.column_stack([p, np.roll(p, -1)]) for p in bf])
        kd = cKDTree(sample_segments(P[e], 0.005 / 8 / 64))
        zs = np.linspace(0.1, 0.9, 9) * self.H
        # distance from the mouth edges to the nearest wall edge of the mesh, in level-3 cells
        miss = [kd.query(np.column_stack([np.full(9, x), np.full(9, y), zs]))[0] / (0.005 / 8) for x, y in E]
        return np.array(miss)

    def test_mouths_kept(self):
        miss = self.mesh({})
        self.assertLess(miss.max(), 0.1)
        # without the concave pass most of the diagonal mouths are chamfered
        miss0 = self.mesh({"concaveSnapMax": 0.5})
        self.assertGreater((miss0 > 0.2).any(1).sum(), 4)


class PinchedCut(unittest.TestCase):
    """The tip of a wedge of solid on a cell face (a vertex on the concave edge, e.g. a
    hanging vertex at a level transition) with the walls leaving through the opposite
    faces: the kept side is two pieces joined along the edge.  The face chords bend
    through the tip and the cell splits into two cells instead of chamfering the edge."""

    class Wedge:
        """solid: x > |y - 1|"""
        def sdf(self, p):
            p = np.atleast_2d(p)
            return np.minimum(p[:, 0] - (p[:, 1] - 1), p[:, 0] + (p[:, 1] - 1)) / np.sqrt(2)

    def test_two_cells(self):
        dom = self.Wedge()
        X = [(0, 0), (0, 1), (0, 2), (2, 2), (2, 0)]          # bottom loop, tip (0, 1) on the left face
        pos = [[x, y, z] for z in (0.0, 1.0) for x, y in X]
        b, t = list(range(5)), list(range(5, 10))
        faces = [(0, b), (1, t[::-1]),                         # outward
                 (2, [b[0], b[1], t[1], t[0]][::-1]), (3, [b[1], b[2], t[2], t[1]][::-1]),
                 (4, [b[2], b[3], t[3], t[2]][::-1]), (5, [b[3], b[4], t[4], t[3]][::-1]),
                 (6, [b[4], b[0], t[0], t[4]][::-1])]
        phi = [float(dom.sdf(np.array(x))[0]) for x in pos]
        for v in (b[1], t[1]):
            phi[v] = 0.0
        ctx = CutContext(pos, phi, 100)
        ctx.pinch_hook = make_pinch_hook(dom, FeatureSet(np.array([[[0, 1, 0], [0, 1, 1.0]]])), lambda v: 2.0)
        neg, _, caps = ctx.cut(faces, shared_cap=False)
        comps = split_components(neg, ctx.last_pinch)
        self.assertEqual(len(comps), 2)
        P = np.array(ctx.pos)
        for comp in comps:
            de = {}
            for _, q in comp:
                for k in range(len(q)):
                    de[(q[k], q[(k + 1) % len(q)])] = de.get((q[k], q[(k + 1) % len(q)]), 0) + 1
            self.assertTrue(all(de.get((y, x), 0) == c for (x, y), c in de.items()))   # closed
            vol = sum(np.dot(P[q[0]], np.cross(P[q[k]], P[q[k + 1]])) for _, q in comp
                      for k in range(1, len(q) - 1)) / 6
            self.assertAlmostEqual(vol, 0.5, places=9)     # triangle (0,1)-(0,0|2)-(1,0|2) x 1
        # nothing of the wedge kept: all kept vertices on the fluid side
        kept = set(v for comp in comps for _, q in comp for v in q)
        self.assertTrue((dom.sdf(P[sorted(kept)]) <= 1e-12).all())


if __name__ == "__main__":
    unittest.main()
