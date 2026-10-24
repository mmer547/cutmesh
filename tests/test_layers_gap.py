"""Prism layers in a narrow gap: the thickness is limited to maxGapFraction x the distance
to the facing wall, so the layers from both walls do not meet (layers.py).

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
from cutmesh.layers import add_layers, _gap_width             # noqa: E402
from cutmesh.mesher import load_config, generate              # noqa: E402
from test_thin_gaps import voxel_stl, _write_json             # noqa: E402


def _split_patch(R, key):
    """split the single wall patch of R into wall_a (key(face centre) true) and wall_b"""
    nn, vv = (np.asarray(a, np.int64) for a in R["faces"])
    own, nei = np.asarray(R["owner"], np.int64), np.asarray(R["neighbour"], np.int64)
    (p,) = [p for p in R["patches"] if p["type"] == "wall"]
    nI, s0, s1 = len(nei), p["startFace"], p["startFace"] + p["nFaces"]
    ptr = np.concatenate([[0], np.cumsum(nn)])
    P = np.asarray(R["points"])
    a = np.array([bool(key(P[vv[ptr[f]:ptr[f + 1]]].mean(0))) for f in range(s0, s1)])
    order = np.concatenate([np.arange(s0), s0 + np.nonzero(a)[0], s0 + np.nonzero(~a)[0],
                            np.arange(s1, len(nn))])
    flat = np.concatenate([vv[ptr[f]:ptr[f + 1]] for f in order])
    patches = []
    for q in R["patches"]:
        if q is p:
            patches.append(dict(q, name="wall_a", nFaces=int(a.sum())))
            patches.append(dict(q, name="wall_b", startFace=s0 + int(a.sum()), nFaces=int((~a).sum())))
        else:
            patches.append(q)
    assert s0 >= nI
    return dict(R, faces=(nn[order], flat), owner=own[order], patches=patches)


class LayersInGap(unittest.TestCase):
    h = 0.01
    y0, y1 = 22, 37                             # slot 1.5 h wide (voxels of h / 10)
    _mesh = None

    @classmethod
    def slot_mesh(cls):
        if cls._mesh is None:
            h, v = cls.h, cls.h / 10
            G = np.zeros((60, 60, 90), bool)
            G[:, :, 0:30] = True                # chamber z in [0, 3h]
            G[:, :, 60:90] = True               # chamber z in [6h, 9h]
            G[10:50, cls.y0:cls.y1, 30:60] = True
            with tempfile.TemporaryDirectory() as d:
                voxel_stl(os.path.join(d, "s.stl"), G, v)
                _write_json(os.path.join(d, "m.json"), {"domain": "s.stl", "baseCellSize": h,
                                                        "surfaceRefinement": {"default": 0},
                                                        "featureLevel": 0})
                cls._mesh = generate(load_config(os.path.join(d, "m.json")), log=lambda s: None)
        return cls._mesh

    def _slot(self, X):
        h = self.h
        return (X[:, 2] > 3.3 * h) & (X[:, 2] < 5.7 * h) & (X[:, 0] > 1.3 * h) & (X[:, 0] < 4.7 * h)

    def test_one_side_only(self):
        # layers on one slot wall only: the other wall (no layers) still limits them
        h, gf = self.h, 0.3
        R = _split_patch(self.slot_mesh(), lambda c: c[1] < (self.y0 + self.y1) * h / 20)
        gap = (self.y1 - self.y0) * h / 10
        P = np.asarray(R["points"])
        L = add_layers(R, {"patches": ["wall_a"], "nLayers": 3, "maxGapFraction": gf}, log=lambda s: None)
        W, T = L["layer_thickness"]["points"], L["layer_thickness"]["T"]
        X = P[W]
        side = self._slot(X) & (np.abs(X[:, 1] - self.y0 * h / 10) < 1e-6)
        self.assertGreater(side.sum(), 5)
        self.assertLessEqual(T[side].max(), gf * gap * 1.05)

    def test_gap_width_dense_wall(self):
        # 40 x 40 points on the own wall within reach of every point: the facing wall
        # lies beyond the 48 nearest points but must still be found
        g = np.linspace(-1, 1, 81)
        a = np.array([(x, y, 0.0) for x in g for y in g])
        b = np.array([(x, y, 0.5) for x in g[::8] for y in g[::8]])
        PO = np.concatenate([a, b])
        nO = np.concatenate([np.tile([0, 0, -1.0], (len(a), 1)), np.tile([0, 0, 1.0], (len(b), 1))])
        X = np.array([[0.0, 0.0, 0.0]])
        w = _gap_width(X, np.array([[0, 0, 1.0]]), np.array([0.05]), np.array([1.0]), PO, nO)
        self.assertAlmostEqual(w[0], 0.5)
        w = _gap_width(X, np.array([[0, 0, 1.0]]), np.array([0.05]), np.array([0.4]), PO, nO)
        self.assertEqual(w[0], np.inf)                       # beyond the range

    def test_slot_layers_limited(self):
        h = self.h
        R = self.slot_mesh()
        gap = (self.y1 - self.y0) * h / 10
        P = np.asarray(R["points"])
        kept = {}
        for gf in (0.3, 0.0):
            L = add_layers(R, {"nLayers": 3, "maxGapFraction": gf}, log=lambda s: None)
            q = quality.check(L["points"], L["faces"], L["owner"], L["neighbour"], log=lambda s: None)
            self.assertEqual(q["nNegVol"], 0)
            self.assertEqual(q["nBadPyramids"], 0)
            W, T = L["layer_thickness"]["points"], L["layer_thickness"]["T"]
            X = P[W]
            slot = self._slot(X)
            self.assertGreater(slot.sum(), 10)
            kept[gf] = (T[slot] > 0).mean()
            if gf:
                self.assertLessEqual(T[slot].max(), gf * gap * 1.05)
                self.assertGreater(kept[gf], 0.8)                # the slot walls keep layers
        self.assertGreaterEqual(kept[0.3], kept[0.0])


if __name__ == "__main__":
    unittest.main()
