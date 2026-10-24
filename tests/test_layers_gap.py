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
from cutmesh.layers import add_layers                         # noqa: E402
from cutmesh.mesher import load_config, generate              # noqa: E402
from test_thin_gaps import voxel_stl, _write_json             # noqa: E402


class LayersInGap(unittest.TestCase):
    h = 0.01

    def test_slot_layers_limited(self):
        h, v = self.h, self.h / 10
        G = np.zeros((60, 60, 90), bool)
        G[:, :, 0:30] = True                    # chamber z in [0, 3h]
        G[:, :, 60:90] = True                   # chamber z in [6h, 9h]
        y0, y1 = 22, 37                         # slot 1.5 h wide
        G[10:50, y0:y1, 30:60] = True
        with tempfile.TemporaryDirectory() as d:
            voxel_stl(os.path.join(d, "s.stl"), G, v)
            _write_json(os.path.join(d, "m.json"), {"domain": "s.stl", "baseCellSize": h,
                                                    "surfaceRefinement": {"default": 0}, "featureLevel": 0})
            R = generate(load_config(os.path.join(d, "m.json")), log=lambda s: None)
        gap = (y1 - y0) * v
        P = np.asarray(R["points"])
        kept = {}
        for gf in (0.3, 0.0):
            L = add_layers(R, {"nLayers": 3, "maxGapFraction": gf}, log=lambda s: None)
            q = quality.check(L["points"], L["faces"], L["owner"], L["neighbour"], log=lambda s: None)
            self.assertEqual(q["nNegVol"], 0)
            self.assertEqual(q["nBadPyramids"], 0)
            W, T = L["layer_thickness"]["points"], L["layer_thickness"]["T"]
            X = P[W]
            slot = (X[:, 2] > 3.3 * h) & (X[:, 2] < 5.7 * h) & (X[:, 0] > 1.3 * h) & (X[:, 0] < 4.7 * h)
            self.assertGreater(slot.sum(), 10)
            kept[gf] = (T[slot] > 0).mean()
            if gf:
                self.assertLessEqual(T[slot].max(), gf * gap * 1.05)
                self.assertGreater(kept[gf], 0.8)                # the slot walls keep layers
        self.assertGreaterEqual(kept[0.3], kept[0.0])


if __name__ == "__main__":
    unittest.main()
