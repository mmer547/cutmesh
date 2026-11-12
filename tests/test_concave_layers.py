"""Prism layers at concave corners (layers.py, concaveSlide): in a closed cylinder the
flat ends meet the side wall at a concave edge.  The wall points of one wall next to the
edge, moving along their own normal, end up behind the corner point (which moves along
the bisector) and the layer cells between them invert; sliding them along their wall
keeps the layers at full thickness up to the corner.

python -m unittest discover -s tests
"""
import json
import os
import re
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from cutmesh import quality                                   # noqa: E402
from cutmesh.geometry import write_stl_ascii                  # noqa: E402
from cutmesh.layers import add_layers                         # noqa: E402
from cutmesh.mesher import load_config, generate              # noqa: E402


def cylinder_stl(path, R, z0, z1, n=96):
    """closed cylinder around the z axis, outward normals"""
    t = np.linspace(0, 2 * np.pi, n + 1)[:-1]
    c = np.stack([R * np.cos(t), R * np.sin(t)], 1)
    tris = []
    for i in range(n):
        a, b = c[i], c[(i + 1) % n]
        A0, B0, A1, B1 = np.r_[a, z0], np.r_[b, z0], np.r_[a, z1], np.r_[b, z1]
        tris += [[A0, B0, B1], [A0, B1, A1], [np.r_[0, 0, z0], B0, A0], [np.r_[0, 0, z1], A1, B1]]
    write_stl_ascii(path, [("walls", np.array(tris))])


class ConcaveCorner(unittest.TestCase):
    h = 0.01
    _mesh = None

    @classmethod
    def mesh(cls):
        if cls._mesh is None:
            with tempfile.TemporaryDirectory() as d:
                cylinder_stl(os.path.join(d, "s.stl"), 4.37 * cls.h, -3.13 * cls.h, 3.13 * cls.h)
                with open(os.path.join(d, "m.json"), "w") as f:
                    json.dump({"domain": "s.stl", "baseCellSize": cls.h,
                               "surfaceRefinement": {"default": 0}, "featureLevel": 0}, f)
                cls._mesh = generate(load_config(os.path.join(d, "m.json")), log=lambda s: None)
        return cls._mesh

    def layers(self, spec):
        msgs = []
        L = add_layers(self.mesh(), spec, log=msgs.append)
        full = float(re.search(r"full thickness ([0-9.]+)%", " ".join(msgs)).group(1)) / 100
        return L, full, msgs

    def test_full_thickness_up_to_the_corner(self):
        _, full_off, _ = self.layers({"concaveSlide": None})
        L, full, msgs = self.layers({})
        self.assertGreater(full, 0.9)
        self.assertGreater(full, full_off + 0.3)
        self.assertFalse(any("WARNING" in m for m in msgs))
        q = quality.check(L["points"], L["faces"], L["owner"], L["neighbour"], log=lambda s: None)
        self.assertIsNotNone(q)

    def test_corner_point_on_the_bisector(self):
        # the points on the edge move off both walls by the same thickness
        L, _, _ = self.layers({"nIter": 0, "nCollapseIter": 0})
        P0 = np.asarray(self.mesh()["points"])
        W, T = L["layer_thickness"]["points"], L["layer_thickness"]["T"]
        X, D = P0[W], np.asarray(L["points"])[W] - P0[W]
        R, Z = 4.37 * self.h, 3.13 * self.h
        r = np.hypot(X[:, 0], X[:, 1])
        edge = (np.abs(np.abs(X[:, 2]) - Z) < 1e-9) & (np.abs(r - R) < 1e-3 * self.h) & (T > 0)
        self.assertGreater(edge.sum(), 10)
        radial = -np.einsum("ij,ij->i", D[edge][:, :2], X[edge][:, :2]) / r[edge]
        axial = -np.sign(X[edge, 2]) * D[edge, 2]
        np.testing.assert_allclose(radial / T[edge], 1.0, atol=0.08)
        np.testing.assert_allclose(axial / T[edge], 1.0, atol=0.08)


if __name__ == "__main__":
    unittest.main()
