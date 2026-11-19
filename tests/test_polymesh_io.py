"""polyMesh writing in binary / ASCII and reading back (cutmesh/polymesh.py, cutmesh/readfoam.py).

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

from cutmesh.geometry import write_stl_ascii                  # noqa: E402
from cutmesh.mesher import load_config, generate, run         # noqa: E402
from cutmesh.polymesh import write_polymesh                   # noqa: E402
from cutmesh.readfoam import read_mesh, FoamFileError         # noqa: E402


def hex_grid(n=3):
    """n^3 unit hexes: points, CSR faces (internal first, upper-triangular), owner, neighbour,
    one wall patch, a cell zone and a face zone"""
    P = np.array([(x, y, z) for z in range(n + 1) for y in range(n + 1) for x in range(n + 1)], float)
    v = lambda i, j, k: (k * (n + 1) + j) * (n + 1) + i
    c = lambda i, j, k: (k * n + j) * n + i
    internal, boundary = [], []
    for k in range(n):
        for j in range(n):
            for i in range(n):
                me = c(i, j, k)
                quads = [  # (face vertices pointing to +axis, neighbour index or None, side)
                    ([v(i + 1, j, k), v(i + 1, j + 1, k), v(i + 1, j + 1, k + 1), v(i + 1, j, k + 1)],
                     c(i + 1, j, k) if i + 1 < n else None),
                    ([v(i, j + 1, k), v(i, j + 1, k + 1), v(i + 1, j + 1, k + 1), v(i + 1, j + 1, k)],
                     c(i, j + 1, k) if j + 1 < n else None),
                    ([v(i, j, k + 1), v(i + 1, j, k + 1), v(i + 1, j + 1, k + 1), v(i, j + 1, k + 1)],
                     c(i, j, k + 1) if k + 1 < n else None),
                ]
                for q, nb in quads:
                    (internal if nb is not None else boundary).append((q, me, nb))
                if i == 0:
                    boundary.append(([v(0, j, k), v(0, j, k + 1), v(0, j + 1, k + 1), v(0, j + 1, k)], me, None))
                if j == 0:
                    boundary.append(([v(i, 0, k), v(i + 1, 0, k), v(i + 1, 0, k + 1), v(i, 0, k + 1)], me, None))
                if k == 0:
                    boundary.append(([v(i, j, 0), v(i, j + 1, 0), v(i + 1, j + 1, 0), v(i + 1, j, 0)], me, None))
    internal.sort(key=lambda t: (t[1], t[2]))
    faces = [q for q, _, _ in internal + boundary]
    owner = np.array([o for _, o, _ in internal + boundary], np.int64)
    nei = np.array([nb for _, _, nb in internal], np.int64)
    sizes = np.array([len(q) for q in faces], np.int64)
    flat = np.array([x for q in faces for x in q], np.int64)
    patches = [dict(name="walls", type="wall", startFace=len(internal), nFaces=len(boundary))]
    cz = {"rotor": np.arange(0, n ** 3, 2, dtype=np.int64)}
    fz = {"rotor_interface": (np.arange(3, dtype=np.int64), np.array([True, False, True]))}
    return P, (sizes, flat), owner, nei, patches, cz, fz


class RoundTrip(unittest.TestCase):
    def check(self, fmt):
        P, F, own, nei, patches, cz, fz = hex_grid()
        P = P * math.pi / 7 + 1e-3                  # values not exact in 12 digits
        with tempfile.TemporaryDirectory() as d:
            write_polymesh(d, P, F, own, nei, patches, cz, fz, fmt=fmt)
            m = read_mesh(d)
        self.assertEqual(m["format"], fmt)
        if fmt == "binary":
            np.testing.assert_array_equal(m["points"], P)          # bit-exact
        else:
            np.testing.assert_allclose(m["points"], P, rtol=1e-11)
        np.testing.assert_array_equal(m["faces"][0], F[0])
        np.testing.assert_array_equal(m["faces"][1], F[1])
        np.testing.assert_array_equal(m["owner"], own)
        np.testing.assert_array_equal(m["neighbour"], nei)
        self.assertEqual(m["patches"], patches)
        np.testing.assert_array_equal(m["cell_zones"]["rotor"], cz["rotor"])
        np.testing.assert_array_equal(m["face_zones"]["rotor_interface"][0], fz["rotor_interface"][0])
        np.testing.assert_array_equal(m["face_zones"]["rotor_interface"][1], fz["rotor_interface"][1])

    def test_binary(self):
        self.check("binary")

    def test_ascii(self):
        self.check("ascii")

    def test_list_faces_input(self):
        # faces given as a list of vertex lists (python -m cutmesh.layers path) write the same
        P, F, own, nei, patches, cz, fz = hex_grid(2)
        sizes, flat = F
        ptr = np.concatenate([[0], np.cumsum(sizes)])
        faces = [flat[ptr[i]:ptr[i + 1]].tolist() for i in range(len(sizes))]
        with tempfile.TemporaryDirectory() as d:
            write_polymesh(d, P, faces, own, nei, patches, fmt="binary")
            m = read_mesh(d)
        np.testing.assert_array_equal(m["faces"][1], flat)

    def test_binary_layout(self):
        # OpenFOAM's layout: arch in the header, faceCompactList, "N\n(" raw ")" per list,
        # an empty list is "0" with no brackets
        P, F, own, nei, patches, _, _ = hex_grid(1)            # one cell: no internal faces
        with tempfile.TemporaryDirectory() as d:
            write_polymesh(d, P, F, own, nei, patches, fmt="binary")
            pm = os.path.join(d, "constant", "polyMesh")
            faces = open(os.path.join(pm, "faces"), "rb").read()
            neigh = open(os.path.join(pm, "neighbour"), "rb").read()
            bnd = open(os.path.join(pm, "boundary"), "rb").read()
            m = read_mesh(d)
        self.assertIn(b'format      binary;', faces)
        self.assertIn(b'arch        "LSB;label=32;scalar=64";', faces)
        self.assertIn(b'class       faceCompactList;', faces)
        body = faces[faces.index(b"* //\n") + 5:].lstrip()
        off = np.arange(0, 28, 4, dtype="<i4").tobytes()       # 6 quads: offsets 0, 4, ..., 24
        self.assertTrue(body.startswith(b"7\n(" + off + b")\n"), body[:40])
        self.assertTrue(neigh.rstrip().endswith(b"\n0"), neigh[-20:])
        self.assertIn(b"format      ascii;", bnd)
        self.assertEqual(len(m["neighbour"]), 0)
        self.assertEqual(len(m["faces"][0]), 6)

    def test_arch_header_with_semicolons(self):
        # the quoted arch value holds ';': it must not end the entry at the first one
        from cutmesh.readfoam import _read_file, _dtypes
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x")
            with open(p, "wb") as f:
                f.write(b'FoamFile\n{\n    version     2.0;\n    format      binary;\n'
                        b'    arch        "LSB;label=64;scalar=32";\n    class       labelList;\n'
                        b'    note        "nPoints:1  nCells:2";\n    object      owner;\n}\n0\n')
            hdr, _, _ = _read_file(p)
        self.assertEqual(hdr["arch"], "LSB;label=64;scalar=32")
        self.assertEqual(hdr["note"], "nPoints:1  nCells:2")
        self.assertEqual(hdr["format"], "binary")
        lab, sca = _dtypes(hdr)
        self.assertEqual((lab, sca), (np.dtype("<i8"), np.dtype("<f4")))

    def test_binary_64bit_labels(self):
        # meshes too large for 32-bit labels are written with label=64; they read back
        from cutmesh.polymesh import _write_binary
        P, F, own, nei, patches, cz, fz = hex_grid(2)
        with tempfile.TemporaryDirectory() as d:
            write_polymesh(d, P, F, own, nei, patches, cz, fz, fmt="binary")
            pm = os.path.join(d, "constant", "polyMesh")
            _write_binary(pm, P, F[0], F[1], own, nei, "", label_bits=64)
            self.assertIn(b"label=64", open(os.path.join(pm, "owner"), "rb").read(400))
            m = read_mesh(d)
        np.testing.assert_array_equal(m["faces"][1], F[1])
        np.testing.assert_array_equal(m["owner"], own)
        np.testing.assert_array_equal(m["neighbour"], nei)

    def test_binary_single_precision_points(self):
        # OpenFOAM built with 32-bit scalars writes scalar=32
        P, F, own, nei, patches, _, _ = hex_grid(1)
        with tempfile.TemporaryDirectory() as d:
            write_polymesh(d, P, F, own, nei, patches, fmt="binary")
            p = os.path.join(d, "constant", "polyMesh", "points")
            b = open(p, "rb").read()
            head = b[:b.index(b"* //\n") + 5].replace(b"scalar=64", b"scalar=32")
            with open(p, "wb") as f:
                f.write(head + b"\n8\n(" + P.astype("<f4").tobytes() + b")\n")
            m = read_mesh(d)
        np.testing.assert_array_equal(m["points"], P)

    def test_truncated_binary(self):
        P, F, own, nei, patches, _, _ = hex_grid(2)
        with tempfile.TemporaryDirectory() as d:
            write_polymesh(d, P, F, own, nei, patches, fmt="binary")
            p = os.path.join(d, "constant", "polyMesh", "owner")
            b = open(p, "rb").read()
            open(p, "wb").write(b[:-10])
            with self.assertRaises(FoamFileError):
                read_mesh(d)

    def test_bad_format(self):
        P, F, own, nei, patches, _, _ = hex_grid(1)
        with tempfile.TemporaryDirectory() as d, self.assertRaises(ValueError):
            write_polymesh(d, P, F, own, nei, patches, fmt="text")


def cylinder_stl(path, r, h, n=32):
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


class MesherOutput(unittest.TestCase):
    def test_binary_default_and_ascii_same_mesh(self):
        with tempfile.TemporaryDirectory() as d:
            cylinder_stl(os.path.join(d, "tank.stl"), 0.1, 0.2)
            md = {"domain": "tank.stl", "baseCellSize": 0.04,
                  "rotatingZones": [{"name": "rotor", "cylinder": {"point1": [0, 0, 0.06],
                                     "point2": [0, 0, 0.14], "radius": 0.05}}]}
            out = {}
            for fmt in ("binary", "ascii"):
                m = dict(md, case=fmt)
                if fmt == "ascii":
                    m["writeFormat"] = "ascii"
                p = os.path.join(d, f"{fmt}.json")
                with open(p, "w") as f:
                    json.dump(m, f)
                run(p, log=lambda s: None, check=False)
                out[fmt] = read_mesh(os.path.join(d, fmt))
        self.assertEqual(out["binary"]["format"], "binary")     # the default
        self.assertEqual(out["ascii"]["format"], "ascii")
        a, b = out["ascii"], out["binary"]
        np.testing.assert_allclose(a["points"], b["points"], rtol=0, atol=1e-12)
        for k in ("owner", "neighbour"):
            np.testing.assert_array_equal(a[k], b[k])
        np.testing.assert_array_equal(a["faces"][1], b["faces"][1])
        self.assertEqual(a["patches"], b["patches"])
        np.testing.assert_array_equal(a["cell_zones"]["rotor"], b["cell_zones"]["rotor"])

    def test_bad_write_format(self):
        from cutmesh.config import ConfigError
        with tempfile.TemporaryDirectory() as d:
            cylinder_stl(os.path.join(d, "tank.stl"), 0.1, 0.2)
            p = os.path.join(d, "m.json")
            with open(p, "w") as f:
                json.dump({"domain": "tank.stl", "baseCellSize": 0.04, "writeFormat": "bin"}, f)
            with self.assertRaises(ConfigError):
                generate(load_config(p), log=lambda s: None)


if __name__ == "__main__":
    unittest.main()
