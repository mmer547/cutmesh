"""Sliding-mesh interfaces (cutmesh/ami.py): "ami": true zones and python -m cutmesh.ami.

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

from cutmesh import ami, quality                                # noqa: E402
from cutmesh.config import ConfigError                          # noqa: E402
from cutmesh.geometry import Cylinder, write_stl_ascii          # noqa: E402
from cutmesh.mesher import load_config, generate, run           # noqa: E402
from cutmesh.readfoam import read_mesh                          # noqa: E402


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


def box_stl(path, lo, hi, name="zone"):
    x0, y0, z0 = lo; x1, y1, z1 = hi
    V = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], float)
    F = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4),
         (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7)]
    write_stl_ascii(path, [(name, V[np.array(F)])])


TWO_ZONES = [
    {"name": "rotor", "cylinder": {"point1": [0, 0, 0.02], "point2": [0, 0, 0.08], "radius": 0.05},
     "ami": True, "omega": 10.0, "surfaceLevel": 2},
    {"name": "mrf", "cylinder": {"point1": [0, 0, 0.12], "point2": [0, 0, 0.18], "radius": 0.05},
     "omega": 5.0},
]


def make_case(d, name, **md):
    cylinder_stl(os.path.join(d, "tank.stl"), 0.1, 0.2)
    cfg = {"domain": "tank.stl", "baseCellSize": 0.025, "rotatingZones": TWO_ZONES, "case": name}
    cfg.update(md)
    p = os.path.join(d, name + ".json")
    with open(p, "w") as f:
        json.dump(cfg, f)
    return p


def rotate_z(P, deg):
    t = math.radians(deg)
    R = np.array([[math.cos(t), -math.sin(t), 0], [math.sin(t), math.cos(t), 0], [0, 0, 1]])
    return P @ R.T


class SlidingMesh(unittest.TestCase):
    def test_com_patches_points_and_rotation(self):
        with tempfile.TemporaryDirectory() as d:
            logs = []
            R = run(make_case(d, "c"), log=logs.append, check=True)
            m = read_mesh(os.path.join(d, "c"))
            const = os.listdir(os.path.join(d, "c", "constant"))
            dyn = open(os.path.join(d, "c", "constant", "dynamicMeshDict")).read()
            mrf = open(os.path.join(d, "c", "constant", "MRFProperties")).read()
        self.assertIn("dynamicMeshDict", const)
        pa = {p["name"]: p for p in m["patches"]}
        a, b = pa["rotor_AMI1"], pa["rotor_AMI2"]
        self.assertEqual((a["type"], b["type"]), ("cyclicAMI", "cyclicAMI"))
        self.assertEqual(a["extra"]["neighbourPatch"], "rotor_AMI2")
        self.assertEqual(b["extra"]["neighbourPatch"], "rotor_AMI1")
        self.assertEqual(a["extra"]["transform"], "noOrdering")
        self.assertEqual(a["nFaces"], b["nFaces"])
        self.assertGreater(a["nFaces"], 100)
        # the MRF zone keeps its faceZone and its MRFProperties entry; the AMI one has neither
        self.assertNotIn("rotor_interface", m["face_zones"])
        self.assertIn("mrf_interface", m["face_zones"])
        self.assertIn("MRF_mrf", mrf)
        self.assertNotIn("MRF_rotor", mrf)
        self.assertIn("dynamicMotionSolverFvMesh", dyn)
        self.assertIn("cellZone        rotor;", dyn)
        self.assertIn("omega       10;", dyn)
        self.assertEqual(R["quality"]["nNegVol"], 0)
        # both patches cover the same area, on the cylinder
        P, faces, own, nei = m["points"], m["faces"], m["owner"], m["neighbour"]
        ff = quality.FlatFaces(faces)
        Cf, Sf = quality.face_geometry(P, ff)
        area = lambda p: np.linalg.norm(Sf[p["startFace"]:p["startFace"] + p["nFaces"]], axis=1).sum()
        self.assertAlmostEqual(area(a), area(b), places=10)
        # no point shared by the zone's cells and the others: the zone can turn on its own
        zp = ami.zone_points(m, "rotor")
        nC = int(max(own.max(), nei.max())) + 1
        inz = np.zeros(nC, bool); inz[m["cell_zones"]["rotor"]] = True
        out = ~inz[own]; out[:len(nei)] |= ~inz[nei]
        op = np.unique(faces[1][np.repeat(out, faces[0])])
        self.assertEqual(len(np.intersect1d(zp, op)), 0)
        for deg in (3.3, 17.0):
            P2 = P.copy(); P2[zp] = rotate_z(P[zp], deg)
            q = quality.check(P2, faces, own, nei, log=None)
            self.assertEqual(q["nNegVol"], 0, deg)
            self.assertEqual(q["openCellEdges"], 0, deg)
            self.assertLess(q["maxClosedness"], 1e-12, deg)
            Cf2, _ = quality.face_geometry(P2, quality.FlatFaces(faces))
            s = slice(a["startFace"], a["startFace"] + a["nFaces"])
            side = (Cf2[s, 2] > 0.021) & (Cf2[s, 2] < 0.079)
            self.assertLess(np.abs(np.hypot(Cf2[s, 0], Cf2[s, 1])[side] - 0.05).max(), 0.002)

    def test_interface_follows_the_rims(self):
        """the cells outside the zone's rims are cut so that no interface face is folded over
        a rim: every face lies on the cap or on the side, and its centre on the cylinder"""
        h = 0.025 / 4
        out = {}
        with tempfile.TemporaryDirectory() as d:
            for key, flag in (("exact", True), ("plain", False)):
                logs = []
                R = run(make_case(d, key, amiExactRims=flag), log=logs.append, check=True)
                m = read_mesh(os.path.join(d, key))
                out[key] = (R["quality"], m, logs)
        cyl = Cylinder([0, 0, 0.02], [0, 0, 0.08], 0.05)

        def interface(m):
            pa = {p["name"]: p for p in m["patches"]}["rotor_AMI1"]
            sizes, flat = m["faces"]
            ptr = np.concatenate([[0], np.cumsum(sizes)])
            ids = range(pa["startFace"], pa["startFace"] + pa["nFaces"])
            return [m["points"][flat[ptr[f]:ptr[f + 1]]] for f in ids]

        def folded(m):
            n = 0
            for Q in interface(m):
                t, r, _ = cyl._local(Q)
                side = np.abs(r - cyl.R) < 1e-3 * h
                cap = (np.abs(t) < 1e-3 * h) | (np.abs(t - cyl.H) < 1e-3 * h)
                n += bool((side & ~cap).any() and (cap & ~side).any())
            return n

        def max_off(m):
            return max(abs(float(cyl.sdf(Q.mean(0)[None])[0])) for Q in interface(m))

        q, m, logs = out["exact"]
        self.assertGreater(folded(out["plain"][1]), 10)        # the case has folds to remove
        self.assertEqual(folded(m), 0)
        self.assertLess(max_off(m), 0.05 * h)
        self.assertGreater(max_off(out["plain"][1]), 0.1 * h)
        self.assertTrue(any("cells outside a rim cut in two" in s for s in logs))
        for k in ("nNegVol", "nBadPyramids", "openCellEdges"):
            self.assertEqual(q[k], 0, k)
        self.assertTrue(q["upperTriangular"])
        self.assertLess(q["maxNonOrth"], 70)
        # the volume is unchanged; the AMI pair still matches face by face
        self.assertAlmostEqual(q["totalVol"], out["plain"][0]["totalVol"], places=12)
        pa = {p["name"]: p for p in m["patches"]}
        self.assertEqual(pa["rotor_AMI1"]["nFaces"], pa["rotor_AMI2"]["nFaces"])

    def test_rims_deterministic(self):
        """the rim cuts give the same mesh from run to run (string hashes are salted per
        process, so a set of cut-edge tokens is ordered differently every time)"""
        import subprocess
        root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
        code = ("import hashlib, os, sys; sys.path.insert(0, sys.argv[1]); from cutmesh.mesher import run; "
                "run(sys.argv[2], log=lambda s: None, check=False); d = os.path.join(sys.argv[3], 'constant', 'polyMesh'); "
                "h = hashlib.sha256(); [h.update(open(os.path.join(d, n), 'rb').read()) for n in sorted(os.listdir(d))]; "
                "print(h.hexdigest())")
        with tempfile.TemporaryDirectory() as d:
            p = make_case(d, "det")
            digests = set()
            for seed in ("1", "2", "3"):
                env = dict(os.environ, PYTHONHASHSEED=seed)
                r = subprocess.run([sys.executable, "-c", code, root, p, os.path.join(d, "det")], env=env, cwd=d,
                                   capture_output=True, text=True)
                self.assertEqual(r.returncode, 0, r.stderr[-2000:])
                digests.add(r.stdout.strip().splitlines()[-1])
            self.assertEqual(len(digests), 1, digests)

    def test_org12_patches_and_hint(self):
        with tempfile.TemporaryDirectory() as d:
            logs = []
            run(make_case(d, "o", openfoam="org12"), log=logs.append, check=False)
            m = read_mesh(os.path.join(d, "o"))
            dyn = open(os.path.join(d, "o", "constant", "dynamicMeshDict")).read()
        pa = {p["name"]: p for p in m["patches"]}
        self.assertEqual(pa["rotor_nonCouple1"]["type"], "patch")
        self.assertEqual(pa["rotor_nonCouple2"]["type"], "patch")
        self.assertNotIn("extra", pa["rotor_nonCouple1"])
        self.assertTrue(any("createNonConformalCouples -overwrite rotor_nonCouple1 rotor_nonCouple2" in s
                            for s in logs))
        self.assertIn("type            motionSolver;", dyn)
        self.assertIn('libs            ("libfvMotionSolvers.so");', dyn)

    def test_cli_matches_run(self):
        with tempfile.TemporaryDirectory() as d:
            # the CLI has no zone geometry: compare without the rim cut
            run(make_case(d, "inrun", amiExactRims=False), log=lambda s: None, check=False)
            plain = [dict(z) for z in TWO_ZONES]; plain[0].pop("ami")
            run(make_case(d, "plain", rotatingZones=plain), log=lambda s: None, check=False)
            ret = ami.main([os.path.join(d, "plain"), "-zone", "rotor", "-out", os.path.join(d, "cli")])
            self.assertEqual(ret, 0)
            a, b = read_mesh(os.path.join(d, "inrun")), read_mesh(os.path.join(d, "cli"))
            dyn_a = open(os.path.join(d, "inrun", "constant", "dynamicMeshDict")).read()
            dyn_b = open(os.path.join(d, "cli", "constant", "dynamicMeshDict")).read()
        self.assertEqual([(p["name"], p["nFaces"]) for p in a["patches"]],
                         [(p["name"], p["nFaces"]) for p in b["patches"]])
        np.testing.assert_allclose(a["points"], b["points"], atol=1e-12)
        np.testing.assert_array_equal(a["owner"], b["owner"])
        self.assertEqual(dyn_a.split("\n", 8)[8], dyn_b.split("\n", 8)[8])   # after the header

    def test_dynamic_mesh_dict_several_zones(self):
        zs = [dict(name="r1", origin=[0, 0, 0], axis=[0, 0, 1], omega=1.0),
              dict(name="r2", origin=[1, 0, 0], axis=[0, 0, -1], omega=2.0)]
        com = ami.dynamic_mesh_dict(zs, "com")
        self.assertIn("motionSolver    multiSolidBodyMotionSolver;", com)
        self.assertIn("rotatingMotionCoeffs", com)
        org = ami.dynamic_mesh_dict(zs, "org10")
        self.assertIn("motionSolver    motionSolverList;", org)
        self.assertIn('("libfvMeshMovers.so" "libfvMotionSolvers.so")', org)
        self.assertIn("cellZone        r2;", org)
        with self.assertRaises(ValueError):
            ami.dynamic_mesh_dict(zs, "org-dev")

    def test_split_errors(self):
        with tempfile.TemporaryDirectory() as d:
            run(make_case(d, "e"), log=lambda s: None, check=False)
            m = read_mesh(os.path.join(d, "e"))
        with self.assertRaises(ValueError):          # already split: no faceZone any more
            ami.split_interface(m, "rotor")
        with self.assertRaises(ValueError):
            ami.split_interface(m, "mrf", "bad")

    def test_config_errors(self):
        with tempfile.TemporaryDirectory() as d:
            zs = [dict(TWO_ZONES[0], conforming=False)]
            with self.assertRaises(ConfigError):
                generate(load_config(make_case(d, "x", rotatingZones=zs)), log=lambda s: None)
            with self.assertRaises(ConfigError):
                generate(load_config(make_case(d, "y", openfoam="org9")), log=lambda s: None)

    def test_non_round_zone_warns(self):
        with tempfile.TemporaryDirectory() as d:
            box_stl(os.path.join(d, "box.stl"), (-0.04, -0.04, 0.03), (0.04, 0.04, 0.09))
            zs = [{"name": "rotor", "stl": "box.stl", "origin": [0, 0, 0], "axis": [0, 0, 1], "ami": True}]
            logs = []
            run(make_case(d, "w", rotatingZones=zs), log=logs.append, check=False)
        self.assertTrue(any(s.startswith("WARNING: zone 'rotor': its interface is not a surface of"
                                         " revolution") for s in logs))

    def test_cli_in_place_drops_converted_mrf_entry(self):
        # converting a zone in place: its MRF entry goes (the mesh turns it now), the other stays
        with tempfile.TemporaryDirectory() as d:
            plain = [dict(z) for z in TWO_ZONES]; plain[0].pop("ami")
            run(make_case(d, "p", rotatingZones=plain), log=lambda s: None, check=False)
            case = os.path.join(d, "p")
            before = open(os.path.join(case, "constant", "MRFProperties")).read()
            self.assertEqual(ami.main([case, "-zone", "rotor"]), 0)
            after = open(os.path.join(case, "constant", "MRFProperties")).read()
            dyn = open(os.path.join(case, "constant", "dynamicMeshDict")).read()
        self.assertIn("MRF_rotor", before)
        self.assertNotIn("MRF_rotor", after)
        self.assertIn("MRF_mrf", after)
        self.assertIn("FoamFile", after)
        self.assertIn("omega       10;", dyn)                   # read before the entry went

    def test_rerun_with_all_zones_ami_clears_mrf(self):
        # a case meshed with MRF zones, meshed again with every zone sliding: no stale entries
        with tempfile.TemporaryDirectory() as d:
            plain = [dict(z) for z in TWO_ZONES]; plain[0].pop("ami")
            run(make_case(d, "r", rotatingZones=plain), log=lambda s: None, check=False)
            both = [dict(z, ami=True) for z in TWO_ZONES]
            logs = []
            run(make_case(d, "r", rotatingZones=both), log=logs.append, check=False)
            mrf = open(os.path.join(d, "r", "constant", "MRFProperties")).read()
        self.assertNotIn("MRF_rotor", mrf)
        self.assertNotIn("MRF_mrf", mrf)
        self.assertTrue(any("MRFProperties: removed 2 entries" in s for s in logs))

    def test_read_mrf(self):
        with tempfile.TemporaryDirectory() as d:
            run(make_case(d, "m", rotatingZones=[dict(TWO_ZONES[1])]), log=lambda s: None, check=False)
            r = ami.read_mrf(os.path.join(d, "m"))
        self.assertEqual(r["mrf"]["omega"], 5.0)
        self.assertEqual(r["mrf"]["axis"], [0, 0, 1])
        self.assertEqual(r["mrf"]["origin"], [0, 0, 0.12])


if __name__ == "__main__":
    unittest.main()
