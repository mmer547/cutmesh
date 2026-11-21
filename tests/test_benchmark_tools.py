"""benchmark/mesh_hash.py and the mesh check in run_bench.compare (--same-mesh)."""
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "benchmark"))

import mesh_hash  # noqa: E402
import run_bench  # noqa: E402


def write_case(root, owner=b"0 1 2"):
    d = os.path.join(root, "constant", "polyMesh")
    os.makedirs(d)
    for name, data in (("owner", owner), ("points", b"(0 0 0)"), ("faces", b"3(0 1 2)")):
        with open(os.path.join(d, name), "wb") as f:
            f.write(data)
    return root


def result(name, h):
    return {"name": name, "status": "PASS", "meshHash": h, "quality": {}}


class MeshHash(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        t = self.tmp.name
        self.a = mesh_hash.mesh_hash(write_case(os.path.join(t, "a")))
        self.b = mesh_hash.mesh_hash(write_case(os.path.join(t, "b")))
        self.c = mesh_hash.mesh_hash(write_case(os.path.join(t, "c"), owner=b"0 1 3"))

    def tearDown(self):
        self.tmp.cleanup()

    def test_hash(self):
        self.assertEqual(self.a, self.b)
        self.assertEqual(mesh_hash.diff(self.a, self.c), ["owner"])
        # the polyMesh directory itself works too
        d = os.path.join(self.tmp.name, "a", "constant", "polyMesh")
        self.assertEqual(mesh_hash.mesh_hash(d), self.a)
        self.assertEqual(mesh_hash.main([os.path.join(self.tmp.name, "a"), os.path.join(self.tmp.name, "c")]), 1)

    def test_compare(self):
        base = {"platform": run_bench.platform_id(), "results": [result("x", self.a)]}
        same = run_bench.compare([result("x", self.b)], base, same_mesh=True)
        self.assertEqual(same, ["same   x: same mesh"])
        changed = run_bench.compare([result("x", self.c)], base, same_mesh=True)
        self.assertTrue(changed[0].startswith("WORSE") and "owner" in changed[0], changed)
        info = run_bench.compare([result("x", self.c)], base)
        self.assertTrue(info[0].startswith("mesh"), info)
        other = dict(base, platform="another machine")
        lines = run_bench.compare([result("x", self.c)], other, same_mesh=True)
        self.assertTrue(lines[0].startswith("note") and not any(s.startswith("WORSE") for s in lines), lines)


if __name__ == "__main__":
    unittest.main()
