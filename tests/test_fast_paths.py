"""Faster code paths that must give the same answers as the plain ones: the open cell
edge count of the quality check, duplicate-free surface sampling, sorted-query search.

python -m unittest discover -s tests
"""
import io
import contextlib
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cutmesh import geometry, quality                          # noqa: E402
from test_polymesh_io import hex_grid                           # noqa: E402


def open_edges_brute(faces, owner, neighbour):
    """directed edges (cell, a, b) whose reverse (cell, b, a) is missing, by sets"""
    nI = len(neighbour)
    edges = []
    for f, p in enumerate(faces):
        for i in range(len(p)):
            a, b = p[i], p[(i + 1) % len(p)]
            edges.append((owner[f], a, b))
            if f < nI:
                edges.append((neighbour[f], b, a))
    have = set(edges)
    return sum(1 for c, a, b in edges if (c, b, a) not in have)


class OpenCellEdges(unittest.TestCase):
    def run_check(self, P, faces, own, nei):
        with contextlib.redirect_stdout(io.StringIO()):
            return quality.check(P, faces, own, nei, log=lambda *a: None)

    def test_closed_and_broken(self):
        P, (sizes, flat), own, nei, _, _, _ = hex_grid(3)
        ptr = np.concatenate([[0], np.cumsum(sizes)])
        faces = [flat[ptr[i]:ptr[i + 1]].tolist() for i in range(len(sizes))]
        self.assertEqual(self.run_check(P, faces, own, nei)["openCellEdges"], 0)
        # drop a vertex from a few faces (internal and boundary): their cells get open edges
        rng = np.random.default_rng(3)
        for f in rng.choice(len(faces), 6, replace=False).tolist():
            faces[f] = faces[f][1:]
        got = self.run_check(P, faces, own, nei)["openCellEdges"]
        self.assertGreater(got, 0)
        self.assertEqual(got, open_edges_brute(faces, own, nei))


class SurfaceSamples(unittest.TestCase):
    def test_same_points_without_repeats(self):
        rng = np.random.default_rng(0)
        tris = rng.uniform(-1, 1, (40, 3, 3))
        tris[5] = tris[4][[1, 2, 0]]                  # a repeated triangle stays two inputs
        for sp in (0.05, 0.2, 5.0):
            p, ti = geometry._sample_triangles(tris, sp)
            T, par = geometry.subdivide_triangles(tris, sp)
            q = np.concatenate([T.reshape(-1, 3), T.mean(1)])
            qi = np.concatenate([np.repeat(par, 3), par])

            def rows(x, i):
                k = np.empty((len(x), 4), np.int64)
                k[:, :3] = np.ascontiguousarray(x).view(np.int64)
                k[:, 3] = i
                return {tuple(r) for r in k.tolist()}
            a, b = rows(p, ti), rows(q, qi)
            self.assertEqual(a, b)                    # the same (point, triangle) set
            self.assertEqual(len(a), len(p))          # and each once


class SortedSearch(unittest.TestCase):
    def test_matches_numpy(self):
        rng = np.random.default_rng(1)
        a = np.unique(rng.integers(0, 10 ** 9, 50000))
        for n in (10, 100000):
            v = rng.integers(-5, 10 ** 9 + 5, (n, 2))
            np.testing.assert_array_equal(geometry.searchsorted(a, v), np.searchsorted(a, v))


if __name__ == "__main__":
    unittest.main()
