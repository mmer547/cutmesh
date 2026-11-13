"""STL repair on load (geometry.repair_triangles): what is removed, what is reported, and that
every change is logged."""
import unittest

import numpy as np

from cutmesh.geometry import Surface, repair_triangles


def box():
    v = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], float)
    f = [[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
         [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]]
    return v[np.array(f)]


class TestRepair(unittest.TestCase):
    def run_repair(self, tris):
        lines = []
        S = Surface([("walls", tris)], log=lines.append)
        return S, lines

    def test_clean_box_unchanged(self):
        S, lines = self.run_repair(box())
        self.assertEqual(len(S.tris), 12)
        self.assertAlmostEqual(S.repair["volume"], 1.0)
        self.assertFalse(any("REPAIR" in s or "WARNING" in s for s in lines))
        self.assertTrue(any("closed, consistently oriented" in s for s in lines))

    def test_same_orientation_duplicates_keep_one(self):
        t = box()
        S, lines = self.run_repair(np.concatenate([t, t[:3], t[:1]]))
        self.assertEqual(len(S.tris), 12)
        self.assertEqual(S.repair["duplicates"], 4)
        self.assertEqual(S.repair["openEdges"], 0)
        self.assertTrue(any(s.startswith("REPAIR:") and "4 duplicated" in s for s in lines))

    def test_opposite_pair_is_removed(self):
        t = box()
        membrane = np.array([[[0, 0, .5], [1, 0, .5], [1, 1, .5]]])
        S, lines = self.run_repair(np.concatenate([t, membrane, membrane[:, ::-1]]))
        self.assertEqual(len(S.tris), 12)
        self.assertEqual(S.repair["membranes"], 1)
        self.assertTrue(any("zero-thickness membranes" in s for s in lines))

    def test_mixed_copies_cancel_pairs_and_keep_majority(self):
        # 3 copies of one face in its own orientation + 1 reversed: one pair cancels, one copy
        # of the majority stays; a face duplicated once in each orientation goes completely
        t = box()
        extra = [t[:1], t[:1], t[:1, ::-1], t[1:2, ::-1]]
        _, _, rep = repair_triangles(np.concatenate([t] + extra), np.zeros(16, int))
        self.assertEqual(rep["membranes"], 2)
        self.assertEqual(rep["duplicates"], 1)
        self.assertEqual(rep["kept"], 11)
        self.assertEqual(rep["openEdges"], 3)

    def test_zero_area_removed(self):
        t = box()
        S, lines = self.run_repair(np.concatenate([t, np.repeat(t[:2, :1], 3, 1)]))
        self.assertEqual(len(S.tris), 12)
        self.assertEqual(S.repair["zeroArea"], 2)
        self.assertTrue(any("2 zero-area" in s for s in lines))

    def test_open_and_inconsistent_are_reported_not_changed(self):
        t = box()
        S, lines = self.run_repair(t[2:])
        self.assertEqual(len(S.tris), 10)
        self.assertEqual(S.repair["openEdges"], 4)
        self.assertTrue(any(s.startswith("WARNING") and "4 open edges" in s for s in lines))
        t2 = t.copy(); t2[0] = t2[0, ::-1]
        _, _, rep = repair_triangles(t2, np.zeros(12, int))
        self.assertEqual(rep["inconsistentEdges"], 3)

    def test_regions_follow_kept_triangles(self):
        t = box()
        S = Surface([("a", t[:6]), ("b", np.concatenate([t[6:], t[6:7]]))])
        self.assertEqual(len(S.tris), 12)
        self.assertEqual(int((S.region == 0).sum()), 6)
        self.assertEqual(int((S.region == 1).sum()), 6)


if __name__ == "__main__":
    unittest.main()
