"""人工参考只做检查：用内存合成数据测试，不向真实结果目录写模拟匹配。"""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import match_facade_roma as matching


class ManualValidationTests(unittest.TestCase):
    def setUp(self):
        self.points = np.array([[1., 1.], [8., 1.], [8., 6.], [1., 6.], [4., 3.]])
        self.manual = {"identity": {"image_sha256": "photo", "citygml_sha256": "gml"},
                       "wall_id": "wall", "control_pairs": [[p.tolist(), p.tolist()] for p in self.points[:4]],
                       "check_pairs": [[self.points[4].tolist(), self.points[4].tolist()]]}
        self.origin = np.array([691000., 5336000., 500.])
        self.frame = {"origin": self.origin, "u": np.array([1., 0, 0]), "v": np.array([0., 1, 0])}
        self.manifest = {"source_sha256": "gml", "front_reference_wall_id": "wall", "source_gml": "unused.gml",
                         "world_origin_m": self.origin.tolist()}
        self.camera = {"local_to_pixel_depth": np.diag([10., 10., 1., 1.]).tolist(), "pixels_per_meter": 10.}
        self.H = np.diag([10., 10., 1.])

    def evaluate(self, H):
        with patch.object(matching, "MANUAL_CHECK_PATH", Path(__file__)), \
             patch.object(matching, "read_json", return_value=self.manual), \
             patch("map_facade_to_3d.load_walls", return_value=[{"id": "wall", "frame": self.frame}]):
            return matching.check_against_manual(H, self.manifest, self.camera, {"photo_sha256": "photo"})

    def test_all_reference_points_checked_without_mutating_homography(self):
        before = self.H.copy()
        result = self.evaluate(self.H)
        self.assertTrue(result["passed"])
        self.assertEqual(result["count"], 5)
        self.assertFalse(result["used_for_fitting"])
        np.testing.assert_array_equal(self.H, before)

    def test_consistent_one_window_shift_fails_reference_check(self):
        shifted = self.H.copy()
        shifted[0, 2] = 25.  # 变换本身完全自洽，但整体错移2.5m，应拒绝。
        result = self.evaluate(shifted)
        self.assertFalse(result["passed"])
        self.assertAlmostEqual(result["max_error_m"], 2.5)

    def test_identity_mismatch_cannot_be_used_as_validation(self):
        self.manual["identity"]["citygml_sha256"] = "other-model"
        result = self.evaluate(self.H)
        self.assertFalse(result["available"])
        self.assertFalse(result["passed"])

    def test_nonfinite_check_projection_is_explicit_failure_and_serializable(self):
        bad = self.H.copy()
        bad[2] = [1., 0., -1.]  # x=1的检查点投影到无穷远。
        result = self.evaluate(bad)
        self.assertFalse(result["passed"])
        self.assertGreater(result["nonfinite_count"], 0)
        import json
        json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
