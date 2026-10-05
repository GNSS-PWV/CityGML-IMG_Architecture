"""自动入口不得依赖人工标定，也不能因为细化失败而替换有效初值。"""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import match_facade_roma as matching


class AutomaticWorkflowTests(unittest.TestCase):
    def test_rejected_structure_keeps_original_transform_and_quality(self):
        original = np.eye(3)
        wrong = original.copy()
        wrong[0, 2] = 100
        report = {"gate_passed": False, "failed_checks": ["holdout"]}
        selected, method, quality = matching.choose_automatic_transform(
            original, report, wrong, {"gate_passed": False})
        np.testing.assert_array_equal(selected, original)
        self.assertEqual(method, "roma")
        self.assertFalse(quality["gate_passed"])

    def test_matching_detection_requires_exact_photo_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "detections.json"
            path.write_text(json.dumps({"image_sha256": "other-image", "image_size": [100, 80],
                                       "coordinate_system": "original_image_pixels_xyxy"}), encoding="utf-8")
            with patch.object(matching, "DETECTION_JSON", path):
                with self.assertRaisesRegex(ValueError, "指纹"):
                    matching.load_detection_result({"photo_sha256": "this-image"}, (100, 80))

    def test_main_succeeds_without_any_manual_reference_read(self):
        # 合成无错位的平面匹配；只验证无人标定的完整控制流，输出仅写临时目录。
        wh = (100, 80)
        points = np.array([(x, y) for y in np.linspace(4, 75, 10) for x in np.linspace(4, 95, 15)])
        normalized_xy = (points + .5) / wh * 2 - 1
        normalized = np.column_stack((normalized_xy, normalized_xy))
        picture = Image.new("RGB", wh)
        camera = {"image_size_wh": wh, "pixel_depth_to_local": np.eye(4).tolist()}
        depth = np.full((80, 100), 10., dtype=np.float32)
        objects = np.zeros((80, 100), dtype=np.int32)
        identity = {"crop_xyxy": [0, 0, 100, 80], "photo_sha256": "synthetic"}
        prepared = (picture, picture, picture, camera, depth, objects, [0],
                    {"world_origin_m": [0, 0, 0]}, Path("synthetic-render"), identity)
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(matching, "OUTPUT_ROOT", Path(folder)), \
             patch.object(matching, "USE_WINDOW_STRUCTURE", False), \
             patch.object(matching, "prepare_pair", return_value=prepared), \
             patch.object(matching, "run_model", return_value=(normalized, np.ones(len(points)), {})), \
             patch.object(matching, "check_against_manual", side_effect=AssertionError("不允许读取人工点")) as manual, \
             patch.object(matching, "draw_matches"), patch.object(matching, "draw_alignment"):
            matching.main()
            manual.assert_not_called()
            run_dir = Path(json.loads((Path(folder) / "latest_run.json").read_text())["run_dir"])
            result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "auto_checks_passed")
            self.assertFalse(result["manual_calibration_used"])
            self.assertFalse(result["independent_accuracy_verified"])
            self.assertNotIn("independent_manual_validation", result)
            with np.load(run_dir / "verified_correspondences.npz") as data:
                np.testing.assert_allclose(data["homography"], np.eye(3), atol=1e-5)
                self.assertTrue(data["surface_valid"].all())


if __name__ == "__main__":
    unittest.main()
