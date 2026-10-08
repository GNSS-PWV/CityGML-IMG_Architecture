"""结构细化的行为测试：已知几何、错误留出点、局部覆盖、过大修正。"""
from pathlib import Path
import json
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from facade_match_geometry import project_homography
from facade_match_structure import assess_door_layout, assess_window_layout, refine_with_window_structure


def scene(photo_centers, target_centers=None):
    """构造只有可见构件编号的合成立面，不依赖项目真实图像或人工点。"""
    photo_centers = np.asarray(photo_centers, dtype=float)
    target_centers = photo_centers if target_centers is None else np.asarray(target_centers)
    object_map = np.full((600, 1000), -1, dtype=int)
    manifest = {"front_reference_wall_id": "front", "objects": []}
    detector = {"detections": []}
    for i, (photo, target) in enumerate(zip(photo_centers, target_centers)):
        box = np.r_[photo-[10, 15], photo+[10, 15]]
        detector["detections"].append({"class": "window", "score": .7, "box": box.tolist()})
        x, y = np.rint(target).astype(int)
        object_map[y-15:y+16, x-10:x+11] = i
        manifest["objects"].append({"index": i, "kind": "Window", "wall_id": "front", "id": str(i)})
    return detector, object_map, manifest


def grid(xs=None, ys=None):
    xx, yy = np.meshgrid(np.linspace(60, 940, 9) if xs is None else xs,
                         np.linspace(50, 550, 5) if ys is None else ys)
    return np.column_stack((xx.ravel(), yy.ravel()))


def refine(initial, data):
    return refine_with_window_structure(initial, [1000, 600], [1000, 600], *data)


class FacadeMatchStructureTests(unittest.TestCase):
    def test_door_layout_rejects_displaced_door_pattern_when_evidence_is_sufficient(self):
        object_map = np.zeros((100, 200), dtype=int)
        for index, x in enumerate((20, 80, 140), 1):
            object_map[40:61, x:x+11] = index
        manifest = {"objects": [{"kind": "Door", "wall_id": "w", "index": index}
                                for index in (1, 2, 3)]}
        detections = {"detections": [{"class": "door", "score": .8,
                                       "box": [x, 40, x+10, 60]} for x in (20, 80, 140)]}
        good = assess_door_layout(np.eye(3), (200, 100), (200, 100),
                                  detections, object_map, manifest, "w")
        bad = assess_door_layout(np.array([[1., 0., 35.], [0., 1., 0.], [0., 0., 1.]]),
                                 (200, 100), (200, 100), detections, object_map, manifest, "w")
        self.assertTrue(good["required"])
        self.assertTrue(good["passed"])
        self.assertFalse(bad["passed"])

    def test_door_layout_marks_sparse_doors_as_unchecked(self):
        result = assess_door_layout(np.eye(3), (100, 100), (100, 100),
                                    {"detections": [{"class": "door", "score": .8,
                                                     "box": [20, 20, 40, 60]}]},
                                    np.zeros((100, 100), dtype=int), {"objects": []}, "w")
        self.assertFalse(result["required"])

    def test_layout_gate_rejects_many_windows_that_cover_too_little_width(self):
        report = {"eligible_photo_windows": 19, "visible_model_windows": 19, "association_count": 17,
                  "train_photo_coverage": {"convex_hull_fraction": .33, "x_span_fraction": .61, "y_span_fraction": .65}}
        layout = assess_window_layout(report)
        self.assertTrue(layout["required"])
        self.assertFalse(layout["passed"])
        self.assertIn("window_layout_photo_x_span_at_least_0_65", layout["failed_checks"])

    def test_layout_gate_is_not_required_for_a_small_window_set(self):
        report = {"eligible_photo_windows": 4, "visible_model_windows": 20, "association_count": 4,
                  "train_photo_coverage": {"convex_hull_fraction": .1, "x_span_fraction": .2, "y_span_fraction": .2}}
        layout = assess_window_layout(report)
        self.assertFalse(layout["required"])
        self.assertTrue(layout["passed"])

    def test_good_coarse_alignment_is_refined_without_manual_inputs(self):
        initial = np.array([[1., 0., 4.], [0., 1., -3.], [0., 0., 1.]])
        H, report, arrays = refine(initial, scene(grid()))
        self.assertTrue(report["gate_passed"], report)
        self.assertFalse(report["manual_points_used"])
        self.assertFalse(report["independent_validation_passed"])
        np.testing.assert_allclose(project_homography([[0, 0], [999, 599]], H), [[0, 0], [999, 599]], atol=1e-6)
        self.assertFalse(np.any(arrays["train_mask"] & arrays["holdout_mask"]))
        self.assertEqual(len(np.unique(arrays["model_object_indices"])), len(arrays["model_object_indices"]))
        json.dumps(report, allow_nan=False)

    def test_corrupting_only_holdout_does_not_refit_training_transform(self):
        centers = grid()
        H, _, arrays = refine(np.eye(3), scene(centers))
        changed_targets = centers.copy()
        # 使用第一次返回的留出身份，修改这些未用于拟合的模型窗位置。
        heldout_indices = arrays["detection_indices"][arrays["holdout_mask"]]
        changed_targets[heldout_indices] += [0., 12.]
        changed_H, report, _ = refine(np.eye(3), scene(centers, changed_targets))
        np.testing.assert_allclose(changed_H, H, atol=1e-10)
        self.assertFalse(report["gate_passed"])
        self.assertIn("holdout_inlier_fraction_at_least_0_70", report["failed_checks"])
        self.assertGreater(report["holdout_all_errors"]["median_px"], 10.)

    def test_many_perfect_windows_in_small_patch_cannot_pass(self):
        centers = grid(np.linspace(300, 620, 9), np.linspace(200, 400, 5))
        _, report, _ = refine(np.eye(3), scene(centers))
        self.assertFalse(report["gate_passed"])
        self.assertIn("train_photo_hull_fraction_at_least_0_30", report["failed_checks"])
        self.assertIn("train_photo_x_span_at_least_0_65", report["failed_checks"])

    def test_refinement_close_to_half_window_spacing_is_rejected(self):
        # 40像素窗间距，粗对齐偏19像素仍关联到原窗；但修正接近半个窗间距，
        # 容易产生窗列歧义，不能因为最终训练/留出残差为0就直接接受。
        centers = grid(np.arange(60, 941, 40))
        initial = np.eye(3)
        initial[0, 2] = 19.
        _, report, _ = refine(initial, scene(centers))
        self.assertFalse(report["gate_passed"])
        self.assertIn("refinement_stays_near_roma_initial", report["failed_checks"])
        self.assertGreater(report["initial_change"]["max_initial_change_px"], 18.)

    def test_full_column_error_in_initial_can_remain_ambiguous(self):
        # 明确记录方法边界：周期立面错整列可同时通过自动一致性；不能把
        # gate_passed 宣传成独立准确率，也不能据此设置“真值验证通过”。
        centers = grid(np.arange(60, 941, 40))
        initial = np.eye(3)
        initial[0, 2] = 40.
        H, report, _ = refine(initial, scene(centers))
        self.assertIsNotNone(H)
        self.assertFalse(report["independent_validation_passed"])
        self.assertIn("重复窗列错位", report["interpretation"])
        if report["gate_passed"]:
            self.assertGreater(abs(H[0, 2]), 35.)

    def test_clipped_low_score_and_ambiguous_detections_are_not_anchors(self):
        data = scene(grid())
        extra = [{"class": "window", "score": .8, "box": [0, 40, 20, 70]},
                 {"class": "window", "score": .1, "box": [200, 100, 220, 130]},
                 {"class": "unresolved", "score": .9, "box": [230, 100, 250, 130]}]
        data[0]["detections"].extend(extra)
        _, report, arrays = refine(np.eye(3), data)
        self.assertTrue(report["gate_passed"], report)
        self.assertEqual(report["rejected_detections"]["clipped_or_near_photo_border"], 1)
        self.assertEqual(report["rejected_detections"]["not_window_or_low_score"], 2)
        self.assertTrue(np.all(arrays["detection_indices"] < len(grid())))

    def test_missing_model_windows_returns_explanatory_failure(self):
        data = scene(grid())
        data[2]["objects"] = []
        H, report, arrays = refine(np.eye(3), data)
        self.assertIsNone(H)
        self.assertFalse(report["gate_passed"])
        self.assertIn("both_images_have_eligible_windows", report["failed_checks"])
        self.assertEqual(arrays["photo_xy"].shape, (0, 2))
        json.dumps(report, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
