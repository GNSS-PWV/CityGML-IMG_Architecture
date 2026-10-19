"""本地窗中心实验：一对一关联与无法排除高位候选时的保守判定。"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_experiment_window_centers import (choose_candidate, pair_from_initial,
                                           choose_dense_fallback, source_gate_with_current_policy)


class WindowCenterExperimentTests(unittest.TestCase):
    def test_four_window_pairs_reuse_only_approved_dense_match(self):
        source_result = {"status": "mapped", "building_id": "A", "wall_id": "wa",
                         "view_name": "wall_001_0",
                         "selected_candidate": "c1", "selection": {
                             "accepted": True, "reason": "automatic_checks_passed"}}
        candidate = {"candidate_id": "c1", "building_id": "A", "wall_id": "wa",
                     "view_name": "wall_001_0",
                     "selected_method": "minima_roma", "gate_passed": True,
                     "wall_vote": {"accepted": True},
                     "roma_validation": {"gate_passed": True},
                     "layout_validation": {"passed": True},
                     "door_layout_validation": {"passed": True},
                     "view": {"orbit_deg": 0},
                     "roma_homography_photo_to_render": np.eye(3).tolist()}
        center = [{"candidate_id": "c1", "pair_count": 4, "accepted": False}]
        selection = {"reason": "no_source_approved_candidate_passed_window_checks"}
        chosen, info = choose_dense_fallback(source_result, [candidate], center, selection)
        self.assertEqual(chosen["candidate_id"], "c1")
        self.assertEqual(info["window_pair_count"], 4)

        # 原结果若未自动确认，或实际靠窗结构而非稠密匹配通过，不允许回退。
        source_result["status"] = "needs_review"
        self.assertIsNone(choose_dense_fallback(source_result, [candidate], center, selection)[0])
        source_result["status"] = "mapped"
        candidate["selected_method"] = "minima_roma_window_structure"
        self.assertIsNone(choose_dense_fallback(source_result, [candidate], center, selection)[0])
        candidate["selected_method"] = "minima_roma"
        center[0]["pair_count"] = 6
        self.assertIsNone(choose_dense_fallback(source_result, [candidate], center, selection)[0])

    def test_association_uses_unique_window_ids_and_saved_transform(self):
        photo = [{"id": f"P{i}", "center": [float(i * 50), 20.]} for i in range(1, 4)]
        render = [{"id": f"R{i}", "center": [float(i * 50 + 7), 24.]} for i in range(1, 4)]
        answer = pair_from_initial(photo, render, np.array([[1., 0., 7.], [0., 1., 4.],
                                                              [0., 0., 1.]]))
        self.assertEqual([(x["photo"], x["render"]) for x in answer["pairs"]],
                         [("P1", "R1"), ("P2", "R2"), ("P3", "R3")])
        self.assertEqual(answer["same_wall"], "uncertain")  # 不足 6 对不能拟合并留出检查。

    def test_cached_dense_ratio_near_half_can_pass_without_overriding_other_checks(self):
        candidate = {"gate_passed": False, "selected_method": "minima_roma",
                     "roma_validation": {
                         "holdout_inlier_fraction": .497,
                         "checks": {"train_inliers_at_least_40": True,
                                    "holdout_inlier_fraction_at_least_0_50": False}},
                     "layout_validation": {"passed": True},
                     "door_layout_validation": {"passed": True}}
        self.assertTrue(source_gate_with_current_policy(candidate))
        candidate["layout_validation"]["passed"] = False
        self.assertFalse(source_gate_with_current_policy(candidate))

    def test_mild_layout_relaxation_requires_dense_and_wall_evidence(self):
        candidate = {"gate_passed": False, "selected_method": "minima_roma",
                     "roma_validation": {
                         "holdout_inlier_fraction": .56,
                         "checks": {"train_inliers_at_least_40": True,
                                    "holdout_inlier_fraction_at_least_0_50": True}},
                     "layout_validation": {"required": True, "passed": False},
                     "structure_validation": {"association_count": 16,
                                              "train_photo_coverage": {
                                                  "convex_hull_fraction": .279,
                                                  "x_span_fraction": .581,
                                                  "y_span_fraction": .689}},
                     "door_layout_validation": {"passed": True},
                     "wall_vote": {"accepted": True}}
        self.assertFalse(source_gate_with_current_policy(candidate))
        self.assertTrue(source_gate_with_current_policy(candidate, "mild"))
        candidate["wall_vote"]["accepted"] = False
        self.assertFalse(source_gate_with_current_policy(candidate, "mild"))
        candidate["wall_vote"]["accepted"] = True
        candidate["roma_validation"]["holdout_inlier_fraction"] = .33
        self.assertFalse(source_gate_with_current_policy(candidate, "mild"))

    def test_unresolved_higher_ranked_building_blocks_automatic_mapping(self):
        rows = [{"candidate_id": "a", "source_order": 1, "building_id": "A",
                 "wall_id": "wa", "accepted": False, "score": 0.,
                 "error": "ValueError: source_matcher_has_no_initial_transform"},
                {"candidate_id": "b", "source_order": 2, "building_id": "B",
                 "wall_id": "wb", "accepted": True, "source_gate_passed": True, "score": .9}]
        selected, info = choose_candidate(rows)
        self.assertIsNone(selected)
        self.assertEqual(info["reason"], "higher_ranked_building_unresolved")

    def test_clear_candidate_can_be_selected(self):
        rows = [{"candidate_id": "a", "source_order": 1, "building_id": "A",
                 "wall_id": "wa", "accepted": True, "source_gate_passed": True, "score": .8},
                {"candidate_id": "b", "source_order": 2, "building_id": "B",
                 "wall_id": "wb", "accepted": True, "source_gate_passed": True, "score": .3}]
        selected, info = choose_candidate(rows)
        self.assertEqual(selected["candidate_id"], "a")
        self.assertEqual(info["reason"], "window_checks_and_margin_passed")

    def test_window_centers_cannot_rescue_a_source_rejected_candidate(self):
        rows = [{"candidate_id": "a", "source_order": 1, "building_id": "A",
                 "wall_id": "wa", "accepted": True, "source_gate_passed": False, "score": .95}]
        selected, info = choose_candidate(rows)
        self.assertIsNone(selected)
        self.assertEqual(info["reason"], "no_source_approved_candidate_passed_window_checks")


if __name__ == "__main__":
    unittest.main()
