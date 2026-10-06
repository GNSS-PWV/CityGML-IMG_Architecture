"""粗H引导局部匹配的CPU模拟测试；不加载模型、不联网、不用GPU。"""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import match_facade_roma as roma


def normalized(points, size):
    return (np.asarray(points, dtype=float)+.5)*2/np.asarray(size)-1


class TiledMatchingTests(unittest.TestCase):
    def test_horizontal_offsets_wall_filter_and_duplicate_pixel_choose_highest_overlap(self):
        photo, render = Image.new("RGB", (2000, 1000)), Image.new("RGB", (2200, 1200))
        H = np.array([[1., 0, 100], [0, 1, 80], [0, 0, 1]])
        allowed = np.ones((1200, 2200), dtype=bool)
        allowed[780, 200] = False
        source = [np.array([[900.25, 300.25], [100.25, 200.25], [600, 500],
                            [100, 700], [-1, 100], [np.nan, 200]]),
                  np.array([[900.35, 300.30], [1900.25, 600.25]])]
        confidence = [np.array([.5, .8, .1, .8, .8, .8]), np.array([.9, .7])]
        calls = []

        def inference(model, a, b):
            index = len(calls)
            calls.append((a.size, b.size))
            photo_offset = np.array([[0, 0], [800, 0]][index])
            render_offset = np.array([[0, 0], [780, 0]][index])
            target = source[index]+[100, 80]
            # 高可信度重复点故意偏离粗H：函数不应凭靠近初值筛掉真正待验证的误差。
            if index == 1:
                target[0] += [20, 0]
            matches = np.column_stack((normalized(source[index]-photo_offset, a.size),
                                       normalized(target-render_offset, b.size)))
            return matches, confidence[index]

        with patch.object(roma, "infer_pair", side_effect=inference):
            a, b, scores, report = roma.tiled_correspondences(object(), photo, render, H, allowed)
        self.assertEqual(calls, [((1200, 1000), (1420, 1180)), ((1200, 1000), (1420, 1180))])
        np.testing.assert_allclose(a, [[100.25, 200.25], [900.35, 300.3], [1900.25, 600.25]], atol=1e-9)
        np.testing.assert_allclose(b, [[200.25, 280.25], [1020.35, 380.3], [2000.25, 680.25]], atol=1e-9)
        np.testing.assert_allclose(scores, [.8, .9, .7])
        self.assertEqual(report["matches_before_deduplication"], 4)
        self.assertEqual(report["duplicate_source_pixels_removed"], 1)
        self.assertEqual(report["tiles"][1]["render_crop_xyxy"], [780, 0, 2200, 1180])
        self.assertEqual(report["tiles"][0]["kept_after_deduplication"], 1)
        self.assertEqual(report["status"], "complete")
        self.assertNotIn("gate_passed", report)
        self.assertFalse(report["independent_accuracy_verified"])

    def test_vertical_tile_offset_and_halfpixel_coordinates_with_scaled_initial_H(self):
        photo, render = Image.new("RGB", (800, 1700)), Image.new("RGB", (500, 1000))
        H = np.array([[.5, 0, 50], [0, .5, 60], [0, 0, 1]])
        calls = []

        def inference(model, a, b):
            index = len(calls)
            calls.append((a.size, b.size))
            full_a = np.array([[200., 200.+700*index]])
            full_b = full_a*.5+[50, 60]
            return np.column_stack((normalized(full_a-[0, 700*index], a.size),
                                    normalized(full_b-[10, 10+350*index], b.size))), np.array([.8])

        with patch.object(roma, "infer_pair", side_effect=inference):
            a, b, _, report = roma.tiled_correspondences(object(), photo, render, H, np.ones((1000, 500), bool))
        np.testing.assert_allclose(a, [[200, 200], [200, 900]], atol=1e-9)
        np.testing.assert_allclose(b, [[150, 160], [150, 510]], atol=1e-9)
        self.assertEqual(report["tiles"][1]["photo_crop_xyxy"], [0, 700, 800, 1700])
        self.assertEqual(report["tiles"][1]["render_crop_xyxy"], [10, 360, 491, 961])

    def test_local_failure_recorded_and_other_tile_preserved(self):
        photo, render = Image.new("RGB", (2000, 1000)), Image.new("RGB", (2000, 1000))
        calls = []

        def inference(model, a, b):
            calls.append(a.size)
            if len(calls) == 1:
                raise RuntimeError("synthetic local inference failure")
            return np.array([[0., 0., 0., 0.]]), np.array([.8])

        with patch.object(roma, "infer_pair", side_effect=inference):
            a, b, scores, report = roma.tiled_correspondences(object(), photo, render, np.eye(3),
                                                            np.ones((1000, 2000), bool))
        self.assertEqual(a.shape, (1, 2))
        self.assertEqual(b.shape, (1, 2))
        self.assertEqual(scores.shape, (1,))
        self.assertEqual(report["failed_tile_count"], 1)
        self.assertEqual(report["completed_tile_count"], 1)
        self.assertEqual(report["status"], "partial")
        self.assertEqual(report["tiles"][0]["error_type"], "RuntimeError")

    def test_poles_tiny_crops_and_empty_wall_are_not_sent_to_model(self):
        photo, render = Image.new("RGB", (2000, 1000)), Image.new("RGB", (2000, 1000))
        pole = np.array([[1., 0, 0], [0, 1, 0], [.001, 0, -.8]])
        tiny = np.diag([.001, .001, 1.])
        with patch.object(roma, "infer_pair") as inference:
            for H, mask in [(pole, np.ones((1000, 2000), bool)),
                            (tiny, np.ones((1000, 2000), bool)),
                            (np.eye(3), np.zeros((1000, 2000), bool))]:
                a, b, scores, report = roma.tiled_correspondences(object(), photo, render, H, mask)
                self.assertEqual(a.shape, (0, 2))
                self.assertEqual(b.shape, (0, 2))
                self.assertEqual(scores.shape, (0,))
                self.assertEqual(report["status"], "no_matches")
                self.assertEqual(report["skipped_tile_count"], 2)
            inference.assert_not_called()

    def test_invalid_inputs_rejected_before_inference(self):
        photo = render = Image.new("RGB", (100, 100))
        with patch.object(roma, "infer_pair") as inference:
            with self.assertRaisesRegex(ValueError, "非奇异"):
                roma.tiled_correspondences(None, photo, render, np.zeros((3, 3)), np.ones((100, 100), bool))
            with self.assertRaisesRegex(ValueError, "布尔"):
                roma.tiled_correspondences(None, photo, render, np.eye(3), np.ones((100, 100), int))
            with self.assertRaisesRegex(ValueError, "同尺寸"):
                roma.tiled_correspondences(None, photo, render, np.eye(3), np.ones((99, 100), bool))
            inference.assert_not_called()


if __name__ == "__main__":
    unittest.main()
