"""匹配几何测试：不联网、不下载模型、不使用真实标定点训练。"""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from facade_match_geometry import (backproject_render_points, fit_and_check,
                                   project_homography, roma_to_image_pixels)


class FacadeMatchGeometryTests(unittest.TestCase):
    def test_half_pixel_and_crop_offset_round_trip(self):
        source = np.array([[0., 0.], [2939., 832.], [120.25, 300.75]])
        target = np.array([[100., 200.], [699., 599.], [255.5, 420.25]])
        offset = np.array([100., 200.])
        normalized = np.column_stack(((source + .5) / [2940, 833] * 2 - 1,
                                       (target - offset + .5) / [600, 400] * 2 - 1))
        actual_source, actual_target = roma_to_image_pixels(normalized, [2940, 833], [600, 400], offset)
        np.testing.assert_allclose(actual_source, source, atol=1e-10)
        np.testing.assert_allclose(actual_target, target, atol=1e-10)

    def test_spatial_holdout_checks_unseen_matches_with_outliers(self):
        rng = np.random.default_rng(19)
        photo = rng.uniform([20., 20.], [980., 580.], size=(1200, 2))
        true_h = np.array([[.78, .012, 90.], [-.012, .7, 110.], [2e-5, -1e-5, 1.]])
        render = project_homography(photo, true_h) + rng.normal(0., .25, photo.shape)
        bad = rng.choice(len(photo), size=240, replace=False)
        render[bad] = rng.uniform([10., 10.], [990., 690.], size=(len(bad), 2))
        H, details, arrays = fit_and_check(photo, render, [1000, 600], [1000, 700])
        self.assertTrue(details["gate_passed"], details)
        self.assertFalse(details["independent_validation_passed"])
        self.assertFalse(np.any(arrays["train_mask"] & arrays["holdout_mask"]))
        self.assertTrue(np.all(arrays["photo_cells_xy"][arrays["holdout_mask"]].sum(axis=1) % 3 == 0))
        self.assertGreater(details["holdout_all_errors"]["p90_px"], 20.)
        self.assertLess(details["holdout_inlier_errors"]["p90_px"], 1.)
        corners = np.array([[50., 50.], [950., 550.], [500., 300.]])
        np.testing.assert_allclose(project_homography(corners, H), project_homography(corners, true_h), atol=.25)

        # 只打乱留出点，训练矩阵应逐元素相同，证明没有在检查后用全部点重拟合。
        changed = render.copy()
        changed[arrays["holdout_mask"]] += [45., 0.]
        H_changed, details_changed, _ = fit_and_check(photo, changed, [1000, 600], [1000, 700])
        np.testing.assert_allclose(H_changed, H, atol=1e-12)
        self.assertFalse(details_changed["gate_passed"])

    def test_small_cluster_cannot_pass_despite_many_perfect_matches(self):
        rng = np.random.default_rng(3)
        photo = rng.uniform([350., 250.], [650., 350.], size=(600, 2))
        render = photo * [.8, .8] + [80., 50.]
        _, details, _ = fit_and_check(photo, render, [1000, 600], [1000, 700])
        self.assertFalse(details["gate_passed"])
        self.assertIn("train_photo_hull_fraction_at_least_0_30", details["failed_checks"])

    def test_mirror_is_rejected_even_when_geometrically_exact(self):
        rng = np.random.default_rng(9)
        photo = rng.uniform([10., 10.], [990., 590.], size=(500, 2))
        render = photo.copy()
        render[:, 0] = 999 - render[:, 0]
        # 某些 OpenCV 的 USAC 实现会直接排除镜像采样；这也是合法的拒绝方式。
        try:
            _, details, _ = fit_and_check(photo, render, [1000, 600], [1000, 600])
        except ValueError as error:
            self.assertIn("有效单应矩阵", str(error))
        else:
            self.assertFalse(details["gate_passed"])
            self.assertIn("no_mirrored_or_collapsed_photo", details["failed_checks"])

    def test_degenerate_points_have_explanatory_error(self):
        xy = np.column_stack((np.arange(100.), np.arange(100.)))
        with self.assertRaisesRegex(ValueError, "直线|不足"):
            fit_and_check(xy, xy, [100, 100], [100, 100])

    def test_backprojection_uses_nearest_pixel_axis_depth_and_large_origin(self):
        # 非单位、非纯平移相机逆矩阵，验证不是把 depth 错当世界 Z。
        matrix = np.array([[.2, 0., -.8, 10.], [.3, 0., .6, -4.], [0., -.5, 0., 20.], [0., 0., 0., 1.]])
        camera = {"image_size_wh": [6, 5], "pixel_depth_to_local": matrix.tolist()}
        depth = np.full((5, 6), 12., dtype=np.float32)
        objects = np.full((5, 6), 7, dtype=int)
        depth[0, 0] = np.nan
        objects[1, 1] = -1
        objects[2, 2] = 9
        origin = np.array([691070., 5336100., 514.])
        points = np.array([[3.3, 3.6], [0., 0.], [1., 1.], [2., 2.], [-1., 3.], [6., 2.], [np.nan, 2.]])
        world, valid = backproject_render_points(points, camera, depth, objects, origin, [7])
        np.testing.assert_array_equal(valid, [True, False, False, False, False, False, False])
        expected = (matrix @ [3., 4., 12., 1.])[:3] + origin
        np.testing.assert_allclose(world[0], expected, atol=1e-9)
        self.assertTrue(np.isnan(world[1:]).all())


if __name__ == "__main__":
    unittest.main()
