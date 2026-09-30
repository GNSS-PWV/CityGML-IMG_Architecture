"""几何回归：合成对应关系，不运行模型，不生成混入真实结果的模拟文件。"""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import map_facade_to_3d as mapping


class FacadeMappingTests(unittest.TestCase):
    def test_known_perspective_mapping_and_independent_check(self):
        expected = np.array([[.025, .001, -3], [.0008, -.022, 18], [.00002, .0001, 1.]])
        control = np.array([[0, 0], [2940, 0], [2940, 833], [0, 833], [1000, 600], [2000, 300]])
        targets = mapping.apply_homography(control, expected)
        matrix = mapping.fit_homography(control, targets)
        checks = np.array([[700, 350], [2300, 500]])
        np.testing.assert_allclose(mapping.apply_homography(checks, matrix),
                                   mapping.apply_homography(checks, expected), atol=1e-9)
        quality = mapping.calibration_quality(np.stack((control, targets), axis=1),
                  np.stack((checks, mapping.apply_homography(checks, expected)), axis=1), matrix)
        self.assertLess(quality["check_max_error_m"], 1e-9)

    def test_collinear_points_are_rejected(self):
        points = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
        with self.assertRaises(ValueError):
            mapping.fit_homography(points, points)

    def test_bad_independent_check_does_not_pass_exact_four_point_fit(self):
        control = np.array([[0, 0], [100, 0], [100, 100], [0, 100]])
        targets = control * .1
        matrix = mapping.fit_homography(control, targets)
        with self.assertRaises(ValueError):
            mapping.calibration_quality(np.stack((control, targets), axis=1),
                                        [[[50, 50], [7, 5]]], matrix)

    def test_check_point_cannot_reuse_a_control_point(self):
        control = np.array([[0, 0], [100, 0], [100, 100], [0, 100]])
        pairs = np.stack((control, control*.1), axis=1)
        with self.assertRaises(ValueError):
            mapping.calibration_quality(pairs, [pairs[0]], np.diag([.1, .1, 1]))

    def test_world_plane_roundtrip_with_large_geo_coordinates(self):
        origin = np.array([691064., 5336062., 514.59])
        u = np.array([.37, .93, 0.]); u /= np.linalg.norm(u)
        rings = [origin + np.array([[0, 0], [66, 0], [66, 18], [0, 18], [0, 0]]) @ np.stack((u, [0, 0, 1.]))]
        frame = mapping.plane_frame(rings)
        np.testing.assert_allclose(mapping.to_xyz(mapping.to_uv(rings[0], frame), frame), rings[0], atol=1e-8)
        self.assertLess(frame["max_plane_deviation_m"], 1e-8)

    def test_nonplanar_wall_is_rejected(self):
        with self.assertRaises(ValueError):
            mapping.plane_frame([np.array([[0, 0, 0], [10, 0, 0], [10, 0, 10], [0, 1, 10]])])

    def test_prediction_mapping_preserves_ambiguity_and_rejects_outside(self):
        wall = {"id": "test_wall", "uv": [np.array([[0, 0], [10, 0], [10, 10], [0, 10]])],
                "frame": {"origin": np.array([691000., 5336000., 500.]),
                          "u": np.array([1., 0, 0]), "v": np.array([0, 0, 1.])}}
        predictions = [{"class": "window", "box": [10, 10, 20, 20], "score": .5},
                       {"class": "ambiguous", "box": [20, 20, 30, 30], "score": .6,
                        "candidates": [{"class": "door"}, {"class": "window"}]},
                       {"class": "window", "box": [90, 90, 110, 110], "score": .4}]
        matrix = np.diag([.1, .1, 1])
        mapped, rejected = mapping.map_detections(predictions, matrix, wall,
                                                  [[0, 0], [100, 0], [100, 100], [0, 100]])
        self.assertEqual(len(mapped), 2)
        self.assertEqual(mapped[1]["class"], "ambiguous")
        self.assertEqual(len(mapped[1]["detection"]["candidates"]), 2)
        np.testing.assert_allclose(mapped[0]["vertices_xyz"][0], [691001., 5336000., 501.])
        self.assertEqual(len(rejected), 1)
        self.assertIn("region", rejected[0]["reason"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "preview.obj"
            mapping.export_obj(path, mapped)
            text = path.read_text()
            self.assertEqual(sum(line.startswith("v ") for line in text.splitlines()), 8)
            self.assertIn("usemtl ambiguous", text)
            self.assertIn("f 5 6 7 8", text)

    def test_region_closure_boundary_and_crossing(self):
        region = mapping.validate_region([[0, 0], [100, 0], [100, 100], [0, 100], [0, 0]], [100, 100])
        self.assertEqual(len(region), 4)
        self.assertTrue(mapping.inside_polygon([[0, 0], [50, 50], [100, 100]], region).all())
        with self.assertRaises(ValueError):
            mapping.validate_region([[0, 0], [100, 100], [0, 100], [100, 0]], [100, 100])
        # 四角全在墙内，但框的一条边跨过凹缺口，仍必须拒绝。
        notched_wall = [[0, 0], [10, 0], [10, 10], [6, 10], [6, 3], [4, 3], [4, 10], [0, 10]]
        self.assertFalse(mapping.polygon_inside_ring([[2, 2], [8, 2], [8, 8], [2, 8]], notched_wall))
        boundary_notch = [[0, 0], [10, 0], [10, 10], [6, 10], [6, 4], [4, 4], [4, 10], [0, 10]]
        self.assertFalse(mapping.polygon_inside_ring([[4, 4], [6, 4], [6, 8], [4, 8]], boundary_notch, .03))
        # 中心在墙内，但边与凹缺口的角相切后经过墙外，交点区间仍应检出。
        self.assertFalse(mapping.polygon_inside_ring([[3, 1], [7, 1], [6, 5], [4, 5]], boundary_notch, .03))
        self.assertTrue(mapping.polygon_inside_ring([[1, 1], [3, 1], [3, 9], [1, 9]], boundary_notch, .03))

    def test_interactive_click_flow_saves_controls_checks_and_region(self):
        # 用 Matplotlib 自身的鼠标事件走完整个回调链，验证按钮/点选流程；不会打开窗口。
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backend_bases import MouseEvent
        from PIL import Image
        wall = {"id": "test_wall", "uv": [np.array([[0, 0], [10, 0], [10, 10], [0, 10]])],
                "size": np.array([10., 10.]), "references": []}

        def interact(**_):
            figure = plt.gcf()

            def click(axis, point):
                figure.canvas.draw()
                x, y = axis.transData.transform(point)
                event = MouseEvent("button_press_event", figure.canvas, x, y, button=1)
                figure.canvas.callbacks.process("button_press_event", event)
                release = MouseEvent("button_release_event", figure.canvas, x, y, button=1)
                figure.canvas.callbacks.process("button_release_event", release)

            for point in ([5, 5], [95, 5], [95, 95], [5, 95]):
                click(figure.axes[0], point)
                click(figure.axes[1], np.asarray(point)*.1)
            click(figure.axes[3], [.5, .5])  # Check pair 按钮。
            figure.axes[0].set_xlim(0, 60)
            figure.axes[0].set_ylim(60, 0)
            figure.axes[1].set_xlim(0, 6)
            figure.axes[1].set_ylim(0, 6)
            click(figure.axes[0], [50, 50])
            np.testing.assert_allclose(figure.axes[0].get_xlim(), [0, 60])
            np.testing.assert_allclose(figure.axes[0].get_ylim(), [60, 0])
            click(figure.axes[1], [5, 5])
            np.testing.assert_allclose(figure.axes[1].get_xlim(), [0, 6])
            np.testing.assert_allclose(figure.axes[1].get_ylim(), [0, 6])
            figure.axes[0].set_xlim(-.5, 99.5)
            figure.axes[0].set_ylim(99.5, -.5)
            click(figure.axes[4], [.5, .5])  # Wall region 按钮。
            for point in ([1, 1], [98, 1], [98, 98], [1, 98]):
                click(figure.axes[0], point)
            click(figure.axes[6], [.5, .5])  # Save + map 按钮。

        with patch.object(plt, "show", side_effect=interact):
            result = mapping.interactive_calibration(Image.new("RGB", (100, 100)), wall)
        self.assertEqual(len(result["control_pairs"]), 4)
        self.assertEqual(len(result["check_pairs"]), 1)
        self.assertEqual(len(result["image_wall_region"]), 4)
        self.assertLess(result["quality"]["check_max_error_m"], .1)


if __name__ == "__main__":
    unittest.main()
