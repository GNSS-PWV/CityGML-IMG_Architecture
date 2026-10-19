"""编号对应的几何门槛：已知仿射通过，留出错配不能被训练掩盖。"""
from pathlib import Path
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vlm_window_alignment import call_openai, fit_window_pairs


def grid_scene():
    xs, ys = np.meshgrid(np.linspace(60, 940, 4), np.linspace(60, 540, 4))
    photo = np.column_stack((xs.ravel(), ys.ravel()))
    render = photo @ np.array([[.72, .03], [.02, .78]]) + [70, 35]
    p = [{"id": f"P{i:02d}", "center": point.tolist()} for i, point in enumerate(photo, 1)]
    r = [{"id": f"R{i:02d}", "center": point.tolist()} for i, point in enumerate(render, 1)]
    answer = {"same_wall": "yes", "reason": "synthetic", "pairs": [
        {"photo": row["id"], "render": r[i]["id"]} for i, row in enumerate(p)]}
    return answer, p, r


class VlmWindowAlignmentTests(unittest.TestCase):
    def test_six_windows_use_affine_with_two_held_out(self):
        photo_xy = np.array([[90., 128.], [318., 128.], [539., 129.],
                             [92., 387.], [316., 388.], [539., 389.]])
        render_xy = photo_xy @ np.array([[.65, 0.], [0., .64]]) + [241., 25.]
        photo = [{"id": f"P{i}", "center": p.tolist()} for i, p in enumerate(photo_xy)]
        render = [{"id": f"R{i}", "center": p.tolist()} for i, p in enumerate(render_xy)]
        answer = {"same_wall": "yes", "pairs": [
            {"photo": f"P{i}", "render": f"R{i}"} for i in range(6)]}
        H, report = fit_window_pairs(answer, photo, render, [647, 833], [900, 600])
        self.assertTrue(report["accepted"], report)
        self.assertEqual(report["holdout_count"], 2)
        self.assertIn("affine", report["method"])
        np.testing.assert_allclose(H[2], [0., 0., 1.])

        # P4 是留出点；把它错误配到另一列不能因训练点够少而蒙混过关。
        render[4]["center"][0] += 90.
        _, rejected = fit_window_pairs(answer, photo, render, [647, 833], [900, 600])
        self.assertFalse(rejected["accepted"])
        self.assertFalse(rejected["checks"]["all_holdout_errors_within_limit"])

    def test_spread_affine_correspondences_pass(self):
        answer, photo, render = grid_scene()
        H, report = fit_window_pairs(answer, photo, render, [1000, 600], [900, 600])
        self.assertTrue(report["accepted"], report)
        self.assertLess(max(report["holdout_errors_px"]), 1e-3)
        self.assertEqual(H.shape, (3, 3))

    def test_perspective_window_centers_are_not_forced_into_an_affine_model(self):
        answer, photo, _ = grid_scene()
        source = np.asarray([row["center"] for row in photo], dtype=float)
        perspective = np.array([[.74, .02, 65.], [.01, .81, 31.],
                                [.00012, -.00008, 1.]])
        target = np.column_stack((source, np.ones(len(source)))) @ perspective.T
        target = target[:, :2] / target[:, 2:]
        render = [{"id": f"R{i:02d}", "center": point.tolist()} for i, point in enumerate(target, 1)]
        H, report = fit_window_pairs(answer, photo, render, [1000, 600], [900, 600])
        self.assertTrue(report["accepted"], report)
        np.testing.assert_allclose(H / H[2, 2], perspective / perspective[2, 2], atol=1e-5)

    def test_wrong_heldout_window_fails(self):
        answer, photo, render = grid_scene()
        # x 排序的第3个点 P09 被留作检查；把它配到错位的渲染窗。
        render[8]["center"][0] += 90
        _, report = fit_window_pairs(answer, photo, render, [1000, 600], [900, 600])
        self.assertFalse(report["accepted"])
        self.assertFalse(report["checks"]["all_holdout_errors_within_limit"])

    def test_uncertain_wall_never_maps(self):
        answer, photo, render = grid_scene()
        answer["same_wall"] = "uncertain"
        H, report = fit_window_pairs(answer, photo, render, [1000, 600], [900, 600])
        self.assertIsNone(H)
        self.assertFalse(report["accepted"])

    def test_paid_api_response_is_cached_separately_from_manual_pairs(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            image = output / "numbered.png"
            Image.new("RGB", (16, 16), "white").save(image)
            (output / "input_identity.json").write_text('{"test":"same-input"}', encoding="utf-8")
            (output / "pairs.json").write_text('{"same_wall":"uncertain","pairs":[]}', encoding="utf-8")
            expected = {"same_wall": "no", "reason": "different", "pairs": []}
            response = Mock()
            response.json.return_value = {"id": "resp_test", "status": "completed", "output": [{"type": "message",
                "content": [{"type": "output_text", "text": json.dumps(expected)}]}]}
            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-secret"}), \
                 patch("requests.post", return_value=response) as post:
                self.assertEqual(call_openai(image, image, output), expected)
                self.assertEqual(call_openai(image, image, output), expected)
                self.assertEqual(post.call_count, 1)
            self.assertEqual(json.loads((output / "pairs.json").read_text())["same_wall"], "uncertain")
            self.assertNotIn("test-secret", (output / "api_request_started.json").read_text())


if __name__ == "__main__":
    unittest.main()
