"""Numerical regression tests for the pixel-coordinate CSV adapter (no weights)."""
import sys
import tempfile
import unittest
import json
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).absolute().parents[1]))
from yolo_backend import YoloBackend
from window_detector import tile_origins, merge_tiles


def backend(fmt='yolo11-onnx'):
    result = YoloBackend.__new__(YoloBackend)
    result.format = fmt
    result.size = 640
    result.names = ['door', 'window', 'other']
    result.ids = {'door': 0, 'window': 1}
    result.threshold = 0.25
    result.nms_threshold = 0.45
    return result


class YoloTests(unittest.TestCase):
    def test_letterbox_coordinate_inverse(self):
        b = backend()
        # 1000x333 image -> 640x213 with asymmetric vertical padding.
        sx, sy, top = 640 / 1000, 213 / 333, 213
        box = [100, 40, 550, 300]
        row = [(box[0] + box[2]) / 2 * sx, (box[1] + box[3]) / 2 * sy + top,
               (box[2] - box[0]) * sx, (box[3] - box[1]) * sy, 0, 0.9, 0]
        windows, doors = b.postprocess(np.array([row]), 1000, 333, 0, top, sx, sy)
        np.testing.assert_allclose(windows, [box])
        self.assertEqual(doors, [])

    def test_class_filter_and_class_aware_nms(self):
        rows = np.array([
            [100, 100, 80, 80, .1, .9, .1],
            [101, 101, 80, 80, .1, .8, .1],
            [100, 100, 80, 80, .95, .7, .1],
            [250, 250, 80, 80, .4, .5, .99],
            [400, 400, 80, 80, .1, .2, .1],
        ])
        w, d = backend().postprocess(rows, 640, 640, 0, 0, 1, 1)
        self.assertEqual(w, [[60, 60, 140, 140]])
        self.assertEqual(d, [[60, 60, 140, 140]])

    def test_clip_empty_invalid_and_padding(self):
        rows = np.array([
            [0, 0, 20, 20, 0, .9, 0],
            [-100, -100, 20, 20, 0, .9, 0],
            [100, 100, -1, 20, 0, .9, 0],
            [np.nan, 100, 20, 20, 0, .9, 0],
        ])
        b = backend()
        self.assertEqual(b.postprocess(rows, 640, 640, 0, 0, 1, 1), ([[0, 0, 10, 10]], []))
        self.assertEqual(b.postprocess(np.zeros((0, 7)), 640, 640, 0, 0, 1, 1), ([], []))

    def test_nms_precedes_clipping(self):
        # Both clip to the same box, but unclipped IoU=0.2 so both survive.
        rows = np.array([[0, 50, 200, 100, 0, .9, 0], [50, 50, 900, 100, 0, .8, 0]])
        w, _ = backend().postprocess(rows, 100, 100, 0, 0, 1, 1)
        self.assertEqual(len(w), 2)

    def test_darknet_scores_are_not_multiplied_twice(self):
        b = backend('darknet')
        row = np.array([[.5, .5, .2, .4, .5, .05, .4, .1]])
        result = b.decode([row])
        np.testing.assert_allclose(result[0], [320, 320, 128, 256, .05, .4, .1])
        self.assertEqual(len(b.postprocess(result, 640, 640, 0, 0, 1, 1)[0]), 1)

    def test_yolo11_output_layout(self):
        b = backend()
        self.assertEqual(b.decode([np.zeros((1, 7, 8400))]).shape, (8400, 7))
        with self.assertRaises(ValueError):
            b.decode([np.zeros((1, 8400, 7))])

    def test_tile_origins_cover_full_length(self):
        self.assertEqual(tile_origins(1000, 640, .25)[-1], 360)
        self.assertEqual(tile_origins(640, 640, .25), [0])

    def test_merge_tiles_prefers_non_edge_detection(self):
        detections = [
            {'bbox': [10, 10, 100, 100], 'confidence': .8, 'tile_id': 0, 'internal_edge': True},
            {'bbox': [12, 12, 102, 102], 'confidence': .7, 'tile_id': 1, 'internal_edge': False},
        ]
        merged = merge_tiles(detections)
        self.assertEqual(len(merged), 1)
        self.assertFalse(merged[0]['internal_edge'])

    def test_reject_incomplete_weight_before_opencv(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)
            (p / 'bad.weights').write_bytes(b'partial')
            (p / 'model.json').write_text(json.dumps({'weights': 'bad.weights', 'sha256': {'weights': '0' * 64}}))
            with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
                YoloBackend(p / 'model.json', .25)

    def test_predict_actual_letterbox_pixels(self):
        b = backend()
        class Net:
            def setInput(self, blob):
                self.blob = blob
            def forward(self, names):
                return [np.array([[[320], [320], [320], [160], [.1], [.9], [.1]]])]
        b.net = Net()
        b.output_names = ['output0']
        image = np.full((200, 400, 3), [10, 20, 30], dtype=np.uint8)
        windows, doors = b.predict(image)
        np.testing.assert_allclose(windows, [[100, 50, 300, 150]])
        np.testing.assert_allclose(b.net.blob[0, :, 0, 0], [114 / 255] * 3)
        np.testing.assert_allclose(b.net.blob[0, :, 200, 0], [30 / 255, 20 / 255, 10 / 255])


if __name__ == '__main__':
    unittest.main(verbosity=2)



