"""Scored window-only adapters and overlap tiling. No storey/height/3D logic."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from yolo_backend import YoloBackend

ROOT = Path(__file__).absolute().parent
MANIFESTS = {'yolov3': 'yolov3-openimages.json', 'yolo11': 'yolo11-door-window.json'}


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def overlap(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    aa = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    ab = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return intersection / max(aa + ab - intersection, 1e-12), intersection / max(min(aa, ab), 1e-12)


def merge_tiles(detections, iou=.5, edge_ios=.8):
    # Prefer complete observations over views cut by an INTERNAL tile boundary.
    ordered = sorted(detections, key=lambda d: (d['internal_edge'], -d['confidence']))
    kept = []
    for d in ordered:
        duplicate = False
        for other in kept:
            ov, ios = overlap(d['bbox'], other['bbox'])
            cross = d['tile_id'] != other['tile_id']
            if cross and (ov > iou or ((d['internal_edge'] or other['internal_edge']) and ios > edge_ios)):
                duplicate = True
                break
        if not duplicate:
            kept.append(d)
    return sorted(kept, key=lambda d: (d['bbox'][1], d['bbox'][0], -d['confidence']))


def tile_origins(length, size, overlap_ratio):
    if size < 32 or not 0 <= overlap_ratio < 1:
        raise ValueError('tile size >=32 and overlap in [0,1) required')
    if length <= size:
        return [0]
    step = max(1, round(size * (1 - overlap_ratio)))
    return sorted(set(list(range(0, length - size + 1, step)) + [length - size]))


class ScoredWindowBackend:
    def __init__(self, mode, threshold=None, nms=None):
        self.mode = mode
        self.manifest = ROOT / 'configs' / MANIFESTS[mode]
        self.spec = json.loads(self.manifest.read_text(encoding='utf-8'))
        self.threshold = threshold if threshold is not None else self.spec.get('confidence_threshold', .25)
        self.nms = nms if nms is not None else self.spec.get('nms_threshold', .45)
        if not 0 < self.threshold <= 1 or not 0 < self.nms <= 1:
            raise ValueError('thresholds must be in (0,1]')
        self.legacy = None
        if mode != 'paper-window':
            self.legacy = YoloBackend(self.manifest, self.threshold, self.nms)
            self.net = self.legacy.net
            self.names = self.legacy.names
            self.size = self.legacy.size
            self.window_id = self.legacy.ids['window']
            self.runtime = 'opencv'
            self.rect = False
            self.pad = 128 if self.legacy.format == 'darknet' else 114
        else:
            if self.spec.get('status') != 'verified':
                raise RuntimeError(self.spec.get('blocked_reason', 'Author checkpoint not yet verified/exported'))
            for key, digest in self.spec['sha256'].items():
                path = self.manifest.parent / self.spec[key]
                if file_sha256(path) != digest:
                    raise ValueError(f'SHA-256 mismatch: {path}')
            self.names = self.spec['class_names']
            self.window_id = self.spec['window_class_id']
            if self.names[self.window_id].lower() != 'window':
                raise ValueError('Verified window class required')
            self.size = self.spec['input_size']
            self.rect = self.spec['rect']
            self.pad = 114
            self.runtime = self.spec['runtime']
            weights = self.manifest.parent / self.spec['weights']
            if weights.suffix != '.onnx':
                raise ValueError('Only ONNX is allowed for daily paper-window inference')
            if self.runtime == 'opencv':
                self.net = cv2.dnn.readNetFromONNX(np.fromfile(str(weights), np.uint8))
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            elif self.runtime == 'onnxruntime':
                import onnxruntime as ort
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 4
                self.net = ort.InferenceSession(str(weights), sess_options=opts, providers=['CPUExecutionProvider'])
            else:
                raise ValueError('Unsupported runtime')

    def predict(self, image):
        h, w = image.shape[:2]
        ratio = min(self.size / w, self.size / h)
        nw, nh = max(1, round(w * ratio)), max(1, round(h * ratio))
        dw, dh = self.size - nw, self.size - nh
        if self.rect:
            stride = self.spec['stride']
            dw, dh = dw % stride, dh % stride
        left, top = dw // 2, dh // 2
        canvas = np.full((nh + dh, nw + dw, 3), self.pad, np.uint8)
        canvas[top:top + nh, left:left + nw] = cv2.resize(image, (nw, nh))
        blob = cv2.dnn.blobFromImage(canvas, 1 / 255., swapRB=True, crop=False)
        if self.runtime == 'onnxruntime':
            outputs = self.net.run(None, {self.net.get_inputs()[0].name: blob})
        else:
            self.net.setInput(blob)
            outputs = self.net.forward(self.net.getUnconnectedOutLayersNames())
        if self.legacy:
            rows = self.legacy.decode(outputs)
        else:
            raw = outputs[0]
            if raw.ndim != 3 or raw.shape[:2] != (1, 4 + len(self.names)):
                raise ValueError(f'Unexpected author ONNX output {raw.shape}')
            rows = raw[0].T
        rows = rows[np.isfinite(rows).all(axis=1)]
        scores = rows[:, 4 + self.window_id]
        valid = (scores > self.threshold) & (rows[:, 2] > 0) & (rows[:, 3] > 0)
        if self.mode != 'yolov3':
            valid &= rows[:, 4:].argmax(axis=1) == self.window_id
        rows, scores = rows[valid], scores[valid]
        boxes = np.column_stack((rows[:, 0] - rows[:, 2] / 2, rows[:, 1] - rows[:, 3] / 2, rows[:, 2:4])).astype(float)
        # Author Ultralytics scales back with the common gain; historical
        # adapters use the actual rounded x/y scales. Preserve each baseline.
        sx, sy = (ratio, ratio) if self.mode == 'paper-window' else (nw / w, nh / h)
        if self.mode == 'paper-window':
            sx = sy = min(canvas.shape[0] / h, canvas.shape[1] / w)
            left = round((canvas.shape[1] - w * sx) / 2 - .1)
            top = round((canvas.shape[0] - h * sy) / 2 - .1)
        boxes[:, 0] = (boxes[:, 0] - left) / sx
        boxes[:, 1] = (boxes[:, 1] - top) / sy
        boxes[:, 2] /= sx
        boxes[:, 3] /= sy
        selected = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), self.threshold, self.nms)
        selected = np.asarray(selected).reshape(-1)
        max_det = self.spec.get('max_det', 300 if self.mode == 'paper-window' else 100000)
        result = []
        for i in selected[:max_det]:
            x, y, bw, bh = boxes[i]
            b = [float(np.clip(x, 0, w)), float(np.clip(y, 0, h)), float(np.clip(x + bw, 0, w)), float(np.clip(y + bh, 0, h))]
            if b[2] > b[0] and b[3] > b[1]:
                result.append({'bbox': b, 'confidence': float(scores[i]), 'class_id': self.window_id, 'class_name': 'window'})
        return sorted(result, key=lambda d: (d['bbox'][1], d['bbox'][0]))


def predict_image(backend, image, strategy='whole', tile_size=640, tile_overlap=.25, merge_iou=.5):
    h, w = image.shape[:2]
    if strategy == 'whole':
        detections = backend.predict(image)
        for d in detections:
            d.update(tile_id=0, tile_xyxy=[0, 0, w, h], internal_edge=False)
        return detections, {'tiles': 1, 'before_merge': len(detections), 'suppressed': 0}
    detections, tile_id = [], 0
    for y in tile_origins(h, tile_size, tile_overlap):
        for x in tile_origins(w, tile_size, tile_overlap):
            x1, y1 = min(w, x + tile_size), min(h, y + tile_size)
            for d in backend.predict(image[y:y1, x:x1]):
                b = d['bbox']
                internal = ((x > 0 and b[0] <= 2) or (y > 0 and b[1] <= 2) or
                            (x1 < w and b[2] >= x1 - x - 2) or (y1 < h and b[3] >= y1 - y - 2))
                d.update(bbox=[b[0] + x, b[1] + y, b[2] + x, b[3] + y], tile_id=tile_id,
                         tile_xyxy=[x, y, x1, y1], internal_edge=internal)
                detections.append(d)
            tile_id += 1
    final = merge_tiles(detections, merge_iou)
    return final, {'tiles': tile_id, 'before_merge': len(detections), 'suppressed': len(detections) - len(final)}




