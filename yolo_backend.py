"""Local OpenCV DNN adapters; never deserialize PyTorch pickle files."""
from __future__ import annotations

import json
import hashlib
import re
from pathlib import Path

import cv2
import numpy as np


class YoloBackend:
    def __init__(self, manifest: Path, threshold: float, nms_threshold: float = 0.45):
        if not 0 < threshold <= 1 or not 0 < nms_threshold <= 1:
            raise ValueError("score/nms threshold must be in (0,1]")
        self.spec = json.loads(manifest.read_text(encoding="utf-8"))
        for key, expected in self.spec.get("sha256", {}).items():
            path = (manifest.parent / self.spec[key])
            if not path.is_file():
                raise FileNotFoundError(f"Model asset not found: {path}. See YOLO_MODELS.md")
            with path.open('rb') as handle:
                digest = hashlib.sha256()
                for block in iter(lambda: handle.read(1024 * 1024), b''):
                    digest.update(block)
            if digest.hexdigest() != expected:
                raise ValueError(f"SHA-256 mismatch: {path}; incomplete or changed model asset")
        self.threshold = threshold
        self.nms_threshold = nms_threshold
        self.size = int(self.spec["input_size"])
        if self.size <= 0 or self.size % 32:
            raise ValueError("YOLO input_size must be a positive multiple of 32")
        self.format = self.spec["format"]
        names_file = (manifest.parent / self.spec["names"])
        self.names = [line.strip() for line in names_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        normalized = [name.lower() for name in self.names]
        if "door" not in normalized or "window" not in normalized:
            raise ValueError("Model must contain actual door AND window classes; COCO weights are not supported")
        self.ids = {kind: normalized.index(kind) for kind in ("window", "door")}
        if "class_ids" in self.spec and self.ids != self.spec["class_ids"]:
            raise ValueError("Door/window class IDs do not match the verified model manifest")
        weights = (manifest.parent / self.spec["weights"])
        if not weights.is_file():
            raise FileNotFoundError(f"Model not found: {weights}. See YOLO_MODELS.md")
        if self.format == "darknet":
            cfg = (manifest.parent / self.spec["config"])
            config_text = cfg.read_text(encoding='utf-8')
            counts = [int(n) for n in re.findall(r"(?m)^classes\s*=\s*(\d+)", config_text)]
            if not counts or any(n != len(self.names) for n in counts):
                raise ValueError("Darknet cfg class count does not match names")
            # OpenCV otherwise silently zeros scores below 0.2 in Region layers.
            # Override inference settings in memory, preserving the downloaded cfg.
            config_text = re.sub(r'(?m)^\s*(?:thresh|nms_threshold)\s*=.*$', '', config_text)
            config_text = config_text.replace('[yolo]', '[yolo]\nthresh=0\nnms_threshold=0')
            # Load buffers instead of filenames for Chinese Windows paths.
            self.net = cv2.dnn.readNetFromDarknet(np.frombuffer(config_text.encode('utf-8'), np.uint8), np.fromfile(str(weights), np.uint8))
            self.output_names = self.net.getUnconnectedOutLayersNames()
        elif self.format == "yolo11-onnx":
            if weights.suffix.lower() != ".onnx":
                raise ValueError("YOLO11 requires an approved ONNX export; .pt pickle loading is intentionally unsupported")
            self.net = cv2.dnn.readNetFromONNX(np.fromfile(str(weights), np.uint8))
            self.output_names = self.net.getUnconnectedOutLayersNames()
        else:
            raise ValueError(f"Unsupported format: {self.format}")
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        print(f"Model: {weights.name}; classes={len(self.names)}; {self.ids}; CPU", flush=True)

    def predict(self, image):
        windows, doors = self.predict_scored(image)
        return ([d['bbox'] for d in windows], [d['bbox'] for d in doors])

    def predict_scored(self, image):
        h, w = image.shape[:2]
        scale = min(self.size / w, self.size / h)
        nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
        left, top = (self.size - nw) // 2, (self.size - nh) // 2
        pad = 128 if self.format == "darknet" else 114
        canvas = np.full((self.size, self.size, 3), pad, np.uint8)
        canvas[top:top + nh, left:left + nw] = cv2.resize(image, (nw, nh))
        blob = cv2.dnn.blobFromImage(canvas, 1 / 255.0, (self.size, self.size), swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_names)
        rows = self.decode(outputs)
        return self.postprocess_scored(rows, w, h, left, top, nw / w, nh / h)

    def decode(self, outputs):
        if self.format == "darknet":
            rows = np.concatenate([o.reshape(-1, o.shape[-1]) for o in outputs])
            if rows.shape[1] != 5 + len(self.names):
                raise ValueError("Darknet output class count does not match manifest")
            # OpenCV's Region layer already multiplies class scores by objectness.
            return np.concatenate((rows[:, :4] * self.size, rows[:, 5:]), axis=1)
        if len(outputs) != 1:
            raise ValueError("Expected one raw YOLO11 ONNX output (export with nms=False)")
        raw = outputs[0]
        expected = 4 + len(self.names)
        if raw.ndim != 3 or raw.shape[0] != 1 or raw.shape[1] != expected:
            raise ValueError(f"Expected YOLO11 output [1,{expected},N], got {raw.shape}; export nms=False")
        return raw[0].T

    def postprocess(self, rows, width, height, left, top, sx, sy):
        scored_windows, scored_doors = self.postprocess_scored(rows, width, height, left, top, sx, sy)
        return ([d['bbox'] for d in scored_windows], [d['bbox'] for d in scored_doors])

    def postprocess_scored(self, rows, width, height, left, top, sx, sy):
        rows = rows[np.isfinite(rows).all(axis=1)]
        results = []
        for kind in ("window", "door"):
            class_id = self.ids[kind]
            scores = rows[:, 4 + class_id]
            valid = scores > self.threshold
            # Match Ultralytics' default single-label YOLO11 inference. Darknet
            # Open Images retains independent sigmoid labels (hierarchical classes).
            if self.format == 'yolo11-onnx':
                valid &= rows[:, 4:].argmax(axis=1) == class_id
            candidates = rows[valid, :4]
            confidence = scores[valid]
            boxes, kept_scores = [], []
            for (cx, cy, bw, bh), score in zip(candidates, confidence):
                if bw <= 0 or bh <= 0:
                    continue
                x0 = float((cx - bw / 2 - left) / sx)
                y0 = float((cy - bh / 2 - top) / sy)
                x1 = float((cx + bw / 2 - left) / sx)
                y1 = float((cy + bh / 2 - top) / sy)
                if x1 > x0 and y1 > y0:
                    boxes.append([x0, y0, x1 - x0, y1 - y0])
                    kept_scores.append(float(score))
            selected = cv2.dnn.NMSBoxes(boxes, kept_scores, self.threshold, self.nms_threshold)
            # NMS uses the full predicted boxes; clip only afterward so padding
            # and image edges do not change IoU and cause premature suppression.
            final = []
            for i in np.asarray(selected).reshape(-1):
                x, y, w, h = boxes[i]
                x0, x1 = np.clip([x, x + w], 0, width).tolist()
                y0, y1 = np.clip([y, y + h], 0, height).tolist()
                if x1 > x0 and y1 > y0:
                    final.append({'bbox': [x0, y0, x1, y1], 'confidence': float(kept_scores[i]),
                                  'class_name': kind, 'class_id': class_id})
            results.append(sorted(final, key=lambda d: (d['bbox'][1], d['bbox'][0])))
        return tuple(results)

