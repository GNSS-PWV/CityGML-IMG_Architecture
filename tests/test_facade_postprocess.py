"""去重、二维分块和批量流程检查；不加载真实模型，不联网。"""
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import csv
import json
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import detect_my_facade as app
from detect_my_facade import box_to_full_image, detection_views, remove_duplicates


def record(kind, score, box):
    return {"class": kind, "score": score, "box": box, "source": "test"}


class PostprocessChecks(unittest.TestCase):
    def test_tall_image_grid_and_vertical_boundary(self):
        views = detection_views(1700, 1500, True, False)
        self.assertTrue(all(b[2]-b[0] <= 1200 and b[3]-b[1] <= 1000 for _, b in views))
        for x in range(0, 1700, 17):
            for y in range(0, 1500, 17):
                self.assertTrue(any(b[0] <= x < b[2] and b[1] <= y < b[3] for _, b in views))
        self.assertEqual(box_to_full_image([20,30,80,90], (500,500,1700,1500), (1700,1500),12),
                         [520,530,580,590])
        self.assertIsNone(box_to_full_image([20,0,80,90], (500,500,1700,1500),(1700,1500),12))
        self.assertEqual(box_to_full_image([20,900,80,1000], (500,500,1700,1500),(1700,1500),12),
                         [520,1400,580,1500])

    def test_native_tiles_cover_full_photo_without_full_frame(self):
        views = detection_views(2940, 833, use_tiles=True, include_full_image=False)
        self.assertEqual(views, [("tile_1", (0, 0, 1200, 833)),
                                 ("tile_2", (900, 0, 2100, 833)),
                                 ("tile_3", (1740, 0, 2940, 833))])
        # 每列原始像素都被至少一个图块覆盖，裁剪并不等于把整图缩小。
        self.assertTrue(all(any(box[0] <= x < box[2] for _, box in views) for x in range(2940)))

    def test_optional_full_frame_and_narrow_photo(self):
        views = detection_views(2940, 833, use_tiles=True, include_full_image=True)
        self.assertEqual(len(views), 4)
        self.assertEqual(views[0], ("full", (0, 0, 2940, 833)))
        self.assertEqual(detection_views(900, 833, True, False), [("full", (0, 0, 900, 833))])
        self.assertEqual(detection_views(2940, 833, False, False), [("full", (0, 0, 2940, 833))])

    def test_observed_nested_windows(self):
        # 来自本次实际检测：小框完全包含在大框内，但 IoU 只有约 0.341。
        outer = record("window", .559, [2311.3, 141.2, 2417.4, 277.3])
        inner = record("window", .502, [2331.3, 168.1, 2384.2, 261.4])
        result = remove_duplicates([outer, inner])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["box"], outer["box"])
        self.assertEqual(result[0]["class"], "window")

    def test_observed_conflict_does_not_force_higher_score_class(self):
        # 实际框：门的分数反而较高。几何去重不能据此证明它就是门。
        door = record("door", .532, [1906.1, 593.6, 1990.8, 824.8])
        window = record("window", .428, [1905.3, 595.1, 1987.5, 826.9])
        result = remove_duplicates([door, window])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["class"], "ambiguous")
        self.assertTrue(result[0]["review_required"])
        self.assertEqual({r["class"] for r in result[0]["candidates"]}, {"door", "window"})

    def test_nearby_windows_remain_separate(self):
        a = record("window", .6, [0, 0, 100, 100])
        b = record("window", .5, [90, 0, 190, 100])
        self.assertEqual(len(remove_duplicates([a, b])), 2)

    def test_small_window_inside_large_door_remains_separate(self):
        # 包含关系不一定是重复：门上方/门内的小玻璃不能一律吞掉。
        door = record("door", .6, [0, 0, 100, 200])
        glass = record("window", .5, [40, 20, 60, 40])
        self.assertEqual(len(remove_duplicates([door, glass])), 2)

    def test_lower_score_class_evidence_survives_same_class_merge(self):
        window = record("window", .6, [0, 0, 100, 100])
        window2 = record("window", .5, [5, 5, 95, 95])
        door = record("door", .4, [8, 8, 92, 92])
        result = remove_duplicates([window, window2, door])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["class"], "ambiguous")
        self.assertEqual(len(result[0]["candidates"]), 3)

    def test_overlap_chain_does_not_join_separate_endpoints(self):
        # A 接近 B、B 接近 C，不等于 A、B、C 一定是同一扇窗。
        a = record("window", .9, [0, 0, 100, 100])
        b = record("window", .8, [30, 0, 130, 100])
        c = record("window", .7, [60, 0, 160, 100])
        self.assertEqual(len(remove_duplicates([a, b, c])), 2)

    def test_does_not_modify_original_predictions(self):
        a = record("window", .6, [0, 0, 100, 100])
        b = record("door", .5, [0, 0, 100, 100])
        remove_duplicates([a, b])
        self.assertEqual(a, record("window", .6, [0, 0, 100, 100]))
        self.assertEqual(b, record("door", .5, [0, 0, 100, 100]))


class BatchChecks(unittest.TestCase):
    def test_batch_loads_once_continues_after_bad_photo_and_links_outputs(self):
        from PIL import Image

        # 使用小型合成图和假的模型输出，驱动真实批量循环/保存/汇总，不执行神经网络。
        class Array:
            def __init__(self, values): self.values = values
            def detach(self): return self
            def cpu(self): return self
            def tolist(self): return self.values

        class Inputs(dict):
            input_ids = [1]
            def to(self, device): return self

        class Processor:
            def __call__(self, images, text, do_resize, return_tensors):
                self.prompt = text
                return Inputs(pixel_values=SimpleNamespace(shape=(1, 3, images.height, images.width)))

            def post_process_grounded_object_detection(self, outputs, input_ids, text_threshold,
                                                        target_sizes, box_threshold):
                found = "window" in self.prompt
                return [{"boxes": Array([[20,20,60,70]] if found else []),
                         "scores": Array([.8] if found else [])}]

        class Model:
            def to(self, device): return self
            def eval(self): pass
            def __call__(self, **inputs): return None

        processor_loader = MagicMock(return_value=Processor())
        model_loader = MagicMock(return_value=Model())
        fake_transformers = SimpleNamespace(__version__="test", AutoProcessor=SimpleNamespace(from_pretrained=processor_loader),
                                            AutoModelForZeroShotObjectDetection=SimpleNamespace(from_pretrained=model_loader))
        fake_torch = SimpleNamespace(__version__="test", cuda=SimpleNamespace(is_available=lambda: False),
                                     inference_mode=nullcontext)
        with tempfile.TemporaryDirectory(dir=str(ROOT / "tests"), prefix="batch_check_") as temporary:
            base = Path(temporary).resolve()
            # TemporaryDirectory退出时仅清理本检查创建的目录，确认它位于项目tests内。
            self.assertEqual(base.parent, (ROOT / "tests").resolve())
            photos = base / "textures"
            photos.mkdir()
            for name in ("100_front.png", "102_right.PNG"):
                Image.new("RGB", (100,90), "white").save(photos / name)
            (photos / "101_broken.png").write_bytes(b"not an image")
            (base / "gt_masks").mkdir()
            Image.new("RGB", (100,90)).save(base / "gt_masks/answer.png")
            for folder in ("citygml", "obj"):
                (base / folder).mkdir()
            for number in (100,102):
                (base / f"citygml/DEBY_LOD3_{number}.gml").write_text("fixture", encoding="utf-8")
                (base / f"obj/DEBY_LOD2_{number}.obj").write_text("fixture", encoding="utf-8")
            output = base / "results"
            with patch.multiple(app, IMAGE_DIR=photos, DATASET_DIR=base, OUTPUT_ROOT=output), \
                    patch.object(app, "local_model_directory", return_value=base), \
                    patch.dict(sys.modules, {"torch": fake_torch, "transformers": fake_transformers}), \
                    patch("builtins.print"):
                with self.assertRaises(SystemExit) as stopped:
                    app.main()
                self.assertEqual(stopped.exception.code, 1)
            self.assertEqual(processor_loader.call_count, 1)
            self.assertEqual(model_loader.call_count, 1)
            self.assertTrue(model_loader.call_args[1]["local_files_only"])
            latest = json.loads((output / "latest_run.json").read_text(encoding="utf-8"))
            run = Path(latest["run_directory"])
            summary = json.loads((run / "batch_summary.json").read_text(encoding="utf-8"))
            self.assertEqual((summary["image_count"], summary["success_count"], summary["failed_count"]), (3,2,1))
            self.assertEqual(summary["state"], "completed_with_errors")
            with (run / "output_stage2.csv").open(encoding="utf-8", newline="") as f:
                self.assertEqual([r["texture_filename"] for r in csv.DictReader(f)], ["100_front", "102_right"])
            items = json.loads((run / "mapping_manifest.json").read_text(encoding="utf-8"))["items"]
            self.assertEqual([r["status"] for r in items], ["awaiting_calibration", "detection_failed", "awaiting_calibration"])
            for item in (items[0], items[2]):
                self.assertTrue(Path(item["detection_json"]).is_file())
                self.assertTrue((Path(item["result_directory"]) / "detections.png").is_file())
                self.assertIsNone(item["wall_id"])

    def test_duplicate_output_names_fail_before_loading_model(self):
        with tempfile.TemporaryDirectory(dir=str(ROOT / "tests"), prefix="names_check_") as temporary:
            folder = Path(temporary).resolve()
            self.assertEqual(folder.parent, (ROOT / "tests").resolve())
            (folder / "same.png").write_bytes(b"unused")
            (folder / "same.jpg").write_bytes(b"unused")
            with patch.object(app, "IMAGE_DIR", folder), self.assertRaises(ValueError):
                app.find_images()


if __name__ == "__main__":
    unittest.main(verbosity=2)
