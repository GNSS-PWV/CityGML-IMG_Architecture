"""实验 D 批量流程的离线检查：只用合成图片和模拟 API，不联网、不读取真实 Token。"""
import base64
import copy
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from PIL import Image
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import try_grounding_dino16 as app


def tile(name, crop_box, objects):
    return {"name": name, "crop_box": crop_box, "result": {"objects": objects}}


def detection(kind="window", bbox=(30, 40, 90, 100), score=0.7):
    return {"category": kind, "bbox": list(bbox), "score": score}


def response(data):
    result = MagicMock()
    result.json.return_value = {"code": 0, "data": data}
    return result


class TileGeometryChecks(unittest.TestCase):
    def test_native_grid_covers_photo_and_small_image_once(self):
        self.assertEqual(app.tile_views(2940, 833), [
            ("tile_1", (0, 0, 1200, 833)),
            ("tile_2", (900, 0, 2100, 833)),
            ("tile_3", (1740, 0, 2940, 833)),
        ])
        self.assertEqual(app.tile_views(700, 600), [("tile_1", (0, 0, 700, 600))])
        views = app.tile_views(1700, 1500)
        for x in range(0, 1700, 37):
            for y in range(0, 1500, 41):
                self.assertTrue(any(b[0] <= x < b[2] and b[1] <= y < b[3]
                                    for _, b in views))
        self.assertTrue(all(b[2] - b[0] <= 1200 and b[3] - b[1] <= 1000
                            for _, b in views))

    def test_original_coordinates_use_offset_without_1536_rescaling(self):
        result = app.merge_tile_results([
            tile("tile_4", (500, 500, 1700, 1500), [detection(bbox=(30, 40, 90, 100))])
        ], (1700, 1500))
        self.assertEqual(result["objects"][0]["bbox"], [530, 540, 590, 600])
        self.assertEqual(result["objects"][0]["local_box"], [30, 40, 90, 100])

    def test_overlap_duplicates_merge_and_preserve_class_conflict(self):
        inputs = [
            tile("tile_1", (0, 0, 1200, 833), [
                detection(bbox=(1000, 100, 1100, 300), score=.6),
                detection(bbox=(1020, 120, 1080, 280), score=.5),
            ]),
            tile("tile_2", (900, 0, 2100, 833), [
                detection("door", (100, 100, 200, 300), .8),
                detection("window", (250, 100, 330, 300), .7),
            ]),
        ]
        original = copy.deepcopy(inputs)
        result = app.merge_tile_results(inputs, (2940, 833))
        self.assertEqual(len(result["objects"]), 2)
        conflict = next(obj for obj in result["objects"] if obj["category"] == "ambiguous")
        self.assertEqual(conflict["bbox"], [1000, 100, 1100, 300])
        self.assertEqual(conflict["class_scores"], {"door": .8, "window": .6})
        self.assertTrue(conflict["review_required"])
        self.assertEqual(len(conflict["candidates"]), 3)
        self.assertEqual(len(result["candidates_before_merge"]), 4)
        self.assertEqual(inputs, original)

    def test_internal_tile_edges_rejected_but_original_image_edges_kept(self):
        result = app.merge_tile_results([
            tile("tile_1", (0, 0, 1200, 833), [
                detection(bbox=(0, 0, 100, 100)),
                detection(bbox=(1130, 100, 1200, 200)),
            ]),
            tile("tile_2", (900, 0, 2100, 833), [detection(bbox=(0, 100, 100, 200))]),
            tile("tile_3", (1740, 0, 2940, 833), [detection(bbox=(1130, 750, 1200, 833))]),
        ], (2940, 833))
        self.assertEqual(len(result["rejected_detections"]), 2)
        boxes = [obj["bbox"] for obj in result["objects"]]
        self.assertIn([0, 0, 100, 100], boxes)
        self.assertIn([2870, 750, 2940, 833], boxes)

    def test_invalid_values_are_recorded_instead_of_drawn(self):
        result = app.merge_tile_results([
            tile("tile_1", (0, 0, 100, 100), [
                detection(bbox=(10, 10, 5, 5)),
                detection(bbox=(10, 10, float("nan"), 50)),
                detection(score=2),
                detection(kind="sky"),
            ])
        ], (100, 100))
        self.assertEqual(result["objects"], [])
        self.assertEqual(len(result["rejected_detections"]), 4)


class PollingChecks(unittest.TestCase):
    def test_queue_longer_than_three_minutes_finishes_and_reports_progress(self):
        # 模拟服务端 200 秒后才完成；推进虚拟时钟，不真正等待或联网。
        clock = {"seconds": 0.0}
        result = {"objects": [detection()]}
        session = MagicMock()
        session.get.side_effect = lambda *args, **kwargs: response(
            {"status": "waiting"} if clock["seconds"] < 200 else
            {"status": "success", "result": result})

        def sleep(seconds):
            clock["seconds"] += seconds

        with patch.object(app.time, "monotonic", side_effect=lambda: clock["seconds"]), \
                patch.object(app.time, "sleep", side_effect=sleep), patch("builtins.print") as output:
            self.assertEqual(app.wait_for_task(session, "offline-task-slow"), result)
        session.post.assert_not_called()
        self.assertGreaterEqual(clock["seconds"], 200)
        progress = [str(call.args[0]) for call in output.call_args_list]
        self.assertTrue(any("已等待 180" in line and "waiting" in line for line in progress))

    def test_deadline_keeps_task_identity_and_last_status_in_error(self):
        clock = {"seconds": 0.0}
        session = MagicMock()
        session.get.return_value = response({"status": "waiting"})

        def sleep(seconds):
            clock["seconds"] += seconds

        with patch.object(app, "WAIT_SECONDS", 4), \
                patch.object(app.time, "monotonic", side_effect=lambda: clock["seconds"]), \
                patch.object(app.time, "sleep", side_effect=sleep), patch("builtins.print"):
            with self.assertRaises(TimeoutError) as stopped:
                app.wait_for_task(session, "offline-task-timeout")
        self.assertIn("offline-task-timeout", str(stopped.exception))
        self.assertIn("waiting", str(stopped.exception))
        self.assertEqual(clock["seconds"], 4)
        session.post.assert_not_called()


class OfflineApiFlowChecks(unittest.TestCase):
    def run_main(self, base, session, resume=None, **settings):
        parameters = dict(ROOT=base, IMAGE_DIR=base / "textures",
                          OUTPUT_ROOT=base / "output", RESUME_RUN_DIR=resume)
        parameters.update(settings)
        with patch.multiple(app, **parameters), \
                patch.dict(app.os.environ, {"DDS_API_TOKEN": "offline-test-token"}, clear=True), \
                patch.object(app.requests, "Session", return_value=session), \
                patch.object(app.time, "sleep", side_effect=AssertionError("Unexpected polling wait")), \
                patch("builtins.print"):
            return app.main()

    def temporary_directory(self):
        # TemporaryDirectory 只清理本测试创建的目录；先核实其解析后的父目录。
        temporary = tempfile.TemporaryDirectory(prefix="grounding_dino16_check_")
        self.addCleanup(temporary.cleanup)
        base = Path(temporary.name).resolve()
        self.assertEqual(base.parent, Path(tempfile.gettempdir()).resolve())
        (base / "textures").mkdir()
        return base

    def add_photo(self, base, name="a_front.png", size=(700, 600), color=(16, 64, 128)):
        image_path = base / "textures" / name
        with Image.new("RGB", size, color) as image:
            image.save(image_path)
        return image_path

    def fake_session(self, count=4):
        session = MagicMock()
        session.__enter__.return_value = session
        session.__exit__.return_value = False
        session.post.side_effect = [response({"task_uuid": "offline-task-{}".format(i)})
                                    for i in range(1, count + 1)]
        session.get.side_effect = [response({"status": "success", "result": {"objects": [detection()]}})
                                   for _ in range(count)]
        return session

    def output_directory(self, base):
        outputs = list((base / "output").glob("D_*"))
        self.assertEqual(len(outputs), 1)
        return outputs[0]

    def read_json(self, path):
        return json.loads(path.read_text(encoding="utf-8"))

    def test_two_photos_native_pngs_and_complete_per_image_and_batch_outputs(self):
        base = self.temporary_directory()
        self.add_photo(base, "a_front.png", (2940, 833))
        self.add_photo(base, "b_front.png", (700, 600), (32, 96, 160))
        session = self.fake_session()
        output = self.run_main(base, session)
        self.assertIsInstance(output, Path)
        self.assertEqual(output, self.output_directory(base))
        self.assertEqual(session.post.call_count, 4)
        self.assertEqual(session.get.call_count, 4)
        keys = set()
        for index, call in enumerate(session.post.call_args_list):
            body = call.kwargs["json"]
            self.assertEqual(body["bbox_threshold"], .20)
            self.assertEqual(body["iou_threshold"], .8)
            self.assertEqual(body["model"], "GroundingDino-1.6-Pro")
            self.assertEqual(body["prompt"], {"type": "text", "text": "window.door."})
            prefix, encoded = body["image"].split(",", 1)
            self.assertEqual(prefix, "data:image/png;base64")
            with Image.open(io.BytesIO(base64.b64decode(encoded))) as crop:
                self.assertEqual(crop.size, (1200, 833) if index < 3 else (700, 600))
                self.assertEqual(crop.getpixel((0, 0)),
                                 (16, 64, 128) if index < 3 else (32, 96, 160))
            keys.add(call.kwargs["headers"]["Idempotency-Key"])
        self.assertEqual(len(keys), 4)
        for stem, size, tile_count in (("a_front", (2940, 833), 3), ("b_front", (700, 600), 1)):
            directory = output / stem
            for filename in ("request_info.json", "result.json", "detections.png",
                             "detections.json", "output_stage2.csv", "needs_review.json"):
                self.assertTrue((directory / filename).is_file(), str(directory / filename))
            record = self.read_json(directory / "request_info.json")
            self.assertEqual(record["state"], "completed")
            self.assertEqual(record["planned_requests"], tile_count)
            self.assertEqual([part["state"] for part in record["tiles"]], ["completed"] * tile_count)
            self.assertEqual(len(list((directory / "tiles").glob("*_result.json"))), tile_count)
            with Image.open(directory / "detections.png") as drawn:
                self.assertEqual(drawn.size, size)
            report = self.read_json(directory / "detections.json")
            self.assertEqual(len(report["detections"]), tile_count)
            self.assertEqual(report["window_count"], tile_count)
            with (directory / "output_stage2.csv").open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["texture_filename"], stem)
            self.assertEqual(len(json.loads(rows[0]["bboxes_window"])), tile_count)
            self.assertEqual(json.loads(rows[0]["bboxes_door"]), [])
        result = self.read_json(output / "a_front/result.json")
        self.assertEqual([obj["bbox"] for obj in result["objects"]], [
            [30, 40, 90, 100], [930, 40, 990, 100], [1770, 40, 1830, 100],
        ])
        for filename in ("batch_summary.json", "batch_summary.csv", "mapping_manifest.json"):
            self.assertTrue((output / filename).is_file())
        summary = self.read_json(output / "batch_summary.json")
        self.assertEqual(summary["state"], "completed")
        self.assertEqual((summary["image_count"], summary["planned_requests"]), (2, 4))
        self.assertEqual((summary["success_count"], summary["failed_count"], summary["pending_count"]),
                         (2, 0, 0))
        self.assertEqual([item["detection_status"] for item in summary["items"]], ["success", "success"])
        self.assertEqual(self.read_json(output / "mapping_manifest.json")["items"], summary["items"])
        self.assertEqual(self.read_json(base / "output/latest_run.json")["run_directory"], str(output))
        for saved in output.rglob("*.json"):
            self.assertNotIn("offline-test-token", saved.read_text(encoding="utf-8"))

    def test_second_photo_submission_failure_preserves_first_and_stops_batch(self):
        base = self.temporary_directory()
        for name in ("a_front.png", "b_front.png", "c_front.png"):
            self.add_photo(base, name)
        session = self.fake_session()
        session.post.side_effect = [response({"task_uuid": "offline-task-1"}),
                                    requests.exceptions.ProxyError("offline synthetic failure")]
        with self.assertRaises(requests.exceptions.ProxyError):
            self.run_main(base, session)
        self.assertEqual(session.post.call_count, 2)
        self.assertEqual(session.get.call_count, 1)
        output = self.output_directory(base)
        completed = self.read_json(output / "a_front/request_info.json")
        self.assertEqual(completed["state"], "completed")
        record = self.read_json(output / "b_front/request_info.json")
        self.assertEqual(record["state"], "failed")
        self.assertEqual(record["tiles"][0]["state"], "error")
        self.assertEqual(record["tiles"][0]["failure_stage"], "submitting")
        self.assertTrue((output / "a_front/tiles/tile_1_result.json").is_file())
        self.assertTrue((output / "a_front/detections.png").is_file())
        self.assertFalse((output / "b_front/result.json").exists())
        self.assertFalse((output / "c_front/result.json").exists())
        summary = self.read_json(output / "batch_summary.json")
        self.assertEqual(summary["state"], "failed")
        self.assertEqual((summary["success_count"], summary["failed_count"], summary["pending_count"]),
                         (1, 1, 1))
        self.assertEqual([item["detection_status"] for item in summary["items"]],
                         ["success", "failed", "pending"])
        self.assertEqual([item["window_count"] for item in summary["items"]], [1, None, None])
        self.assertEqual([item["final_object_count"] for item in summary["items"]], [1, None, None])
        with (output / "output_stage2.csv").open(encoding="utf-8", newline="") as stream:
            self.assertEqual([row["texture_filename"] for row in csv.DictReader(stream)], ["a_front"])

    def test_class_conflict_is_reviewable_and_not_written_as_both_window_and_door(self):
        base = self.temporary_directory()
        self.add_photo(base)
        session = self.fake_session()
        session.get.side_effect = [response({"status": "success", "result": {"objects": [
            detection("window", score=.7), detection("door", score=.6),
        ]}})]
        output = self.run_main(base, session) / "a_front"
        report = self.read_json(output / "detections.json")
        self.assertEqual((report["window_count"], report["door_count"], report["ambiguous_count"]),
                         (0, 0, 1))
        pending = self.read_json(output / "needs_review.json")
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["class_scores"], {"window": .7, "door": .6})
        with (output / "output_stage2.csv").open(encoding="utf-8", newline="") as stream:
            row = next(csv.DictReader(stream))
        self.assertEqual(json.loads(row["bboxes_window"]), [])
        self.assertEqual(json.loads(row["bboxes_door"]), [])

    def test_resume_known_task_only_polls_and_does_not_repost_successful_photos(self):
        base = self.temporary_directory()
        self.add_photo(base, "a_front.png")
        self.add_photo(base, "b_front.png")
        session = self.fake_session()
        session.get.side_effect = [response({"status": "success", "result": {"objects": [detection()]}}),
                                   requests.exceptions.ReadTimeout("offline interrupted polling")]
        with self.assertRaises(requests.exceptions.ReadTimeout):
            self.run_main(base, session)
        output = self.output_directory(base)
        first_result = (output / "a_front/result.json").read_bytes()
        record = self.read_json(output / "b_front/request_info.json")
        self.assertEqual(record["tiles"][0]["task_uuid"], "offline-task-2")
        resume_session = self.fake_session()
        self.assertEqual(self.run_main(base, resume_session, resume=output), output)
        resume_session.post.assert_not_called()
        self.assertEqual(resume_session.get.call_count, 1)
        self.assertTrue(resume_session.get.call_args.args[0].endswith("/offline-task-2"))
        self.assertEqual((output / "a_front/result.json").read_bytes(), first_result)
        self.assertEqual(self.read_json(output / "b_front/request_info.json")["state"], "completed")

    def test_resume_retains_successful_tiles_within_interrupted_photo(self):
        base = self.temporary_directory()
        self.add_photo(base, size=(2940, 833))
        session = self.fake_session()
        session.get.side_effect = [response({"status": "success", "result": {"objects": [detection()]}}),
                                   requests.exceptions.ReadTimeout("offline interrupted polling")]
        with self.assertRaises(requests.exceptions.ReadTimeout):
            self.run_main(base, session)
        output = self.output_directory(base)
        resume_session = self.fake_session()
        self.run_main(base, resume_session, resume=output)
        self.assertEqual(resume_session.get.call_count, 2)
        self.assertEqual(resume_session.post.call_count, 1)
        self.assertTrue(resume_session.get.call_args_list[0].args[0].endswith("/offline-task-2"))
        self.assertEqual(self.read_json(output / "a_front/request_info.json")["state"], "completed")

    def test_resume_complete_batch_makes_no_api_calls(self):
        base = self.temporary_directory()
        self.add_photo(base)
        output = self.run_main(base, self.fake_session())
        original_result = (output / "a_front/result.json").read_bytes()
        session = self.fake_session()
        self.assertEqual(self.run_main(base, session, resume=output), output)
        session.post.assert_not_called()
        session.get.assert_not_called()
        self.assertEqual((output / "a_front/result.json").read_bytes(), original_result)
        summary = self.read_json(output / "batch_summary.json")
        self.assertEqual((summary["state"], summary["success_count"]), ("completed", 1))

    def test_resume_missing_csv_reexports_from_local_tiles_without_api_calls(self):
        base = self.temporary_directory()
        self.add_photo(base)
        output = self.run_main(base, self.fake_session())
        csv_path = (output / "a_front/output_stage2.csv").resolve()
        original_csv = csv_path.read_bytes()
        # 仅删除本测试 TemporaryDirectory 内生成的这一份文件。
        csv_path.relative_to(base)
        csv_path.unlink()
        session = self.fake_session()
        self.assertEqual(self.run_main(base, session, resume=output), output)
        session.post.assert_not_called()
        session.get.assert_not_called()
        self.assertEqual(csv_path.read_bytes(), original_csv)
        summary = self.read_json(output / "batch_summary.json")
        self.assertEqual((summary["state"], summary["success_count"]), ("completed", 1))

    def test_unknown_submission_outcome_is_never_automatically_reposted(self):
        base = self.temporary_directory()
        self.add_photo(base)
        session = self.fake_session()
        session.post.side_effect = requests.exceptions.ProxyError("offline unknown submission")
        with self.assertRaises(requests.exceptions.ProxyError):
            self.run_main(base, session)
        output = self.output_directory(base)
        resume_session = self.fake_session()
        with self.assertRaises(RuntimeError):
            self.run_main(base, resume_session, resume=output)
        resume_session.post.assert_not_called()
        resume_session.get.assert_not_called()

    def test_changed_configuration_or_image_rejected_before_network_on_resume(self):
        for change in ("configuration", "image"):
            with self.subTest(change=change):
                base = self.temporary_directory()
                self.add_photo(base)
                output = self.run_main(base, self.fake_session())
                parameters = {"BOX_THRESHOLD": .25} if change == "configuration" else {}
                if change == "image":
                    self.add_photo(base, color=(200, 150, 100))
                resume_session = self.fake_session()
                with self.assertRaises((ValueError, RuntimeError)):
                    self.run_main(base, resume_session, resume=output, **parameters)
                resume_session.post.assert_not_called()
                resume_session.get.assert_not_called()

    def test_duplicate_stems_rejected_before_any_api_submission(self):
        base = self.temporary_directory()
        self.add_photo(base, "same.png")
        self.add_photo(base, "SAME.jpg")
        session = self.fake_session()
        with self.assertRaises((ValueError, RuntimeError)):
            self.run_main(base, session)
        session.post.assert_not_called()
        session.get.assert_not_called()

    def test_all_photos_are_decoded_before_first_api_submission(self):
        base = self.temporary_directory()
        self.add_photo(base)
        (base / "textures/z_broken.png").write_bytes(b"invalid synthetic PNG")
        session = self.fake_session()
        with self.assertRaises((OSError, ValueError, RuntimeError)):
            self.run_main(base, session)
        session.post.assert_not_called()
        session.get.assert_not_called()


if __name__ == "__main__":
    unittest.main()
