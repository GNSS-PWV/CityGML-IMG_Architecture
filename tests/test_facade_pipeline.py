"""全流程安全边界：候选模糊拒绝，检测缓存必须匹配内容，不读取文件名身份。"""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_facade_pipeline as pipeline
import try_grounding_dino16 as detector
from PIL import Image


def candidate(name, building, wall, score, passed=True):
    return {"candidate_id": name, "building_id": building, "wall_id": wall,
            "geometry_score": score, "gate_passed": passed, "wall_vote": {"accepted": True}}


class PipelineTests(unittest.TestCase):
    def test_batch_saves_each_image_and_continues_after_one_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            photos = root/"photos"
            photos.mkdir()
            for name in ("a.png", "b.png", "c.png"):
                Image.new("RGB", (10,10)).save(photos/name)
            def run(photo, *_):
                summaries = list((root/"out").glob("batch_*/summary.json"))
                before = len(pipeline.read_json(summaries[0])["records"]) if summaries else 0
                self.assertEqual(before, "abc".index(photo.stem))
                if photo.stem == "b":
                    raise ValueError("synthetic failure")
                output = root/photo.stem
                output.mkdir()
                pipeline.write_json(output/"result.json", {"status": "needs_review"})
                return output
            with patch.object(pipeline, "OUTPUT_ROOT", root/"out"), patch.object(pipeline, "run_pipeline", side_effect=run):
                batch = pipeline.run_photo_directory(photos)
                self.assertEqual([r["status"] for r in pipeline.read_json(batch/"summary.json")["records"]],
                                 ["needs_review", "failed", "needs_review"])

    def test_model_source_and_tiling_changes_invalidate_cache_identity(self):
        original = pipeline.tiled_cache_identity(np.eye(3))
        with patch.object(detector, "TILE_OVERLAP", detector.TILE_OVERLAP+1):
            self.assertNotEqual(original, pipeline.tiled_cache_identity(np.eye(3)))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"model.py"
            path.write_text("revision = 1", encoding="utf-8")
            before = pipeline.roma.code_sha256(Path(folder))
            path.write_text("revision = 2", encoding="utf-8")
            self.assertNotEqual(before, pipeline.roma.code_sha256(Path(folder)))

    def test_two_views_disagreeing_in_world_space_block_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            camera = {"projection": "orthographic", "image_size_wh": [100, 100],
                      "pixel_depth_to_local": np.eye(4).tolist()}
            manifest = {"source_gml": "unused.gml", "world_origin_m": [0, 0, 0],
                        "objects": [{"index": 0, "wall_id": "wall"}]}
            pipeline.write_json(root/"camera.json", camera)
            pipeline.write_json(root/"manifest.json", manifest)
            np.savez(root/"geometry.npz", object_index=np.zeros((100,100),dtype=int))
            view = {"camera": str(root/"camera.json"), "manifest": str(root/"manifest.json"),
                    "geometry": str(root/"geometry.npz")}
            first = dict(candidate("a", "1", "wall", .9), view_name="a", view=view,
                         homography_photo_to_render=np.eye(3).tolist())
            shifted = np.eye(3)
            shifted[0,2] = 1.
            second = dict(candidate("b", "1", "wall", .8), view_name="b", view=view,
                          homography_photo_to_render=shifted.tolist())
            wall = {"id": "wall", "frame": {"origin": np.array([0,0,10]),
                    "u": np.array([1,0,0]), "v": np.array([0,1,0])}}
            with patch("facade_auto_mapping.validated_wall", return_value=wall):
                report = pipeline.check_view_agreement(first, [first,second], [100,100])
                self.assertFalse(report["passed"])
                self.assertAlmostEqual(report["comparisons"][0]["median_m"], 1.)
                second["homography_photo_to_render"] = np.eye(3).tolist()
                self.assertTrue(pipeline.check_view_agreement(first, [first,second], [100,100])["passed"])

    def test_single_passed_view_is_not_confirmation(self):
        row = dict(candidate("a", "building", "wall", .9), view_name="view_000")
        report = pipeline.check_view_agreement(row, [row], [100, 100])
        self.assertFalse(report["available"])
        self.assertFalse(report["passed"])
        self.assertEqual(report["reason"], "insufficient_cross_view_confirmation")

    def test_confirmation_uses_unseen_visible_views_of_the_same_wall(self):
        row = dict(candidate("a", "b", "wall", .8), view_name="seen")
        def view(name, pixels, building="b", wall="wall"):
            return {"view_name": name, "building_id": building,
                    "visible_walls": [{"wall_id": wall, "pixel_count": pixels}]}
        views = [view("seen", 9999), view("b", 600), view("a", 1000), view("c", 800),
                 view("wrong_building", 9999, "other"), view("wrong_wall", 9999, wall="other"),
                 view("edge", 4)]
        selected = pipeline.confirmation_views(row, [row], views)
        self.assertEqual([v["view_name"] for v in selected], ["a", "c"])

    def test_high_scoring_failed_geometry_is_not_accepted(self):
        chosen, decision = pipeline.select_candidate([candidate("bad", "1", "a", .99, False)])
        self.assertIsNone(chosen)
        self.assertFalse(decision["accepted"])

    def test_close_buildings_and_close_walls_both_require_review(self):
        for building, wall in [("2", "b"), ("1", "b")]:
            chosen, decision = pipeline.select_candidate([candidate("a", "1", "a", .80),
                                                           candidate("b", building, wall, .76)])
            self.assertIsNone(chosen)
            self.assertEqual(decision["reason"], "ambiguous_candidates")

    def test_same_wall_views_are_not_false_competitors(self):
        a = candidate("view_a", "1", "a", .80)
        chosen, decision = pipeline.select_candidate([a, candidate("view_b", "1", "a", .79),
                                                         candidate("different", "2", "b", .40)])
        self.assertEqual(chosen, a)
        self.assertTrue(decision["accepted"])
        self.assertFalse(decision["independent_accuracy_verified"])

    def test_failed_but_competing_evidence_blocks_overconfident_choice(self):
        chosen, _ = pipeline.select_candidate([candidate("a", "1", "a", .80),
                                               candidate("b", "2", "b", .85, False)])
        self.assertIsNone(chosen)

    def test_hash_cache_reuses_renamed_photo_but_never_changed_photo(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            photo = root / "unrelated_name.png"
            Image.new("RGB", (30, 20)).save(photo)
            cache = root / "my_results/grounding_dino16/batch_detection/D_saved/original_name"
            cache.mkdir(parents=True)
            data = {"image_sha256": pipeline.roma.sha256(photo), "image_size": [30, 20],
                    "model": detector.MODEL, "parameters": detector.experiment_parameters(),
                    "coordinate_system": "original_image_pixels_xyxy", "building_id": "must_not_be_used",
                    "detections": []}
            (cache / "detections.json").write_text(json.dumps(data), encoding="utf-8")
            with patch.object(pipeline, "ROOT", root), patch.object(pipeline, "OUTPUT_ROOT", root / "pipeline"), \
                    patch.dict("os.environ", {"DDS_API_TOKEN": ""}):
                result, source = pipeline.detection_for_photo(photo, root)
                self.assertEqual(result, data)
                provenance = pipeline.read_json(root / "detection_source.json")
                self.assertFalse(provenance["building_id_field_used"])
                Image.new("RGB", (30, 20), "red").save(photo)
                with self.assertRaisesRegex(RuntimeError, "新照片没有可复用检测"):
                    pipeline.detection_for_photo(photo, root)


if __name__ == "__main__":
    unittest.main()
