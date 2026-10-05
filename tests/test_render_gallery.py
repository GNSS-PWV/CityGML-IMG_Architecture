"""批量图库的完整性：不能把失败、残缺或不同参数的渲染混进索引。"""
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

import build_render_gallery as gallery


class RenderGalleryTests(unittest.TestCase):
    def make_render(self, root):
        directory = root / "render"
        directory.mkdir()
        manifest = {"status": "complete", "source_sha256": "source", "script_sha256": "renderer",
                    "front_manually_identified": False, "parameters": {"image_size_wh": [64, 48]},
                    "building_gml_id": "gml-building", "coordinate_system": "EPSG:25832",
                    "world_origin_m": [1, 2, 3], "views": []}
        for i in range(8):
            view = {"name": f"view_{i}", "image": f"{i}.png", "camera": f"{i}.json", "geometry": f"{i}.npz",
                    "orbit_deg": i * 45, "elevation_deg": 0,
                    "visible_walls": [{"wall_id": f"wall_{i}", "pixel_count": 100, "object_indices": [i]}]}
            (directory / view["image"]).write_bytes(b"test-payload")
            (directory / view["geometry"]).write_bytes(b"test-payload")
            gallery.write_json(directory / view["camera"], {"image_size_wh": [64, 48]})
            manifest["views"].append(view)
        (directory / "overview.png").write_bytes(b"test-payload")
        gallery.write_json(directory / "manifest.json", manifest)
        pointer = root / "latest_run.json"
        gallery.write_json(pointer, {"run_dir": str(directory)})
        return directory, pointer, manifest

    def test_missing_depth_or_changed_inputs_are_never_reused(self):
        with tempfile.TemporaryDirectory() as folder:
            directory, pointer, _ = self.make_render(Path(folder))
            self.assertEqual(gallery.reusable_render(pointer, "source", "renderer", (64, 48)), directory)
            self.assertIsNone(gallery.reusable_render(pointer, "changed-model", "renderer", (64, 48)))
            self.assertIsNone(gallery.reusable_render(pointer, "source", "new-renderer", (64, 48)))
            self.assertIsNone(gallery.reusable_render(pointer, "source", "renderer", (128, 96)))
            (directory / "7.npz").unlink()
            self.assertIsNone(gallery.reusable_render(pointer, "source", "renderer", (64, 48)))

    def test_manual_direction_or_failed_render_cannot_pass_as_auto_complete(self):
        with tempfile.TemporaryDirectory() as folder:
            directory, pointer, manifest = self.make_render(Path(folder))
            for state, manual in [("failed", False), ("running", False), ("complete", True)]:
                with self.subTest(state=state, manual=manual):
                    manifest.update(status=state, front_manually_identified=manual)
                    gallery.write_json(directory / "manifest.json", manifest)
                    self.assertIsNone(gallery.reusable_render(pointer, "source", "renderer", (64, 48)))

    def test_index_keeps_each_view_wall_identity_and_excludes_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            directory, _, _ = self.make_render(Path(folder))
            views = gallery.collect_views([{"building_id": "building-A", "status": "complete", "render_dir": str(directory)},
                                           {"building_id": "building-B", "status": "failed"}])
            self.assertEqual(len(views), 8)
            self.assertEqual({v["building_id"] for v in views}, {"building-A"})
            self.assertEqual(views[3]["visible_walls"][0]["wall_id"], "wall_3")
            self.assertEqual(views[3]["world_origin_m"], [1, 2, 3])

    def test_worker_records_geometry_failure_without_success_path(self):
        task = {"source_gml": "bad.gml", "building_id": "bad", "buildings_root": ".", "image_size_wh": [64, 48]}
        with patch.object(gallery.renderer, "render_building", side_effect=ValueError("nonplanar source polygon")):
            result = gallery.render_one(task)
        self.assertEqual(result["status"], "failed")
        self.assertNotIn("render_dir", result)
        self.assertIn("nonplanar", result["error"])


if __name__ == "__main__":
    unittest.main()
