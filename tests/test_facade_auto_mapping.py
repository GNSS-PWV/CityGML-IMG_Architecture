"""自动三维映射的合成几何测试：不联网、不调用GPU、不使用真实人工点。"""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import facade_auto_mapping as mapping
from map_facade_to_3d import file_hash, load_walls
from render_building_views import make_camera


class AutoMappingTests(unittest.TestCase):
    def fixture(self, folder, polygon=None, nonplanar=False):
        root = Path(folder)
        origin = np.array([691000., 5336000., 500.])
        polygon = polygon or [[0, 0], [10, 0], [10, 10], [0, 10]]
        points = origin+np.array([[x, 0, z] for x, z in polygon+[polygon[0]]])
        if nonplanar:
            points[2, 1] += .5
        other = origin+np.array([[0, 3, 0], [10, 3, 0], [10, 3, 10], [0, 3, 10], [0, 3, 0]])
        source = root/"unrelated_source_name.gml"
        surfaces = []
        for wall_id, ring in [("wall-A", points), ("wall-B", other)]:
            pos = " ".join(str(value) for value in ring.ravel())
            surfaces.append('<b:boundedBy><b:WallSurface g:id="'+wall_id+'"><b:lod3MultiSurface>'
                            '<g:MultiSurface><g:surfaceMember><g:Polygon><g:exterior><g:LinearRing>'
                            '<g:posList>'+pos+'</g:posList></g:LinearRing></g:exterior></g:Polygon>'
                            '</g:surfaceMember></g:MultiSurface></b:lod3MultiSurface></b:WallSurface></b:boundedBy>')
        source.write_text('<core:CityModel xmlns:core="http://www.opengis.net/citygml/2.0" '
                          'xmlns:g="http://www.opengis.net/gml" xmlns:b="http://www.opengis.net/citygml/building/2.0">'
                          '<core:cityObjectMember><b:Building g:id="building-A">'+"".join(surfaces)+
                          '</b:Building></core:cityObjectMember></core:CityModel>', encoding="utf8")
        # 相机沿+Y看墙，深度20米处为主墙Y=0；左上像素(10,10)对应墙(0,0,10)。
        inverse = np.array([[.1, 0, 0, -1], [0, 0, 1, -20], [0, -.1, 0, 11], [0, 0, 0, 1.]])
        camera = {"projection": "orthographic", "image_size_wh": [121, 121],
                  "pixel_depth_to_local": inverse.tolist(), "local_to_pixel_depth": np.linalg.inv(inverse).tolist(),
                  "pixels_per_meter": 10.}
        depth = np.full((121, 121), np.nan, dtype=np.float32)
        objects = np.full((121, 121), -1, dtype=np.int32)
        depth[10:111, 10:111] = 20.
        objects[10:111, 10:111] = 0
        depth[20:41, 20:41] = 18.5  # 窗面凸出1.5米，映射仍应回到主墙。
        objects[20:41, 20:41] = 1
        depth[80:91, 80:91] = 23.
        objects[80:91, 80:91] = 2
        manifest = {"building_id": "selected-building-A", "building_gml_id": "building-A",
                    "source_gml": str(source), "source_sha256": file_hash(source),
                    "world_origin_m": origin.tolist(), "coordinate_system": "EPSG:25832",
                    "objects": [{"index": 0, "kind": "WallSurface", "id": "wall-A", "wall_id": "wall-A"},
                                {"index": 1, "kind": "Window", "id": "window-A", "wall_id": "wall-A"},
                                {"index": 2, "kind": "WallSurface", "id": "wall-B", "wall_id": "wall-B"}]}
        detections = {"image_size": [121, 121], "image_sha256": "synthetic-photo",
                      "image": "a_wrong_building_id_in_filename.jpg", "building_id": "wrong-building-B",
                      "coordinate_system": "original_image_pixels_xyxy", "detections": []}
        return detections, camera, {"depth_m": depth, "object_index": objects}, manifest

    def test_pixel_rays_recover_same_world_plane_from_two_views(self):
        with tempfile.TemporaryDirectory() as folder:
            _, _, _, manifest = self.fixture(folder)
            frame = next(w["frame"] for w in load_walls(Path(manifest["source_gml"])) if w["id"] == "wall-A")
            origin = np.asarray(manifest["world_origin_m"])
            targets = origin+np.array([[1, 0, 2], [5, 0, 5], [8, 0, 9.]])
            context = np.array([[0, 0, 0], [10, 0, 0], [10, 3, 10], [0, 3, 10]])
            recovered = []
            for direction in ([0, -1, 0], [.7, -1, .2]):
                camera = make_camera(context, direction, image_size=(300, 220))
                matrix = np.asarray(camera["local_to_pixel_depth"])
                projected = (targets-origin)@matrix[:3, :3].T+matrix[:3, 3]
                xyz, uv, axis_depth = mapping.render_pixels_to_wall(projected[:, :2], camera, frame, origin)
                np.testing.assert_allclose(xyz, targets, atol=1e-8)
                np.testing.assert_allclose(axis_depth, projected[:, 2], atol=1e-8)
                recovered.append(xyz)
            np.testing.assert_allclose(recovered[0], recovered[1], atol=1e-8)

    def test_nearly_parallel_view_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            _, camera, _, manifest = self.fixture(folder)
            frame = next(w["frame"] for w in load_walls(Path(manifest["source_gml"])) if w["id"] == "wall-A")
            inverse = np.eye(4)
            inverse[:3, 2] = [1, 0, 0]
            camera["pixel_depth_to_local"] = inverse.tolist()
            with self.assertRaisesRegex(ValueError, "平行"):
                mapping.render_pixels_to_wall([[20, 20]], camera, frame, manifest["world_origin_m"])

    def test_wall_vote_uses_window_ancestor_and_rejects_tied_walls(self):
        with tempfile.TemporaryDirectory() as folder:
            _, _, geometry, manifest = self.fixture(folder)
            vote = mapping.choose_wall_from_matches([[25, 25], [30, 30], [35, 35], [50, 50], [85, 85]],
                                                    geometry, manifest, min_votes=3)
            self.assertTrue(vote["accepted"])
            self.assertEqual(vote["wall_id"], "wall-A")
            self.assertEqual(vote["votes"][0]["count"], 4)
            tied = mapping.choose_wall_from_matches([[25, 25], [50, 50], [83, 83], [87, 87]],
                                                    geometry, manifest, min_votes=2)
            self.assertFalse(tied["accepted"])
            self.assertIsNone(tied["wall_id"])
            mostly_background = mapping.choose_wall_from_matches([[25, 25], [30, 30]]+[[0, 0]]*10,
                                                                 geometry, manifest, min_votes=2)
            self.assertFalse(mostly_background["accepted"])

    def test_mapping_preserves_classes_projects_to_plane_and_rejects_unsafe_boxes(self):
        with tempfile.TemporaryDirectory() as folder:
            detections, camera, geometry, manifest = self.fixture(folder)
            detections["detections"] = [
                {"class": "window", "score": .8, "box": [20, 20, 40, 40]},
                {"class": "ambiguous", "score": .7, "box": [50, 50, 60, 70],
                 "candidates": [{"class": "window", "score": .7}, {"class": "door", "score": .6}]},
                {"class": "door", "score": .75, "box": [63, 50, 75, 75]},
                {"class": "window", "score": .5, "box": [109, 50, 118, 70]},
                {"class": "window", "score": .5, "box": [80, 80, 90, 90]},
                {"class": "window", "score": .5, "box": [0, 20, 15, 40]},
                {"class": "window", "score": .5, "box": [20, 20, 20, 40]},
            ]
            before = copy.deepcopy(detections)
            output = Path(folder)/"result"
            result = mapping.map_detections(detections, np.eye(3), camera, geometry, manifest, "wall-A", output)
            self.assertEqual(result["mapped_count"], 3)
            self.assertEqual(result["rejected_count"], 4)
            self.assertEqual(result["building_id"], "selected-building-A")
            self.assertEqual(result["predictions"][1]["class"], "ambiguous")
            self.assertEqual(len(result["predictions"][1]["detection"]["candidates"]), 2)
            self.assertEqual(result["mapped_class_counts"], {"window": 1, "ambiguous": 1, "door": 1})
            first = result["predictions"][0]
            np.testing.assert_allclose(np.array(first["vertices_xyz"])[:, 1], 5336000., atol=1e-9)
            np.testing.assert_allclose(first["camera_axis_depth_m"], 20., atol=1e-9)
            self.assertAlmostEqual(first["width_m"], 2.)
            self.assertAlmostEqual(first["height_m"], 2.)
            reasons = [record["reason"] for record in result["rejected"]]
            self.assertIn("mapped_box_not_fully_inside_wall_outer_boundary", reasons)
            self.assertIn("mapped_box_not_visibly_supported_by_selected_wall", reasons)
            self.assertIn("box_touches_photo_border_may_be_incomplete", reasons)
            self.assertEqual(detections, before)
            self.assertEqual(file_hash(Path(manifest["source_gml"])), manifest["source_sha256"])
            self.assertFalse(result["source_gml_modified"])
            for path in result["files"].values():
                self.assertTrue(Path(path).is_file())
            with Image.open(result["files"]["preview"]) as preview:
                self.assertEqual(preview.size, (2400, 1050))
            obj = Path(result["files"]["obj"]).read_text(encoding="utf8")
            self.assertIn("usemtl ambiguous", obj)
            vertices = [list(map(float, line.split()[1:])) for line in obj.splitlines() if line.startswith("v ")]
            self.assertEqual(len(vertices), 12)
            self.assertLess(np.max(np.abs(vertices)), 20.)
            stored = json.loads(Path(result["files"]["json"]).read_text(encoding="utf8"))
            self.assertEqual(stored["mapped_count"], 3)

    def test_concave_wall_gap_and_projective_pole_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(mapping, "_save_preview", return_value=[]):
            polygon = [[0, 0], [10, 0], [10, 10], [6, 10], [6, 3], [4, 3], [4, 10], [0, 10]]
            detections, camera, geometry, manifest = self.fixture(folder, polygon=polygon)
            detections["detections"] = [{"class": "window", "score": .5, "box": [30, 30, 90, 90]}]
            result = mapping.map_detections(detections, np.eye(3), camera, geometry, manifest, "wall-A", Path(folder)/"concave")
            self.assertEqual(result["mapped_count"], 0)
            self.assertEqual(result["status"], "no_detections_mapped")
            self.assertEqual(result["rejected"][0]["reason"], "mapped_box_not_fully_inside_wall_outer_boundary")
            H = np.array([[1., 0, 0], [0, 1, 0], [1, 0, -60]])
            result = mapping.map_detections(detections, H, camera, geometry, manifest, "wall-A", Path(folder)/"pole")
            self.assertEqual(result["rejected"][0]["reason"], "projective_pole_crosses_detection")

    def test_one_window_crossing_two_triangle_patches_of_same_wall_is_preserved(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(mapping, "_save_preview", return_value=[]):
            detections, camera, geometry, manifest = self.fixture(folder)
            walls = load_walls(Path(manifest["source_gml"]))
            wall = next(w for w in walls if w["id"] == "wall-A")
            ring = wall["uv"][0]
            wall["uv"] = [ring[[0, 1, 2]], ring[[0, 2, 3]]]
            detections["detections"] = [{"class": "window", "score": .8, "box": [40, 40, 70, 70]}]
            # 两个三角形共享一条对角线；框跨线，但完整位于两片合起来的墙内。
            with patch.object(mapping, "load_walls", return_value=walls):
                result = mapping.map_detections(detections, np.eye(3), camera, geometry, manifest,
                                                "wall-A", Path(folder)/"seam")
            self.assertEqual(result["mapped_count"], 1)
            self.assertEqual(result["rejected_count"], 0)
            self.assertEqual(result["wall_boundary"]["source_outer_ring_count"], 2)
            self.assertEqual(result["wall_boundary"]["merged_outer_ring_count"], 1)
            self.assertEqual(len(result["wall_boundary"]["wall_uv_outer_rings_m"][0]), 4)
            uv = result["predictions"][0]["wall_uv_m"]
            self.assertFalse(any(mapping.polygon_inside_ring(uv, triangle, .03) for triangle in wall["uv"]))

    def test_union_uses_exteriors_but_keeps_disconnected_components_and_concave_notches(self):
        # 四块墙片围出封闭内部孔；预测框依据整体外边界，不依据模型旧孔洞。
        def rectangle(x0, y0, x1, y1):
            return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)
        wall = {"uv": [rectangle(0, 0, 4, 1), rectangle(0, 3, 4, 4),
                       rectangle(0, 1, 1, 3), rectangle(3, 1, 4, 3)]}
        rings = mapping._wall_outer_rings(wall)
        self.assertEqual(len(rings), 1)
        self.assertTrue(mapping.polygon_inside_ring(rectangle(1.5, 1.5, 2.5, 2.5), rings[0], .03))
        wall["uv"] = [rectangle(0, 0, 1, 4), rectangle(3, 0, 4, 4)]
        rings = mapping._wall_outer_rings(wall)
        self.assertEqual(len(rings), 2)
        self.assertFalse(any(mapping.polygon_inside_ring(rectangle(.5, 1, 3.5, 2), r, .03) for r in rings))
        # U形缺口连通外界，union不能把它扩为包围盒或凸包。
        wall["uv"].append(rectangle(0, 0, 4, 1))
        rings = mapping._wall_outer_rings(wall)
        self.assertEqual(len(rings), 1)
        self.assertFalse(mapping.polygon_inside_ring(rectangle(.5, 1.5, 3.5, 2.5), rings[0], .03))

    def test_changed_model_and_nonplanar_selected_wall_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            detections, camera, geometry, manifest = self.fixture(folder)
            changed = dict(manifest, source_sha256="wrong-hash")
            with self.assertRaisesRegex(ValueError, "模型已变化"):
                mapping.map_detections(detections, np.eye(3), camera, geometry, changed, "wall-A", Path(folder)/"bad")
        with tempfile.TemporaryDirectory() as folder:
            detections, camera, geometry, manifest = self.fixture(folder, nonplanar=True)
            with self.assertRaisesRegex(ValueError, "不是单一平面"):
                mapping.map_detections(detections, np.eye(3), camera, geometry, manifest, "wall-A", Path(folder)/"bad")

    def test_invalid_unselected_context_wall_does_not_abort_valid_mapping(self):
        with tempfile.TemporaryDirectory() as folder:
            detections, camera, geometry, manifest = self.fixture(folder)
            walls = load_walls(Path(manifest["source_gml"]))
            other = next(w for w in walls if w["id"] == "wall-B")
            # 背景参考墙自交；选中的A墙仍完全合法，绘图不能让整个映射失败。
            other["uv"] = [other["uv"][0][[0, 2, 1, 3]]]
            other["rings"] = [other["rings"][0][[0, 2, 1, 3]]]
            detections["detections"] = [{"class": "window", "score": .8, "box": [20, 20, 40, 40]}]
            with patch.object(mapping, "load_walls", return_value=walls):
                report = mapping.map_detections(detections, np.eye(3), camera, geometry, manifest,
                                               "wall-A", Path(folder)/"valid_selected_wall")
            self.assertEqual(report["mapped_count"], 1)
            self.assertEqual(report["preview_context_warnings"][0]["wall_id"], "wall-B")
            self.assertEqual(report["preview_context_warnings"][0]["action"], "raw_3d_outline_only")
            self.assertTrue(Path(report["files"]["preview"]).is_file())
            self.assertEqual(file_hash(Path(manifest["source_gml"])), manifest["source_sha256"])

    def test_validated_wall_rejects_nonplanar_missing_and_invalid_selected_boundaries(self):
        with tempfile.TemporaryDirectory() as folder:
            _, _, _, manifest = self.fixture(folder, nonplanar=True)
            with self.assertRaisesRegex(ValueError, "不是单一平面.*wall-A"):
                mapping.validated_wall(manifest, "wall-A")
            with self.assertRaisesRegex(ValueError, "不存在.*missing-wall"):
                mapping.validated_wall(manifest, "missing-wall")
        with tempfile.TemporaryDirectory() as folder:
            detections, camera, geometry, manifest = self.fixture(folder)
            wall = mapping.validated_wall(manifest, "wall-A")
            self.assertIn("frame", wall)
            self.assertEqual(len(wall["outer_uv"]), 1)
            walls = load_walls(Path(manifest["source_gml"]))
            selected = next(w for w in walls if w["id"] == "wall-A")
            selected["uv"] = [selected["uv"][0][[0, 2, 1, 3]]]
            with patch.object(mapping, "load_walls", return_value=walls):
                with self.assertRaisesRegex(ValueError, "目标墙边界无效.*wall-A"):
                    mapping.validated_wall(manifest, "wall-A")
                with self.assertRaisesRegex(ValueError, "目标墙边界无效"):
                    mapping.map_detections(detections, np.eye(3), camera, geometry, manifest,
                                           "wall-A", Path(folder)/"invalid_selected_wall")
            self.assertFalse((Path(folder)/"invalid_selected_wall").exists())

    def test_geometry_path_must_use_its_own_camera(self):
        with tempfile.TemporaryDirectory() as folder:
            detections, camera, geometry, manifest = self.fixture(folder)
            root = Path(folder)
            path = root/"view_000_geometry.npz"
            np.savez_compressed(path, **geometry)
            (root/"view_000_camera.json").write_text(json.dumps(camera), encoding="utf8")
            manifest["views"] = [{"name": "view_000", "geometry": path.name, "camera": "view_000_camera.json"}]
            changed = dict(camera, pixels_per_meter=11.)
            with self.assertRaisesRegex(ValueError, "对应的视图相机不同"):
                mapping.map_detections(detections, np.eye(3), changed, path, manifest, "wall-A", root/"bad")

    def test_invalid_depth_object_mask_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            _, _, geometry, manifest = self.fixture(folder)
            geometry["depth_m"][25, 25] = np.nan
            with self.assertRaisesRegex(ValueError, "掩码不一致"):
                mapping.choose_wall_from_matches([[25, 25]], geometry, manifest)


if __name__ == "__main__":
    unittest.main()
