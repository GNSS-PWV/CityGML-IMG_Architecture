"""纯几何渲染检查：不访问网络、不读真实照片，也不生成正式渲染结果。"""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import render_building_views as renderer


def project(points, camera):
    matrix = np.asarray(camera["local_to_pixel_depth"])
    return np.asarray(points) @ matrix[:3, :3].T + matrix[:3, 3]


def unproject(pixel_depth, camera):
    matrix = np.asarray(camera["pixel_depth_to_local"])
    return np.asarray(pixel_depth) @ matrix[:3, :3].T + matrix[:3, 3]


class BuildingRenderTests(unittest.TestCase):
    def test_boundary_touching_opening_is_repaired_as_cutout(self):
        # 门洞与墙体下边界相接（或略微越界）时，并非合法的内部孔环。
        # 应将其转成外轮廓凹口，而不是填上整个洞或改变源坐标。
        for lower_edge in [0., -.1]:
            with self.subTest(lower_edge=lower_edge):
                outer = np.array([[0., 0., 0.], [6., 0., 0.], [6., 4., 0.],
                                  [0., 4., 0.], [0., 0., 0.]])
                hole = np.array([[2., lower_edge, 0.], [2., 2., 0.], [4., 2., 0.],
                                 [4., lower_edge, 0.], [2., lower_edge, 0.]])
                rings = [outer, hole]
                before = [ring.copy() for ring in rings]
                repair = {}
                triangles, _, _ = renderer.triangulate_rings(rings, repair)
                area = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                               triangles[:, 2] - triangles[:, 0]), axis=1).sum() / 2
                self.assertAlmostEqual(area, 20.)
                self.assertAlmostEqual(repair["repaired_area_m2"], 20.)
                self.assertTrue(repair["reason"])
                self.assertIn("original_algebraic_area_m2", repair)
                for source, original in zip(rings, before):
                    np.testing.assert_array_equal(source, original)

                raster_triangles = triangles.copy()
                raster_triangles[:, :, :2] = triangles[:, :, :2] * 20 + 5
                raster_triangles[:, :, 2] = 2.
                camera = {"image_size_wh": [131, 91], "local_to_pixel_depth": np.eye(4).tolist()}
                colors = np.tile([100, 120, 110], (len(triangles), 1))
                _, depth, object_map = renderer.rasterize(raster_triangles, colors,
                                                         np.zeros(len(triangles), dtype=int), camera)
                self.assertTrue(np.isnan(depth[25, 65]))  # 凹口内 (3,1)。
                self.assertEqual(object_map[25, 65], -1)
                self.assertEqual(object_map[25, 25], 0)   # 墙体 (1,1)。
                self.assertEqual(object_map[65, 65], 0)   # 门洞上方墙体 (3,3)。

    def test_concave_polygon_preserves_window_hole_and_area(self):
        # 外环面积 22，孔洞面积 1。把它旋转到三维斜平面，避免只验证 XY 特例。
        outer = np.array([[0, 0], [6, 0], [6, 5], [4, 5], [4, 3], [0, 3], [0, 0]])
        hole = np.array([[1, 1], [1, 2], [2, 2], [2, 1], [1, 1]])
        basis = np.array([[.6, .8, 0.], [0., 0., 1.]])
        offset = np.array([7., -3., 12.])
        triangles, normal, deviation = renderer.triangulate_rings(
            [outer @ basis + offset, hole @ basis + offset])
        area = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                       triangles[:, 2] - triangles[:, 0]), axis=1).sum() / 2
        self.assertAlmostEqual(area, 21.)
        self.assertAlmostEqual(np.linalg.norm(normal), 1.)
        self.assertLess(deviation, 1e-12)

        # 将三角化结果真正光栅化：窗洞、凹口应透明，不能靠面积巧合抵消错误。
        uv = (triangles - offset) @ basis.T
        raster_triangles = np.concatenate((uv * 20 + 5, np.full((*uv.shape[:2], 1), 2.)), axis=2)
        camera = {"image_size_wh": [131, 111], "local_to_pixel_depth": np.eye(4).tolist()}
        colors = np.tile([100, 120, 110], (len(triangles), 1))
        _, depth, object_map = renderer.rasterize(raster_triangles, colors,
                                                 np.zeros(len(triangles), dtype=int), camera)
        for x, y in [(1.25, 1.25), (1.75, 1.75), (1.25, 1.75), (1.75, 1.25), (2., 4.)]:
            px, py = np.rint(np.array([x, y]) * 20 + 5).astype(int)
            self.assertTrue(np.isnan(depth[py, px]))
            self.assertEqual(object_map[py, px], -1)
        for x, y in [(.5, .5), (3., 2.), (5., 4.)]:
            px, py = np.rint(np.array([x, y]) * 20 + 5).astype(int)
            self.assertEqual(object_map[py, px], 0)

    def test_crossing_surfaces_use_per_pixel_depth_independent_of_order(self):
        # 两个面在画面内相交：A 左边近、右边远，B 深度固定。
        # 仅按整个面的中心排序，无法同时正确显示两边。
        triangles = np.array([[[2., 2., 2.], [18., 2., 6.], [2., 18., 2.]],
                              [[2., 2., 4.], [18., 2., 4.], [2., 18., 4.]]])
        colors = np.array([[200, 40, 60], [30, 160, 100]], dtype=np.uint8)
        ids = np.array([7, 19])
        camera = {"image_size_wh": [21, 21], "local_to_pixel_depth": np.eye(4).tolist()}
        rgb, depth, object_map = renderer.rasterize(triangles, colors, ids, camera)
        reverse_rgb, reverse_depth, reverse_objects = renderer.rasterize(
            triangles[::-1], colors[::-1], ids[::-1], camera)
        # 深度恰好相等处允许先画的面保留；非共面交线外必须与绘制顺序无关。
        non_tie = np.indices(depth.shape)[1] != 10
        np.testing.assert_array_equal(rgb[non_tie], reverse_rgb[non_tie])
        np.testing.assert_allclose(depth, reverse_depth, equal_nan=True)
        np.testing.assert_array_equal(object_map[non_tie], reverse_objects[non_tie])
        self.assertEqual(object_map[4, 4], 7)
        self.assertAlmostEqual(float(depth[4, 4]), 2.5)
        self.assertEqual(object_map[3, 13], 19)
        self.assertAlmostEqual(float(depth[3, 13]), 4.)
        self.assertEqual(object_map[20, 20], -1)
        self.assertTrue(np.isnan(depth[20, 20]))

    def test_orthographic_roundtrip_includes_large_world_origin(self):
        bounds = np.array([[x, y, z] for x in [-8., 8.] for y in [-4., 4.] for z in [-6., 6.]])
        camera = renderer.make_camera(bounds, [2., -3., 1.2], image_size=(301, 201), margin=.08)
        origin = np.array([691064., 5336062., 514.59])
        local = np.random.default_rng(51).uniform(-1, 1, (25, 3)) * [8., 4., 6.]
        pixels = project(local, camera)
        world = unproject(pixels, camera) + origin
        np.testing.assert_allclose(world, local + origin, rtol=0, atol=1e-8)
        self.assertTrue(np.all(pixels[:, 2] > 0))
        corners = project(bounds, camera)
        self.assertGreaterEqual(float(corners[:, 0].min()), .08 * 300 - 1e-10)
        self.assertLessEqual(float(corners[:, 0].max()), .92 * 300 + 1e-10)
        self.assertGreaterEqual(float(corners[:, 1].min()), .08 * 200 - 1e-10)
        self.assertLessEqual(float(corners[:, 1].max()), .92 * 200 + 1e-10)

    def test_camera_eye_agrees_with_principal_pixel_for_asymmetric_model(self):
        # 实际建筑相对于包围盒中心并不对称。eye 元数据必须描述图像中心射线，
        # 这样后续改用标准渲染相机也不会产生横向偏移。
        points = np.array([[0., 0., 0.], [10., 0., 0.], [0., 4., 3.]])
        camera = renderer.make_camera(points, [1., -1., 1.], image_size=(301, 201))
        camera_center = unproject([150., 100., 0.], camera)
        np.testing.assert_allclose(camera_center, camera["eye_local_m"], atol=1e-10)

    def test_rendered_depth_backprojects_to_original_slanted_plane(self):
        triangles = np.array([[[-3., -2., -1.], [3., -2., 2.], [-3., 2., 1.]]])
        camera = renderer.make_camera(triangles.reshape(-1, 3), [1., -2., 3.], image_size=(101, 81))
        _, depth, _ = renderer.rasterize(triangles, np.array([[100, 120, 110]]), np.array([3]), camera)
        yy, xx = np.where(np.isfinite(depth))
        self.assertGreater(len(xx), 100)
        restored = unproject(np.column_stack((xx, yy, depth[yy, xx])), camera)
        normal = np.cross(triangles[0, 1] - triangles[0, 0], triangles[0, 2] - triangles[0, 0])
        normal /= np.linalg.norm(normal)
        distances = (restored - triangles[0, 0]) @ normal
        # npz 保存 float32 深度，因此允许几微米的回投误差。
        self.assertLess(float(np.max(np.abs(distances))), 3e-6)

    def test_front_right_views_agree_with_image_right_and_world_up(self):
        front = np.array([.6, -.8, 0.])
        views = list(renderer.view_definitions(front))
        indexed = {view["name"]: view for view in views}
        bounds = np.array([[-2., -3., -1.], [2., 3., 1.]])
        camera = renderer.make_camera(bounds, front, image_size=(101, 81))
        image_right = np.asarray(camera["image_right_world"])
        np.testing.assert_allclose(indexed["right"]["eye_direction"], image_right, atol=1e-12)
        np.testing.assert_allclose(indexed["back"]["eye_direction"], -front, atol=1e-12)
        np.testing.assert_allclose(indexed["left"]["eye_direction"], -image_right, atol=1e-12)
        central_pixel = project(np.zeros(3), camera)
        self.assertGreater(project(image_right, camera)[0], central_pixel[0])
        self.assertLess(project([0., 0., 1.], camera)[1], central_pixel[1])
        front_right = indexed["front_right"]["eye_direction"]
        self.assertGreater(float(front_right @ front), 0.)
        self.assertGreater(float(front_right @ image_right), 0.)
        self.assertGreater(float(front_right[2]), 0.)
        self.assertEqual(len(views), 8)
        self.assertEqual(len({view["name"] for view in views}), 8)


if __name__ == "__main__":
    unittest.main()
