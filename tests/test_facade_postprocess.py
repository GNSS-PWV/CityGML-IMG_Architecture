"""去重与实验 D 二维分块检查；不加载真实模型，不联网。"""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from facade_geometry import box_to_full_image, remove_duplicates
from try_grounding_dino16 import tile_views


def record(kind, score, box):
    return {"class": kind, "score": score, "box": box, "source": "test"}


class PostprocessChecks(unittest.TestCase):
    def test_tall_image_grid_and_vertical_boundary(self):
        views = tile_views(1700, 1500)
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
        views = tile_views(2940, 833)
        self.assertEqual(views, [("tile_1", (0, 0, 1200, 833)),
                                 ("tile_2", (900, 0, 2100, 833)),
                                 ("tile_3", (1740, 0, 2940, 833))])
        # 每列原始像素都被至少一个图块覆盖，裁剪并不等于把整图缩小。
        self.assertTrue(all(any(box[0] <= x < box[2] for _, box in views) for x in range(2940)))

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
