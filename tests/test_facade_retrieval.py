"""检索的身份隔离、候选分组及缓存失效测试；不运行 GPU/网络或产生真实实验结果。"""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import facade_retrieval as retrieval


class RetrievalTests(unittest.TestCase):
    def test_depth_crop_ignores_nan_background_and_preserves_border(self):
        depth = np.full((10, 12), np.nan)
        depth[2:8, 3:9] = 4
        self.assertEqual(retrieval.foreground_box(depth), (2, 1, 10, 9))
        depth[:, :] = 4
        self.assertEqual(retrieval.foreground_box(depth), (0, 0, 12, 10))
        with self.assertRaises(ValueError):
            retrieval.foreground_box(np.full((4, 5), np.nan))

    def test_crop_rejects_unrelated_geometry_size(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            Image.new("RGB", (10, 8)).save(root / "image.png")
            np.savez(root / "depth.npz", depth_m=np.ones((8, 11)))
            with self.assertRaisesRegex(ValueError, "尺寸不一致"):
                retrieval.read_rgb(root / "image.png", root / "depth.npz")

    def test_candidate_quota_prevents_one_building_taking_all_slots(self):
        views = [{"building_id": key, "view_name": str(i)} for i, key in enumerate(["A", "A", "B", "C"])]
        values = [[1, 0], [.99, .01], [.8, .2], [0, 1]]
        result = retrieval.rank_descriptors([1, 0], values, views, top_k_buildings=2, views_per_building=1)
        self.assertEqual(result["ranked_building_ids"], ["A", "B"])
        self.assertEqual([v["building_id"] for v in result["selected_candidates"]], ["A", "B"])
        self.assertEqual([v["retrieval_rank"] for v in result["selected_candidates"]], [1, 3])

    def test_rejects_invalid_descriptors_instead_of_arbitrary_ranking(self):
        for array in [[[0, 0]], [[np.nan, 1]], [[np.inf, 1]]]:
            with self.assertRaises(ValueError):
                retrieval.unit_rows(array)

    def test_cache_key_changes_for_image_depth_model_and_preprocess(self):
        records = [{"image": "view.png", "image_sha256": "image1", "geometry_sha256": "depth1"}]
        model = {"checkpoint_sha256": "model1"}
        original = retrieval.cache_key(records, model, 384)[0]
        for name in ("image_sha256", "geometry_sha256"):
            changed = [dict(records[0], **{name: "different"})]
            self.assertNotEqual(original, retrieval.cache_key(changed, model, 384)[0])
        self.assertNotEqual(original, retrieval.cache_key(records, {"checkpoint_sha256": "model2"}, 384)[0])
        self.assertNotEqual(original, retrieval.cache_key(records, model, 256)[0])
        with patch.object(retrieval, "PREPROCESS_VERSION", "different"):
            self.assertNotEqual(original, retrieval.cache_key(records, model, 384)[0])

    def test_query_rename_does_not_use_filename_id_and_gallery_cache_reuses(self):
        class FakeEncoder:
            def __init__(self):
                self.sizes = []
            def __call__(self, images):
                self.sizes.append(len(images))
                return np.asarray([np.asarray(im, dtype=float).mean(axis=(0, 1)) for im in images])

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            views = []
            for key, color in [("real_red", (240, 10, 10)), ("wrong_green", (10, 240, 10))]:
                Image.new("RGB", (12, 10), color).save(root / (key + ".png"))
                np.savez(root / (key + ".npz"), depth_m=np.ones((10, 12)))
                views.append({"building_id": key, "view_name": "view_000", "image": key + ".png", "geometry": key + ".npz"})
            index = root / "index.json"
            index.write_text(json.dumps({"status": "complete", "views": views}), encoding="utf-8")
            query = root / "wrong_green_front.png"
            Image.new("RGB", (12, 10), (240, 10, 10)).save(query)
            encoder = FakeEncoder()
            kwargs = dict(gallery_index_path=index, cache_dir=root / "cache", model_cache=root,
                          encoder=encoder, progress=None, top_k_buildings=1)
            with patch.object(retrieval, "model_identity", return_value={"test_only": True}):
                first = retrieval.retrieve_gallery(query, **kwargs)
                renamed = root / "another_name.png"
                renamed.write_bytes(query.read_bytes())
                second = retrieval.retrieve_gallery(renamed, **kwargs)
                self.assertEqual(first["ranked_building_ids"], ["real_red"])
                self.assertEqual(first["ranked_views"], second["ranked_views"])
                self.assertFalse(first["ranking_uses_query_filename"])
                self.assertFalse(first["cache_hit"])
                self.assertTrue(second["cache_hit"])
                self.assertEqual(encoder.sizes, [2, 1, 1])
                # 改动渲染内容后，必须重提图库特征而不是用同名旧向量。
                Image.new("RGB", (12, 10), (20, 220, 10)).save(root / "real_red.png")
                third = retrieval.retrieve_gallery(renamed, **kwargs)
                self.assertFalse(third["cache_hit"])


if __name__ == "__main__":
    unittest.main()
