"""0°对照实验的公平性与事后计数边界。"""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_experiment_front0 as experiment


class FrontZeroTests(unittest.TestCase):
    def test_gallery_filters_identical_zero_degree_views_for_both_models(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "source"
            source.mkdir()
            views = [{"building_id": "1", "orbit_deg": angle, "view_name": str(angle)}
                     for angle in (-45, 0, 45)]
            experiment.write_json(source / "gallery_index.json", {
                "status": "complete", "views": views, "view_count": 3, "building_count": 1})
            pointer = root / "latest_run.json"
            experiment.write_json(pointer, {"run_dir": str(source)})
            with patch.object(experiment, "SOURCE_GALLERY_POINTER", pointer), patch.object(experiment, "OUTPUT", root / "out"):
                first, data = experiment.gallery_index()
                second, again = experiment.gallery_index()
            self.assertEqual(first, second)
            self.assertEqual(data, again)
            self.assertEqual([v["orbit_deg"] for v in data["views"]], [0])

    def test_wrong_building_only_measured_after_mapping(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            experiment.write_json(run / "retrieval.json", {"selected_candidates": [{"building_id": "123"}]})
            experiment.write_json(run / "candidates.json", [
                {"building_id": "123", "roma_validation": {"gate_passed": True}, "gate_passed": True}])
            experiment.write_json(run / "result.json", {"status": "needs_review", "selection": {"reason": "ambiguous"}})
            photo = Path("123_front.jpg")
            pending = experiment.diagnose(1, photo, run)
            self.assertTrue(pending["correct_building_entered"])
            self.assertTrue(pending["correct_building_geometry_passed"])
            self.assertFalse(pending["wrong_building_mapped"])
            experiment.write_json(run / "result.json", {"status": "mapped", "building_id": "999",
                         "mapping": {"mapped_count": 2}, "selection": {"reason": "automatic_checks_passed"}})
            mapped = experiment.diagnose(1, photo, run)
            self.assertTrue(mapped["wrong_building_mapped"])
            self.assertEqual(mapped["mapped_openings"], 2)


if __name__ == "__main__":
    unittest.main()
