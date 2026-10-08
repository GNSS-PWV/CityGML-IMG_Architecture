"""实验 E：为每栋 CityGML 建筑建立“每个可映射外立面 3 视角”的独立图库。

此脚本与 build_render_gallery.py 的 8 个全局视角图库并存，绝不覆盖它。这里的
“墙”是通过单平面、竖直和面积筛选的可映射外立面，不是 CityGML 中窗台、饰线等
细碎 WallSurface。每栋完成后立即写入 gallery_index.json，失败的建筑不会混入索引。
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
import argparse
import csv
import time

import build_render_gallery as base_gallery
import render_building_views as renderer


ROOT = Path(__file__).resolve().parent
DATASET = next((path for path in (ROOT / "Drills/Texture2LoD3_dataset",
                                  ROOT / "CityGML-IMG_Architecture/Drills/Texture2LoD3_dataset") if path.is_dir()),
               ROOT / "Drills/Texture2LoD3_dataset")
OUTPUT_ROOT = ROOT / "my_results/experiment_e/wall_view_gallery"
BUILDING_IDS = None
IMAGE_SIZE = (1800, 1200)
MAX_WORKERS = 2
REUSE_COMPLETED = True


def reusable_wall_render(pointer, source_hash, renderer_hash):
    """只复用同一模型、同一渲染代码和同一实验参数的完整墙面视图。"""
    try:
        directory = Path(base_gallery.read_json(pointer)["run_dir"])
        manifest = base_gallery.read_json(directory / "manifest.json")
        params = manifest.get("parameters", {})
        expected = 3 * int(manifest["eligible_wall_count"])
        if (manifest.get("status") != "complete" or manifest.get("source_sha256") != source_hash or
                manifest.get("script_sha256") != renderer_hash or params.get("image_size_wh") != list(IMAGE_SIZE) or
                params.get("wall_view_yaws_deg") != list(renderer.WALL_VIEW_YAWS_DEG) or
                len(manifest.get("views", [])) != expected):
            return None
        for view in manifest["views"]:
            if not all((directory / view[key]).is_file() for key in ("image", "camera", "geometry")):
                return None
        return directory
    except (OSError, ValueError, KeyError, TypeError):
        return None


def render_one(task):
    started = time.perf_counter()
    try:
        directory = renderer.render_wall_views(task["source_gml"], task["buildings_root"], task["building_id"],
                                               image_size=tuple(task["image_size_wh"]))
        return {"status": "complete", "render_dir": str(directory), "reused": False,
                "seconds": round(time.perf_counter() - started, 3)}
    except Exception as exc:
        return {"status": "failed", "error_type": type(exc).__name__, "error": str(exc), "reused": False,
                "seconds": round(time.perf_counter() - started, 3)}


def collect_views(rows):
    """复用与基线相同的索引记录结构，RoMa 匹配无需知道图库来自哪种视角。"""
    return base_gallery.collect_views(rows)


def save_progress(batch_dir, rows, renderer_hash, final=False):
    counts = {name: sum(row["status"] == name for row in rows) for name in ("complete", "failed", "pending")}
    status = ("complete" if counts["complete"] == len(rows) else "partial") if final else "running"
    views = collect_views(rows)
    summary = {"schema_version": 1, "experiment": "E", "status": status, "view_strategy": "three wall-front/near-front views",
               "building_count": len(rows), "completed_count": counts["complete"], "failed_count": counts["failed"],
               "pending_count": counts["pending"], "view_count": len(views), "image_size_wh": list(IMAGE_SIZE),
               "renderer_sha256": renderer_hash, "wall_view_yaws_deg": list(renderer.WALL_VIEW_YAWS_DEG),
               "eligible_wall_rule": {"max_tilt_deg": renderer.WALL_VIEW_MAX_TILT_DEG,
                                       "min_area_m2": renderer.WALL_VIEW_MIN_AREA_M2,
                                       "relative_to_building_max_area": renderer.WALL_VIEW_RELATIVE_AREA},
               "buildings": rows,
               "scope": "Independent Experiment E gallery. Uses CityGML geometry only; no photos, detections, labels, or manual calibration."}
    base_gallery.write_json(batch_dir / "batch_summary.json", summary)
    base_gallery.write_json(batch_dir / "gallery_index.json", {"schema_version": 1, "status": status, "experiment": "E",
                           "view_strategy": "each eligible planar facade wall at yaw " + "/".join(
                               f"{yaw:g}" for yaw in renderer.WALL_VIEW_YAWS_DEG) + " degrees", "view_count": len(views),
                           "building_count": counts["complete"], "views": views, "manual_calibration_used": False,
                           "photos_or_detections_used": False, "scope": summary["scope"]})
    with (batch_dir / "batch_summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=["building_id", "status", "reused", "seconds", "render_dir", "source_gml", "error"],
                                extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    return summary


def build_gallery(building_ids=BUILDING_IDS):
    """建立或安全复用实验 E 墙面图库，返回本次批次目录。"""
    source_dir = DATASET / "citygml"
    models = base_gallery.discover_models(source_dir, building_ids)
    batch = OUTPUT_ROOT / datetime.now().strftime("wall_gallery_%Y%m%d_%H%M%S_%f")
    batch.mkdir(parents=True, exist_ok=False)
    renderer_hash = base_gallery.sha256(Path(renderer.__file__))
    rows, pending = [], []
    for building_id, source in models:
        row = {"building_id": building_id, "source_gml": str(source), "status": "pending", "reused": False}
        pointer = OUTPUT_ROOT / "buildings" / building_id / "latest_wall_views.json"
        reused = reusable_wall_render(pointer, base_gallery.sha256(source), renderer_hash) if REUSE_COMPLETED else None
        if reused is not None:
            row.update({"status": "complete", "render_dir": str(reused), "reused": True, "seconds": 0.0})
        else:
            pending.append({"building_id": building_id, "source_gml": str(source),
                            "buildings_root": str(OUTPUT_ROOT / "buildings"), "image_size_wh": list(IMAGE_SIZE)})
        rows.append(row)
    save_progress(batch, rows, renderer_hash)
    if pending:
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
            jobs = {pool.submit(render_one, task): task["building_id"] for task in pending}
            for job in as_completed(jobs):
                building_id, result = jobs[job], job.result()
                next(row for row in rows if row["building_id"] == building_id).update(result)
                save_progress(batch, rows, renderer_hash)
                print(f"{building_id}: {result['status']} ({result['seconds']:.1f}s)", flush=True)
    save_progress(batch, rows, renderer_hash, final=True)
    base_gallery.write_json(OUTPUT_ROOT / "latest_run.json", {"run_dir": str(batch)})
    print("实验 E 墙面图库：", batch, flush=True)
    return batch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--image-size", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"), default=IMAGE_SIZE,
                        help="渲染尺寸。实验记录须与实际值一致；默认 1800 1200。")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    arguments = parser.parse_args()
    if min(arguments.image_size) < 2 or arguments.workers < 1:
        raise ValueError("图像宽高至少为2，workers至少为1。")
    OUTPUT_ROOT, IMAGE_SIZE, MAX_WORKERS = arguments.output_root, tuple(arguments.image_size), arguments.workers
    build_gallery()
