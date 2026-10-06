"""批量建立建筑渲染图库：在 PyCharm 中直接运行本文件。

读取已有 CityGML 模型，不读照片、人工标定或门窗预测。每栋生成8个基础视角，
保存RGB、深度、相机和可见墙面，供下一步图像检索与RoMa候选匹配使用。
默认两栋并行；每栋结束立即更新索引。完整且输入/参数一致的建筑可直接复用。
本文件负责批次管理，几何和渲染复用 render_building_views.py。
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
import csv
import hashlib
import json
import time

from PIL import Image, ImageDraw, ImageFont
import render_building_views as renderer

ROOT = Path(__file__).resolve().parent
DATASET = next((p for p in (ROOT / "Drills/Texture2LoD3_dataset",
                           ROOT / "CityGML-IMG_Architecture/Drills/Texture2LoD3_dataset") if p.is_dir()),
               ROOT / "Drills/Texture2LoD3_dataset")
OUTPUT_ROOT = ROOT / "my_results/render_gallery"
BUILDING_IDS = None  # None遍历所有GML；试跑可写 ["4906970", "4959323"]。
IMAGE_SIZE = (1800, 1200)
MAX_WORKERS = 2       # 内存不足可改1；每个子进程只处理一栋，结束即释放内存。
REUSE_COMPLETED = True


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    """先写临时文件再替换，避免中途停止留下半个JSON。"""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def discover_models(directory, selected=None):
    models = [(path.stem.removeprefix("DEBY_LOD3_"), path)
              for path in sorted(Path(directory).glob("DEBY_LOD3_*.gml"))]
    if selected is not None:
        selected = set(map(str, selected))
        missing = selected - {key for key, _ in models}
        if missing:
            raise ValueError("找不到这些建筑模型：" + ", ".join(sorted(missing)))
        models = [(key, path) for key, path in models if key in selected]
    if not models:
        raise ValueError("没有找到 CityGML 模型：" + str(directory))
    return models


def reusable_render(pointer, source_hash, renderer_hash, image_size):
    """不复用残缺批次、旧源模型、不同分辨率或人工指定起点的结果。"""
    try:
        directory = Path(read_json(pointer)["run_dir"])
        manifest = read_json(directory / "manifest.json")
        params = manifest.get("parameters", {})
        if (manifest["status"] != "complete" or manifest["source_sha256"] != source_hash or
                manifest["script_sha256"] != renderer_hash or manifest["front_manually_identified"] or
                params.get("image_size_wh") != list(image_size) or len(manifest["views"]) != 8):
            return None
        if not (directory / "overview.png").is_file():
            return None
        for view in manifest["views"]:
            if not all((directory / view[key]).is_file() for key in ("image", "camera", "geometry")):
                return None
            if read_json(directory / view["camera"]).get("image_size_wh") != list(image_size):
                return None
        return directory
    except (OSError, ValueError, KeyError, TypeError):
        return None


def render_one(task):
    """子进程只渲染一栋；一个模型失败不会丢掉其他建筑已完成的结果。"""
    started = time.perf_counter()
    try:
        directory = renderer.render_building(Path(task["source_gml"]), Path(task["buildings_root"]),
                                             building_id=task["building_id"], front_wall_id=None,
                                             image_size=tuple(task["image_size_wh"]))
        return {"status": "complete", "render_dir": str(directory),
                "seconds": round(time.perf_counter() - started, 3), "reused": False}
    except Exception as exc:
        return {"status": "failed", "error": str(exc), "error_type": type(exc).__name__,
                "seconds": round(time.perf_counter() - started, 3), "reused": False}


def collect_views(rows):
    """每个检索候选都带回三维所需路径；建筑ID只做模型身份，不来自照片。"""
    views = []
    for row in rows:
        if row["status"] != "complete":
            continue
        directory = Path(row["render_dir"])
        manifest = read_json(directory / "manifest.json")
        for view in manifest["views"]:
            views.append({"building_id": row["building_id"], "building_gml_id": manifest["building_gml_id"],
                          "view_name": view["name"], "orbit_deg": view["orbit_deg"],
                          "elevation_deg": view["elevation_deg"],
                          "image": str(directory / view["image"]),
                          "camera": str(directory / view["camera"]),
                          "geometry": str(directory / view["geometry"]),
                          "manifest": str(directory / "manifest.json"),
                          "source_sha256": manifest["source_sha256"],
                          "coordinate_system": manifest["coordinate_system"],
                          "world_origin_m": manifest["world_origin_m"],
                          "visible_walls": view["visible_walls"]})
    return views


def save_progress(batch_dir, rows, renderer_hash, final=False):
    counts = {state: sum(row["status"] == state for row in rows) for state in ("complete", "failed", "pending")}
    status = ("complete" if counts["complete"] == len(rows) else "partial") if final else "running"
    views = collect_views(rows)
    summary = {"status": status, "building_count": len(rows), "completed_count": counts["complete"],
               "failed_count": counts["failed"], "pending_count": counts["pending"], "view_count": len(views),
               "image_size_wh": list(IMAGE_SIZE), "renderer_sha256": renderer_hash,
               "batch_script_sha256": sha256(__file__), "buildings": rows,
               "scope": "多建筑8视角基础图库；未执行图像检索、RoMa候选验证或三维门窗写入。"}
    write_json(batch_dir / "batch_summary.json", summary)
    write_json(batch_dir / "gallery_index.json", {"schema_version": 1, "status": status,
               "view_count": len(views), "building_count": counts["complete"], "views": views,
               "manual_calibration_used": False, "photos_or_detections_used": False,
               "scope": "8个环绕方向的基础图库；复杂建筑可继续补充墙面正视图。"})
    with (batch_dir / "batch_summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        fields = ["building_id", "status", "reused", "seconds", "render_dir", "source_gml", "error"]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return summary


def make_gallery_overview(batch_dir, rows):
    """总览选可见门窗像素较多的已有视图；不改变216张图或推断照片身份。"""
    columns, card_w, card_h, pad = 4, 480, 355, 42
    nrows = (len(rows) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * card_w + pad * 2, nrows * card_h + 180), renderer.BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    def font(size):
        return ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", size)
    draw.text((pad, 28), "多建筑渲染图库", font=font(38), fill="#3F4845")
    done = sum(row["status"] == "complete" for row in rows)
    draw.text((pad, 85), f"{done}/{len(rows)} 栋完成 · 每栋8视角 · 总览展示门窗较清楚的已有视角", font=font(22), fill="#76817C")
    selected_views = []
    for i, row in enumerate(rows):
        y, x = divmod(i, columns)
        x, y = pad + x * card_w, 140 + y * card_h
        draw.text((x + 12, y), row["building_id"], font=font(24), fill="#3F4845")
        if row["status"] == "complete":
            directory = Path(row["render_dir"])
            manifest = read_json(directory / "manifest.json")
            opening_ids = {obj["index"] for obj in manifest["objects"] if obj["kind"] in ("Window", "Door")}
            def visibility(view):
                return sum(obj["pixel_count"] for obj in view["visible_objects"] if obj["object_index"] in opening_ids)
            selected = max(manifest["views"], key=visibility)
            selected_views.append({"building_id": row["building_id"], "view_name": selected["name"],
                                   "opening_pixel_count": visibility(selected), "render_dir": str(directory)})
            with Image.open(directory / selected["image"]) as source:
                image = source.convert("RGB")
            # crop只影响总览缩略图，不改变用于匹配的相机图像。
            from PIL import ImageChops
            bbox = ImageChops.difference(image, Image.new("RGB", image.size, renderer.BACKGROUND)).getbbox()
            if bbox:
                image = image.crop(bbox)
            image.thumbnail((card_w - 30, card_h - 85), Image.Resampling.LANCZOS)
            canvas.paste(image, (x + (card_w-image.width)//2, y + 45 + (card_h-85-image.height)//2))
        else:
            draw.text((x + 12, y + 130), "未完成，见批次报告", font=font(22), fill="#987867")
        draw.line((x + 12, y + card_h - 15, x + card_w - 18, y + card_h - 15), fill="#D3D5CD", width=1)
    canvas.save(batch_dir / "overview.png")
    write_json(batch_dir / "overview_metadata.json", {"selection": "largest visible window+door pixel count; display only",
               "generator_sha256": sha256(__file__), "selected_views": selected_views})


def main():
    models = discover_models(DATASET / "citygml", BUILDING_IDS)
    batch_dir = OUTPUT_ROOT / datetime.now().strftime("gallery_%Y%m%d_%H%M%S_%f")
    batch_dir.mkdir(parents=True, exist_ok=False)
    buildings_root = OUTPUT_ROOT / "buildings"
    renderer_hash = sha256(Path(renderer.__file__))
    rows, tasks = [], []
    for key, path in models:
        source_hash = sha256(path)
        row = {"building_id": key, "source_gml": str(path), "source_sha256": source_hash, "status": "pending"}
        previous = reusable_render(buildings_root / key / "latest_run.json", source_hash, renderer_hash, IMAGE_SIZE) if REUSE_COMPLETED else None
        if previous:
            row.update(status="complete", render_dir=str(previous), reused=True, seconds=0.)
        else:
            tasks.append({"building_id": key, "source_gml": str(path), "buildings_root": str(buildings_root),
                          "image_size_wh": list(IMAGE_SIZE), "size_bytes": path.stat().st_size})
        rows.append(row)
    indexed = {row["building_id"]: row for row in rows}
    save_progress(batch_dir, rows, renderer_hash)
    write_json(OUTPUT_ROOT / "latest_run.json", {"run_dir": str(batch_dir)})
    print(f"[图库] 共{len(rows)}栋，复用{len(rows)-len(tasks)}栋，新渲染{len(tasks)}栋；每栋8视角。", flush=True)
    print("[图库] 输出：", batch_dir, flush=True)
    # 大模型先启动，两进程均衡耗时；不同时把所有模型装进内存。
    with ProcessPoolExecutor(max_workers=MAX_WORKERS, max_tasks_per_child=1) as executor:
        futures = {executor.submit(render_one, task): task["building_id"]
                   for task in sorted(tasks, key=lambda t: t["size_bytes"], reverse=True)}
        for future in as_completed(futures):
            key = futures[future]
            try:
                indexed[key].update(future.result())
            except Exception as exc:
                indexed[key].update(status="failed", error=str(exc), error_type=type(exc).__name__)
            summary = save_progress(batch_dir, rows, renderer_hash)
            print(f"[图库] {key}: {indexed[key]['status']}；完成 {summary['completed_count']}/{len(rows)}，失败 {summary['failed_count']}。", flush=True)
    summary = save_progress(batch_dir, rows, renderer_hash, final=True)
    make_gallery_overview(batch_dir, rows)
    print(f"[图库] 结束：{summary['completed_count']}/{len(rows)}栋，{summary['view_count']}张。请看overview.png和batch_summary.csv。", flush=True)
    if summary["failed_count"]:
        print("[图库] 失败原因逐栋保存在batch_summary.json；修复后重跑会复用已完整成功的建筑。", flush=True)
    return summary


if __name__ == "__main__":
    main()
