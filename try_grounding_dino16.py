"""实验 D：Grounding DINO 1.6 Pro 批量立面照片分块检测，PyCharm 直接运行。

准备：在此脚本的运行配置 → 环境变量中添加 DDS_API_TOKEN=自己的 API Token。
运行会把照片发送给 DeepDataSpace，并使用账号的调用额度；无需本地模型权重。
仅依赖当前环境已有的 requests、Pillow，无需本地检测模型。

实验 D 参数：框阈值 0.20，图块最大 1200×1000，重叠 300 像素。
2940×833 照片分成 3 块，每块一次 window.door. 联合检测，共 3 次 API 任务。
不额外检测全图，不在本地缩放图块；图块用 PNG 编码以避免再次有损压缩。
官方云端仍按长边 1536 像素推理，返回的是上传图块的坐标。
所以只需给框加上图块左上角偏移，不能再乘以原图/1536 的缩放倍数。

使用 facade_geometry.py 的坐标还原与几何去重函数；该模块不调用模型。
内部切边附近的残框交给重叠块补充，照片真正四周的边缘框保留。
跨图块重复框合并；同一位置门/窗冲突保留为黄色 ambiguous，需人工确认。
默认检测 textures 目录下全部 17 张立面照片，不把全景或 gt_masks 当输入。
输出：my_results/grounding_dino16/batch_detection/D_时间/照片名/。
每张分别保存原始图块结果、合并 JSON、画框图、CSV 和三维映射所需 detections.json。
全批 batch_summary.csv 可以用 Excel 打开；mapping_manifest.json 仅关联照片与建筑，
并不代表已完成墙面标定。map_facade_to_3d.py 默认读取本流程最新批次。

网络或 API 出错会停止后续提交，已经完成的结果保留。
要续跑，把 RESUME_RUN_DIR 改为控制台显示的批次目录，仍运行本文件：
已保存的图块不重复调用；已有 task_uuid 的任务只查询，不重新提交。
如果提交后没有收到任务编号，必须先检查平台记录，程序不会自动重发收费请求。
RESUME_RUN_DIR=None 表示新建批次；重新跑一批会再次使用平台额度。

接口依据（核对日期 2026-09-30）：
https://github.com/deepdataspace/dds-cloudapi-sdk/blob/main/examples.py
https://github.com/deepdataspace/dds-cloudapi-sdk/blob/main/dds_cloudapi_sdk/tasks/base.py
https://github.com/deepdataspace/dds-cloudapi-sdk/blob/main/dds_cloudapi_sdk/tasks/v2_task.py
HTTP 字段与官方 V2 接口一致；本文件的读图、轮询和保存代码为教学适配。
接口及缩放说明：https://algos.deepdataspace.com/zh-cn/model/grounding_dino.md
"""

import base64
import csv
from datetime import datetime
import io
import hashlib
import json
import math
import os
from pathlib import Path
import time
import uuid

import requests
from PIL import Image, ImageDraw, ImageFont
# 公共模块只含几何运算与建筑文件关联，不导入 PyTorch 或 Transformers。
import facade_geometry


# ==================== 主要修改这里 ====================
ROOT = Path(__file__).resolve().parent
IMAGE_DIR = ROOT / "Drills/Texture2LoD3_dataset/textures"
OUTPUT_ROOT = ROOT / "my_results/grounding_dino16/batch_detection"
# 当前按用户选择：每次新建批次，从第一张重新检测；每完成一张立即保存。
# 示例：RESUME_RUN_DIR = Path(r"E:\workinCODEX\prog_3D\my_results\grounding_dino16\batch_detection\D_实际时间")
RESUME_RUN_DIR = None
PROMPT = "window.door."  # 英文类别以句点分隔，一次请求同时寻找窗和门。
MODEL = "GroundingDino-1.6-Pro"  # 官方 API 模型名，不是 Hugging Face 仓库名。
API_ROOT = "https://api.deepdataspace.com"
BOX_THRESHOLD = 0.20  # 实验 D：重叠分块 + 0.20；保持此值即可复用 D 的设置。
API_NMS_IOU = 0.8  # 服务端重叠框筛选，与原脚本自定义去重不是同一套规则。
TILE_WIDTH = 1200
TILE_HEIGHT = 1000
TILE_OVERLAP = 300
TILE_EDGE_MARGIN = 12  # 只过滤图块内部切边附近的残框，不过滤原照片真正的四周。
WAIT_SECONDS = 900  # 每次查询同一任务最多等 15 分钟；每次 HTTP 请求另有超时。
PROGRESS_SECONDS = 30  # 状态没有变化时，也每隔 30 秒报告仍在等待。


def write_json(path, value):
    """写完临时文件再替换，避免中断时留下半份进度 JSON。"""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def tile_views(width, height):
    """用二维重叠网格完整覆盖原照片；小于图块的照片只提交一次。"""
    if min(width, height, TILE_WIDTH, TILE_HEIGHT) <= 0:
        raise ValueError("图片及图块的宽高必须大于 0。")
    if not 0 <= TILE_OVERLAP < min(TILE_WIDTH, TILE_HEIGHT):
        raise ValueError("重叠宽度必须非负，并小于图块宽和高。")

    def starts(length, size):
        if length <= size:
            return [0]
        positions = list(range(0, length - size + 1, size - TILE_OVERLAP))
        if positions[-1] != length - size:
            positions.append(length - size)  # 最后一块贴齐右/下边缘，不留空白。
        return positions

    boxes = [(x, y, min(x + TILE_WIDTH, width), min(y + TILE_HEIGHT, height))
             for y in starts(height, TILE_HEIGHT) for x in starts(width, TILE_WIDTH)]
    return [("tile_{}".format(i + 1), box) for i, box in enumerate(boxes)]


def merge_tile_results(tile_results, image_size):
    """图块坐标 → 原图坐标 → 去重；原始返回结果另存，不在这里改写。"""
    candidates, rejected = [], []
    for tile in tile_results:
        for obj in tile["result"]["objects"]:
            reason = None
            try:
                box = [float(v) for v in obj["bbox"]]
                score = float(obj["score"])
                if len(box) != 4 or not all(math.isfinite(v) for v in box + [score]):
                    raise ValueError("无效数值")
                if not 0 <= score <= 1:
                    raise ValueError("评分越界")
                kind = str(obj["category"]).strip().lower().rstrip(".")
                if kind not in ("window", "door"):
                    reason = "unexpected_category"
                else:
                    full_box = facade_geometry.box_to_full_image(
                        box, tile["crop_box"], image_size, TILE_EDGE_MARGIN)
                    if full_box is None:
                        reason = "invalid_or_internal_tile_edge"
            except (KeyError, TypeError, ValueError, OverflowError):
                reason = "invalid_bbox_or_score"
            if reason:
                rejected.append({"source": tile["name"], "object": obj, "reason": reason})
                continue
            candidates.append({"class": kind, "score": score, "box": full_box,
                               "source": tile["name"], "local_box": box,
                               "crop_box": list(tile["crop_box"])})

    # 使用已经验证的 IoU/包含关系分组；这里仅做几何后处理。
    # 必须在加完偏移后统一比较；门窗冲突不能单纯按最高分强行归类。
    objects = []
    for detection in facade_geometry.remove_duplicates(candidates):
        obj = dict(detection)
        obj["category"] = obj.pop("class")
        obj["bbox"] = obj.pop("box")
        objects.append(obj)
    return {
        "objects": objects,
        "coordinate_system": "original_image_pixels_xyxy",
        "raw_object_count": sum(len(t["result"]["objects"]) for t in tile_results),
        "candidates_before_merge": candidates,
        "rejected_detections": rejected,
        "postprocess": {"method": "facade_geometry.remove_duplicates",
                        "local_nms_iou": facade_geometry.NMS_IOU,
                        "nested_ios": facade_geometry.NESTED_IOS,
                        "nested_min_area_ratio": facade_geometry.NESTED_MIN_AREA_RATIO,
                        "nested_center_fraction": facade_geometry.NESTED_CENTER_FRACTION},
    }


def read_api_data(response):
    """HTTP 成功还不够：API JSON 中 code=0 才表示此次请求成功。"""
    response.raise_for_status()
    payload = response.json()
    if payload.get("code") != 0:
        raise RuntimeError("API 返回错误 code={}: {}".format(
            payload.get("code"), payload.get("msg", "请查看平台调用记录")))
    return payload["data"]


def wait_for_task(session, task_id):
    """只查询已经创建的任务，不自动重发 POST，以免意外增加调用次数。"""
    started = time.monotonic()
    deadline = started + WAIT_SECONDS
    previous_status = None
    last_report = started
    while time.monotonic() < deadline:
        response = session.get(API_ROOT + "/v2/task_status/" + task_id, timeout=(15, 30))
        data = read_api_data(response)
        status = data["status"]
        now = time.monotonic()
        if status != previous_status or now - last_report >= PROGRESS_SECONDS:
            print("    云端状态：{}；已等待 {:.0f} 秒（本次上限 {} 秒）".format(
                status, now - started, WAIT_SECONDS), flush=True)
            previous_status = status
            last_report = now
        if status == "success":
            return data["result"]
        if status == "failed":
            raise RuntimeError("云端检测失败：{}".format(data.get("error")))
        if status not in ("waiting", "running", "triggering"):
            raise RuntimeError("未知任务状态：" + status)
        time.sleep(min(2, max(0, deadline - now)))
    raise TimeoutError("等待超时：本次上限 {} 秒，最后状态={}，任务编号={}。"
                       "本张尚未取得结果；此前已完成照片的文件仍保留。"
                       "可用任务编号核对平台记录。".format(WAIT_SECONDS, previous_status, task_id))


def draw_predictions(image, result, output_path):
    """把返回的 XYXY 像素坐标画在原图上；空列表也会保存图片。"""
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 20)
    except OSError:
        font = ImageFont.load_default()
    counts = {}
    for obj in result["objects"]:
        box = obj.get("bbox")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            raise ValueError("返回对象没有有效 bbox；请检查已保存的 result.json。")
        name = str(obj.get("category", obj.get("category_id", "unknown")))
        counts[name] = counts.get(name, 0) + 1
        color = "deepskyblue" if "window" in name.lower() else (
            "darkorange" if "door" in name.lower() else "yellow")
        draw.rectangle(box, outline=color, width=3)
        score = obj.get("score")
        label = name if score is None else "{} {:.2f}".format(name, float(score))
        if name == "ambiguous":
            # 代表候选的最高分不是“待确认类别”的置信度；两类分数看 JSON 的 class_scores。
            label = "window/door ?"
        draw.text((box[0], max(0, box[1] - 22)), label, fill=color, font=font,
                  stroke_width=1, stroke_fill="black")
    canvas.save(output_path)
    return counts


def experiment_parameters():
    """同时记录云端与本地规则；续跑时必须一致，避免一批里混用参数。"""
    return {"model": MODEL, "api_root": API_ROOT, "prompt": PROMPT,
            "bbox_threshold": BOX_THRESHOLD, "iou_threshold": API_NMS_IOU,
            "client_resize": False, "tile_size": [TILE_WIDTH, TILE_HEIGHT],
            "tile_overlap": TILE_OVERLAP, "tile_edge_margin": TILE_EDGE_MARGIN,
            "local_nms_iou": facade_geometry.NMS_IOU,
            "nested_ios": facade_geometry.NESTED_IOS,
            "nested_min_area_ratio": facade_geometry.NESTED_MIN_AREA_RATIO,
            "nested_center_fraction": facade_geometry.NESTED_CENTER_FRACTION,
            "class_conflict_policy": "keep_as_ambiguous"}


def find_images():
    """只枚举指定照片目录；不进入同级的全景、标注或历史结果目录。"""
    if not IMAGE_DIR.is_dir():
        raise FileNotFoundError("找不到立面照片目录：" + str(IMAGE_DIR))
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    images = sorted((p.resolve() for p in IMAGE_DIR.iterdir()
                     if p.is_file() and p.suffix.lower() in extensions), key=lambda p: p.name.lower())
    if not images:
        raise ValueError("照片目录为空：" + str(IMAGE_DIR))
    if len({p.stem.lower() for p in images}) != len(images):
        raise ValueError("存在同名照片（不含扩展名），请先改名，避免结果互相覆盖。")
    return images


def plan_images(images, run_dir):
    """先在本地检查全部照片，并计算任务数；此步骤不联网、不消耗额度。"""
    items = []
    for path in images:
        image_bytes = path.read_bytes()
        with Image.open(io.BytesIO(image_bytes)) as source:
            source.load()  # 真正解码，提前发现损坏照片，不只检查文件名。
            size = list(source.size)
        item = facade_geometry.mapping_item(path, run_dir)
        item.update({"image_size": size, "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
                     "planned_requests": len(tile_views(*size)), "final_object_count": None,
                     "raw_object_count": None, "rejected_count": None})
        items.append(item)
    return items


def write_batch_reports(run_dir, batch):
    """每完成一张就写汇总；失败/未处理与“成功但零个框”明确区分。"""
    items = batch["items"]
    success = [item for item in items if item["detection_status"] == "success"]
    failed = [item for item in items if item["detection_status"] == "failed"]
    batch.update({"updated_at": datetime.now().astimezone().isoformat(),
                  "success_count": len(success), "failed_count": len(failed),
                  "pending_count": len(items) - len(success) - len(failed)})
    write_json(run_dir / "batch_summary.json", batch)
    write_json(run_dir / "mapping_manifest.json", {
        "run_directory": str(run_dir), "state": batch["state"], "model": MODEL,
        "note": "2D predictions only; each image requires verified wall calibration.", "items": items})
    fields = ["image_key", "building_id", "detection_status", "window_count", "door_count",
              "ambiguous_count", "final_object_count", "planned_requests", "raw_object_count",
              "rejected_count", "result_directory", "error"]
    with (run_dir / "batch_summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(items)
    # 兼容参考 Stage 2 的三列接口，冲突框只保留在 JSON，不重复写成门和窗。
    with (run_dir / "output_stage2.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["texture_filename", "bboxes_window", "bboxes_door"])
        writer.writeheader()
        for item in success:
            with (Path(item["result_directory"]) / "output_stage2.csv").open(encoding="utf-8", newline="") as source:
                writer.writerows(csv.DictReader(source))
    write_json(run_dir.parent / "latest_run.json", {
        "run_directory": str(run_dir), "state": batch["state"],
        "success_count": len(success), "failed_count": len(failed)})


def save_image_outputs(image, merged, item, output_dir, experiment):
    """保留实验 D 原来的图/JSON，并导出原三维脚本可读取的字段。"""
    merged.update({"experiment": experiment, "model": MODEL, "image": item["image_path"],
                   "image_size": list(image.size), "image_sha256": item["image_sha256"]})
    write_json(output_dir / "result.json", merged)
    counts = draw_predictions(image, merged, output_dir / "detections.png")
    detections = []
    for obj in merged["objects"]:
        detection = dict(obj)
        detection["class"] = detection.pop("category")
        detection["box"] = detection.pop("bbox")
        detections.append(detection)
    uncertain = [d for d in detections if d["class"] == "ambiguous"]
    report = {"image": item["image_path"], "image_size": list(image.size),
              "image_sha256": item["image_sha256"], "model": MODEL, "experiment": experiment,
              "source_type": "facade_texture", "building_id": item["building_id"],
              "coordinate_system": merged["coordinate_system"], "parameters": experiment_parameters(),
              "detections": detections, "window_count": counts.get("window", 0),
              "door_count": counts.get("door", 0), "ambiguous_count": len(uncertain),
              "opening_count": len(detections), "csv_excludes_ambiguous": True}
    write_json(output_dir / "detections.json", report)
    write_json(output_dir / "needs_review.json", uncertain)
    with (output_dir / "output_stage2.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["texture_filename", "bboxes_window", "bboxes_door"])
        writer.writeheader()
        writer.writerow({"texture_filename": item["image_key"],
                         "bboxes_window": json.dumps([d["box"] for d in detections if d["class"] == "window"]),
                         "bboxes_door": json.dumps([d["box"] for d in detections if d["class"] == "door"])})
    return counts


def detect_one_image(session, item, experiment):
    """一张照片的分块流程；可从本张 request_info.json 安全续跑。"""
    image_path, output_dir = Path(item["image_path"]), Path(item["result_directory"])
    image_bytes = image_path.read_bytes()
    if hashlib.sha256(image_bytes).hexdigest() != item["image_sha256"]:
        raise ValueError("照片内容在预检后发生变化，请新建批次：" + image_path.name)
    with Image.open(io.BytesIO(image_bytes)) as source:
        image = source.convert("RGB")
    try:
        return detect_image_tiles(session, image, item, output_dir, experiment)
    finally:
        image.close()


def detect_image_tiles(session, image, item, output_dir, experiment):
    """已完成图块读本地；已提交任务只查询；仅 pending 图块才新建 API 任务。"""
    views = tile_views(*image.size)
    parameters = experiment_parameters()
    record_path = output_dir / "request_info.json"
    if record_path.is_file():
        record = json.loads(record_path.read_text(encoding="utf-8"))
        expected_views = [(name, list(box)) for name, box in views]
        recorded_views = [(t["name"], t["crop_box"]) for t in record["tiles"]]
        if (record.get("parameters") != parameters or record["image_sha256"] != item["image_sha256"]
                or record["image"] != item["image_path"] or recorded_views != expected_views):
            raise ValueError("单图进度与当前照片/参数不一致，请新建批次。")
    else:
        if output_dir.is_dir() and any(output_dir.iterdir()):
            raise ValueError("输出目录已有文件但缺少 request_info.json，不能判断哪些任务已经提交。")
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "tiles").mkdir(exist_ok=True)
        record = {"experiment": experiment, "mode": "tiles_only", "state": "running",
                  "image": item["image_path"], "image_size": list(image.size),
                  "image_sha256": item["image_sha256"], "parameters": parameters,
                  "documented_server_inference_long_edge": 1536, "planned_requests": len(views),
                  "tiles": [{"name": name, "crop_box": list(box), "state": "pending",
                             "idempotency_key": uuid.uuid4().hex} for name, box in views]}
        record.update(parameters)
        write_json(record_path, record)
    tile_results, active_tile = [], None
    try:
        record["state"] = "running"
        write_json(record_path, record)
        for index, tile in enumerate(record["tiles"], 1):
            active_tile = tile
            raw_path = output_dir / "tiles" / (tile["name"] + "_result.json")
            if raw_path.is_file():
                print("    复用已保存图块 {}/{}".format(index, len(views)), flush=True)
                result = json.loads(raw_path.read_text(encoding="utf-8"))
            else:
                if not tile.get("task_uuid"):
                    # POST 超时不能证明服务端未收到；不以新请求冒险重复扣费。
                    if tile["state"] != "pending":
                        raise RuntimeError(
                            "{} 提交状态不明且无任务编号，请先核对平台调用记录；禁止自动重复提交。进度：{}".format(
                                tile["name"], record_path))
                    buffer = io.BytesIO()
                    with image.crop(tile["crop_box"]) as crop:
                        crop.save(buffer, format="PNG")  # 原像素、无损编码，不使用 resize。
                    body = {"model": MODEL,
                            "image": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii"),
                            "prompt": {"type": "text", "text": PROMPT}, "targets": ["bbox"],
                            "bbox_threshold": BOX_THRESHOLD, "iou_threshold": API_NMS_IOU}
                    tile["state"] = "submitting"
                    write_json(record_path, record)
                    print("    提交图块 {}/{}：{}（使用平台额度）".format(index, len(views), tile["name"]), flush=True)
                    response = session.post(API_ROOT + "/v2/task/grounding_dino/detection", json=body,
                                            headers={"Idempotency-Key": tile["idempotency_key"]}, timeout=(15, 60))
                    task_id = read_api_data(response)["task_uuid"]
                    if not isinstance(task_id, str) or not task_id:
                        raise ValueError("API 没有返回有效任务编号，请检查平台调用记录。")
                    tile["task_uuid"] = task_id
                tile["state"] = "waiting"
                write_json(record_path, record)
                print("    查询任务：" + tile["task_uuid"], flush=True)
                result = wait_for_task(session, tile["task_uuid"])
                write_json(raw_path, result)
            if not isinstance(result, dict) or not isinstance(result.get("objects"), list):
                raise ValueError("API 返回结构不一致，请检查 " + str(raw_path))
            tile_results.append({"name": tile["name"], "crop_box": tile["crop_box"], "result": result})
            tile.update({"state": "completed", "raw_object_count": len(result["objects"])})
            tile.pop("failure_stage", None)
            write_json(record_path, record)
            active_tile = None
        merged = merge_tile_results(tile_results, image.size)
        counts = save_image_outputs(image, merged, item, output_dir, experiment)
        record.update({"state": "completed", "final_counts": counts,
                       "raw_object_count": merged["raw_object_count"],
                       "rejected_count": len(merged["rejected_detections"]),
                       "final_object_count": len(merged["objects"])})
        record.pop("error_type", None)
        write_json(record_path, record)
        return record
    except (Exception, KeyboardInterrupt) as exc:
        if active_tile is not None:
            if active_tile["state"] != "error":
                active_tile["failure_stage"] = active_tile["state"]
            active_tile["state"] = "error"
        record["state"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        record["error_type"] = type(exc).__name__
        write_json(record_path, record)
        raise


def outputs_complete(item):
    required = ("result.json", "detections.png", "detections.json", "output_stage2.csv", "needs_review.json")
    return all((Path(item["result_directory"]) / name).is_file() for name in required)


def main():
    """批量入口：先本地预检 → 按文件名顺序检测 → 每张保存 → 更新全批汇总。"""
    token = os.environ.get("DDS_API_TOKEN", "").strip()
    if not token:
        raise RuntimeError("请先在 PyCharm 此脚本的运行配置中设置环境变量 DDS_API_TOKEN。")
    images = find_images()
    parameters = experiment_parameters()
    default_d = (BOX_THRESHOLD, API_NMS_IOU, PROMPT, TILE_WIDTH, TILE_HEIGHT,
                 TILE_OVERLAP, TILE_EDGE_MARGIN) == (0.20, 0.8, "window.door.", 1200, 1000, 300, 12)
    experiment = "D" if default_d else "tiled_custom"
    run_dir = (Path(RESUME_RUN_DIR).resolve() if RESUME_RUN_DIR is not None else
               OUTPUT_ROOT.resolve() / (experiment + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")))
    planned_items = plan_images(images, run_dir)
    if RESUME_RUN_DIR is not None:
        batch = json.loads((run_dir / "batch_summary.json").read_text(encoding="utf-8"))
        identity = lambda items: [(i["image_path"], i["image_sha256"], i["image_size"]) for i in items]
        if (batch["parameters"] != parameters or batch["run_directory"] != str(run_dir)
                or identity(batch["items"]) != identity(planned_items)):
            raise ValueError("续跑的照片清单、内容或参数已经变化，请恢复原设置或将 RESUME_RUN_DIR 设为 None 新建批次。")
        # 最终导出文件被删掉时，重新用缓存图块导出；不把缺文件的图片提前列为成功。
        for item in batch["items"]:
            if item["detection_status"] == "success" and not outputs_complete(item):
                item.update({"detection_status": "pending", "status": "pending_detection",
                             "window_count": None, "door_count": None, "ambiguous_count": None,
                             "final_object_count": None})
        print("[1] 续跑已有批次；已完成的图片/图块不会重复提交。", flush=True)
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
        batch = {"version": 1, "experiment": experiment, "run_directory": str(run_dir),
                 "source_directory": str(IMAGE_DIR.resolve()), "parameters": parameters,
                 "started_at": datetime.now().astimezone().isoformat(), "image_count": len(images),
                 "planned_requests": sum(i["planned_requests"] for i in planned_items), "items": planned_items}
        print("[1] 新建实验 {} 批次。".format(experiment), flush=True)
    print("[2] 共 {} 张立面照片，完整新跑需 {} 个图块任务；阈值={}。".format(
        len(images), batch["planned_requests"], BOX_THRESHOLD), flush=True)
    print("    批次目录：{}".format(run_dir), flush=True)
    for item in batch["items"]:
        print("    {}：{}x{}，{} 块".format(item["image_key"], *item["image_size"], item["planned_requests"]), flush=True)
    batch["state"] = "running"
    write_batch_reports(run_dir, batch)
    active_item = None
    try:
        with requests.Session() as session:
            session.headers.update({"Token": token})
            for index, item in enumerate(batch["items"], 1):
                if item["detection_status"] == "success" and outputs_complete(item):
                    print("[3] {}/{} {}：已完成，跳过。".format(index, len(images), item["image_key"]), flush=True)
                    continue
                active_item = item
                print("\n[3] 图片 {}/{}：{}".format(index, len(images), item["image_key"]), flush=True)
                record = detect_one_image(session, item, experiment)
                counts = record["final_counts"]
                has_model = item["citygml_path"] and Path(item["citygml_path"]).is_file()
                item.update({"detection_status": "success", "status": "awaiting_calibration" if has_model else "needs_building_match",
                             "window_count": counts.get("window", 0), "door_count": counts.get("door", 0),
                             "ambiguous_count": counts.get("ambiguous", 0), "error": "",
                             "final_object_count": record["final_object_count"],
                             "raw_object_count": record["raw_object_count"], "rejected_count": record["rejected_count"]})
                print("    完成：窗 {}，门 {}，待确认 {}。".format(
                    item["window_count"], item["door_count"], item["ambiguous_count"]), flush=True)
                active_item = None
                write_batch_reports(run_dir, batch)
                print("    本张结果及批次汇总已保存：{}".format(item["result_directory"]), flush=True)
    except (Exception, KeyboardInterrupt) as exc:
        batch["state"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        if active_item is not None:
            active_item.update({"detection_status": "failed", "status": "detection_failed", "error": type(exc).__name__,
                                "window_count": None, "door_count": None, "ambiguous_count": None,
                                "final_object_count": None})
        write_batch_reports(run_dir, batch)
        print("批次已停止，此前完成的图片及汇总已保存：\n{}".format(run_dir), flush=True)
        raise
    batch["state"] = "completed"
    write_batch_reports(run_dir, batch)
    print("\n[4] 批量完成：{}/{} 张。".format(batch["success_count"], len(images)), flush=True)
    print("    先查看：{}".format(run_dir / "batch_summary.csv"), flush=True)
    print("    每张画框图：批次目录/照片名/detections.png；黄色为门窗待确认。", flush=True)
    print("    已导出三维关联清单；映射仍需逐图标定，本脚本不直接修改建筑。", flush=True)
    return run_dir


if __name__ == "__main__":
    try:
        main()
    except requests.RequestException as exc:
        # 不打印请求头或整份请求体，避免把 Token、图片 Base64 放进日志。
        status_code = exc.response.status_code if exc.response is not None else "无 HTTP 响应"
        print("网络/API 请求失败：{}；状态={}。".format(type(exc).__name__, status_code), flush=True)
        print("请检查网络、Token 和平台额度；若已提交任务，先查看调用记录再决定是否重跑。", flush=True)
        raise SystemExit(1)
    except (RuntimeError, ValueError, OSError) as exc:
        print("运行停止：{}".format(exc), flush=True)
        raise SystemExit(1)
    except KeyboardInterrupt:
        print("已中断，此前完成的图片和批次进度已保存。", flush=True)
        raise SystemExit(130)
