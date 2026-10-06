"""一张真实立面照片与模型渲染图的 RoMa v2 匹配实验（PyCharm 直接运行）。

模型：Parskatt/RoMaV2 官方 v2.0.1 预训练权重；不是 Grounding DINO。
官方源码放在 references/RoMaV2；本文件负责自己的数据准备、验证及可视化。
流程：裁掉渲染图留白 → RoMa 候选点 → RANSAC → 已检测窗中心辅助对齐 → 自动检查。
主入口不读取人工标定，不要求点选。自动几何检查通过不等于真实精度已验证。
本实验处理已指定的建筑正面，不修改原有标定，也不把检测框写入三维模型。
"""
from pathlib import Path
from datetime import datetime
import csv
import hashlib
import json
import sys
import time

if sys.version_info < (3, 10):
    raise RuntimeError("RoMa v2 需要 Python>=3.10；请使用升级后的 torch_1（Python 3.12）。")

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from facade_match_geometry import (roma_to_image_pixels, fit_and_check,
                                   project_homography, backproject_render_points)

# ===== 只需关注的实验参数 =====
ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "Drills/Texture2LoD3_dataset"
PHOTO_PATH = DATASET / "textures/4959323_front.jpg"
RENDER_ROOT = ROOT / "my_results/rendered_buildings/4959323"
RENDER_RUN_DIR = None  # None：读取 latest_run.json；也可明确指定已有 render_时间 目录。
OUTPUT_ROOT = ROOT / "my_results/roma_matching/4959323_front"
USE_WINDOW_STRUCTURE = True  # 用实验D已有窗框与模型窗构件自动细化；不重新调用检测API。
DETECTION_JSON = None       # None：使用实验D最新批次中当前照片的 detections.json。
# 以下三项只供旧实验的 check_against_manual() 重现报告；main() 完全不使用它们。
MANUAL_CHECK_PATH = ROOT / "my_results/calibrations/4959323_front.json"
MODEL_CACHE = ROOT / "model_cache/roma_v2/hub"
SETTING = "base"       # 640×640，8 GB 显存先用此配置；precise已比较，改善有限且更占内存。
NUM_MATCHES = 2000      # 官方采样会暂时扩展到4倍；过大会显著占用显存。
MIN_OVERLAP = 0.20      # RoMa 的对应可信度筛选值，不是已校准的正确率。
RANSAC_THRESHOLD_PX = 4.0  # 完整渲染图上的像素误差阈值。
MAX_MANUAL_ERROR_M = 0.30
MEDIAN_MANUAL_ERROR_M = 0.20
SEED = 42

# 留用原始 RoMa 输出，之后只调筛选/绘图时可复用，不必重新推理。
# 默认 None，每次新建匹配实验；设置为已有run目录将校验输入与渲染元数据指纹。
REUSE_MATCHES_DIR = None
BG, INK, MUTED = (246, 245, 241), (59, 69, 65), (113, 125, 117)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def load_detection_result(identity, photo_size):
    """只复用检测结果，校验照片指纹，避免把别的照片或缩放后的框混进来。"""
    if DETECTION_JSON is None:
        pointer = ROOT / "my_results/grounding_dino16/batch_detection/latest_run.json"
        if not pointer.exists():
            return None, {"available": False, "reason": "没有已有检测批次，使用纯RoMa。"}
        path = Path(read_json(pointer)["run_directory"]) / PHOTO_PATH.stem / "detections.json"
    else:
        path = Path(DETECTION_JSON)
    if not path.exists():
        return None, {"available": False, "reason": "当前照片没有已保存检测框，使用纯RoMa。", "path": str(path)}
    data = read_json(path)
    if (data.get("image_sha256") != identity["photo_sha256"] or
            data.get("image_size") != list(photo_size) or
            data.get("coordinate_system") != "original_image_pixels_xyxy"):
        raise ValueError("检测结果的照片指纹、尺寸或坐标定义不一致，拒绝用于结构细化。")
    return data, {"available": True, "path": str(path), "sha256": sha256(path),
                  "model": data.get("model"), "experiment": data.get("experiment")}


def choose_automatic_transform(roma_h, roma_report, structure_h=None, structure_report=None):
    """只依据自动几何指标选择候选，不读取人工误差、人工点或历史实验最优矩阵。"""
    if structure_h is not None and structure_report and structure_report.get("gate_passed"):
        return structure_h, "roma_window_structure", structure_report
    return roma_h, "roma", roma_report


def font(size):
    return ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", size)


def prepare_pair():
    """裁剪范围只来自三维渲染的构件编号，完全不读取照片标注/人工对应点。"""
    render_dir = Path(RENDER_RUN_DIR) if RENDER_RUN_DIR else Path(read_json(RENDER_ROOT / "latest_run.json")["run_dir"])
    manifest = read_json(render_dir / "manifest.json")
    if manifest["status"] != "complete":
        raise ValueError("只能使用已经完整渲染成功的批次。")
    if sha256(manifest["source_gml"]) != manifest["source_sha256"]:
        raise ValueError("源模型在渲染之后发生变化，需重新渲染，不能混用旧相机/深度。")
    view = next(v for v in manifest["views"] if v["name"] == "front")
    render_path = render_dir / view["image"]
    camera = read_json(render_dir / view["camera"])
    with np.load(render_dir / view["geometry"]) as arrays:
        depth, object_map = arrays["depth_m"], arrays["object_index"]
    wall_id = manifest["front_reference_wall_id"]
    allowed = [obj["index"] for obj in manifest["objects"] if obj["wall_id"] == wall_id]
    valid = np.isin(object_map, allowed) & np.isfinite(depth)
    ys, xs = np.where(valid)
    if not len(xs):
        raise ValueError("渲染图中看不到目标墙及门窗。")
    # 保留几像素边距；原始图、相机与深度不裁剪，只有送模型的B图裁剪。
    height, width = depth.shape
    crop = [max(0, int(xs.min()) - 4), max(0, int(ys.min()) - 4),
            min(width, int(xs.max()) + 5), min(height, int(ys.max()) + 5)]
    photo = Image.open(PHOTO_PATH).convert("RGB")
    render = Image.open(render_path).convert("RGB")
    identity = {"photo_sha256": sha256(PHOTO_PATH), "render_sha256": sha256(render_path),
                "render_manifest_sha256": sha256(render_dir / "manifest.json"),
                "camera_sha256": sha256(render_dir / view["camera"]),
                "geometry_sha256": sha256(render_dir / view["geometry"]), "crop_xyxy": crop}
    return photo, render, render.crop(crop), camera, depth, object_map, allowed, manifest, render_dir, identity


def run_model(photo, crop_image, output, identity):
    """此函数中才真正运行预训练模型；坐标、验证等步骤不需要神经网络。"""
    if REUSE_MATCHES_DIR is not None:
        previous = Path(REUSE_MATCHES_DIR)
        old = read_json(previous / "run_info.json")
        if old["input_identity"] != identity:
            raise ValueError("复用结果与当前输入/渲染元数据不一致，不能混用。")
        data = dict(np.load(previous / "raw_matches.npz"))
        np.savez_compressed(output / "raw_matches.npz", **data)
        return data["matches"], data["overlap"], dict(old["model"], reused_from=str(previous))

    import torch
    model, info = load_roma_model()
    tick = time.perf_counter()
    print("[3] 推理并采样对应点……", flush=True)
    normalized, confidence = infer_pair(model, photo, crop_image)
    np.savez_compressed(output / "raw_matches.npz", matches=normalized, overlap=confidence)
    info["inference_seconds"] = round(time.perf_counter() - tick, 3)
    info["peak_cuda_memory_GB"] = round(torch.cuda.max_memory_allocated() / 1024**3, 3) if torch.cuda.is_available() else None
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return normalized, confidence, info


def load_roma_model():
    """共用的本地模型入口；多建筑流程加载一次后逐个匹配候选。"""
    import torch
    import romav2
    import PIL
    from romav2 import RoMaV2
    checkpoint = MODEL_CACHE / "checkpoints/romav2.0.1.pt"
    backbone = MODEL_CACHE / "facebookresearch_dinov3_adc254450203739c8149213a7a69d8d905b4fcfa"
    if not checkpoint.is_file() or not backbone.is_dir():
        raise FileNotFoundError("缺少本地 RoMa 权重或 DINOv3 源码缓存。请先完成首次模型准备；本入口不重复下载。")
    torch.hub.set_dir(str(MODEL_CACHE))
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_float32_matmul_precision("highest")  # 官方 forward 明确要求。
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.cuda.reset_peak_memory_stats()
    print("[2] 加载本地 RoMa v2.0.1 权重，setting=" + SETTING, flush=True)
    model = RoMaV2(RoMaV2.Cfg(setting=SETTING, compile=False))
    info = {"name": "RoMa v2.0.1", "official_source": "https://github.com/Parskatt/RoMaV2",
            "setting": SETTING, "num_matches": NUM_MATCHES, "seed": SEED,
            "checkpoint_sha256": sha256(checkpoint), "torch_version": torch.__version__,
            "device": str(next(model.parameters()).device),
            "feature_backbone": "DINOv3 ViT-L/16 (included in RoMa checkpoint)",
            "input_resize": [model.W_lr, model.H_lr],
            "high_resolution_resize": [model.W_hr, model.H_hr] if model.H_hr is not None else None,
            "bidirectional": model.bidirectional, "input_coordinates": "normalized align_corners=False"}
    info["implementation"] = {
        "romav2_source_sha256": code_sha256(Path(romav2.__file__).parent),
        "dinov3_source_sha256": code_sha256(backbone),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "numpy": np.__version__, "pillow": PIL.__version__,
        "device": str(next(model.parameters()).device),
        "device_name": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
        "matmul_precision": torch.get_float32_matmul_precision()}
    return model, info


def code_sha256(directory):
    """模型包实际源码指纹，避免更新依赖后复用旧推理。"""
    digest = hashlib.sha256()
    for path in sorted(Path(directory).rglob("*.py")):
        digest.update(path.relative_to(directory).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def infer_pair(model, photo, crop_image):
    """每对图固定采样种子，避免候选排序改变采样结果。"""
    import torch
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    with torch.inference_mode():
        predictions = model.match(photo, crop_image)
        matches, overlap, _, _ = model.sample(predictions, NUM_MATCHES)
    return matches.float().cpu().numpy(), overlap.float().cpu().numpy()


def tiled_correspondences(model, photo, render, initial_h, allowed_mask):
    """粗H引导局部RoMa匹配，返回原照片点、完整渲染点、可信度和逐块记录。

    复用实验D的1200×1000/重叠300网格，但这里只运行本地RoMa，不调用云API。
    将照片块边界经粗H投到渲染图，包围框各边扩展其宽/高的10%，再匹配两块。
    原图没有预先缩小；RoMa内部仍按自身设置缩放，每个图块因此占用更多模型像素。
    粗H只决定搜索裁剪范围，不用它的重投影误差筛点，避免人为制造几何通过。
    返回点统一还原到两张完整图的像素中心坐标，不拟合H、不更改质量门槛。
    重叠块中的同一照片像素只保留最高可信度的一对；不平均可能冲突的目标位置。
    """
    import try_grounding_dino16 as detector

    H = np.asarray(initial_h, dtype=float)
    mask = np.asarray(allowed_mask)
    if H.shape != (3, 3) or not np.isfinite(H).all() or np.linalg.matrix_rank(H) < 3:
        raise ValueError("分块匹配需要有限、非奇异的照片→完整渲染图H。")
    H = H/np.linalg.norm(H)
    if mask.dtype != np.bool_ or mask.shape != (render.height, render.width):
        raise ValueError("allowed_mask必须是与完整渲染图同尺寸的布尔墙面掩码。")
    report = {"method": "coarse-H-guided local RoMa correspondences",
              "photo_size_wh": list(photo.size), "render_size_wh": list(render.size),
              "initial_homography_photo_to_render": H.tolist(),
              "policy": {"tile_size_wh": [detector.TILE_WIDTH, detector.TILE_HEIGHT],
                         "tile_overlap_px": detector.TILE_OVERLAP,
                         "render_margin_fraction_each_side": .10, "min_render_crop_side_px": 32,
                         "min_overlap": MIN_OVERLAP,
                         "deduplication": "highest overlap per nearest original-photo pixel"},
              "manual_points_used": False, "gt_masks_used": False,
              "independent_accuracy_verified": False, "tiles": [],
              "interpretation": "局部对应点还需原有空间留出几何检查；粗H若错列，分块仍可能错列。"}
    photo_points, render_points, confidences, source_tiles = [], [], [], []

    def inside(points, size):
        return (np.isfinite(points).all(axis=1) & (points >= -.5).all(axis=1)
                & (points < np.asarray(size)-.5).all(axis=1))

    for tile_index, (name, box) in enumerate(detector.tile_views(*photo.size)):
        row = {"name": name, "photo_crop_xyxy": list(box), "status": "skipped"}
        report["tiles"].append(row)
        x0, y0, x1, y1 = box
        # PIL裁剪起点是整数像素中心，连续边界比其小半像素。
        corners = np.array([[x0-.5, y0-.5], [x1-.5, y0-.5],
                            [x1-.5, y1-.5], [x0-.5, y1-.5]])
        denominator = np.column_stack((corners, np.ones(4)))@H[2]
        if not ((denominator > 1e-12).all() or (denominator < -1e-12).all()):
            row["reason"] = "projective_pole_crosses_photo_tile"
            continue
        projected = project_homography(corners, H)
        if not np.isfinite(projected).all():
            row["reason"] = "nonfinite_projected_tile"
            continue
        span = np.ptp(projected, axis=0)
        area = abs(float(np.sum(projected[:, 0]*np.roll(projected[:, 1], -1)
                                - projected[:, 1]*np.roll(projected[:, 0], -1))))/2
        if np.any(span < 1.) or area < 1.:
            row["reason"] = "degenerate_projected_tile"
            continue
        # 先与图像边界相交，再转整数，避免极大有限坐标溢出int。
        lower = np.maximum(projected.min(axis=0)-.10*span, [-.5, -.5])
        upper = np.minimum(projected.max(axis=0)+.10*span, np.asarray(render.size)-.5)
        if np.any(upper-lower < 32.):
            row["reason"] = "render_crop_outside_image_or_too_small"
            continue
        start = np.floor(lower+.5).astype(int)
        stop = np.ceil(upper+.5).astype(int)
        crop = [int(start[0]), int(start[1]), int(stop[0]), int(stop[1])]
        row["render_crop_xyxy"] = crop
        if not mask[crop[1]:crop[3], crop[0]:crop[2]].any():
            row["reason"] = "no_allowed_wall_in_render_crop"
            continue
        try:
            photo_crop, render_crop = photo.crop(box), render.crop(crop)
            normalized, overlap = infer_pair(model, photo_crop, render_crop)
            overlap = np.asarray(overlap, dtype=float)
            a, b = roma_to_image_pixels(normalized, photo_crop.size, render_crop.size, crop[:2])
            if overlap.shape != (len(a),):
                raise ValueError("RoMa匹配点与可信度数组长度不一致。")
            row["raw_match_count"] = len(a)
            valid = (inside(a, photo_crop.size) & inside(b-np.asarray(crop[:2]), render_crop.size)
                     & np.isfinite(overlap) & (overlap >= MIN_OVERLAP))
            a += np.asarray(box[:2])
            valid &= inside(a, photo.size) & inside(b, render.size)
            indices = np.flatnonzero(valid)
            pixels = np.floor(b[indices]+.5).astype(int)
            valid[indices] &= mask[pixels[:, 1], pixels[:, 0]]
            row.update({"status": "completed", "kept_before_deduplication": int(valid.sum())})
            photo_points.append(a[valid]); render_points.append(b[valid]); confidences.append(overlap[valid])
            source_tiles.append(np.full(int(valid.sum()), tile_index, dtype=int))
        except (ValueError, RuntimeError) as error:
            # 单块失败有明确记录，不伪装成已完成；全部失败返回空点交给上游拒绝。
            row.update({"status": "failed", "reason": "local_inference_failed",
                        "error_type": type(error).__name__, "error": str(error)})

    a = np.concatenate(photo_points) if photo_points else np.empty((0, 2), dtype=float)
    b = np.concatenate(render_points) if render_points else np.empty((0, 2), dtype=float)
    confidence = np.concatenate(confidences) if confidences else np.empty(0, dtype=float)
    origins = np.concatenate(source_tiles) if source_tiles else np.empty(0, dtype=int)
    # 按可信度稳定排序，同分时保留较早图块的点，结果可复现。
    order = np.argsort(-confidence, kind="stable")
    pixels = np.floor(a[order]+.5).astype(np.int64)
    _, first = np.unique(pixels, axis=0, return_index=True)
    kept = np.sort(order[first])
    for tile_index, row in enumerate(report["tiles"]):
        row["kept_after_deduplication"] = int(np.count_nonzero(origins[kept] == tile_index))
    completed = sum(row["status"] == "completed" for row in report["tiles"])
    failed = sum(row["status"] == "failed" for row in report["tiles"])
    report.update({"tile_count": len(report["tiles"]), "completed_tile_count": completed,
                   "failed_tile_count": failed, "skipped_tile_count": len(report["tiles"])-completed-failed,
                   "matches_before_deduplication": len(a), "duplicate_source_pixels_removed": len(a)-len(kept),
                   "match_count": len(kept), "status": "no_matches" if not len(kept) else
                   ("complete" if completed == len(report["tiles"]) else "partial")})
    return a[kept], b[kept], confidence[kept], report


def check_against_manual(H, manifest, camera, identity):
    """旧人工点仅用于独立检查。这一步不改变 H，也不筛掉不利的检查点。"""
    if not MANUAL_CHECK_PATH.exists():
        return {"available": False, "passed": False, "reason": "没有独立人工核对点，需人工复核。"}
    manual = read_json(MANUAL_CHECK_PATH)
    if (manual["identity"]["image_sha256"] != identity["photo_sha256"] or
        manual["identity"]["citygml_sha256"] != manifest["source_sha256"] or
        manual["wall_id"] != manifest["front_reference_wall_id"]):
        return {"available": False, "passed": False, "reason": "旧人工标定的照片/模型/墙面身份不同，不能当检查依据。"}
    # 复用原三维脚本的“读墙面、墙面UV转世界坐标”，保持旧人工点的坐标定义。
    from map_facade_to_3d import load_walls, to_xyz
    walls = load_walls(Path(manifest["source_gml"]))
    frame = next(w["frame"] for w in walls if w["id"] == manual["wall_id"])
    pairs = np.asarray(manual["control_pairs"] + manual["check_pairs"], dtype=float)
    world = to_xyz(pairs[:, 1], frame)
    local = world - np.asarray(manifest["world_origin_m"])
    projection = np.asarray(camera["local_to_pixel_depth"])
    expected = (local @ projection[:3, :3].T + projection[:3, 3])[:, :2]
    actual = project_homography(pairs[:, 0], H)
    errors_px = np.linalg.norm(actual - expected, axis=1)
    errors_m = errors_px / camera["pixels_per_meter"]
    passed = bool(np.isfinite(errors_m).all() and len(errors_m) >= 4 and
                  np.max(errors_m) <= MAX_MANUAL_ERROR_M and np.median(errors_m) <= MEDIAN_MANUAL_ERROR_M)
    return {"available": True, "passed": passed, "count": len(pairs),
            "source": str(MANUAL_CHECK_PATH), "source_sha256": sha256(MANUAL_CHECK_PATH),
            "used_for_fitting": False, "used_for_model_input": False,
            "development_comparison": True,
            "max_allowed_error_m": MAX_MANUAL_ERROR_M, "median_allowed_error_m": MEDIAN_MANUAL_ERROR_M,
            "median_error_m": float(np.median(errors_m)) if np.isfinite(errors_m).all() else None,
            "max_error_m": float(np.max(errors_m)) if np.isfinite(errors_m).all() else None,
            "nonfinite_count": int((~np.isfinite(errors_m)).sum()),
            "points": [{"id": i + 1, "photo_xy": pairs[i, 0].tolist(), "reference_render_xy": expected[i].tolist(),
                        "predicted_render_xy": actual[i].tolist() if np.isfinite(actual[i]).all() else None,
                        "error_px": float(errors_px[i]) if np.isfinite(errors_px[i]) else None,
                        "error_m": float(errors_m[i]) if np.isfinite(errors_m[i]) else None} for i in range(len(pairs))],
            "limitation": "旧人工点本身存在点选误差；这是本栋已知立面的独立对照，不是测量真值或全数据集准确率。"}


def draw_matches(photo, crop_image, photo_xy, render_xy, crop_xy, arrays, output,
                 filename="matches.png", title="RoMa v2 / 真实照片与三维渲染图", quality_label=None):
    width, margin, gap = 1600, 55, 125
    scale_a = width / photo.width
    scale_b = width / crop_image.width
    ha, hb = round(photo.height * scale_a), round(crop_image.height * scale_b)
    canvas = Image.new("RGB", (width + 2 * margin, ha + hb + gap + 175), BG)
    draw = ImageDraw.Draw(canvas)
    y_a, y_b = 82, 82 + ha + gap
    canvas.paste(photo.resize((width, ha), Image.Resampling.LANCZOS), (margin, y_a))
    canvas.paste(crop_image.resize((width, hb), Image.Resampling.LANCZOS), (margin, y_b))
    draw.text((margin, 22), title, font=font(30), fill=INK)
    ids = np.flatnonzero(arrays["inlier_mask"])
    # 按照片中的位置均匀挑显示点，图面不堆满连线；完整点保存在CSV/NPZ。
    chosen, occupied = [], set()
    for i in ids[np.argsort(arrays["errors_px"][ids])]:
        cell = tuple(np.floor(photo_xy[i] / np.array(photo.size) * [16, 4]).astype(int))
        if cell not in occupied:
            chosen.append(i); occupied.add(cell)
    for i in chosen[:50]:
        a = np.array([margin, y_a]) + photo_xy[i] * scale_a
        b = np.array([margin, y_b]) + (render_xy[i] - crop_xy) * scale_b
        color = (109, 135, 133) if arrays["train_mask"][i] else (157, 125, 105)
        draw.line((tuple(a), tuple(b)), fill=color, width=1)
        for p in (a, b):
            draw.ellipse((p[0]-3, p[1]-3, p[0]+3, p[1]+3), fill=color, outline=BG)
    draw.text((margin, y_b - 48), "上：实拍照片   下：渲染立面裁剪   蓝绿：拟合内点   棕：留出检查内点", font=font(22), fill=INK)
    draw.text((margin, y_b + hb + 20), quality_label or "仅显示部分几何内点；自动一致性不等于真实精度验证。", font=font(21), fill=MUTED)
    canvas.save(output / filename)


def draw_alignment(photo, render, crop, H, manual, output, quality_label=None):
    warped = cv2.warpPerspective(np.asarray(photo), H, render.size, flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=BG)
    valid = cv2.warpPerspective(np.ones((photo.height, photo.width), np.uint8), H, render.size,
                               flags=cv2.INTER_NEAREST).astype(bool)
    blend = np.array(render).copy()
    blend[valid] = (blend[valid] * .5 + warped[valid] * .5).astype(np.uint8)
    rows = [("模型渲染（参照）", render.crop(crop)),
            ("根据自动匹配变换后的照片", Image.fromarray(warped).crop(crop)),
            ("两图 50% 叠加：检查窗框是否对齐", Image.fromarray(blend).crop(crop))]
    width, margin = 1600, 55
    height = round((crop[3]-crop[1]) * width / (crop[2]-crop[0]))
    canvas = Image.new("RGB", (width+margin*2, 120+(height+80)*3), BG)
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 20), "对齐检查  /  同一渲染坐标系", font=font(32), fill=INK)
    if quality_label:
        draw.text((margin, 60), quality_label, font=font(20), fill=MUTED)
    for index, (label, image) in enumerate(rows):
        y = 85 + index * (height + 80)
        draw.text((margin, y), label, font=font(24), fill=INK)
        canvas.paste(image.resize((width, height), Image.Resampling.LANCZOS), (margin, y+38))
    canvas.save(output / "alignment.png")
    if manual and manual.get("available"):
        scale = width / (crop[2]-crop[0])
        checks = Image.new("RGB", (width+margin*2, height+220), BG)
        checks.paste(render.crop(crop).resize((width,height), Image.Resampling.LANCZOS), (margin, 92))
        draw = ImageDraw.Draw(checks)
        draw.text((margin, 20), "独立人工点对照  /  这些点未参与模型匹配或拟合", font=font(29), fill=INK)
        for point in manual["points"]:
            if point["predicted_render_xy"] is None:
                continue  # 不可投影点已在JSON计为失败，不能假画到图像边界上。
            a = (np.array(point["reference_render_xy"])-crop[:2])*scale+[margin,92]
            b = (np.array(point["predicted_render_xy"])-crop[:2])*scale+[margin,92]
            draw.line((tuple(a),tuple(b)), fill=(151,95,79), width=2)
            draw.ellipse((a[0]-7,a[1]-7,a[0]+7,a[1]+7), outline=(79,126,114), width=3)
            draw.line((b[0]-7,b[1],b[0]+7,b[1]),fill=(151,95,79),width=3)
            draw.line((b[0],b[1]-7,b[0],b[1]+7),fill=(151,95,79),width=3)
            draw.text((a[0]+10,a[1]-28), "%d / %.3f m" % (point["id"],point["error_m"]),font=font(19),fill=INK)
        draw.text((margin,height+118), "绿圈：旧人工参考点   红十字：自动变换预测位置",font=font(23),fill=INK)
        draw.text((margin,height+160), "对照通过" if manual["passed"] else "对照未通过：请勿直接用于自动三维写入",font=font(23),fill=INK)
        checks.save(output / "independent_checks.png")


def main():
    tick = time.perf_counter()
    photo, render, crop_image, camera, depth, object_map, allowed, manifest, render_dir, identity = prepare_pair()
    output = OUTPUT_ROOT / datetime.now().strftime("match_%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    crop_image.save(output / "render_input_crop.png")
    print("[1] 照片：%s；渲染批次：%s" % (PHOTO_PATH.name, render_dir.name), flush=True)
    print("    输出目录：", output, flush=True)
    normalized, confidence, model_info = run_model(photo, crop_image, output, identity)
    info = {"photo_path": str(PHOTO_PATH), "render_run_dir": str(render_dir), "input_identity": identity,
            "python": sys.executable, "python_version": sys.version, "model": model_info,
            "min_overlap": MIN_OVERLAP, "ransac_threshold_px": RANSAC_THRESHOLD_PX,
            "manual_calibration_used": False, "use_window_structure": USE_WINDOW_STRUCTURE,
            "script_sha256": sha256(__file__), "geometry_source": "known existing LoD3 front facade",
            "building_and_wall_selected_in_advance": True}
    write_json(output / "run_info.json", info)
    photo_xy, render_xy = roma_to_image_pixels(normalized, photo.size, crop_image.size, identity["crop_xyxy"][:2])
    world, surface_valid = backproject_render_points(render_xy, camera, depth, object_map,
                                                     manifest["world_origin_m"], allowed)
    candidate_mask = (confidence >= MIN_OVERLAP) & surface_valid & np.isfinite(photo_xy).all(axis=1)
    photo_good, render_good = photo_xy[candidate_mask], render_xy[candidate_mask]
    print("[4] 原始 %d 点，可信度与目标墙过滤后 %d 点；开始空间留出检查。" % (len(normalized),len(photo_good)), flush=True)
    try:
        roma_h, geometry, roma_arrays = fit_and_check(photo_good, render_good, photo.size, render.size, RANSAC_THRESHOLD_PX)
    except ValueError as exc:
        write_json(output / "result.json", {"status": "insufficient_correspondences", "reliable": False,
                                            "reason": str(exc), "sampled_count": len(normalized),
                                            "candidate_count": len(photo_good)})
        print("匹配检查未通过：", exc, "；原始结果已经保存。", flush=True)
        return
    print("[5] 自动进行窗中心关联与几何检查；不读取人工标定。", flush=True)
    structure_h, structure_arrays = None, None
    structure_report = {"gate_passed": False, "reason": "结构细化已关闭。"}
    detection_source = {"available": False}
    if USE_WINDOW_STRUCTURE:
        try:
            detections, detection_source = load_detection_result(identity, photo.size)
            if detections is not None:
                from facade_match_structure import refine_with_window_structure
                structure_h, structure_report, structure_arrays = refine_with_window_structure(
                    roma_h, photo.size, render.size, detections, object_map, manifest)
            else:
                structure_report = dict(detection_source, gate_passed=False)
        except (ValueError, KeyError, OSError) as exc:
            # 辅助数据有问题时完整记录原因，不把失败候选冒充成功结果。
            structure_report = {"gate_passed": False, "reason": str(exc)}
            print("    未采用结构细化：", exc, flush=True)
    H, method, automatic = choose_automatic_transform(roma_h, geometry, structure_h, structure_report)
    accepted = bool(automatic["gate_passed"])
    if method == "roma_window_structure":
        final_photo, final_render = structure_arrays["photo_xy"], structure_arrays["render_xy"]
        arrays = {k: v for k, v in structure_arrays.items() if k not in ("photo_xy", "render_xy")}
        scores = np.asarray([detections["detections"][int(i)]["score"] for i in arrays["detection_indices"]])
        score_type = "window_detection_score"
    else:
        final_photo, final_render, arrays = photo_good, render_good, roma_arrays
        scores, score_type = confidence[candidate_mask], "roma_overlap"
    final_world, final_surface_valid = backproject_render_points(final_render, camera, depth, object_map,
                                                               manifest["world_origin_m"], allowed)
    result = {"schema_version": 2,
              "status": "auto_checks_passed" if accepted else "auto_checks_failed", "auto_accepted": accepted,
              "manual_calibration_used": False, "independent_accuracy_verified": False,
              "selected_method": method, "automatic_validation": automatic,
              "sampled_count": len(normalized), "candidate_count": len(photo_good),
              "selected_correspondence_count": len(final_photo),
              "photo_to_full_render_homography": H.tolist(), "roma_photo_to_full_render_homography": roma_h.tolist(),
              "geometric_validation": geometry, "structure_validation": structure_report,
              "detection_source": detection_source, "confidence_type": score_type,
              "validation_scope": "自动几何一致性；重复窗列错配仍可能通过，真实米制精度未验证。",
              "scope": "已知建筑与正面；验证照片-渲染匹配，不代表建筑检索已实现，也不评估纯LoD2条件。",
              "world_coordinates": "EPSG:25832; nearest rendered visible surface, not snapped to wall plane",
              "total_seconds": round(time.perf_counter()-tick,3)}
    write_json(output / "result.json", result)
    np.savez_compressed(output / "roma_correspondences.npz", photo_xy=photo_good,
                        render_xy=render_good, overlap=confidence[candidate_mask], homography=roma_h, **roma_arrays)
    np.savez_compressed(output / "verified_correspondences.npz", photo_xy=final_photo,
                        render_xy=final_render, world_xyz=final_world, surface_valid=final_surface_valid,
                        confidence=scores, confidence_type=score_type, homography=H, **arrays)
    with (output / "correspondences.csv").open("w",newline="",encoding="utf-8-sig") as f:
        writer=csv.writer(f)
        writer.writerow(["photo_x","photo_y","render_x","render_y","confidence","confidence_type","split","inlier",
                         "error_render_px","surface_valid","world_x","world_y","world_z"])
        for i,(a,b,xyz,score) in enumerate(zip(final_photo,final_render,final_world,scores)):
            writer.writerow([*a,*b,float(score),score_type,"train" if arrays["train_mask"][i] else "holdout",
                             bool(arrays["inlier_mask"][i]),float(arrays["errors_px"][i]),bool(final_surface_valid[i]),
                             *[float(v) if np.isfinite(v) else "" for v in xyz]])
    label = ("自动检查通过" if accepted else "自动检查未通过") + "；无人点选，真实距离精度未验证"
    draw_matches(photo,crop_image,photo_good,render_good,np.array(identity["crop_xyxy"][:2]),roma_arrays,output,
                 filename="roma_matches.png")
    draw_matches(photo,crop_image,final_photo,final_render,np.array(identity["crop_xyxy"][:2]),arrays,output,
                 title="自动匹配 / " + ("RoMa + 窗中心结构细化" if method == "roma_window_structure" else "RoMa"),
                 quality_label=label)
    draw_alignment(photo,render,identity["crop_xyxy"],H,None,output,quality_label=label)
    report_lines = [label, "选择方法：" + method,
                    "当前为已指定建筑和正面的单对图匹配，不包含建筑检索。",
                    "原RoMa自动几何检查：" + str(geometry["gate_passed"]),
                    "窗中心结构检查：" + str(structure_report.get("gate_passed", False)),
                    "选中方法未通过项：" + str(automatic.get("failed_checks", [])),
                    "主流程没有读取任何人工标定点；旧标定及三维建筑未修改。",
                    "空间留出检查衡量自动关联的一致性，不能排除规则窗列整体错位。",
                    "matches.png是选中方法；roma_matches.png是纯RoMa对照；alignment.png是最终叠加。",
                    "所有误差和门槛见 result.json；失败结果仍保存，但不视作自动接受。"]
    (output / "quality_report.txt").write_text("\n".join(report_lines)+"\n", encoding="utf-8")
    write_json(OUTPUT_ROOT / "latest_run.json", {"run_dir":str(output)})
    print("[6] 完成。", label, flush=True)
    print("    选择方法：", method, "；失败项：", automatic.get("failed_checks", []), flush=True)
    print("    请查看 matches.png、alignment.png、quality_report.txt 与 result.json。",flush=True)


if __name__ == "__main__":
    main()
