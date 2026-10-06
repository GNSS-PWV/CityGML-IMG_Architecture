"""照片→DINO检索→RoMa候选匹配→自动几何检查→门窗三维映射。

PyCharm直接运行；只改PHOTO_PATH即可换照片。建筑编号不从照片名读取。
已有检测按图片内容哈希复用；新照片通过Grounding DINO 1.6云API检测，会使用额度。
只对自动检查通过且候选差距足够的结果输出三维门窗；不改源CityGML，不开孔。
"""
from pathlib import Path
from datetime import datetime
import argparse
import csv
import hashlib
import inspect
import json
import os
import time

import numpy as np
from PIL import Image, ImageDraw

import match_facade_roma as roma
from facade_match_geometry import fit_and_check, roma_to_image_pixels
from facade_match_structure import refine_with_window_structure

ROOT = Path(__file__).resolve().parent
DATASET = next((p for p in (ROOT / "Drills/Texture2LoD3_dataset",
                           ROOT / "CityGML-IMG_Architecture/Drills/Texture2LoD3_dataset") if p.is_dir()),
               ROOT / "Drills/Texture2LoD3_dataset")
PHOTO_PATH = DATASET / "textures/4959323_front.jpg"
GALLERY_POINTER = ROOT / "my_results/render_gallery/latest_run.json"
OUTPUT_ROOT = ROOT / "my_results/facade_pipeline"
TOP_BUILDINGS = 5
VIEWS_PER_BUILDING = 2
MAX_CONFIRMATION_VIEWS = 2
# 固定公开的候选区分门槛，不是建筑正确概率，也不是米制精度保证。
MIN_CANDIDATE_MARGIN = .08
MIN_GEOMETRY_SCORE = .35


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def detection_for_photo(photo_path, output_dir):
    """只以图像哈希、尺寸、D参数复用预测，建筑身份字段不进入检索/选择。"""
    import try_grounding_dino16 as detector
    digest = roma.sha256(photo_path)
    with Image.open(photo_path) as photo:
        size = list(photo.size)
    detection_root = ROOT / "my_results/grounding_dino16/batch_detection"
    paths = sorted(detection_root.glob("*/**/detections.json"), reverse=True)
    # 此流程新照片的付费任务也固定存入内容哈希目录，重跑不会重复提交未知状态的任务。
    owned = OUTPUT_ROOT / "detection_cache" / digest / "detections.json"
    for path in [owned] + paths:
        if not path.is_file():
            continue
        data = read_json(path)
        if (data.get("image_sha256") == digest and data.get("image_size") == size
                and data.get("model") == detector.MODEL and data.get("parameters") == detector.experiment_parameters()
                and data.get("coordinate_system") == "original_image_pixels_xyxy"):
            write_json(output_dir / "detection_source.json", {"path": str(path), "sha256": roma.sha256(path),
                        "reused": True, "building_id_field_used": False})
            return data, path
    token = os.environ.get("DDS_API_TOKEN", "").strip()
    if not token:
        raise RuntimeError("新照片没有可复用检测。请在PyCharm运行配置设置DDS_API_TOKEN；新检测会使用平台额度。")
    directory = owned.parent
    item = {"image_path": str(photo_path), "image_key": photo_path.stem, "building_id": None,
            "image_sha256": digest, "image_size": size, "result_directory": str(directory)}
    print("[检测] 新照片提交Grounding DINO 1.6，使用平台额度；每块立即保存。", flush=True)
    with detector.requests.Session() as session:
        session.headers.update({"Token": token})
        detector.detect_one_image(session, item, "D")
    write_json(output_dir / "detection_source.json", {"path": str(owned), "sha256": roma.sha256(owned),
                "reused": False, "building_id_field_used": False})
    return read_json(owned), owned


def wall_crops(view, manifest, objects, depth):
    """从可见像素取最多两面大墙；同一张图的两墙不能混进一个平面拟合。"""
    walls = sorted(view["visible_walls"], key=lambda w: w["pixel_count"], reverse=True)
    if not walls:
        return []
    result = []
    for wall in walls[:2]:
        if wall["pixel_count"] < max(500, walls[0]["pixel_count"] * .30):
            continue
        allowed = [o["index"] for o in manifest["objects"] if o.get("wall_id") == wall["wall_id"]]
        ys, xs = np.where(np.isin(objects, allowed) & np.isfinite(depth))
        if len(xs) == 0 or np.ptp(xs) < 32 or np.ptp(ys) < 32:
            continue
        h, w = depth.shape
        crop = [max(0, int(xs.min()) - 4), max(0, int(ys.min()) - 4),
                min(w, int(xs.max()) + 5), min(h, int(ys.max()) + 5)]
        result.append((wall["wall_id"], allowed, crop))
    return result


def geometry_score(report):
    """几何分数用于排序，不用DINO相似度覆盖几何失败，也不当成置信概率。"""
    fraction = report.get("holdout_inlier_fraction", 0.)
    hull = report.get("train_photo_coverage", {}).get("convex_hull_fraction", 0.)
    return float(fraction * min(1., hull / .60))


def select_candidate(candidates):
    """建筑/墙面有接近的竞争者时拒绝自动选定；同墙不同视角合为一个身份。"""
    viable = [c for c in candidates if c.get("gate_passed") and c.get("wall_vote", {}).get("accepted")]
    if not viable:
        return None, {"accepted": False, "reason": "no_candidate_passed_geometry"}
    best = max(viable, key=lambda c: c["geometry_score"])
    alternative = [c for c in candidates if (c["building_id"], c.get("wall_id")) !=
                   (best["building_id"], best["wall_id"]) and "geometry_score" in c]
    runner = max(alternative, key=lambda c: c["geometry_score"]) if alternative else None
    margin = best["geometry_score"] - runner["geometry_score"] if runner else best["geometry_score"]
    accepted = best["geometry_score"] >= MIN_GEOMETRY_SCORE and margin >= MIN_CANDIDATE_MARGIN
    return (best if accepted else None), {
        "accepted": bool(accepted), "reason": "automatic_checks_passed" if accepted else "ambiguous_candidates",
        "best_candidate_id": best["candidate_id"], "runner_up_id": runner["candidate_id"] if runner else None,
        "geometry_score": best["geometry_score"], "margin": margin,
        "min_score": MIN_GEOMETRY_SCORE, "min_margin": MIN_CANDIDATE_MARGIN,
        "independent_accuracy_verified": False}


def check_view_agreement(selected, candidates, photo_size):
    """必须有同墙第二视角通过且回投一致；单视图不能默认放行。"""
    from facade_auto_mapping import render_pixels_to_wall, validated_wall
    from facade_match_geometry import project_homography
    others = [c for c in candidates if c.get("gate_passed") and c.get("wall_vote", {}).get("accepted")
              and c["building_id"] == selected["building_id"]
              and c["wall_id"] == selected["wall_id"] and c["view_name"] != selected["view_name"]]
    report = {"available": bool(others), "passed": bool(others), "comparisons": [],
              "median_limit_m": .30, "p90_limit_m": .75, "independent_accuracy_verified": False}
    if not others:
        report["reason"] = "insufficient_cross_view_confirmation"
        return report
    manifest = read_json(selected["view"]["manifest"])
    try:
        wall = validated_wall(manifest, selected["wall_id"])
    except ValueError as exc:
        report.update({"passed": False, "reason": "wall_not_supported_by_planar_reader", "error": str(exc)})
        return report
    xx, yy = np.meshgrid(np.linspace(.1, .9, 9)*photo_size[0], np.linspace(.1, .9, 5)*photo_size[1])
    points = np.column_stack((xx.ravel(), yy.ravel()))

    def world(row):
        xy = project_homography(points, row["homography_photo_to_render"])
        camera = read_json(row["view"]["camera"])
        xyz, _, _ = render_pixels_to_wall(xy, camera, wall["frame"], manifest["world_origin_m"])
        with np.load(row["view"]["geometry"], allow_pickle=False) as source:
            objects = source["object_index"]
        ij = np.floor(xy+.5).astype(int)
        valid = (ij[:, 0]>=0) & (ij[:, 1]>=0) & (ij[:, 0]<objects.shape[1]) & (ij[:, 1]<objects.shape[0])
        allowed = [o["index"] for o in manifest["objects"] if o.get("wall_id") == selected["wall_id"]]
        valid[valid] &= np.isin(objects[ij[valid, 1], ij[valid, 0]], allowed)
        return xyz, valid
    first, visible = world(selected)
    for row in others:
        second, other_visible = world(row)
        errors = np.linalg.norm(first[visible & other_visible]-second[visible & other_visible], axis=1)
        median = float(np.median(errors)) if len(errors) else None
        p90 = float(np.percentile(errors, 90)) if len(errors) else None
        passed = len(errors)>=6 and median<=.30 and p90<=.75
        report["comparisons"].append({"candidate_id": row["candidate_id"], "point_count": len(errors),
                                      "median_m": median, "p90_m": p90, "passed": bool(passed)})
        report["passed"] &= bool(passed)
    return report


def confirmation_views(selected, candidates, ranked_views):
    """从图库补找同一墙的大可见视角，重新独立匹配，不由首个H构造对应点。"""
    seen = {(c["building_id"], c["wall_id"], c["view_name"]) for c in candidates}
    choices = []
    for view in ranked_views:
        if view["building_id"] != selected["building_id"]:
            continue
        if (view["building_id"], selected["wall_id"], view["view_name"]) in seen:
            continue
        walls = sorted(view["visible_walls"], key=lambda w: w["pixel_count"], reverse=True)
        target = next((w for w in walls[:2] if w["wall_id"] == selected["wall_id"]), None)
        if target and target["pixel_count"] >= max(500, walls[0]["pixel_count"]*.30):
            choices.append((target["pixel_count"], view))
    return [v for _, v in sorted(choices, key=lambda item: (-item[0], item[1]["view_name"]))[:MAX_CONFIRMATION_VIEWS]]


def tiled_cache_identity(initial_h):
    """分块策略和函数均参与缓存身份，改参数后必须真正重新匹配。"""
    import try_grounding_dino16 as detector
    return {"initial_h": np.asarray(initial_h).tolist(),
            "tile_size": [detector.TILE_WIDTH, detector.TILE_HEIGHT], "tile_overlap": detector.TILE_OVERLAP,
            "minimum_overlap": roma.MIN_OVERLAP,
            "tile_views_code": hashlib.sha256(inspect.getsource(detector.tile_views).encode()).hexdigest(),
            "code": hashlib.sha256(inspect.getsource(roma.tiled_correspondences).encode()).hexdigest()}


def candidate_match(photo, detections, view, wall_id, allowed, crop, manifest,
                    objects, depth, model, model_info, photo_hash, directory):
    """一面候选墙上的RoMa拟合和独立空间留出；全图坐标贯穿后续映射。"""
    from facade_auto_mapping import choose_wall_from_matches
    render = Image.open(view["image"]).convert("RGB")
    crop_image = render.crop(crop)
    identity = {"photo_sha256": photo_hash, "render_sha256": roma.sha256(view["image"]),
                "wall_id": wall_id,
                "geometry_sha256": roma.sha256(view["geometry"]), "camera_sha256": roma.sha256(view["camera"]),
                "manifest_sha256": roma.sha256(view["manifest"]), "crop": crop,
                "model": model_info["checkpoint_sha256"], "setting": roma.SETTING,
                "implementation": model_info["implementation"],
                "sample_count": roma.NUM_MATCHES, "seed": roma.SEED,
                "inference_code_sha256": roma.sha256(Path(roma.__file__))}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache = OUTPUT_ROOT / "matches_cache" / key
    cache.mkdir(parents=True, exist_ok=True)
    if (cache / "raw.npz").is_file() and (cache / "identity.json").is_file() and read_json(cache / "identity.json") == identity:
        with np.load(cache / "raw.npz", allow_pickle=False) as values:
            normalized, confidence = values["matches"], values["overlap"]
    else:
        normalized, confidence = roma.infer_pair(model, photo, crop_image)
        np.savez_compressed(cache / "raw.npz", matches=normalized, overlap=confidence)
        write_json(cache / "identity.json", identity)
    a, b = roma_to_image_pixels(normalized, photo.size, crop_image.size, crop[:2])
    valid = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1) & (confidence >= roma.MIN_OVERLAP)
    ix = np.floor(np.nan_to_num(b) + .5).astype(int)
    inside = (ix[:, 0] >= 0) & (ix[:, 1] >= 0) & (ix[:, 0] < render.width) & (ix[:, 1] < render.height)
    on_wall = np.zeros(len(a), bool)
    on_wall[inside] = np.isin(objects[ix[inside, 1], ix[inside, 0]], allowed)
    a, b = a[valid & on_wall], b[valid & on_wall]
    H, report, arrays = fit_and_check(a, b, photo.size, render.size, roma.RANSAC_THRESHOLD_PX)
    coarse_h, coarse_report = H.copy(), report
    dense_method, tiled_report = "roma", {"attempted": False}
    # 长图整张缩到RoMa输入尺寸后细节不足：仅用有空间证据的粗匹配确定局部搜索范围。
    # 不降低验收门槛，也不用建筑真值或照片名决定是否放大。
    if (max(photo.size)>1400 and report["train_inlier_count"]>=40 and
            report["holdout_inlier_fraction"]>=.20 and report["checks"]["no_projective_pole_in_photo"]):
        tiled_identity = tiled_cache_identity(H)
        tiled_key = hashlib.sha256(json.dumps(tiled_identity, sort_keys=True).encode()).hexdigest()
        tiled_cache = cache / ("tiles_"+tiled_key+".npz")
        if tiled_cache.is_file():
            with np.load(tiled_cache, allow_pickle=False) as saved:
                fa, fb, fc = saved["photo_xy"], saved["render_xy"], saved["confidence"]
                tile_info = json.loads(str(saved["tile_info"].item()))
        else:
            fa, fb, fc, tile_info = roma.tiled_correspondences(model, photo, render, H, np.isin(objects, allowed))
            np.savez_compressed(tiled_cache, photo_xy=fa, render_xy=fb, confidence=fc,
                                tile_info=np.array(json.dumps(tile_info, ensure_ascii=False)))
        tiled_report = {"attempted": True, "tiles": tile_info, "raw_cache": str(tiled_cache)}
        try:
            fine_h, fine_report, fine_arrays = fit_and_check(fa, fb, photo.size, render.size, roma.RANSAC_THRESHOLD_PX)
            tiled_report["validation"] = fine_report
            np.savez_compressed(directory/"tiled_matches.npz", photo_xy=fa, render_xy=fb, **fine_arrays)
            roma.draw_matches(photo, crop_image, fa, fb, crop[:2], fine_arrays, directory,
                              filename="tiled_matches.png", quality_label="局部放大几何检查："+("通过" if fine_report["gate_passed"] else "未通过"))
            if fine_report["gate_passed"]:
                H, report, arrays, a, b, dense_method = fine_h, fine_report, fine_arrays, fa, fb, "roma_tiled"
        except ValueError as exc:
            tiled_report["error"] = str(exc)
    vote = choose_wall_from_matches(b[arrays["inlier_mask"]], {"depth_m": depth, "object_index": objects}, manifest)
    refined_h, structure, struct_arrays = refine_with_window_structure(H, photo.size, render.size,
                                                                detections, objects, manifest, wall_id)
    selected_h, method, selected_report = roma.choose_automatic_transform(H, report, refined_h, structure)
    uses_structure = method == "roma_window_structure"
    method = dense_method + ("_window_structure" if uses_structure else "")
    # 排序仍使用原始RoMa留出证据，不能因为少量结构点被挑得好就压倒其他建筑。
    result = {"wall_id": wall_id, "wall_vote": vote, "gate_passed": bool(selected_report["gate_passed"]),
              "geometry_score": geometry_score(report), "selected_method": method,
              "homography_photo_to_render": selected_h.tolist(), "roma_validation": report,
              "roma_homography_photo_to_render": H.tolist(),
              "coarse_roma_homography": coarse_h.tolist(), "coarse_roma_validation": coarse_report,
              "tiled_roma": tiled_report,
              "structure_validation": structure, "raw_matches": str(cache / "raw.npz"), "input_identity": identity,
              "crop_xyxy": crop, "independent_accuracy_verified": False}
    np.savez_compressed(directory / "roma_matches.npz", photo_xy=a, render_xy=b, **arrays)
    roma.draw_matches(photo, crop_image, a, b, crop[:2], arrays, directory, filename="roma_matches.png",
                      quality_label="原RoMa几何检查：" + ("通过" if report["gate_passed"] else "未通过"))
    if uses_structure:
        a, b = struct_arrays["photo_xy"], struct_arrays["render_xy"]
        arrays = {k: v for k, v in struct_arrays.items() if k not in ("photo_xy", "render_xy")}
    np.savez_compressed(directory / "geometry_matches.npz", photo_xy=a, render_xy=b, **arrays)
    label = "自动几何检查通过；真实精度未验证" if result["gate_passed"] else "自动几何检查未通过"
    roma.draw_matches(photo, crop_image, a, b, crop[:2], arrays, directory, quality_label=label)
    roma.draw_alignment(photo, render, crop, selected_h, None, directory, quality_label=label)
    return result


def draw_retrieval(photo_path, candidates, output):
    """直接拼接真实输入与候选渲染，不重绘或改变实验内容。"""
    width, height = 480, 340
    canvas = Image.new("RGB", (width * 3, height * ((len(candidates) + 3) // 3)), roma.BG)
    draw = ImageDraw.Draw(canvas)
    entries = [("输入照片", photo_path)] + [(f"{c['building_id']} / {c['view_name']}", c["image"]) for c in candidates]
    for i, (label, path) in enumerate(entries):
        left, top = i % 3 * width, i // 3 * height
        image = Image.open(path).convert("RGB")
        image.thumbnail((width - 24, height - 62))
        canvas.paste(image, (left + (width-image.width)//2, top + 48 + (height-62-image.height)//2))
        draw.text((left+14, top+12), label, fill=roma.INK, font=roma.font(22))
    canvas.save(output)


def run_pipeline(photo_path=PHOTO_PATH, top_buildings=TOP_BUILDINGS, views_per_building=VIEWS_PER_BUILDING):
    from facade_retrieval import retrieve_gallery
    from facade_auto_mapping import map_detections
    photo_path = Path(photo_path).resolve()
    if not photo_path.is_file():
        raise FileNotFoundError(photo_path)
    if not GALLERY_POINTER.is_file():
        raise FileNotFoundError("请先运行build_render_gallery.py建立多建筑渲染图库。")
    gallery = Path(read_json(GALLERY_POINTER)["run_dir"]) / "gallery_index.json"
    started = time.perf_counter()
    output = OUTPUT_ROOT / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True)
    photo_hash = roma.sha256(photo_path)
    status = {"status": "running", "photo": str(photo_path), "photo_sha256": photo_hash,
              "gallery": str(gallery), "gallery_sha256": roma.sha256(gallery), "output_dir": str(output),
              "filename_building_id_used": False, "manual_calibration_used": False, "gt_masks_used": False}
    status["source_sha256"] = {name: roma.sha256(ROOT/name) for name in (
        "run_facade_pipeline.py", "facade_retrieval.py", "facade_auto_mapping.py", "match_facade_roma.py",
        "facade_match_geometry.py", "facade_match_structure.py", "map_facade_to_3d.py", "try_grounding_dino16.py")}
    write_json(output / "result.json", status)
    write_json(OUTPUT_ROOT / "latest_run.json", {"run_dir": str(output)})
    try:
        print("[1/5] 准备门窗检测（已有相同照片优先复用）", flush=True)
        detections, detection_path = detection_for_photo(photo_path, output)
        print("[2/5] DINOv3检索多建筑图库", flush=True)
        retrieval = retrieve_gallery(photo_path, gallery, OUTPUT_ROOT / "retrieval_cache", roma.MODEL_CACHE,
                                     top_k_buildings=top_buildings, views_per_building=views_per_building)
        write_json(output / "retrieval.json", retrieval)
        views = retrieval["selected_candidates"]
        draw_retrieval(photo_path, views, output / "retrieval.png")
        print("[3/5] RoMa逐个匹配候选", flush=True)
        model, model_info = roma.load_roma_model()
        write_json(output / "model.json", model_info)
        photo = Image.open(photo_path).convert("RGB")
        candidates = []
        def evaluate_view(view, target_wall=None, source="retrieval"):
            manifest = read_json(view["manifest"])
            if roma.sha256(manifest["source_gml"]) != manifest["source_sha256"]:
                raise ValueError("图库源模型已变化，请重新建立图库。")
            with np.load(view["geometry"], allow_pickle=False) as arrays:
                objects, depth = arrays["object_index"], arrays["depth_m"]
            for wall_id, allowed, crop in wall_crops(view, manifest, objects, depth):
                if target_wall is not None and wall_id != target_wall:
                    continue
                number = len(candidates) + 1
                directory = output / f"candidate_{number:02d}"
                directory.mkdir()
                row = {"candidate_id": directory.name, "building_id": view["building_id"],
                       "wall_id": wall_id, "view_name": view["view_name"], "view": view,
                       "directory": str(directory), "candidate_source": source}
                print(f"    候选{number}: {view['building_id']} / {view['view_name']}", flush=True)
                try:
                    from facade_auto_mapping import validated_wall
                    validated_wall(manifest, wall_id)
                    row.update(candidate_match(photo, detections, view, wall_id, allowed, crop, manifest,
                                               objects, depth, model, model_info, photo_hash, directory))
                except ValueError as exc:
                    row.update({"gate_passed": False, "error": str(exc)})
                write_json(directory / "result.json", row)
                candidates.append(row)
                write_json(output / "candidates.json", candidates)
        for view in views:
            evaluate_view(view)
        # DINO的前两个视角可能分别看见不同墙；不能把缺少同墙第二视角当成通过。
        preliminary, _ = select_candidate(candidates)
        if preliminary is not None:
            confirmed = any(c.get("gate_passed") and c.get("wall_vote", {}).get("accepted") and
                            c["building_id"] == preliminary["building_id"] and c["wall_id"] == preliminary["wall_id"] and
                            c["view_name"] != preliminary["view_name"] for c in candidates)
            if not confirmed:
                for view in confirmation_views(preliminary, candidates, retrieval["ranked_views"]):
                    evaluate_view(view, preliminary["wall_id"], "same_wall_confirmation")
        del model
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[4/5] 对比候选建筑、墙面与几何证据", flush=True)
        selected, decision = select_candidate(candidates)
        if selected is not None:
            try:
                agreement = check_view_agreement(selected, candidates, photo.size)
            except ValueError as exc:
                agreement = {"available": False, "passed": False, "reason": "invalid_wall_projection", "error": str(exc)}
            decision["cross_view_agreement"] = agreement
            if not agreement["passed"]:
                decision.update({"accepted": False, "reason": agreement.get("reason", "views_disagree_on_world_position")})
                selected = None
        status.update({"selection": decision, "candidate_count": len(candidates),
                       "detection_source": str(detection_path), "independent_accuracy_verified": False})
        if selected is None:
            status["status"] = "needs_review"
            status["mapping"] = None
            print("    证据不足，不强行确定建筑、不输出三维门窗。", flush=True)
        else:
            print("[5/5] 门窗框投到选中三维墙面", flush=True)
            view = selected["view"]
            try:
                mapping = map_detections(detections, selected["homography_photo_to_render"], read_json(view["camera"]),
                                         view["geometry"], read_json(view["manifest"]), selected["wall_id"], output / "mapping_3d")
            except ValueError as exc:
                decision.update({"accepted": False, "reason": "invalid_mapping_geometry", "error": str(exc)})
                status.update({"status": "needs_review", "mapping": None})
            else:
                status.update({"status": "mapped" if mapping["mapped_count"] else "no_detections_mapped", "building_id": selected["building_id"],
                               "wall_id": selected["wall_id"], "view_name": selected["view_name"], "mapping": mapping,
                               "selected_candidate": selected["candidate_id"]})
        with (output / "candidates.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            fields = ["candidate_id", "building_id", "view_name", "wall_id", "geometry_score", "gate_passed", "error"]
            writer = csv.DictWriter(stream, fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(candidates)
        status["elapsed_seconds"] = round(time.perf_counter()-started, 2)
        write_json(output / "result.json", status)
        print("结果：", status["status"], "；目录：", output, flush=True)
        return output
    except Exception as exc:
        status.update({"status": "failed", "error_type": type(exc).__name__})
        # 不将HTTP请求对象、Token或照片base64写入日志。
        write_json(output / "result.json", status)
        exc.pipeline_run_dir = str(output)
        raise


def run_photo_directory(directory, top_buildings=TOP_BUILDINGS, views_per_building=VIEWS_PER_BUILDING):
    """批量入口每张立即落盘；一张异常保留记录并继续，不重复提交未知的付费任务。"""
    directory = Path(directory).resolve()
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    photos = sorted(p for p in directory.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png") and p.is_file())
    if not photos:
        raise ValueError("目录中没有 JPG/PNG 照片。请指定真实照片目录，不能选择标注掩码目录。")
    batch = OUTPUT_ROOT / datetime.now().strftime("batch_%Y%m%d_%H%M%S_%f")
    batch.mkdir(parents=True)
    summary = {"photo_directory": str(directory), "photo_count": len(photos), "records": []}
    for i, photo in enumerate(photos, 1):
        print(f"\n批量 {i}/{len(photos)}：{photo.name}", flush=True)
        try:
            output = run_pipeline(photo, top_buildings, views_per_building)
            result = read_json(output/"result.json")
            row = {"photo": photo.name, "run_dir": str(output), "status": result["status"],
                   "building_id": result.get("building_id"), "mapped_count": (result.get("mapping") or {}).get("mapped_count", 0)}
        except Exception as exc:
            row = {"photo": photo.name, "status": "failed", "error_type": type(exc).__name__,
                   "run_dir": getattr(exc, "pipeline_run_dir", None)}
            print("    本张失败："+type(exc).__name__+"；已有结果保留。", flush=True)
        summary["records"].append(row)
        write_json(batch/"summary.json", summary)
    print("批次汇总：", batch/"summary.json", flush=True)
    return batch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--photo", type=Path, default=PHOTO_PATH)
    inputs.add_argument("--photo-dir", type=Path, help="批量处理一个真实照片目录，每张保存；新检测仍需额度")
    parser.add_argument("--top-buildings", type=int, default=TOP_BUILDINGS)
    parser.add_argument("--views-per-building", type=int, default=VIEWS_PER_BUILDING)
    arguments = parser.parse_args()
    if arguments.photo_dir is not None:
        run_photo_directory(arguments.photo_dir, arguments.top_buildings, arguments.views_per_building)
    else:
        run_pipeline(arguments.photo, arguments.top_buildings, arguments.views_per_building)
