"""视觉模型窗户对应实验：照片检测框 -> 0°渲染窗 -> 三维预览。

PyCharm 直接运行，先用 MODE="prepare" 生成带编号的两张图。视觉模型只输出
编号对应，不输出像素坐标或建筑真值；像素中心、单应拟合、留出验证和三维回投
均由本地代码完成。MODE="api" 会发起收费的 OpenAI 图像请求，必须显式修改。
本入口只处理一张照片的一个候选墙；单个候选通过不构成跨建筑唯一性证明。
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from facade_auto_mapping import map_detections
from facade_match_geometry import project_homography


# ===== PyCharm 中只需先改这三个配置 =====
ROOT = Path(__file__).resolve().parent
RUN_DIR = (ROOT.parent / "my_results/experiment_front0/"
           "experiment_front0_20261010_172758_423064/experiment_2_runs/"
           "run_20261010_192522_899052")  # 示例：4959323_front；换照片时换成其 run_... 目录。
CANDIDATE_RANK = 1  # retrieval.json 中的候选序号，从 1 开始；不使用照片文件名选建筑。
MODE = "prepare"  # prepare：生成编号图；apply：读取 pairs.json；api：收费请求后应用。

OUTPUT_ROOT = ROOT.parent / "my_results/vlm_window_alignment"
TRIAL_NAME = "my_trial"  # 改这个名字可新开一次试验，避免覆盖已有配对结果。
OPENAI_MODEL = "gpt-6-astra"  # 可用性取决于你的 API 账号；可改为账号支持的视觉模型。
MIN_WINDOW_SCORE = 0.20


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_font(size):
    for name in ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/msyh.ttc"):
        if Path(name).is_file():
            return ImageFont.truetype(name, size)
    return ImageFont.load_default()


def numbered_image(source, items, path, color):
    image = Image.open(source).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = image_font(max(15, min(24, image.width // 125)))
    line_width = max(2, image.width // 1000)
    for item in items:
        x0, y0, x1, y1 = item["box"]
        draw.rectangle((x0, y0, x1, y1), outline=color, width=line_width)
        label = item["id"]
        bounds = draw.textbbox((0, 0), label, font=font)
        tw, th = bounds[2] - bounds[0], bounds[3] - bounds[1]
        lx = max(0, min(image.width - tw - 6, round(x0)))
        ly = max(0, min(image.height - th - 6, round(y0) - th - 8))
        draw.rectangle((lx, ly, lx + tw + 5, ly + th + 5), fill=color)
        draw.text((lx + 2, ly + 1), label, font=font, fill="white")
    image.save(path)


def load_inputs(run_dir, candidate_rank):
    run_dir = Path(run_dir).resolve()
    result = read_json(run_dir / "result.json")
    photo_path = Path(result["photo"])
    retrieval = read_json(run_dir / "retrieval.json")
    candidates = retrieval["selected_candidates"]
    if not 1 <= candidate_rank <= len(candidates):
        raise ValueError(f"候选序号必须在 1～{len(candidates)} 之间")
    view = candidates[candidate_rank - 1]
    if float(view.get("orbit_deg", 999)) != 0:
        raise ValueError("本实验只使用 0° 正视渲染图")
    if retrieval["query_image_sha256"] != sha256(photo_path):
        raise ValueError("检索记录与原照片内容不一致")
    detection_path = Path(read_json(run_dir / "detection_source.json")["path"])
    detections = read_json(detection_path)
    if detections["image_sha256"] != sha256(photo_path):
        raise ValueError("检测框与原照片内容不一致")
    with Image.open(photo_path) as photo:
        photo_size = photo.size
    if detections["image_size"] != list(photo_size):
        raise ValueError("检测框不是原照片像素坐标")
    manifest = read_json(view["manifest"])
    camera = read_json(view["camera"])
    with np.load(view["geometry"], allow_pickle=False) as arrays:
        objects = arrays["object_index"]
    if list(camera["image_size_wh"]) != [objects.shape[1], objects.shape[0]]:
        raise ValueError("渲染相机与构件编号图尺寸不一致")
    if manifest["source_sha256"] != sha256(manifest["source_gml"]):
        raise ValueError("源 CityGML 已改变，旧图库不能继续使用")
    wall = max(view["visible_walls"], key=lambda item: item["pixel_count"])
    return photo_path, view, manifest, camera, objects, wall["wall_id"], detections


def photo_windows(detections):
    width, height = detections["image_size"]
    rows = []
    for index, detection in enumerate(detections["detections"]):
        if detection.get("class") != "window" or float(detection.get("score", 0)) < MIN_WINDOW_SCORE:
            continue
        box = np.asarray(detection["box"], dtype=float)
        if (box.shape != (4,) or not np.isfinite(box).all() or
                (box[2:] <= box[:2]).any() or (box[:2] < 0).any() or
                box[2] > width or box[3] > height or
                box[0] <= 0 or box[1] <= 0 or box[2] >= width or box[3] >= height):
            continue  # 边界截断/无效框不用于建立对应，但仍保留在原检测结果中。
        rows.append({"detection_index": index, "box": box.tolist(),
                     "center": ((box[:2] + box[2:]) / 2).tolist()})
    rows.sort(key=lambda row: (row["center"][1], row["center"][0]))
    for index, row in enumerate(rows, 1):
        row["id"] = f"P{index:02d}"
    return rows


def render_windows(manifest, wall_id, object_map):
    rows = []
    for obj in manifest["objects"]:
        if obj.get("kind") != "Window" or obj.get("wall_id") != wall_id:
            continue
        ys, xs = np.where(object_map == int(obj["index"]))
        if not len(xs):
            continue
        box = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        rows.append({"object_index": int(obj["index"]), "box": box,
                     "center": [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]})
    rows.sort(key=lambda row: (row["center"][1], row["center"][0]))
    for index, row in enumerate(rows, 1):
        row["id"] = f"R{index:02d}"
    return rows


PROMPT = """第一张图是真实照片中的检测窗 P 编号；第二张图是某一候选建筑墙面的0°渲染窗 R 编号。
判断它们是否为同一面墙。重点比较楼层、窗列、拱门/入口、边缘与不对称结构；
重复窗列可能整体错移一列，不能只凭图案相似就认定。
如果不是或无法确定，same_wall 分别写 no 或 uncertain，pairs 返回空数组。
如果能够明确对应，请只列高把握的 P-R 编号对，最好至少8对，覆盖左右和上下；
同一个 P 或 R 只能使用一次。不确定的窗不要硬配。不要输出或猜测像素坐标。
图片上 P/R 标签的位置是定位辅助，输出须使用确切的标签编号。"""


RESPONSE_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "same_wall": {"type": "string", "enum": ["yes", "no", "uncertain"]},
        "reason": {"type": "string"},
        "pairs": {"type": "array", "items": {"type": "object", "additionalProperties": False,
            "properties": {"photo": {"type": "string"}, "render": {"type": "string"}},
            "required": ["photo", "render"]}},
    },
    "required": ["same_wall", "reason", "pairs"],
}


def call_openai(photo_labeled, render_labeled, output):
    """只有显式 MODE=api 才调用；请求前落盘，未知结果时禁止自动重试扣费。"""
    import requests

    cached = output / "api_pairs.json"
    if cached.is_file():
        print("已复用 api_pairs.json，没有再次调用 API")
        return read_json(cached)
    marker = output / "api_request_started.json"
    if marker.exists():
        raise RuntimeError("上次 API 请求结果不明；先查平台调用记录，再决定是否重试。")
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("PyCharm 运行配置缺少 OPENAI_API_KEY；prepare 模式无需 API key。")
    def image_part(path):
        encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
        return {"type": "input_image", "image_url": "data:image/png;base64," + encoded,
                "detail": "original"}
    payload = {"model": OPENAI_MODEL, "store": False,
               "input": [{"role": "user", "content": [
                   {"type": "input_text", "text": PROMPT},
                   image_part(photo_labeled), image_part(render_labeled)]}],
               "text": {"format": {"type": "json_schema", "name": "window_pairs",
                                   "strict": True, "schema": RESPONSE_SCHEMA}}}
    write_json(marker, {"model": OPENAI_MODEL, "status": "request_started",
                        "input_identity": read_json(output / "input_identity.json")})
    response = requests.post("https://api.openai.com/v1/responses", json=payload,
                             headers={"Authorization": "Bearer " + key}, timeout=180)
    response.raise_for_status()
    body = response.json()
    if body.get("status") != "completed":
        raise RuntimeError("API 请求尚未完整结束；已保留请求标记，先核对平台记录。")
    chunks = [part.get("text", "") for item in body.get("output", [])
              if item.get("type") == "message" for part in item.get("content", [])
              if part.get("type") == "output_text"]
    if len(chunks) != 1:
        raise RuntimeError("API 未返回唯一的结构化文本；已保留请求标记，避免自动重试。")
    pairs = json.loads(chunks[0])
    write_json(cached, pairs)
    write_json(marker, {"model": OPENAI_MODEL, "status": "response_saved",
                        "response_id": body.get("id"),
                        "input_identity": read_json(output / "input_identity.json")})
    return pairs


def fit_window_pairs(answer, photo_rows, render_rows, photo_size, render_size):
    """本地以真实窗框中心拟合变换；至少两对完全留出。

    6–7 对时用仿射，避免只有四个训练点却拟合八自由度单应矩阵；
    8 对以上用单应处理照片的透视差。两者都返回 3×3 矩阵。
    这里只用窗框中心，不从编号标签或图像像素中反推“人工点”。
    留出的点不会参与模型估计，返回的 H 也始终是仅用训练锚点得到的 H。
    """
    report = {"accepted": False, "method": "window ID pairs -> transform RANSAC -> holdout",
              "independent_accuracy_verified": False,
              "note": "留出窗与训练窗来自同一配对来源；几何自洽不能证明建筑身份正确。"}
    if answer.get("same_wall") != "yes":
        report["reason"] = "vision_model_rejected_or_uncertain"
        return None, report
    photo_by_id = {row["id"]: row for row in photo_rows}
    render_by_id = {row["id"]: row for row in render_rows}
    rows = answer.get("pairs", [])
    ids_photo = [row.get("photo") for row in rows]
    ids_render = [row.get("render") for row in rows]
    if (len(rows) < 6 or len(set(ids_photo)) != len(rows) or
            len(set(ids_render)) != len(rows) or
            any(p not in photo_by_id for p in ids_photo) or
            any(r not in render_by_id for r in ids_render)):
        report["reason"] = "need_at_least_6_unique_known_pairs"
        return None, report
    photo = np.asarray([photo_by_id[p]["center"] for p in ids_photo], dtype=np.float64)
    render = np.asarray([render_by_id[r]["center"] for r in ids_render], dtype=np.float64)
    spans = np.ptp(photo, axis=0) / np.asarray(photo_size, dtype=float)
    if spans[0] < .55 or spans[1] < .30:
        report.update({"reason": "pairs_do_not_cover_wall", "photo_span_fraction": spans.tolist()})
        return None, report
    # x 排序后每第三对留出；留出点绝不参与 RANSAC 或重拟合。
    order = np.lexsort((photo[:, 1], photo[:, 0]))
    holdout = order[2::3]
    train = np.array([i for i in range(len(rows)) if i not in set(holdout)], dtype=int)
    if len(train) < 4 or len(holdout) < 2 or np.linalg.matrix_rank(photo[train] - photo[train].mean(0)) < 2:
        report["reason"] = "insufficient_spread_after_holdout"
        return None, report
    threshold = max(8., .01 * render_size[0])
    cv2.setRNGSeed(0)
    if len(rows) < 8:
        affine, solver_mask = cv2.estimateAffine2D(
            photo[train], render[train], method=cv2.RANSAC,
            ransacReprojThreshold=threshold, maxIters=10000, confidence=.999)
        H = np.vstack((affine, [0., 0., 1.])) if affine is not None else None
        report["method"] = "window ID pairs -> affine RANSAC -> holdout"
    else:
        H, solver_mask = cv2.findHomography(
            photo[train], render[train], cv2.RANSAC,
            ransacReprojThreshold=threshold, maxIters=10000, confidence=.999)
        report["method"] = "window ID pairs -> homography RANSAC -> holdout"
    if (H is None or not np.isfinite(H).all() or abs(H[2, 2]) < 1e-12 or
            np.linalg.matrix_rank(H) < 3):
        report["reason"] = "homography_fit_failed"
        return None, report
    H = H / H[2, 2]
    errors = np.linalg.norm(project_homography(photo, H) - render, axis=1)
    train_inliers = int(np.sum(errors[train] <= threshold))
    holdout_limit = (max(12., .015 * render_size[0]) if len(rows) < 8
                     else max(18., .025 * render_size[0]))
    holdout_errors = errors[holdout]
    corners = np.asarray([[0, 0], [photo_size[0], 0],
                          [photo_size[0], photo_size[1]], [0, photo_size[1]]], dtype=float)
    mapped_corners = project_homography(corners, H)
    denominator = np.column_stack((corners, np.ones(4))) @ H[2]
    signed_area = float(np.sum(mapped_corners[:, 0] * np.roll(mapped_corners[:, 1], -1)
                               - mapped_corners[:, 1] * np.roll(mapped_corners[:, 0], -1)) / 2)
    checks = {
        "train_inliers_at_least_4": train_inliers >= 4,
        "train_inlier_fraction_at_least_0_70": train_inliers / len(train) >= .70,
        "all_holdout_errors_within_limit": bool(np.all(holdout_errors <= holdout_limit)),
        "no_projective_pole_or_mirror": bool((denominator > 1e-8).all() and signed_area > 1.),
        "wall_inside_render_bounds": bool(np.all(mapped_corners >= -np.asarray(render_size) * .15) and
            np.all(mapped_corners <= np.asarray(render_size) * 1.15)),
    }
    report.update({"accepted": all(checks.values()), "reason": "checks_passed" if all(checks.values())
                   else "geometry_checks_failed", "checks": checks,
                   "pair_count": len(rows), "train_count": len(train), "holdout_count": len(holdout),
                   "train_inlier_count": train_inliers, "train_threshold_px": threshold,
                   "holdout_limit_px": holdout_limit, "holdout_errors_px": holdout_errors.tolist(),
                   "pair_errors_px": errors.tolist(), "photo_span_fraction": spans.tolist(),
                   "mapped_photo_signed_area_px2": signed_area,
                   "matrix_photo_to_render": H.tolist(), "solver_train_inliers": int(solver_mask.sum())})
    return H, report


def save_overlay(photo_path, render_path, H, output):
    photo = np.asarray(Image.open(photo_path).convert("RGB"))
    render = np.asarray(Image.open(render_path).convert("RGB"))
    warped = cv2.warpPerspective(photo, H, (render.shape[1], render.shape[0]))
    valid = cv2.warpPerspective(np.full(photo.shape[:2], 255, dtype=np.uint8), H,
                               (render.shape[1], render.shape[0])) > 0
    blend = render.copy()
    blend[valid] = ((render[valid].astype(np.uint16) + warped[valid].astype(np.uint16)) // 2).astype(np.uint8)
    Image.fromarray(blend).save(output)


def save_center_overlay(photo_path, render_path, H, answer, photo_rows, render_rows, report, output):
    """保存中心锚点检查图：绿=低残差，红=不一致，白=投影后的照片中心。"""
    photo = np.asarray(Image.open(photo_path).convert("RGB"))
    render = np.asarray(Image.open(render_path).convert("RGB"))
    warped = cv2.warpPerspective(photo, H, (render.shape[1], render.shape[0]))
    valid = cv2.warpPerspective(np.full(photo.shape[:2], 255, dtype=np.uint8), H,
                                (render.shape[1], render.shape[0])) > 0
    canvas = render.copy()
    canvas[valid] = ((render[valid].astype(np.uint16) + warped[valid].astype(np.uint16)) // 2).astype(np.uint8)
    photo_by_id = {row["id"]: row for row in photo_rows}
    render_by_id = {row["id"]: row for row in render_rows}
    errors = report.get("pair_errors_px", [])
    threshold = float(report.get("holdout_limit_px", 18.))
    for index, pair in enumerate(answer.get("pairs", [])):
        source, target = photo_by_id.get(pair.get("photo")), render_by_id.get(pair.get("render"))
        if source is None or target is None:
            continue
        projected = project_homography(np.asarray([source["center"]], dtype=float), H)[0]
        actual = np.asarray(target["center"], dtype=float)
        error = errors[index] if index < len(errors) else np.inf
        color = (55, 210, 55) if np.isfinite(error) and error <= threshold else (40, 40, 235)
        a, b = tuple(np.rint(projected).astype(int)), tuple(np.rint(actual).astype(int))
        cv2.line(canvas, a, b, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, b, 6, color, 2, cv2.LINE_AA)
        cv2.circle(canvas, a, 2, (255, 255, 255), -1, cv2.LINE_AA)
    text = "50% overlay; green=within limit, red=outside; white=projected photo center"
    cv2.putText(canvas, text, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, .45, (20, 20, 20), 3, cv2.LINE_AA)
    cv2.putText(canvas, text, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1, cv2.LINE_AA)
    Image.fromarray(canvas).save(output)


def main():
    photo_path, view, manifest, camera, objects, wall_id, detections = load_inputs(RUN_DIR, CANDIDATE_RANK)
    output = OUTPUT_ROOT / TRIAL_NAME / (Path(RUN_DIR).name + f"_candidate_{CANDIDATE_RANK:02d}")
    output.mkdir(parents=True, exist_ok=True)
    identity = {"photo_sha256": sha256(photo_path), "detection_sha256": sha256(read_json(Path(RUN_DIR) / "detection_source.json")["path"]),
                "render_sha256": sha256(view["image"]), "geometry_sha256": sha256(view["geometry"]),
                "manifest_sha256": sha256(view["manifest"]), "building_id": view["building_id"],
                "wall_id": wall_id}
    identity_path = output / "input_identity.json"
    if identity_path.exists() and read_json(identity_path) != identity:
        raise ValueError("此输出目录属于另一组输入；请换输出目录，不能混用旧编号。")
    write_json(identity_path, identity)
    photo_rows = photo_windows(detections)
    render_rows = render_windows(manifest, wall_id, objects)
    if len(photo_rows) < 8 or len(render_rows) < 8:
        raise ValueError("照片或渲染图不足8扇可用窗，本方法不适用此候选。")
    photo_labeled, render_labeled = output / "photo_numbered.png", output / "render_numbered.png"
    numbered_image(photo_path, photo_rows, photo_labeled, "#245d77")
    numbered_image(view["image"], render_rows, render_labeled, "#a36639")
    write_json(output / "window_ids.json", {"photo": photo_rows, "render": render_rows,
        "candidate": {"building_id": view["building_id"], "view_name": view["view_name"],
                      "wall_id": wall_id}, "photo_size": detections["image_size"],
        "render_size": camera["image_size_wh"]})
    (output / "prompt.txt").write_text(PROMPT + "\n", encoding="utf-8")
    pairs_path = output / "pairs.json"
    if MODE == "prepare":
        template = output / "pairs_template.json"
        if not template.exists():
            write_json(template, {"same_wall": "uncertain", "reason": "把模型回答另存为 pairs.json",
                                  "pairs": [{"photo": "P01", "render": "R01"}]})
        print(f"已准备编号图与提示词：{output}\n照片窗 {len(photo_rows)}，渲染窗 {len(render_rows)}。")
        return output
    if MODE == "api":
        answer = call_openai(photo_labeled, render_labeled, output)
    elif MODE == "apply":
        if not pairs_path.is_file():
            raise FileNotFoundError("请先把 ChatGPT 的 JSON 回答保存为本目录的 pairs.json")
        answer = read_json(pairs_path)
    else:
        raise ValueError("MODE 只能是 prepare、apply 或 api")
    H, report = fit_window_pairs(answer, photo_rows, render_rows,
                                  detections["image_size"], camera["image_size_wh"])
    report.update({"photo": str(photo_path), "candidate_building_id": view["building_id"],
                   "wall_id": wall_id, "answer_source": "openai_api" if MODE == "api" else "chatgpt_json",
                   "model_answer": answer,
                   "status": "preview_created" if report["accepted"] else "needs_review"})
    if H is not None:
        save_overlay(photo_path, view["image"], H, output / "alignment_overlay.png")
        save_center_overlay(photo_path, view["image"], H, answer, photo_rows, render_rows, report,
                            output / "window_center_overlay.png")
    if report["accepted"]:
        report["mapping"] = map_detections(detections, H, camera, view["geometry"],
                                            manifest, wall_id, output / "mapping")
    write_json(output / "alignment_report.json", report)
    print(f"结果：{report['status']}；报告：{output / 'alignment_report.json'}")
    return output


if __name__ == "__main__":
    main()
