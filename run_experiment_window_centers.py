"""实验 1.1 / 2.1：用已有视觉匹配初值，按门窗中心重新对齐 17 张照片。

实验 1.1 复用实验 1 的 RoMa 初值；实验 2.1 复用实验 2 的 MINIMA-RoMa
初值。窗户配对由本地一对一距离分配产生，并非 GPT 视觉模型回答。至少
6 对时按配对数重新拟合仿射或单应矩阵；少于 6 对仅可复用原流程已自动
确认的稠密匹配变换。通过后尝试映射到三维墙面。
只读取原实验和 Grounding DINO 1.6 的缓存，不调用收费 API，不修改 CityGML。
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from facade_auto_mapping import map_detections
from facade_match_geometry import (MIN_HOLDOUT_INLIER_FRACTION,
                                   dense_gate_from_saved_report, project_homography)
from vlm_window_alignment import (fit_window_pairs, photo_windows, read_json,
                                  render_windows, save_center_overlay, save_overlay, sha256, write_json)


ROOT = Path(__file__).resolve().parent
SOURCE_BATCH = (ROOT.parent / "my_results/experiment_front0/"
                "experiment_front0_20261010_172758_423064")
OUTPUT_ROOT = ROOT.parent / "my_results/experiment_window_centers"
EXPERIMENTS = ("1.1", "2.1")
EXPECTED_PHOTOS = 17
# 只允许初值附近的窗配对。大于此距离的对应不能用“窗列规则”猜出来。
MAX_ASSOCIATION_DISTANCE_PX = 35.0
MIN_ACCEPT_MARGIN = 0.12
MIN_WINDOW_PAIRS = 6
POLICY_VERSION = "six_pairs_affine_dense_holdout_0.45_sparse_dense_fallback"
LAYOUT_PROFILES = {
    "baseline": {"hull": .30, "x_span": .65, "y_span": .60},
    "mild": {"hull": .25, "x_span": .58, "y_span": .55},
}


def policy_version(layout_profile: str) -> str:
    if layout_profile not in LAYOUT_PROFILES:
        raise ValueError(f"未知窗布局实验配置：{layout_profile}")
    return POLICY_VERSION if layout_profile == "baseline" else POLICY_VERSION + "_layout_mild"


def source_rows(source_batch: Path, experiment: str) -> list[dict]:
    base = experiment.split(".")[0]
    with (source_batch / f"experiment_{base}_per_photo.csv").open(
            encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != EXPECTED_PHOTOS or len({r["photo"] for r in rows}) != EXPECTED_PHOTOS:
        raise ValueError(f"实验 {base} 的逐照片记录不是完整的 17 张")
    return sorted(rows, key=lambda row: row["photo"])


def pair_from_initial(photo_rows: list[dict], render_rows: list[dict], initial_h,
                      max_distance=MAX_ASSOCIATION_DISTANCE_PX) -> dict:
    """已有模型匹配只提供初值；实际对应点固定为检测框/构件框中心。"""
    if not photo_rows or not render_rows:
        return {"same_wall": "uncertain", "reason": "no_visible_windows", "pairs": []}
    H = np.asarray(initial_h, dtype=float)
    if H.shape != (3, 3) or not np.isfinite(H).all() or np.linalg.matrix_rank(H) < 3:
        return {"same_wall": "uncertain", "reason": "invalid_initial_transform", "pairs": []}
    photo = np.asarray([row["center"] for row in photo_rows], dtype=float)
    render = np.asarray([row["center"] for row in render_rows], dtype=float)
    projected = project_homography(photo, H)
    if not np.isfinite(projected).all():
        return {"same_wall": "uncertain", "reason": "invalid_initial_projection", "pairs": []}
    distances = np.linalg.norm(projected[:, None, :] - render[None, :, :], axis=2)
    ix, iy = linear_sum_assignment(distances)
    pairs = [{"photo": photo_rows[i]["id"], "render": render_rows[j]["id"],
              "initial_distance_px": float(distances[i, j])}
             for i, j in zip(ix, iy) if distances[i, j] <= max_distance]
    return {"same_wall": "yes" if len(pairs) >= MIN_WINDOW_PAIRS else "uncertain",
            "reason": "local_assignment_from_previously_saved_matcher_transform",
            "pairs": pairs}


def candidate_score(report: dict, photo_count: int, render_count: int) -> float:
    """仅排序候选，不解释成概率；留出误差和覆盖不足会降低分数。"""
    if not report.get("accepted"):
        return 0.0
    pairs = report["pair_count"]
    fraction = pairs / max(1, min(photo_count, render_count))
    coverage = np.prod(report["photo_span_fraction"])
    error = float(np.median(report["holdout_errors_px"]))
    limit = float(report["holdout_limit_px"])
    return float(min(1., fraction) * np.sqrt(coverage) * max(0., 1. - error / (2 * limit)))


def choose_candidate(candidates: list[dict]) -> tuple[dict | None, dict]:
    """跨建筑、跨墙面比较；分数接近就保留待复核。"""
    # 中心锚点只精配准已经通过稠密几何筛选的候选。规则窗列可以形成漂亮的
    # 自洽 H，所以它不能替一个上游已经拒绝的建筑或墙面“翻案”。
    valid = sorted((row for row in candidates
                    if row.get("accepted") and row.get("source_gate_passed")),
                   key=lambda row: row["score"], reverse=True)
    if not valid:
        center_only = [row["candidate_id"] for row in candidates if row.get("accepted")]
        return None, {"reason": "no_source_approved_candidate_passed_window_checks",
                      "center_only_candidate_ids": center_only, "margin": None}
    best = valid[0]
    # 检索排序更靠前的另一栋楼若根本没有可用初值，就没有证据排除它。
    # 不能仅因为较后候选的规则窗列碰巧自洽，就自动写入错误建筑。
    unresolved_earlier = [row for row in candidates
        if row["source_order"] < best["source_order"] and
        row["building_id"] != best["building_id"] and
        "source_matcher_has_no_initial_transform" in row.get("error", "")]
    if unresolved_earlier:
        return None, {"reason": "higher_ranked_building_unresolved",
                      "unresolved_candidate_ids": [row["candidate_id"] for row in unresolved_earlier],
                      "best_candidate_id": best["candidate_id"], "margin": None}
    runner = next((row for row in valid[1:] if
                   (row["building_id"], row["wall_id"]) !=
                   (best["building_id"], best["wall_id"])), None)
    margin = best["score"] - runner["score"] if runner else best["score"]
    info = {"reason": "window_checks_and_margin_passed" if margin >= MIN_ACCEPT_MARGIN
            else "ambiguous_candidate_walls", "margin": float(margin),
            "required_margin": MIN_ACCEPT_MARGIN, "runner_up_candidate_id":
            runner["candidate_id"] if runner else None}
    return (best if margin >= MIN_ACCEPT_MARGIN else None), info


def source_gate_with_current_policy(candidate: dict, layout_profile="baseline") -> bool:
    """从旧报告重算稠密门槛；mild 仅放宽窗布局覆盖，不跳过墙面及门检查。"""
    if candidate.get("gate_passed"):
        return True
    if candidate.get("error") or candidate.get("selected_method", "").endswith("_window_structure"):
        return False
    if not (dense_gate_from_saved_report(candidate.get("roma_validation") or {}) and
            bool((candidate.get("door_layout_validation") or {}).get("passed"))):
        return False
    if layout_profile != "baseline" and not (candidate.get("wall_vote") or {}).get("accepted"):
        return False
    layout = candidate.get("layout_validation") or {}
    if layout_profile == "baseline" or layout.get("passed"):
        return bool(layout.get("passed"))
    thresholds = LAYOUT_PROFILES[layout_profile]
    structure = candidate.get("structure_validation") or {}
    coverage = structure.get("train_photo_coverage") or {}
    return bool((candidate.get("wall_vote") or {}).get("accepted") and
                layout.get("required") and
                structure.get("association_count", 0) >= 12 and
                all(isinstance(coverage.get(key), (int, float)) and
                    np.isfinite(coverage[key]) and coverage[key] >= minimum
                    for key, minimum in (("convex_hull_fraction", thresholds["hull"]),
                                         ("x_span_fraction", thresholds["x_span"]),
                                         ("y_span_fraction", thresholds["y_span"]))))


def choose_dense_fallback(source_result: dict, source_candidates: list[dict],
                          center_rows: list[dict], center_selection: dict) -> tuple[dict | None, dict]:
    """少于 6 对窗时仅复用原流程已自动确认的稠密匹配，不借窗列重新选楼。"""
    if center_selection["reason"] != "no_source_approved_candidate_passed_window_checks":
        return None, {"reason": "center_candidate_requires_review"}
    if source_result.get("status") != "mapped" or not source_result.get("selection", {}).get("accepted"):
        return None, {"reason": "source_dense_match_not_auto_approved"}
    candidate_id = (source_result.get("selected_candidate") or
                    source_result["selection"].get("best_candidate_id"))
    candidate = next((c for c in source_candidates if c["candidate_id"] == candidate_id), None)
    center = next((r for r in center_rows if r["candidate_id"] == candidate_id), None)
    if candidate is None or center is None or center.get("pair_count") is None:
        return None, {"reason": "source_candidate_has_no_window_pair_report"}
    pair_count = center["pair_count"]
    if pair_count >= MIN_WINDOW_PAIRS:
        return None, {"reason": "source_candidate_has_enough_window_pairs"}
    if (str(candidate["building_id"]) != str(source_result.get("building_id")) or
            candidate["wall_id"] != source_result.get("wall_id") or
            candidate["view_name"] != source_result.get("view_name") or
            candidate.get("selected_method", "").endswith("_window_structure") or
            not candidate.get("gate_passed") or
            not (candidate.get("wall_vote") or {}).get("accepted") or
            not (candidate.get("roma_validation") or {}).get("gate_passed") or
            not (candidate.get("layout_validation") or {}).get("passed") or
            not (candidate.get("door_layout_validation") or {}).get("passed") or
            float(candidate["view"].get("orbit_deg", 999)) != 0):
        return None, {"reason": "source_dense_evidence_not_sufficient"}
    H = np.asarray(candidate.get("roma_homography_photo_to_render"), dtype=float)
    if H.shape != (3, 3) or not np.isfinite(H).all() or np.linalg.matrix_rank(H) < 3:
        return None, {"reason": "source_dense_transform_invalid"}
    return candidate, {"reason": "source_dense_match_fallback_for_sparse_windows",
                       "source_candidate_id": candidate_id, "window_pair_count": pair_count,
                       "source_selection_reason": source_result["selection"].get("reason"),
                       "source_geometry_score": candidate.get("geometry_score"),
                       "independent_accuracy_verified": False}


def analyze_photo(source_row: dict, experiment: str, output: Path,
                  layout_profile="baseline") -> dict:
    run_dir = Path(source_row["run_dir"])
    source_result = read_json(run_dir / "result.json")
    photo_path = Path(source_result["photo"])
    detection_ref = read_json(run_dir / "detection_source.json")
    detections = read_json(detection_ref["path"])
    if sha256(photo_path) != detections["image_sha256"]:
        raise ValueError("照片与检测缓存哈希不一致")
    photo_rows = photo_windows(detections)
    source_candidates = read_json(run_dir / "candidates.json")
    output.mkdir(parents=True, exist_ok=True)
    input_identity = {"photo_sha256": sha256(photo_path),
                      "detections_sha256": sha256(detection_ref["path"]),
                      "source_candidates_sha256": sha256(run_dir / "candidates.json"),
                      "experiment": experiment, "policy_version": policy_version(layout_profile)}
    identity_path = output / "input_identity.json"
    if identity_path.exists() and read_json(identity_path) != input_identity:
        raise ValueError("输出目录属于不同输入，不能复用")
    write_json(identity_path, input_identity)

    rows = []
    for index, candidate in enumerate(source_candidates, 1):
        view = candidate["view"]
        row = {"candidate_id": candidate["candidate_id"],
               "source_order": index, "building_id": str(candidate["building_id"]),
               "wall_id": candidate["wall_id"], "view_name": candidate["view_name"],
               "accepted": False, "score": 0.0, "photo_windows": len(photo_rows),
               "source_gate_original": bool(candidate.get("gate_passed")),
               "source_gate_passed": source_gate_with_current_policy(candidate, layout_profile),
               "source_geometry_score": candidate.get("geometry_score"),
               "independent_accuracy_verified": False}
        try:
            if float(view.get("orbit_deg", 999)) != 0:
                raise ValueError("candidate_not_front_view")
            H0 = candidate.get("roma_homography_photo_to_render")
            if H0 is None:
                raise ValueError("source_matcher_has_no_initial_transform")
            manifest = read_json(view["manifest"])
            if sha256(manifest["source_gml"]) != manifest["source_sha256"]:
                raise ValueError("CityGML_has_changed_since_render")
            camera = read_json(view["camera"])
            with np.load(view["geometry"], allow_pickle=False) as geom:
                object_map = geom["object_index"]
            render_rows = render_windows(manifest, candidate["wall_id"], object_map)
            answer = pair_from_initial(photo_rows, render_rows, H0)
            H, report = fit_window_pairs(answer, photo_rows, render_rows,
                                         detections["image_size"], camera["image_size_wh"])
            row.update({"render_windows": len(render_rows), "pair_count": len(answer["pairs"]),
                        "pairs": answer["pairs"], "alignment_report": report,
                        "accepted": bool(report["accepted"]),
                        "score": candidate_score(report, len(photo_rows), len(render_rows))})
            if H is not None:
                row["matrix_photo_to_render"] = H.tolist()
        except (FileNotFoundError, KeyError, ValueError, TypeError) as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    write_json(output / "candidates.json", rows)

    selected, selection = choose_candidate(rows)
    fallback = None
    fallback_info = None
    if selected is None:
        fallback, fallback_info = choose_dense_fallback(source_result, source_candidates, rows, selection)
        if fallback is not None:
            selection = fallback_info
    result = {"experiment": experiment, "photo": photo_path.name,
              "source_run_dir": str(run_dir), "source_status": source_row["status"],
              "candidate_count": len(rows), "photo_window_count": len(photo_rows),
              "candidate_pass_count": sum(r["accepted"] for r in rows),
              "status": "needs_review", "selection": selection,
              "alignment_path": "none",
              "independent_accuracy_verified": False,
              "method": "local one-to-one window centers, affine for 6–7 pairs or homography for 8+ pairs; no GPT API",
              "policy_version": policy_version(layout_profile)}
    if fallback_info is not None and fallback is None:
        result["fallback_check"] = fallback_info
    if selected is not None or fallback is not None:
        match = (next(c for c in source_candidates if c["candidate_id"] == selected["candidate_id"])
                 if selected is not None else fallback)
        view = match["view"]
        H = np.asarray(selected["matrix_photo_to_render"] if selected is not None else
                       match["roma_homography_photo_to_render"], dtype=float)
        result.update({"candidate_id": match["candidate_id"],
                       "building_id": str(match["building_id"]), "wall_id": match["wall_id"],
                       "view_name": match["view_name"],
                       "score": selected["score"] if selected is not None else match["geometry_score"],
                       "alignment_path": "window_centers" if selected is not None else "dense_fallback"})
        if fallback is not None:
            result["method"] = "saved auto-approved dense matcher transform; no window-center refit; no GPT API"
        save_overlay(photo_path, view["image"], H, output / "alignment_overlay.png")
        if selected is not None:
            # 中心检查图必须用被选候选的 Window 构件，不能复用循环末尾另一栋楼的编号。
            selected_manifest = read_json(view["manifest"])
            with np.load(view["geometry"], allow_pickle=False) as selected_geometry:
                selected_render_rows = render_windows(selected_manifest, selected["wall_id"],
                                                      selected_geometry["object_index"])
            save_center_overlay(photo_path, view["image"], H, {"pairs": selected["pairs"]},
                                photo_rows, selected_render_rows, selected["alignment_report"],
                                output / "window_center_overlay.png")
        try:
            mapping = map_detections(detections, H, read_json(view["camera"]),
                                     view["geometry"], read_json(view["manifest"]),
                                     match["wall_id"], output / "mapping")
            result["mapping"] = mapping
            if mapping["mapped_count"]:
                result["status"] = "mapped"
            else:
                result["selection"]["reason"] = "no_detections_mapped"
        except (FileNotFoundError, KeyError, ValueError, TypeError) as exc:
            result["selection"]["reason"] = "three_d_mapping_rejected"
            result["mapping_error"] = f"{type(exc).__name__}: {exc}"
    write_json(output / "result.json", result)
    return result


def report_row(result: dict, source_row: dict) -> dict:
    expected = source_row["expected_building_posthoc"]
    mapped = result["status"] == "mapped"
    chosen = result.get("building_id", "")
    candidates = read_json(Path(result["output_dir"]) / "candidates.json")
    return {"experiment": result["experiment"], "photo": result["photo"],
            "expected_building_posthoc": expected,
            "source_status": source_row["status"], "status": result["status"],
            "correct_building_entered": source_row["correct_building_entered"],
            "correct_building_window_passed": any(c["accepted"] and c["building_id"] == expected
                                                  for c in candidates),
            "any_window_passed": any(c["accepted"] for c in candidates),
            "candidate_pass_count": result["candidate_pass_count"],
            "alignment_path": result["alignment_path"],
            "final_mapped": mapped, "wrong_building_mapped": mapped and chosen != expected,
            "needs_review": result["status"] == "needs_review", "mapped_building": chosen if mapped else "",
            "mapped_openings": (result.get("mapping") or {}).get("mapped_count", 0),
            "decision_reason": result["selection"]["reason"],
            "output_dir": result["output_dir"]}


def write_reports(batch: Path, states: dict) -> None:
    fields = ["experiment", "photo", "expected_building_posthoc", "source_status", "status",
              "correct_building_entered", "correct_building_window_passed", "any_window_passed",
              "candidate_pass_count", "alignment_path", "final_mapped", "wrong_building_mapped", "needs_review",
              "mapped_building", "mapped_openings", "decision_reason", "output_dir"]
    totals = {}
    for experiment in EXPERIMENTS:
        rows = list(states[experiment]["rows"].values())
        with (batch / f"experiment_{experiment}_per_photo.csv").open(
                "w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(sorted(rows, key=lambda row: row["photo"]))
        totals[experiment] = {"completed_photos": len(rows),
            **{key: sum(str(row.get(key, "")).lower() == "true" for row in rows)
               for key in ("correct_building_entered", "correct_building_window_passed",
                           "any_window_passed", "final_mapped", "wrong_building_mapped", "needs_review")},
            "correct_building_mapped": sum(str(row.get("final_mapped", "")).lower() == "true" and
                                           str(row.get("wrong_building_mapped", "")).lower() != "true"
                                           for row in rows),
            "dense_fallback_mapped": sum(str(row.get("final_mapped", "")).lower() == "true" and
                                         row.get("alignment_path") == "dense_fallback" for row in rows)}
    write_json(batch / "summary.json", {"method": "local_window_center_alignment_no_vlm_api",
               "policy_version": states["policy_version"],
               "layout_profile": states.get("layout_profile", "baseline"),
               "minimum_window_pairs": MIN_WINDOW_PAIRS,
               "minimum_dense_holdout_inlier_fraction": MIN_HOLDOUT_INLIER_FRACTION,
               "sparse_window_policy": "reuse only a previously auto-approved dense-only source match",
               "expected_photo_count": EXPECTED_PHOTOS, "totals": totals,
               "fully_completed": all(v["completed_photos"] == EXPECTED_PHOTOS for v in totals.values()),
               "accuracy_scope": "building ID inferred from filename only after run; no independent wall/3D GT"})
    source = Path(states["source_batch"])
    old = read_json(source / "summary.json")["totals"]
    labels = (("correct_building_entered", "正确建筑进入候选"),
              ("correct_building_mapped", "正确建筑自动映射"),
              ("wrong_building_mapped", "错配建筑自动映射"),
              ("needs_review", "待复核"))
    lines = ["# 实验 1.1 / 2.1：本地窗中心重新对齐", "",
             "本次复用实验 1/2 的 RoMa / MINIMA-RoMa 初值、Grounding DINO 1.6 检测和 0° 渲染图库。",
             "根据初值对照片窗与模型窗做一对一分配：6–7 对拟合仿射，8 对以上拟合单应，",
             "RoMa 留出点内点比例门槛为 0.45；旧缓存按新门槛重算。",
             ("窗布局覆盖使用原门槛。" if states.get("layout_profile", "baseline") == "baseline" else
              "温和放宽窗布局覆盖：面积 0.30→0.25，横向 0.65→0.58，纵向 0.60→0.55；其余几何及墙面检查不变。"),
             "若配对少于 6 对，仅当原稠密匹配已自动选定同一建筑墙面时回退；"
             "不会凭 4 对窗重新选楼。",
             "留出部分窗作几何检查。**未调用 GPT 或任何视觉语言模型 API。**", "",
             "注意：实验 1/2 与本批 1.1/2.1 同时存在窗中心方法和验收门槛差异，",
             "不能把总指标变化单独归因于窗中心方法。", "",
             "| 指标（17 张） | 实验 1 | 实验 1.1 | 实验 2 | 实验 2.1 |",
             "|---|---:|---:|---:|---:|"]
    for key, label in labels:
        lines.append(f"| {label} | {old['1'][key]} | {totals['1.1'][key]} | "
                     f"{old['2'][key]} | {totals['2.1'][key]} |")
    lines.extend(["", f"其中稠密匹配回退通过：实验 1.1 为 {totals['1.1']['dense_fallback_mapped']} 张，"
                  f"实验 2.1 为 {totals['2.1']['dense_fallback_mapped']} 张。"])
    lines.extend(["", "结果按每张照片保存；`experiment_1.1_per_photo.csv` 与 "
                  "`experiment_2.1_per_photo.csv` 列出逐照片判定。",
                  "`experiment_1.1/` 和 `experiment_2.1/` 下的照片文件夹保存候选报告、"
                  "成功对齐的叠加图和三维预测面预览。", "",
                  "## 与原实验逐照片相比", ""])
    for base, experiment in (("1", "1.1"), ("2", "2.1")):
        old_rows = {row["photo"]: row for row in source_rows(source, experiment)}
        new_rows = states[experiment]["rows"]
        gained = sorted(name for name, row in new_rows.items()
                        if row.get("status") == "mapped" and
                        row.get("mapped_building") == row.get("expected_building_posthoc") and
                        old_rows[name]["status"] != "mapped")
        lost = sorted(name for name, row in new_rows.items()
                      if row.get("status") != "mapped" and old_rows[name]["status"] == "mapped")
        lines.extend([f"- 实验 {base} → {experiment}：新增正确建筑映射 " +
                      ("、".join(gained) if gained else "无") + "；原已映射却转为待复核 " +
                      ("、".join(lost) if lost else "无") + "。"])
    lines.extend(["",
                  "**解释边界：**建筑 ID 仅按照片文件名事后核对；墙面和三维位置没有独立真值。"
                  "窗列重复会产生几何自洽的错配，`mapped` 不等于真实匹配成功。"
                  "三维输出仅为预览，不修改原 CityGML。", ""])
    (batch / "comparison.md").write_text("\n".join(lines), encoding="utf-8")


def run(source_batch=SOURCE_BATCH, output_root=OUTPUT_ROOT, experiments=EXPERIMENTS,
        resume: Path | None = None, layout_profile="baseline") -> Path:
    version = policy_version(layout_profile)
    source_batch = Path(source_batch).resolve()
    batch = Path(resume).resolve() if resume else Path(output_root) / datetime.now().strftime(
        "window_centers_%Y%m%d_%H%M%S_%f")
    batch.mkdir(parents=True, exist_ok=True)
    state_path = batch / "state.json"
    if state_path.exists():
        states = read_json(state_path)
        if states["source_batch"] != str(source_batch):
            raise ValueError("续跑目录属于其他来源实验")
        if states.get("policy_version") != version or states.get("layout_profile", "baseline") != layout_profile:
            raise ValueError("续跑目录属于旧版策略，请新建批次")
    else:
        states = {"source_batch": str(source_batch), "policy_version": version,
                  "layout_profile": layout_profile,
                  "1.1": {"source_matcher": "RoMa v2", "rows": {}},
                  "2.1": {"source_matcher": "MINIMA-RoMa", "rows": {}}}
        write_json(state_path, states)
    print(f"实验 1.1/2.1 输出目录：{batch}", flush=True)
    for experiment in experiments:
        for index, source_row in enumerate(source_rows(source_batch, experiment), 1):
            photo = source_row["photo"]
            if photo in states[experiment]["rows"]:
                continue
            print(f"实验 {experiment}，照片 {index}/{EXPECTED_PHOTOS}：{photo}", flush=True)
            output = batch / f"experiment_{experiment}" / Path(photo).stem
            try:
                result = analyze_photo(source_row, experiment, output, layout_profile)
                result["output_dir"] = str(output)
                write_json(output / "result.json", result)
                row = report_row(result, source_row)
            except Exception as exc:
                row = {"experiment": experiment, "photo": photo,
                       "expected_building_posthoc": source_row["expected_building_posthoc"],
                       "source_status": source_row["status"], "status": "failed",
                       "decision_reason": f"{type(exc).__name__}: {exc}", "output_dir": str(output)}
                print(f"  失败：{row['decision_reason']}", flush=True)
            states[experiment]["rows"][photo] = row
            write_json(state_path, states)
            write_reports(batch, states)
            print(f"  {row['status']}；{row['decision_reason']}", flush=True)
    return batch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("1.1", "2.1", "both"), default="both")
    parser.add_argument("--layout-profile", choices=tuple(LAYOUT_PROFILES), default="baseline")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    run(experiments=EXPERIMENTS if args.experiment == "both" else (args.experiment,),
        resume=args.resume, layout_profile=args.layout_profile)
