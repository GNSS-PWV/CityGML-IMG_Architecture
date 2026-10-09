"""PyCharm 右键运行：生成 ±45° 实验的逐照片失败表。

默认只分析已经完成的批次，不调用模型或收费检测。把 REPORT_ONLY 改为 False，
才会重新建立 ±45° 图库、匹配 17 张照片，然后生成同一张诊断表。
照片文件名中的建筑编号只用于推理结束后的开发集核对，绝不传给检索或匹配。
"""

from collections import Counter
from pathlib import Path
import csv
import json
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
PHOTOS = ROOT / "Drills/Texture2LoD3_dataset/textures"
OUTPUT = Path(r"E:\workinCODEX\prog_3D\my_results\experiment_e45_user")
GALLERY = OUTPUT / "wall_view_gallery"
PIPELINE = OUTPUT / "wall_view_pipeline"

# 第一次看失败表请保持 True：直接读取最近一次完整的 ±45° 结果，不重跑模型。
# 若要完整重做实验，改为 False；缺少检测缓存时可能调用收费云端 API。
REPORT_ONLY = True
# 如需分析指定批次，填入它的 summary.json 绝对路径；None 表示最近完整批次。
REPORT_BATCH_SUMMARY = None

PHOTO_COLUMNS = {
    "photo": "照片", "expected_building": "对应建筑ID_仅事后核对",
    "building_rank": "DINO对应建筑排名", "top1": "正确建筑在Top1",
    "top3": "正确建筑在Top3", "top5": "正确建筑在Top5",
    "top2_gap": "DINO前两栋余弦分差", "entered_candidates": "正确建筑进入候选",
    "attempted_walls": "对应建筑已尝试墙ID_不代表正确墙",
    "correct_wall_truth": "正确墙面真值", "correct_candidate_count": "对应建筑候选墙次数",
    "correct_gate_count": "对应建筑几何布局通过次数",
    "diagnostic_candidate": "用于诊断的对应建筑候选", "diagnostic_view": "诊断候选视角",
    "diagnostic_wall": "诊断候选墙ID", "diagnostic_method": "诊断候选匹配方式",
    "roma_matches": "RoMa有效输入匹配数", "train_inliers": "训练内点数",
    "holdout_inliers": "留出内点数", "holdout_fraction": "留出内点比例",
    "holdout_p90_px": "全部留出点误差P90_渲染像素",
    "photo_hull_fraction": "训练内点照片凸包覆盖率",
    "photo_x_span": "训练内点照片横向跨度", "photo_y_span": "训练内点照片纵向跨度",
    "roma_geometry": "RoMa几何检查", "structure_geometry": "窗中心细化检查",
    "window_layout": "窗布局检查", "door_layout": "门布局检查",
    "wall_vote": "候选墙内点支持", "candidate_gate": "候选最终门槛",
    "failed_checks": "诊断候选未过检查项", "candidate_error": "候选错误",
    "selection_score": "全局候选排序分数", "minimum_score": "全局最低分门槛",
    "selection_margin": "全局候选分差", "minimum_margin": "全局最低分差门槛",
    "decision_reason": "全局决策原因", "failure_stage": "本张主要卡点",
    "status": "最终状态", "mapped_building": "最终映射建筑ID",
    "building_identity_correct": "映射建筑ID事后核对", "run_dir": "逐图结果目录",
}

CANDIDATE_COLUMNS = {
    "photo": "照片", "expected_building": "对应建筑ID_仅事后核对",
    "candidate_id": "候选编号", "building_id": "候选建筑ID",
    "view_name": "渲染视角", "wall_id": "候选墙ID",
    "retrieval_rank": "DINO视图排名", "source": "候选来源",
    "selected_method": "最终采用的匹配方式",
    "roma_matches": "RoMa有效输入匹配数", "train_inliers": "训练内点数",
    "holdout_inliers": "留出内点数", "holdout_fraction": "留出内点比例",
    "holdout_p90_px": "全部留出点误差P90_渲染像素",
    "photo_hull_fraction": "训练内点照片凸包覆盖率",
    "roma_geometry": "RoMa几何检查", "structure_geometry": "窗中心细化检查",
    "window_layout": "窗布局检查", "door_layout": "门布局检查",
    "wall_vote": "候选墙内点支持", "candidate_gate": "候选最终门槛",
    "failed_checks": "各模块未过检查项_不一定用于最终验收", "candidate_error": "候选错误",
}


def run(script, *arguments):
    """始终使用 PyCharm 当前解释器；子脚本出错时停止后续阶段。"""
    command = [sys.executable, str(ROOT / script), *map(str, arguments)]
    print("\n运行：", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def existing_summary():
    if REPORT_BATCH_SUMMARY is not None:
        path = Path(REPORT_BATCH_SUMMARY)
        if not path.is_file():
            raise FileNotFoundError(f"指定的批次汇总不存在：{path}")
        return path
    summaries = sorted(PIPELINE.glob("batch_*/summary.json"), reverse=True)
    for path in summaries:
        data = read_json(path)
        if len(data.get("records", [])) == data.get("photo_count"):
            return path
    raise FileNotFoundError(
        f"没有完整的 ±45° 批次：{PIPELINE}。如需新跑，把 REPORT_ONLY 改为 False。")


def check_state(report):
    """区分未执行、证据不足未触发、真正通过和失败。"""
    if not report:
        return "未执行"
    if report.get("required") is False:
        return "未触发"
    return "通过" if report.get("passed") else "失败"


def gate_state(report):
    if not report:
        return "未执行"
    return "通过" if report.get("gate_passed") else "失败"


def candidate_diagnosis(photo, expected, candidate):
    roma = candidate.get("roma_validation") or {}
    structure = candidate.get("structure_validation") or {}
    coverage = roma.get("train_photo_coverage") or {}
    errors = roma.get("holdout_all_errors") or {}
    view = candidate.get("view") or {}
    failures = []
    for label, report in (("RoMa", roma), ("窗中心", structure),
                          ("窗布局", candidate.get("layout_validation") or {}),
                          ("门布局", candidate.get("door_layout_validation") or {})):
        # 未触发的布局不能被当成“通过”，也不应产生失败项。
        if report.get("required") is False:
            continue
        failures.extend(f"{label}:{name}" for name in report.get("failed_checks", []))
    vote = candidate.get("wall_vote") or {}
    if vote and not vote.get("accepted"):
        failures.append("墙内点支持不足")
    return {
        "photo": photo, "expected_building": expected,
        "candidate_id": candidate.get("candidate_id"), "building_id": candidate.get("building_id"),
        "view_name": candidate.get("view_name"), "wall_id": candidate.get("wall_id"),
        "retrieval_rank": view.get("retrieval_rank"), "source": candidate.get("candidate_source"),
        "selected_method": candidate.get("selected_method"),
        "roma_matches": roma.get("valid_count"), "train_inliers": roma.get("train_inlier_count"),
        "holdout_inliers": roma.get("holdout_inlier_count"),
        "holdout_fraction": roma.get("holdout_inlier_fraction"),
        "holdout_p90_px": errors.get("p90_px"),
        "photo_hull_fraction": coverage.get("convex_hull_fraction"),
        "photo_x_span": coverage.get("x_span_fraction"),
        "photo_y_span": coverage.get("y_span_fraction"),
        "roma_geometry": gate_state(roma), "structure_geometry": gate_state(structure),
        "window_layout": check_state(candidate.get("layout_validation")),
        "door_layout": check_state(candidate.get("door_layout_validation")),
        "wall_vote": "通过" if vote.get("accepted") else ("失败" if vote else "未执行"),
        "candidate_gate": "通过" if candidate.get("gate_passed") else "失败",
        "failed_checks": "; ".join(failures), "candidate_error": candidate.get("error", ""),
    }


def failure_stage(status, expected, mapped, rank, entered, correct_candidates,
                  correct_passed, selection, reason):
    if status == "mapped":
        return "建筑ID核对一致；三维位置尚无独立真值" if mapped == expected else "错误自动接受_建筑ID不符"
    if status == "failed":
        return "运行异常"
    if rank is None:
        return "对应建筑未出现在检索排序或检索尚未完成"
    if not entered:
        return "正确建筑未进入Top5候选"
    if not correct_candidates:
        return "正确建筑已入选但没有可检验墙面候选"
    if not correct_passed:
        return "对应建筑候选几何或布局未通过"
    if reason == "ambiguous_candidates":
        score, minimum = selection.get("geometry_score"), selection.get("min_score")
        if score is not None and minimum is not None and score < minimum:
            return "候选排序分数不足"
        return "候选建筑或墙面区分不足"
    if reason in ("insufficient_cross_view_confirmation", "views_disagree_on_world_position"):
        return "双视角确认未通过"
    if reason == "invalid_mapping_geometry":
        return "三维映射几何未通过"
    return "自动验收其他环节未通过"


def diagnose_record(batch_row):
    """文件名只在此事后分析函数里转为已知建筑编号。"""
    photo = batch_row["photo"]
    expected = Path(photo).stem.split("_", 1)[0]
    run_dir = Path(batch_row["run_dir"]) if batch_row.get("run_dir") else None
    result = read_json(run_dir / "result.json") if run_dir and (run_dir / "result.json").is_file() else {}
    retrieval = read_json(run_dir / "retrieval.json") if run_dir and (run_dir / "retrieval.json").is_file() else {}
    candidates = read_json(run_dir / "candidates.json") if run_dir and (run_dir / "candidates.json").is_file() else []
    unique_buildings = []
    first_views = []
    for view in retrieval.get("ranked_views", []):
        building = str(view["building_id"])
        if building not in unique_buildings:
            unique_buildings.append(building)
            first_views.append(view)
    rank = unique_buildings.index(expected) + 1 if expected in unique_buildings else None
    entered = expected in {str(view["building_id"]) for view in retrieval.get("selected_candidates", [])}
    correct = [candidate for candidate in candidates if str(candidate.get("building_id")) == expected]
    passed = [candidate for candidate in correct if candidate.get("gate_passed")
              and (candidate.get("wall_vote") or {}).get("accepted")]
    def diagnostic_rank(item):
        score = item.get("geometry_score")
        return item in passed, float(score) if isinstance(score, (int, float)) else float("-inf")
    best = max(correct, key=diagnostic_rank) if correct else None
    details = [candidate_diagnosis(photo, expected, candidate) for candidate in candidates]
    diagnostic = candidate_diagnosis(photo, expected, best) if best else {}
    selection = result.get("selection") or {}
    status = result.get("status", batch_row.get("status", "未记录"))
    mapped = result.get("building_id") or batch_row.get("building_id")
    reason = selection.get("reason") or result.get("error_type") or batch_row.get("error_type", "")
    top2_gap = (round(first_views[0]["cosine_similarity"] - first_views[1]["cosine_similarity"], 6)
                if len(first_views) >= 2 else None)
    row = {
        "photo": photo, "expected_building": expected, "building_rank": rank,
        "top1": rank <= 1 if rank is not None else None,
        "top3": rank <= 3 if rank is not None else None,
        "top5": rank <= 5 if rank is not None else None,
        "top2_gap": top2_gap, "entered_candidates": entered if retrieval else None,
        "attempted_walls": "; ".join(sorted({str(item["wall_id"]) for item in correct})),
        "correct_wall_truth": "未提供；不能按候选墙ID推定",
        "correct_candidate_count": len(correct), "correct_gate_count": len(passed),
        "diagnostic_candidate": diagnostic.get("candidate_id"),
        "diagnostic_view": diagnostic.get("view_name"), "diagnostic_wall": diagnostic.get("wall_id"),
        "diagnostic_method": best.get("selected_method") if best else None,
        "roma_matches": diagnostic.get("roma_matches"), "train_inliers": diagnostic.get("train_inliers"),
        "holdout_inliers": diagnostic.get("holdout_inliers"),
        "holdout_fraction": diagnostic.get("holdout_fraction"),
        "holdout_p90_px": diagnostic.get("holdout_p90_px"),
        "photo_hull_fraction": diagnostic.get("photo_hull_fraction"),
        "photo_x_span": diagnostic.get("photo_x_span"),
        "photo_y_span": diagnostic.get("photo_y_span"),
        "roma_geometry": diagnostic.get("roma_geometry", "未执行"),
        "structure_geometry": diagnostic.get("structure_geometry", "未执行"),
        "window_layout": diagnostic.get("window_layout", "未执行"),
        "door_layout": diagnostic.get("door_layout", "未执行"),
        "wall_vote": diagnostic.get("wall_vote", "未执行"),
        "candidate_gate": diagnostic.get("candidate_gate", "未执行"),
        "failed_checks": diagnostic.get("failed_checks", ""),
        "candidate_error": diagnostic.get("candidate_error", ""),
        "selection_score": selection.get("geometry_score"),
        "minimum_score": selection.get("min_score"),
        "selection_margin": selection.get("margin"),
        "minimum_margin": selection.get("min_margin"),
        "decision_reason": reason,
        "failure_stage": failure_stage(status, expected, str(mapped) if mapped is not None else None,
                                       rank, entered, correct, passed, selection, reason),
        "status": status, "mapped_building": mapped,
        "building_identity_correct": (str(mapped) == expected) if status == "mapped" else "未映射",
        "run_dir": str(run_dir) if run_dir else "",
    }
    return row, details


def write_csv(path, rows, columns):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(columns.values()), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            values = {}
            for key, label in columns.items():
                value = row.get(key)
                values[label] = ("是" if value else "否") if isinstance(value, bool) else value
            writer.writerow(values)


def make_failure_report(batch_summary):
    """对已有结果做事后审计；不导入模型、不重跑检测/匹配。"""
    batch_summary = Path(batch_summary)
    batch = read_json(batch_summary)
    photo_rows, candidate_rows = [], []
    for source in batch.get("records", []):
        row, details = diagnose_record(source)
        photo_rows.append(row)
        candidate_rows.extend(details)
    report = {
        "scope": "仅限本批开发照片的事后诊断；文件名建筑ID不进入推理；墙面真值未提供；不是独立测试精度",
        "source_summary": str(batch_summary.resolve()),
        "planned_photo_count": batch.get("photo_count"), "recorded_photo_count": len(photo_rows),
        "stage_counts": dict(Counter(row["failure_stage"] for row in photo_rows)),
        "photo_records": photo_rows, "candidate_records": candidate_rows,
    }
    destination = batch_summary.parent
    write_csv(destination / "逐照片失败表.csv", photo_rows, PHOTO_COLUMNS)
    write_csv(destination / "逐候选失败明细.csv", candidate_rows, CANDIDATE_COLUMNS)
    (destination / "逐照片失败表.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(f"\n失败诊断已生成：{len(photo_rows)}/{batch.get('photo_count')} 张照片；"
          f"{len(candidate_rows)} 个候选墙面。", flush=True)
    print("  一张照片一行：", destination / "逐照片失败表.csv")
    print("  所有候选详细原因：", destination / "逐候选失败明细.csv")
    print("  可编程完整记录：", destination / "逐照片失败表.json")
    for stage, count in report["stage_counts"].items():
        print(f"  {stage}：{count} 张")
    print("注意：‘未触发’不等于通过；候选墙ID不等于有真值确认的正确墙面。")
    return report


def main():
    if REPORT_ONLY:
        batch_summary = existing_summary()
        print(f"只分析现有 ±45° 批次，不重跑模型：{batch_summary}", flush=True)
        make_failure_report(batch_summary)
        return
    if not PHOTOS.is_dir():
        raise FileNotFoundError(f"找不到立面照片目录：{PHOTOS}")
    photos = [p for p in PHOTOS.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    if not photos:
        raise ValueError(f"立面照片目录为空：{PHOTOS}")
    print(f"解释器：{sys.executable}\n照片：{len(photos)} 张\n输出：{OUTPUT}", flush=True)

    print("\n[1/2] 建立每面墙 -45°、0°、+45° 三视角图库", flush=True)
    run("build_wall_view_gallery.py", "--output-root", GALLERY,
        "--image-size", 900, 600, "--workers", 2)
    gallery_batch = Path(read_json(GALLERY / "latest_run.json")["run_dir"])
    gallery_summary = read_json(gallery_batch / "batch_summary.json")
    gallery_index = gallery_batch / "gallery_index.json"
    if (gallery_summary["status"] != "complete" or
            gallery_summary["wall_view_yaws_deg"] != [-45.0, 0.0, 45.0] or
            not gallery_index.is_file()):
        raise RuntimeError(f"图库未完整生成或视角参数错误，请检查：{gallery_batch}")
    print(f"图库完成：{gallery_summary['building_count']} 栋、{gallery_summary['view_count']} 张图", flush=True)

    print("\n[2/2] 使用新图库批量匹配照片；已有合格检测会复用缓存。"
          "若缓存缺失，原入口可能调用收费检测 API。", flush=True)
    before = set(PIPELINE.glob("batch_*/summary.json"))
    run("run_facade_pipeline.py", "--photo-dir", PHOTOS,
        "--top-buildings", 5, "--views-per-building", 2,
        "--gallery-index", gallery_index, "--output-root", PIPELINE)
    created = set(PIPELINE.glob("batch_*/summary.json")) - before
    if len(created) != 1:
        raise RuntimeError(f"无法唯一确定本次匹配汇总；请查看：{PIPELINE}")
    batch_summary = created.pop()
    result = read_json(batch_summary)
    counts = Counter(row["status"] for row in result["records"])
    print(f"\n完成：{counts.get('mapped', 0)} 张自动映射，"
          f"{counts.get('needs_review', 0)} 张待复核，{counts.get('failed', 0)} 张失败。")
    print(f"逐张结果和原因：{batch_summary}")
    make_failure_report(batch_summary)


if __name__ == "__main__":
    main()
