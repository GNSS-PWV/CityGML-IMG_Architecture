"""审计实验 E 与 8 视角基线的自动匹配结果。

照片文件名只在本脚本的事后开发集统计中转换为已知建筑编号；推理、检索、RoMa
匹配和三维映射均不读取该编号。输出的数量是流程通过数量，不是独立精度声明。
"""
from pathlib import Path
import argparse
import csv
import json
from collections import Counter, defaultdict


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def expected_building(photo_name):
    """数据集命名约定，仅限实验完成后的已知真值审计。"""
    return Path(photo_name).stem.split("_", 1)[0]


def ranked_building_rank(retrieval, building_id):
    seen = []
    for row in retrieval.get("ranked_views", []):
        item = str(row["building_id"])
        if item not in seen:
            seen.append(item)
    return seen.index(building_id) + 1 if building_id in seen else None


def one_record(row):
    run_dir = Path(row["run_dir"])
    result = read_json(run_dir / "result.json")
    retrieval = read_json(run_dir / "retrieval.json") if (run_dir / "retrieval.json").is_file() else {}
    candidates = read_json(run_dir / "candidates.json") if (run_dir / "candidates.json").is_file() else []
    expected = expected_building(row["photo"])
    candidate_ids = {str(item["building_id"]) for item in retrieval.get("selected_candidates", [])}
    passed = [item for item in candidates if item.get("gate_passed") and item.get("wall_vote", {}).get("accepted")]
    correct_passed = [item for item in passed if str(item["building_id"]) == expected]
    by_wall = defaultdict(set)
    for item in correct_passed:
        by_wall[item["wall_id"]].add(item["view_name"])
    double_supported = any(len(names) >= 2 for names in by_wall.values())
    status = result.get("status")
    mapped_id = str(result.get("building_id")) if result.get("building_id") is not None else None
    return {"photo": row["photo"], "expected_building_for_evaluation_only": expected, "status": status,
            "retrieval_building_rank": ranked_building_rank(retrieval, expected),
            "correct_building_entered_candidates": expected in candidate_ids,
            "correct_building_geometry_passed": bool(correct_passed),
            "correct_building_two_view_supported": double_supported,
            "final_mapping_correct": status == "mapped" and mapped_id == expected,
            "final_mapping_wrong": status == "mapped" and mapped_id != expected,
            "needs_review": status == "needs_review", "failed": status == "failed",
            "mapped_building_id": mapped_id, "candidate_count": len(candidates), "run_dir": str(run_dir)}


def audit(batch_summary):
    summary = read_json(batch_summary)
    records = [one_record(row) for row in summary["records"] if row.get("run_dir") and Path(row["run_dir"]).is_dir()]
    metrics = {
        "input_photo_count": len(records),
        "correct_building_entered_candidates": sum(row["correct_building_entered_candidates"] for row in records),
        "correct_building_geometry_passed": sum(row["correct_building_geometry_passed"] for row in records),
        "correct_building_two_view_supported": sum(row["correct_building_two_view_supported"] for row in records),
        "final_automatic_mapping_correct": sum(row["final_mapping_correct"] for row in records),
        "wrong_automatic_mapping": sum(row["final_mapping_wrong"] for row in records),
        "needs_review": sum(row["needs_review"] for row in records),
        "failed": sum(row["failed"] for row in records),
        "status_counts": dict(Counter(row["status"] for row in records)),
    }
    return {"batch_summary": str(Path(batch_summary).resolve()), "records": records, "metrics": metrics,
            "scope": "Post-hoc development-set audit; filename-derived expected IDs are not inference inputs and results are not independent accuracy estimates."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True, help="8视角基线的 summary.json")
    parser.add_argument("--wall-views", type=Path, required=True, help="实验E墙面视角的 summary.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline, wall_views = audit(args.baseline), audit(args.wall_views)
    output = args.output
    result = {"experiment": "E", "comparison": {"baseline_global_8_views": baseline, "wall_front_3_views": wall_views},
              "definitions": {"correct_building_entered_candidates": "known correct building appears in DINO Top-5 selected building candidates",
                              "geometry_passed": "at least one candidate of known correct building passes the existing geometric gate",
                              "two_view_supported": "two distinct rendered views of one known correct wall pass the existing gate",
                              "automatic_mapping": "pipeline final status is mapped; wrong mapping is counted separately",
                              "needs_review": "pipeline abstains rather than forcing a building/wall"}}
    write_json(output / "experiment_e_audit.json", result)
    labels = [("正确建筑进入候选", "correct_building_entered_candidates"), ("正确建筑几何通过", "correct_building_geometry_passed"),
              ("正确建筑双视角支持", "correct_building_two_view_supported"), ("最终正确自动映射", "final_automatic_mapping_correct"),
              ("错配自动映射", "wrong_automatic_mapping"), ("待复核", "needs_review"), ("失败", "failed")]
    with (output / "experiment_e_metrics.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream); writer.writerow(["指标", "8个全局视角", "每面墙3个正视/近正视视角"])
        for label, key in labels:
            writer.writerow([label, baseline["metrics"][key], wall_views["metrics"][key]])
    print(json.dumps({"baseline": baseline["metrics"], "wall_views": wall_views["metrics"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
