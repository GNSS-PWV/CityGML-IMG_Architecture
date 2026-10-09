"""PyCharm 直接运行：实验1 RoMa v2 vs 实验2 MINIMA-RoMa，共用0°图库和17张照片。

默认只读已有实验D门窗检测，不调用收费 API。两组保持同一 DINOv3 检索、
Top5建筑×每栋2张候选、几何/布局阈值与三维映射逻辑。仅匹配器不同。
0°只有每墙一个图，因此两组都明确停用“同墙第二视角”验收；此结果
不能与原来的双视角实验直接比较自动接受率，也不代表米制精度有真值。
照片文件名中的建筑ID仅在模型运行后的报告中作建筑级核对。
"""

from collections import Counter
from datetime import datetime
from pathlib import Path
import argparse
import csv
import json

import match_facade_roma as roma
import run_facade_pipeline as pipeline


ROOT = Path(__file__).resolve().parent
PHOTOS = ROOT / "Drills/Texture2LoD3_dataset/textures"
SOURCE_GALLERY_POINTER = ROOT.parent / "my_results/experiment_e45_user/wall_view_gallery/latest_run.json"
OUTPUT = ROOT.parent / "my_results/experiment_front0"
TOP_BUILDINGS = 5
VIEWS_PER_BUILDING = 2
RUN_EXPERIMENTS = (1, 2)  # PyCharm 可改成 (1,) 或 (2,)。
RESUME_RUN_DIR = None    # 中断后可填入本程序打印的 experiment_front0_... 目录。

FIELDS = ["experiment", "photo", "expected_building_posthoc", "status", "error_type",
          "correct_building_entered", "correct_building_geometry_passed",
          "correct_building_candidate_gate_passed", "any_geometry_passed",
          "any_candidate_gate_passed", "final_mapped", "wrong_building_mapped",
          "needs_review", "mapped_building", "mapped_openings", "decision_reason", "run_dir"]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    pipeline.write_json(path, value)


def gallery_index():
    """复用已经渲染完成的0°像素、相机与深度；两组引用完全相同文件。"""
    source = Path(read_json(SOURCE_GALLERY_POINTER)["run_dir"]) / "gallery_index.json"
    data = read_json(source)
    if data.get("status") != "complete":
        raise ValueError("源渲染图库尚未完成")
    views = [row for row in data["views"] if float(row.get("orbit_deg", 999)) == 0]
    if not views or len(views) * 3 != len(data["views"]):
        raise ValueError("源图库缺少完整的 -45°、0°、+45° 三视角配套数据")
    index = OUTPUT / "gallery_0_degree/gallery_index.json"
    index.parent.mkdir(parents=True, exist_ok=True)
    filtered = {**data, "experiment": "front0_matcher_comparison",
                "view_strategy": "one already rendered 0 degree view per eligible wall",
                "view_count": len(views), "views": views,
                "source_gallery": str(source), "source_gallery_sha256": roma.sha256(source)}
    if index.exists() and read_json(index) != filtered:
        raise ValueError("现有0°图库与源图库不一致；请指定新的实验输出目录核对后运行。")
    if not index.exists():
        write_json(index, filtered)
    return index, filtered


def image_list():
    photos = sorted(p for p in PHOTOS.iterdir() if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if len(photos) != 17:
        raise ValueError(f"预期17张立面照片，实际 {len(photos)} 张：{PHOTOS}")
    return photos


def preflight_detections(photos):
    """付费API硬停止；全部图片的D缓存可用才启动模型。"""
    import tempfile
    with tempfile.TemporaryDirectory(prefix="front0_detection_check_") as directory:
        for photo in photos:
            pipeline.detection_for_photo(photo, Path(directory), offline_only=True)


def diagnose(experiment, photo, run_dir):
    """事后使用文件名建筑编号评价建筑层面；墙面及三维坐标暂无独立真值。"""
    expected = photo.stem.split("_", 1)[0]
    result = read_json(run_dir / "result.json")
    retrieval = read_json(run_dir / "retrieval.json") if (run_dir / "retrieval.json").is_file() else {}
    candidates = read_json(run_dir / "candidates.json") if (run_dir / "candidates.json").is_file() else []
    selected = retrieval.get("selected_candidates", [])
    geometry = lambda c: bool((c.get("roma_validation") or {}).get("gate_passed") or
                              (c.get("structure_validation") or {}).get("gate_passed"))
    entered = any(str(c["building_id"]) == expected for c in selected)
    mapped = result["status"] == "mapped"
    mapped_building = str(result.get("building_id", "")) if mapped else ""
    return {
        "experiment": experiment, "photo": photo.name, "expected_building_posthoc": expected,
        "status": result["status"], "error_type": result.get("error_type", ""),
        "correct_building_entered": entered,
        "correct_building_geometry_passed": any(geometry(c) and str(c["building_id"]) == expected for c in candidates),
        "correct_building_candidate_gate_passed": any(c.get("gate_passed") and str(c["building_id"]) == expected for c in candidates),
        "any_geometry_passed": any(geometry(c) for c in candidates),
        "any_candidate_gate_passed": any(c.get("gate_passed") for c in candidates),
        "final_mapped": mapped, "wrong_building_mapped": mapped and mapped_building != expected,
        "needs_review": result["status"] == "needs_review",
        "mapped_building": mapped_building,
        "mapped_openings": (result.get("mapping") or {}).get("mapped_count", 0),
        "decision_reason": (result.get("selection") or {}).get("reason", ""),
        "run_dir": str(run_dir),
    }


def write_reports(batch, states):
    totals = {}
    for experiment in (1, 2):
        rows = states[str(experiment)]["rows"]
        path = batch / f"experiment_{experiment}_per_photo.csv"
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows([rows[name] for name in sorted(rows)])
        totals[str(experiment)] = {"completed_photos": len(rows),
            "correct_building_entered": sum(bool(r.get("correct_building_entered")) for r in rows.values()),
            "correct_building_geometry_passed": sum(bool(r.get("correct_building_geometry_passed")) for r in rows.values()),
            "correct_building_candidate_gate_passed": sum(bool(r.get("correct_building_candidate_gate_passed")) for r in rows.values()),
            "final_mapped": sum(bool(r.get("final_mapped")) for r in rows.values()),
            "wrong_building_mapped": sum(bool(r.get("wrong_building_mapped")) for r in rows.values()),
            "needs_review": sum(bool(r.get("needs_review")) for r in rows.values()),
            "statuses": dict(Counter(r["status"] for r in rows.values()))}
    write_json(batch / "summary.json", {"expected_photo_count": 17, "totals": totals,
                                        "fully_completed": all(t["completed_photos"] == 17 for t in totals.values())})
    lines = ["# 0°正视图匹配实验", "", "| 指标 | 实验1：RoMa v2 | 实验2：MINIMA-RoMa |", "|---|---:|---:|"]
    for label, key in (("已处理照片", "completed_photos"), ("正确建筑进入候选", "correct_building_entered"),
                       ("正确建筑几何通过", "correct_building_geometry_passed"),
                       ("正确建筑通过几何+布局候选门槛", "correct_building_candidate_gate_passed"),
                       ("最终自动映射", "final_mapped"), ("错配（建筑ID）", "wrong_building_mapped"),
                       ("待复核", "needs_review")):
        lines.append(f"| {label} | {totals['1'][key]}/17 | {totals['2'][key]}/17 |")
    lines.extend(["", "两组均使用同一批145张0°渲染图、17张照片、DINOv3检索和离线实验D门窗检测。",
                  "0°每墙只有一个视角，两组均停用同墙第二视角确认；自动映射数不能与双视角实验直接对比。",
                  "正确建筑与错配仅按照片文件名中的建筑ID事后核对；墙面、相机位姿和三维门窗位置无独立真值。",
                  "几何通过是几何模块门槛；候选通过还包含门窗布局与墙面支持。", ""])
    (batch / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    return totals


def main(experiments=RUN_EXPERIMENTS, resume=RESUME_RUN_DIR):
    gallery, gallery_data = gallery_index()
    photos = image_list()
    preflight_detections(photos)
    import match_facade_minima_roma as minima
    if 2 in experiments:
        for path, size in minima.EXPECTED_BYTES.items():
            if not path.is_file() or path.stat().st_size != size:
                raise FileNotFoundError(f"实验2模型权重未准备完整：{path}")
    if resume:
        batch = Path(resume).resolve()
        states = read_json(batch / "state.json")
        if states["gallery_sha256"] != roma.sha256(gallery):
            raise ValueError("续跑图库与原批次不同")
    else:
        batch = OUTPUT / datetime.now().strftime("experiment_front0_%Y%m%d_%H%M%S_%f")
        batch.mkdir(parents=True)
        states = {"gallery": str(gallery), "gallery_sha256": roma.sha256(gallery),
                  "gallery_view_count": gallery_data["view_count"],
                  "photo_directory": str(PHOTOS),
                  "1": {"matcher": "RoMa v2.0.1", "rows": {}},
                  "2": {"matcher": "MINIMA-RoMa", "rows": {}}}
        write_json(batch / "state.json", states)
    print(f"实验目录：{batch}\n0°图库：{gallery_data['view_count']}视图，{gallery_data['building_count']}栋建筑", flush=True)
    for experiment in experiments:
        matcher = roma if experiment == 1 else minima
        output_root = batch / f"experiment_{experiment}_runs"
        for i, photo in enumerate(photos, 1):
            if photo.name in states[str(experiment)]["rows"]:
                continue
            print(f"\n实验{experiment} 照片{i}/17：{photo.name}", flush=True)
            try:
                run_dir = pipeline.run_pipeline(photo, TOP_BUILDINGS, VIEWS_PER_BUILDING,
                    gallery_index=gallery, output_root=output_root, matcher_api=matcher,
                    single_view_experiment=True, offline_detection=True,
                    retrieval_cache_root=OUTPUT / "shared_retrieval_cache")
                row = diagnose(experiment, photo, run_dir)
            except Exception as exc:
                run_dir = getattr(exc, "pipeline_run_dir", None)
                row = diagnose(experiment, photo, Path(run_dir)) if run_dir else {
                    "experiment": experiment, "photo": photo.name,
                    "expected_building_posthoc": photo.stem.split("_", 1)[0],
                    "status": "failed", "error_type": type(exc).__name__,
                    "final_mapped": False, "wrong_building_mapped": False,
                    "needs_review": False, "run_dir": run_dir or ""}
                print(f"本张失败：{type(exc).__name__}: {exc}", flush=True)
            states[str(experiment)]["rows"][photo.name] = row
            write_json(batch / "state.json", states)
            totals = write_reports(batch, states)
            print("累计：", totals[str(experiment)], flush=True)
    return batch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("1", "2", "both"), default="both")
    parser.add_argument("--resume-run-dir", type=Path, default=RESUME_RUN_DIR)
    args = parser.parse_args()
    main((1, 2) if args.experiment == "both" else (int(args.experiment),), args.resume_run_dir)
