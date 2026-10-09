"""PyCharm 右键运行：建立 ±45° 墙面图库，再用它批量匹配 17 张立面照片。

本脚本只负责把已有两个入口顺序接起来。它不修改旧的 ±15°、±30° 图库，
也不把照片文件名当作建筑答案。运行前在 PyCharm 选好 torch_1 解释器。
"""

from collections import Counter
from pathlib import Path
import json
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
PHOTOS = ROOT / "Drills/Texture2LoD3_dataset/textures"
OUTPUT = Path(r"E:\workinCODEX\prog_3D\my_results\experiment_e45_user")
GALLERY = OUTPUT / "wall_view_gallery"
PIPELINE = OUTPUT / "wall_view_pipeline"


def run(script, *arguments):
    """始终使用 PyCharm 当前解释器；子脚本出错时停止后续阶段。"""
    command = [sys.executable, str(ROOT / script), *map(str, arguments)]
    print("\n运行：", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
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


if __name__ == "__main__":
    main()
