"""将实验 E 审计结果制作成适合汇报的静态结果图。"""
from pathlib import Path
import argparse
import json

import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image


INK, MUTED, PAPER, LINE = "#34434A", "#718087", "#F5F1EA", "#D7D3CA"
BASE, WALL, AMBER = "#B7C6C7", "#6F91A8", "#C7A36B"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def find_view(index, building_id, preferred=None):
    views = [item for item in index["views"] if str(item["building_id"]) == str(building_id)]
    if preferred:
        choice = next((item for item in views if item["view_name"] == preferred), None)
        if choice:
            return choice
    return views[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--baseline-gallery", type=Path, required=True)
    parser.add_argument("--wall-gallery", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = read_json(args.audit)["comparison"]
    baseline = audit["baseline_global_8_views"]["metrics"]
    wall = audit["wall_front_3_views"]["metrics"]
    baseline_view = find_view(read_json(args.baseline_gallery), "4959323", "view_000")
    wall_view = find_view(read_json(args.wall_gallery), "4959323")
    font = font_manager.FontProperties(fname="C:/Windows/Fonts/msyh.ttc")
    plt.rcParams["axes.unicode_minus"] = False
    figure = plt.figure(figsize=(16, 10), facecolor=PAPER)
    grid = figure.add_gridspec(16, 18, left=.055, right=.96, top=.92, bottom=.07, hspace=.95, wspace=.85)
    figure.text(.055, .955, "实验 E｜墙面正视/近正视图库对自动匹配的影响", fontsize=23, color=INK, fontproperties=font, weight="bold")
    figure.text(.055, .922, "同一批 17 张开发照片 · Top-5 建筑 × 每栋 2 个检索视图 · 既有 DINO / RoMa / 几何规则不变",
                fontsize=11.5, color=MUTED, fontproperties=font)
    labels = ["正确建筑\n进入候选", "正确建筑\n几何通过", "正确建筑\n双视角支持", "最终正确\n自动映射", "待复核", "错配\n自动映射"]
    keys = ["correct_building_entered_candidates", "correct_building_geometry_passed", "correct_building_two_view_supported",
            "final_automatic_mapping_correct", "needs_review", "wrong_automatic_mapping"]
    axis = figure.add_subplot(grid[2:10, :10], facecolor=PAPER)
    y = list(range(len(keys))); height = .34
    axis.barh([i + height/2 for i in y], [baseline[key] for key in keys], height, color=BASE, label="8 个全局视角")
    axis.barh([i - height/2 for i in y], [wall[key] for key in keys], height, color=WALL, label="每面墙 3 个视角")
    axis.set_yticks(y, labels, fontproperties=font, fontsize=11); axis.invert_yaxis(); axis.set_xlim(0, max(17, *(baseline[k] for k in keys), *(wall[k] for k in keys)) + 1)
    axis.xaxis.grid(True, color=LINE, linewidth=.8); axis.set_axisbelow(True)
    for spine in axis.spines.values(): spine.set_visible(False)
    axis.tick_params(axis="x", colors=MUTED); axis.tick_params(axis="y", length=0, colors=INK)
    for i, key in enumerate(keys):
        axis.text(baseline[key] + .16, i + height/2, str(baseline[key]), va="center", fontsize=10, color=INK)
        axis.text(wall[key] + .16, i - height/2, str(wall[key]), va="center", fontsize=10, color=INK)
    legend = axis.legend(frameon=False, loc="lower right", prop=font); [text.set_color(INK) for text in legend.get_texts()]
    axis.set_title("17 张开发照片的流程通过数量", loc="left", fontsize=15, color=INK, fontproperties=font, pad=14)
    for column, (name, values, color) in enumerate((("8 个全局视角", baseline, BASE), ("墙面 3 视角", wall, WALL))):
        card = figure.add_subplot(grid[11:15, column*5:(column+1)*5-1]); card.set_facecolor("#FAF8F4"); card.set_xticks([]); card.set_yticks([])
        for spine in card.spines.values(): spine.set_edgecolor(LINE); spine.set_linewidth(1.1)
        card.text(.08, .76, name, color=MUTED, fontsize=11, fontproperties=font, transform=card.transAxes)
        card.text(.08, .36, f"{values['final_automatic_mapping_correct']} / {values['input_photo_count']}", color=color, fontsize=25, weight="bold", transform=card.transAxes)
        card.text(.08, .16, "正确自动映射", color=INK, fontsize=10, fontproperties=font, transform=card.transAxes)
    for col, (title, item) in enumerate((("基线：完整建筑全局视图", baseline_view), ("实验 E：目标墙正视/近正视", wall_view))):
        axis_image = figure.add_subplot(grid[2:9, 10+col*4:14+col*4])
        axis_image.imshow(Image.open(item["image"]).convert("RGB")); axis_image.set_xticks([]); axis_image.set_yticks([])
        axis_image.set_title(title, fontsize=11, color=INK, fontproperties=font, pad=10)
        for spine in axis_image.spines.values(): spine.set_edgecolor(LINE); spine.set_linewidth(1.3)
    note = ("读图：蓝色柱为实验 E，灰绿色柱为原图库。‘待复核’代表系统拒绝强行映射；错配仅统计状态为 mapped 且与事后数据集对应关系不符的记录。\n"
            "实验 E 为开发集对照，照片文件名只在统计结束后用于已知建筑核对；不进入检索、匹配或映射推理。")
    figure.text(.055, .025, note, fontsize=9.5, color=MUTED, fontproperties=font, linespacing=1.65)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=220, facecolor=PAPER)


if __name__ == "__main__":
    main()
