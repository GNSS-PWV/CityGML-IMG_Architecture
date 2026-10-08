"""用已保存的审计 JSON 绘制实验 E 的 30° + 门窗布局检查结果图。"""
from pathlib import Path
import argparse
import json

from matplotlib import font_manager
from matplotlib import pyplot as plt


PAPER = "#F5F3EF"
INK = "#33434D"
MUTED = "#687A82"
GRID = "#D8DCD9"
GLOBAL = "#BFCBCB"
E15 = "#95ABB5"
E30 = "#668BA3"


def metrics(path, key):
    audit = json.loads(Path(path).read_text(encoding="utf-8"))
    return audit["comparison"][key]["metrics"]


def draw(old_audit, new_audit, output):
    baseline = metrics(old_audit, "baseline_global_8_views")
    e15 = metrics(old_audit, "wall_front_3_views")
    e30 = metrics(new_audit, "wall_front_3_views")
    groups = [("8 个全局视角", baseline, GLOBAL), ("墙面 ±15°", e15, E15),
              ("墙面 ±30° + 门窗布局检查", e30, E30)]
    labels = ["正确建筑进入候选", "正确建筑几何通过", "正确建筑双视角支持",
              "最终正确自动映射", "错误自动映射", "待复核"]
    keys = ["correct_building_entered_candidates", "correct_building_geometry_passed",
            "correct_building_two_view_supported", "final_automatic_mapping_correct",
            "wrong_automatic_mapping", "needs_review"]
    font = font_manager.FontProperties(fname="C:/Windows/Fonts/msyh.ttc")
    plt.rcParams["axes.unicode_minus"] = False
    fig = plt.figure(figsize=(15, 9), facecolor=PAPER)
    grid = fig.add_gridspec(12, 15, left=.19, right=.96, top=.88, bottom=.07,
                           hspace=.5, wspace=.5)
    fig.text(.075, .94, "实验 E｜扩大视角间隔并检查门窗布局", fontsize=23,
             color=INK, fontproperties=font)
    fig.text(.075, .902, "17 张相同立面照片  ·  27 栋建筑  ·  每面入选墙 3 张渲染图",
             fontsize=11, color=MUTED, fontproperties=font)

    ax = fig.add_subplot(grid[:9, :11], facecolor=PAPER)
    offsets = (-.25, 0, .25)
    for (name, values, color), offset in zip(groups, offsets):
        ys = [i + offset for i in range(len(keys))]
        widths = [values[key] for key in keys]
        ax.barh(ys, widths, height=.22, color=color, label=name)
        for y, width in zip(ys, widths):
            ax.text(width + .14, y, str(width), va="center", fontsize=9, color=INK)
    ax.set_yticks(range(len(labels)), labels, fontproperties=font, fontsize=11, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 18)
    ax.grid(axis="x", color=GRID, linewidth=.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", colors=MUTED)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -.055), ncol=3,
              frameon=False, prop=font)

    card = fig.add_subplot(grid[1:8, 11:])
    card.set_facecolor("#FAFAF8")
    card.set_xticks([])
    card.set_yticks([])
    for spine in card.spines.values():
        spine.set_edgecolor(GRID)
    card.text(.10, .87, "关键样本", color=MUTED, fontsize=11,
              fontproperties=font, transform=card.transAxes)
    card.text(.10, .75, "4907518_left", color=INK, fontsize=18,
              transform=card.transAxes)
    card.text(.10, .59, "±15°：误配至 4907520", color="#A4776D", fontsize=11,
              fontproperties=font, transform=card.transAxes)
    card.text(.10, .47, "±30° + 布局：待复核", color=E30, fontsize=11,
              fontproperties=font, transform=card.transAxes)
    card.text(.10, .30, "正确自动映射保留 4 张", color=INK, fontsize=11,
              fontproperties=font, transform=card.transAxes)
    card.text(.10, .20, "错配由 1 张降为 0 张", color=INK, fontsize=11,
              fontproperties=font, transform=card.transAxes)

    fig.text(.075, .115, "图中统计为开发集事后核对。新配置同时改变视角和布局门槛，不能单独归因于其中一项。",
             color=MUTED, fontsize=10, fontproperties=font)
    fig.text(.075, .082, "窗布局检查需双方各至少 12 个窗；门布局检查需双方各至少 3 个门。本批目标墙最多 2 个可见门，门检查未触发。",
             color=MUTED, fontsize=10, fontproperties=font)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, facecolor=PAPER)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-audit", type=Path, required=True)
    parser.add_argument("--new-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    draw(arguments.old_audit, arguments.new_audit, arguments.output)
