"""立面检测的公共几何与建筑文件关联函数，不调用任何检测模型。

这些函数从旧的 Tiny 入口中拆出，供 try_grounding_dino16.py 的实验 D 使用。
它们是本项目的适配实现，不是 Grounding DINO 网络或原参考作者的模型代码。
迁移仅改变文件位置，保留已有坐标还原、重复框分组和类别冲突规则。
本文件只依赖 Python 标准库，无需在 PyCharm 中单独运行。
"""

from pathlib import Path
import re


DATASET_DIR = Path(__file__).resolve().parent / "Drills/Texture2LoD3_dataset"

# IoU 超过此值，或满足下方严格的包含关系，才合并为同一个开口。
NMS_IOU = 0.45
NESTED_IOS = 0.90  # 小框至少 90% 被大框覆盖。
NESTED_MIN_AREA_RATIO = 0.25  # 小框不能小到只是大门里的一小块玻璃。
NESTED_CENTER_FRACTION = 0.30  # 两框的中心还必须足够接近。


def overlap(a, b):
    """两个 XYXY 框的 IoU＝交集面积 / 并集面积；不是模型置信度。"""
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return intersection / max(area_a + area_b - intersection, 1e-12)


def same_opening(a, b):
    """判断两个框是否指向同一开口；不判断它究竟是门还是窗。

    除 IoU 外，再检查交集/小框面积、两框面积比和中心距离，处理窗框包窗框。
    不能只用“包含”判断：大门内的小玻璃仍可能是独立目标。
    """
    if overlap(a, b) > NMS_IOU:
        return True
    aw, ah = a[2] - a[0], a[3] - a[1]
    bw, bh = b[2] - b[0], b[3] - b[1]
    small_area, large_area = sorted((aw * ah, bw * bh))
    if small_area <= 0:
        return False
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1]))
    return (intersection / small_area >= NESTED_IOS
            and small_area / large_area >= NESTED_MIN_AREA_RATIO
            and abs((a[0] + a[2] - b[0] - b[2]) / 2) <= NESTED_CENTER_FRACTION * max(aw, bw)
            and abs((a[1] + a[3] - b[1] - b[3]) / 2) <= NESTED_CENTER_FRACTION * max(ah, bh))


def remove_duplicates(records):
    """按位置分组，每组保留最高分候选的框，门窗冲突标为 ambiguous。

    分数不是已校准的门窗分类概率，不能仅凭门的分数更高就判成门。
    candidates 保留全部组内候选，供人工核对；不使用 GT 自动修改类别。
    """
    groups = []
    for record in sorted(records, key=lambda r: r["score"], reverse=True):
        # 必须与组内每个框都接近，避免通过相邻框连锁合并一整排窗。
        for group in groups:
            if all(same_opening(record["box"], old["box"]) for old in group):
                group.append(record)
                break
        else:
            groups.append([record])
    detections = []
    for group in groups:
        result = dict(group[0])
        result["candidates"] = group
        classes = {r["class"] for r in group}
        if len(classes) > 1:
            result["class"] = "ambiguous"
            result["review_required"] = True
            result["class_scores"] = {kind: max(r["score"] for r in group if r["class"] == kind)
                                      for kind in sorted(classes)}
        detections.append(result)
    # 几何规则仍有局限：大框覆盖多个真实开口时不保证正确，结果需要核查。
    return detections


def box_to_full_image(box, view_box, image_size, edge_margin):
    """图块坐标加偏移后回到原照片；无效框和内部切边残框返回 None。

    例如图块左上角 x=900，框内 x=100，则原图 x=1000。照片的真实四周边缘
    仍保留；只过滤靠近内部切边的框，让重叠图块补充完整预测。
    """
    left, top, right, bottom = view_box
    view_width, view_height = right - left, bottom - top
    x0, y0, x1, y1 = box
    x0, x1 = max(0, min(view_width, x0)), max(0, min(view_width, x1))
    y0, y1 = max(0, min(view_height, y0)), max(0, min(view_height, y1))
    if x1 <= x0 or y1 <= y0:
        return None
    if ((left > 0 and x0 <= edge_margin) or (right < image_size[0] and x1 >= view_width - edge_margin)
            or (top > 0 and y0 <= edge_margin) or (bottom < image_size[1] and y1 >= view_height - edge_margin)):
        return None
    return [x0 + left, y0 + top, x1 + left, y1 + top]


def mapping_item(image_path, run_dir):
    """根据文件名建立照片与建筑文件的关联，不猜墙面或像素到米的变换。

    例如 4959323_front 对应建筑编号 4959323；其具体墙面仍需人工确认和标定。
    """
    match = re.match(r"(\d+)(?:_|$)", image_path.stem)
    building_id = match.group(1) if match else None
    return {"image_key": image_path.stem, "image_path": str(image_path),
            "source_type": "facade_texture", "building_id": building_id,
            "citygml_path": str(DATASET_DIR / f"citygml/DEBY_LOD3_{building_id}.gml") if building_id else None,
            "obj_path": str(DATASET_DIR / f"obj/DEBY_LOD2_{building_id}.obj") if building_id else None,
            "wall_id": None, "status": "pending_detection", "detection_status": "pending",
            "result_directory": str(run_dir / image_path.stem),
            "detection_json": str(run_dir / image_path.stem / "detections.json"),
            "window_count": None, "door_count": None, "ambiguous_count": None, "error": ""}
