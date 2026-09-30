"""把批量检测结果标定到已有三维墙面：在 PyCharm 中直接运行。

这是新增的教学适配实现，使用 NumPy 的线性代数和 Matplotlib 的交互窗口，
不调用原 Stage 3 的 30 米图高假设，也不调用神经网络或下载权重。

先运行 detect_my_facade.py。随后本文件按以下顺序工作：
1. 找到最新批次，选择一张已成功检测的照片，并由你确认对应的平面墙。
2. 左图点照片上的位置，右图点同一个物理位置，至少建立 4 组拟合点。
3. 另选至少 1 组没有参与拟合的检查点，并在照片上圈定本次映射的墙面区域。
4. 计算“像素 XY → 墙面 UV（米）→ 三维 XYZ”，检查误差后保存标定。
5. 只映射区域内的检测框，保留黄色 ambiguous；保存 JSON、OBJ 和预览图。

灰色门窗轮廓来自现有 CityGML，只用于人工寻找对应点，绝不复制到预测集合。
这种使用已有标注几何辅助的标定不能当作独立的三维精度评测。
输出是贴在墙面上的预测四边形预览，没有切割墙体、生成洞或修改原 CityGML。
每张照片都必须独立标定；换一个视角不能套用前一张照片的对应关系。
"""

from pathlib import Path
import hashlib
import json
import xml.etree.ElementTree as ET

import numpy as np


# ==================== 用户参数：先用当前已经熟悉的正面照片 ====================
ROOT = Path(__file__).resolve().parent
LATEST_RUN = ROOT / "my_results/batch_detection/latest_run.json"
IMAGE_KEY = "4959323_front"  # 改为 mapping_manifest.json 中其他图片的 image_key。
WALL_ID = None  # 可填写完整 WallSurface ID；None 表示在窗口中选择并确认。
RECALIBRATE = True  # 已存标定且图片/模型未变时复用；True 表示重新点选。

# 这些是人工标定的质量门槛，不是神经网络的框评分阈值。
MAX_CONTROL_RMSE_M = 0.20
MAX_CHECK_ERROR_M = 0.30
PLANE_TOLERANCE_M = 0.03  # 超过此偏离距离的墙面不适合单个平面变换。
KNOWN_FRONT_WALL = "DEBY_LOD2_4959323_5150f87d-bb94-4ffe-bdac-c5d180e863d5"
NS = {"g": "http://www.opengis.net/gml", "b": "http://www.opengis.net/citygml/building/2.0"}
COLORS = {"window": "#1689ff", "door": "#ff941a", "ambiguous": "#d6ad00"}


def file_hash(path):
    """用内容指纹防止把旧照片或旧墙面的标定套到已经改变的文件上。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def plane_frame(rings):
    """为墙建立正交坐标：origin + u*U + v*V；U、V 的长度单位为米。

    SVD 找到最贴近墙面顶点的平面。V 尽量朝上，U 的方向固定，避免重复运行翻转。
    原模型的坐标很大，先减去中心再运算，提高数值稳定性。
    """
    points = np.concatenate(rings)
    origin = points.mean(axis=0)
    _, values, axes = np.linalg.svd(points - origin, full_matrices=False)
    if len(values) < 3 or values[1] < 1e-6:
        raise ValueError("墙面顶点共线，无法建立二维墙面坐标")
    normal = axes[-1]
    deviation = float(np.abs((points - origin) @ normal).max())
    if deviation > PLANE_TOLERANCE_M:
        raise ValueError(f"墙面非平面，最大偏离 {deviation:.3f} m")
    vertical = np.array([0., 0., 1.])
    v = vertical - np.dot(vertical, normal) * normal
    if np.linalg.norm(v) < 0.1:
        raise ValueError("这不是可按立面方式标定的墙面")
    v /= np.linalg.norm(v)
    u = np.cross(v, normal)
    if u[np.argmax(np.abs(u))] < 0:
        u = -u
    return {"origin": origin, "u": u, "v": v, "max_plane_deviation_m": deviation}


def to_uv(points, frame):
    return (np.asarray(points) - frame["origin"]) @ np.column_stack((frame["u"], frame["v"]))


def to_xyz(points, frame):
    return frame["origin"] + np.asarray(points) @ np.stack((frame["u"], frame["v"]))


def read_rings(element):
    """只读取给定几何范围中的 exterior，不把已有门窗洞当成预测。"""
    rings = []
    for pos in element.findall(".//g:Polygon/g:exterior/g:LinearRing/g:posList", NS):
        values = np.fromstring(pos.text or "", sep=" ")
        if len(values) >= 9 and len(values) % 3 == 0:
            rings.append(values.reshape(-1, 3))
    return rings


def load_walls(path):
    """读已有墙的外轮廓，并单独保存灰色参考开口。支持当前数据的 CityGML 2.0。"""
    root = ET.parse(path).getroot()
    walls = []
    for index, node in enumerate(root.findall(".//b:WallSurface", NS)):
        wall_id = node.get("{" + NS["g"] + "}id") or f"unnamed_wall_{index}"
        rings, references = [], []
        for child in node:
            # 只读取 WallSurface 自身的几何，不跨入 bldg:opening 的子树。
            if child.tag.rsplit("}", 1)[-1] in ("lod2MultiSurface", "lod3MultiSurface"):
                rings.extend(read_rings(child))
                # 直接取墙面已有洞口边界作为灰色参照，不展开门窗细部的上万个三角面。
                # 某些触地开口属于外轮廓的一部分，已由 rings 的黑色边界显示。
                for pos in child.findall(".//g:Polygon/g:interior/g:LinearRing/g:posList", NS):
                    values = np.fromstring(pos.text or "", sep=" ")
                    if len(values) >= 9 and len(values) % 3 == 0:
                        references.append(values.reshape(-1, 3))
        if not rings:
            continue
        try:
            frame = plane_frame(rings)
        except ValueError:
            continue  # 非平面或退化墙不能用单个 homography；不冒充已匹配。
        uv = [to_uv(ring, frame) for ring in rings]
        span = np.ptp(np.concatenate(uv), axis=0)
        walls.append({"id": wall_id, "rings": rings, "uv": uv, "frame": frame,
                      "references": references, "size": span, "area_rank": float(np.prod(span))})
    if not walls:
        raise ValueError("CityGML 中没有可读取的平面 WallSurface，不能自动猜测墙面。")
    return sorted(walls, key=lambda wall: wall["area_rank"], reverse=True)


def apply_homography(points, matrix):
    points = np.asarray(points, dtype=float)
    homogeneous = np.column_stack((points, np.ones(len(points)))) @ np.asarray(matrix).T
    if not np.isfinite(homogeneous).all() or np.any(np.abs(homogeneous[:, 2]) < 1e-9):
        raise ValueError("变换在该位置趋于无穷，需重新选择标定点")
    return homogeneous[:, :2] / homogeneous[:, 2, None]


def fit_homography(image_points, wall_points):
    """归一化 DLT：至少 4 个不退化的对应点，解出 3×3 的平面投影变换。

    4 个拟合点可以被精确拟合，拟合误差接近 0 并不能证明对应关系正确，
    所以交互流程另外要求检查点。建议选 6～8 个覆盖墙面上下左右的拟合点。
    """
    src, dst = np.asarray(image_points, float), np.asarray(wall_points, float)
    if src.ndim != 2 or src.shape[1:] != (2,) or src.shape != dst.shape or len(src) < 4:
        raise ValueError("需要至少 4 组完整的二维对应点")
    if not np.isfinite(src).all() or not np.isfinite(dst).all():
        raise ValueError("对应点中含无效数值")

    def normalize(points):
        center = points.mean(axis=0)
        if np.linalg.matrix_rank(points - center) < 2:
            raise ValueError("对应点不能集中在同一条直线上")
        scale = np.sqrt(2.) / np.linalg.norm(points - center, axis=1).mean()
        transform = np.array([[scale, 0, -scale * center[0]],
                              [0, scale, -scale * center[1]], [0, 0, 1]])
        return apply_homography(points, transform), transform

    src_n, src_t = normalize(src)
    dst_n, dst_t = normalize(dst)
    rows = []
    for (x, y), (u, v) in zip(src_n, dst_n):
        rows.extend([[-x, -y, -1, 0, 0, 0, u*x, u*y, u],
                     [0, 0, 0, -x, -y, -1, v*x, v*y, v]])
    if np.linalg.matrix_rank(rows) < 8:
        raise ValueError("点的布局退化，请将点分散到墙面的不同区域")
    _, _, right = np.linalg.svd(rows, full_matrices=True)
    matrix = np.linalg.inv(dst_t) @ right[-1].reshape(3, 3) @ src_t
    if abs(np.linalg.det(matrix)) < 1e-12 or np.linalg.cond(matrix) > 1e12:
        raise ValueError("标定变换不稳定，请检查点的顺序和分布")
    return matrix / np.linalg.norm(matrix)


def inside_polygon(points, polygon, tolerance=0.0):
    """边界算在区域内。显式首尾闭合，避免 Matplotlib 忽略最后一个角。"""
    from matplotlib.path import Path as PlotPath
    points, polygon = np.asarray(points), np.asarray(polygon)
    closed = np.vstack((polygon, polygon[0]))
    inside = PlotPath(closed).contains_points(points)
    for start, end in zip(closed[:-1], closed[1:]):
        delta = end - start
        if np.dot(delta, delta) == 0:
            distance = np.linalg.norm(points - start, axis=1)
        else:
            t = np.clip((points - start) @ delta / np.dot(delta, delta), 0, 1)
            distance = np.linalg.norm(points - start - t[:, None] * delta, axis=1)
        inside |= distance <= max(tolerance, 1e-8)
    return inside


def inside_wall(points, wall):
    return np.logical_or.reduce([inside_polygon(points, ring, 0.03) for ring in wall["uv"]])


def polygon_inside_ring(polygon, ring, tolerance=0.):
    """检查角点、中心和每条边的所有边界交点之间的区间，包含贴边/触点情况。

    只检查角点和严格相交不够：一个框可正好卡在凹缺口的边界上，而内部仍在墙外。
    检测框经有效 homography 变成凸四边形，此处检查它的完整边界及内部代表点。
    """
    polygon, ring = np.asarray(polygon), np.asarray(ring)
    if not inside_polygon(np.vstack((polygon, polygon.mean(axis=0))), ring, tolerance).all():
        return False
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        ab = b-a
        length_squared = np.dot(ab, ab)
        if length_squared < 1e-16:
            continue
        divisions = [0., 1.]
        for c, d in zip(ring, np.roll(ring, -1, axis=0)):
            cd = d-c
            denominator = np.cross(ab, cd)
            if abs(denominator) > 1e-12:
                t, u = np.cross(c-a, cd)/denominator, np.cross(c-a, ab)/denominator
                if -1e-9 <= t <= 1+1e-9 and -1e-9 <= u <= 1+1e-9:
                    divisions.append(float(np.clip(t, 0, 1)))
            elif abs(np.cross(ab, c-a)) < 1e-9:
                # 共线的边也必须按端点切分，不能只检查“严格穿越”。
                divisions.extend(float(np.clip(np.dot(point-a, ab)/length_squared, 0, 1))
                                 for point in (c, d))
        divisions = np.unique(divisions)
        midpoints = a + ((divisions[:-1] + divisions[1:])/2)[:, None]*ab
        if not inside_polygon(midpoints, ring, tolerance).all():
            return False
    return True


def validate_region(region, image_size):
    """拒绝越界、面积为零或交叉的墙面区域；不能把点按任意顺序连成蝴蝶形。"""
    polygon = np.asarray(region, dtype=float)
    if len(polygon) > 1 and np.linalg.norm(polygon[-1] - polygon[0]) < 1e-6:
        polygon = polygon[:-1]
    if polygon.ndim != 2 or polygon.shape[1:] != (2,) or len(polygon) < 3 or not np.isfinite(polygon).all():
        raise ValueError("请用 Wall region 沿墙面边界依次点至少 3 个不同角")
    if np.any(polygon < 0) or np.any(polygon > np.asarray(image_size)):
        raise ValueError("墙面区域的点超出照片边界，请撤销后在照片内部点选")
    edges = np.roll(polygon, -1, axis=0) - polygon
    if np.any(np.linalg.norm(edges, axis=1) < 1e-6):
        raise ValueError("区域边界有重复点，请撤销重复点")
    area = abs(np.sum(polygon[:, 0]*np.roll(polygon[:, 1], -1)
                      - polygon[:, 1]*np.roll(polygon[:, 0], -1))) / 2
    if area < 1:
        raise ValueError("墙面区域面积太小或边界交叉，请按顺序沿外边界点角")
    for i in range(len(polygon)):
        a, b = polygon[i], polygon[(i+1) % len(polygon)]
        for j in range(i+2, len(polygon)):
            if (j+1) % len(polygon) == i:
                continue
            c, d = polygon[j], polygon[(j+1) % len(polygon)]
            if np.cross(b-a, c-a)*np.cross(b-a, d-a) < 0 and np.cross(d-c, a-c)*np.cross(d-c, b-c) < 0:
                raise ValueError("墙面区域边界交叉，请沿边界顺时针或逆时针依次点角")
    return polygon.tolist()


def calibration_quality(controls, checks, matrix):
    if len(checks) < 1:
        raise ValueError("还需要至少 1 组不参与拟合的检查点，建议 2～3 组")
    controls, checks = np.asarray(controls), np.asarray(checks)
    # 防止拿原来的拟合点重复充当“独立检查点”。
    distances = np.linalg.norm(checks[:, None, 0, :] - controls[None, :, 0, :], axis=2)
    if np.any(distances.min(axis=1) < 5):
        raise ValueError("检查点离拟合点不到 5 像素，请换一个不同位置")
    control_errors = np.linalg.norm(apply_homography(controls[:, 0], matrix) - controls[:, 1], axis=1)
    check_errors = np.linalg.norm(apply_homography(checks[:, 0], matrix) - checks[:, 1], axis=1)
    rmse = float(np.sqrt(np.mean(control_errors ** 2)))
    if rmse > MAX_CONTROL_RMSE_M or check_errors.max() > MAX_CHECK_ERROR_M:
        raise ValueError(f"标定误差较大：拟合 RMSE={rmse:.3f} m，检查最大误差={check_errors.max():.3f} m。"
                         "请核对点的顺序、墙面身份及左右方向。")
    return {"control_rmse_m": rmse, "check_errors_m": check_errors.tolist(),
            "check_max_error_m": float(check_errors.max()),
            "note": "人工点选的一致性检查，不是独立三维精度评测。"}


def draw_wall(ax, wall, show_references=True):
    ax.clear()
    for ring in wall["uv"]:
        closed = np.vstack((ring, ring[0]))
        ax.plot(closed[:, 0], closed[:, 1], color="black", linewidth=1)
    if show_references:
        for ring in wall["references"]:
            uv = to_uv(ring, wall["frame"])
            ax.plot(uv[:, 0], uv[:, 1], color="0.65", linewidth=0.65)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Wall U (m)")
    ax.set_ylabel("Wall V (m)")
    ax.grid(alpha=.15)


def choose_wall(image, walls, preferred_id=None):
    """先看照片、墙面平面图和建筑俯视位置，再确认；绝不只按名字猜墙。"""
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button
    index = next((i for i, wall in enumerate(walls) if wall["id"] == preferred_id), 0)
    state = {"index": index, "selected": None}
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.subplots_adjust(bottom=.20, top=.82)

    def redraw():
        wall = walls[state["index"]]
        axes[0].clear()
        axes[0].imshow(image)
        axes[0].set_title("Your photo")
        axes[0].axis("off")
        axes[1].clear()
        origin = wall["frame"]["origin"][:2]
        for candidate in walls:
            for ring in candidate["rings"]:
                xy = ring[:, :2] - origin
                axes[1].plot(xy[:, 0], xy[:, 1], color="0.8", linewidth=.6)
        for ring in wall["rings"]:
            xy = ring[:, :2] - origin
            axes[1].plot(xy[:, 0], xy[:, 1], color="red", linewidth=3)
        axes[1].set_aspect("equal", adjustable="datalim")
        axes[1].set_title("Building top view: selected wall in red")
        draw_wall(axes[2], wall)
        axes[2].set_title("Selected wall; gray = existing references")
        fig.suptitle(f"Choose the SAME wall as the photo: {state['index']+1}/{len(walls)}\n"
                     f"{wall['id']}\nwidth/height: {wall['size'][0]:.2f} / {wall['size'][1]:.2f} m")
        fig.canvas.draw_idle()

    def move(step):
        state["index"] = (state["index"] + step) % len(walls)
        redraw()

    def choose(_):
        state["selected"] = walls[state["index"]]
        plt.close(fig)

    buttons = [Button(fig.add_axes(rect), label) for rect, label in
               [([.18, .05, .18, .07], "Previous wall"), ([.41, .05, .18, .07], "Next wall"),
                ([.64, .05, .18, .07], "Use this wall")]]
    buttons[0].on_clicked(lambda _: move(-1))
    buttons[1].on_clicked(lambda _: move(1))
    buttons[2].on_clicked(choose)
    redraw()
    plt.show(block=True)
    if state["selected"] is None:
        raise ValueError("已取消选墙，没有生成三维结果。")
    return state["selected"]


def interactive_calibration(image, wall):
    """左图→右图构成一对；支持撤销和独立检查点，最后圈出属于该平面的区域。

    Matplotlib 工具栏可以缩放。缩放/平移模式开启时不会收集坐标，点选前要关闭。
    按 Save + map 仅在拟合点、检查点和区域都合格时结束；直接关窗表示取消。
    """
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button
    controls, checks, region = [], [], []
    state = {"mode": "Fit pair", "pending": None, "result": None, "error": "", "drawn": False}
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.subplots_adjust(bottom=.23, top=.78)

    def redraw():
        # 点一个角之后保留当前放大位置，避免宽立面每点一次就退回全图。
        limits = [(ax.get_xlim(), ax.get_ylim()) for ax in axes] if state["drawn"] else None
        axes[0].clear()
        axes[0].imshow(image)
        axes[0].set_title("PHOTO: click a recognizable physical point")
        axes[0].set_xlabel("Original image X (pixels)")
        axes[0].set_ylabel("Original image Y (pixels)")
        draw_wall(axes[1], wall)
        axes[1].set_title("WALL: click the SAME point; gray outlines are reference GT")
        for name, pairs, color in (("F", controls, "#178bd3"), ("C", checks, "#ba36bf")):
            for i, pair in enumerate(pairs, 1):
                for ax, point in zip(axes, pair):
                    ax.plot(*point, marker="+", color=color, markersize=9)
                    ax.annotate(f"{name}{i}", point, color=color, fontsize=9)
        if region:
            polygon = np.asarray(region + ([region[0]] if len(region) >= 3 else []))
            axes[0].plot(polygon[:, 0], polygon[:, 1], "g.-", linewidth=1)
        if state["pending"] is not None:
            axes[0].plot(*state["pending"], "rx")
        destination = "PHOTO, then WALL" if state["pending"] is None else "WALL (same physical point)"
        if state["mode"] == "Wall region":
            destination = "PHOTO only: click wall boundary in order; 3+ vertices"
        fig.suptitle(f"Mode: {state['mode']} | Next: {destination}\n"
                     f"Fit pairs: {len(controls)} (4+); check pairs: {len(checks)} (1+); "
                     f"region vertices: {len(region)} (3+)\n"
                     "Fit points must spread across the wall. Check points must be different.\n"
                     + state["error"], fontsize=10)
        if limits is not None:
            for ax, (xlim, ylim) in zip(axes, limits):
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
        state["drawn"] = True
        fig.canvas.draw_idle()

    def set_mode(mode):
        state.update(mode=mode, pending=None, error="")
        redraw()

    def click(event):
        toolbar = getattr(fig.canvas.manager, "toolbar", None)
        if event.button != 1 or (toolbar is not None and toolbar.mode):
            return
        if event.inaxes not in axes or event.xdata is None:
            return
        point = [float(event.xdata), float(event.ydata)]
        if state["mode"] == "Wall region":
            if event.inaxes == axes[0]:
                region.append(point)
        elif state["pending"] is None and event.inaxes == axes[0]:
            state["pending"] = point
        elif state["pending"] is not None and event.inaxes == axes[1]:
            if not inside_wall([point], wall)[0]:
                state["error"] = "Selected WALL point is outside the wall boundary."
            else:
                (controls if state["mode"] == "Fit pair" else checks).append([state["pending"], point])
                state["pending"] = None
                state["error"] = ""
        redraw()

    def undo(_):
        if state["pending"] is not None:
            state["pending"] = None
        else:
            values = region if state["mode"] == "Wall region" else controls if state["mode"] == "Fit pair" else checks
            if values:
                values.pop()
        state["error"] = ""
        redraw()

    def save(_):
        try:
            if state["pending"] is not None:
                raise ValueError("还有一组未完成的对应点，请补齐或撤销")
            matrix = fit_homography([pair[0] for pair in controls], [pair[1] for pair in controls])
            checked_region = validate_region(region, image.size)
            all_image_points = [pair[0] for pair in controls + checks]
            if not inside_polygon(all_image_points, checked_region, 1.).all():
                raise ValueError("拟合点或检查点在所圈墙面区域外，请核对区域边界")
            span = np.ptp(np.asarray([pair[1] for pair in controls]), axis=0)
            if np.any(span < wall["size"] * .25):
                raise ValueError("拟合点过于集中，请分散到墙面的上下左右")
            quality = calibration_quality(controls, checks, matrix)
            state["result"] = {"image_to_wall_homography": matrix.tolist(), "control_pairs": controls,
                               "check_pairs": checks, "image_wall_region": checked_region, "quality": quality}
            plt.close(fig)
        except ValueError as error:
            print(f"标定未通过：{error}", flush=True)
            state["error"] = "Cannot save yet. See the PyCharm console for the reason."
            redraw()

    buttons = []
    for i, label in enumerate(("Fit pair", "Check pair", "Wall region", "Undo", "Save + map")):
        button = Button(fig.add_axes([.04 + i*.19, .065, .17, .065]), label)
        button.on_clicked((lambda _, mode=label: set_mode(mode)) if i < 3 else undo if i == 3 else save)
        buttons.append(button)
    fig.canvas.mpl_connect("button_press_event", click)
    redraw()
    plt.show(block=True)
    if state["result"] is None:
        raise ValueError("已取消标定，没有生成三维结果。")
    return state["result"]


def map_detections(detections, matrix, wall, image_region):
    """预测框的四角映射到墙上；不完整落在选定平面的框保留为未映射记录。

    这里只产生平面四边形，矩形框不会自动变成真实拱形轮廓。检测类别直接沿用，
    ambiguous 不会因为附近有一个 GT Window 就被改成 window。
    """
    mapped, rejected = [], []
    for index, detection in enumerate(detections, 1):
        record = {"prediction_id": f"P{index:04d}", "class": detection["class"],
                  "box_image_xyxy": detection["box"], "detection": detection}
        box = np.asarray(detection["box"], dtype=float)
        if box.shape != (4,) or not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
            record["reason"] = "invalid_detection_box"
            rejected.append(record)
            continue
        x0, y0, x1, y1 = box
        corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        if not polygon_inside_ring(corners, image_region, 1.):
            record["reason"] = "box_not_fully_inside_selected_image_wall_region"
            rejected.append(record)
            continue
        try:
            uv = apply_homography(corners, matrix)
        except ValueError:
            record["reason"] = "unstable_homography_at_detection"
            rejected.append(record)
            continue
        # 保守地要求四边形落在一个外轮廓内；不跨越离散墙片，也不填掉原模型的凹缺口。
        if not any(polygon_inside_ring(uv, ring, .03) for ring in wall["uv"]):
            record["reason"] = "mapped_box_not_fully_inside_wall_outer_boundary"
            rejected.append(record)
            continue
        record.update({"wall_uv_m": uv.tolist(), "vertices_xyz": to_xyz(uv, wall["frame"]).tolist(),
                       "geometry_role": "prediction_preview_only", "wall_id": wall["id"]})
        mapped.append(record)
    return mapped, rejected


def export_obj(path, predictions):
    """保存原模型坐标系下的预测面，不写 GT 面，不对建筑做布尔开孔。"""
    material_path = path.with_suffix(".mtl")
    materials = {"window": (0.09, .54, 1.), "door": (1., .58, .10), "ambiguous": (1., .88, .20)}
    material_path.write_text("\n".join(f"newmtl {name}\nKd {r} {g} {b}\nd 0.85\n"
                                        for name, (r, g, b) in materials.items()), encoding="utf-8")
    lines = ["# PREDICTION PREVIEW ONLY: no wall holes; ambiguous stays unresolved.",
             "# Coordinates are the source CityGML XYZ, not local image coordinates.",
             f"mtllib {material_path.name}"]
    for index, prediction in enumerate(predictions):
        lines.extend([f"g {prediction['prediction_id']}_{prediction['class']}",
                      f"usemtl {prediction['class']}"])
        lines.extend("v " + " ".join(f"{v:.8f}" for v in point) for point in prediction["vertices_xyz"])
        lines.append("f " + " ".join(str(index*4 + i) for i in range(1, 5)))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_obj_faces(path):
    """只读已有 OBJ 的顶点/面，用于三维灰色楼体背景。"""
    vertices, faces = [], []
    if not path or not Path(path).is_file():
        return []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        words = line.split()
        if words and words[0] == "v":
            vertices.append([float(value) for value in words[1:4]])
        elif words and words[0] == "f":
            indices = [int(word.split("/")[0]) for word in words[1:]]
            faces.append(np.array([vertices[i-1 if i > 0 else i] for i in indices]))
    return faces


def show_result(output_dir, wall, predictions, obj_path):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig = plt.figure(figsize=(16, 7))
    flat = fig.add_subplot(121)
    draw_wall(flat, wall, show_references=False)
    flat.set_title("Mapped PREDICTIONS (GT openings are not drawn)")
    for record in predictions:
        uv = np.asarray(record["wall_uv_m"])
        uv = np.vstack((uv, uv[0]))
        flat.plot(uv[:, 0], uv[:, 1], color=COLORS.get(record["class"], "red"))
    space = fig.add_subplot(122, projection="3d")
    origin = wall["frame"]["origin"]
    faces = read_obj_faces(obj_path)
    if faces:
        space.add_collection3d(Poly3DCollection([face-origin for face in faces], facecolor="0.85",
                                               edgecolor="0.6", alpha=.12, linewidth=.4))
    # 红色墙边界表示当前用于标定的 CityGML 平面。它可能与 LoD2 OBJ 表面有模型差异。
    for ring in wall["rings"]:
        xyz = ring - origin
        space.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color="red", linewidth=.8)
    for record in predictions:
        xyz = np.asarray(record["vertices_xyz"]) - origin
        space.add_collection3d(Poly3DCollection([xyz], facecolor=COLORS.get(record["class"], "red"),
                                               edgecolor="black", linewidth=.3, alpha=.65))
    all_points = np.concatenate(faces if faces else wall["rings"]) - origin
    low, high = all_points.min(axis=0), all_points.max(axis=0)
    for setter, a, b in zip((space.set_xlim, space.set_ylim, space.set_zlim), low, high):
        setter(a-1, b+1)
    space.set_box_aspect(np.maximum(high-low, 1.))
    space.set_xlabel("X - origin (m)")
    space.set_ylabel("Y - origin (m)")
    space.set_zlabel("Z - origin (m)")
    space.set_title("3D preview: rotate with the mouse\nBlue=window; orange=door; yellow=unresolved")
    fig.suptitle("Preview geometry only: the building has NOT been cut or modified.")
    fig.tight_layout()
    fig.savefig(output_dir / "mapping_preview.png", dpi=170)
    plt.show(block=True)


def main():
    """主入口：不调用模型，不自动猜标定点，没有人工标定就不生成假三维结果。"""
    # 使用当前环境已有的 Tk 独立窗口，避免 PyCharm 的静态 Plots 面板拦截点选。
    # 只有主入口设置；几何测试可以继续使用不弹窗的 Agg 后端。
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from PIL import Image

    if not LATEST_RUN.is_file():
        print("尚无批量检测记录。请先在 PyCharm 运行 detect_my_facade.py，再运行本文件。")
        return
    latest = json.loads(LATEST_RUN.read_text(encoding="utf-8"))
    run_dir = Path(latest["run_directory"])
    manifest = json.loads((run_dir / "mapping_manifest.json").read_text(encoding="utf-8"))
    items = manifest["items"]
    item = next((record for record in items if record["image_key"] == IMAGE_KEY), None)
    if item is None:
        raise ValueError(f"找不到 IMAGE_KEY={IMAGE_KEY}，可选：{[record['image_key'] for record in items]}")
    if item["status"] in ("detection_failed", "pending_detection") or not Path(item["detection_json"]).is_file():
        raise ValueError("该照片还没有成功检测；请先运行 detect_my_facade.py 并检查批量汇总。")
    if not item.get("citygml_path") or not Path(item["citygml_path"]).is_file():
        raise ValueError("该照片尚未匹配已有 CityGML，需要先确认 building_id/citygml_path。")
    report = json.loads(Path(item["detection_json"]).read_text(encoding="utf-8"))
    image_path, citygml_path = Path(item["image_path"]), Path(item["citygml_path"])
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    if list(image.size) != report["image_size"] or Path(report["image"]).resolve() != image_path.resolve():
        raise ValueError("检测记录与当前照片的路径/尺寸不一致，请重新检测。")
    identity = {"image_key": IMAGE_KEY, "image_path": str(image_path.resolve()),
                "image_sha256": file_hash(image_path), "citygml_path": str(citygml_path.resolve()),
                "citygml_sha256": file_hash(citygml_path), "image_size": list(image.size)}
    if report.get("image_sha256") and report["image_sha256"] != identity["image_sha256"]:
        raise ValueError("照片内容与检测时不同，即使尺寸相同也不能使用旧框，请重新检测。")
    walls = load_walls(citygml_path)
    calibration_path = ROOT / "my_results/calibrations" / f"{IMAGE_KEY}.json"
    calibration = None
    if calibration_path.is_file() and not RECALIBRATE:
        saved = json.loads(calibration_path.read_text(encoding="utf-8"))
        if saved.get("identity") != identity:
            raise ValueError("已存标定与当前照片/CityGML 的内容或身份不一致。确认数据后设 RECALIBRATE=True 重新标定。")
        if WALL_ID and saved.get("wall_id") != WALL_ID:
            raise ValueError("WALL_ID 与保存的标定不同，请设 RECALIBRATE=True 重新标定。")
        wall = next((candidate for candidate in walls if candidate["id"] == saved["wall_id"]), None)
        if wall is None:
            raise ValueError("保存标定中的墙面在当前模型里不存在。")
        calibration = saved
        print(f"复用已核对照片/模型指纹的标定：{calibration_path}")
    else:
        preferred = WALL_ID or item.get("wall_id") or (KNOWN_FRONT_WALL if IMAGE_KEY == "4959323_front" else None)
        if WALL_ID and not any(wall["id"] == WALL_ID for wall in walls):
            raise ValueError("WALL_ID 不在可用的平面墙列表中，请核对 ID。")
        print("第一步：确认照片与红色墙的位置相同，点击 Use this wall。")
        print("第二步：Fit pair 左图→右图至少 4 对，建议 6～8 对分散的点。")
        print("第三步：Check pair 另选至少 1 对不同点；Wall region 沿照片中的墙边界依次点角。")
        print("第四步：Save + map。若有错误，控制台会解释；关窗则取消。")
        print("灰色轮廓是已有 CityGML 标注，仅辅助点选，不是本次预测或独立评测。")
        wall = choose_wall(image, walls, preferred)
        calibration = interactive_calibration(image, wall)
        calibration.update({"version": 1, "identity": identity, "wall_id": wall["id"],
                            "reference_geometry_used_for_manual_calibration": True})
        write_json(calibration_path, calibration)

    matrix = np.asarray(calibration["image_to_wall_homography"], dtype=float)
    quality = calibration_quality(calibration["control_pairs"], calibration["check_pairs"], matrix)
    region = validate_region(calibration["image_wall_region"], image.size)
    denominators = np.column_stack((region, np.ones(len(region)))) @ matrix[2]
    if denominators.min()*denominators.max() <= 0:
        raise ValueError("投影变换的无穷远线穿过所选区域，请重新检查标定点，不能映射这片区域。")
    mapped, rejected = map_detections(report["detections"], matrix, wall, region)
    output_dir = run_dir / "mapping_3d" / IMAGE_KEY
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {"scope": "Predicted planar quadrilateral preview, no wall cutting or CityGML modification.",
              "source_image": str(image_path), "source_detection_json": item["detection_json"],
              "source_detection_sha256": file_hash(item["detection_json"]), "source_citygml": str(citygml_path),
              "building_obj_for_context_only": item.get("obj_path"), "wall_id": wall["id"],
              "calibration_path": str(calibration_path), "calibration_quality": quality,
              "coordinate_system": "Same XYZ coordinates and units as the source CityGML.",
              "frame": {name: value.tolist() if isinstance(value, np.ndarray) else value
                        for name, value in wall["frame"].items()},
              "mapped_count": len(mapped), "rejected_count": len(rejected),
              "predictions": mapped, "unmapped_detections": rejected,
              "ambiguous_policy": "Preserved as unresolved yellow preview; never converted from GT labels."}
    write_json(output_dir / "predictions_3d.json", result)
    write_json(output_dir / "calibration.json", calibration)
    export_obj(output_dir / "predictions_preview.obj", mapped)
    print(f"完成三维预览坐标转换：{len(mapped)} 个框映射，{len(rejected)} 个框未映射。")
    print(f"结果：{output_dir}\n未修改已有建筑，没有开孔；待确认类别仍然待确认。")
    if not mapped:
        print("没有框通过墙面范围检查，请查看 predictions_3d.json 的 unmapped_detections。")
    show_result(output_dir, wall, mapped, item.get("obj_path"))
    plt.close("all")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, KeyError) as error:
        print(f"三维映射未完成：{error}")
        raise SystemExit(1)
