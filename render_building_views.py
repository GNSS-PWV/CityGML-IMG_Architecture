"""给一栋已有 CityGML 建筑生成多视角图库——直接在 PyCharm 运行本文件。

本文件是为当前数据集新写的渲染入口，不是参考仓库的 Stage 1/2/3 原代码。
不调用 Grounding DINO、DINO 或 RoMa，不使用检测框，也不上传照片。
读入的是数据集已有 LoD3 模型；图上的门窗来自模型本身。

依赖：numpy、Pillow、mapbox-earcut==2.0.0、shapely==2.0.6（Python 3.12 的 torch_1）。
earcut 只负责把“带孔的多边形”切成三角形；不是神经网络，无模型权重。
随后用 CPU 深度缓冲渲染真实三角形，保存图像、深度及相机坐标变换。
"""
from pathlib import Path
from datetime import datetime
from collections import Counter
import hashlib
import json
import time
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import Polygon
from shapely.validation import make_valid, explain_validity
from shapely.ops import unary_union
try:
    import mapbox_earcut
except ImportError as exc:
    raise ImportError("请在当前解释器安装 mapbox-earcut==2.0.0；详见 04_单栋建筑多视角渲染教程.md") from exc


# ===== 1. 日常需要修改的参数在这里 =====
ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "Drills/Texture2LoD3_dataset"
BUILDING_ID = "4959323"
GML_PATH = DATASET / "citygml" / ("DEBY_LOD3_" + BUILDING_ID + ".gml")
OUTPUT_ROOT = ROOT / "my_results/rendered_buildings"

# 这个墙面是前面人工标定 4959323_front 时已核对的正面。
# 它只用来定义“从哪个方向开始看”，没有使用照片给模型贴图。
# 换建筑时请设为 None：程序会选面积最大的外墙作为起始方向，名称改为 view。
FRONT_WALL_ID = "DEBY_LOD2_4959323_5150f87d-bb94-4ffe-bdac-c5d180e863d5"
IMAGE_SIZE = (1800, 1200)       # 输出宽、高；不会改变原始模型几何。
MARGIN_FRACTION = 0.06         # 模型与画面边缘的留白比例。
OBLIQUE_ELEVATION_DEG = 22.0   # 四张斜视图稍微俯看，可同时看到屋顶和两个立面。
MAX_REFERENCE_WALL_TILT_DEG = 5.0  # 仅筛选环绕起点：墙法向与水平面的夹角上限。
# 实验 E 只渲染能作为照片映射目标的“大型竖直单平面外立面”。CityGML 会把
# 一整面楼拆成窗台、饰线等许多 WallSurface；它们不是独立的拍摄立面。
WALL_VIEW_MAX_TILT_DEG = 10.0
WALL_VIEW_MIN_AREA_M2 = 10.0
WALL_VIEW_RELATIVE_AREA = .12
WALL_VIEW_YAWS_DEG = (-15.0, 0.0, 15.0)
USE_MODEL_COLORS = True       # 使用已有 diffuseColor；无颜色的面用下方默认色。
BACKGROUND = (246, 245, 241)

# 采用不透明、柔和的材质。原模型的玻璃透明度不参与渲染，保证每个可见像素
# 有唯一的第一层几何深度。不会凭空补造砖纹、窗户、建筑装饰或真实照片纹理。
FALLBACK_COLORS = {"WallSurface": (202, 196, 184), "RoofSurface": (157, 145, 135),
                   "GroundSurface": (164, 160, 150), "Window": (94, 112, 109),
                   "Door": (115, 107, 93)}
NS = {"g": "http://www.opengis.net/gml", "b": "http://www.opengis.net/citygml/building/2.0",
      "a": "http://www.opengis.net/citygml/appearance/2.0"}
GML_ID = "{" + NS["g"] + "}id"


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def clean_ring(points):
    """GML 环最后一点通常重复第一点；三角化前去掉闭合重复点及连续重复点。"""
    points = np.asarray(points, dtype=np.float64)
    if len(points) > 1:
        points = points[np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-9]]
    if len(points) > 1 and np.linalg.norm(points[-1] - points[0]) < 1e-9:
        points = points[:-1]
    return points


def triangulate_rings(rings, repair_info=None):
    """外环+孔环 → 三角形的三维顶点。返回 None 表示毫米取整后已退化的面。

    先把平面上的三维点投影到二维，再交给 earcut，最后按原索引取回三维点。
    特别保留墙上的窗洞，不能简单把整个墙矩形填满。
    """
    outer = clean_ring(rings[0])
    if len(outer) < 3:
        return None
    center = outer.mean(axis=0)
    _, singular, axes = np.linalg.svd(outer - center, full_matrices=False)
    if singular[1] < 1e-8:
        return None
    cleaned = [outer] + [p for p in (clean_ring(r) for r in rings[1:]) if len(p) >= 3]
    points = np.concatenate(cleaned)
    deviation = float(np.max(np.abs((points - center) @ axes[2])))
    if deviation > 0.01:
        raise ValueError("多边形偏离平面超过 1 cm，不能直接进行平面三角化。")
    # 沿法向量最大分量轴投影（而不旋转二维坐标），使竖直线仍能保持严格
    # 共线，避免极细窗框的回折边被浮点旋转变成数值上自交的狭长三角形。
    dropped_axis = int(np.argmax(np.abs(axes[2])))
    kept_axes = [axis for axis in range(3) if axis != dropped_axis]
    uv = np.ascontiguousarray((points - center)[:, kept_axes])
    area_scale = 1 / abs(axes[2, dropped_axis])
    def restore_projected(tri_uv):
        relative = np.zeros(tri_uv.shape[:-1] + (3,))
        relative[..., kept_axes] = tri_uv
        relative[..., dropped_axis] = -(tri_uv @ axes[2, kept_axes]) / axes[2, dropped_axis]
        return center + relative
    ends = np.cumsum([len(r) for r in cleaned], dtype=np.uint32)
    # 实际模型有几个孔略微越出墙体边界，还有毫米级取整造成的自交细面。
    # 先检查平面拓扑；外环减去孔的并集，可把越界孔正确转成墙边的凹口。
    # 只修复渲染用的临时几何，不改原始 GML；每次修复均写入审计文件。
    uv_rings = []
    start = 0
    for end in ends:
        uv_rings.append(uv[start:end])
        start = end
    polygon = Polygon(uv_rings[0], uv_rings[1:])
    if not polygon.is_valid:
        reason = explain_validity(polygon)
        exterior = make_valid(Polygon(uv_rings[0]))
        holes = [make_valid(Polygon(r)) for r in uv_rings[1:]]
        repaired = exterior.difference(unary_union(holes)) if holes else exterior
        if repair_info is not None:
            repair_info.update({"reason": reason, "original_algebraic_area_m2": float(polygon.area * area_scale),
                                "repaired_area_m2": float(repaired.area * area_scale),
                                "projection_max_deviation_m": deviation})
        def polygon_parts(geometry):
            if geometry.geom_type == "Polygon":
                yield geometry
            elif hasattr(geometry, "geoms"):
                for part in geometry.geoms:
                    yield from polygon_parts(part)
        pieces = []
        for part in polygon_parts(repaired):
            part_rings = [np.asarray(part.exterior.coords)[:-1]] + [np.asarray(r.coords)[:-1] for r in part.interiors]
            part_uv = np.ascontiguousarray(np.concatenate(part_rings))
            part_ends = np.cumsum([len(r) for r in part_rings], dtype=np.uint32)
            part_indices = mapbox_earcut.triangulate_float64(part_uv, part_ends).reshape(-1, 3)
            pieces.append(part_uv[part_indices])
        if not pieces:
            return None
        tri_uv = np.concatenate(pieces)
        triangles = restore_projected(tri_uv)
        expected_area = repaired.area
    else:
        indices = mapbox_earcut.triangulate_float64(uv, ends).reshape(-1, 3)
        triangles = points[indices]
        tri_uv = uv[indices]
        expected_area = polygon.area
    # 面积核对基于修复后的合法多边形，不能放宽容差来掩盖窗洞被填满。
    edge_a, edge_b = tri_uv[:, 1] - tri_uv[:, 0], tri_uv[:, 2] - tri_uv[:, 0]
    actual_area = np.abs(edge_a[:, 0] * edge_b[:, 1] - edge_a[:, 1] * edge_b[:, 0]).sum() * .5
    if expected_area <= 1e-10:
        return None
    if not np.isclose(actual_area, expected_area, rtol=1e-5, atol=1e-7):
        raise ValueError("带孔多边形三角化面积不一致。")
    # 恢复原始外环的正反面朝向，供材质/光照使用；渲染本身不剔除背面。
    relative = outer - center
    normal = np.cross(relative, np.roll(relative, -1, axis=0)).sum(axis=0)
    norm = np.linalg.norm(normal)
    if norm < 1e-12:
        return None
    normal /= norm
    valid = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                   triangles[:, 2] - triangles[:, 0]), axis=1) > 1e-12
    return triangles[valid], normal, deviation


def load_building(path):
    """只读取建筑语义节点直属几何，避免将 Window 内的面重复当成墙面。"""
    tree = ET.parse(path)
    root = tree.getroot()
    buildings = root.findall(".//b:Building", NS)
    if len(buildings) != 1:
        raise ValueError("此入口要求 GML 恰好包含一栋 Building。")
    envelope = root.find(".//g:Envelope", NS)
    lo = np.fromstring(envelope.findtext("g:lowerCorner", namespaces=NS), sep=" ")
    hi = np.fromstring(envelope.findtext("g:upperCorner", namespaces=NS), sep=" ")
    origin = (lo + hi) / 2   # 从百万级大坐标转为局部米制坐标，减少数值误差。

    materials = {True: {}, False: {}}
    for material in root.findall(".//a:X3DMaterial", NS):
        color = material.findtext("a:diffuseColor", namespaces=NS)
        if color is None:
            continue
        front = material.findtext("a:isFront", default="true", namespaces=NS).lower() != "false"
        rgb = np.clip(np.fromstring(color, sep=" ") * 255, 0, 255)
        for target in material.findall("a:target", NS):
            materials[front][target.text.strip().lstrip("#")] = rgb
    texture_uris = sorted(set(x.text.strip() for x in root.findall(".//a:imageURI", NS)))
    missing_textures = [uri for uri in texture_uris if not (path.parent / uri).exists()]

    triangles, normals, colors_front, colors_back, object_indices, objects, wall_frames = [], [], [], [], [], [], []
    stats = Counter()
    skipped, repairs = [], []
    max_deviation = 0.0

    def visit(node, parent_wall=None):
        nonlocal max_deviation
        kind = node.tag.split("}")[-1]
        node_id = node.get(GML_ID)
        if kind == "WallSurface":
            parent_wall = node_id
        geometry = None
        if kind in FALLBACK_COLORS:
            geometry = node.find("b:lod3MultiSurface", NS)
            if geometry is None:
                geometry = node.find("b:lod2MultiSurface", NS)
        if geometry is not None:
            polygons = geometry.findall(".//g:Polygon", NS)
            if polygons:
                object_index = len(objects)
                objects.append({"index": object_index, "id": node_id, "kind": kind, "wall_id": parent_wall})
                stats[kind + "_objects"] += 1
                for polygon in polygons:
                    stats["source_polygons"] += 1
                    outer = polygon.find("g:exterior/g:LinearRing/g:posList", NS)
                    if outer is None:
                        raise ValueError("发现不支持的外环表达，不能静默省略几何。")
                    positions = [outer] + polygon.findall("g:interior/g:LinearRing/g:posList", NS)
                    rings = [np.fromstring(p.text, sep=" ").reshape(-1, 3) - origin for p in positions]
                    stats["interior_rings"] += len(rings) - 1
                    polygon_id = polygon.get(GML_ID)
                    repair = {}
                    try:
                        result = triangulate_rings(rings, repair)
                    except ValueError as exc:
                        raise ValueError("多边形 %s：%s" % (polygon_id, exc)) from exc
                    if repair:
                        repairs.append(dict(repair, polygon_id=polygon_id, kind=kind, object_id=node_id))
                    if result is None:
                        skipped.append(polygon_id)
                        continue
                    tri, normal, deviation = result
                    max_deviation = max(max_deviation, deviation)
                    count = len(tri)
                    triangles.append(tri)
                    normals.append(np.tile(normal, (count, 1)))
                    fallback = FALLBACK_COLORS[kind]
                    front = materials[True].get(polygon_id, fallback) if USE_MODEL_COLORS else fallback
                    back = materials[False].get(polygon_id, fallback) if USE_MODEL_COLORS else fallback
                    colors_front.append(np.tile(front, (count, 1)))
                    colors_back.append(np.tile(back, (count, 1)))
                    object_indices.append(np.full(count, object_index, dtype=np.int32))
                    if kind == "WallSurface":
                        # 这里只保存外墙的方向，不把嵌在墙内的窗台/窗框当成外墙。
                        n = normal.copy()
                        if np.dot(n, rings[0].mean(axis=0)) < 0:
                            n = -n
                        area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum() / 2
                        wall_frames.append({"id": node_id, "normal": n, "area": float(area)})
        for child in node:
            if child.tag.split("}")[-1] not in {"lod2MultiSurface", "lod3MultiSurface", "lod2Solid", "lod3Solid"}:
                visit(child, parent_wall)

    visit(buildings[0])
    if not triangles:
        raise ValueError("模型中没有可渲染的面。")
    stats["degenerate_polygons_skipped"] = len(skipped)
    stats["rendered_polygons"] = stats["source_polygons"] - len(skipped)
    stats["triangles"] = sum(len(t) for t in triangles)
    stats["topology_repaired_polygons"] = len(repairs)
    return {"triangles": np.concatenate(triangles), "normals": np.concatenate(normals),
            "colors_front": np.concatenate(colors_front), "colors_back": np.concatenate(colors_back),
            "object_indices": np.concatenate(object_indices), "objects": objects, "walls": wall_frames,
            "origin": origin, "building_gml_id": buildings[0].get(GML_ID), "srs": envelope.get("srsName"),
            "stats": dict(stats), "skipped_polygon_ids": skipped, "topology_repairs": repairs, "max_plane_deviation_m": max_deviation,
            "texture_uris": texture_uris, "missing_textures": missing_textures}


def make_camera(points, eye_direction, image_size=IMAGE_SIZE, margin=MARGIN_FRACTION):
    """正交相机：没有近大远小，便于先看清立面，并能精确回投到三维。

    像素坐标约定：左上像素中心为 (0,0)，x 向右、y 向下。
    depth 是沿相机观察轴到表面的米制距离，不是欧氏距离，也不是高度 Z。
    """
    width, height = image_size
    direction = np.asarray(eye_direction, dtype=float)
    direction /= np.linalg.norm(direction)
    forward = -direction
    right = np.cross(forward, [0., 0., 1.])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    xy = points @ np.stack((right, up)).T
    center_xy = (xy.min(axis=0) + xy.max(axis=0)) / 2
    span = np.ptp(xy, axis=0)
    scale = float(np.min((np.array([width - 1, height - 1]) * (1 - 2 * margin)) / span))
    distance = float(np.linalg.norm(np.ptp(points, axis=0)) * 2)
    eye = direction * distance + right * center_xy[0] + up * center_xy[1]
    matrix = np.eye(4)
    matrix[0, :3] = right * scale
    matrix[0, 3] = (width - 1) / 2 - center_xy[0] * scale
    matrix[1, :3] = -up * scale
    matrix[1, 3] = (height - 1) / 2 + center_xy[1] * scale
    matrix[2, :3] = forward
    matrix[2, 3] = -eye @ forward
    return {"projection": "orthographic", "image_size_wh": [width, height],
            "pixel_convention": "top-left pixel center=(0,0), x right, y down",
            "depth_convention": "positive camera-axis distance in meters; NaN=background",
            "eye_local_m": eye.tolist(), "view_direction": forward.tolist(),
            "image_right_world": right.tolist(), "image_up_world": up.tolist(),
            "pixels_per_meter": scale, "local_to_pixel_depth": matrix.tolist(),
            "pixel_depth_to_local": np.linalg.inv(matrix).tolist()}


def rasterize(triangles, colors, object_indices, camera, background=BACKGROUND):
    """三角形逐像素光栅化 + Z-buffer，保证近处表面挡住远处表面。

    不能简单按面中心排序涂色：那会把后墙门窗画到前墙上。这里对每个像素
    比较实际深度，同时记录该像素属于哪个 GML 构件。
    """
    width, height = camera["image_size_wh"]
    matrix = np.asarray(camera["local_to_pixel_depth"])
    projected = triangles @ matrix[:3, :3].T + matrix[:3, 3]
    rgb = np.empty((height, width, 3), dtype=np.uint8)
    rgb[:] = background
    depth = np.full((height, width), np.inf, dtype=np.float64)
    object_map = np.full((height, width), -1, dtype=np.int32)
    mins = np.maximum(np.ceil(projected[:, :, :2].min(axis=1)).astype(int), 0)
    maxs = np.minimum(np.floor(projected[:, :, :2].max(axis=1)).astype(int), [width - 1, height - 1])
    for i in np.flatnonzero(np.all(maxs >= mins, axis=1)):
        tri = projected[i]
        x0, y0, z0 = tri[0]
        x1, y1, z1 = tri[1]
        x2, y2, z2 = tri[2]
        denominator = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(denominator) < 1e-9:
            continue
        xmin, ymin = mins[i]
        xmax, ymax = maxs[i]
        yy, xx = np.ogrid[ymin:ymax + 1, xmin:xmax + 1]
        w0 = ((y1 - y2) * (xx - x2) + (x2 - x1) * (yy - y2)) / denominator
        w1 = ((y2 - y0) * (xx - x2) + (x0 - x2) * (yy - y2)) / denominator
        w2 = 1 - w0 - w1
        z = w0 * z0 + w1 * z1 + w2 * z2
        block = depth[ymin:ymax + 1, xmin:xmax + 1]
        visible = (w0 >= -1e-8) & (w1 >= -1e-8) & (w2 >= -1e-8) & (z > 0) & (z < block - 1e-8)
        block[visible] = z[visible]
        rgb[ymin:ymax + 1, xmin:xmax + 1][visible] = colors[i]
        object_map[ymin:ymax + 1, xmin:xmax + 1][visible] = object_indices[i]
    depth[~np.isfinite(depth)] = np.nan
    return rgb, depth.astype(np.float32), object_map


def view_definitions(front_direction, known_front=True):
    """从已知正面起，每90°一张立面图；中间再插入四张带俯角的斜视图。"""
    front = np.asarray(front_direction, dtype=float).copy()
    front[2] = 0
    front /= np.linalg.norm(front)
    # 正对正面时，画面中的右边就是该建筑的“右侧”。不是地图的正东方。
    right = np.cross([0., 0., 1.], front)
    names = [("front", "正面"), ("right", "右侧"), ("back", "背面"), ("left", "左侧"),
             ("front_right", "正面 · 右侧"), ("back_right", "背面 · 右侧"),
             ("back_left", "背面 · 左侧"), ("front_left", "正面 · 左侧")]
    for index, angle in enumerate([0, 90, 180, 270, 45, 135, 225, 315]):
        elev = 0. if index < 4 else OBLIQUE_ELEVATION_DEG
        a, e = np.deg2rad([angle, elev])
        direction = (np.cos(a) * front + np.sin(a) * right) * np.cos(e)
        direction[2] = np.sin(e)
        name, label = names[index] if known_front else ("view_%03d" % angle, "相对起点 %d°" % angle)
        yield {"name": name, "label": label, "orbit_deg": angle, "elevation_deg": elev, "eye_direction": direction}


def make_contact_sheet(output, views, stats, building_id):
    """汇报用总览单独制作；用于匹配的单张 PNG 不叠文字、坐标轴或边框。"""
    canvas = Image.new("RGB", (2200, 1830), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    def font(size):
        return ImageFont.truetype(str(font_path), size)
    ink, muted, line = "#3F4845", "#76817C", "#D3D5CD"
    draw.text((76, 42), str(building_id) + "  /  建筑多视角渲染", font=font(42), fill=ink)
    draw.text((78, 105), "已有 CityGML 几何 · 正交相机 · 模型颜色与默认材质 · 未使用照片贴图", font=font(23), fill=muted)
    draw.line((76, 156, 2124, 156), fill=line, width=2)
    for i, view in enumerate(views):
        row, col = divmod(i, 2)
        x, y = 76 + col * 1060, 182 + row * 382
        draw.text((x, y), "%02d  %s" % (i + 1, view["label"]), font=font(25), fill=ink)
        draw.text((x + 545, y + 5), "方位 %d°  /  俯角 %d°" % (view["orbit_deg"], view["elevation_deg"]), font=font(18), fill=muted)
        image = Image.open(output / view["image"]).convert("RGB")
        # 仅总览压缩留白；原始PNG保持统一相机画幅。
        arr = np.asarray(image)
        mask = np.any(arr != np.asarray(BACKGROUND), axis=2)
        ys, xs = np.where(mask)
        if len(xs):
            image = image.crop((max(0, xs.min() - 20), max(0, ys.min() - 20),
                                min(image.width, xs.max() + 21), min(image.height, ys.max() + 21)))
        image.thumbnail((990, 305), Image.Resampling.LANCZOS)
        canvas.paste(image, (x + (990 - image.width) // 2, y + 44 + (305 - image.height) // 2))
        draw.line((x, y + 365, x + 988, y + 365), fill=line, width=1)
    draw.text((78, 1735), "门窗来自已有模型；每个视角保留相机、深度及构件编号，用于后续照片检索与匹配。", font=font(22), fill=ink)
    draw.text((78, 1777), "%d 个视角  ·  %s 个三角形  ·  渲染图只展示模型几何，不表示照片匹配已通过。"
              % (len(views), format(stats.get("triangles", 0), ",")), font=font(20), fill=muted)
    canvas.save(output / "overview.png")


def _file_sha256(path):
    """逐块记录文件指纹，避免给较大的 CityGML 额外分配整文件内存。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reference_wall(walls, front_wall_id):
    """选择有ID、非退化且接近竖直的墙多边形，不改变或删除模型中的面。

    自动模式只确定环绕相机的起点，不意味着识别出了照片对应的“正面”。
    有些 WallSurface 内含多个多边形，比较的是各有效墙多边形的真实面积。
    """
    candidates = []
    vertical_limit = np.sin(np.deg2rad(MAX_REFERENCE_WALL_TILT_DEG))
    for wall in walls:
        normal = np.asarray(wall["normal"], dtype=float)
        norm = np.linalg.norm(normal)
        if (wall.get("id") is None or normal.shape != (3,) or
                not np.isfinite(normal).all() or norm < 1e-12 or
                not np.isfinite(wall["area"]) or wall["area"] <= 1e-10):
            continue
        if abs(normal[2]) / norm > vertical_limit:
            continue
        if front_wall_id is None or wall["id"] == front_wall_id:
            candidates.append(wall)
    if not candidates:
        if front_wall_id is not None:
            raise ValueError("指定的正面墙ID不存在或没有有效竖直墙面；换建筑时可将 front_wall_id 设为 None。")
        raise ValueError("模型中没有带ID的有效竖直墙，无法自动确定环绕渲染起点。")
    return max(candidates, key=lambda wall: wall["area"])


def _wall_metadata(walls):
    """同一 wall_id 可能含多个平面，逐面保存法向，不把非共面的墙强合成一个面。"""
    grouped = {}
    for wall in walls:
        key = wall["id"]
        item = grouped.setdefault(key, {"wall_id": key, "area_m2": 0., "polygons": []})
        item["area_m2"] += wall["area"]
        item["polygons"].append({"normal": np.asarray(wall["normal"]).tolist(), "area_m2": wall["area"]})
    return list(grouped.values())


def render_building(gml_path, output_root, building_id=None, front_wall_id=None,
                    image_size=(1800, 1200)) -> Path:
    """渲染一栋建筑的八视图，返回完成目录；可由批量入口逐栋调用。

    输出：output_root/建筑ID/render_时间/。逐视图落盘，只有八图和总览全部成功
    后才更新该建筑的 latest_run.json。失败会记录 manifest 并原样抛出异常，
    批量入口可继续其他建筑，不会把不完整批次标为完成。

    front_wall_id=None：从最大有效竖直墙起按0/90/180/270及四个斜向渲染，
    名称为 view_000 等；不读取人工标定，也不推断真实照片对应哪栋建筑。
    """
    start = time.perf_counter()
    gml_path, output_root = Path(gml_path).resolve(), Path(output_root).resolve()
    if not gml_path.is_file():
        raise FileNotFoundError("找不到 CityGML 文件：%s" % gml_path)
    if building_id is None:
        building_id = gml_path.stem
        for prefix in ("DEBY_LOD3_", "DEBY_LOD2_"):
            if building_id.startswith(prefix):
                building_id = building_id[len(prefix):]
                break
    building_id = str(building_id)
    if (not building_id or building_id in (".", "..") or
            any(char in building_id for char in '/\\<>:"|?*') or
            building_id != building_id.rstrip(". ")):
        raise ValueError("building_id 必须是单个有效文件夹名，不能包含路径或特殊字符。")
    size = np.asarray(image_size, dtype=float)
    if (size.shape != (2,) or not np.isfinite(size).all() or
            (size < 2).any() or not np.equal(size, np.floor(size)).all()):
        raise ValueError("image_size 必须是两个至少为2的整数：[宽, 高]。")
    image_size = tuple(int(value) for value in size)
    output = output_root / building_id / datetime.now().strftime("render_%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    light = np.array([-.35, -.45, .82]); light /= np.linalg.norm(light)
    manifest = {"schema_version": 2, "status": "running", "stage": "loading_model",
                "building_id": building_id, "source_gml": str(gml_path), "views": [],
                "parameters": {"image_size_wh": list(image_size), "projection": "orthographic",
                               "margin_fraction": MARGIN_FRACTION,
                               "oblique_elevation_deg": OBLIQUE_ELEVATION_DEG,
                               "background_rgb": list(BACKGROUND), "light_direction": light.tolist(),
                               "requested_front_wall_id": front_wall_id,
                               "reference_wall_max_tilt_deg": MAX_REFERENCE_WALL_TILT_DEG,
                               "reference_selection": "largest_valid_vertical_wall_polygon" if front_wall_id is None else "specified_wall_id",
                               "use_model_diffuse_colors": USE_MODEL_COLORS,
                               "expected_view_count": 8}}
    try:
        write_json(output / "manifest.json", manifest)
        manifest["source_sha256"] = _file_sha256(gml_path)
        manifest["script_sha256"] = _file_sha256(__file__)
        print("[1] 读取已有三维建筑：", gml_path, flush=True)
        mesh = load_building(gml_path)
        print("    几何统计：", mesh["stats"], flush=True)
        print("    缺少 %d 个纹理文件；使用模型颜色和默认材质。" % len(mesh["missing_textures"]), flush=True)
        front_wall = _reference_wall(mesh["walls"], front_wall_id)
        manifest.update({"stage": "rendering_views", "building_gml_id": mesh["building_gml_id"],
                         "coordinate_system": mesh["srs"], "world_origin_m": mesh["origin"].tolist(),
                         "front_reference_wall_id": front_wall["id"],
                         "front_manually_identified": front_wall_id is not None,
                         "reference_wall_normal": np.asarray(front_wall["normal"]).tolist(),
                         "reference_wall_polygon_area_m2": front_wall["area"],
                         "geometry_source": "dataset existing CityGML; no predicted detections or facade photographs used",
                         "rendering": "CPU triangle z-buffer; flat double-sided shading; opaque materials; no shadows",
                         "texture_mapping_applied": False, "use_model_diffuse_colors": USE_MODEL_COLORS,
                         "transparency_applied": False, "missing_texture_files": mesh["missing_textures"],
                         "stats": mesh["stats"], "max_plane_deviation_m": mesh["max_plane_deviation_m"],
                         "objects": mesh["objects"], "walls": _wall_metadata(mesh["walls"]),
                         "wall_normal_orientation": "sign chosen relative to model bounding-box center"})
        write_json(output / "manifest.json", manifest)
        write_json(output / "skipped_degenerate_polygons.json", mesh["skipped_polygon_ids"])
        write_json(output / "render_geometry_repairs.json", mesh["topology_repairs"])
        print("[2] 逐张渲染并保存到：", output, flush=True)
        points = mesh["triangles"].reshape(-1, 3)
        objects_by_index = {obj["index"]: obj for obj in mesh["objects"]}
        for view in view_definitions(front_wall["normal"], front_wall_id is not None):
            tick = time.perf_counter()
            direction = view.pop("eye_direction")
            manifest["current_view"] = view["name"]
            write_json(output / "manifest.json", manifest)
            # 必须显式传入本次尺寸，不能依赖函数定义时捕获的 IMAGE_SIZE 默认值。
            camera = make_camera(points, direction, image_size=image_size, margin=MARGIN_FRACTION)
            dot = mesh["normals"] @ direction
            colors = np.where((dot >= 0)[:, None], mesh["colors_front"], mesh["colors_back"])
            brightness = .60 + .25 * np.abs(dot) + .15 * np.abs(mesh["normals"] @ light)
            colors = np.clip(colors * brightness[:, None], 0, 255).astype(np.uint8)
            rgb, depth, object_map = rasterize(mesh["triangles"], colors, mesh["object_indices"], camera)
            name = view["name"]
            Image.fromarray(rgb).save(output / (name + ".png"))
            # 渲染像素+轴向深度乘逆矩阵，再加 world_origin_m，恢复三维世界坐标。
            np.savez_compressed(output / (name + "_geometry.npz"), depth_m=depth, object_index=object_map)
            write_json(output / (name + "_camera.json"), camera)
            ids, counts = np.unique(object_map[object_map >= 0], return_counts=True)
            wall_counts = Counter()
            wall_object_indices = {}
            for index, count in zip(ids, counts):
                wall_id = objects_by_index[int(index)].get("wall_id")
                if wall_id is not None:
                    # 同一墙的本体、窗和门都归入这堵墙，但每个像素只计一次。
                    wall_counts[wall_id] += int(count)
                    wall_object_indices.setdefault(wall_id, []).append(int(index))
            view.update({"image": name + ".png", "camera": name + "_camera.json",
                         "geometry": name + "_geometry.npz", "visible_objects":
                         [{"object_index": int(i), "pixel_count": int(c)} for i, c in zip(ids, counts)],
                         "visible_walls": [{"wall_id": wall_id, "pixel_count": count,
                                            "object_indices": wall_object_indices[wall_id]}
                                           for wall_id, count in wall_counts.most_common()],
                         "foreground_pixel_count": int(counts.sum()),
                         "render_seconds": round(time.perf_counter() - tick, 3)})
            manifest["views"].append(view)
            write_json(output / "manifest.json", manifest)
            print("    已保存 %s.png（%.1f 秒）" % (name, view["render_seconds"]), flush=True)
        manifest["stage"] = "contact_sheet"
        manifest.pop("current_view", None)
        write_json(output / "manifest.json", manifest)
        make_contact_sheet(output, manifest["views"], mesh["stats"], building_id)
        manifest.update({"status": "complete", "stage": "complete",
                         "total_seconds": round(time.perf_counter() - start, 3)})
        write_json(output / "manifest.json", manifest)
        write_json(output.parent / "latest_run.json", {"run_dir": str(output)})
        print("[3] 完成。先打开 overview.png；单张 PNG 用于后续检索/匹配。", flush=True)
        return output
    except (Exception, KeyboardInterrupt) as exc:
        manifest.update({"status": "failed", "total_seconds": round(time.perf_counter() - start, 3),
                         "error": {"type": type(exc).__name__, "message": str(exc)}})
        try:
            write_json(output / "manifest.json", manifest)
        except OSError:
            pass  # 磁盘写入也失败时，保留并抛出原始异常，不覆盖真正失败原因。
        raise


def eligible_wall_facades(gml_path, mesh):
    """返回实验 E 的可映射立面，不把 CityGML 的小构件片段当成独立墙面。

    规则在所有建筑上一致：墙必须能通过 ``load_walls`` 的单平面检查、近似竖直，
    且其二维包围面积不少于 10 平方米及本楼最大候选墙面积的 12%。这个选择只看
    CityGML 几何，不读照片、检索或门窗检测结果。
    """
    # 延迟导入避免渲染模块与映射模块在测试时形成初始化环；两者共用同一平面定义。
    from map_facade_to_3d import load_walls

    mesh_normals = {}
    for item in mesh["walls"]:
        if item.get("id") is not None:
            mesh_normals.setdefault(item["id"], []).append(item)
    candidates = []
    tilt_limit = np.sin(np.deg2rad(WALL_VIEW_MAX_TILT_DEG))
    for wall in load_walls(gml_path):
        normal = np.cross(wall["frame"]["u"], wall["frame"]["v"])
        normal /= np.linalg.norm(normal)
        # map_facade_to_3d 的 SVD 法向没有正反；沿用渲染器按模型中心确定的外侧符号。
        references = mesh_normals.get(wall["id"], [])
        if not references:
            continue
        reference = max(references, key=lambda item: item["area"])
        if np.dot(normal, reference["normal"]) < 0:
            normal = -normal
        if abs(normal[2]) > tilt_limit:
            continue
        area = float(wall["area_rank"])
        if not np.isfinite(area) or area <= 0:
            continue
        candidates.append(dict(wall, outward_normal=normal, facade_area_m2=area))
    if not candidates:
        raise ValueError("模型没有可用于实验 E 的竖直单平面外立面。")
    minimum = max(WALL_VIEW_MIN_AREA_M2,
                  max(item["facade_area_m2"] for item in candidates) * WALL_VIEW_RELATIVE_AREA)
    selected = [item for item in candidates if item["facade_area_m2"] >= minimum]
    if not selected:
        raise ValueError("大型立面筛选后为空，不能建立墙面视图库。")
    return sorted(selected, key=lambda item: (-item["facade_area_m2"], item["id"]))


def _wall_view_definitions(wall):
    """每个目标墙生成左近正视、正视和右近正视三个正交相机方向。"""
    normal = np.asarray(wall["outward_normal"], dtype=float).copy()
    normal[2] = 0.
    normal /= np.linalg.norm(normal)
    for yaw in WALL_VIEW_YAWS_DEG:
        radians = np.deg2rad(yaw)
        direction = np.array([np.cos(radians) * normal[0] - np.sin(radians) * normal[1],
                              np.sin(radians) * normal[0] + np.cos(radians) * normal[1], 0.])
        suffix = "m%02d" % abs(int(yaw)) if yaw < 0 else ("p%02d" % int(yaw) if yaw > 0 else "0")
        yield {"name_suffix": suffix, "yaw_deg": float(yaw), "eye_direction": direction}


def make_wall_contact_sheet(output, views, building_id):
    """墙面图库的汇报缩略图；匹配仍只读取无文字覆盖的单张渲染图。"""
    columns, card_w, card_h, pad = 3, 560, 340, 38
    rows = max(1, int(np.ceil(len(views) / columns)))
    canvas = Image.new("RGB", (columns * card_w + pad * 2, rows * card_h + 180), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    font = lambda size: ImageFont.truetype(str(font_path), size)
    draw.text((pad, 28), f"{building_id}  /  立面正视与近正视图库", font=font(36), fill="#3F4845")
    draw.text((pad, 82), "每个可映射大型竖直墙面 3 视角；仅使用已有 CityGML 几何", font=font(20), fill="#76817C")
    for index, view in enumerate(views):
        row, col = divmod(index, columns)
        x, y = pad + col * card_w, 130 + row * card_h
        draw.text((x, y), f"{view['target_wall_index']:02d} · {view['yaw_deg']:+.0f}°", font=font(20), fill="#3F4845")
        image = Image.open(output / view["image"]).convert("RGB")
        image.thumbnail((card_w - 28, card_h - 62), Image.Resampling.LANCZOS)
        canvas.paste(image, (x + (card_w-image.width)//2, y + 36 + (card_h-62-image.height)//2))
    canvas.save(output / "overview.png")


def render_wall_views(gml_path, output_root, building_id=None, image_size=(1800, 1200)) -> Path:
    """渲染实验 E 的墙面 3 视角图库，保存与八视角相同的几何回投数据。

    每个入选 CityGML 外立面独立执行 -15°、0°、+15° 三个视角。渲染整个建筑而
    相机按目标墙取景，故遮挡仍由完整三维 Z-buffer 计算；不会修改 CityGML。
    """
    start = time.perf_counter()
    gml_path, output_root = Path(gml_path).resolve(), Path(output_root).resolve()
    if not gml_path.is_file():
        raise FileNotFoundError("找不到 CityGML 文件：%s" % gml_path)
    if building_id is None:
        building_id = gml_path.stem.removeprefix("DEBY_LOD3_").removeprefix("DEBY_LOD2_")
    image_size = tuple(int(value) for value in image_size)
    if len(image_size) != 2 or min(image_size) < 2:
        raise ValueError("image_size 必须是两个至少为2的整数：[宽, 高]。")
    output = output_root / str(building_id) / datetime.now().strftime("wall_views_%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    light = np.array([-.35, -.45, .82]); light /= np.linalg.norm(light)
    manifest = {"schema_version": 1, "status": "running", "stage": "loading_model", "building_id": str(building_id),
                "source_gml": str(gml_path), "views": [], "parameters": {
                    "image_size_wh": list(image_size), "projection": "orthographic", "margin_fraction": MARGIN_FRACTION,
                    "wall_view_yaws_deg": list(WALL_VIEW_YAWS_DEG), "wall_view_max_tilt_deg": WALL_VIEW_MAX_TILT_DEG,
                    "wall_view_min_area_m2": WALL_VIEW_MIN_AREA_M2, "wall_view_relative_area": WALL_VIEW_RELATIVE_AREA,
                    "selection": "valid planar vertical WallSurface above fixed absolute and relative area thresholds"}}
    try:
        write_json(output / "manifest.json", manifest)
        manifest["source_sha256"] = _file_sha256(gml_path)
        manifest["script_sha256"] = _file_sha256(__file__)
        mesh = load_building(gml_path)
        walls = eligible_wall_facades(gml_path, mesh)
        manifest.update({"stage": "rendering_wall_views", "building_gml_id": mesh["building_gml_id"],
                         "coordinate_system": mesh["srs"], "world_origin_m": mesh["origin"].tolist(),
                         "geometry_source": "dataset existing CityGML; no photographs, detections, or labels used",
                         "objects": mesh["objects"], "walls": _wall_metadata(mesh["walls"]),
                         "eligible_wall_count": len(walls), "stats": mesh["stats"],
                         "max_plane_deviation_m": mesh["max_plane_deviation_m"],
                         "wall_selection_note": "Large vertical planar facade surfaces only; architectural fragments are excluded by area rule."})
        write_json(output / "manifest.json", manifest)
        objects_by_index = {obj["index"]: obj for obj in mesh["objects"]}
        for wall_index, wall in enumerate(walls, 1):
            wall_points = np.concatenate(wall["rings"]) - mesh["origin"]
            for definition in _wall_view_definitions(wall):
                tick = time.perf_counter()
                name = f"wall_{wall_index:03d}_{definition['name_suffix']}"
                manifest["current_view"] = name
                write_json(output / "manifest.json", manifest)
                direction = definition["eye_direction"]
                camera = make_camera(wall_points, direction, image_size=image_size, margin=MARGIN_FRACTION)
                dot = mesh["normals"] @ direction
                colors = np.where((dot >= 0)[:, None], mesh["colors_front"], mesh["colors_back"])
                brightness = .60 + .25 * np.abs(dot) + .15 * np.abs(mesh["normals"] @ light)
                colors = np.clip(colors * brightness[:, None], 0, 255).astype(np.uint8)
                rgb, depth, object_map = rasterize(mesh["triangles"], colors, mesh["object_indices"], camera)
                Image.fromarray(rgb).save(output / (name + ".png"))
                np.savez_compressed(output / (name + "_geometry.npz"), depth_m=depth, object_index=object_map)
                write_json(output / (name + "_camera.json"), camera)
                ids, counts = np.unique(object_map[object_map >= 0], return_counts=True)
                wall_counts, wall_object_indices = Counter(), {}
                for object_index, count in zip(ids, counts):
                    wall_id = objects_by_index[int(object_index)].get("wall_id")
                    if wall_id is not None:
                        wall_counts[wall_id] += int(count)
                        wall_object_indices.setdefault(wall_id, []).append(int(object_index))
                view = {"name": name, "label": f"墙 {wall_index:02d} / {definition['yaw_deg']:+.0f}°", "orbit_deg": definition["yaw_deg"],
                        "elevation_deg": 0., "view_kind": "wall_front_or_near_front", "target_wall_id": wall["id"],
                        "target_wall_index": wall_index, "target_wall_area_m2": wall["facade_area_m2"], "yaw_deg": definition["yaw_deg"],
                        "image": name + ".png", "camera": name + "_camera.json", "geometry": name + "_geometry.npz",
                        "visible_objects": [{"object_index": int(i), "pixel_count": int(c)} for i, c in zip(ids, counts)],
                        "visible_walls": [{"wall_id": wall_id, "pixel_count": count, "object_indices": wall_object_indices[wall_id]}
                                          for wall_id, count in wall_counts.most_common()],
                        "foreground_pixel_count": int(counts.sum()), "render_seconds": round(time.perf_counter() - tick, 3)}
                manifest["views"].append(view)
                write_json(output / "manifest.json", manifest)
        manifest.pop("current_view", None)
        make_wall_contact_sheet(output, manifest["views"], building_id)
        manifest.update({"status": "complete", "stage": "complete", "total_seconds": round(time.perf_counter() - start, 3)})
        write_json(output / "manifest.json", manifest)
        write_json(output.parent / "latest_wall_views.json", {"run_dir": str(output)})
        return output
    except (Exception, KeyboardInterrupt) as exc:
        manifest.update({"status": "failed", "total_seconds": round(time.perf_counter() - start, 3),
                         "error": {"type": type(exc).__name__, "message": str(exc)}})
        try:
            write_json(output / "manifest.json", manifest)
        except OSError:
            pass
        raise


def main():
    render_building(GML_PATH, OUTPUT_ROOT, building_id=BUILDING_ID,
                    front_wall_id=FRONT_WALL_ID, image_size=IMAGE_SIZE)


if __name__ == "__main__":
    main()
