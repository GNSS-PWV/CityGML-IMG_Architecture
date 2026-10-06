"""自动匹配之后，把预测门窗框投到选中建筑的一个平面墙上。

输入只有检测结果、照片→渲染图单应矩阵、相机/深度及建筑模型；本模块不读取
人工标定、gt_masks，也不使用照片文件名或检测文件里的建筑编号决定建筑身份。
复用 map_facade_to_3d 的墙面读取/平面坐标数学；不调用其交互标定或 main。

一个 H 只能用于一个平面。调用方先完成建筑/视角检索与匹配质量检查，再将同一
候选的 H、相机、几何和 wall_id 一起传入。这里的输出是独立预测面预览，既没有
对原建筑开孔，也没有写入源 GML，更不代表已验证真实世界的定位精度。
"""
from collections import Counter, defaultdict
from pathlib import Path
import json

import numpy as np

from facade_match_geometry import project_homography
from map_facade_to_3d import file_hash, load_walls, polygon_inside_ring


POLICY = {"wall_boundary_tolerance_m": .03, "min_polygon_area_m2": 1e-6,
          "ray_plane_min_abs_cosine": 1e-4, "reject_photo_border_boxes": True,
          "visibility_sample_grid": 5, "visibility_sample_inset": .10,
          "min_same_wall_sample_fraction": .80, "max_other_surface_sample_fraction": .05}
COLORS = {"window": "#568F98", "door": "#BA855F", "ambiguous": "#C6A653"}


def _geometry(source):
    if isinstance(source, (str, Path)):
        with np.load(source, allow_pickle=False) as data:
            depth, objects = data["depth_m"], data["object_index"]
    else:
        depth, objects = np.asarray(source["depth_m"]), np.asarray(source["object_index"])
    if depth.ndim != 2 or objects.shape != depth.shape:
        raise ValueError("深度和构件编号必须是同尺寸的二维数组。")
    if not np.issubdtype(objects.dtype, np.integer):
        raise ValueError("构件编号图必须使用整数类型。")
    mask = objects >= 0
    if not np.array_equal(np.isfinite(depth), mask) or np.any(depth[mask] <= 0):
        raise ValueError("深度与构件编号掩码不一致，或有效深度不是正数。")
    return depth, objects


def _object_samples(points, objects):
    """按像素中心约定查询可见构件；越界/NaN 返回背景编号 -1。"""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("像素坐标必须是 N×2 数组。")
    wh = np.asarray(objects.shape[::-1])
    valid = np.isfinite(points).all(axis=1) & (points >= -.5).all(axis=1) & (points < wh-.5).all(axis=1)
    result = np.full(len(points), -1, dtype=np.int64)
    pixels = np.floor(points[valid]+.5).astype(int)
    result[valid] = objects[pixels[:, 1], pixels[:, 0]]
    return result


def choose_wall_from_matches(render_xy, geometry_npz, manifest, weights=None,
                             min_votes=8, min_fraction=.60, min_visible_fraction=.50):
    """由匹配点可见构件的祖先墙投票；不把屋顶、背景或未知编号算成墙。

    若照片明显包含多墙且没有主墙，返回 accepted=False、wall_id=None，让调用方
    分墙匹配。该票数只确定几何归属，不能证明照片和建筑配对正确。
    """
    _, objects = _geometry(geometry_npz)
    ids = _object_samples(render_xy, objects)
    weights = np.ones(len(ids)) if weights is None else np.asarray(weights, dtype=float)
    if weights.shape != (len(ids),) or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("投票权重必须与点数一致，且为有限非负数。")
    if min_votes < 1 or not (0 < min_fraction <= 1) or not (0 < min_visible_fraction <= 1):
        raise ValueError("墙面投票门槛无效。")
    definitions = {int(obj["index"]): obj for obj in manifest["objects"]}
    counts, scores = Counter(), defaultdict(float)
    eligible = weights > 0
    for index, weight in zip(ids[eligible], weights[eligible]):
        obj = definitions.get(int(index))
        if obj and obj.get("kind") in ("WallSurface", "Window", "Door") and obj.get("wall_id") is not None:
            wall_id = obj["wall_id"]
            counts[wall_id] += 1
            scores[wall_id] += float(weight)
    total_score = sum(scores.values())
    visible_fraction = sum(counts.values())/max(1, int(eligible.sum()))
    votes = [{"wall_id": key, "count": counts[key], "weight": scores[key],
              "weight_fraction": scores[key]/total_score}
             for key in sorted(scores, key=lambda key: (-scores[key], -counts[key], str(key)))]
    accepted = bool(votes and votes[0]["count"] >= min_votes and
                    votes[0]["weight_fraction"] >= min_fraction and visible_fraction >= min_visible_fraction)
    return {"accepted": accepted, "wall_id": votes[0]["wall_id"] if accepted else None,
            "suggested_wall_id": votes[0]["wall_id"] if votes else None, "votes": votes,
            "point_count": len(ids), "positive_weight_count": int(eligible.sum()),
            "visible_wall_fraction": visible_fraction,
            "policy": {"min_votes": min_votes, "min_fraction": min_fraction,
                       "min_visible_fraction": min_visible_fraction},
            "reason": "dominant_visible_wall" if accepted else "no_dominant_visible_wall; split_matches_by_wall_or_reject",
            "independent_accuracy_verified": False}


def render_pixels_to_wall(render_xy, camera, frame, world_origin):
    """渲染像素发射正交相机射线，与主墙平面求交，返回世界XYZ、墙UV和轴向深度。

    直接拿窗框或玻璃的深度回投会落在凹凸表面；这里使用确定的主墙平面。
    在局部坐标中计算交点，最后才加回百万级地理坐标，降低数值误差。
    """
    points = np.asarray(render_xy, dtype=float)
    inverse = np.asarray(camera["pixel_depth_to_local"], dtype=float)
    origin = np.asarray(world_origin, dtype=float)
    if (points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all() or
            inverse.shape != (4, 4) or not np.isfinite(inverse).all() or
            origin.shape != (3,) or not np.isfinite(origin).all()):
        raise ValueError("射线回投的像素、相机矩阵或世界原点无效。")
    if camera.get("projection") != "orthographic" or not np.allclose(inverse[3], [0, 0, 0, 1], atol=1e-12):
        raise ValueError("本模块只支持图库保存的正交相机，不能把透视相机按此公式回投。")
    normal = np.cross(frame["u"], frame["v"])
    normal = normal/np.linalg.norm(normal)
    direction = inverse[:3, 2]
    direction_norm = np.linalg.norm(direction)
    denominator = float(direction@normal)
    if direction_norm < 1e-12 or abs(denominator)/direction_norm < POLICY["ray_plane_min_abs_cosine"]:
        raise ValueError("视线与目标墙近乎平行，不能稳定地映射到该墙。")
    p0 = np.column_stack((points, np.zeros(len(points)), np.ones(len(points))))@inverse.T
    wall_origin_local = np.asarray(frame["origin"])-origin
    depths = ((wall_origin_local-p0[:, :3])@normal)/denominator
    if not np.isfinite(depths).all() or np.any(depths <= 0):
        raise ValueError("墙面交点位于相机后方或无穷远。")
    local = p0[:, :3]+depths[:, None]*direction
    uv = (local-wall_origin_local)@np.column_stack((frame["u"], frame["v"]))
    return local+origin, uv, depths


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _wall_outer_rings(wall):
    """合并同墙共面的所有面片，只返回整体外边界，消除三角化/拼接的内部接缝。

    使用真实多边形并集，不取包围盒或凸包，所以墙的凹缺口仍被保留。
    并集形成的封闭内部孔不用于约束预测框：本模块使用墙外轮廓，而不是拿原模型
    门窗孔洞替换预测。多个不连通墙片仍分别保留，不能跨空隙连接成一整块。
    """
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    from shapely.errors import GEOSException

    polygons = [Polygon(ring) for ring in wall["uv"]]
    if not polygons or any(p.is_empty or not p.is_valid or p.area <= 0 for p in polygons):
        raise ValueError("墙面外环不是有效多边形，不能静默修补后用于三维映射。")
    try:
        merged = unary_union(polygons)
    except GEOSException as error:
        raise ValueError("墙面外环无法合并成合法边界，拒绝用于三维映射。") from error
    if merged.geom_type == "Polygon":
        components = [merged]
    elif merged.geom_type == "MultiPolygon":
        components = list(merged.geoms)
    else:
        raise ValueError("墙面外环的并集不是面几何，无法确定整体墙边界。")
    return [np.asarray(p.exterior.coords, dtype=float)[:-1]
            for p in sorted(components, key=lambda p: (-p.area, p.bounds))]


def validated_wall(manifest, wall_id):
    """取得唯一、可用一个H表达的合法墙；无效候选抛ValueError供上游明确拒绝。

    候选预检、跨视角回投和最终映射共用同一检查，不把能渲染的非平面WallSurface
    自动视为可标定平面。不修补源模型、不放宽原0.03米平面门槛。
    返回原frame/uv/rings，并增加合并后的outer_uv；不读取人工点或照片身份。
    """
    source = Path(manifest["source_gml"])
    if file_hash(source) != manifest["source_sha256"]:
        raise ValueError("源模型已变化，不能复用旧渲染图建立三维位置。")
    try:
        walls = load_walls(source)
    except ValueError as error:
        raise ValueError("目标墙不存在、不是单一平面或退化，不能用于自动映射："+str(wall_id)) from error
    selected = [wall for wall in walls if wall["id"] == wall_id]
    if wall_id is None or len(selected) != 1:
        raise ValueError("目标墙不存在、不是单一平面或不唯一，不能用于自动映射："+str(wall_id))
    wall = dict(selected[0])
    try:
        wall["outer_uv"] = _wall_outer_rings(wall)
    except ValueError as error:
        raise ValueError("目标墙边界无效，拒绝自动映射："+str(wall_id)+"；"+str(error)) from error
    return wall


def _write_obj(path, predictions, origin):
    """只写预测四边形，使用局部米制坐标；世界原点保存在注释与JSON中。"""
    material = path.with_suffix(".mtl")
    material_lines = []
    for kind, color in COLORS.items():
        rgb = [int(color[index:index+2], 16)/255 for index in (1, 3, 5)]
        material_lines.extend(["newmtl "+kind, "Kd "+" ".join("%.6f" % value for value in rgb), "d 0.85", ""])
    material.write_text("\n".join(material_lines), encoding="utf8")
    lines = ["# PREDICTIONS ONLY: no wall cuts; source building not modified.",
             "# Local meters; world XYZ = OBJ xyz + origin below.",
             "# origin_world_m " + " ".join("%.9f" % value for value in origin), "mtllib "+material.name]
    for index, item in enumerate(predictions):
        lines.extend(["g "+item["prediction_id"]+"_"+item["class"], "usemtl "+item["class"]])
        for point in np.asarray(item["vertices_xyz"])-origin:
            lines.append("v "+" ".join("%.9f" % value for value in point))
        lines.append("f "+" ".join(str(index*4+corner) for corner in (1, 2, 3, 4)))
    path.write_text("\n".join(lines)+"\n", encoding="utf8")


def _save_preview(path, wall, walls, predictions, origin, rejected_count):
    """离线科研绘图：左侧墙UV，右侧建筑墙轮廓与预测面；不打开交互窗口。"""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.font_manager import FontProperties
    from matplotlib.lines import Line2D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    font = FontProperties(fname="C:/Windows/Fonts/msyh.ttc")
    figure = Figure(figsize=(16, 7), dpi=150, facecolor="#F6F5F1")
    FigureCanvasAgg(figure)
    flat = figure.add_subplot(121)
    space = figure.add_subplot(122, projection="3d")
    for axis in (flat, space):
        axis.set_facecolor("#F6F5F1")
    for ring in _wall_outer_rings(wall):
        ring = np.asarray(ring)
        ring = np.vstack((ring, ring[0]))
        flat.plot(ring[:, 0], ring[:, 1], color="#5D6963", linewidth=1.)
    context, warnings = [], []
    for current in walls:
        try:
            outlines = _wall_outer_rings(current)
            frame = current["frame"]
            # 绘制与边界检查相同的整体外环，避免预览留下已合并的内部对角线。
            world_rings = [ring@np.asarray([frame["u"], frame["v"]])+np.asarray(frame["origin"])
                           for ring in outlines]
        except ValueError as error:
            if current["id"] == wall["id"]:
                raise  # 选中墙无效必须拒绝，不能靠背景绘图的容错绕过几何检查。
            warnings.append({"wall_id": current["id"], "reason": str(error), "action": "raw_3d_outline_only"})
            # 其他墙只提供空间背景，保留原始有限折线并明确记录，不修补、不填面。
            world_rings = [ring for ring in current["rings"] if np.isfinite(ring).all()]
        for ring in world_rings:
            points = np.asarray(ring)-origin
            context.append(points)
            closed = np.vstack((points, points[0]))
            space.plot(closed[:, 0], closed[:, 1], closed[:, 2], color="#9DA49E", linewidth=.5, alpha=.65)
    for item in predictions:
        uv = np.asarray(item["wall_uv_m"])
        xy = np.vstack((uv, uv[0]))
        flat.fill(uv[:, 0], uv[:, 1], color=COLORS[item["class"]], alpha=.15)
        flat.plot(xy[:, 0], xy[:, 1], color=COLORS[item["class"]], linewidth=1.1)
        xyz = np.asarray(item["vertices_xyz"])-origin
        space.add_collection3d(Poly3DCollection([xyz], facecolor=COLORS[item["class"]],
                                               edgecolor=COLORS[item["class"]], linewidth=.6, alpha=.75))
    all_points = np.concatenate(context if context else [ring-origin for ring in wall["rings"]])
    lo, hi = all_points.min(axis=0), all_points.max(axis=0)
    for setter, lower, upper in zip((space.set_xlim, space.set_ylim, space.set_zlim), lo, hi):
        setter(lower-.5, upper+.5)
    space.set_box_aspect(np.maximum(hi-lo, 1.))
    # 从选中墙外侧斜看，尽量避免默认视角将预测面显示成一条线。
    outward = np.cross(wall["frame"]["u"], wall["frame"]["v"])
    if outward@(wall["frame"]["origin"]-origin) < 0:
        outward = -outward
    space.view_init(elev=16, azim=float(np.degrees(np.arctan2(outward[1], outward[0])))+15)
    flat.set_aspect("equal", adjustable="box")
    flat.set_xlabel("Wall U (m)"); flat.set_ylabel("Wall V (m)")
    space.set_xlabel("X - origin (m)"); space.set_ylabel("Y - origin (m)"); space.set_zlabel("Z - origin (m)")
    flat.set_title("预测门窗在选中墙面上的位置", fontproperties=font, fontsize=14)
    space.set_title("独立预测面预览：未开孔、未修改建筑", fontproperties=font, fontsize=14)
    labels = [("window", "窗"), ("door", "门"), ("ambiguous", "门/窗待确认")]
    flat.legend([Line2D([], [], color=COLORS[kind], lw=2) for kind, _ in labels],
                [label for _, label in labels], prop=font, loc="best", frameon=False)
    figure.suptitle("自动映射预览  /  保留 %d 个，拒绝 %d 个；真实位置精度取决于上游匹配"
                    % (len(predictions), rejected_count), fontproperties=font, fontsize=16, color="#414C47")
    figure.tight_layout(rect=[.01, .02, .99, .92])
    figure.savefig(path, facecolor=figure.get_facecolor())
    figure.clear()
    return warnings


def map_detections(detections, H_photo_to_render, camera, geometry_npz, manifest, wall_id, output_dir):
    """自动投影并保存 predictions.json、OBJ/MTL、mapping_preview.png，返回报告字典。

    detections 是现有检测JSON字典，必须包含 image_size 和 detections；建筑身份
    只取上游选中的 manifest。geometry_npz 接受路径或 {depth_m, object_index}。
    此函数不判断上游照片是否检索到正确建筑：调用方必须先检查匹配质量。
    """
    H = np.asarray(H_photo_to_render, dtype=float)
    if H.shape != (3, 3) or not np.isfinite(H).all() or np.linalg.matrix_rank(H) < 3:
        raise ValueError("照片到渲染图的单应矩阵必须是有限、非奇异的3×3矩阵。")
    H = H/np.linalg.norm(H)
    photo_wh = np.asarray(detections.get("image_size"), dtype=float)
    if photo_wh.shape != (2,) or not np.isfinite(photo_wh).all() or np.any(photo_wh <= 0):
        raise ValueError("检测结果必须含原照片的有效 image_size=[宽,高]。")
    if detections.get("coordinate_system", "original_image_pixels_xyxy") != "original_image_pixels_xyxy":
        raise ValueError("检测框必须使用原始照片的 xyxy 坐标。")
    depth, objects = _geometry(geometry_npz)
    if list(camera["image_size_wh"]) != list(depth.shape[::-1]):
        raise ValueError("相机和深度图尺寸不一致。")
    inverse = np.asarray(camera["pixel_depth_to_local"], dtype=float)
    forward = np.asarray(camera["local_to_pixel_depth"], dtype=float)
    if (inverse.shape != (4, 4) or forward.shape != (4, 4) or
            not np.isfinite(inverse).all() or not np.isfinite(forward).all() or
            not np.allclose(inverse@forward, np.eye(4), atol=1e-8, rtol=1e-8)):
        raise ValueError("相机正反变换不一致。")
    source = Path(manifest["source_gml"])
    wall = validated_wall(manifest, wall_id)
    geometry_provenance = {"in_memory": not isinstance(geometry_npz, (str, Path))}
    if isinstance(geometry_npz, (str, Path)):
        geometry_path = Path(geometry_npz).resolve()
        views = [view for view in manifest.get("views", []) if view["geometry"] == geometry_path.name]
        if len(views) != 1:
            raise ValueError("无法在候选建筑 manifest 中唯一确定该深度文件所属视图。")
        saved_camera = json.loads((geometry_path.parent/views[0]["camera"]).read_text(encoding="utf8"))
        if saved_camera != camera:
            raise ValueError("传入的相机与该深度文件对应的视图相机不同。")
        geometry_provenance.update({"path": str(geometry_path), "sha256": file_hash(geometry_path), "view_name": views[0]["name"]})
    definitions = {int(obj["index"]): obj for obj in manifest["objects"]}
    allowed = [index for index, obj in definitions.items() if obj.get("wall_id") == wall_id
               and obj.get("kind") in ("WallSurface", "Window", "Door")]
    if wall_id is None or not allowed or not np.isin(objects, allowed).any():
        raise ValueError("目标墙及其门窗在此渲染视图中不可见，拒绝猜测墙面。")
    walls = load_walls(source)
    outer_rings = wall["outer_uv"]
    origin = np.asarray(manifest["world_origin_m"], dtype=float)
    # 在处理框以前检查射线/平面是否可解，避免无效视角生成一批貌似正常的结果。
    render_pixels_to_wall(np.asarray(camera["image_size_wh"])[None, :]/2, camera, wall["frame"], origin)
    mapped, rejected = [], []
    for index, detection in enumerate(detections.get("detections", []), 1):
        record = {"prediction_id": "P%04d" % index, "class": detection.get("class"),
                  "detection": _json_safe(detection), "wall_id": wall_id}
        def reject(reason):
            record["reason"] = reason
            rejected.append(record)
        if record["class"] not in COLORS:
            reject("unsupported_detection_class")
            continue
        try:
            box = np.asarray(detection.get("box"), dtype=float)
        except (ValueError, TypeError):
            box = np.empty(0)
        if box.shape != (4,) or not np.isfinite(box).all() or np.any(box[2:] <= box[:2]):
            reject("invalid_detection_box")
            continue
        record["box_image_xyxy"] = box.tolist()
        if np.any(box[:2] < 0) or np.any(box[2:] > photo_wh):
            reject("box_outside_photo")
            continue
        if POLICY["reject_photo_border_boxes"] and (np.any(box[:2] <= 1e-6) or np.any(box[2:] >= photo_wh-1e-6)):
            reject("box_touches_photo_border_may_be_incomplete")
            continue
        x0, y0, x1, y1 = box
        corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        denominator = np.column_stack((corners, np.ones(4)))@H[2]
        if not ((denominator > 1e-12).all() or (denominator < -1e-12).all()):
            reject("projective_pole_crosses_detection")
            continue
        render = project_homography(corners, H)
        render_wh = np.asarray(camera["image_size_wh"])
        if (not np.isfinite(render).all() or np.any(render < -.5) or np.any(render >= render_wh-.5)):
            reject("mapped_box_outside_render_image")
            continue
        try:
            xyz, uv, axis_depth = render_pixels_to_wall(render, camera, wall["frame"], origin)
        except ValueError as error:
            reject("invalid_ray_plane_intersection: "+str(error))
            continue
        area = abs(float(np.sum(uv[:, 0]*np.roll(uv[:, 1], -1)-uv[:, 1]*np.roll(uv[:, 0], -1))))/2
        if area < POLICY["min_polygon_area_m2"]:
            reject("mapped_polygon_is_degenerate")
            continue
        # 只使用墙外轮廓，不用原有 Window/Door 的轮廓替换检测框。
        if not any(polygon_inside_ring(uv, ring, POLICY["wall_boundary_tolerance_m"]) for ring in outer_rings):
            reject("mapped_box_not_fully_inside_wall_outer_boundary")
            continue
        inset = POLICY["visibility_sample_inset"]
        fx, fy = np.meshgrid(np.linspace(inset, 1-inset, POLICY["visibility_sample_grid"]),
                             np.linspace(inset, 1-inset, POLICY["visibility_sample_grid"]))
        samples = np.column_stack((x0+fx.ravel()*(x1-x0), y0+fy.ravel()*(y1-y0)))
        ids = _object_samples(project_homography(samples, H), objects)
        same_wall = np.isin(ids, allowed)
        other = (ids >= 0) & ~same_wall
        visibility = {"sample_count": len(ids), "same_wall_fraction": float(same_wall.mean()),
                      "other_surface_fraction": float(other.mean()), "background_fraction": float(np.mean(ids < 0))}
        record["visibility"] = visibility
        if (not same_wall[len(same_wall)//2] or same_wall.mean() < POLICY["min_same_wall_sample_fraction"] or
                other.mean() > POLICY["max_other_surface_sample_fraction"]):
            reject("mapped_box_not_visibly_supported_by_selected_wall")
            continue
        width_m = (np.linalg.norm(uv[1]-uv[0])+np.linalg.norm(uv[2]-uv[3]))/2
        height_m = (np.linalg.norm(uv[3]-uv[0])+np.linalg.norm(uv[2]-uv[1]))/2
        record.update({"geometry_role": "prediction_preview_only", "render_corners_xy": render.tolist(),
                       "wall_uv_m": uv.tolist(), "vertices_xyz": xyz.tolist(),
                       "width_m": float(width_m), "height_m": float(height_m), "area_m2": area,
                       "camera_axis_depth_m": axis_depth.tolist()})
        mapped.append(record)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": 1, "status": "preview_created" if mapped else "no_detections_mapped",
              "preview_only": True, "source_gml_modified": False, "wall_holes_created": False,
              "manual_points_used": False, "gt_masks_used": False, "independent_accuracy_verified": False,
              "upstream_match_quality_checked_here": False,
              "building_id": manifest.get("building_id"), "building_gml_id": manifest.get("building_gml_id"),
              "wall_id": wall_id, "coordinate_system": manifest.get("coordinate_system"),
              "source_gml": str(source), "source_sha256": manifest["source_sha256"],
              "image_sha256": detections.get("image_sha256"), "geometry_provenance": geometry_provenance,
              "method": "photo homography -> orthographic render ray -> selected wall plane",
              "photo_to_render_homography": H.tolist(), "wall_frame": _json_safe(wall["frame"]),
              "obj_origin_world_m": origin.tolist(), "obj_coordinates": "local meters; world=OBJ+origin",
              "policy": dict(POLICY), "input_count": len(detections.get("detections", [])),
              "wall_boundary": {"method": "union of coplanar outer rings; polygon exteriors only",
                                "source_outer_ring_count": len(wall["uv"]),
                                "merged_outer_ring_count": len(outer_rings),
                                "wall_uv_outer_rings_m": [ring.tolist() for ring in outer_rings]},
              "mapped_count": len(mapped), "rejected_count": len(rejected),
              "mapped_class_counts": dict(Counter(item["class"] for item in mapped)),
              "predictions": mapped, "rejected": rejected,
              "scope": "单个已选平面墙上的预测四边形预览；框不是精细拱形轮廓，类别待确认保持待确认。",
              "files": {"json": str(output/"predictions.json"), "obj": str(output/"predictions_preview.obj"),
                        "mtl": str(output/"predictions_preview.mtl"), "preview": str(output/"mapping_preview.png")}}
    _write_obj(output/"predictions_preview.obj", mapped, origin)
    report["preview_context_warnings"] = _save_preview(output/"mapping_preview.png", wall, walls, mapped,
                                                       origin, len(rejected)) or []
    (output/"predictions.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf8")
    return report
