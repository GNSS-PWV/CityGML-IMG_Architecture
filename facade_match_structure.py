"""用现有检测窗与模型窗的中心，自动细化 RoMa 的照片→渲染图变换。

输入只有检测结果、渲染构件编号图和 RoMa 初值。本模块不访问任何文件，不读取
人工标定或 GT 掩膜，不调用检测 API。调用入口负责核对图片和模型的身份。

通过检查只表示“自动结构一致”：重复窗列整体错移仍可能自洽，不能据此声称
已得到独立真值验证。本模块只修正一个已有粗对齐，不承担建筑/视角检索。
"""

import numpy as np

from facade_match_geometry import (_coverage, _error_summary, _inside, _size,
                                   project_homography)


# 这些门槛固定公开，与纯 RoMa 的密集点检查分别报告；不拿少量窗中心套用
# “至少40个训练内点”的密集点门槛，也不修改纯 RoMa 原来的检查结果。
STRUCTURE_POLICY = {
    "min_detection_score": .20,
    "photo_border_margin_px": 4.,
    "min_size_ratio": .55,
    "max_size_ratio": 1.8,
    "max_association_distance_render_px": 30.,
    "fit_ransac_render_px": 4.,
    "min_train_inliers": 12,
    "min_holdout_matches": 6,
    "min_train_inlier_fraction": .70,
    "min_holdout_inlier_fraction": .70,
    "min_photo_hull_fraction": .30,
    "min_photo_x_span": .65,
    "min_photo_y_span": .60,
    "max_initial_change_render_px": 30.,
    "max_initial_change_spacing_fraction": .45,
    "split": "8x4 photo cells; (column+row)%3==0 held out; no refit on holdout",
}

# 这是候选选择前的窗布局证据门槛，不重拟合 RoMa，也不把窗中心当作真值。
# 只有照片和模型墙面都至少可见 12 个窗时才要求它，避免少窗立面因证据本来不足
# 被误判为错误。若窗数充足却只能覆盖局部区域，重复窗列可能造成看似自洽的错配。
LAYOUT_POLICY = {
    "minimum_windows_to_require_layout": 12,
    "minimum_mutual_associations": 12,
    "minimum_photo_hull_fraction": .30,
    "minimum_photo_x_span_fraction": .65,
    "minimum_photo_y_span_fraction": .60,
}

# 门的检测通常很少，只有双方都至少有 3 个可见门时才检查门的相对位置。
# 证据不足时明确返回 required=False，不把“未检查”误写成门布局验证通过。
DOOR_LAYOUT_POLICY = {
    "minimum_doors_to_require_layout": 3,
    "minimum_mutual_associations": 2,
    "minimum_photo_association_fraction": .50,
    "maximum_center_distance_render_fraction": .06,
}


def assess_window_layout(report):
    """审计窗布局是否覆盖整张立面；返回可单独记录的证据而非准确率声明。"""
    photo_count = int(report.get("eligible_photo_windows", 0) or 0)
    model_count = int(report.get("visible_model_windows", 0) or 0)
    required = min(photo_count, model_count) >= LAYOUT_POLICY["minimum_windows_to_require_layout"]
    coverage = report.get("train_photo_coverage", {})
    checks = {
        "mutual_window_associations_at_least_12": int(report.get("association_count", 0) or 0) >= LAYOUT_POLICY["minimum_mutual_associations"],
        "window_layout_photo_hull_at_least_0_30": float(coverage.get("convex_hull_fraction", 0.) or 0.) >= LAYOUT_POLICY["minimum_photo_hull_fraction"],
        "window_layout_photo_x_span_at_least_0_65": float(coverage.get("x_span_fraction", 0.) or 0.) >= LAYOUT_POLICY["minimum_photo_x_span_fraction"],
        "window_layout_photo_y_span_at_least_0_60": float(coverage.get("y_span_fraction", 0.) or 0.) >= LAYOUT_POLICY["minimum_photo_y_span_fraction"],
    }
    return {"required": required, "passed": bool(all(checks.values())) if required else True,
            "policy": dict(LAYOUT_POLICY), "photo_window_count": photo_count, "model_window_count": model_count,
            "checks": checks, "failed_checks": [name for name, value in checks.items() if not value] if required else [],
            "interpretation": "仅在两侧窗数充足时要求匹配窗覆盖立面的横向和纵向范围；通过不等于独立真值验证。"}


def assess_door_layout(homography, photo_size, render_size, detector_json,
                       object_map, manifest, wall_id):
    """用投影后门中心与模型可见门中心的互为最近邻关系检查门布局。"""
    photo_wh, render_wh = _size(photo_size), _size(render_size)
    objects = np.asarray(object_map)
    photo_centers = []
    for detection in detector_json.get("detections", []):
        box = _box(detection.get("box"))
        try:
            score = float(detection.get("score", 0.))
        except (TypeError, ValueError):
            continue
        if (detection.get("class") != "door" or box is None or not np.isfinite(score)
                or score < STRUCTURE_POLICY["min_detection_score"]):
            continue
        if min(*box[:2], *(photo_wh-box[2:])) < STRUCTURE_POLICY["photo_border_margin_px"]:
            continue
        photo_centers.append((box[:2]+box[2:])/2)
    model_centers = []
    for obj in manifest.get("objects", []):
        if obj.get("kind") != "Door" or obj.get("wall_id") != wall_id:
            continue
        ys, xs = np.where(objects == obj["index"])
        if len(xs) == 0:
            continue
        model_centers.append(((xs.min()+xs.max())/2, (ys.min()+ys.max())/2))
    photo_count, model_count = len(photo_centers), len(model_centers)
    required = min(photo_count, model_count) >= DOOR_LAYOUT_POLICY["minimum_doors_to_require_layout"]
    report = {"required": required, "passed": True, "photo_door_count": photo_count,
              "model_door_count": model_count, "mutual_association_count": 0,
              "policy": dict(DOOR_LAYOUT_POLICY), "failed_checks": [],
              "interpretation": "仅在照片和目标墙各有至少 3 个有效门时检查；未触发不代表门布局通过验证。"}
    if not required:
        return report
    projected = project_homography(np.asarray(photo_centers), homography)
    model = np.asarray(model_centers)
    distances = np.linalg.norm(projected[:, None, :]-model[None, :, :], axis=2)
    limit = DOOR_LAYOUT_POLICY["maximum_center_distance_render_fraction"] * min(render_wh)
    pairs = [(i, j) for i, j in enumerate(distances.argmin(axis=1))
             if np.isfinite(distances[i, j]) and distances[i, j] <= limit and distances[:, j].argmin() == i]
    count = len(pairs)
    checks = {"mutual_door_associations_at_least_2": count >= DOOR_LAYOUT_POLICY["minimum_mutual_associations"],
              "photo_door_association_fraction_at_least_0_50":
              count/photo_count >= DOOR_LAYOUT_POLICY["minimum_photo_association_fraction"]}
    report.update({"passed": all(checks.values()), "mutual_association_count": count,
                   "maximum_center_distance_render_px": limit, "checks": checks,
                   "failed_checks": [name for name, value in checks.items() if not value]})
    return report


def _box(value):
    try:
        box = np.asarray(value, dtype=float)
    except (ValueError, TypeError):
        return None
    if box.shape != (4,) or not np.isfinite(box).all() or (box[2:] <= box[:2]).any():
        return None
    return box


def _pose_checks(H, initial_h, photo_wh, render_centers):
    """拒绝镜像/极点及过大的细化；窗间距自适应上限防止细化时跨窗列。"""
    corners = np.array([[-.5, -.5], [photo_wh[0]-.5, -.5],
                        [photo_wh[0]-.5, photo_wh[1]-.5], [-.5, photo_wh[1]-.5]])
    denominator = np.column_stack((corners, np.ones(4))) @ H[2]
    no_pole = bool((denominator > 1e-8).all() or (denominator < -1e-8).all())
    mapped = project_homography(corners, H)
    area = float(np.sum(mapped[:, 0]*np.roll(mapped[:, 1], -1)
                        - mapped[:, 1]*np.roll(mapped[:, 0], -1))/2)
    # 检查整个照片，不能只在少数拟合点附近检查后向外任意外推。
    xx, yy = np.meshgrid(np.linspace(-.5, photo_wh[0]-.5, 9),
                         np.linspace(-.5, photo_wh[1]-.5, 5))
    grid = np.column_stack((xx.ravel(), yy.ravel()))
    shifts = np.linalg.norm(project_homography(grid, H)-project_homography(grid, initial_h), axis=1)
    distances = np.linalg.norm(render_centers[:, None]-render_centers[None, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    nearest = distances.min(axis=1)
    spacing = float(np.median(nearest[np.isfinite(nearest)])) if np.isfinite(nearest).any() else 0.
    limit = min(STRUCTURE_POLICY["max_initial_change_render_px"],
                STRUCTURE_POLICY["max_initial_change_spacing_fraction"]*spacing)
    max_shift = float(shifts.max()) if np.isfinite(shifts).all() else None
    return {
        "no_projective_pole_in_photo": no_pole,
        "no_mirrored_or_collapsed_photo": bool(np.isfinite(area) and area > 1.),
        "refinement_stays_near_roma_initial": bool(max_shift is not None and max_shift <= limit),
    }, {"median_visible_window_spacing_px": spacing,
        "max_allowed_initial_change_px": limit, "max_initial_change_px": max_shift,
        "check_grid": "9x5 over full photo including corners"}


def refine_with_window_structure(initial_h, photo_size, render_size, detector_json,
                                 object_map, manifest, wall_id=None):
    """返回 (候选H或None, report, arrays)，调用方按 gate_passed 决定是否采用。

    先用初始 H 将检测窗投到渲染图，筛选尺寸接近、相距≤30像素的候选，再取
    互为最近邻的一对一关联。仅使用完整且分类为 window 的检测框；未确定门窗
    不参与拟合。渲染侧只使用指定墙上实际可见的 Window 构件。

    照片8×4空间格划分训练/留出点；留出点从不参与拟合/重拟合。求解失败返回
    明确 failed_checks 和已有点数组，正常但未过门槛仍返回候选 H 供诊断。
    arrays 各行对应一组关联中心，所有 mask/errors 与这组行号一致。
    """
    import cv2

    initial_h = np.asarray(initial_h, dtype=float)
    if initial_h.shape != (3, 3) or not np.isfinite(initial_h).all() or np.linalg.matrix_rank(initial_h) < 3:
        raise ValueError("RoMa 初始单应矩阵必须为非奇异、有限的3×3矩阵。")
    photo_wh, render_wh = _size(photo_size), _size(render_size)
    objects = np.asarray(object_map)
    if objects.ndim != 2 or objects.shape != tuple(render_wh[::-1].astype(int)):
        raise ValueError("构件编号图尺寸与完整渲染图尺寸不一致。")
    wall_id = wall_id if wall_id is not None else manifest.get("front_reference_wall_id")
    report = {
        "method": "RoMa-guided mutual-nearest window centers + spatial-holdout homography",
        "transform": "photo pixel centers -> full render pixel centers",
        "policy": dict(STRUCTURE_POLICY), "wall_id": wall_id,
        "gate_passed": False, "checks": {}, "failed_checks": [],
        "manual_points_used": False, "gt_masks_used": False,
        "independent_validation_passed": False,
        "interpretation": "自动几何一致性，不是独立真值验证；重复窗列错位仍可能通过，尤其当RoMa初值已错移。",
    }
    empty_xy = np.empty((0, 2), dtype=float)
    arrays = {"photo_xy": empty_xy.copy(), "render_xy": empty_xy.copy(),
              "detection_indices": np.empty(0, dtype=int), "model_object_indices": np.empty(0, dtype=int)}

    def fail(reason):
        report["checks"][reason] = False
        report["failed_checks"] = [key for key, passed in report["checks"].items() if not passed]
        return None, report, arrays

    if wall_id is None:
        return fail("reference_wall_id_available")
    model_centers, model_sizes, model_indices = [], [], []
    for obj in manifest.get("objects", []):
        if obj.get("kind") != "Window" or obj.get("wall_id") != wall_id:
            continue
        ys, xs = np.where(objects == obj["index"])
        if len(xs) == 0:
            continue
        box = _box([xs.min(), ys.min(), xs.max(), ys.max()])
        if box is None:
            continue
        model_centers.append((box[:2]+box[2:])/2)
        model_sizes.append(box[2:]-box[:2])
        model_indices.append(obj["index"])
    photo_centers, photo_sizes, detection_indices = [], [], []
    rejected = {"not_window_or_low_score": 0, "invalid_box_or_score": 0,
                "clipped_or_near_photo_border": 0, "invalid_initial_projection": 0}
    for index, detection in enumerate(detector_json.get("detections", [])):
        box = _box(detection.get("box"))
        try:
            score = float(detection.get("score", 0.))
        except (TypeError, ValueError):
            score = np.nan
        if box is None or not np.isfinite(score):
            rejected["invalid_box_or_score"] += 1
            continue
        if detection.get("class") != "window" or score < STRUCTURE_POLICY["min_detection_score"]:
            rejected["not_window_or_low_score"] += 1
            continue
        if min(*box[:2], *(photo_wh-box[2:])) < STRUCTURE_POLICY["photo_border_margin_px"]:
            rejected["clipped_or_near_photo_border"] += 1
            continue
        x0, y0, x1, y1 = box
        mapped = project_homography([[x0,y0], [x1,y0], [x1,y1], [x0,y1]], initial_h)
        if not np.isfinite(mapped).all() or (np.ptp(mapped, axis=0) <= 0).any():
            rejected["invalid_initial_projection"] += 1
            continue
        photo_centers.append((box[:2]+box[2:])/2)
        photo_sizes.append(np.ptp(mapped, axis=0))
        detection_indices.append(index)
    report.update({"visible_model_windows": len(model_centers),
                   "eligible_photo_windows": len(photo_centers), "rejected_detections": rejected})
    if not model_centers or not photo_centers:
        return fail("both_images_have_eligible_windows")
    pc, rc = np.asarray(photo_centers), np.asarray(model_centers)
    projected = project_homography(pc, initial_h)
    distances = np.linalg.norm(projected[:, None]-rc[None, :], axis=2)
    ratio = np.asarray(photo_sizes)[:, None]/np.asarray(model_sizes)[None, :]
    eligible = ((ratio >= STRUCTURE_POLICY["min_size_ratio"]) & (ratio <= STRUCTURE_POLICY["max_size_ratio"])).all(axis=2)
    eligible &= distances <= STRUCTURE_POLICY["max_association_distance_render_px"]
    costs = np.where(eligible, distances, np.inf)
    pairs = [(i, j) for i, j in enumerate(costs.argmin(axis=1))
             if np.isfinite(costs[i, j]) and costs[:, j].argmin() == i]
    report["association_count"] = len(pairs)
    if not pairs:
        return fail("mutual_nearest_associations_available")
    pair = np.asarray(pairs)
    photo, render = pc[pair[:, 0]], rc[pair[:, 1]]
    arrays.update({"photo_xy": photo, "render_xy": render,
                   "detection_indices": np.asarray(detection_indices)[pair[:, 0]],
                   "model_object_indices": np.asarray(model_indices)[pair[:, 1]],
                   "initial_distance_px": distances[pair[:, 0], pair[:, 1]]})
    valid = _inside(photo, photo_wh) & _inside(render, render_wh)
    cells = np.minimum(np.floor((photo+.5)/photo_wh*[8, 4]).astype(int), [7, 3])
    holdout = valid & (cells.sum(axis=1) % 3 == 0)
    train = valid & ~holdout
    arrays.update({"valid_mask": valid, "photo_cells_xy": cells,
                   "train_mask": train, "holdout_mask": holdout})
    report.update({"train_count": int(train.sum()), "holdout_count": int(holdout.sum())})
    if train.sum() < 4:
        return fail("at_least_four_training_associations")
    if any(np.linalg.matrix_rank(xy-xy.mean(axis=0)) < 2 for xy in (photo[train], render[train])):
        return fail("training_associations_not_collinear")
    cv2.setRNGSeed(0)
    method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
    H, solver_mask = cv2.findHomography(photo[train], render[train], method,
                                      STRUCTURE_POLICY["fit_ransac_render_px"],
                                      maxIters=10000, confidence=.999)
    if H is None or not np.isfinite(H).all() or abs(H[2, 2]) < 1e-12 or np.linalg.matrix_rank(H) < 3:
        return fail("nonsingular_homography_found")
    H = H/H[2, 2]
    predicted = project_homography(photo, H)
    errors = np.linalg.norm(predicted-render, axis=1)
    errors[~valid | ~np.isfinite(errors)] = np.inf
    inliers = valid & (errors <= STRUCTURE_POLICY["fit_ransac_render_px"])
    train_inliers, holdout_inliers = train & inliers, holdout & inliers
    solver_inliers = np.zeros(len(photo), dtype=bool)
    if solver_mask is not None:
        solver_inliers[train] = solver_mask.ravel().astype(bool)
    coverage = _coverage(photo[train_inliers], photo_wh, cv2)
    train_fraction = float(train_inliers.sum()/max(1, train.sum()))
    holdout_fraction = float(holdout_inliers.sum()/max(1, holdout.sum()))
    checks = {
        "train_inliers_at_least_12": bool(train_inliers.sum() >= STRUCTURE_POLICY["min_train_inliers"]),
        "holdout_matches_at_least_6": bool(holdout.sum() >= STRUCTURE_POLICY["min_holdout_matches"]),
        "train_inlier_fraction_at_least_0_70": train_fraction >= STRUCTURE_POLICY["min_train_inlier_fraction"],
        "holdout_inlier_fraction_at_least_0_70": holdout_fraction >= STRUCTURE_POLICY["min_holdout_inlier_fraction"],
        "train_photo_hull_fraction_at_least_0_30": coverage["convex_hull_fraction"] >= STRUCTURE_POLICY["min_photo_hull_fraction"],
        "train_photo_x_span_at_least_0_65": coverage["x_span_fraction"] >= STRUCTURE_POLICY["min_photo_x_span"],
        "train_photo_y_span_at_least_0_60": coverage["y_span_fraction"] >= STRUCTURE_POLICY["min_photo_y_span"],
    }
    pose_checks, change = _pose_checks(H, initial_h, photo_wh, rc)
    checks.update(pose_checks)
    report.update({"solver": "USAC_MAGSAC" if method == getattr(cv2, "USAC_MAGSAC", None) else "RANSAC",
                   "train_inlier_count": int(train_inliers.sum()), "holdout_inlier_count": int(holdout_inliers.sum()),
                   "train_inlier_fraction": train_fraction, "holdout_inlier_fraction": holdout_fraction,
                   "train_inlier_errors": _error_summary(errors[train_inliers]),
                   "holdout_all_errors": _error_summary(errors[holdout]),
                   "holdout_inlier_errors": _error_summary(errors[holdout_inliers]),
                   "train_photo_coverage": coverage, "initial_change": change,
                   "checks": checks, "gate_passed": bool(all(checks.values())),
                   "failed_checks": [key for key, passed in checks.items() if not passed]})
    arrays.update({"predicted_render_xy": predicted, "errors_px": errors,
                   "inlier_mask": inliers, "train_inlier_mask": train_inliers,
                   "holdout_inlier_mask": holdout_inliers, "solver_train_inlier_mask": solver_inliers})
    return H, report, arrays
