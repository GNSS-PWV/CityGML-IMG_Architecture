"""真实照片 ↔ 渲染图匹配的坐标转换、空间留出检查和三维回投。

这里不调用 RoMa、不下载权重，也不读取人工标定点。RoMa 给出候选对应点后，
本模块检查这些点能否由同一个平面投影解释。**几何自洽不能证明匹配的是同一
扇窗**：重复窗户整体错移一列，也可能拟合得很好。自动模式可据此筛选候选，
但不能仅凭自洽指标宣称真实米制精度或唯一正确的窗列对应已经验证。
"""

import numpy as np


def _points(value, columns=2):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != columns:
        raise ValueError("点数组必须为 N×%d。" % columns)
    return value


def _size(value):
    value = np.asarray(value, dtype=np.float64)
    if value.shape != (2,) or not np.isfinite(value).all() or (value <= 0).any():
        raise ValueError("图片尺寸必须为正数 [width, height]。")
    return value


def roma_to_image_pixels(normalized, photo_wh, crop_wh, crop_xy=(0, 0)):
    """将 RoMa 的 [-1,1] 坐标还原到两张原图的像素中心坐标。

    每行是 [photo_x, photo_y, render_crop_x, render_crop_y]。
    RoMa 使用 align_corners=False 的归一化坐标，故需减去半个像素；
    渲染图如果曾裁剪，最后加回裁剪框左上角，才能查询原始深度图。
    不把越界值强行截到边缘，交给后续有效性检查剔除。
    """
    matches = _points(normalized, 4)
    offset = np.asarray(crop_xy, dtype=np.float64)
    if offset.shape != (2,) or not np.isfinite(offset).all():
        raise ValueError("裁剪偏移必须为有限的 [x, y]。")
    return ((matches[:, :2] + 1) * _size(photo_wh) / 2 - .5,
            (matches[:, 2:] + 1) * _size(crop_wh) / 2 - .5 + offset)


def project_homography(points, homography):
    """用 3×3 单应矩阵投影 Nx2 点；投影到无穷远的点返回 NaN。"""
    points = _points(points)
    homography = np.asarray(homography, dtype=np.float64)
    if homography.shape != (3, 3) or not np.isfinite(homography).all():
        raise ValueError("单应矩阵必须为有限的 3×3 数组。")
    homogeneous = np.column_stack((points, np.ones(len(points)))) @ homography.T
    result = np.full((len(points), 2), np.nan)
    valid = np.isfinite(homogeneous).all(axis=1) & (np.abs(homogeneous[:, 2]) > 1e-12)
    result[valid] = homogeneous[valid, :2] / homogeneous[valid, 2:3]
    return result


def _inside(points, wh):
    # 像素中心约定：图像连续边界是 -0.5 到 width/height-0.5。
    return np.isfinite(points).all(axis=1) & (points >= -.5).all(axis=1) & (points < wh - .5).all(axis=1)


def _coverage(points, wh, cv2):
    if len(points) < 3:
        return {"convex_hull_fraction": 0., "x_span_fraction": 0., "y_span_fraction": 0.}
    hull = cv2.convexHull(np.asarray(points, dtype=np.float32))
    spans = np.ptp(points, axis=0) / wh
    return {"convex_hull_fraction": float(cv2.contourArea(hull) / np.prod(wh)),
            "x_span_fraction": float(spans[0]), "y_span_fraction": float(spans[1])}


def _error_summary(errors):
    # 若出现无穷远投影，保留为失败，不仅统计剩余的有限误差。
    if len(errors) == 0:
        return {"median_px": None, "p90_px": None, "max_px": None}
    finite = np.isfinite(errors)
    ordered = np.sort(np.where(finite, errors, np.inf))
    def percentile(q):
        result = float(ordered[min(len(ordered) - 1, int(np.ceil(q * len(ordered))) - 1)])
        return result if np.isfinite(result) else None
    return {"median_px": percentile(.5), "p90_px": percentile(.9),
            "max_px": float(ordered[-1]) if finite.all() else None,
            "nonfinite_count": int((~finite).sum())}


def fit_and_check(photo_xy, render_xy, photo_wh, render_wh, ransac_px=4.0):
    """空间分区拟合单应性 H（照片 → 完整渲染图），并返回可审计的质量指标。

    照片分成 8×4 格，(列号+行号)%3==0 的格子全部留作检查，其余格子拟合。
    留出点不参与 RANSAC 或最终重拟合，因此不是用拟合点报告训练误差。
    固定的初步几何门槛：训练内点≥40，留出点≥20且留出内点≥20，留出内点
    比例≥50%，训练内点覆盖照片面积≥30%，宽≥65%、高≥60%，变换不能镜像或
    在照片范围内穿过无穷远。误差阈值默认是完整渲染图上的4像素。

    gate_passed 仅表示自动一致性通过；永不据此设置“真实精度已确认”。
    全部留出误差和内点误差分别报告，不隐藏被拒绝的错配。
    返回 (H, details, arrays)。arrays 的所有掩码/误差均与输入行一一对应。
    """
    import cv2

    photo = _points(photo_xy)
    render = _points(render_xy)
    if len(photo) != len(render):
        raise ValueError("照片点和渲染图点的数量不一致。")
    photo_wh, render_wh = _size(photo_wh), _size(render_wh)
    if not np.isfinite(ransac_px) or ransac_px <= 0:
        raise ValueError("RANSAC 像素阈值必须为正数。")
    valid = _inside(photo, photo_wh) & _inside(render, render_wh)
    cells = np.full((len(photo), 2), -1, dtype=int)
    cells[valid] = np.minimum(np.floor((photo[valid] + .5) / photo_wh * [8, 4]).astype(int), [7, 3])
    holdout = valid & (cells.sum(axis=1) % 3 == 0)
    train = valid & ~holdout
    if train.sum() < 4:
        raise ValueError("空间留出后不足4组有效训练点，无法拟合平面变换。")
    for xy in (photo[train], render[train]):
        if np.linalg.matrix_rank(xy - xy.mean(axis=0)) < 2:
            raise ValueError("训练点集中在一条直线上，无法建立稳定的平面变换。")
    cv2.setRNGSeed(0)
    method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
    H, solver_mask = cv2.findHomography(photo[train], render[train], method=method,
                                      ransacReprojThreshold=float(ransac_px),
                                      maxIters=10000, confidence=.999)
    if H is None or not np.isfinite(H).all() or abs(H[2, 2]) < 1e-12 or np.linalg.matrix_rank(H) < 3:
        raise ValueError("鲁棒拟合没有得到有效单应矩阵；请检查匹配是否集中、重复或大量错误。")
    H = H / H[2, 2]
    predicted = project_homography(photo, H)
    errors = np.linalg.norm(predicted - render, axis=1)
    errors[~valid | ~np.isfinite(errors)] = np.inf
    # 统一按同一像素阈值重新判定两组内点；另外保留求解器给出的训练内点掩码。
    inliers = valid & (errors <= ransac_px)
    train_inliers = train & inliers
    holdout_inliers = holdout & inliers
    solver_inliers = np.zeros(len(photo), dtype=bool)
    if solver_mask is not None:
        solver_inliers[train] = solver_mask.ravel().astype(bool)
    coverage = _coverage(photo[train_inliers], photo_wh, cv2)
    render_coverage = _coverage(render[train_inliers], render_wh, cv2)
    holdout_fraction = float(holdout_inliers.sum() / max(1, holdout.sum()))

    corners = np.array([[-.5, -.5], [photo_wh[0] - .5, -.5],
                        [photo_wh[0] - .5, photo_wh[1] - .5], [-.5, photo_wh[1] - .5]])
    denominator = np.column_stack((corners, np.ones(4))) @ H[2]
    no_pole = bool((denominator > 1e-8).all() or (denominator < -1e-8).all())
    mapped_corners = project_homography(corners, H)
    signed_area = float(np.sum(mapped_corners[:, 0] * np.roll(mapped_corners[:, 1], -1)
                               - mapped_corners[:, 1] * np.roll(mapped_corners[:, 0], -1)) / 2)
    checks = {
        "train_inliers_at_least_40": bool(train_inliers.sum() >= 40),
        "holdout_matches_at_least_20": bool(holdout.sum() >= 20),
        "holdout_inliers_at_least_20": bool(holdout_inliers.sum() >= 20),
        "holdout_inlier_fraction_at_least_0_50": bool(holdout_fraction >= .5),
        "train_photo_hull_fraction_at_least_0_30": coverage["convex_hull_fraction"] >= .30,
        "train_photo_x_span_at_least_0_65": coverage["x_span_fraction"] >= .65,
        "train_photo_y_span_at_least_0_60": coverage["y_span_fraction"] >= .60,
        "no_projective_pole_in_photo": no_pole,
        "no_mirrored_or_collapsed_photo": bool(np.isfinite(signed_area) and signed_area > 1.),
    }
    details = {
        "method": "USAC_MAGSAC" if method == getattr(cv2, "USAC_MAGSAC", None) else "RANSAC",
        "transform": "photo pixel centers -> full render pixel centers",
        "holdout_policy": "photo 8x4 spatial cells; (column+row)%3==0 held out; no refit on holdout",
        "ransac_px": float(ransac_px), "input_count": len(photo), "valid_count": int(valid.sum()),
        "train_count": int(train.sum()), "holdout_count": int(holdout.sum()),
        "train_inlier_count": int(train_inliers.sum()), "holdout_inlier_count": int(holdout_inliers.sum()),
        "train_inlier_fraction": float(train_inliers.sum() / max(1, train.sum())),
        "holdout_inlier_fraction": holdout_fraction,
        "train_inlier_errors": _error_summary(errors[train_inliers]),
        "holdout_all_errors": _error_summary(errors[holdout]),
        "holdout_inlier_errors": _error_summary(errors[holdout_inliers]),
        "train_photo_coverage": coverage, "train_render_coverage": render_coverage,
        "checks": checks, "gate_passed": bool(all(checks.values())),
        "failed_checks": [key for key, passed in checks.items() if not passed],
        "independent_validation_passed": False,
        "interpretation": "仅检查候选匹配的平面几何自洽；重复窗户错位仍可能通过，真实精度未验证。",
    }
    arrays = {"valid_mask": valid, "train_mask": train, "holdout_mask": holdout,
              "inlier_mask": inliers, "train_inlier_mask": train_inliers,
              "holdout_inlier_mask": holdout_inliers, "solver_train_inlier_mask": solver_inliers,
              "errors_px": errors, "predicted_render_xy": predicted, "photo_cells_xy": cells}
    return H, details, arrays


def backproject_render_points(render_xy, camera, depth, object_map, world_origin,
                              allowed_object_indices=None):
    """渲染图像素 → 三维世界坐标；无效点返回 NaN，另返回有效掩码。

    深度图记录的是相机轴向距离，不是 Z 高程；必须乘渲染器保存的逆矩阵。
    采用最近像素查询，回投也使用同一个整数像素中心，避免跨构件插值。
    allowed_object_indices 可限制到目标墙及其门窗，不能把背景或背面点当墙面。
    本函数只恢复被渲染表面上的点，不会自动把窗框凹凸面压回主墙平面。
    """
    points = _points(render_xy)
    depth = np.asarray(depth)
    objects = np.asarray(object_map)
    if depth.ndim != 2 or objects.shape != depth.shape:
        raise ValueError("深度图和构件编号图必须为尺寸相同的二维数组。")
    height, width = depth.shape
    if list(camera.get("image_size_wh", [width, height])) != [width, height]:
        raise ValueError("深度图尺寸与相机记录不一致。")
    matrix = np.asarray(camera["pixel_depth_to_local"], dtype=np.float64)
    origin = np.asarray(world_origin, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all() or origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError("相机逆矩阵或世界坐标原点无效。")
    valid = _inside(points, np.array([width, height]))
    pixels = np.zeros((len(points), 2), dtype=int)
    pixels[valid] = np.floor(points[valid] + .5).astype(int)
    selected = np.flatnonzero(valid)
    pixel_depth = np.full(len(points), np.nan)
    selected_objects = np.full(len(points), -1, dtype=np.int64)
    pixel_depth[selected] = depth[pixels[selected, 1], pixels[selected, 0]]
    selected_objects[selected] = objects[pixels[selected, 1], pixels[selected, 0]]
    valid &= np.isfinite(pixel_depth) & (pixel_depth > 0) & (selected_objects >= 0)
    if allowed_object_indices is not None:
        valid &= np.isin(selected_objects, list(allowed_object_indices))
    world = np.full((len(points), 3), np.nan)
    pixel_depth_h = np.column_stack((pixels[valid], pixel_depth[valid], np.ones(valid.sum())))
    local_h = pixel_depth_h @ matrix.T
    local_valid = np.isfinite(local_h).all(axis=1) & (np.abs(local_h[:, 3]) > 1e-12)
    valid_indices = np.flatnonzero(valid)
    good_indices = valid_indices[local_valid]
    world[good_indices] = local_h[local_valid, :3] / local_h[local_valid, 3:4] + origin
    valid[valid_indices[~local_valid]] = False
    return world, valid
