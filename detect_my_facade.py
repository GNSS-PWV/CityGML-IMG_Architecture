"""批量检测自己的全部立面照片：在 PyCharm 中直接运行本文件。

【这份代码是谁做什么的】
这是为当前 Texture2LoD3_dataset 编写的教学适配脚本。
Python 在这里负责读取照片、调用已有模型、筛选结果、保存文件；神经网络本身由库实现。
本文件没有从零实现或训练神经网络，也没有调用参考仓库的 Python 函数。

【参考代码在哪里，具体参考了什么】
参考仓库：chrise96/3D_building_reconstruction；参考文件：stage_2/test.py。
重点对照它的 main()、instances_to_dict()、non_max_suppression()。
沿用了“照片检测 → 按窗/门分组 → 去重 → 保存坐标”的任务流程，尤其是
texture_filename、bboxes_window、bboxes_door 这三个输出字段的约定。
下面的推理调用、几何去重函数、批量图片读取及文件保存代码是新增适配实现，
并非直接复制执行上述参考文件；去重算法的细节也与参考实现不同。

【原模型与现在的模型】
参考 Stage 2：Detectron2 的 Mask R-CNN，使用 R50-FPN 配置，类别为
sky/window/door，需要作者的专用 model_output/model.pth 权重。
当前参考仓库没有这份专用权重，因此本脚本改用公开的 Grounding DINO Tiny：
https://huggingface.co/IDEA-Research/grounding-dino-tiny
通过 Hugging Face Transformers 调用，PyTorch 负责张量运算和 GPU 计算。
Grounding DINO 能结合照片与文字提示寻找目标；“预训练”表示权重已经由发布方训练过，本脚本只做推理，不用你的数据更新权重。这属于替换检测模型的适配，
不能称为原 Mask R-CNN 实验的完全复现。

【先按这个顺序读】
参数区 → main() 批量循环 → detect_image() 单图检测 → save_results() 保存结果。
运行时 Python 先读取参数、定义函数，最后从文件末尾的 main() 调用开始处理照片。
输入：Texture2LoD3_dataset/textures 内全部17张立面照片；不是 gt_masks 标注或原始全景。
直接运行本文件即可：加载一次本地模型 → 逐张原像素分块检测 → 去重 → 分别保存。
输出：my_results/batch_detection/run_时间/照片名/ 下的 detections.png、output_stage2.csv、
detections.json、needs_review.json。黄色框为门窗冲突，保留在 JSON 中，暂不计入 CSV 门窗列。
每次运行新建时间目录，并输出批量汇总和 mapping_manifest.json，供后续三维标定使用。
某张图片失败时记录错误并继续下一张；最终汇总明确区分成功、失败、尚未处理。
默认不缩放图块，也不额外推理整张超宽图；USE_TILES/INCLUDE_FULL_IMAGE/DO_RESIZE 可调整。
离线模式不会下载权重，若本地文件缺失会报出缺失路径。

【目前完成到哪里】
只输出二维检测框，坐标单位是像素；不读取 gt_masks，也不生成或修改 CityGML。
gt_masks 可在检测后用于验证，但需先解决标注图和照片的对应、配准问题。
参考 Stage 3 还需要墙面身份、图像到三维墙面的对应关系和实际尺寸等信息；
三列 CSV 的格式相同，不意味着可直接套用原 Stage 3 的 30 米图高假设。
"""

# 下面都是 Python 标准库，不需要分别用 pip 安装。
# Path：拼接文件路径；用 / 拼接路径时，这里的 / 不是数学除法。
from pathlib import Path
# csv：把检测结果写成表格文件。
import csv
# inspect：查看库函数接受哪些参数，用来兼容阈值参数的不同名称。
import inspect
# json：把 Python 列表/字典转成可保存、可再读取的 JSON 文本。
import json
# os：在导入模型库前启用离线模式。
import os
# sys.executable：打印“这次真正运行程序的 Python 在哪里”。
import sys
from datetime import datetime
import gc
import re
import hashlib
import io


# ==================== 参数区：初学时主要修改这里 ====================
# 【新增适配】以这个 .py 文件所在目录为项目根目录。
# __file__ 是当前脚本路径，resolve() 得到绝对路径，parent 取上一级目录。
# 因此这些数据路径不依赖你从哪个终端目录启动程序。
ROOT = Path(__file__).resolve().parent

# 【你的数据】这一层含 citygml、obj、textures、panoramas、gt_masks。
DATASET_DIR = ROOT / "Drills/Texture2LoD3_dataset"
# 仅扫描立面照片目录。gt_masks是答案图；panoramas是待定位/提取立面的原始全景。
IMAGE_DIR = DATASET_DIR / "textures"
OUTPUT_ROOT = ROOT / "my_results/batch_detection"

# 【使用的模型】这不是本地 .py 文件名，而是 Hugging Face 上的模型仓库 ID。
# 对应 Grounding DINO Tiny；缓存配置里视觉骨干为 Swin，文本编码器为 BERT。
# 具体网络层、注意力计算等在 Transformers 库内，本文件通过接口调用。
MODEL_ID = "IDEA-Research/grounding-dino-tiny"

# 已经下载的模型配置、分词文件和预训练权重所在缓存目录。
# 本版只读该目录，不再访问模型仓库；缓存缺失时明确报错，不自动下载。
CACHE_DIR = ROOT / "model_cache"

# ==================== 本次主要设置：一个文件直接运行 ====================
# False：不改变照片/图块的宽高；像素值仍会按模型要求归一化，这是必要的数据转换。
# True：恢复权重配套的默认缩放（短边 800，长边不超过 1333）。
DO_RESIZE = False
# True：把宽图切成重叠图块，依次处理，最后恢复到原图坐标；不会缩小原始像素。
USE_TILES = True
# False：默认只检测图块，降低整张原图一次推理的显存压力。
# True：在图块之外再检测一次全图；当 DO_RESIZE=False 时，全图同样不缩放，更占显存。
# 若 USE_TILES=False，直接检测整张图，此项不再起作用。
INCLUDE_FULL_IMAGE = False

# 检测框评分门槛。保留你当前的0.28设置；后处理保留 score > BOX_THRESHOLD 的框。
# 调低可能增加检出，也可能增加误检；调高更严格，也可能漏掉真实目标。
# 这是筛选参数，不代表检测准确率。
BOX_THRESHOLD = 0.28

# 文本 token（文字被切分后的单元）的匹配门槛，用于后处理生成文字标签。
# 【版本细节】你当前的 Transformers 4.46.3 用 BOX_THRESHOLD 筛选框，
# TEXT_THRESHOLD 用来拼出返回的 labels。本脚本没有使用返回的 labels，
# 而是用下面循环中的 kind 赋类别，因此在该版本中单独调这个值不会改变
# 最终保存的框、评分和 window/door 类别。不要把它当作第二道框评分门槛。
TEXT_THRESHOLD = 0.25

# 重叠框判定门槛。先用 IoU 判断接近的框，再用下面的包含规则补充。
# 调低会更容易合并，但过低也可能误合并相邻目标；调高则更容易留下重复框。
NMS_IOU = 0.45
# 小框的至少 90% 被大框覆盖，才考虑“外框/内框属于同一个开口”。
NESTED_IOS = 0.90
# 同时要求小框面积至少是大框的 25%，两个中心在横纵方向都比较接近。
# 降低把大门中很小的玻璃、相距较远的两个目标当成重复的风险。
NESTED_MIN_AREA_RATIO = 0.25
NESTED_CENTER_FRACTION = 0.30

# 【分块参数】保持每块原像素，避免宽照片按默认规则缩小后丢失小窗细节。
# 横向和纵向都限制图块尺寸，较高的照片也不用整幅高度同时进入显存。
# 2940×833照片仍生成原来的三个横向图块；高于1000的照片还会纵向分块。
TILE_WIDTH = 1200
TILE_HEIGHT = 1000
TILE_OVERLAP = 300
# 距“内部切边”12像素以内的框可能不完整，交由重叠块补充；原图真正边缘不过滤。
TILE_EDGE_MARGIN = 12


def local_model_directory():
    """找到已下载模型的本地快照，纯文件读取；缺文件时不会尝试联网补齐。

    Hugging Face 缓存的 refs/main 保存当前版本编号，snapshots/版本编号存模型文件。
    这样无需硬编码版本号，也不需要向服务器查询版本。
    """
    repo_dir = CACHE_DIR / ("models--" + MODEL_ID.replace("/", "--"))
    ref_file = repo_dir / "refs" / "main"
    if not ref_file.is_file():
        raise FileNotFoundError(f"本地模型版本记录不存在：{ref_file}。当前是离线模式，不会下载。")
    model_dir = repo_dir / "snapshots" / ref_file.read_text(encoding="utf-8").strip()
    required = ("config.json", "preprocessor_config.json", "tokenizer_config.json",
                "tokenizer.json", "vocab.txt", "model.safetensors")
    missing = [name for name in required if not (model_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"本地模型文件不完整：{model_dir}，缺少 {missing}。离线模式不会下载。")
    return model_dir


def overlap(a, b):
    """【新增实现】计算两个框的 IoU（交并比），供 remove_duplicates 调用。

    a、b 都是 [左 x, 上 y, 右 x, 下 y]，例如 [10, 20, 50, 80]。
    IoU = 两框相交的面积 / 两框合并后的面积。
    完全重合为 1，完全不相交为 0。它描述两个框的重叠程度，不是模型评分。
    这里使用连续像素坐标面积，不额外给宽、高加 1。
    """
    # 相交宽 = 较小的右边界 - 较大的左边界；相交高同理。
    # 若两框错开，差值可能为负；max(0, ...) 把这种情况处理成 0。
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1]))
    # 分别计算两个矩形自身的面积。
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    # 并集面积 = A 面积 + B 面积 - 相交面积；1e-12 防止分母为 0。
    return intersection / max(area_a + area_b - intersection, 1e-12)


def same_opening(a, b):
    """【新增几何规则】两个 XYXY 框是否很可能指向同一个开口。

    IoU = 交集 / 并集；IoS = 交集 / 较小框面积。
    小框完全落在大框内时 IoS=1，即使 IoU 只有 0.34 也可被发现。
    IoS 配合面积比和中心距离使用，不能仅凭“包含”就把所有框合并。
    这里只判断几何关系，不判断这个开口究竟是门还是窗。
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
    """【新增适配，修正旧 NMS】按位置分组，同位置只画一个框，保留类别冲突。

    旧实现仅做同类 IoU-NMS，会留下“外窗框包着内窗框”和“同处门/窗”两个问题。
    现在按 same_opening 分组，每组使用最高分候选的坐标，不平均或扩张坐标。
    同组只有 window 或只有 door，保留该类别；两类都有，标成 ambiguous（待确认）。
    评分不是已校准的门窗分类概率，不能仅凭门的分数较高就断言它是门。
    candidates 保存组内原始预测，供后续核对，不用 GT 自动改类别。

    本函数是本项目的适配实现，不是原作者代码，也不是 Soft-NMS。
    """
    groups = []
    for record in sorted(records, key=lambda r: r["score"], reverse=True):
        # 要求与组内每个框都接近，避免 A 接近 B、B 接近 C 就连锁合并整排窗户。
        for group in groups:
            if all(same_opening(record["box"], old["box"]) for old in group):
                group.append(record)
                break
        else:  # 没找到符合条件的组，建立新组。
            groups.append([record])
    detections = []
    for group in groups:
        result = dict(group[0])  # 复制记录，避免修改传入的原始预测。
        result["candidates"] = group
        classes = {r["class"] for r in group}
        if len(classes) > 1:
            result["class"] = "ambiguous"
            result["review_required"] = True
            result["class_scores"] = {kind: max(r["score"] for r in group if r["class"] == kind)
                                      for kind in sorted(classes)}
        detections.append(result)
    # ponytail: 几何分组仍是启发式，大框覆盖多个真实开口时不保证正确，需核查或专用模型。
    return detections


def tile_boxes(width, height, tile_width, overlap_width):
    """返回二维重叠图块的 XYXY 范围，不缩放或修改原始照片。

    默认不缩放图块，降低单次输入尺寸而保留局部像素；重叠减少目标被切断的影响。
    若照片宽高均不超过一块，返回空列表，由主流程只做一次全图检测。
    """
    if min(width, height, tile_width, TILE_HEIGHT) <= 0:
        raise ValueError("照片宽高和图块宽高必须大于0")
    if not 0 <= overlap_width < min(tile_width, TILE_HEIGHT):
        raise ValueError("TILE_OVERLAP 必须非负，且小于图块宽和高")
    if width <= tile_width and height <= TILE_HEIGHT:
        return []

    def starts(length, size):
        if length <= size:
            return [0]
        values = list(range(0, length - size + 1, size - overlap_width))
        if values[-1] != length - size:
            values.append(length - size)
        return values

    return [(left, top, min(left + tile_width, width), min(top + TILE_HEIGHT, height))
            for top in starts(height, TILE_HEIGHT) for left in starts(width, tile_width)]


def detection_views(width, height, use_tiles, include_full_image):
    """生成（名称，XYXY 范围）；窄图无需切块时自动只处理一次全图。"""
    crops = tile_boxes(width, height, TILE_WIDTH, TILE_OVERLAP) if use_tiles else []
    views = [(f"tile_{i + 1}", box) for i, box in enumerate(crops)]
    if include_full_image or not views:
        views.insert(0, ("full", (0, 0, width, height)))
    return views


def box_to_full_image(box, view_box, image_size, edge_margin):
    """【新增坐标转换】图块内部的框 → 原照片的框；无效/切边框返回 None。

    例：图块从原图 x=900 开始，图块内 x=100 的位置对应原图 x=1000。
    图块可横纵分割，四个内部切边都要检查；原图真正的四周边缘不受此限制。
    全图视图没有内部切边，其裁界规则与原来的单图流程一致。
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


def save_results(image, report, output_dir):
    """保存一张照片的结果，保证图/JSON/CSV使用同一组最终框。

    蓝框=窗预测，橙框=门预测，黄框=门窗冲突待确认。
    待确认框保存在 JSON 中，但不写入 CSV 的窗/门列，避免同一开口写入三维两次。
    少一个重复框不代表类别已经判对；需检查 needs_review.json 中的原始候选。
    """
    from PIL import ImageDraw, ImageFont

    output_dir.mkdir(parents=True, exist_ok=True)
    detections = report["detections"]
    windows = [r["box"] for r in detections if r["class"] == "window"]
    doors = [r["box"] for r in detections if r["class"] == "door"]
    uncertain = [r for r in detections if r["class"] == "ambiguous"]
    # 按位置编号，便于看图与 needs_review.json 对照。编号仅适用于这一次结果。
    for i, r in enumerate(sorted(uncertain, key=lambda r: (r["box"][0], r["box"][1])), 1):
        r["review_id"] = f"R{i:02d}"
    report.update({"window_count": len(windows), "door_count": len(doors),
                   "ambiguous_count": len(uncertain), "opening_count": len(detections),
                   "csv_excludes_ambiguous": True,
                   "postprocess": {"method": "iou_or_guarded_containment_groups_v2",
                                   "nms_iou": NMS_IOU, "nested_ios": NESTED_IOS,
                                   "min_area_ratio": NESTED_MIN_AREA_RATIO,
                                   "center_fraction": NESTED_CENTER_FRACTION,
                                   "class_conflict_policy": "keep_as_ambiguous"}})

    # 【沿用参考接口】三列名称保持与原 Stage 2 一致；待确认开口另存，不强塞类别。
    csv_path = output_dir / "output_stage2.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["texture_filename", "bboxes_window", "bboxes_door"])
        writer.writeheader()
        writer.writerow({"texture_filename": Path(report["image"]).stem,
                         "bboxes_window": json.dumps(windows), "bboxes_door": json.dumps(doors)})
    (output_dir / "detections.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "needs_review.json").write_text(
        json.dumps(uncertain, indent=2, ensure_ascii=False), encoding="utf-8")

    # 【可视化】始终在原照片副本上画最终结果，不能在旧 detections.png 上叠加新框。
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    font_path = Path("C:/Windows/Fonts/arial.ttf")
    font = ImageFont.truetype(str(font_path), 22) if font_path.exists() else ImageFont.load_default()
    colors = {"window": "#1689ff", "door": "#ff941a", "ambiguous": "#ffe033"}
    for record in detections:
        color = colors[record["class"]]
        draw.rectangle(record["box"], outline=color, width=4)
        x0, y0 = record["box"][:2]
        # 冲突框不显示一个“获胜类别/评分”，防止把几何代表框的分数理解成类别结论。
        # W/D = window/door；缩短文字，避免相邻开口的标签挤在一起。
        label = (f"{record['review_id']} W/D ?" if record["class"] == "ambiguous"
                 else f"{record['class']} {record['score']:.2f}")
        draw.text((x0, max(0, y0 - 25)), label, fill=color, font=font,
                  stroke_width=1, stroke_fill="black")
    plot_path = output_dir / "detections.png"
    canvas.save(plot_path)
    print(f"[4] Kept {len(windows)} windows, {len(doors)} doors, "
          f"{len(uncertain)} ambiguous openings.", flush=True)
    print(f"[5] Saved: {plot_path}\n    CSV: {csv_path}\n    JSON: {output_dir / 'detections.json'}", flush=True)
    if uncertain:
        print(f"REVIEW: {len(uncertain)} yellow openings are NOT included in the window/door CSV. "
              f"See {output_dir / 'needs_review.json'}", flush=True)
    if not detections:
        print("No objects passed the current threshold. Inspect the photo and model settings.")


def detect_image(image_path, processor, model, device, model_dir):
    """检测一张照片；processor/model由批量主流程传入，因此不会逐张重载权重。"""

    use_tiles = USE_TILES

    # ==================== 步骤 1：加载第三方库 ====================
    # 这些包需要安装在 PyCharm 当前选中的解释器环境中（你现在用 torch_1）。
    # 放在 main() 内导入，使单独研究上面的 IoU/NMS 函数时不必先加载模型库。
    import torch          # PyTorch：张量计算、CPU/GPU 设备、推理上下文。
    import transformers   # Hugging Face 的模型实现库；这里也记录它的版本。
    from PIL import Image  # Pillow：读图片；保存函数再导入画图所需对象。

    # ==================== 步骤 2：读取自己的照片 ====================
    # assert：如果条件不成立，立即停止并显示这里写的中文错误。
    assert image_path.is_file(), f"找不到照片：{image_path}"
    # with 会在读取完成后关闭文件。convert("RGB") 统一为红、绿、蓝三通道。
    # 参考 Stage 2 用 OpenCV 的 BGR 图片供 Detectron2 使用；这里是 PIL RGB，
    # 后续交给 Grounding DINO 的 processor 处理，不要照搬原代码交换通道。
    # 对实际读入的照片字节做指纹，后续三维标定可检查照片有没有被同名替换。
    image_bytes = image_path.read_bytes()
    with Image.open(io.BytesIO(image_bytes)) as source:
        image = source.convert("RGB")
    image_sha256 = hashlib.sha256(image_bytes).hexdigest()
    del image_bytes
    width, height = image.size  # PIL 返回顺序是（宽，高），单位：像素。

    print(f"    Image: {image_path.name}; width={width}, height={height}; device={device}", flush=True)

    # 你遇到的 cl/DLL 警告来自库尝试加载可变形注意力的自定义加速扩展。
    # 当前 4.46.3 在扩展不可用时可回退到普通 PyTorch 实现；模型/输入位于
    # CUDA 时仍可在 GPU 运算。是否成功应看后续检测、输出和退出状态。

    # ==================== 步骤 4：准备视图和后处理，分别寻找窗户和门 ====================
    # 【模型调用位置 B：取得配套后处理函数】网络原始输出还不是最终像素框，
    # 这个函数负责分数筛选、框格式转换和恢复到原图尺寸。
    # 不同 Transformers 版本中，框门槛参数可能叫 threshold 或 box_threshold。
    # inspect.signature(...).parameters 读取参数名，不会执行一次检测。
    postprocess = processor.post_process_grounded_object_detection
    threshold_name = "threshold" if "threshold" in inspect.signature(postprocess).parameters else "box_threshold"
    # 用原像素网格覆盖整图；可用 INCLUDE_FULL_IMAGE=True 额外加入全图。
    # 重叠能照顾部分跨切边目标，但超大目标仍可能需要全图上下文。
    views = detection_views(width, height, use_tiles, INCLUDE_FULL_IMAGE)
    has_tiles = any(name.startswith("tile_") for name, _ in views)
    has_full = any(name == "full" for name, _ in views)
    inference_mode = "full_plus_tiles" if has_full and has_tiles else "tiles_only" if has_tiles else "full_only"
    records = []  # 所有视图通过阈值、裁界和切边检查的候选记录，尚未统一去重。
    view_stats = []  # 记录每次检测的来源和数量，方便检查分块究竟增加了什么。
    print(f"    Mode: {inference_mode}; views={len(views)}; do_resize={DO_RESIZE}", flush=True)

    # 模型只加载一次；下面重复使用它处理全图和各图块，不会每块重新下载权重。
    for source_name, view_box in views:
        view = image if source_name == "full" else image.crop(view_box)
        view_width, view_height = view.size
        # 【当前模型与原模型的区别】原 Mask R-CNN 使用 sky/window/door 固定类别。
        # 这里每个视图先配 a window.，再配 a door.，各执行一次预测。
        # kind 是本脚本保存用的类别名，prompt 是实际送进模型的英文文字提示。
        for kind, prompt in (("window", "a window."), ("door", "a door.")):
            # 【模型调用位置 C：预处理】默认关闭空间缩放，保留归一化和文本 token 转换。
            # do_rescale（像素数值除以255）与 do_resize（改变宽高）不同，不要一起关闭。
            # pt 表示 PyTorch 张量；输入与模型放到同一个 CPU/GPU 上。
            inputs = processor(images=view, text=prompt, do_resize=DO_RESIZE,
                               return_tensors="pt").to(device)
            model_height, model_width = inputs["pixel_values"].shape[-2:]
            print(f"[3] {source_name}: {kind}; crop={view_box}; "
                  f"source={view_width}x{view_height}; model_input={model_width}x{model_height}", flush=True)
            # inference_mode 不记录反向传播所需的梯度；本脚本不会更新模型权重。
            with torch.inference_mode():
                # 【模型调用位置 D：真正执行神经网络】**inputs 展开输入字典。
                outputs = model(**inputs)

            # 【模型调用位置 E：后处理】必须先恢复到“当前视图”的像素尺寸！
            # 不能直接填整张照片的尺寸，否则图块里的坐标会被错误拉伸。
            # target_sizes 的顺序是（高，宽）；只有一张图，所以返回列表取 [0]。
            # **{threshold_name: ...} 动态传入当前版本支持的参数名。
            result = postprocess(
                outputs, inputs.input_ids, text_threshold=TEXT_THRESHOLD,
                target_sizes=[(view_height, view_width)], **{threshold_name: BOX_THRESHOLD})[0]

            # ==================== 步骤 5：整理当前视图的框，恢复原图坐标 ====================
            # boxes 与 scores 按位置配对；detach/cpu/tolist 只转换输出数据，
            # 不会把整个模型切到 CPU。坐标顺序是 [左 x, 上 y, 右 x, 下 y]。
            boxes = result["boxes"].detach().cpu().tolist()
            scores = result["scores"].detach().cpu().tolist()
            accepted = 0
            for box, score in zip(boxes, scores):
                # 新增 helper：裁界、排除无效/内部切边框，再加图块起点偏移量。
                # 例：tile_2 起点 x=900，局部 x=100 → 原图 x=1000。
                full_box = box_to_full_image(box, view_box, (width, height), TILE_EDGE_MARGIN)
                if full_box is not None:
                    # 类别取本轮 kind；未读取 result["labels"] 或 gt_masks。
                    # source 标明全图/第几块的预测，保留在 JSON 中供你回溯。
                    records.append({"class": kind, "score": score, "box": full_box,
                                    "source": source_name})
                    accepted += 1
            view_stats.append({"source": source_name, "crop_box": list(view_box),
                               "class": kind, "model_input_size": [model_width, model_height],
                               "postprocess_count": len(boxes), "accepted_count": accepted,
                               "discarded_boundary_or_invalid_count": len(boxes) - accepted})
            # 坐标/评分已经转成 Python 数字，及时释放本轮张量的引用，
            # 避免下一次预测期间还保留上一次的模型输出；不会删除模型或缓存权重。
            del outputs, result, inputs

    # ==================== 步骤 6：几何去重，并记录门窗类别冲突 ====================
    # 【参考任务流程 + 新增算法实现】原 Stage 2 也按类去重，但本脚本调用的是
    # 上方自己实现的 remove_duplicates，而不是参考文件中的函数。
    # 分块模式下，必须等框都恢复到原图坐标后再统一去重，才能合并重叠区域的重复框。
    # 新版同时检查包含关系；同一位置出现门和窗时标成待确认，不强行取高分类别。
    detections = remove_duplicates(records)
    # ==================== 步骤 7：整理实验记录 ====================
    # raw_detection_count 实际是“本脚本 NMS 前的候选数”：已经经过 BOX_THRESHOLD、
    # 裁界、有效框/切边检查；它不是神经网络输出的全部原始候选数。
    report = {"scope": "RGB inference with public Grounding DINO; no GT masks used; 2D only.",
              "model": MODEL_ID, "model_local_path": str(model_dir), "local_files_only": True,
              "python": sys.executable, "device": device,
              "torch_version": torch.__version__, "transformers_version": transformers.__version__,
              "image": str(image_path), "image_sha256": image_sha256, "image_size": [width, height],
              "box_threshold": BOX_THRESHOLD, "text_threshold": TEXT_THRESHOLD,
              "nms_iou": NMS_IOU, "raw_detection_count": len(records),
              "postprocess_input_count": len(records), "candidate_source": "before_geometric_cleanup",
              "candidates_before_cleanup": records, "detections": detections}
    report.update({"inference_mode": inference_mode, "do_resize": DO_RESIZE,
                   "tiling": {"enabled": has_tiles, "include_full_image": has_full,
                              "width": TILE_WIDTH, "height": TILE_HEIGHT,
                              "overlap": TILE_OVERLAP, "edge_margin": TILE_EDGE_MARGIN},
                   "views": view_stats})
    # 把原图与结果交回批量主流程，由 save_results 保存并统计本张结果。
    return image, report


def find_images():
    """只查找 IMAGE_DIR 中的真实照片，按文件名排序；同名输出冲突时提前报错。"""
    if not IMAGE_DIR.is_dir():
        raise FileNotFoundError(f"找不到立面照片目录：{IMAGE_DIR}")
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    images = sorted((p.resolve() for p in IMAGE_DIR.rglob("*")
                     if p.is_file() and p.suffix.lower() in extensions), key=lambda p: str(p).lower())
    if not images:
        raise ValueError(f"照片目录为空：{IMAGE_DIR}")
    keys = [p.stem.lower() for p in images]
    if len(set(keys)) != len(keys):
        raise ValueError("照片存在相同的不带扩展名文件名，请先改为不同名称，避免输出互相覆盖")
    return images


def mapping_item(image_path, run_dir):
    """建立后续3D的文件关联，不猜测墙面方向或像素到米的转换。

    4959323_front -> 建筑4959323；建筑身份不等于已经知道正面墙上每个像素的位置。
    wall_id暂留空，需在map_facade_to_3d.py中确认墙面并标定。
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


def write_json(path, value):
    """先写临时文件再替换，避免中断时留下半份批量索引。"""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def write_batch_reports(run_dir, items, started_at, state):
    """每完成一张就更新汇总，出错/中断时已完成的结果仍可找到。"""
    success = [item for item in items if item["detection_status"] == "success"]
    failed = [item for item in items if item["detection_status"] == "failed"]
    summary = {"run_directory": str(run_dir), "source_directory": str(IMAGE_DIR),
               "started_at": started_at, "updated_at": datetime.now().astimezone().isoformat(),
               "state": state, "image_count": len(items), "success_count": len(success),
               "failed_count": len(failed), "pending_count": len(items) - len(success) - len(failed),
               "model": MODEL_ID, "local_files_only": True,
               "parameters": {"box_threshold": BOX_THRESHOLD, "text_threshold": TEXT_THRESHOLD,
                              "nms_iou": NMS_IOU, "nested_ios": NESTED_IOS,
                              "nested_min_area_ratio": NESTED_MIN_AREA_RATIO,
                              "nested_center_fraction": NESTED_CENTER_FRACTION,
                              "do_resize": DO_RESIZE, "use_tiles": USE_TILES,
                              "include_full_image": INCLUDE_FULL_IMAGE, "tile_width": TILE_WIDTH,
                              "tile_height": TILE_HEIGHT, "tile_overlap": TILE_OVERLAP,
                              "tile_edge_margin": TILE_EDGE_MARGIN}, "items": items}
    write_json(run_dir / "batch_summary.json", summary)
    write_json(run_dir / "mapping_manifest.json", {
        "run_directory": str(run_dir), "state": state,
        "note": "2D predictions only; each image needs its own verified wall calibration. GT openings are not predictions.",
        "items": items})
    fields = ["image_key", "building_id", "detection_status", "window_count", "door_count",
              "ambiguous_count", "status", "result_directory", "error"]
    with (run_dir / "batch_summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(items)
    # 汇总CSV只收成功图片；失败不是“0个窗”，在batch_summary里明确记录。
    with (run_dir / "output_stage2.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["texture_filename", "bboxes_window", "bboxes_door"])
        writer.writeheader()
        for item in success:
            with (Path(item["result_directory"]) / "output_stage2.csv").open(encoding="utf-8", newline="") as source:
                writer.writerows(csv.DictReader(source))
    write_json(OUTPUT_ROOT / "latest_run.json", {"run_directory": str(run_dir), "state": state,
                                                "success_count": len(success), "failed_count": len(failed)})


def main():
    """批量入口：枚举照片 → 仅加载一次模型 → 循环检测 → 单图结果和全批汇总。"""
    images = find_images()
    model_dir = local_model_directory()
    # 必须在导入模型库前启用离线；每个from_pretrained还单独限制只读本地。
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    import transformers
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[1] Python: {sys.executable}\n    torch={torch.__version__}, transformers={transformers.__version__}", flush=True)
    print(f"    Found {len(images)} facade photos: {IMAGE_DIR}; device={device}", flush=True)
    print(f"[2] Loading LOCAL model ONCE (offline; no downloads): {model_dir}", flush=True)
    processor = AutoProcessor.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        str(model_dir), local_files_only=True, use_safetensors=True).to(device)
    model.eval()  # 评估模式；推理不会更新模型权重。

    started_at = datetime.now().astimezone().isoformat()
    # 微秒保证快速连续启动也不会覆盖同一目录，路径不含Windows不允许的冒号。
    run_dir = OUTPUT_ROOT / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True, exist_ok=False)
    items = [mapping_item(path, run_dir) for path in images]
    write_batch_reports(run_dir, items, started_at, "running")
    for index, (image_path, item) in enumerate(zip(images, items), 1):
        print(f"\n===== Image {index}/{len(images)}: {image_path.name} =====", flush=True)
        image = None
        try:
            image, report = detect_image(image_path, processor, model, device, model_dir)
            report.update({"building_id": item["building_id"], "source_type": item["source_type"],
                           "batch_run_directory": str(run_dir)})
            save_results(image, report, Path(item["result_directory"]))
            item.update({"detection_status": "success", "window_count": report["window_count"],
                         "door_count": report["door_count"], "ambiguous_count": report["ambiguous_count"]})
            # 模型文件关联尚需存在，之后仍要逐图标定，绝不自动宣称可直接写入3D。
            has_model = (item["citygml_path"] and Path(item["citygml_path"]).is_file()
                         and item["obj_path"] and Path(item["obj_path"]).is_file())
            item["status"] = "awaiting_calibration" if has_model else "needs_building_match"
        except Exception as error:
            # 一张损坏照片/显存不足不会把之前已完成的图片丢掉，也不会伪装成零检测。
            item.update({"detection_status": "failed", "status": "detection_failed",
                         "error": f"{type(error).__name__}: {error}"})
            print(f"FAILED {image_path.name}: {item['error']}", flush=True)
        finally:
            if image is not None:
                image.close()
        # 异常处理结束后释放临时引用；模型始终保留，下一张继续复用。
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        write_batch_reports(run_dir, items, started_at, "running")

    failures = sum(item["detection_status"] == "failed" for item in items)
    write_batch_reports(run_dir, items, started_at, "completed_with_errors" if failures else "completed")
    print(f"\nBatch finished: {len(items) - failures}/{len(items)} succeeded; {failures} failed.", flush=True)
    print(f"Results: {run_dir}\nSummary: {run_dir / 'batch_summary.csv'}", flush=True)
    print("Next: run map_facade_to_3d.py to confirm a wall, calibrate it, and preview predicted 3D openings.", flush=True)
    if failures:
        raise SystemExit(1)  # 有失败时不把整批标成成功；详情在batch_summary.csv。


# ==================== 程序入口 ====================
# 在 PyCharm 直接运行本文件时，__name__ 等于 "__main__"，于是执行 main()。
# 若其他脚本只是 import 本文件以研究 IoU/NMS，下面的 main() 不会自动执行。
if __name__ == "__main__":
    main()
