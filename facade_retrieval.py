"""DINOv3 图像检索：从已下载的 RoMa checkpoint 严格恢复完整预训练骨干。

只比较图像特征，不从照片文件名提取建筑编号。建筑 ID 仅来自图库元数据，
用于把视觉排序结果按建筑分组。真实照片与无纹理渲染跨域，cosine 不是正确概率；
候选还必须经过 RoMa/几何验证。本模块不调用云端 API，也不下载任何权重。

来源：缓存的官方 DINOv3 hub/backbones.py、models/vision_transformer.py；
RoMaV2/src/romav2/features.py 的默认骨干为 dinov3_vitl16，完整权重在 f.*。
"""
from contextlib import nullcontext
from pathlib import Path
import gc
import hashlib
import json

import numpy as np
from PIL import Image

DINOV3_COMMIT = "adc254450203739c8149213a7a69d8d905b4fcfa"
CHECKPOINT_SHA256 = "1557dec0d21b62366465f7ff4d5fdf228cc695d0582e196ad2b80e05230828b7"
PREPROCESS_VERSION = "v1-depth-bbox-pad2pct-photo-full-rgb-bicubic-imagenet-cls-l2"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unit_rows(values):
    """余弦比较前做 L2 归一化；拒绝坏缓存和零向量。"""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("描述子必须是有限二维数组")
    lengths = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(lengths < 1e-12):
        raise ValueError("描述子包含零向量")
    return values / lengths


def foreground_box(depth, margin_fraction=0.02):
    """仅依据渲染深度裁掉留白，不用照片标签或门窗预测控制检索。"""
    depth = np.asarray(depth)
    if depth.ndim != 2:
        raise ValueError("渲染深度必须是二维数组")
    mask = np.isfinite(depth) & (depth > 0)
    ys, xs = np.where(mask)
    if not len(xs):
        raise ValueError("渲染图没有有效前景深度")
    height, width = mask.shape
    pad = max(1, int(np.ceil(max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1) * margin_fraction)))
    return (max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad),
            min(width, int(xs.max()) + 1 + pad), min(height, int(ys.max()) + 1 + pad))


def read_rgb(path, geometry=None):
    """照片保留全图；模型图按自己的深度前景裁剪，返回原图裁剪范围。"""
    with Image.open(path) as source:
        image = source.convert("RGB")
    box = (0, 0, image.width, image.height)
    if geometry is not None:
        with np.load(geometry, allow_pickle=False) as arrays:
            depth = arrays["depth_m"]
            if depth.shape != (image.height, image.width):
                raise ValueError("图片与深度尺寸不一致：" + str(path))
            box = foreground_box(depth)
        image = image.crop(box)
    return image, list(box)


def rank_descriptors(query, descriptors, views, top_k_buildings=5, views_per_building=2):
    """纯视觉排序，再按建筑分组保留候选；不接受查询建筑 ID 作为参数。"""
    if top_k_buildings < 1 or views_per_building < 1 or not views:
        raise ValueError("图库和 Top K 参数必须非空/为正")
    query = unit_rows(np.asarray(query).reshape(1, -1))[0]
    descriptors = unit_rows(descriptors)
    if descriptors.shape != (len(views), len(query)):
        raise ValueError("图库记录与描述子尺寸不一致")
    scores = descriptors @ query
    ranked = []
    for rank, index in enumerate(np.argsort(-scores, kind="stable"), 1):
        row = dict(views[int(index)])
        row.update(retrieval_rank=rank, cosine_similarity=float(np.clip(scores[index], -1, 1)))
        ranked.append(row)
    selected, building_order, counts = [], [], {}
    for row in ranked:
        key = str(row["building_id"])
        if key not in counts:
            if len(building_order) == top_k_buildings:
                continue
            building_order.append(key)
            counts[key] = 0
        if counts[key] < views_per_building:
            counts[key] += 1
            selected.append(dict(row, building_retrieval_rank=building_order.index(key) + 1))
    # 每栋相邻，便于父流程按栋尝试多个视角；visual rank 仍在每条记录里。
    selected.sort(key=lambda row: (row["building_retrieval_rank"], row["retrieval_rank"]))
    return {"ranked_views": ranked, "selected_candidates": selected,
            "ranked_building_ids": building_order}


def cache_key(records, model_identity, image_size):
    identity = {"schema_version": 1, "model": model_identity,
                "preprocess": PREPROCESS_VERSION, "image_size": image_size, "images": records}
    payload = json.dumps(identity, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), identity


def model_identity(model_cache):
    hub = Path(model_cache)
    checkpoint = hub / "checkpoints/romav2.0.1.pt"
    source = hub / ("facebookresearch_dinov3_" + DINOV3_COMMIT)
    if not checkpoint.is_file() or not (source / "hubconf.py").is_file():
        raise FileNotFoundError("需要已有 RoMa 权重和固定 DINOv3 源码缓存，不会自动下载：" + str(hub))
    digest = sha256(checkpoint)
    if digest != CHECKPOINT_SHA256:
        raise ValueError("RoMa checkpoint SHA256 不符，拒绝使用未知/残缺权重")
    code = hashlib.sha256()
    for path in sorted(source.rglob("*.py")):
        code.update(path.relative_to(source).as_posix().encode("utf-8"))
        code.update(path.read_bytes())
    return {"name": "DINOv3 ViT-L/16 from RoMa v2.0.1 f.*", "checkpoint_sha256": digest,
            "checkpoint_size_bytes": checkpoint.stat().st_size, "source_commit": DINOV3_COMMIT,
            "source_sha256": code.hexdigest(), "descriptor": "final normalized CLS token, L2 normalized"}


class DinoV3Encoder:
    """只加载 f.*：不加载 RoMa 匹配器，严格检查所有预训练骨干参数。"""
    def __init__(self, model_cache, image_size=384, device=None):
        import torch
        # hubconf 会导入 torchvision；先在真实 CPU 上初始化其注册逻辑，不能放进 meta 上下文。
        import torchvision  # noqa: F401
        self.torch = torch
        self.image_size = int(image_size)
        if self.image_size < 16 or self.image_size % 16:
            raise ValueError("DINO 输入尺寸应为16的正整数倍")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        hub = Path(model_cache)
        source = hub / ("facebookresearch_dinov3_" + DINOV3_COMMIT)
        # meta 只建立网络结构、不初始化真实随机权重；随后把 checkpoint 的完整 f.* 赋进去。
        with torch.device("meta"):
            self.model = torch.hub.load(str(source), "dinov3_vitl16", source="local",
                                        pretrained=False, weights=None)
        checkpoint = torch.load(hub / "checkpoints/romav2.0.1.pt", map_location="cpu",
                                weights_only=True, mmap=True)
        state = {key[2:]: value for key, value in checkpoint.items() if key.startswith("f.")}
        self.model.load_state_dict(state, strict=True, assign=True)
        if any(value.is_meta for value in list(self.model.parameters()) + list(self.model.buffers())):
            raise RuntimeError("有未恢复的 DINO 参数/缓冲，拒绝生成检索特征")
        self.loaded_key_count = len(state)
        self.dtype = torch.bfloat16 if self.device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
        self.model = self.model.to(device=self.device, dtype=self.dtype).eval()
        self.model.requires_grad_(False)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device)[None, :, None, None]
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device)[None, :, None, None]

    def __call__(self, images):
        # 官方 DINOv3 使用平方 Resize 和 ImageNet 归一化；此处固定 Pillow bicubic，缓存记录版本。
        torch = self.torch
        pixels = np.stack([np.asarray(image.resize((self.image_size, self.image_size), Image.Resampling.BICUBIC),
                                      dtype=np.float32) / 255 for image in images])
        tensor = torch.from_numpy(pixels).permute(0, 3, 1, 2).to(self.device)
        tensor = (tensor - self.mean) / self.std
        amp = torch.autocast("cuda", dtype=self.dtype) if self.dtype == torch.bfloat16 else nullcontext()
        with torch.inference_mode(), amp:
            # forward_features 不依赖 RoMa 对 forward() 的临时包装。
            features = self.model.forward_features(tensor)["x_norm_clstoken"]
        return unit_rows(features.float().cpu().numpy())

    def close(self):
        self.model = None
        self.mean = self.std = None
        gc.collect()
        if self.device.type == "cuda":
            self.torch.cuda.empty_cache()


def retrieve_gallery(photo_path, gallery_index_path, cache_dir, model_cache, *,
                     top_k_buildings=5, views_per_building=2, image_size=384, batch_size=4,
                     device=None, encoder=None, progress=print):
    """返回所有视图排序及 Top K 栋×每栋 N 视图；父流程负责保存报告/做 RoMa。

    如传入 encoder(images)->L2描述子，则调用者负责释放；默认内部创建并释放 DINO。
    缓存绑定图片、裁剪深度、官方权重、源码和预处理版本。相同照片文件内容改名不会改变排序。
    """
    photo_path, gallery_index_path = Path(photo_path), Path(gallery_index_path)
    index = json.loads(gallery_index_path.read_text(encoding="utf-8-sig"))
    if index.get("status") != "complete" or not index.get("views"):
        raise ValueError("请使用完整完成的 gallery_index.json")
    if batch_size < 1:
        raise ValueError("batch_size 必须大于0")
    identity = model_identity(model_cache)
    views, records = [], []
    for original in index["views"]:
        row = dict(original)
        for name in ("image", "geometry", "camera", "manifest"):
            if name in row:
                path = Path(row[name])
                row[name] = str(path.resolve() if path.is_absolute() else (gallery_index_path.parent / path).resolve())
        views.append(row)
        records.append({"image": row["image"], "image_sha256": sha256(row["image"]),
                        "geometry_sha256": sha256(row["geometry"])})
    key, cache_identity = cache_key(records, identity, image_size)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / ("dinov3_" + key + ".npz")
    descriptors, crops, cache_hit = None, [], False
    if path.exists():
        try:
            with np.load(path, allow_pickle=False) as cached:
                saved_identity = json.loads(str(cached["identity"].item()))
                if saved_identity == cache_identity:
                    descriptors = unit_rows(cached["descriptors"])
                    crops = cached["crop_boxes"].tolist()
                    if descriptors.shape[0] != len(views) or len(crops) != len(views):
                        raise ValueError("缓存图像数量不符")
                    cache_hit = True
        except (OSError, ValueError, KeyError):
            descriptors, crops = None, []
    own_encoder = encoder is None
    if own_encoder:
        encoder = DinoV3Encoder(model_cache, image_size=image_size, device=device)
    try:
        if descriptors is None:
            batches = []
            for start in range(0, len(views), batch_size):
                images = []
                for row in views[start:start + batch_size]:
                    image, crop = read_rgb(row["image"], row["geometry"])
                    images.append(image)
                    crops.append(crop)
                batches.append(unit_rows(encoder(images)))
                if progress:
                    progress(f"[检索] DINO图库特征 {min(start+batch_size, len(views))}/{len(views)}")
            descriptors = np.concatenate(batches)
            temporary = path.with_suffix(".tmp.npz")
            np.savez_compressed(temporary, descriptors=descriptors, crop_boxes=np.asarray(crops),
                                identity=np.array(json.dumps(cache_identity, ensure_ascii=False)))
            temporary.replace(path)
        photo, _ = read_rgb(photo_path)
        query = unit_rows(encoder([photo]))[0]
        result = rank_descriptors(query, descriptors, views, top_k_buildings, views_per_building)
        result.update(model=identity, preprocessing={"version": PREPROCESS_VERSION, "image_size": image_size,
                      "render_crop": "depth foreground bounding box + 2% margin", "photo_crop": "full image"},
                      query_image_sha256=sha256(photo_path), gallery_index_sha256=sha256(gallery_index_path),
                      gallery_view_count=len(views), gallery_building_count=len({str(v["building_id"]) for v in views}),
                      descriptor_cache=str(path), cache_hit=cache_hit, ranking_uses_query_filename=False,
                      interpretation="cosine仅作跨域候选检索分数；不是正确概率，须继续RoMa/几何验证。")
        return result
    finally:
        if own_encoder:
            encoder.close()
