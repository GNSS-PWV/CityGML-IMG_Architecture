"""MINIMA-RoMa 适配器；只读取本地官方权重，不触发隐式网络下载。

MINIMA 的 RoMa 分支基于官方随附的 RoMa_minima 代码和 DINOv2 主干，
并非 RoMa v2 的另一套参数。本模块只将它的匹配输出转成现有几何检查所用的
[-1, 1] 对应点坐标，后续筛选、照片到三维墙面的映射共用同一套代码。
"""

from pathlib import Path
import sys

import numpy as np

from match_facade_roma import sha256, code_sha256

ROOT = Path(__file__).resolve().parent
CACHE = ROOT.parent / "model_cache/minima"
SOURCE = CACHE / "unpacked/MINIMA-main/third_party/RoMa_minima"
CHECKPOINT = CACHE / "unpacked/MINIMA-main/weights/minima_roma.pth"
DINO_CHECKPOINT = CACHE / "dinov2_vitl14_pretrain.pth"
SETTING = "official_560_864"
NUM_MATCHES = 2000
MIN_OVERLAP = .20
RANSAC_THRESHOLD_PX = 4.0
SEED = 42
# 官方 release 文件大小；截断的文件绝不能被 torch.load 当作可用权重。
EXPECTED_BYTES = {CHECKPOINT: 445638463, DINO_CHECKPOINT: 1217586395}


def load_roma_model():
    import torch

    if not SOURCE.is_dir():
        raise FileNotFoundError(f"缺少官方 MINIMA/RoMa_minima 源码：{SOURCE}")
    for path, size in EXPECTED_BYTES.items():
        if not path.is_file() or path.stat().st_size != size:
            raise FileNotFoundError(f"缺少或未下载完整的本地模型权重：{path}（应为 {size} 字节）")
    if str(SOURCE) not in sys.path:
        sys.path.insert(0, str(SOURCE))
    from romatch import roma_outdoor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
        torch.cuda.reset_peak_memory_stats()
    # 两个 state_dict 均从显式本地文件读入；否则 roma_outdoor 会下载 DINOv2。
    weights = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    dino_weights = torch.load(DINO_CHECKPOINT, map_location="cpu", weights_only=True)
    model = roma_outdoor(device=device, weights=weights, dinov2_weights=dino_weights,
                         coarse_res=560, upsample_res=864).eval()
    del weights, dino_weights
    return model, {
        "name": "MINIMA-RoMa", "official_source": "https://github.com/LSXI7/MINIMA",
        "source_dir": str(SOURCE), "checkpoint_sha256": sha256(CHECKPOINT),
        "dinov2_checkpoint_sha256": sha256(DINO_CHECKPOINT),
        "setting": SETTING, "num_matches": NUM_MATCHES, "seed": SEED,
        "device": str(device), "torch_version": torch.__version__,
        "input_resize": [560, 560], "high_resolution_resize": [864, 864],
        "input_coordinates": "normalized align_corners=False",
        "implementation": {"adapter_sha256": sha256(Path(__file__)),
                           "romatch_source_sha256": code_sha256(SOURCE),
                           "torch": torch.__version__},
    }


def infer_pair(model, photo, crop_image):
    import torch

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    with torch.inference_mode():
        dense_matches, certainty = model.match(photo, crop_image)
        matches, scores = model.sample(dense_matches, certainty, num=NUM_MATCHES)
    matches = matches.float().cpu().numpy()
    scores = scores.float().cpu().numpy()
    if matches.ndim != 2 or matches.shape[1] != 4 or scores.shape != (len(matches),):
        raise ValueError("MINIMA-RoMa 返回的对应点尺寸不符合几何检查接口。")
    return matches, scores
