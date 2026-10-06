# 当前流程的安装与运行

当前统一入口已串联：实验 D 门窗检测、DINOv3 多建筑检索、RoMa 候选墙面匹配、自动检查与三维预测预览。已有 27 栋、216 张基础渲染图库；并非所有照片都能通过匹配检查。输出是独立预测面，尚未修改建筑、开孔或写出新的 LoD3 CityGML。操作和结果判读见 [自动流程教程](pipeline.md)。

本页负责环境与缓存准备；已经配置好的本机可以直接看 [PyCharm 手把手教程](pycharm_tutorial.md)，按图像路径、运行配置和结果文件逐步操作。

2026-10-06 本机与仓库副本各通过 100 项测试；17 张开发照片完整复跑为 3 张映射、14 张待复核、0 运行异常，全部复用检测缓存，新增付费请求 0 次。结果范围与逐张原因见 [实验记录](experiment_log.md#最终复跑17-张开发照片)，这些开发结果不保证新照片必然通过。

## 本机继续工作

已有环境 `torch_1` 为 Python 3.12.14 / PyTorch 2.8.0+cu126。保持原解释器即可。日常入口：

| 文件 | 作用 | 网络或费用 |
|---|---|---|
| `run_facade_pipeline.py` | 当前统一入口；一张照片自动检索建筑、匹配与三维预览 | 合格检测复用；新检测需要 Token 并收费；检索/匹配本地运行 |
| `try_grounding_dino16.py` | 实验D批量门窗检测 | 调用云API，消耗平台额度 |
| `build_render_gallery.py` | 遍历CityGML建立多建筑图库 | 本地CPU；不调用API |
| `render_building_views.py` | 单栋渲染及批量入口共用的几何实现 | 本地CPU |
| `match_facade_roma.py` | 固定建筑的历史对照入口；统一流程复用其 RoMa 推理/绘图函数 | 已准备缓存后本地GPU，不收费 |
| `map_facade_to_3d.py` | 历史人工标定与三维预览入口 | 当前自动匹配流程不要求运行它 |

旧 `detect_my_facade.py` 是已停用的Tiny版本。当前入口以上表为准；本次自动清理被执行策略阻止，旧文件尚未删除。

`my_results/`、数据、环境和模型缓存没有在这次提交中重复上传。现有数据由 Git LFS 管理。正式实验结果在本地保存，仓库 `docs/images/` 和 `docs/results/` 仅放代表图与精简记录。

## 在另一台Windows电脑首次准备

1. 取得仓库及Git LFS数据，确认存在 `Drills/Texture2LoD3_dataset/citygml` 和 `textures`，不能把Git LFS指针文本当成真实GML/照片。
2. 新建或选择Python 3.12环境。在仓库根目录安装已固定的依赖：

   ```powershell
   python -m pip install -r requirements-roma.lock.txt
   ```

   文件包含CUDA 12.6版PyTorch，适用于本项目验证过的NVIDIA环境；其他显卡环境需选择相应PyTorch构建。RoMa源码固定到官方提交 `95c9968145c8906b7b59383258e9f73b02853d89`，需要本机安装Git。

3. **仅第一次准备 RoMa 缓存**时，按下节在 PyCharm 中执行准备代码。它调用官方初始化，会下载约 1.1 GB 权重及固定版本的 DINOv3 源码，不调用门窗云 API。已准备好的本机无需重复下载。

4. 需要云端门窗检测时，在实际运行的脚本配置中添加 `DDS_API_TOKEN` 环境变量，值为自己的平台Token。别的脚本配置里的变量未必自动继承。不要写入代码、截图、Git提交或公开文档。统一入口在图片内容哈希、尺寸、模型和实验 D 参数一致时复用检测；存在同名文件本身不保证能复用。
5. 运行 `python -m unittest discover -s tests -v`。测试不调用收费API，合成测试文件写入临时目录，不混入正式结果。

## 在 PyCharm 中一次性准备完整模型缓存

先在“设置 → 项目 → Python 解释器”选中 Python 3.12 环境。本机解释器是 `C:\Users\YuexuanWang\.conda\envs\torch_1\python.exe`。若是在新电脑，先完成上面的依赖安装；在终端运行 `python -c "import sys; print(sys.executable)"`，确认 pip 所用解释器与 PyCharm 相同。

打开 PyCharm 底部的 **Python Console / Python 控制台**，复制下面整段。只需先把 `project_dir` 改为**你将要运行的 `run_facade_pipeline.py` 所在目录**。本机外层项目使用下面默认路径；如果运行内层仓库副本，改为 `E:\workinCODEX\prog_3D\CityGML-IMG_Architecture`。

```python
from pathlib import Path
import hashlib
import sys
import torch
from romav2 import RoMaV2

project_dir = Path(r"E:\workinCODEX\prog_3D")
assert (project_dir / "run_facade_pipeline.py").is_file(), "project_dir 不是脚本所在目录"
cache_dir = project_dir / "model_cache" / "roma_v2" / "hub"
cache_dir.mkdir(parents=True, exist_ok=True)
print("Python:", sys.executable)
print("PyTorch:", torch.__version__, "CUDA:", torch.cuda.is_available())
print("缓存目录:", cache_dir)

torch.hub.set_dir(str(cache_dir))
torch.set_float32_matmul_precision("highest")
model = RoMaV2(RoMaV2.Cfg(setting="base", compile=False))

checkpoint = cache_dir / "checkpoints" / "romav2.0.1.pt"
with checkpoint.open("rb") as stream:
    digest = hashlib.file_digest(stream, "sha256").hexdigest()
expected = "1557dec0d21b62366465f7ff4d5fdf228cc695d0582e196ad2b80e05230828b7"
assert digest == expected, "权重 SHA256 不符，需要检查下载文件"
backbone = cache_dir / "facebookresearch_dinov3_adc254450203739c8149213a7a69d8d905b4fcfa"
assert (backbone / "hubconf.py").is_file(), "缺少固定版本的 DINOv3 源码缓存"
del model
if torch.cuda.is_available():
    torch.cuda.empty_cache()
print("RoMa 权重和 DINOv3 源码缓存已准备好:", digest)
```

这段代码只初始化和核验模型，不检测照片、不调用收费 API。下载失败时先看具体网络异常；本机此前的代理故障通过关闭有问题的 VPN 解决。准备成功后可以关闭 Python 控制台，再正常运行统一入口。

完整目录至少包含两部分，不能只保留 `.pt` 文件：

```text
model_cache/roma_v2/hub/
├── checkpoints/romav2.0.1.pt
└── facebookresearch_dinov3_adc254450203739c8149213a7a69d8d905b4fcfa/
    ├── hubconf.py
    └── dinov3/ 及其余源码文件
```

官方来源：[RoMa v2](https://github.com/Parskatt/RoMaV2)、[DINOv3](https://github.com/facebookresearch/dinov3)。统一入口日常读取这些本地缓存，DINOv3 检索骨干也从同一权重恢复。跨电脑或跨项目目录搬迁时，可复制这份完整 `hub` 文件夹，再按相同路径核验；不要只复制网络模型权重而漏掉骨干源码。

## 新机器上建议的执行次序

1. `build_render_gallery.py` 建立多建筑图库，完成后查看 `my_results/render_gallery/latest_run.json` 指向的总览与索引。
2. 打开 `run_facade_pipeline.py`，设置 `PHOTO_PATH` 并运行。源码默认保留熟悉的 `4959323_front.jpg` 对照图，它在本轮最终复跑中返回 `needs_review`。第一次想观察已有成功样例，可主动改为 `DATASET / "textures/4907507.png"`；这不代表其他照片也会通过。
3. 检测没有合格缓存时，统一入口会调用实验 D 的检测函数；无需先运行全部 17 张的独立收费检测脚本。已配置的本机可复用完整 D 批次，新机器若未复制这批结果则需为新检测准备 Token。
4. 读取 `my_results/facade_pipeline/latest_run.json` 指向的 `result.json`；只有自动选择通过才进入三维投影。`mapped` 看预览和被拒绝框，`needs_review` 看候选检查与对齐图。完整文件说明见 [pipeline.md](pipeline.md)。

多建筑图库使用 `view_000` 等自动方向命名，统一入口通过检索和候选匹配决定建筑、视角和墙面，不把某个方向名称直接当作照片正面。

所有缓存和输出以脚本所在目录为根。外层工作目录与内层 Git 仓库副本不是同一个输出目录；复制脚本不会自动复制模型、检测结果与图库。

要批量处理数据集，在同一脚本的 PyCharm 运行配置“脚本参数 / Parameters”中填写：

```text
--photo-dir "E:\workinCODEX\prog_3D\CityGML-IMG_Architecture\Drills\Texture2LoD3_dataset\textures"
```

其他电脑请换成自己的 `textures` 路径；仓库根目录终端也可运行 `python run_facade_pipeline.py --photo-dir "Drills/Texture2LoD3_dataset/textures"`。不要传 `gt_masks` 或全景目录，不同时传 `--photo`。每张结果立即写入 `my_results/facade_pipeline/batch_时间/summary.json`，异常记录后继续下一张；清空参数恢复默认单图。状态与费用说明见 [批量模式](pipeline.md#批量模式)。

## 保留的历史对照入口

若要重现固定建筑的单图对照，先运行 `render_building_views.py` 准备 `my_results/rendered_buildings/4959323` 的 `front`，再运行 `match_facade_roma.py`。这条旧主流程预先指定建筑与视角，不能用来评价自动找楼；当前统一入口不要求先执行它。`map_facade_to_3d.py` 的旧人工标定流程也不是统一入口的前置步骤。
