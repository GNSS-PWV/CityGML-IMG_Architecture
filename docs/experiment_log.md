# 多建筑渲染图库、门窗检测与照片匹配：实验记录

更新日期：2026-10-05。当前新增成果是 **27 栋已有 LoD3 建筑的多视角渲染图库**。本记录区分实际完成的输出、尚未通过质量检查的试验，以及后续研究任务；原项目的建筑检索目标没有改变。

## 1. 当前进度

| 内容 | 实际结果 | 不能据此推出的结论 |
|---|---|---|
| 多建筑图库 | 27/27 栋成功、0 失败；每栋 8 视角，共 216 张 1800×1200 图片 | 不能推出照片已匹配到正确建筑 |
| 实验 D 门窗检测 | 17/17 张成功；397 个窗、27 个门、2 个待确认 | 预测数不是准确率，也没有跨照片去重 |
| RoMa 单图匹配 | 本地真实 GPU 推理与自动检查流程已跑通，2026-10-05 已重新生成结果，无需人工点选 | 预先给定建筑与视角；本轮自动质量检查仍未通过 |
| 历史三维预览 | 4959323 正面已有人工标定及 54 个预测四边形 | 不是全批自动标定，也不是完成墙面开孔或 LoD3 写入 |

![多建筑渲染总览，每栋只显示一个方向](./images/多建筑总览.png)

总览从每栋已有 8 个视图中选取门窗可见像素最多的方向，来源记录在本地批次的 `overview_metadata.json`；原始 216 张图未改变。216 组 PNG/NPZ/相机已逐项验证尺寸、正深度、构件与墙面统计、源模型身份及相机往返，本地和仓库副本均通过 67 项测试。[图库摘要](./results/gallery_summary.json)、[图库验证记录](./results/gallery_validation.json)、[实验 D 汇总](./results/detection_D_summary.csv) 与 [自动匹配摘要](./results/automatic_match_summary.json) 保存了可复查的轻量结果。

**这张图里的门窗是已有 LoD3 模型的几何。** 本次没有把实验 D 预测贴上去，也没有给模型贴入真实照片。当前数据副本缺少 GML 引用的纹理资源，因此渲染使用模型颜色与默认材质。已有模型的门窗不能算成本次从照片重建的成果。

## 2. 代码入口与模型分工

| 文件 | 用途 | 使用什么模型或方法 |
|---|---|---|
| [build_render_gallery.py](../build_render_gallery.py) | 批量生成多栋建筑图库 | 调用单栋渲染器；不使用神经网络 |
| [render_building_views.py](../render_building_views.py) | 解析并渲染一栋 CityGML，保存相机与深度 | 多边形三角化、CPU 深度缓冲、正交相机 |
| [try_grounding_dino16.py](../try_grounding_dino16.py) | 实验 D 批量门窗检测 | Grounding DINO 1.6 Pro 云端 API |
| [facade_geometry.py](../facade_geometry.py) | 坐标还原、去重、门窗冲突与模型关联 | 本地几何规则；不加载检测网络 |
| [match_facade_roma.py](../match_facade_roma.py) | 一张照片与指定建筑渲染图自动匹配 | 官方 RoMa v2.0.1 本地预训练模型 |
| [facade_match_geometry.py](../facade_match_geometry.py) | 几何变换与自动检查 | MAGSAC/RANSAC、空间留出、覆盖及变换合法性检查 |
| [facade_match_structure.py](../facade_match_structure.py) | 照片预测窗中心与模型窗中心的自动细化 | RoMa 初值引导的结构对应及自动验收 |
| [map_facade_to_3d.py](../map_facade_to_3d.py) | 保留的人工标定与三维预览工具 | 照片 XY → 墙面 UV → 模型 XYZ |

`detect_my_facade.py` 是 **已弃用的 Tiny 历史入口**，当前不应再用它执行实验 D。历史文件目前仍保留，不能把“弃用”写成“已删除”。当前自动匹配入口也不会调用人工点选流程。

Grounding DINO 负责检测“哪里有门窗”，RoMa 负责找“两张图的哪些位置相对应”；RoMa 内部使用 DINOv3 特征。后续计划的 DINO 图像检索负责先挑选候选建筑图像，这项功能尚未实现。

参考项目 [chrise96/3D_building_reconstruction](https://github.com/chrise96/3D_building_reconstruction) 的 Stage 2 使用 Detectron2 Mask R-CNN、R50-FPN 和 sky/window/door 专用权重；目前取得的参考材料没有其 `model_output/model.pth`。本项目改用公开模型或云端模型推进数据适配，没有声称复现原作者的同一套权重与精度。渲染、批次管理、自动几何检查及结构细化属于本项目实现；RoMa 神经网络来自 [官方 RoMaV2](https://github.com/Parskatt/RoMaV2)。

## 3. 批量建立渲染图库

### 输入和运行

数据集模型应位于仓库的 `Drills/Texture2LoD3_dataset/citygml/`，包含实际 GML 内容，不能只是 Git LFS 指针。当前工作目录外层布局也可由批量入口识别。

在 PyCharm 选择现有 Python 3.12 环境，打开 `build_render_gallery.py`，右键运行，或在仓库目录执行：

```bash
python build_render_gallery.py
```

渲染依赖 NumPy、Pillow、mapbox-earcut 和 Shapely；不需要 DDS Token、RoMa 权重或模型推理 GPU。本机现有环境已配置。首次在新机器运行时，先按 [环境与资源准备说明](./setup.md) 准备依赖和完整数据。

默认配置：

```python
BUILDING_IDS = None
IMAGE_SIZE = (1800, 1200)
MAX_WORKERS = 2
REUSE_COMPLETED = True
```

`None` 遍历全部模型；试跑部分建筑可设为 `["4906970", "4959323"]`。默认同时处理两栋，内存紧张可把 `MAX_WORKERS` 改为 1。

程序每栋完成后更新索引和汇总。再次运行会核对源模型哈希、渲染代码哈希、图像尺寸、完成状态以及各视角文件；满足条件才复用。缺相机、深度、图片或总览的结果不会被当成完整缓存。每次会新建图库汇总，不代表每次都重新渲染全部模型。

### 本次结果和目录

实际批次为 `gallery_20261005_212322_924197`，汇总状态 `complete`，27 栋完成、0 失败、216 个视图。

```text
my_results/render_gallery/
├── latest_run.json
├── gallery_时间/
│   ├── overview.png
│   ├── gallery_index.json
│   ├── batch_summary.csv
│   └── batch_summary.json
└── buildings/
    └── 建筑编号/
        ├── latest_run.json
        └── render_时间/
            ├── manifest.json
            ├── overview.png
            ├── view_000.png
            ├── view_000_camera.json
            ├── view_000_geometry.npz
            └── 其他7个视角的配套文件
```

逐栋状态和错误查看 `batch_summary.csv/json`。`latest_run.json` 仅记录最新路径，是否全部成功仍须看汇总。失败时保留已成功建筑，修复后可复用成功输出继续构建新的图库批次。

每栋自动选择有效的大型竖直墙面作为起始参考，以相对 0°、90°、180°、270° 生成四个平视方向，再以 45°、135°、225°、315° 和 22° 相机高度角生成四个斜视方向。`view_000` 不是人工确认的照片正面，也不是固定北面。

这 8 个方向是基础图库，不保证每个复杂墙面都获得最佳正视图。内凹结构、狭窄侧墙和遮挡区域，后续可以继续按墙面补充正视图。

### 为什么不能只保存 PNG

`gallery_index.json` 为每张图保存建筑 ID、视角、图像/相机/几何文件路径、模型指纹、坐标系及 `world_origin_m`。`visible_walls` 按原模型 `wall_id` 聚合可见墙面与关联构件，并记录像素数量。

相机 JSON 保存投影与逆变换；几何 NPZ 保存 `depth_m` 和 `object_index`。深度是可见表面沿相机观察轴的米制距离，不是建筑高度。后续图像匹配得到渲染图上的点后，需要相机、深度和局部原点才能找回三维坐标，故三类文件必须配套保留。

## 4. 实验 D 的批量门窗检测

本次有效完整批次为 `D_20260930_155603_929347`。已核对其 `batch_summary.csv`：17 张成功、0 失败，完整新跑需要 28 个图块任务。下表是去重后的预测数量，不是标注真值或准确率。

| 照片 | 窗 | 门 | 待确认 |
|---|---:|---:|---:|
| 4906970 | 7 | 6 | 0 |
| 4906972_front_01 | 2 | 0 | 1 |
| 4906972_front_02 | 23 | 0 | 0 |
| 4906972_front_03 | 3 | 1 | 0 |
| 4906981 | 58 | 6 | 0 |
| 4907506 | 41 | 2 | 0 |
| 4907507 | 35 | 1 | 0 |
| 4907514 | 21 | 1 | 0 |
| 4907518_front | 15 | 0 | 1 |
| 4907518_left | 19 | 2 | 0 |
| 4907518_right | 17 | 1 | 0 |
| 4907520_front | 20 | 2 | 0 |
| 4907520_right | 21 | 2 | 0 |
| 4959322 | 20 | 1 | 0 |
| 4959323_front | 65 | 0 | 0 |
| 4959323_left | 14 | 1 | 0 |
| 4959323_right | 16 | 1 | 0 |
| **合计** | **397** | **27** | **2** |

主要配置为 `GroundingDino-1.6-Pro`，提示 `window.door.`，框阈值 0.20，分块最大 1200×1000，重叠 300 px，内部切边过滤 12 px，云端 IoU 0.8。跨块重复框由本地几何规则处理，门窗类别冲突保留为 `ambiguous`，不强行当作窗或门。

要重新做检测，在 PyCharm 的该脚本运行配置里设置自己的 `DDS_API_TOKEN`，然后运行：

```bash
python try_grounding_dino16.py
```

Token 不写进代码或 Git。此脚本会上传照片并使用平台额度；图库生成和已有检测结果的匹配不会重新调用它。当前 `RESUME_RUN_DIR=None` 表示新建收费批次，从第一张开始；每张完成后立即保存，网络/API 出错时停止后续提交，已有结果保留。

结果位于 `my_results/grounding_dino16/batch_detection/D_时间/`。先看总表和每图 `detections.png`；`detections.json` 保存原图像素框及模型来源，供匹配与映射读取；`mapping_manifest.json` 只关联已知照片和模型身份，不代表已经完成墙面标定。

ABCD 对照实验及莫兰迪汇总图保留用于回看参数变化。数量增加不能单独证明准确率提高，仍需配准后的误检、漏检和类别评价。51 张全景图和 `gt_masks` 不作为本次门窗检测输入。

## 5. 无需人工点选的单图 RoMa 匹配

当前单图入口 `match_facade_roma.py` 使用本地 RoMa v2.0.1，流程为：读取指定照片及已有单栋正面渲染 → 裁剪目标立面 → RoMa 初始对应 → RANSAC 与空间留出检查 → 已有预测窗中心/模型窗中心辅助细化 → 自动选择通过门槛的变换。

在已准备好官方源码、完整权重缓存和单栋配套渲染的环境中运行：

```bash
python match_facade_roma.py
```

本机沿用升级后的 `torch_1`，Python 3.12.14、PyTorch 2.8.0+cu126。RoMa 源码及模型缓存需要单独准备，不随普通 Git 源码提交完整权重；默认入口只读取已缓存资产，缺失时明确报错，不自动重新下载。

默认不读取人工标定文件，不调用 `check_against_manual()`，也不要求用户点对应点。它可以读取实验 D 最新完整批次中的窗检测结果并校验输入身份，不重新执行收费检测。

**当前建筑 4959323 和正面视角仍然预先选定。** 新增图库没有让这个脚本自动比较 27 栋，也没有把任意 `view_000` 自动当作照片的正面。DINO 检索与 RoMa 多候选选择尚未实现。

每次输出到 `my_results/roma_matching/4959323_front/match_时间/`：

- `quality_report.txt`：自动通过/失败及原因。
- `matches.png`：最终选中方法的对应点。
- `roma_matches.png`：纯 RoMa 对照。
- `alignment.png`：最终变换的叠加效果。
- `result.json`：`selected_method`、`automatic_validation`、`auto_accepted` 与状态。
- `run_info.json`、CSV 和 NPZ：输入指纹、参数、原始对应及几何掩码。

`auto_checks_passed` 表示通过自动几何门槛，`auto_checks_failed` 表示未通过；前者仍不是独立真实定位精度证明。规则窗列可能整体错位，不能凭连线多或叠加图看起来接近就报告匹配正确。

2026-10-05 新的真实 GPU 单图试验为 `match_20261005_214050_135840`：纯 RoMa 留出内点比例 42.04%，低于 50% 门槛；结构候选为 69.565%，低于 70%，相对初值的全图最大修正 30.0496 px 也超过 30 px 上限，最终未采用结构候选。最终 `selected_method="roma"`、`auto_checks_failed`、`auto_accepted=false`。本次没有读取人工标定或调用检测 API，结果摘要见 [automatic_match_summary.json](./results/automatic_match_summary.json)。

![本次自动匹配叠加，自动检查未通过](./images/自动匹配叠加.png)

这张图用于观察偏差，不代表自动验收通过。此前未保留的旧匹配目录不再作为本次结果链接。

另一次旧结构分析与人工参考点比较的最大偏差约 0.435 m，使用不同检测输入和评价方式，不能混作当前自动试验的米制精度。当前自动模式没有独立真值验证，不会通过放宽门槛把失败改写为成功。

## 6. 下一步与当前限制

1. 利用 DINO 类图像特征建立图库检索，给一张照片返回少量候选建筑与视角。
2. 对候选分别运行 RoMa，检查几何一致性、覆盖范围和重复窗列错位，再决定建筑及墙面。
3. 利用相机/深度/墙面参数，将检测门窗框转换到确认的三维墙面。
4. 检查几何和类别后，再实现墙体开孔、门窗构件及 CityGML 写入与有效性检查。

上述检索、候选选择与实体修改尚未完成。图库基于含门窗的 LoD3；如果最终输入只有 LoD1/LoD2 外壳，必须另外验证缺少门窗几何时的检索和匹配表现。

## 7. 代码、结果与历史文件的保留原则

保留完整数据与权重、实验 D 完整结果、ABCD 对照、有效三维输出、当前渲染图库、正式匹配代码及测试、环境恢复记录。大数据、权重、环境和全套渲染产物不直接提交普通 Git；仓库保存代码、说明、必要的轻量图例与测试。

旧 Tiny 入口和一次性调试文件仍可能出现在本地或仓库中，已按用途标为历史内容。清理计划不等于已经删除；本轮文件删除操作尚未执行。当前运行入口以第 2 节为准。

## 8. 历史概要：Tiny 检测与人工三维预览

2026-09-29 的 Tiny 批次为 `run_20260929_160455_549558`，17 张照片处理成功，预测 349 个窗、12 个门、29 个待确认。它使用公开 `IDEA-Research/grounding-dino-tiny` 权重，未在当前数据上重新训练。该批次已被实验 D 取代作为日常检测入口，数量比较不能代替准确率对比。

2026-09-30 对 `4959323_front` 进行了历史人工标定：5 对拟合点、1 对额外检查点，拟合 RMSE 0.056262 m，检查点误差 0.169840 m；54 个预测框完成三维映射，其中 50 个窗、4 个待确认。另 6 个框超出当时圈定的照片墙面区域而未映射，不能因此算作二维漏检。

![历史人工标定的三维预测预览](./images/facade-4959323-front-3d.png)

这些数值只描述历史人工参考的一致性，不是整栋建筑的独立重建精度。`map_facade_to_3d.py` 保留该人工交互与预览功能；自动匹配不要求执行该点选流程。图中楼体来自已有模型，独立预测 OBJ 仅包含预测四边形，原始 CityGML 未开孔或写入新门窗。
