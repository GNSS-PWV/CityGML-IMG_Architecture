# 一张照片自动找建筑、匹配墙面并生成三维门窗预览

现在的统一入口是 [run_facade_pipeline.py](../run_facade_pipeline.py)。在 PyCharm 中运行它，程序会依次准备门窗检测、检索候选建筑、匹配候选墙面，并在自动检查通过后生成三维预测预览。换照片时，主要修改 `PHOTO_PATH` 一处。

这一步用已有三维模型作为参考。当前渲染图库中的门窗来自数据集原有 LoD3 模型；照片中的门窗框来自实验 D 检测。最终输出的是检测框投影后的独立预测面，源 CityGML 不会被修改，也没有给建筑开孔。

## 1. 先分清三个模型各自做什么

| 模型 | 本程序让它做的事 | 输入与输出 | 运行位置 |
|---|---|---|---|
| Grounding DINO 1.6 Pro | 在照片里找窗和门 | 照片图块 → 类别、评分、二维框 | DeepDataSpace 云端；新检测消耗额度 |
| DINOv3 ViT-L/16 | 从渲染图库里找外观比较相近的建筑 | 照片和渲染图 → 特征向量、相似度排序 | 本地，复用已有 RoMa 缓存里的骨干权重 |
| RoMa v2.0.1 | 在照片与候选渲染图之间寻找对应位置 | 两张图 → 一批候选对应点 | 本地，使用已下载的权重 |

Grounding DINO 和 DINOv3 的名字相似，但这里承担的任务不同。第一步回答“窗和门在哪里”，第二步回答“先尝试匹配哪几栋”，第三步回答“两张图里的哪些位置可能相对应”。DINOv3 的相似度不能直接当作“选对建筑的概率”。

官方来源：[Grounding DINO 云端接口示例](https://github.com/deepdataspace/dds-cloudapi-sdk)、[DINOv3](https://github.com/facebookresearch/dinov3)、[RoMa v2](https://github.com/Parskatt/RoMaV2)。本项目的数据读写、候选筛选、几何检查和三维输出是围绕这些模型编写的适配代码。

你最初拿到的参考项目 Stage 2 使用 Mask R-CNN 专用权重，目前统一入口没有运行那套 Stage 1/2/3 脚本。检测已替换成云 API，RoMa 的网络来自官方实现；渲染、检索调度和三维预测导出属于本项目新增代码。自动映射复用我们旧 `map_facade_to_3d.py` 的几何函数，没有调用其人工点选界面，也没有沿用参考 Stage 3 的固定 30 米图高假设。代码归属与历史复现见 [实验记录](experiment_log.md)。

## 2. 在 PyCharm 中运行第一张照片

### 第一步：使用目前这个项目和解释器

在 PyCharm 中打开 `run_facade_pipeline.py` 所在的项目目录。本机可以使用外层 `E:\workinCODEX\prog_3D`，也可以使用内层 `CityGML-IMG_Architecture` 仓库；每次保持代码、缓存和结果目录配套。本机现有解释器为：

```text
C:\Users\YuexuanWang\.conda\envs\torch_1\python.exe
```

当前已升级为 Python 3.12。已有环境和模型缓存可继续使用，不必为这一步重新安装整套环境。另一台电脑的安装和缓存准备见 [环境说明](setup.md)。

如果提示缺少 `romav2.0.1.pt` 或 DINOv3 源码缓存，环境说明中提供了可直接粘贴到 **PyCharm Python Console** 的完整准备代码；只改 `project_dir` 为当前脚本所在目录。日常匹配不会替你自动联网补齐缺失缓存。

请保持脚本、`my_results` 和 `model_cache` 在同一项目根目录下。根目录脚本和内层 Git 仓库副本各自按所在目录寻找结果，切换运行副本不会自动迁移缓存。

### 第二步：确认渲染图库已存在

项目中需要有：

```text
my_results/render_gallery/latest_run.json
```

打开它，`run_dir` 指向一份已完成图库；对应文件夹内应有 `gallery_index.json`。已有图库可以复用。如果提示“请先运行 build_render_gallery.py”，按 [多建筑渲染图库教程](experiment_log.md#3-批量建立渲染图库) 先建立图库。

本项目已有 27 栋建筑、216 张视图的图库。这只是可检索的参考库规模，不能据此认为 27 栋均已完成照片匹配。

### 第三步：打开统一入口，设置照片

打开 `run_facade_pipeline.py`，在前面找到：

```python
PHOTO_PATH = DATASET / "textures/4959323_front.jpg"
TOP_BUILDINGS = 5
VIEWS_PER_BUILDING = 2
```

源码默认保留你熟悉的 `4959323_front.jpg`。本轮最终结果为检索正确但几何检查未通过，状态是 `needs_review`；这也是需要保留的失败对照。第一次想观察已经通过的完整三维预览，可以主动改成：

```python
PHOTO_PATH = DATASET / "textures/4907507.png"
```

换成数据集内另一张照片，只改为实际存在的文件名，例如：

```python
PHOTO_PATH = DATASET / "textures/4906970.png"
```

换成你另存的照片，可以用完整路径：

```python
PHOTO_PATH = Path(r"E:\我的照片\楼房正面.jpg")
```

这里的路径只用于读图片。程序不会从 `4959323_front.jpg` 这样的文件名中提取建筑编号来决定答案；建筑身份由候选检索和几何检查决定。照片所对应的建筑应当存在于当前图库中，系统没有“图库外新建筑”的可靠识别保证。

### 第四步：检查是否需要新检测

程序会先寻找相同图片内容、尺寸、模型和实验 D 参数的现有检测。匹配成功就复用；仅仅文件名相同不会直接复用，改名也不会改变图片内容哈希。

如果是没有可复用结果的新照片，需要在 **run_facade_pipeline 的运行配置**里添加环境变量 `DDS_API_TOKEN`。在 PyCharm 的“编辑配置”中选择这个脚本，找到“环境变量”，填写变量名和自己的平台 Token。之前给另一个脚本配置的变量，未必会自动带到新配置中。

新检测会上传照片图块并消耗平台额度。Token 只放在环境变量里，不写进代码或文档。已有合格检测时，这个入口不会为了重新匹配而再次提交同一份检测。

### 第五步：运行并观察控制台

右键编辑区 → “运行 run_facade_pipeline”。不要同时启动同一照片的多份运行。

正常进度包含：

```text
[1/5] 准备门窗检测（已有相同照片优先复用）
[2/5] DINOv3检索多建筑图库
[3/5] RoMa逐个匹配候选
[4/5] 对比候选建筑、墙面与几何证据
[5/5] 门窗框投到选中三维墙面
```

第 5 步仅在自动选择通过时执行。若第 4 步显示“证据不足”，本次结果会是 `needs_review`，不会强行输出三维门窗。这是程序的质量判定结果，不等同于程序崩溃。

默认先取 5 栋、每栋 2 个视角；每个视角最多尝试 2 面主要可见墙，初选最多约 20 个墙面候选。若缺少同墙第二个合格视角，会从完整图库再补充最多两个该墙可见范围较大的视角。首次需要计算图库特征，后续满足缓存校验条件时会复用；候选数量越多，RoMa 匹配通常越耗时。

### 批量处理全部立面照片，仍然运行同一个脚本

单图与批量都使用 `run_facade_pipeline.py`，不用新建脚本，也不用反复改 `PHOTO_PATH`。

1. 在 PyCharm 顶部运行配置下拉菜单中选择“编辑配置”，找到 `run_facade_pipeline`。
2. 在“脚本参数 / Parameters”中填入下面一整行；不是填入环境变量框：

   ```text
   --photo-dir "E:\workinCODEX\prog_3D\CityGML-IMG_Architecture\Drills\Texture2LoD3_dataset\textures"
   ```

3. 保持 `torch_1` 解释器，应用配置后运行。此参数选择批量模式；不要同时填写 `--photo`。以后清空参数就回到源码 `PHOTO_PATH` 指定的单图模式。
4. 每处理完一张，立即更新 `my_results/facade_pipeline/batch_时间/summary.json`；正常返回的记录含 `run_dir`，按它打开单图详细结果。异常时，已建立报告的记录保留 `run_dir`，尚未建目录则为 `null`；先看 `error_type` 和控制台。

这里只遍历所选目录直接包含的 JPG/JPEG/PNG，不递归子目录。请选 `textures` 真实立面照片；**不要选择 `gt_masks` 标注目录，也不要把全景图混进来。** 程序按文件格式读取，不能自动判断一张 PNG 是照片还是标注。

某张异常会记为 `failed`，保存 `error_type` 后继续下一张；`needs_review` 也会保留，不强制投影。因此批量进程退出代码为 0 仍要检查逐张状态。`photo_count` 是计划处理的照片数，`records` 才是已经写下的处理记录。新照片没有检测缓存时仍会调用收费 API，批量并不改变这一规则。

若在 Git 仓库根目录的终端运行，相对路径命令为：

```powershell
python run_facade_pipeline.py --photo-dir "Drills/Texture2LoD3_dataset/textures"
```

### 可选：在 PyCharm 复现实验 E 第二轮对照

实验 E 不替换日常 8 视角图库。当前源码生成每面墙 `-30° / 0° / +30°` 三张图；历史第一轮的 ±15° 图库已经单独保存，当前设置复现的是第二轮。先右键运行 `build_wall_view_gallery.py`，在“编辑配置 → 脚本参数”填入：

```text
--output-root E:\workinCODEX\prog_3D\my_results\experiment_e2\wall_view_gallery_30deg --image-size 900 600 --workers 2
```

运行完成后，打开 `wall_view_gallery_30deg/latest_run.json`，复制其中 `run_dir`，末尾补上 `\gallery_index.json`。再编辑 `run_facade_pipeline` 的运行配置，填入：

```text
--photo-dir Drills\Texture2LoD3_dataset\textures --top-buildings 5 --views-per-building 2 --gallery-index E:\...\wall_gallery_时间\gallery_index.json --output-root E:\workinCODEX\prog_3D\my_results\experiment_e2\wall_view_pipeline_30deg_opening_layout
```

这会复用相同照片和已有实验 D 检测，结果写进独立目录。最后运行 `evaluate_experiment_e.py`，参数依次填写基线 `summary.json`、墙面图库批次 `summary.json` 和输出目录 `docs\results\experiment_e2`。评价脚本把照片名转为建筑编号只用于运行结束后的表格核对；模型推理从不读取这个编号。查看 [第二轮对照图](images/实验E_30度与窗布局检查.png) 时，应同时看“正确自动映射”和“错配自动映射”，不能只看映射总数。

打开某张照片的 `candidate_XX/result.json`，看 `layout_validation` 与 `door_layout_validation`。`required: true` 才表示该项布局检查实际启用；`required: false` 是证据不足，不能读成“门窗布局已验证”。这批目标墙可见门最多 2 个，门布局检查没有触发；请主要看窗布局与几何检查。`4907518_left` 在第一轮错配，第二轮改为 `needs_review`，可用两轮的逐图报告对照候选及失败理由。

## 3. 运行后按什么顺序看结果

控制台会打印本次目录，也可以打开：

```text
my_results/facade_pipeline/latest_run.json
```

里面的 `run_dir` 指向最新一轮。每轮报告保存在新文件夹里，不覆盖上一轮报告；这个指针本身不表示运行成功。

```text
my_results/facade_pipeline/
├── latest_run.json
├── retrieval_cache/                 # 图库 DINO 特征缓存
├── matches_cache/                   # 照片—候选视图的 RoMa 原始匹配缓存
├── detection_cache/                 # 本入口为新照片保存的检测与任务记录
└── run_时间/
    ├── result.json                  # 首先看最终状态、选择理由和建筑编号
    ├── detection_source.json        # 用了哪份检测，是否复用
    ├── retrieval.png                # 输入照片与入选渲染视图拼图
    ├── retrieval.json               # 全部视图排序、入选建筑和模型信息
    ├── model.json                   # RoMa 权重和推理设置
    ├── candidates.csv              # 全部墙面候选的简表
    ├── candidates.json             # 完整候选记录
    ├── candidate_01/               # 后面可能有 candidate_02 等
    │   ├── result.json             # 本候选通过/失败、检查指标和 H 矩阵
    │   ├── roma_matches.png        # 结构细化前采用的 RoMa 对应
    │   ├── roma_matches.npz        # 与该图配套的点和检查掩码
    │   ├── tiled_matches.png       # 长图局部分块匹配，有该阶段时保存
    │   ├── tiled_matches.npz       # 局部分块对应与检查掩码
    │   ├── matches.png             # 最终采用方法的对应点示意
    │   ├── alignment.png           # 照片变换后与渲染图的对齐对照
    │   └── geometry_matches.npz    # 最终方法的点坐标和掩码，与 matches.png 对应
    └── mapping_3d/                 # 仅自动选择通过后创建
        ├── mapping_preview.png    # 墙面展开图 + 三维预览
        ├── predictions.json       # 三维顶点、尺寸、保留/拒绝的检测及原因
        ├── predictions_preview.obj
        └── predictions_preview.mtl
```

某候选过早失败时，可能只有 `result.json`；整轮中途出错时，后续图表也可能尚未生成。应先看状态，再判断缺少的文件是否异常。

### 先看 `result.json`

| `status` | 含义 | 接着看什么 |
|---|---|---|
| `mapped` | 自动选择通过，至少一个检测框生成了三维预测面 | `selected_candidate`、`mapping_3d/mapping_preview.png` |
| `needs_review` | 没有候选满足条件，或多个建筑/墙面证据过于接近 | `selection.reason`、`candidates.csv`、候选对齐图 |
| `no_detections_mapped` | 选出了候选，但检测框全部未通过投影检查 | `mapping_3d/predictions.json` 中的 `rejected` |
| `failed` | 文件、模型、网络或其他运行错误 | `error_type` 和 PyCharm 控制台最后的异常 |
| `running` | 尚未结束，或上次运行被直接中断 | 先确认 PyCharm 是否仍在运行 |

`mapped` 表示通过了当前自动规则并产生结果；`independent_accuracy_verified: false` 表示尚未用独立真值证明真实定位精度，不能把它写成“达到厘米级精度”或“已完全匹配正确”。

`selection.cross_view_agreement` 是跨视角一致性检查。收紧后的规则要求同一堵墙至少有两个分别推理且检查通过的视角，再将同一组照片采样点分别投到墙面，比较三维位置差。查看 `available`、`passed`、各项 `point_count`、`median_m` 和 `p90_m`；这些数值表示两条投影路线是否一致，不表示它们与真实位置的误差。缺少第二个合格视角或位置不一致，都停止投影并返回 `needs_review`。旧初轮报告里可能出现 `available: false, passed: true`，这是已经被收紧的旧行为，不能当作双视角确认。

### 再看图，最后看数字

1. 打开 `retrieval.png`：候选建筑整体形状、窗列和楼层是否接近？它只展示初选结果，不表示已经接受这些建筑。
2. 打开最终 `selected_candidate` 对应的 `alignment.png`：从左到右逐列看窗，检查有无整列错移、左右颠倒、上下错层。重复窗格可能让局部对齐看起来合理，因此要看整面墙。
3. 打开 `mapping_preview.png`：左边是墙面坐标，右边是三维预览。窗、门、待确认分别采用蓝绿、棕橙、黄褐色。`ambiguous` 仍表示门窗类别未确认，映射不会替你解决分类冲突。
4. 打开 `predictions.json`：`mapped_count` 是保留数量，`rejected_count` 是未映射数量；每个被拒绝的框都有 `reason`。`width_m`、`height_m` 来自当前模型与变换，是估计尺寸。

候选中的 `selected_method` 说明最终用了哪套对应。`roma` 是整图 RoMa，`roma_tiled` 表示采用局部分块 RoMa；名称再带 `_window_structure` 表示继续采用窗中心结构细化。`roma_matches.*` 保存结构细化前采用的稠密对应，可能已经来自分块；`matches.png` 与 `geometry_matches.npz` 保存最终方法的对应。初始整图的 `H` 和检查指标另记在 `coarse_roma_homography`、`coarse_roma_validation`，原始对应可按 `raw_matches` 找到。不要把两套点混在一起计算误差。

OBJ 里只有预测四边形，单独打开不会自动带出整栋建筑。它使用局部米制坐标；世界坐标等于 OBJ 坐标加上 JSON 中的 `obj_origin_world_m`。整栋建筑轮廓与预测共同显示在 `mapping_preview.png` 中。

## 4. 程序怎样把“这张照片”对应到“这栋建筑”

可以顺着下面这条链理解：

```text
照片 → 图像特征 → 候选建筑及视图 → 候选墙上的对应点
     → 几何检查与候选比较 → 一面选定墙 → 三维预测门窗
```

**检索：** DINOv3 分别把照片和图库视图变成特征向量，再用余弦相似度排序。图库保留视图所属建筑编号，程序按这个编号把视图归组，取前几栋。这里使用的是渲染图的身份信息，没有用照片文件名提供答案。

**匹配：** 每个入选视角分别裁出主要可见墙，由 RoMa 给出照片点与渲染点的候选对应关系。同一张渲染图里的两面墙分开拟合，避免把两个不同平面混在一起。

**计算变换：** 用对应点拟合一个 3×3 的单应矩阵 `H`，表示“一面近似平面墙在两张图之间怎样变换”。照片划成 8×4 网格，一部分区域用于拟合，另一部分区域的点留出检查。程序还检查点是否覆盖足够大的范围、投影是否镜像等。

**长图局部匹配：** 对长边超过 1400 像素、且粗匹配已有一定支持的照片，程序会尝试把照片按最大 1200×1000、重叠 300 像素分块。粗 `H` 只用于在渲染图中找对应搜索区域，并向外留边距；然后重新让 RoMa 匹配局部图块。这样每块里的窗能占更多模型输入像素。这一阶段只运行本地 RoMa，不重新调用门窗 API。所有点仍换回完整图坐标，再重新进行原来的几何检查；4 像素误差阈值和 50% 留出内点门槛没有因此降低。分块结果检查通过才替代整图结果，失败则保留记录。粗匹配若已经错了一列窗，局部匹配仍可能沿着错误位置匹配，因此它不是必然修好的保证。

**结构辅助：** 以 RoMa 初始结果为基础，把照片中检测到的窗中心与模型原有窗中心建立候选对应。结构检查通过时可采用细化后的变换。这里确实使用了参考 LoD3 模型原有门窗信息；没有读取 `gt_masks`，也没有读取历史人工标定点。这种辅助不能证明在没有门窗细节的普通 LoD2 模型上同样有效。

**比较候选：** 不直接接受 DINO 排名第一的建筑。程序比较各候选的几何证据，并要求最佳候选与其他建筑/墙面之间有足够差距。同一墙面的不同渲染视角视为同一个身份，避免自己与自己竞争。

候选报告里的 `wall_vote` 只是“这面候选墙内有多少匹配点支持”。因为传入的点已经按候选墙过滤，它不是独立判断建筑或墙面身份正确的证据，不能把票数高理解为已核对真值。

**双视角核对：** 同墙至少需要两个分别推理且合格的视角，再比较两种变换得到的三维位置。单视角时会从完整图库补充匹配，仍无第二份证据就返回待复核。它是在自动结果之间做一致性检查，没有新增人工点或独立测量真值，即使两视角一致也不是准确率保证。

**映射到三维：** 将检测框四个角从原照片坐标通过 `H` 变到渲染图坐标，再沿渲染相机的射线与选定墙面求交，得到四个三维顶点。深度和构件编号图用于判断墙面可见性、归属和遮挡；不会把窗玻璃凹进去的深度直接当作墙面位置。

这条链一次处理一张照片、选定一面平面墙。照片里同时有两面建筑立面时，不能期望用一个 `H` 把两面都正确投影。

## 5. 为什么有些二维框没有进入三维

二维检测阶段保留的框，映射时还要通过墙面与投影检查。常见原因如下：

| `rejected` 中的原因 | 通俗解释 |
|---|---|
| `box_touches_photo_border_may_be_incomplete` | 框碰到原照片边界，目标可能只拍到一部分，暂不按完整门窗生成三维面 |
| `mapped_box_outside_render_image` | 投影后的框超出渲染图范围 |
| `mapped_box_not_fully_inside_wall_outer_boundary` | 框没有完整落在选定墙的外轮廓内 |
| `mapped_box_not_visibly_supported_by_selected_wall` | 框覆盖了背景、别的墙或其他遮挡面，归属不够可靠 |
| `invalid_detection_box` | 框坐标本身无效 |

因此，二维图里框的数量和三维面数量可能不同。应查拒绝原因，不能仅为凑齐数量就关闭检查。

还要区分“目标墙不合格”和“背景参考墙画图有问题”。`4907514` 的目标墙最大离平面约 0.257 m，超过 0.03 m 门槛，程序明确拒绝它，不能因为能看到渲染图就强行映射。`4907518_front` 之前则是未选中的参考墙自交导致预览中断；现在对这类背景只保留原始三维折线，并写入 `preview_context_warnings`，合法目标墙的映射继续。目标墙的严格检查没有放宽，也没有修改原模型。

## 6. 如果结果不理想，先调整哪一项

如果 `retrieval.png` 根本没有正确建筑，可以先把 `TOP_BUILDINGS` 从 `5` 改为 `10`；如果正确建筑入选但视角不合适，可把 `VIEWS_PER_BUILDING` 从 `2` 改为 `4`。每次先只改一项，保留两轮结果目录比较。增加候选会增加本地匹配量，不等于必然提高准确率。

如果正确建筑与视角已入选，但 `alignment.png` 明显错位，应查看对应候选的 `roma_validation`、`structure_validation`、`checks`/`failed_checks`。门窗排列重复、照片遮挡、照片与模型外观差异都可能造成失败；仅降低门槛会把错误投影放行。

当前候选比较的最低几何分数为 `0.35`，与其他建筑/墙面的最低分数差为 `0.08`。它们是项目的初步筛选规则，不是已校准的准确率。建议先保留这些值。

检索输入会缩放到 `384×384`，RoMa 当前使用 `base` 设置。长图局部分块会增加局部细节在输入中的占比，但 RoMa 内部仍按自身设置缩放，不能说“模型全程不缩放”。原照片文件没有被覆盖，检测框始终保留在原图坐标里，匹配结果也会换回原图/完整渲染图坐标。坐标还原不能补回未识别的细节。

如果需要命令行运行，在项目目录的终端输入：

```powershell
python run_facade_pipeline.py --photo "E:\我的照片\楼房正面.jpg" --top-buildings 10 --views-per-building 2
```

检查终端的 `python` 是否就是 `torch_1`。PyCharm 直接运行时则使用前面的配置即可。

## 7. 每个代码文件负责哪一段

| 文件 | 主要职责 |
|---|---|
| [run_facade_pipeline.py](../run_facade_pipeline.py) | 串起检测、检索、匹配、候选决策和三维输出；设置输入照片 |
| [facade_retrieval.py](../facade_retrieval.py) | 从已有 RoMa 权重恢复 DINOv3 骨干，计算特征、缓存、按建筑检索 |
| [try_grounding_dino16.py](../try_grounding_dino16.py) | 实验 D 分块门窗检测、API 任务保存和结果合并 |
| [match_facade_roma.py](../match_facade_roma.py) | 复用其本地 RoMa 加载、推理、可视化函数；统一入口不会调用旧的固定建筑主流程 |
| [facade_match_geometry.py](../facade_match_geometry.py) | 匹配坐标还原、单应矩阵估计、空间留出检查 |
| [facade_match_structure.py](../facade_match_structure.py) | 结合预测窗中心与模型原有窗中心细化变换 |
| [facade_auto_mapping.py](../facade_auto_mapping.py) | 候选墙内点支持、射线与平面求交、过滤并保存预测三维面 |
| [map_facade_to_3d.py](../map_facade_to_3d.py) | 提供读取墙面、墙面坐标等共用函数；自动流程不进入其人工点选界面 |

旧代码清理仍有未完成项，目录中可能还保留 Tiny 入口或历史调试文件。当前流程以本页的统一入口为准。

## 8. 本轮实测记录

2026-10-06 已完成全部 17 张的最终复跑：**3 张生成映射预览、14 张待复核、0 运行异常，共 67 个预测面**。

| 照片 | 选中建筑 | 保留预测 | 拒绝检测框 |
|---|---|---|---:|
| 4907507.png | 4907507 | 35 个窗 | 1 |
| 4907518_front.png | 4907518 | 15 个窗、1 个门/窗待确认 | 0 |
| 4907520_front.png | 4907520 | 16 个窗 | 6 |

3 个被接受建筑的编号均与数据集对应关系一致。其余 14 张未强制输出三维预测，其中 `4907518_left` 缺少合格第二视角，其他照片为没有候选通过几何检查；`4906972_front_01`、`front_03` 的对应建筑还未进入初选 Top-5，`4907514` 的目标墙非平面被明确拒绝。完整 [17 张逐图表和原因](experiment_log.md#最终复跑17-张开发照片)、[汇总 CSV](results/pipeline_summary.csv) 和 [审计 JSON](results/pipeline_summary.json) 可直接查看，实验记录还提供三组预测 JSON、OBJ/MTL 和预览图。

DINO 对应建筑 Top-1 为 **10/17**、Top-5 为 **15/17**；检索到候选不等于几何通过。17 张全部复用检测缓存，新增付费请求 **0 次**。2026-10-09 当前仓库代码通过 **104 项测试**；上述指标是原 8 全局视角基线的历史结果，第二轮墙面图库另见[实验 E 记录](experiment_log.md#32-实验-e-第二轮30-与门窗布局检查)。

初轮曾有 3 张 `mapped`、12 张待复核、2 张异常，其中 `4907518_left` 误配到 `4907520`。收紧双视角规则后该误接受已被拒绝，[初轮记录](results/pipeline_initial_audit.json) 保留。因为这些照片参与了发现问题和修改规则，最终仍是开发集结果，不能称独立测试准确率 100%；实际米制定位精度未独立验证。三维产物仍是预测面，没有给原模型开孔或写入新门窗。
