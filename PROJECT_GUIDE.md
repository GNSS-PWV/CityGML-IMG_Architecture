# 街景影像—CityGML 建筑目标检索项目实施指南

> 适用团队：3 人本科生项目组  
> 项目周期：2026 年 9 月—2027 年 8 月  
> 时间投入：每位成员每周约 1 天  
> 项目主题：基于 CityGML 可见性约束的街景影像—建筑目标检索精度提升

## 目录

- [1. 项目定位](#1-项目定位)
- [2. 目标成果](#2-目标成果)
- [3. 总体技术路线](#3-总体技术路线)
- [4. 自主学习清单](#4-自主学习清单)
- [5. 分阶段实施计划](#5-分阶段实施计划)
- [6. 三人团队分工](#6-三人团队分工)
- [7. Git 团队协作指南](#7-git-团队协作指南)
- [8. 服务器和云算力使用](#8-服务器和云算力使用)
- [9. 每周工作节奏](#9-每周工作节奏)
- [10. 负责人自我要求](#10-负责人自我要求)
- [11. 风险与降级方案](#11-风险与降级方案)
- [12. 第一个项目日清单](#12-第一个项目日清单)

---

## 1. 项目定位

本项目不追求直接完成一套“卫星—街景—精细三维重建系统”，而是先解决其中一个关键问题：

> 给定带有粗略 GPS、拍摄方向和视场角的街景照片，从校园 CityGML 建筑模型中，更准确地找到照片对应的建筑 `building_id`，并判断结果是否可信。

这可以理解为三维模型更新流程中的“先找对楼”。只有找对楼，后续从街景照片中提取的楼层、窗户、材质等信息才不会被写到错误建筑上。

项目边界：

- 使用已有校园 CityGML LoD1 模型和项目组自有街景影像；
- 假设照片已有粗略 GPS、航向和视场角；
- 研究附近候选建筑的检索和排序精度；
- 允许低置信度结果转人工复核；
- 不进行城市尺度无先验视觉定位；
- 不从头训练大型深度学习模型；
- 不宣称提出新的通用定位算法；
- 不把完整三维重建、网页系统或 3DGS 作为必做内容。

三人每周各投入一天，一年理论上约有 144 人日。扣除考试、假期、调试和沟通，应按约 90～110 个有效人日规划。因此，控制范围是按时结题的首要条件。

---

## 2. 目标成果

### 必须完成

1. **小规模标注数据集**
   - 约 300～500 张街景照片；
   - 覆盖约 50～100 栋建筑；
   - 每张照片包含 GPS、航向、视场角、正确 `building_id`；
   - 多主体或无法确认的照片标记为模糊样本。

2. **可运行的 Python 实验原型**
   - 输入：街景照片、相机元数据、CityGML 模型；
   - 输出：候选建筑、Top-1、各项分数、置信度、人工复核标志；
   - 支持通过统一命令重复运行。

3. **完整实验**
   - 最近建筑基准；
   - GPS + 方向过滤基准；
   - CityGML 几何可见性实验；
   - 影像—投影一致性实验；
   - 模块消融实验；
   - GPS 和航向误差扰动实验；
   - 失败案例分析。

4. **可复核的结题报告**
   - 说明哪些约束有效、相对基准提高多少；
   - 说明哪些场景仍会失败；
   - 保留参数、配置、代码版本和实验记录。

### 条件允许时再完成

- 简单可视化演示；
- 软件著作权材料；
- 学生论文或会议投稿材料。

论文和软件著作权不应在前期作为硬性承诺，应以实验结果和导师意见为准。

---

## 3. 总体技术路线

```text
CityGML LoD1 模型                街景照片
建筑轮廓、高度、building_id       GPS、航向、视场角
          │                           │
          └───────────┬───────────────┘
                      ↓
               统一坐标与数据检查
                      ↓
           根据距离生成附近候选建筑
                      ↓
           方向、视锥、立面朝向筛选
                      ↓
                建筑遮挡关系判断
                      ↓
        候选立面投影到街景照片中
                      ↓
        与照片中的建筑区域计算一致性
                      ↓
           候选排序、置信度和拒绝
                      ↓
      building_id + 分数 + 人工复核标志
                      ↓
          基准对比、消融和误差分析
```

每增加一个模块，都必须单独回答：

> 加入这个模块以后，候选召回率、Top-1 准确率或错配率是否改善？

不要等所有模块完成以后才进行第一次评价。

---

## 4. 自主学习清单

### 第一优先级：前两个月必须掌握

#### Git 和团队协作

需要理解仓库、工作区、暂存区、提交、远程仓库、分支、Pull Request、合并冲突和 `.gitignore`。

建议只学习《Pro Git》中的：1.5、1.6、2.1～2.5、3.1～3.3。

- [Pro Git 官方中文版](https://git-scm.com/book/zh/v2)
- [GitHub Pull Request 官方说明](https://docs.github.com/en/pull-requests/get-started/about-pull-requests)

暂时不学复杂 rebase、submodule、自建 Git 服务器、复杂 CI/CD 和 Git 内部原理。

#### Python 工程基础

至少掌握：

- 函数、模块和类的基本用法；
- `pathlib` 路径处理；
- CSV、JSON 读写；
- NumPy、pandas、Matplotlib；
- 虚拟环境和 `pip`；
- 命令行参数、异常处理和简单测试。

- [Python 官方中文教程](https://docs.python.org/zh-cn/3/tutorial/index.html)

目标不是掌握全部 Python，而是能独立运行：

```bash
python run_pipeline.py --config configs/pilot.json
```

#### GIS 和坐标系统

重点理解经纬度与投影坐标、EPSG、WGS84、坐标轴顺序、高程基准、坐标转换和空间索引。

建议使用：

- `pyproj`：坐标转换；
- `shapely`：几何计算；
- `geopandas`：空间表格处理。

- [pyproj 官方文档](https://pyproj4.github.io/pyproj/stable/api/proj.html)

负责人必须将以下信息写入 `docs/data_schema.md`：

- CityGML 和 GPS 的坐标参考系；
- 高程单位；
- 航向角起始方向；
- 顺时针还是逆时针；
- 使用度还是弧度；
- 照片坐标是否经过纠偏。

很多看似算法问题的错误，实际上来自坐标、角度或单位错误。

#### CityGML 基础

重点掌握 `Building`、`building_id`、footprint、高度、LoD、`srsName`、XML 命名空间和建筑几何顶点。

- [OGC CityGML 3.0 标准](https://docs.ogc.org/is/20-010/20-010.html)
- [CityGML 3.0 用户指南](https://docs.ogc.org/guides/20-066.html)

不要编写支持所有 CityGML 版本的通用解析器。先确认老师提供的数据版本，只处理当前数据真实使用的结构。

#### 实验评价基础

必须理解训练集、调试集、测试集、数据泄漏、Top-1 Accuracy、Recall@5、错配率、Precision、Coverage、消融和鲁棒性实验。

数据集必须按照“建筑”划分，而不是随机按照照片划分，避免同一栋建筑同时出现在调试集和测试集中。

- [scikit-learn 模型评价文档](https://scikit-learn.org/stable/modules/model_evaluation.html)

### 第二优先级：做到投影模块前学习

#### 相机模型和投影

需要理解相机内参、外参、航向角、俯仰角、滚转角、FOV、针孔相机模型、坐标系变换和镜头畸变。

核心关系：

```text
s × p = K × [R | t] × P
```

- [OpenCV 相机标定与三维重建文档](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)

建议先完成最小练习：输入一个相机位置、朝向和一个建筑立面的四个顶点，将四边形画到照片上。成功后再处理整栋建筑。

#### 可见性与遮挡

学习方位角、视锥、立面法向量、点积、射线相交、深度顺序和建筑遮挡。建议从二维 footprint 开始，再扩展到简化三维。

#### 预训练图像分割

本项目只需要会加载预训练权重、输入图片、获得建筑 mask、保存结果并检查错误样本。

- [SegFormer 文档](https://huggingface.co/docs/transformers/model_doc/segformer)

不要把某个模型名称写成创新点。先在约 30 张照片上比较现成模型，选择最省事且够用的方案。

### 论文阅读顺序

1. [Robust Building Identification from Street Views, 2024](https://www.mdpi.com/2075-5309/14/3/578)
2. [Visual Localization Using Imperfect 3D Models From the Internet, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Panek_Visual_Localization_Using_Imperfect_3D_Models_From_the_Internet_CVPR_2023_paper.html)
3. [To Glue or Not to Glue?, ISPRS 2025](https://isprs-annals.copernicus.org/articles/X-1-W2-2025/35/2025/isprs-annals-X-1-W2-2025-35-2025.pdf)

阅读时先回答：研究问题是什么、输入输出是什么、使用什么数据、如何评价、哪一部分可以借鉴。

当前不需要深入学习 NeRF、3D Gaussian Splatting、扩散式三维生成或大规模视觉语言模型。

---

## 5. 分阶段实施计划

### 阶段 0：启动与降风险（2026 年 9 月第 1～2 周）

任务：

- 创建私有 GitHub/GitLab 仓库；
- 三个人都完成一次 clone、commit、push 和 PR；
- 确认 CityGML 版本、坐标系和服务器规则；
- 检查 10 张照片的 GPS、航向、时间和相机信息；
- 在 QGIS 中同时显示相机位置和建筑；
- 建立统一数据表和命名规则；
- 使用 10 张照片完成第一次端到端试验。

建议数据字段：

```text
image_id,image_path,longitude,latitude,altitude,
heading,pitch,roll,horizontal_fov,building_id,
is_ambiguous,split,notes
```

阶段出口：

- 三个人都能运行同一个检查脚本；
- 能在地图上看到照片位置和建筑；
- 至少 10 张照片有明确 `building_id`；
- 坐标单位和角度方向已经写清楚。

未达到以上条件时，不进入模型开发。

### 阶段 1：数据集与基准（2026 年 9～10 月）

先使用 30～50 张照片做 pilot，不立即标注 500 张。

制定标注规范：主体建筑如何确定、模糊样本如何标记、`building_id` 来源、谁标注和谁复核。每张照片至少由两人确认。

实现两个基准：

```text
基准 A：照片 GPS → 最近建筑中心或轮廓 → building_id
基准 B：附近建筑 → 方向过滤 → 最近候选 → building_id
```

评价 Recall@5、Top-1 Accuracy、错配率、模糊样本比例和错误类型。

阶段出口：一个命令能重新运行基准，输出 `predictions.csv`，错误样本有可视化或备注。

如果最近建筑基准已经几乎全对，应增加建筑密集、路口或遮挡明显的第二片区域。

### 阶段 2：CityGML 几何可见性（2026 年 11～12 月）

按顺序逐个加入：候选半径、相机方向、视场角、立面朝向、距离、遮挡近似。

| 方法 | Recall@5 | Top-1 | 错配率 |
|---|---:|---:|---:|
| 最近建筑 | 待填写 | 待填写 | 待填写 |
| GPS + 方向 | 待填写 | 待填写 | 待填写 |
| + 视场角 | 待填写 | 待填写 | 待填写 |
| + 立面朝向 | 待填写 | 待填写 | 待填写 |
| + 遮挡 | 待填写 | 待填写 | 待填写 |

阶段出口：每个约束可解释、有消融结果、有相机与候选建筑可视化、几何部分可在 CPU 上运行。

如果三维遮挡实现过难，先使用二维 footprint 射线近似。

### 阶段 3：影像一致性和置信度（2027 年 1～3 月）

这个阶段才开始使用 GPU。

优先使用老师课题组已有模型或现成预训练模型，不从头训练大型网络。

将候选立面投影到照片后，可以计算：

- 投影区域与建筑 mask 的重叠度；
- 投影中心与 mask 中心的距离；
- 投影面积是否合理；
- 投影区域是否超出图像；
- 候选是否被严重遮挡。

最初使用可解释的加权分数：

```text
总分 = 距离分数 + 方向分数 + 立面朝向分数
     + 遮挡分数 + 投影重叠分数
```

权重只在调试集上选择。置信度可以由第一名分数、前两名分差和多个约束是否一致构成。低置信度样本转人工复核。

阶段出口：能输出 Top-1、Top-5、分项分数和置信度，能绘制 Precision-Coverage 曲线并生成投影核验图。

### 阶段 4：正式实验（2027 年 4～5 月）

冻结数据划分和测试集，完成：

- 最近建筑、GPS + 方向、完整方法的对比；
- 分别去除立面朝向、视场角、遮挡、影像一致性和拒绝机制的消融；
- GPS 误差 0、5、10、20 米的初步扰动实验；
- 航向误差 0°、5°、15°、30° 的初步扰动实验；
- GPS 错误、航向错误、相邻建筑、遮挡、CityGML 误差、分割错误等失败案例分类。

具体扰动数值应根据真实设备误差调整。不要只展示成功案例。

### 阶段 5：成果封装（2027 年 6～8 月）

最终应支持：

```bash
python run_pipeline.py --config configs/test.json
python evaluate.py --pred outputs/test_predictions.csv
```

建议仓库结构：

```text
building-retrieval/
├── README.md
├── PROJECT_GUIDE.md
├── requirements.txt
├── .gitignore
├── configs/
├── src/
├── tests/
├── docs/
│   ├── data_schema.md
│   ├── experiment_log.md
│   └── decisions.md
├── data/
│   └── README.md
└── run_pipeline.py
```

目录按需要逐步增加，不要在第一天创建大量空模块。照片、CityGML、模型权重和大量输出不进入普通 Git 仓库；代码放 Git，数据放老师服务器，仓库只保存数据清单和说明。

- [GitHub 大文件和 Git LFS 说明](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage)

---

## 6. 三人团队分工

| 人员 | 主责 | 副责 |
|---|---|---|
| 负责人 | 项目统筹、CityGML 解析、几何可见性 | Git 维护、系统集成、报告框架 |
| 组员 B | 图像预处理、分割、投影一致性 | 服务器环境、结果可视化 |
| 组员 C | 数据标注、评价、实验表格 | 测试、失败案例、文档整理 |

协作规则：

- 主责人实现功能；
- 副责人审查 Pull Request；
- 第三个人至少独立运行一次；
- 至少两台机器能运行后任务才算完成；
- 每个人都要了解项目完整输入和输出。

---

## 7. Git 团队协作指南

### Git 的四个位置

```text
正在修改的文件
      ↓ git add
暂存区
      ↓ git commit
本地版本历史
      ↓ git push
GitHub / GitLab 远程仓库
```

`commit` 只保存到本地；`push` 才会上传到远程仓库。

### 第一次配置

```bash
git --version
git config --global user.name "你的姓名"
git config --global user.email "你的邮箱"
git config --global init.defaultBranch main
git config --global pull.ff only
```

### 第一次下载项目

```bash
git clone <仓库地址>
cd <项目目录>
git status
```

### 每次开始工作

不要直接在 `main` 上开发：

```bash
git switch main
git pull --ff-only
git switch -c feature/nearest-baseline
```

推荐分支名：

```text
feature/data-schema
feature/nearest-baseline
feature/citygml-parser
feature/visibility-filter
feature/image-mask
feature/evaluation
fix/heading-angle
docs/experiment-plan
```

### 修改后提交

```bash
git status
git diff
git add src/baseline.py
git add tests/test_baseline.py
git commit -m "feat: add nearest-building baseline"
git push -u origin feature/nearest-baseline
```

随后在 GitHub/GitLab 上创建 Pull Request/Merge Request，由另一名成员审查。

推荐提交信息：

```text
feat: add visibility filtering
fix: correct heading angle convention
test: add projection test
docs: document coordinate reference system
refactor: simplify candidate scoring
```

避免使用 `update`、`修改`、`111`、`最终版2` 等含义不清的信息。

### PR 合并以后

```bash
git switch main
git pull --ff-only
git branch -d feature/nearest-baseline
```

### 常用检查命令

```bash
git status
git diff
git log --oneline --graph --all
git branch
git remote -v
```

### 常见误操作

取消暂存：

```bash
git restore --staged 文件名
```

临时保存未完成修改：

```bash
git stash push -m "unfinished visibility work"
git stash list
git stash pop
```

错误提交已经推送时，优先使用：

```bash
git revert <提交编号>
```

不要随意使用：

```bash
git reset --hard
git push --force
```

发生问题时先运行 `git status`，保留完整输出，再寻求帮助。

### 团队 Git 规则

- 禁止直接向 `main` 推送；
- 一个任务对应一个分支；
- 一个 PR 只解决一件事；
- 合并前至少一人审查；
- 合并前必须能运行；
- 原始数据、密码、SSH 私钥和云密钥不能进入 Git；
- 每周至少形成一次有意义的提交；
- 正式图表尽量由代码生成。

---

## 8. 服务器和云算力使用

### GPU 需求

| 工作 | 是否需要 GPU |
|---|---|
| CSV/JSON 处理 | 否 |
| CityGML 解析 | 否 |
| 坐标转换 | 否 |
| 候选建筑检索 | 否 |
| 视锥和遮挡计算 | 否 |
| 指标计算 | 否 |
| 预训练分割模型推理 | 建议使用 |
| 大模型微调 | 当前不建议 |

项目初期不要购买云 GPU。几何和基准实验应优先在 CPU 上完成。

### 向管理员确认

- 是否需要校园 VPN；
- SSH 地址、端口和用户名；
- 密码还是 SSH key；
- 是否使用 Slurm；
- 登录节点是否允许运行程序；
- 数据存储目录和容量；
- GPU 是否需要预约；
- 推荐 CUDA、PyTorch 版本；
- 是否有公共 Python/Conda 环境；
- 结果如何备份。

### 基础命令

```bash
ssh 用户名@服务器地址
nvidia-smi
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

PyTorch 安装命令应根据服务器环境从官方页面生成：

- [PyTorch 官方安装页面](https://pytorch.org/get-started/locally/)

如果服务器使用 Slurm，应遵守课题组规则，不在登录节点运行长时间 GPU 任务。

### 实验记录

每次正式实验记录：

```text
实验日期、Git commit、数据版本、配置文件、完整命令、
服务器/GPU、软件版本、输出目录、评价指标、简短结论
```

获取当前版本：

```bash
git rev-parse HEAD
```

### 云服务建议

- 只在确实需要 GPU 时开机；
- 先用 10 张图片验证，再运行全部数据；
- 设置消费提醒和预算上限；
- 同步结果后立即关机；
- 不将云密钥写入代码。

---

## 9. 每周工作节奏

建议安排：

1. 开始前 20 分钟：说明上周结果和本周唯一目标；
2. 上午：成员分别开发；
3. 下午前半段：集成和实验；
4. 下午后半段：互相运行、修复问题；
5. 最后 30 分钟：展示结果、提交 Git、更新实验日志。

每周结束必须留下：

1. 一个提交或 Pull Request；
2. 一个可运行命令；
3. 一张结果图或指标表；
4. 三句话结论。

示例：

```text
本周加入立面朝向筛选。
Top-1 从 72% 提高到 78%，Recall@5 基本不变。
主要剩余错误来自路口相邻建筑和航向角偏差。
```

每四周进行一次里程碑复盘，并向导师展示真实运行结果，而不是只汇报学习过程。

---

## 10. 负责人自我要求

### 控制范围

面对新功能时先问：

> 这件事是否直接帮助提高建筑目标检索精度，或者证明这种提高有效？

如果不能，记录为后续想法，不立即加入计划。

### 为任务定义完成标准

不要写“研究 CityGML”，而要写：

> 读取 10 栋建筑的 `building_id`、轮廓和高度，导出为 GeoJSON，并在 QGIS 中正确显示。

### 不只做行政工作

负责人至少掌握 Git 合并与冲突、完整运行流程、数据字段、评价指标、各模块输入和输出。可以不写完所有代码，但必须能判断结果是否合理。

### 尽早暴露问题

一个问题连续阻塞两周时，应缩小问题、制作最小复现、整理报错和预期结果，并找组员或导师讨论。

### 保护实验可信度

- 测试集不用于调参；
- 同一建筑不跨数据集泄漏；
- 不删除表现差的失败样本；
- 全组使用同一套评价脚本；
- 正式图表能追溯到代码、配置和数据版本；
- 不承诺尚未得到的准确率。

### 公平分工和贡献记录

每月记录各成员完成的模块、标注量、实验和报告内容。提前讨论论文署名和软著参与人。

### 保持导师沟通

建议每月至少一次正式汇报，只讲：上次目标、已完成结果、当前图表、一个关键问题和下月目标。避免只说“最近在学习”或“代码还在调”。

---

## 11. 风险与降级方案

| 风险 | 触发信号 | 降级方案 |
|---|---|---|
| 坐标不一致 | 相机和建筑明显错位 | 暂停算法开发，核实 CRS、轴顺序和单位 |
| 最近建筑基准过强 | 简单区域准确率接近满分 | 增加建筑密集、路口和遮挡区域 |
| CityGML 解析复杂 | 长时间卡在不同几何类型 | 只支持当前 LoD1 子集，必要时预先转换 |
| 三维遮挡困难 | 两周仍不能稳定判断 | 使用二维 footprint 射线近似 |
| 分割效果差 | 建筑 mask 大量错误 | 更换现成模型；固定子集用人工 mask 做上限实验 |
| GPU 不足 | 排队或显存不足 | 降低分辨率和批量大小，按需短租云 GPU |
| 完整方法提升小 | 多个模块无显著增益 | 保留负结果，完成误差和适用条件分析 |
| 进度落后 | 连续两个月未达到阶段出口 | 取消网页、模型训练和扩展区域，保住主线 |

允许得到“某些约束作用有限”的结论，但不允许得到无法复现、无法解释的结果。

---

## 12. 第一个项目日清单

1. 用 30 分钟确认项目边界：只做建筑目标检索精度提升；
2. 向老师确认数据版本、坐标系和服务器使用方法；
3. 创建私有 Git 仓库并邀请三位成员；
4. 三个人分别 clone，各自完成一次小型 PR；
5. 选取 10 张照片和对应区域 CityGML；
6. 建立第一版 `docs/data_schema.md`；
7. 人工确认 10 张照片对应的 `building_id`；
8. 在 QGIS 中同时显示相机位置和建筑；
9. 建立三个任务：读取照片元数据、提取建筑轮廓和 ID、计算最近建筑；
10. 当天结束前提交半页项目日志。

前四周唯一的总目标是：

> 让 30～50 张照片跑通“数据读取 → 最近建筑基准 → 指标计算 → 错误可视化”的完整闭环。

在这个闭环跑通以前，不训练大型模型、不购买云 GPU、不开发网页、不开展 3DGS。先验证数据、坐标、基准和评价流程，是一年内按时完成项目的最短路径。
