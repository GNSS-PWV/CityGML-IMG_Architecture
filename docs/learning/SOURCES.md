# 前期学习资料：官方来源与建议阅读顺序

> 适用项目：基于街景照片与 CityGML LoD1 模型的建筑目标检索  
> 核对日期：2026-08-30  
> 选材原则：只列官方文档、正式标准或项目维护方的一手资料。

这份文件是学习资料索引，不要求一次读完。建议按 **Git → Python → GIS/CRS → CityGML → 实验评价** 的顺序学习；每一部分先完成“必读”，遇到实际任务时再查“按需阅读”。

---

## 1. Git 基础与三人协作

### 必读

1. [Pro Git 官方中文版](https://git-scm.com/book/zh/v2)
   - **1.1 关于版本控制**：理解为什么需要版本历史。
   - **1.3 Git 是什么**：理解快照、已提交、已修改、已暂存。
   - **1.5 安装 Git、1.6 初次运行 Git 前的配置**：完成本机安装与姓名、邮箱配置。
   - **2.1 获取 Git 仓库、2.2 记录每次更新到仓库**：掌握 `clone`、`status`、`diff`、`add`、`commit`。
   - **2.3 查看提交历史、2.4 撤消操作、2.5 远程仓库的使用**：掌握 `log`、安全撤销、`fetch/pull/push`。
   - **3.1 分支简介、3.2 分支的新建与合并、3.3 分支管理**：掌握功能分支和普通合并。
   - **3.4 分支开发工作流、3.5 远程分支**：理解三人如何并行工作。

2. [GitHub：About pull requests](https://docs.github.com/en/pull-requests/get-started/about-pull-requests)
   - 重点读 **Why use pull requests**、**Key parts** 和 **How pull requests fit your workflow**。
   - 目标：理解“分支提交 → 发起 PR → 同伴审查 → 合并”的完整协作流程。

3. [GitHub：About pull request reviews](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/reviewing-changes-in-pull-requests/about-pull-request-reviews)
   - 重点读 **Reviewing pull requests** 和三种审查结果：Comment、Approve、Request changes。
   - 目标：每个 PR 至少由另一名组员检查后再合并。

### 按需阅读

- [gitignore 官方文档](https://git-scm.com/docs/gitignore)：开始提交数据前必读；照片、模型、权重、密钥和虚拟环境不应进入普通 Git 历史。
- [git restore 官方文档](https://git-scm.com/docs/git-restore)：恢复工作区文件或取消暂存。
- [git revert 官方文档](https://git-scm.com/docs/git-revert)：已经共享的错误提交优先用新提交撤销。
- [GitHub：创建 Pull Request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request)：第一次实际创建 PR 时跟着操作。

### 暂时不学

复杂 `rebase`、交互式变基、submodule、自建 Git 服务器和复杂 CI/CD。前期先把小分支、小提交和 PR 审查做稳定。

---

## 2. Python 工程基础

### 必读

1. [Python 官方中文教程](https://docs.python.org/zh-cn/3/tutorial/index.html)
   - **第 4 章：更多控制流工具**：条件、循环、函数、参数。
   - **第 5 章：数据结构**：列表、字典、集合、遍历。
   - **第 6 章：模块**：把代码拆成多个 `.py` 文件和包。
   - **第 7 章：输入与输出**：文件读写、格式化输出、JSON。
   - **第 8 章：错误和异常**：看懂 traceback，正确使用 `try/except`。
   - **第 12 章：虚拟环境和包**：使用 `venv` 与 `pip` 隔离依赖。

2. [pathlib：面向对象的文件系统路径](https://docs.python.org/zh-cn/3/library/pathlib.html)
   - 先读 **基础使用**，掌握 `Path`、`/` 拼接路径、`exists()`、`glob()`、`open()`。
   - 目标：代码不再依赖某台电脑写死的绝对路径或手工拼接的斜杠。

3. [csv 模块](https://docs.python.org/zh-cn/3/library/csv.html) 与 [json 模块](https://docs.python.org/zh-cn/3/library/json.html)
   - `csv` 重点：`DictReader`、`DictWriter`、`newline=''`。
   - `json` 重点：`load`、`dump`、`ensure_ascii`、`indent`。
   - 目标：能稳定读写图像元数据、配置和预测结果。

4. [argparse：命令行参数解析](https://docs.python.org/zh-cn/3/library/argparse.html)
   - 先读 **教程** 和 `ArgumentParser` 基础示例。
   - 目标：脚本能通过 `--config`、`--input`、`--output` 接收参数，而不是每次修改源码。

### 按需阅读

- [logging：日志工具](https://docs.python.org/zh-cn/3/library/logging.html)：正式实验不要只使用零散 `print()`。
- [unittest：单元测试框架](https://docs.python.org/zh-cn/3/library/unittest.html)：为坐标变换、方位角、候选排序等关键函数建立小测试。
- [venv 模块](https://docs.python.org/zh-cn/3/library/venv.html)：遇到环境创建或激活问题时查阅。

### 建议练习出口

能独立实现并运行：读取 CSV → 校验必填字段 → 输出 JSON 汇总；命令、输入和输出路径均由命令行参数指定。

---

## 3. GIS、坐标参考系统与 pyproj

### 必读

1. [GeoPandas：Projections](https://geopandas.org/en/stable/docs/user_guide/projections.html)
   - 重点读 **Coordinate reference systems**、**Setting a projection**、**Re-projecting**、**The axis order of a CRS**。
   - 必须理解：`set_crs()` 是声明现有坐标的含义，`to_crs()` 才是实际重投影；两者不能混用。

2. [pyproj：Getting Started](https://pyproj4.github.io/pyproj/stable/examples.html)
   - 重点读 CRS 创建、坐标变换和 `Transformer` 示例。
   - 目标：会用 `CRS` 检查坐标系，会用 `Transformer.from_crs()` 统一街景 GPS 与 CityGML 坐标。

3. [pyproj：Transformer API](https://pyproj4.github.io/pyproj/stable/api/transformer.html)
   - 重点看 `Transformer.from_crs()`、`transform()`、`always_xy`、`area_of_use` 和 `accuracy`。
   - 项目常用约定为 `x=经度/东坐标，y=纬度/北坐标`；使用前仍须核实数据源轴顺序。

4. [pyproj：Gotchas / FAQ](https://pyproj4.github.io/pyproj/stable/gotchas.html)
   - 必读 **What are the best formats to store the CRS information?**。
   - 必读 **Axis order changes in PROJ 6+**。
   - 必读 **Proj (Not a generic latitude/longitude to projection converter)**。
   - 目标：避免经纬度颠倒、把角度当米、忽略 datum 转换等常见错误。

### 权威查询入口

- [EPSG Geodetic Parameter Dataset](https://epsg.org/home.html)：查询 EPSG 代码、坐标轴、单位、适用范围和 datum；不要凭印象猜坐标系。
- [PROJ 官方 FAQ](https://proj.org/en/stable/faq.html)：理解 CRS 表达格式、轴顺序和转换选择。

### 建议练习出口

打印源 CRS 与目标 CRS 的名称、轴顺序、单位和适用范围；将 5 个已知 GPS 点转换到 CityGML 坐标系，并在 QGIS 中检查它们是否落在正确街区。距离计算只能在合适的米制投影坐标系下进行。

---

## 4. CityGML 与 LoD1

### 先确认数据版本

正式写解析器前，先打开 CityGML 文件头，记录命名空间、`srsName`、CityGML 版本和 GML 版本。CityGML 2.0 与 3.0 的数据结构和 LoD 语义并不完全相同，不能把 3.0 示例直接套到 2.0 文件上。

### 必读：CityGML 3.0

1. [OGC CityGML 3.0 Part 1：Conceptual Model Standard](https://docs.ogc.org/is/20-010/20-010.html)
   - **第 6 章 Overview of CityGML**：理解 CityGML 保存对象、语义、几何与不同 LoD 的目的。
   - **7.2 Core Model**，尤其 **7.2.5 Geometry and LOD**：LoD1 是由水平 footprint 垂直拉伸得到的 solid/block model。
   - **Building 模块的 UML 与数据字典**：理解 Building、BuildingPart、边界面和属性之间的关系。

2. [OGC CityGML 3.0 Conceptual Model Users Guide](https://docs.ogc.org/guides/20-066.html)
   - 重点读 **CityGML 3.0 in a Nutshell**、**Levels of Detail**、建筑示例。
   - 用户指南更适合第一次建立整体认识，标准用于核对准确含义。

3. [OGC CityGML 3.0 Part 2：GML Encoding Standard](https://docs.ogc.org/is/21-006r2/21-006r2.html)
   - **6.1 Core**：对象标识、命名空间、GML 编码规则。
   - **6.5 Building**：Building 模块的 XML/GML 编码。
   - **Annex C.4 Building**：查看 `building.xsd` 与元素定义。
   - 目标：区分“概念模型中的 Building”与“XML 文件中如何编码 Building”。

### 如果老师的数据是 CityGML 2.0

- [OGC CityGML 2.0 Encoding Standard（OGC 12-019，官方存档）](https://portal.ogc.org/files/?artifact_id=47842)
  - 重点查 **第 8 章 CityGML core**、**第 10 章 Building model**、LoD0–LoD4 的几何属性和 `gml:id`。
  - 只实现老师数据中真实出现的元素，不要一开始编写“支持所有 CityGML 文件”的通用解析器。

### 建议练习出口

从一个真实文件中提取至少 10 栋建筑的对象 ID、CRS、footprint/ground surface、LoD1 solid 或可用于恢复轮廓与高度的几何，并导出 GeoJSON 在 QGIS 中核验。

---

## 5. 实验评价与数据划分

### 必读

1. [scikit-learn：Cross-validation — evaluating estimator performance](https://scikit-learn.org/stable/modules/cross_validation.html)
   - 开头部分：理解为什么不能在训练/调参数据上报告最终性能。
   - **Cross-validation iterators for grouped data**：理解具有相关性的样本必须按组划分。
   - **Group K-fold**：确保同一组不会同时出现在训练和验证折中。
   - 本项目应把 `building_id` 作为 group；不能让同一栋建筑的不同照片分散到训练集和测试集。

2. [GroupKFold API](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html)
   - 重点读 `groups` 参数、每组在测试折中的出现规则和示例。
   - 若只做固定的 train/validation/test 划分，也必须先按建筑 ID 分组再划分。

3. [scikit-learn：Metrics and scoring](https://scikit-learn.org/stable/modules/model_evaluation.html)
   - 重点读 **Accuracy score**、**Top-k accuracy score**、**Precision, recall and F-measures**、**Confusion matrix**。
   - 项目核心至少报告 Top-1 Accuracy、Recall@5/Top-5、错配率；加入拒绝机制后还要报告 precision 与 coverage 的关系。

4. [top_k_accuracy_score API](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.top_k_accuracy_score.html)
   - 核对 `y_true`、候选类别分数 `y_score`、`k` 和类别顺序；不要把“候选集合是否含真值”和“分类 Top-k”混为一谈。

### 实验纪律

- 训练集：拟合模型或学习参数。
- 调试/验证集：选择阈值、权重和超参数。
- 测试集：方案冻结后只用于最终评价；不能反复看测试结果再调参数。
- 划分表必须固定保存 `image_id`、`building_id`、`split`，并纳入版本控制。
- 所有正式结果记录数据版本、Git commit、配置文件、随机种子、完整命令和输出目录。
- 消融实验每次只移除或加入一个模块；鲁棒性实验要明确 GPS 和航向扰动的生成方法。

### 建议练习出口

构造一个含多张同建筑照片的小表，证明 train/validation/test 之间不存在重复 `building_id`；再用一组手工预测结果算出 Top-1、Top-5、错配率和混淆统计，并人工核对数值。

---

## 建议的五周最小学习计划

| 周次 | 学习重点 | 当周可检查成果 |
|---|---|---|
| 第 1 周 | Git 基础、分支、PR | 三人各完成一次小 PR，并由另一人审查 |
| 第 2 周 | Python 模块、文件读写、虚拟环境 | CSV → 校验 → JSON 的命令行脚本 |
| 第 3 周 | CRS、pyproj、轴顺序 | 5 个 GPS 点正确叠加到 CityGML 区域 |
| 第 4 周 | CityGML 版本、Building、LoD1 | 提取 10 栋建筑并导出 GeoJSON |
| 第 5 周 | 分组划分与指标 | 固定 split 表，并算出最近建筑基准指标 |

完成标准不是“看完链接”，而是能留下可运行脚本、可复查输出和一段简短结论。
