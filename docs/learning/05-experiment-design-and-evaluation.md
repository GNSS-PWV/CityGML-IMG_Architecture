# 实验设计与评价指标

## 学习目标

完成本教程后，你应该能够：

- 明确一次实验的研究问题、基准和变量；
- 按建筑划分调试集和测试集，避免数据泄漏；
- 计算 Top-1 Accuracy、Recall@5 和错配率；
- 理解低置信度拒绝中的 Precision-Coverage；
- 设计消融实验和 GPS/航向扰动实验；
- 记录可以复现的实验；
- 对失败案例进行分类，而不是只展示成功样本。

## 1. 先从研究问题开始

实验不是“运行代码得到一个数字”，而是回答一个明确问题。

本项目可以拆成：

1. 最近建筑基准会在什么情况下认错？
2. 航向和视场角能否减少相邻建筑错配？
3. 立面朝向和遮挡约束分别贡献多少？
4. 影像—投影一致性能否进一步改善候选排序？
5. 置信度能否识别不可靠结果？
6. GPS 或航向误差增大时，方法如何退化？

每个实验表格和图都应对应其中一个问题。

## 2. 样本和真值

一条基本样本可以包含：

```text
image_id
building_id
is_ambiguous
split
```

真值必须通过人工标注和复核获得。不能把某个算法输出再当作真值评价同一个算法。

### 模糊样本

以下照片可能属于模糊样本：

- 一张照片中有多栋同等重要的建筑；
- 主体建筑被完全遮挡；
- 照片元数据明显错误；
- 标注人员无法可靠判断；
- 数据中找不到对应建筑。

模糊样本不应被悄悄删除。应保留 `is_ambiguous` 和原因，并明确正式指标是否包含它们。

## 3. 数据划分

### 为什么不能随机按照片划分

同一建筑可能有多张照片。如果其中一些用于调参数，另一些进入测试集，系统可能间接利用同一建筑的重复信息，导致测试结果过高。

正确做法：按 `building_id` 分组划分。

```text
调试集建筑：b001、b002、b003...
测试集建筑：b021、b022、b023...
```

同一栋建筑的所有照片只能属于一个集合。

### 三个集合的作用

- **训练集**：只有需要训练或微调模型时使用；
- **调试集/验证集**：选择候选半径、权重和置信度阈值；
- **测试集**：方案确定后只用于最终报告。

如果前期完全不训练模型，可以只使用调试集和测试集，但不能用测试集反复选参数。

## 4. 检索结果格式

建议将每个候选保存为一行：

```text
image_id,rank,predicted_building_id,score,confidence
img_0001,1,b102,0.86,0.72
img_0001,2,b103,0.80,0.72
img_0001,3,b107,0.42,0.72
```

这样可以计算 Top-1、Recall@K、前两名分差和失败案例。

## 5. Top-1 Accuracy

Top-1 表示排名第一的建筑是否等于正确建筑。

```text
Top-1 Accuracy = 排名第一正确的照片数 / 参与评价的照片总数
```

示例：100 张照片中有 76 张第一名正确：

```text
Top-1 Accuracy = 76 / 100 = 0.76
```

Python 示例：

```python
def top1_accuracy(y_true: list[str], y_pred: list[str]) -> float:
    if len(y_true) != len(y_pred):
        raise ValueError("真值和预测数量不一致")
    if not y_true:
        raise ValueError("没有可评价样本")
    correct = sum(true == pred for true, pred in zip(y_true, y_pred))
    return correct / len(y_true)
```

## 6. 错配率

如果每张照片都强制输出一个建筑：

```text
错配率 = 1 - Top-1 Accuracy
```

但引入拒绝机制后，需要区分：

- 错误关联；
- 正确关联；
- 拒绝/人工复核。

此时不能只报告错配率，还要报告覆盖率。

## 7. Recall@K

Recall@K 表示正确建筑是否出现在前 K 个候选中。

```text
Recall@5 = 正确建筑位于前 5 名的照片数 / 照片总数
```

Python 示例：

```python
def recall_at_k(
    y_true: list[str],
    ranked_candidates: list[list[str]],
    k: int,
) -> float:
    if len(y_true) != len(ranked_candidates):
        raise ValueError("真值和候选列表数量不一致")
    if not y_true:
        raise ValueError("没有可评价样本")
    if k <= 0:
        raise ValueError("k 必须为正整数")

    hits = sum(
        true_id in candidates[:k]
        for true_id, candidates in zip(y_true, ranked_candidates)
    )
    return hits / len(y_true)
```

解释：

- Recall@5 低：候选生成阶段就漏掉了正确建筑；
- Recall@5 高但 Top-1 低：正确建筑在候选中，但排序不够好；
- 因此这两个指标分别评价候选生成和候选排序。

## 8. Precision-Coverage

系统可以拒绝低置信度样本，让人工复核。

### Coverage

```text
Coverage = 系统自动给出确定结果的样本数 / 总样本数
```

### Precision

在系统实际接受的样本中：

```text
Precision = 接受结果中正确的样本数 / 被接受的样本数
```

提高置信度阈值通常会：

- 提高 Precision；
- 降低 Coverage；
- 增加人工复核量。

项目不应只选择“准确率最高”的阈值，还应说明该阈值保留了多少覆盖率。

示例表：

| 阈值 | Precision | Coverage | 人工复核比例 |
|---:|---:|---:|---:|
| 0.3 | 0.82 | 0.95 | 0.05 |
| 0.5 | 0.89 | 0.81 | 0.19 |
| 0.7 | 0.95 | 0.54 | 0.46 |

阈值只能在调试集上选择，再在测试集报告结果。

## 9. 基准方法

至少保留：

### 基准 A：最近建筑

只根据相机到建筑轮廓或中心的距离排序。

### 基准 B：GPS + 方向过滤

先根据距离生成候选，再过滤明显不在拍摄方向内的建筑。

### 完整方法

加入视场角、立面朝向、遮挡、影像一致性和置信度。

基准必须使用同一份数据和同一评价脚本。

## 10. 消融实验

消融实验不是“换很多模型”，而是回答每个模块是否有贡献。

推荐顺序：

| 方法 | 说明 |
|---|---|
| Baseline A | 最近建筑 |
| Baseline B | GPS + 方向 |
| B + FOV | 加视场角 |
| B + FOV + Facade | 加立面朝向 |
| + Occlusion | 加遮挡 |
| + Image | 加影像一致性 |
| + Rejection | 加低置信度拒绝 |

报告同一组指标，并分析：

- 模块是否提高平均结果；
- 对哪些场景帮助最大；
- 是否伤害某些样本；
- 增加了多少运行时间。

## 11. GPS 与航向扰动

扰动实验用于模拟元数据误差。

初步方案：

- GPS：0、5、10、20 米；
- 航向：0°、5°、15°、30°。

正式数值应根据采集设备实际误差调整。

原则：

- 对所有方法使用同一组扰动；
- 保存随机种子；
- 每个强度重复多次并报告平均值/波动；
- 不只报告最有利的一次；
- 区分随机误差和固定偏差。

## 12. 失败案例分析

为每个错误样本至少记录一个主要原因：

```text
gps_error
heading_error
dense_buildings
similar_facades
occlusion
citygml_geometry_error
segmentation_error
multiple_targets
unknown
```

推荐输出：

```text
image_id,true_building_id,predicted_building_id,
failure_type,confidence,notes
```

统计每类错误数量，展示典型图片和候选可视化。失败分析可以指导下一步工作，也能防止只挑选成功案例。

## 13. 可复现实验记录

每次正式实验记录：

```text
实验 ID：
日期：
研究问题：
Git commit：
数据版本：
调试/测试划分版本：
配置文件：
完整运行命令：
Python 和依赖版本：
服务器/GPU：
随机种子：
输出目录：
主要指标：
结论：
```

获取 Git commit：

```bash
git rev-parse HEAD
```

所有正式表格最好由评价脚本自动生成，避免从多个终端输出中手工复制数字。

## 14. 正确的实验顺序

```text
确定研究问题
→ 固定数据和真值
→ 实现最简单基准
→ 检查评价脚本
→ 在调试集选择参数
→ 冻结方法和阈值
→ 在测试集运行一次正式评价
→ 做消融和扰动
→ 分析失败案例
→ 写结论与限制
```

不要先在测试集上反复尝试，再把最好结果称为最终结果。

## 15. 项目练习

创建 10 条人工数据：

```python
y_true = ["b1", "b2", "b3", "b4"]
ranked = [
    ["b1", "b2", "b3"],
    ["b5", "b2", "b3"],
    ["b6", "b7", "b3"],
    ["b1", "b2", "b3"],
]
```

完成：

1. 计算 Top-1；
2. 计算 Recall@3；
3. 加入置信度和阈值；
4. 计算不同阈值下的 Precision 和 Coverage；
5. 为每个错误写一个失败类型；
6. 用断言验证手算结果。

随后在 30～50 张真实 pilot 数据上重复同样流程。

## 验收清单

- [ ] 能解释 Top-1 与 Recall@5 的区别；
- [ ] 能按建筑划分数据；
- [ ] 能说明测试集为什么不能用于调参；
- [ ] 能计算错配率；
- [ ] 能解释 Precision-Coverage 权衡；
- [ ] 能设计一个模块消融表；
- [ ] 能设计 GPS/航向扰动实验；
- [ ] 能记录 Git commit、配置和数据版本；
- [ ] 能对失败样本分类。

## 官方资料

- [scikit-learn 模型评价](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [scikit-learn Top-k Accuracy](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.top_k_accuracy_score.html)
- [scikit-learn Precision-Recall](https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html)
- [PyTorch 可复现性说明](https://docs.pytorch.org/docs/stable/notes/randomness.html)

