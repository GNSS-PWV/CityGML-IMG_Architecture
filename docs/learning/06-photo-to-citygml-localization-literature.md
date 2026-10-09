# 真实立面照片与已有 CityGML 建筑自动匹配：文献笔记

资料核对日期：2026-10-10。以下区分**论文已验证的做法**与**对本项目的推断**。这些论文的测试场景、模型精度和定位指标不同，不能把论文中的成功率直接当成本项目的预期成绩。

## 本项目的问题定位

当前流程是：真实立面照片 → DINOv3 检索 27 栋建筑的渲染视角 → RoMa v2 建立照片—渲染图对应 → 几何、双视角和门窗布局检查 → 将检测框投影到 CityGML 墙面。现有文档明确指出，检索只是候选筛选；几何匹配未通过的照片须待复核，不能强制投影。[本项目流程说明](../pipeline.md)

对 2026-10-09 完成的 ±45° 批次逐图核对：17 张照片中，按文件名所含建筑编号**仅在事后评估**所得的正确建筑进入 Top-5 候选 14 张，3 张未进入；最终自动映射 3 张、待复核 14 张。待复核原因中，13 张为 `no_candidate_passed_geometry`，1 张为 `ambiguous_candidates`。因此检索与后续配准都要检查，但不能将全部失败归因于检索模型。文件名编号不参与实际推理。证据是本机 `my_results/experiment_e45_user/wall_view_pipeline/batch_20261009_203320_750873/summary.json` 及各张 `retrieval.json`、`result.json`；本地运行结果未随这份笔记上传。[本项目流程说明](../pipeline.md)

要把失败原因分开看：①正确建筑是否进入检索候选；②候选墙面是否可见且渲染视角充分；③真实照片与渲染图的跨域对应是否稳定；④是否有足够的、不重复的几何约束来确定墙面和相机；⑤门窗检测本身是否可靠。这是依据本项目流程作出的诊断划分，不是下述论文的原文结论。[本项目流程说明](../pipeline.md)

## 六篇优先阅读的论文

### 1. Loeper 等，2026：低细节、无纹理 CityGML 与照片的跨域定位

**准确标题：** *Using textureless, low-detailed 3D city models for visual localization*，ISPRS Archives，2026。[论文及 DOI](https://isprs-archives.copernicus.org/articles/XLIX-B4-2026/407/2026/)

**做法和证据：** 作者把真实照片和城市模型渲染图的语义掩膜 IoU 加入候选位姿评分，并系统比较传统局部特征、学习式匹配器和针对跨模态数据训练的匹配器。论文摘要报告：跨模态匹配器是所测方法中最好的一组；在其测试数据上，四个匹配器分别有 20%、10%、9%、7% 的查询达到“平移误差 <5 m 且旋转误差 <5°”，其他所测方法最高为 1.4%。**这些数字不是本项目 17 张照片的结果，也不足以证明任何方法在本数据上一定成功。**[论文摘要](https://isprs-archives.copernicus.org/articles/XLIX-B4-2026/407/2026/)

**对本项目的直接启发（推断）：** 最值得做的是一个小规模、同数据同候选的对照：在现有 RoMa 之外，测试论文列出的跨模态匹配器（如 MINIMA-RoMa、MatchAnything）；另把真实照片与渲染图都转为一致的建筑/天空/开口语义掩膜，对候选位姿再评分。当前图库具有门窗几何，细节可能高于论文的低细节模型，但照片—渲染图的外观差异仍存在。论文页面未提供可直接运行的官方代码入口，复现成本需要另行核实。[论文方法与结果](https://isprs-archives.copernicus.org/articles/XLIX-B4-2026/407/2026/)

### 2. Panek 等，2022：MeshLoc，用渲染图深度得到真正的 2D–3D 对应

**准确标题：** *MeshLoc: Mesh-Based Visual Localization*，ECCV，2022。[会议论文](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/5648_ECCV_2022_paper.php) · [作者/团队代码](https://github.com/tsattler/meshloc_release)

**做法和证据：** 从带已知相机位姿的模型渲染图检索候选，用照片—渲染图的 2D–2D 匹配，加上渲染深度把匹配点提升为 2D–3D 对应，再用鲁棒位姿求解器估计查询相机的 6-DoF 位姿。论文还测试无颜色、无纹理的几何渲染，说明并非必须依赖逼真纹理。[论文](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136820573.pdf) · [官方实现说明](https://github.com/tsattler/meshloc_release)

**对本项目的直接启发（推断）：** 现有图库已经保存渲染相机和深度信息，适合在 RoMa 匹配之后增加“匹配点 → 深度反投影 → PnP/RANSAC → 重投影/可见性检查”实验。这比只在单面墙上拟合单应矩阵多了一条三维几何验证途径；但若模型与真实场景有明显几何差异，或重复窗列导致错误对应，PnP 同样会失败。官方代码主要面向 Aachen 和 12 Scenes，移植到本项目需要自建数据接口。[本项目图库说明](../pipeline.md) · [MeshLoc 代码支持范围](https://github.com/tsattler/meshloc_release)

**实施前提：** MeshLoc 的查询输入需要相机内参；本项目的立面纹理图可能经过裁切或正视矫正，不能未经核实就当作原始针孔相机照片。应先查明原图/全景图、裁切变换与内参；若没有可用相机模型，先做立面平面配准与结构验证。此外，若所有对应点都落在同一面平墙上，2D–3D 求姿态仍可能存在平面歧义；应优先使用墙角、侧墙、屋檐等非共面结构作验证，不能把 PnP 当成自动消除重复窗错配的方法。[MeshLoc 查询文件要求](https://github.com/tsattler/meshloc_release)

### 3. Panek 等，2023：CADLoc，系统评估不完美三维模型的影响

**准确标题：** *Visual Localization Using Imperfect 3D Models From the Internet*，CVPR，2023。[论文](https://openaccess.thecvf.com/content/CVPR2023/html/Panek_Visual_Localization_Using_Imperfect_3D_Models_From_the_Internet_CVPR_2023_paper.html) · [官方 CADLoc 基准/渲染脚本](https://github.com/v-pnk/cadloc)

**做法和证据：** 使用不同外观与几何保真度的现成三维模型，按照“生成带深度的参考渲染图 → 图像检索 → 实景/渲染局部匹配 → 2D–3D 位姿估计”评估定位。作者明确指出缺少真实纹理、几何过度简化或比例失真会损害定位，并发布基准数据和渲染/评估脚本。[论文](https://openaccess.thecvf.com/content/CVPR2023/papers/Panek_Visual_Localization_Using_Imperfect_3D_Models_From_the_Internet_CVPR_2023_paper.pdf) · [基准仓库](https://github.com/v-pnk/cadloc)

**对本项目的直接启发（推断）：** 在替换匹配网络之前，先审计几个失败样本的 CityGML/OBJ：真实照片中的立面饰线、拱窗、底层门、邻楼遮挡，在模型和渲染图中是否存在、位置是否准确。若模型缺了关键结构，继续调 RoMa 阈值无法补回不存在的几何。CADLoc 仓库是基准与工具，不是针对本数据“下载即用”的权重。[论文](https://openaccess.thecvf.com/content/CVPR2023/html/Panek_Visual_Localization_Using_Imperfect_3D_Models_From_the_Internet_CVPR_2023_paper.html) · [本项目流程说明](../pipeline.md)

### 4. Abe 等，2025：NormalLoc，用表面法线图缩小照片—渲染图模态差异

**准确标题：** *NormalLoc: Visual Localization on Textureless 3D Models using Surface Normals*，ICCV，2025。[论文](https://openaccess.thecvf.com/content/ICCV2025/html/Abe_NormalLoc_Visual_Localization_on_Textureless_3D_Models_using_Surface_Normals_ICCV_2025_paper.html)

**做法和证据：** 从模型表面法线渲染参考图，学习照片与法线图之间的全局描述子和局部匹配，而不是要求渲染颜色与真实照片接近；论文在无纹理 CAD 和城市尺度网格模型上评估。[论文全文](https://openaccess.thecvf.com/content/ICCV2025/papers/Abe_NormalLoc_Visual_Localization_on_Textureless_3D_Models_using_Surface_Normals_ICCV_2025_paper.pdf)

**对本项目的直接启发（推断）：** 当渲染颜色、光照、材质与照片差得很远时，可从现有模型额外导出法线、深度、轮廓图作为参考模态。但**直接把法线图输入当前 RoMa 并不等于复现 NormalLoc**：论文依赖相应的跨模态训练方案。所查论文页面未列出可核实的官方实现/权重入口，宜先读原理，暂不把它列为最容易复跑的方案。[论文方法](https://openaccess.thecvf.com/content/ICCV2025/html/Abe_NormalLoc_Visual_Localization_on_Textureless_3D_Models_using_Surface_Normals_ICCV_2025_paper.html)

### 5. Fond 等，2021：利用窗、门等立面语义联合修正分割和配准

**准确标题：** *Model-image registration of a building’s facade based on dense semantic segmentation*，Computer Vision and Image Understanding，2021。[期刊论文](https://www.sciencedirect.com/science/article/pii/S1077314221000291)

**做法和证据：** 从粗略相机位姿、消失点和立面检测初始化，随后把有语义标签的参考立面与照片语义分割图配准。其 EM 框架使用每个像素对多个类别的分类分数，而非只取最高分；窗和门等结构元素、几何一致性及遮挡共同参与优化。作者明确举“门略高分被误分类为窗”的问题。[期刊正文/方法](https://www.sciencedirect.com/science/article/pii/S1077314221000291)

**对本项目的直接启发（推断）：** 这直接对应本项目的门窗二义性和重复窗列。可保留“窗/门”两类检测分数，把照片预测的开口分布与 CityGML 已有开口布局联合评分；但论文方法属于**已有粗定位后的精配准**，不能单独从 27 栋中直接找到正确建筑。所查期刊页面未列出官方代码入口。[期刊正文](https://www.sciencedirect.com/science/article/pii/S1077314221000291)

### 6. Loeper 等，2024：直接让照片直线与 CityGML 投影线对齐

**准确标题：** *Visual localization in urban environments employing 3D city models*，ISPRS Archives，2024。[论文及 DOI](https://isprs-archives.copernicus.org/articles/XLVIII-2-W8-2024/311/2024/index.html)

**做法和证据：** 从 CityGML LoD2 边界表示提取建筑轮廓线，给定候选位姿投影到照片，比较照片线段与投影线段并搜索相机位姿。实验假设有粗位姿先验，将 XY 搜索范围限制在 20×20 m 或 40×40 m，并限制相机方向；作者仅报告可获得粗定位，也指出线段评分存在歧义。[论文全文，方法与实验限制](https://isprs-archives.copernicus.org/articles/XLVIII-2-W8-2024/311/2024/isprs-archives-XLVIII-2-W8-2024-311-2024.pdf)

**对本项目的直接启发（推断）：** 可把屋檐、墙角、层间横线、拱窗边缘等长结构线作为 RoMa 之后的独立验证，减少“窗口整体错一列仍能匹配”的风险。如果没有大致建筑区域和相机方向，照搬其全局位姿搜索会很昂贵、也可能产生相似建筑错配。所查论文页面未提供官方代码入口。[论文方法](https://isprs-archives.copernicus.org/articles/XLVIII-2-W8-2024/311/2024/isprs-archives-XLVIII-2-W8-2024-311-2024.pdf)

## 对本项目的建议实验顺序（基于上述论文的推断，不是既有结果）

1. **先做失败定位。** 对 17 张逐图记下正确建筑在检索 Top-1/Top-5/Top-10 的位置、正确墙面是否在尝试列表、RoMa 对应可视化、最终拒绝原因。若正确建筑没进候选，优先改检索；若已进候选但几何失败，优先改匹配与验证。[分层定位思想与官方 HLoc 工具](https://github.com/cvg/Hierarchical-Localization) · [本项目现有诊断文件](../pipeline.md)
2. **用固定候选做跨模态匹配对照。** 选正确建筑已进入候选但 RoMa 失败的照片，在相同渲染图、分辨率、相机和验收门槛下比较 RoMa 与论文列出的跨模态匹配器；记录匹配分布和独立几何检验，而非只看匹配点数。[Loeper 等，2026](https://isprs-archives.copernicus.org/articles/XLIX-B4-2026/407/2026/)
3. **把“颜色像不像”与“结构合不合”分开。** 候选渲染图增加语义掩膜/轮廓线；照片端生成对应掩膜/结构线，检查屋顶、墙体、开口和层间结构是否共同一致。需要注意真实照片可能只截取立面的一部分，不能要求整栋外轮廓完全重叠。[Loeper 等，2026](https://isprs-archives.copernicus.org/articles/XLIX-B4-2026/407/2026/) · [Fond 等，2021](https://www.sciencedirect.com/science/article/pii/S1077314221000291) · [Loeper 等，2024](https://isprs-archives.copernicus.org/articles/XLVIII-2-W8-2024/311/2024/index.html)
4. **对已获可靠 2D–2D 对应的候选增加 2D–3D 位姿验证。** 用渲染深度、相机参数和 CityGML 几何得到三维点，以 PnP/RANSAC 与回投误差验证候选建筑和墙面；对同一面重复窗列要检查空间覆盖、深度一致性与邻楼遮挡。[MeshLoc](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/5648_ECCV_2022_paper.php) · [CADLoc](https://openaccess.thecvf.com/content/CVPR2023/html/Panek_Visual_Localization_Using_Imperfect_3D_Models_From_the_Internet_CVPR_2023_paper.html)
5. **保留独立测试。** 现有照片已参与调参，应把上述变化在开发集与未参与调参的新照片上分开报告：正确建筑候选召回、正确墙面候选召回、最终自动接受数、错配数、待复核数及有真值时的相机/墙面定位误差。不同论文的定位阈值不可直接与本项目“自动映射数”互比。[CADLoc 评测形式](https://github.com/v-pnk/cadloc) · [本项目实验边界](../experiment_log.md)

**补充阅读：** [From Coarse to Fine: Robust Hierarchical Localization at Large Scale（CVPR 2019）](https://openaccess.thecvf.com/content_CVPR_2019/html/Sarlin_From_Coarse_to_Fine_Robust_Hierarchical_Localization_at_Large_Scale_CVPR_2019_paper.html) 解释“全局检索 → 局部对应 → PnP”的经典路线；[3D Line Mapping Revisited（CVPR 2023）](https://openaccess.thecvf.com/content/CVPR2023/html/Liu_3D_Line_Mapping_Revisited_CVPR_2023_paper.html) 及其 [LIMAP 官方实现](https://github.com/cvg/limap) 说明如何将点与线一起用于定位，但其完整三维线图构建不等于直接读取现有 CityGML 边线；[Analyzing the impact of semantic LoD3 building models on image-based vehicle localization（ISPRS Annals 2024）](https://isprs-annals.copernicus.org/articles/X-4-W5-2024/55/2024/index.html) 比较 LoD2/LoD3 对可匹配特征的影响，支持先检查模型是否包含真实照片中的关键立面结构。
