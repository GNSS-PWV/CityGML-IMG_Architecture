# 实验 1、2：仅用 0° 渲染图比较匹配器

这两组是后续 **实验 1.1/2.1 窗中心精对齐**的上游输入。最新完整四组结果、修正后的窗中心核对图与运行入口见 [当前成果](CURRENT_STATUS.md)；本页只说明原实验 1/2 如何生成。

这两组实验处理 `Drills/Texture2LoD3_dataset/textures` 中的 17 张照片。实验 1 用现有的 RoMa v2.0.1；实验 2 用 MINIMA-RoMa。两组共用 145 张 0° 墙面渲染图（27 栋建筑）、DINOv3 建筑检索、实验 D 已保存的门窗检测、Top 5 建筑候选、几何与门窗布局验收规则，以及三维墙面映射。照片文件名中的建筑编号只在推理完成后用于建筑级别评价。

注意：MINIMA-RoMa 是 MINIMA 官方在较早 RoMa 架构上训练的匹配器，并不是把 RoMa v2 的权重换了一下。两个模型的内部特征、输入尺寸和密集匹配策略不同。因此这是一项“匹配器整体替换”对照实验，而不是严格的单变量权重消融。

## 在 PyCharm 运行

1. 用 `torch_1` 解释器打开本仓库，确认工作目录是本仓库根目录。
2. 确认旧的 `experiment_e45_user/wall_view_gallery/latest_run.json` 和其指向的渲染文件还在；程序仅筛选其中的 0° 图像，不重渲染。还需有 17 张照片对应的实验 D `detections.json` 缓存。程序会先逐张核对缓存；缺失就停止，绝不提交收费检测任务。
3. 实验 1：运行 `run_experiment_front0.py --experiment 1`。也可直接右键运行脚本，并在顶部把 `RUN_EXPERIMENTS` 改为 `(1,)`。
4. 实验 2 还需 MINIMA 官方 `minima_roma.pth` 与 DINOv2 ViT-L/14 权重。当前项目的本地路径分别在 `../model_cache/minima/unpacked/MINIMA-main/weights/minima_roma.pth` 和 `../model_cache/minima/dinov2_vitl14_pretrain.pth`，RoMa_minima 官方源码在 `../model_cache/minima/unpacked/MINIMA-main/third_party/RoMa_minima`。本项目不把大型权重提交 Git。安装官方依赖 `kornia==0.6.11` 后，运行 `run_experiment_front0.py --experiment 2`；或者把顶部 `RUN_EXPERIMENTS` 改为 `(2,)`。本地权重不完整会在开始前报错，模型适配器不会联网下载。
5. 两组都想新跑时使用 `--experiment both`，或直接右键运行默认设置。由于两组各 17 张且每张有多个候选，可能耗时较长。程序每完成一张即写结果。中断后将顶部 `RESUME_RUN_DIR` 填为控制台打印的 `experiment_front0_...` 目录，或加 `--resume-run-dir 绝对路径` 续跑。

输出在 `../my_results/experiment_front0/experiment_front0_时间/`：`experiment_1_per_photo.csv`、`experiment_2_per_photo.csv` 分别是 17 张照片的逐张记录；`summary.json` 与 `comparison.md` 是计数对照；`experiment_1_runs`、`experiment_2_runs` 中保留每张照片的检索排序、候选、几何诊断、对齐可视化和三维映射。`state.json` 用于中断续跑。未执行的一组在汇总里显示 `0/17`，不应当解读为模型表现。

## 指标口径

- **正确建筑进入候选**：照片对应建筑 ID 出现在 DINOv3 选出的 Top 5 建筑候选中。真实建筑 ID 由文件名事后核对；检索和匹配本身不读取它。
- **正确建筑几何通过**：该建筑的至少一个墙面候选通过原始密集匹配几何门槛或门窗结构辅助几何门槛。
- **正确建筑通过候选门槛**：几何、布局和墙面支持均满足候选验收。
- **最终自动映射**：选中的候选通过全局分数与候选差距门槛，并成功输出至少一个三维门窗框。
- **错配**：自动映射所选的建筑 ID 与照片建筑 ID 不同。建筑 ID 对了仍不能证明墙面或门窗三维坐标准确。
- **待复核**：自动检查未能安全地选择并映射候选；运行异常另列为 `failed`，不能算待复核，也不能算成功。

每墙仅有一张 0° 图，因此没有独立的同墙第二视角可做交叉确认。两组均显式关闭这一关卡，并在逐张 `result.json` 中写入 `validation_policy=single_0_degree_view`。请勿把两组的自动映射数直接与原来的双视角实验比较。当前也没有每张照片的墙面、位姿和门窗三维真值，评价只覆盖建筑 ID 层面，不能报告三维定位“准确率”。

MINIMA 官方代码与权重：https://github.com/LSXI7/MINIMA；RoMa v2 官方代码：https://github.com/Parskatt/RoMaV2。
