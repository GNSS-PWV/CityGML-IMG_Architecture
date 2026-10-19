# 实验 1.1 与实验 2.1：本地窗中心重新对齐

**最新完整批次：**`E:\workinCODEX\prog_3D\my_results\experiment_window_centers\window_centers_20261012_145618_436419`。1.1、2.1 各处理了 17 张；1.1 正确建筑自动映射 **5 张、错配 0 张、待复核 12 张**，2.1 正确映射 **9 张、错配 1 张、待复核 7 张**。相对较早的 2.1 批次，温和放宽窗布局覆盖门槛后新增 `4959323_right.png` 的正确建筑映射；错配 `4906981.jpg` 仍在。新增的少窗回退在这 17 张中**没有触发**。完整对比见 [最新结果快照](results/window_centers_20261012_relaxed/comparison.md)，项目总体说明见 [当前成果](CURRENT_STATUS.md)。

## 温和放宽几何审核的独立试验（2026-10-12）

为了检验窗布局覆盖门槛是否过严，新增可选参数 `--layout-profile mild`。它仅将**窗布局**的训练关联点覆盖面积从 0.30 降到 0.25、照片横向跨度从 0.65 降到 0.58、纵向跨度从 0.60 降到 0.55。RoMa 留出点内点比例仍为 0.45；门布局、窗中心留出误差、墙面有效性、候选分差及三维边界检查均不变。默认 `baseline` 保持上述旧结果，不会因这次试验自动放宽。

完整新跑 17 张后，实验 1.1 仍为 **5 张正确、0 张错配、12 张待复核**；实验 2.1 由 **8/1/8** 变为 **9 张正确、1 张错配、7 张待复核**。唯一新增的是 `4959323_right.png`：先前正确墙候选的稠密匹配已通过，但窗布局面积/横向跨度略低于旧门槛；温和放宽后通过窗中心及三维边界检查，生成 **13 个预测面**，另外 **4 个检测框被三维映射检查拒绝**。建筑 ID 的“正确”只按文件名事后核对；尚无独立墙面或三维位置真值。已有 `4906981.jpg` 错配仍在，因此不宜继续直接降低门槛并宣称整体精度提高。

新批次位于 `E:\workinCODEX\prog_3D\my_results\experiment_window_centers\window_centers_20261012_145618_436419`。重点查看 `experiment_2.1\4959323_right\window_center_overlay.png` 与 `mapping\mapping_preview.png`。在 PyCharm 的**运行 → 编辑配置 → 脚本参数**填写 `--layout-profile mild` 即可独立重跑；不要把旧 `baseline` 批次传给这次试验的 `--resume`。

入口是 [`run_experiment_window_centers.py`](../run_experiment_window_centers.py)。在 PyCharm 中选择已有解释器，打开文件后右键运行即可。默认一次处理全部 17 张立面照片，先做实验 1.1，再做实验 2.1；每张照片处理结束即保存。命令行也可运行 `python run_experiment_window_centers.py`。中断后把**本次新建的**输出目录传给 `--resume <目录>`，已经记录的照片不会重复处理；先前策略的批次不能用新版代码续跑。

这里的 **1.1** 复用实验 1 的 RoMa v2 匹配初值，**2.1** 复用实验 2 的 MINIMA-RoMa 匹配初值。两者都沿用 Grounding DINO 1.6 的既有检测、同一批 0° 模型渲染图和已有建筑检索候选；**没有调用 GPT 视觉模型或付费 API**，也没有重新训练或重新检测照片。

## 重新划分后的职责

这不是用窗户数量重新检索建筑，而是在渲染候选已经筛完之后做局部标定：

```text
照片 + 多建筑渲染图库
  -> 检索 / 稠密特征匹配筛出候选建筑、墙和正视渲染图
  -> 仅对通过该筛选的渲染图提取可见 Window 构件中心
  -> 照片检测窗中心与渲染窗中心作一对一关联
  -> 至少 6 对：6–7 对拟合仿射，8 对以上拟合单应，留出锚点验收
  -> 少于 6 对：原稠密匹配已自动确认该建筑墙面，才复用其原变换
  -> 否则待复核；通过后回投三维墙面
```

这样分层有两个好处：前一层负责“是不是这栋楼/这面墙”，后一层负责“同一墙上每扇窗能否对齐”。重复窗列即使能产生低残差，也不会单凭中心对齐越过前一层的拒绝结果。

对每张照片，程序先取得检测框中的窗中心，以及 CityGML Window 构件在渲染图中的中心。已有 RoMa / MINIMA-RoMa 变换仅用来找到附近可能对应的窗；接着做一对一分配。**6–7 对窗用仿射变换**，避免仅 4 个训练点时用自由度更高的单应矩阵过拟合；**8 对以上用单应矩阵**，处理真实照片相对于正视渲染的透视差。至少 2 对窗完全留出，不参加拟合，只检查位置误差；6–7 对时留出误差门槛更严格。覆盖范围不足、留出误差过大、变换出现极点或镜像，都会保留为待复核。

**少于 6 对窗**时不会用 4 对窗强行拟合。只有原实验的 RoMa/MINIMA 稠密几何、布局、墙面和候选选择已全部自动通过，且原结果已成功映射**同一个候选**，才复用原稠密变换；否则保留 `needs_review`。这种输出的 `alignment_path=dense_fallback`，有 `alignment_overlay.png`，没有窗中心新拟合的 `window_center_overlay.png`。当前 17 张实际批次的稠密回退数为 0；用临时只保留 4 扇检测窗的回放已验证该代码分支能够生成 4 个预测面，不能据此声称新照片匹配准确率。RoMa 留出点内点比例门槛从 0.50 调至 **0.45**；程序会根据旧缓存里的原始检查值重新判定这**一个**门槛，不重算模型对应点，其余检查仍须通过。跨建筑或跨墙面的候选分数接近时也不自动选择。

结果在 `E:\workinCODEX\prog_3D\my_results\experiment_window_centers\window_centers_时间戳`。先打开 `comparison.md` 看四组指标，再打开 `experiment_1.1_per_photo.csv` 和 `experiment_2.1_per_photo.csv` 看逐照片判定，其中 `alignment_path` 标明 `window_centers` 或 `dense_fallback`。每张照片的 `candidates.json` 包含候选墙面的配对数量、留出误差和失败原因；`result.json` 是最终选择。成功的两条路径都有 `alignment_overlay.png` 与 `mapping/` 三维预测面预览，只有窗中心路径有 `window_center_overlay.png`。中心叠加图中绿色圆点是残差合格的真实渲染窗中心，白点是照片窗中心投影位置，红色表示残差超过阈值；请重点检查窗列是否整体错位，不能只看 `mapped` 状态。

## 在 PyCharm 中自己查看或重跑

1. 在项目窗口打开 `E:\workinCODEX\prog_3D\CityGML-IMG_Architecture`，确认右下角解释器为 `C:\Users\YuexuanWang\.conda\envs\torch_1\python.exe`。若只想看结果，先打开上面的完整批次，无需运行代码。
2. 打开 `run_experiment_window_centers.py`，核对顶部 `SOURCE_BATCH` 指向实验 1/2 的同一批完整结果。右键 **运行**，默认 `both` 会建立新的时间戳目录，并从已有的检测、检索、RoMa/MINIMA-RoMa 报告读取数据；不会提交云 API 请求。
3. 控制台每完成一张就打印 `mapped` 或 `needs_review`。若中断，在 PyCharm **运行 → 编辑配置 → 脚本参数**填写 `--resume "E:\workinCODEX\prog_3D\my_results\experiment_window_centers\刚才的批次目录"` 再运行；已经完成的照片会跳过。
4. 完成后按 `comparison.md` → `experiment_1.1_per_photo.csv`/`experiment_2.1_per_photo.csv` → 对应照片的 `result.json` → 该路径的对齐图 → `mapping/mapping_preview.png` 的顺序查看。先读 `alignment_path`：`window_centers` 看 `window_center_overlay.png`，`dense_fallback` 看 `alignment_overlay.png` 和原实验的稠密点诊断。`status=mapped` 仍需结合叠加图和已知建筑 ID 核对。

特别注意：`4906981.jpg` 在实验 2.1 中被自动映射到错误建筑，它是**失败案例**，不能作为成功示例。新通过的 [4906970 少窗仿射核对图](images/window_center_alignment_4906970_relaxed.png)和 [4959322 核对图](images/window_center_alignment_4959322_relaxed.png)可用于查看本轮门槛调整后的视觉效果；中心对齐仍不能独立证明建筑或墙面正确。此前一批中心图曾将另一候选的渲染窗编号误用于绘图，造成很长的绿线；修正后的原样例为 [4907507 核对图](images/window_center_alignment_4907507.png)。

这组实验有明显边界：窗列高度重复时，错楼也可能形成很好的几何拟合。照片文件名里的建筑 ID 只在实验结束后作建筑级核对，不参与候选选择；墙面身份和三维门窗精度没有独立真值。当前三维输出是贴在墙上的预测面，不对源 CityGML 开孔。
