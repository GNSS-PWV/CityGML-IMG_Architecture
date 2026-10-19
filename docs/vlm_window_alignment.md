# 用视觉模型找对应窗户：单张试验教程

这是**尚未进入 17 张批量实验的可选单候选试验**。当前实验 1.1/2.1 采用本地一对一窗中心分配，没有调用本页的视觉模型 API；最新已验证流程和结果见 [当前成果](CURRENT_STATUS.md)。本文件同时提供批量实验复用的中心拟合与核对图函数，但不能因此把本地批量结果称为 GPT 视觉模型结果。

这个入口把已有流程拆成两部分：视觉模型只回答“照片的 P 编号窗对应渲染图的哪个 R 编号窗”；本地程序根据**已有检测框和 CityGML 构件的真实像素中心**拟合照片到渲染图的变换，留出部分窗检查，再调用原有三维映射函数。它不要求视觉模型报像素坐标，也不会改动源 CityGML 或开孔。

## 第一次在 PyCharm 跑

1. 使用你现有的 `torch_1` 解释器打开项目。运行 [vlm_window_alignment.py](../vlm_window_alignment.py)。文件顶部默认 `RUN_DIR` 是 `4959323_front` 的实验 2 单张结果，`CANDIDATE_RANK = 1`，`MODE = "prepare"`。第一次不要改成 `api`。
2. 控制台会显示输出目录。打开其中的 `photo_numbered.png` 和 `render_numbered.png`：`P01`、`P02` 等来自照片上 Grounding DINO 1.6 已保存的窗框；`R01`、`R02` 等来自所选 0° 渲染图的 CityGML Window 构件。数字仅是本次图片的定位标签，**不表示两个相同数字就是对应窗**。
3. 如果先用当前 ChatGPT 手工验证，把两张编号图和 `prompt.txt` 的全文放到同一条消息，让模型只回复 JSON。输出目录中 `pairs_template.json` 是格式范例；将回答另存为 `pairs.json`。例如：

   ```json
   {
     "same_wall": "yes",
     "reason": "楼层、窗列与拱门分组对应",
     "pairs": [
       {"photo": "P21", "render": "R16"},
       {"photo": "P19", "render": "R17"}
     ]
   }
   ```

   上面两行只是**格式演示**，不是经过验证的对应关系。建议给出至少 8 对、分布在墙面左右和上下；若立面只有 6–7 对可靠窗，程序改用仿射拟合并留出至少 2 对检查；少于 6 对不能自动通过。不要把 Markdown 代码块标记复制进 `pairs.json`；看不准时让模型输出 `uncertain` 和空数组。
4. 将脚本顶部的 `MODE` 改成 `"apply"`，再次右键运行。程序会检查编号是否存在且不重复，按配对数拟合仿射或单应矩阵，留出至少两对对应窗作检查。若失败，查看 `alignment_report.json` 中的 `reason`、`checks` 和 `holdout_errors_px`；结果为 `needs_review`，不会继续输出三维门窗。
5. 如果通过，打开 `alignment_overlay.png` 检查窗列是否整体错一列、边缘是否拉斜，再看 `mapping/mapping_preview.png`。`mapping/predictions.json` 和 OBJ 是墙面**预测面预览**；源 CityGML 没有开孔。通过几何检查仍不等于建筑身份或三维精度有独立真值。

默认样例输出到 `E:\workinCODEX\prog_3D\my_results\vlm_window_alignment\my_trial\run_20261010_192522_899052_candidate_01`。修改顶部 `TRIAL_NAME` 可开新试验，避免覆盖此前的编号与对应结果。换照片时，先在 `run_experiment_front0.py` 的逐照片 CSV 找到该照片的 `run_dir`，填到 `RUN_DIR`；用 `CANDIDATE_RANK = 1, 2, ...` 依次检查该次检索得到的候选。只比较 `0°` 候选。不要依据照片文件名里的建筑编号挑候选，否则不能作为自动找楼实验。

当前机器另保存一份**示范**结果于 `my_results/vlm_window_alignment/run_20261010_192522_899052_candidate_01`：对已有检索第 1 候选，ChatGPT 目视给出 45 对窗，15 对留出检查的最大渲染图误差约 11.44 像素，生成 62 个三维预测面预览、拒绝 3 个框。这只证明编号→拟合→预览的代码链可运行；候选已由既有检索提供，配对并未通过 OpenAI API 自动产生，也没有独立墙面真值验证。你自己的 `my_trial` 与这份示范分开保存。

## 让脚本直接调用视觉模型 API

如果你有单独的 OpenAI API 账号和可用的视觉模型，在 PyCharm **运行配置 → 环境变量**设置 `OPENAI_API_KEY`，然后将 `MODE` 改成 `"api"`。此模式会上传两张编号图并产生 API 费用；`prepare` 和 `apply` 不调用 API。脚本使用已有的 `requests` 依赖，不要求额外安装 OpenAI SDK。模型名由顶部 `OPENAI_MODEL` 指定；账号无法使用该模型时，请换成账号支持且能处理两张图片和结构化输出的模型。API 图片输入及 JSON Schema 写法参见 [OpenAI 官方图像文档](https://developers.openai.com/api/docs/guides/images-vision)与[结构化输出文档](https://developers.openai.com/api/docs/guides/structured-outputs)。

调用前脚本会写 `api_request_started.json`。如果网络超时或结果不明，重跑会停下来，防止同一候选重复扣费；先查看 API 平台调用记录。成功返回会保存在 `api_pairs.json`，再次运行会复用它。手工 ChatGPT 回答仍放在 `pairs.json`，二者互不覆盖。密钥只从环境变量读取，不写入结果或 Git。

## 这一步的边界

- 这是**一个候选墙面的可复现试验**，不是已经实现 17 张照片的全自动多建筑定夺。多个建筑都给出类似窗列时，还需比较候选之间的证据，保留“待复核”。
- 模型输出的编号对、几何留出检查以及肉眼叠加图都可能被规则窗列的整列错位欺骗；不能把 `preview_created` 当成独立定位真值。
- 当前门窗三维结果仍是四边形预测面，不对墙体进行布尔开洞。若此路线在失败照片上有效，再批量化并与 RoMa、MINIMA-RoMa 在同一 17 张上比较。
