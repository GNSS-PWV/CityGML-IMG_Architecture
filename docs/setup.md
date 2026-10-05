# 当前流程的安装与运行

当前已有：Grounding DINO 1.6 实验D批量门窗检测、已知立面自动匹配、27栋建筑的216张基础渲染图库。DINO检索、多栋建筑自动选择、自动写出新的LoD3 CityGML仍未实现。

## 本机继续工作

已有环境 `torch_1` 为 Python 3.12.14 / PyTorch 2.8.0+cu126。保持原解释器即可。日常入口：

| 文件 | 作用 | 网络或费用 |
|---|---|---|
| `try_grounding_dino16.py` | 实验D批量门窗检测 | 调用云API，消耗平台额度 |
| `build_render_gallery.py` | 遍历CityGML建立多建筑图库 | 本地CPU；不调用API |
| `render_building_views.py` | 单栋渲染及批量入口共用的几何实现 | 本地CPU |
| `match_facade_roma.py` | 指定照片与已有正面渲染图自动匹配 | 已准备缓存后本地GPU，不收费 |
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

3. **仅第一次准备RoMa缓存**时，联网执行下面命令。它调用官方初始化，会下载约1.1GB权重及固定版本的DINOv3源码；不调用门窗云API。

   ```powershell
   python -c "import torch; from romav2 import RoMaV2; torch.hub.set_dir('model_cache/roma_v2/hub'); torch.set_float32_matmul_precision('highest'); model=RoMaV2(RoMaV2.Cfg(setting='base', compile=False)); print('RoMa cache ready')"
   ```

   权重应位于 `model_cache/roma_v2/hub/checkpoints/romav2.0.1.pt`，SHA256应为：

   ```text
   1557dec0d21b62366465f7ff4d5fdf228cc695d0582e196ad2b80e05230828b7
   ```

   官方来源：[RoMa v2](https://github.com/Parskatt/RoMaV2)。日常 `match_facade_roma.py` 只使用已存在的缓存；已准备好的本机无需重复执行本步骤。

4. 需要云端门窗检测时，在PyCharm运行配置中添加 `DDS_API_TOKEN` 环境变量，值为自己的平台Token。不要写入代码、截图、Git提交或公开文档。已有检测结果时，不需要重新调用API。
5. 运行 `python -m unittest discover -s tests -v`。测试不调用收费API，合成测试文件写入临时目录，不混入正式结果。

## 新机器上建议的执行次序

1. `build_render_gallery.py` 建立多建筑图库，完成后查看 `my_results/render_gallery/latest_run.json` 指向的总览与索引。
2. 当前单图RoMa仍固定使用 `my_results/rendered_buildings/4959323` 的 `front` 视图；运行一次 `render_building_views.py` 准备该样例，然后运行 `match_facade_roma.py`。
3. 若需要结构辅助，先运行实验D检测入口保存当前照片的结果；缺少检测结果时，匹配入口保留纯RoMa路径并报告自动检查状态。

多建筑图库使用 `view_000` 等自动方向命名，不能直接把其中某张当成样例 `front`。后续检索模块将提供建筑、视角和墙面候选，再接入通用RoMa匹配接口。
