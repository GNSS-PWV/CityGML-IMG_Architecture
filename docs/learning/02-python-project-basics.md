# Python 项目基础

## 学习目标

完成本教程后，你应该能够：

- 创建隔离的 Python 环境；
- 运行脚本和模块；
- 使用函数、列表、字典和路径；
- 读取 CSV、JSON 和图像元数据；
- 使用 NumPy/pandas 处理表格与数值；
- 为脚本提供命令行参数；
- 输出清晰错误并编写一个最小测试。

## 1. Python 在本项目中的作用

Python 负责连接整个实验流程：

```text
读取照片元数据
→ 读取 CityGML 建筑
→ 坐标转换和几何计算
→ 候选建筑排序
→ 输出 CSV
→ 计算指标和绘图
```

前期不需要学习复杂 Web 框架、异步编程或元编程。

## 2. 创建虚拟环境

虚拟环境把本项目的依赖与其他项目隔离。

Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Linux 服务器：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

确认解释器：

```bash
python --version
python -c "import sys; print(sys.executable)"
```

退出环境：

```bash
deactivate
```

不要把 `.venv/` 提交到 Git。

## 3. 数据类型

项目中最常用的是：

```python
image_id = "img_0001"       # str
longitude = 121.50          # float
heading = 90.0              # float
is_ambiguous = False        # bool
candidates = ["b01", "b02"]  # list
metadata = {                # dict
    "image_id": image_id,
    "longitude": longitude,
    "heading": heading,
}
```

建议：

- ID 使用字符串，不进行加减；
- 坐标、距离和角度使用浮点数；
- 缺失值不要随便写成 0；
- 字段名在全组内保持一致。

## 4. 条件、循环和函数

```python
def is_valid_heading(heading: float) -> bool:
    return 0.0 <= heading < 360.0


headings = [0.0, 90.0, -1.0, 361.0]

for heading in headings:
    if is_valid_heading(heading):
        print(f"有效航向：{heading}")
    else:
        print(f"无效航向：{heading}")
```

函数应当：

- 名称说明作用；
- 输入和输出清楚；
- 一次只做一件主要事情；
- 对关键单位写在注释或文档中。

例如：

```python
def normalize_heading_degrees(heading: float) -> float:
    """将航向角归一化到 [0, 360) 度。"""
    return heading % 360.0
```

## 5. 使用 pathlib 处理路径

不要手工拼接 Windows 和 Linux 路径。

```python
from pathlib import Path

project_root = Path(__file__).resolve().parent
metadata_path = project_root / "data" / "metadata.csv"

if not metadata_path.exists():
    raise FileNotFoundError(f"找不到数据表：{metadata_path}")
```

读取文件列表：

```python
image_dir = Path("data/images")
image_paths = sorted(image_dir.glob("*.jpg"))
print(f"找到 {len(image_paths)} 张照片")
```

## 6. 读取 CSV

### 使用标准库

数据量很小时，标准库已经够用：

```python
import csv
from pathlib import Path

path = Path("data/metadata.csv")

with path.open("r", encoding="utf-8-sig", newline="") as file:
    rows = list(csv.DictReader(file))

print(rows[0]["image_id"])
```

`utf-8-sig` 可以兼容部分由 Excel 导出的 UTF-8 CSV。

### 使用 pandas

```python
import pandas as pd

df = pd.read_csv("data/metadata.csv")
print(df.head())
print(df.dtypes)
print(df.isna().sum())
```

检查必需字段：

```python
required = {"image_id", "longitude", "latitude", "heading", "building_id"}
missing = required - set(df.columns)

if missing:
    raise ValueError(f"缺少字段：{sorted(missing)}")
```

检查重复 ID：

```python
duplicates = df[df["image_id"].duplicated(keep=False)]
if not duplicates.empty:
    print("发现重复 image_id：")
    print(duplicates[["image_id", "image_path"]])
```

## 7. JSON 配置

前期可以使用标准 JSON，避免为了配置额外增加依赖。

`configs/pilot.json`：

```json
{
  "metadata_path": "data/metadata.csv",
  "candidate_radius_m": 30.0,
  "top_k": 5
}
```

读取：

```python
import json
from pathlib import Path

with Path("configs/pilot.json").open("r", encoding="utf-8") as file:
    config = json.load(file)

print(config["candidate_radius_m"])
```

正式实验不要悄悄在代码里改参数，应当修改配置并保留配置文件。

## 8. NumPy 基础

NumPy 用于向量和矩阵计算：

```python
import numpy as np

camera_xy = np.array([100.0, 200.0])
building_xy = np.array([112.0, 205.0])

distance = np.linalg.norm(building_xy - camera_xy)
print(distance)
```

计算单位方向：

```python
direction = building_xy - camera_xy
length = np.linalg.norm(direction)

if length == 0:
    raise ValueError("相机与建筑点重合，无法计算方向")

unit_direction = direction / length
```

向量运算前必须确认：

- 坐标在同一 CRS；
- 单位相同；
- 轴顺序相同；
- 数组形状符合预期。

## 9. 命令行参数

使用 `argparse` 让脚本可重复运行：

```python
import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"读取：{args.metadata}")
    print(f"写入：{args.output}")
```

运行：

```bash
python check_metadata.py --metadata data/metadata.csv --output outputs/report.json
```

## 10. 错误处理

错误应当尽早出现，并说明问题和文件位置：

```python
if not 0.0 <= heading < 360.0:
    raise ValueError(
        f"image_id={image_id} 的 heading={heading}，应位于 [0, 360)"
    )
```

不要使用：

```python
try:
    risky_operation()
except Exception:
    pass
```

这种写法会把真实错误隐藏掉。

## 11. 最小测试

几何或角度代码必须留下一个能够失败的检查：

```python
def normalize_heading_degrees(heading: float) -> float:
    return heading % 360.0


def test_normalize_heading_degrees() -> None:
    assert normalize_heading_degrees(0.0) == 0.0
    assert normalize_heading_degrees(360.0) == 0.0
    assert normalize_heading_degrees(-10.0) == 350.0
```

前期可以直接运行小测试函数。项目稳定后再统一使用 `pytest`。

## 12. 推荐脚本结构

```python
from pathlib import Path


def load_data(path: Path):
    ...


def validate_data(data):
    ...


def save_report(report, path: Path) -> None:
    ...


def main() -> None:
    ...


if __name__ == "__main__":
    main()
```

避免把所有逻辑直接写在文件顶层，也不需要过早建立复杂的类继承体系。

## 13. 项目练习：数据检查脚本

编写 `check_metadata.py`，完成：

1. 接收 CSV 路径；
2. 检查必需字段；
3. 统计照片数量和建筑数量；
4. 检查重复 `image_id`；
5. 检查缺失 GPS、航向和 `building_id`；
6. 检查航向是否在 `[0, 360)`；
7. 将检查结果保存为 JSON；
8. 对一条正常数据和一条异常数据留下最小测试。

预期输出示例：

```json
{
  "image_count": 30,
  "building_count": 12,
  "duplicate_image_ids": 0,
  "missing_heading": 2,
  "invalid_heading": 1
}
```

## 验收清单

- [ ] 能创建和激活虚拟环境；
- [ ] 能使用 `pathlib` 处理路径；
- [ ] 能读取和检查 CSV；
- [ ] 能读取 JSON 配置；
- [ ] 能使用 NumPy 计算二维距离；
- [ ] 能通过命令行参数运行脚本；
- [ ] 错误信息能指出具体样本；
- [ ] 至少有一个可运行的测试。

## 官方资料

- [Python 官方中文教程](https://docs.python.org/zh-cn/3/tutorial/index.html)
- [Python pathlib](https://docs.python.org/zh-cn/3/library/pathlib.html)
- [Python csv](https://docs.python.org/zh-cn/3/library/csv.html)
- [Python argparse 教程](https://docs.python.org/zh-cn/3/howto/argparse.html)
- [NumPy 快速入门](https://numpy.org/doc/stable/user/quickstart.html)
- [pandas 入门教程](https://pandas.pydata.org/docs/getting_started/intro_tutorials/)

