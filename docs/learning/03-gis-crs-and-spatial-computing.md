# GIS、坐标系统与空间计算

## 学习目标

完成本教程后，你应该能够：

- 区分经纬度坐标和投影坐标；
- 理解 CRS、EPSG、单位和坐标轴顺序；
- 使用 pyproj 将 GPS 转换到目标投影坐标系；
- 使用 Shapely 表示点、线和建筑轮廓；
- 使用 GeoPandas 做附近建筑查询；
- 在计算距离前检查数据是否处于同一 CRS；
- 使用 QGIS 对坐标转换结果进行目视核验。

## 1. 为什么这是项目前期最重要的知识

本项目需要比较：

- 照片的拍摄位置；
- CityGML 建筑的位置；
- 相机到建筑的距离和方向；
- 建筑是否落在相机视野中。

如果照片使用经纬度、建筑使用米制投影坐标，而代码直接把两组数字相减，程序仍然可能运行，但结果没有物理意义。

因此，空间计算的第一条规则是：

> 进行距离、缓冲区、面积和方向计算前，先确认所有对象处于同一坐标参考系，并确认单位。

## 2. 坐标和坐标参考系

坐标只是数字，例如：

```text
(121.5012, 31.2850)
```

只有知道 CRS 后，才能解释：

- 哪个数字是经度；
- 哪个数字是纬度；
- 单位是度还是米；
- 使用什么椭球和基准；
- 坐标覆盖哪个区域。

### 地理坐标系

地理坐标通常用经度和纬度表示，单位是度。GPS 数据经常以 WGS84 表示，例如 EPSG:4326。

经纬度不是平面米制坐标。下面的做法是错误的：

```python
distance = ((lon2 - lon1) ** 2 + (lat2 - lat1) ** 2) ** 0.5
```

结果的单位是“度”，不是米，而且经度一度在不同纬度代表的距离不同。

### 投影坐标系

投影坐标将地球表面映射到平面，常以米作为单位，更适合局部距离、面积、缓冲区和邻近查询。

项目应使用与 CityGML 一致、适用于研究区的投影 CRS。具体 EPSG 不能凭猜测，应从数据 `srsName`、元数据或老师处确认。

## 3. EPSG 编号

EPSG 代码是坐标参考系的常用标识。例如：

```text
EPSG:4326
```

但不能看到 GPS 就假定所有数据都是 EPSG:4326。CityGML 可能使用地方投影、国家坐标系或复合三维坐标系。

项目数据清单至少记录：

| 数据 | 原始 CRS | 单位 | 轴顺序 | 来源 |
|---|---|---|---|---|
| 照片 GPS | 待确认 | 度 | 经度、纬度 | EXIF/采集设备 |
| CityGML | 待确认 | 待确认 | 待确认 | `srsName`/数据说明 |
| 高程 | 待确认 | 米？ | Z | 数据说明 |

## 4. 坐标轴顺序

常见程序接口使用 `(x, y)`，对于经纬度通常对应 `(longitude, latitude)`。但某些 CRS 定义或数据文件可能使用其他顺序。

使用 pyproj 时，建议明确使用 `always_xy=True`：

```python
from pyproj import Transformer

transformer = Transformer.from_crs(
    "EPSG:4326",
    "<目标 EPSG>",
    always_xy=True,
)

x, y = transformer.transform(longitude, latitude)
```

`always_xy=True` 表示输入采用传统 GIS 顺序：经度/东向坐标在前，纬度/北向坐标在后。

不能把 `<目标 EPSG>` 原样放进正式代码，必须替换成真实数据使用的 CRS。

## 5. 使用 pyproj

安装：

```bash
python -m pip install pyproj
```

查看 CRS：

```python
from pyproj import CRS

crs = CRS.from_user_input("EPSG:4326")
print(crs.name)
print(crs.axis_info)
print(crs.is_geographic)
print(crs.is_projected)
```

坐标转换：

```python
from pyproj import Transformer

source_crs = "EPSG:4326"
target_crs = "<CityGML 使用的投影 CRS>"

transformer = Transformer.from_crs(
    source_crs,
    target_crs,
    always_xy=True,
)

longitude = 121.50
latitude = 31.28
x, y = transformer.transform(longitude, latitude)

print(x, y)
```

转换后必须检查：

- 数值量级是否合理；
- 单位是否是米；
- 点是否落在研究区；
- 与 QGIS 显示结果是否一致。

## 6. 高程与三维坐标

水平坐标转换正确，不代表高程一定正确。高程可能使用：

- 椭球高；
- 正常高或海拔；
- 地方高程基准；
- 相对高度。

前期最近建筑和二维候选检索可以先只使用 `(x, y)`。但必须保留原始 `z` 和高程说明，后续立面投影和视线计算会用到。

不要在不了解高程基准时直接混合 GPS altitude 和 CityGML Z。

## 7. Shapely 几何

安装：

```bash
python -m pip install shapely
```

常用对象：

```python
from shapely.geometry import Point, Polygon

camera = Point(100.0, 200.0)
footprint = Polygon([
    (110.0, 190.0),
    (130.0, 190.0),
    (130.0, 210.0),
    (110.0, 210.0),
])

print(camera.distance(footprint))
print(footprint.centroid)
print(footprint.area)
```

### 到轮廓的距离与到中心的距离

这两个量不同：

```python
distance_to_footprint = camera.distance(footprint)
distance_to_centroid = camera.distance(footprint.centroid)
```

狭长建筑或大型建筑中，中心点可能离道路很远。基准实验应明确使用哪种距离，最好同时比较。

### 有效性检查

```python
if not footprint.is_valid:
    raise ValueError("建筑轮廓几何无效")

if footprint.is_empty:
    raise ValueError("建筑轮廓为空")
```

## 8. GeoPandas 与 CRS

安装：

```bash
python -m pip install geopandas
```

创建相机点：

```python
import geopandas as gpd
from shapely.geometry import Point

cameras = gpd.GeoDataFrame(
    {"image_id": ["img_0001"]},
    geometry=[Point(121.50, 31.28)],
    crs="EPSG:4326",
)
```

转换 CRS：

```python
cameras_projected = cameras.to_crs("<CityGML 使用的 CRS>")
```

检查：

```python
print(cameras.crs)
print(cameras_projected.crs)
print(cameras_projected.total_bounds)
```

如果两个 GeoDataFrame 的 CRS 不一致，不应直接进行距离或空间连接。

## 9. 附近建筑候选

最简单的候选方法：对每个相机点，选择半径 30 米以内的建筑。

概念上：

```python
candidate_area = camera.buffer(30.0)
candidates = buildings[buildings.intersects(candidate_area)]
```

注意：只有在 CRS 单位为米时，`30.0` 才表示 30 米。

数据较大时使用 GeoPandas 空间索引，不要手工对所有照片和所有建筑做双重循环。但在 10 张照片、几十栋建筑的最小练习中，简单循环可以接受，先验证正确性。

## 10. 方向与方位角

已知相机 `(x_c, y_c)` 和建筑目标点 `(x_b, y_b)`：

```python
import math

dx = x_b - x_c
dy = y_b - y_c

# 从正北开始，顺时针增加的方位角
bearing = math.degrees(math.atan2(dx, dy)) % 360.0
```

必须确认采集设备的 heading 是否同样采用“正北为 0°、顺时针增加”。如果设备定义不同，应先转换到统一约定。

两个角度的最小差：

```python
def angular_difference_degrees(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)
```

最小测试：

```python
assert angular_difference_degrees(5.0, 355.0) == 10.0
assert angular_difference_degrees(90.0, 100.0) == 10.0
```

## 11. 使用 QGIS 做目视核验

代码结果必须与 GIS 软件交叉检查：

1. 打开 CityGML 转换得到的建筑图层；
2. 设置正确 CRS；
3. 导入照片点 CSV；
4. 指定经度和纬度字段；
5. 将照片点重投影到建筑 CRS；
6. 随机检查 10 个点；
7. 查看是否位于实际道路或采集位置附近；
8. 检查最近建筑是否符合照片内容。

如果整体出现固定方向或固定距离偏移，应优先检查 CRS、轴顺序、基准和坐标来源，而不是立刻修改算法参数。

## 12. 项目练习

使用 10 张照片和对应建筑：

1. 读取 GPS；
2. 转换到 CityGML CRS；
3. 保存为 GeoJSON；
4. 在 QGIS 中与建筑叠加；
5. 计算相机到最近建筑轮廓的距离；
6. 输出最近的 5 栋建筑及距离；
7. 人工检查结果；
8. 记录至少一个异常样本。

输出表：

```text
image_id,candidate_rank,building_id,distance_m
img_0001,1,b_102,8.4
img_0001,2,b_103,13.7
```

## 验收清单

- [ ] 能解释经纬度为什么不能直接计算米制距离；
- [ ] 能从数据中找到 CRS 信息；
- [ ] 能解释 `always_xy=True`；
- [ ] 能使用 pyproj 转换一个坐标；
- [ ] 能使用 Shapely 计算点到建筑轮廓距离；
- [ ] 能在 QGIS 中核验转换结果；
- [ ] 能实现半径候选和 Top-5 最近建筑；
- [ ] 能说明项目航向角约定。

## 官方资料

- [pyproj CRS API](https://pyproj4.github.io/pyproj/stable/api/crs/index.html)
- [pyproj Transformer](https://pyproj4.github.io/pyproj/stable/api/transformer.html)
- [Shapely 用户手册](https://shapely.readthedocs.io/en/stable/manual.html)
- [GeoPandas 投影说明](https://geopandas.org/en/stable/docs/user_guide/projections.html)
- [QGIS 投影教程](https://docs.qgis.org/latest/en/docs/training_manual/vector_analysis/reproject_transform.html)

