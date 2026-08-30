# CityGML 与 LoD1 建筑模型

## 学习目标

完成本教程后，你应该能够：

- 解释 CityGML 与普通三维模型的区别；
- 理解建筑对象、语义、几何和 LoD；
- 检查 CityGML 文件版本与 `srsName`；
- 理解 XML 命名空间和 `gml:id`；
- 从项目数据中提取建筑 ID 和坐标；
- 将 LoD1 建筑简化为 footprint、高度和近似立面；
- 识别解析中常见的坐标、几何和版本问题。

## 1. CityGML 是什么

普通 OBJ/GLB 模型主要关注顶点、面、材质和纹理。CityGML 除几何外，还关注城市对象的语义和关系，例如：

- 这是一个建筑，而不是任意网格；
- 建筑有唯一 ID；
- 建筑可能有墙面、屋顶和地面；
- 同一对象可以有不同细节层级；
- 对象可以带有用途、高度等属性。

本项目需要的关键能力不是“把城市渲染得很漂亮”，而是：

> 从 CityGML 中找到具有唯一标识的建筑，并利用其位置、轮廓和高度生成候选与可见性约束。

## 2. 先确认数据版本

CityGML 2.0 和 3.0 的 XML 命名空间、模块结构和部分元素不同。不要在看到 `.gml` 后假定它使用某个版本。

用文本编辑器查看文件开头，重点找：

```xml
xmlns:core="..."
xmlns:bldg="..."
xmlns:gml="http://www.opengis.net/gml"
srsName="..."
```

把真实值记录到 `docs/data_schema.md`：

```text
CityGML 文件名：
CityGML 版本：
core 命名空间：
building 命名空间：
GML 命名空间：
srsName：
坐标单位：
高程说明：
```

## 3. XML 与命名空间

CityGML 常使用带前缀的 XML：

```xml
<bldg:Building gml:id="building_001">
    ...
</bldg:Building>
```

`bldg` 和 `gml` 是命名空间前缀。解析时真正重要的是命名空间 URI，而不是前缀文字。另一个文件可能把 `bldg` 写成其他前缀，但仍指向同一个 URI。

Python 标准库示例：

```python
from pathlib import Path
from xml.etree import ElementTree as ET

path = Path("data/city_model.gml")
root = ET.parse(path).getroot()

print(root.tag)
print(root.attrib)
```

查看元素局部名称：

```python
def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


for element in root.iter():
    if local_name(element.tag) == "Building":
        print(element.tag)
```

这种写法适合前期检查，但正式解析仍应明确记录和使用真实命名空间。

## 4. 建筑 ID

建筑唯一标识可能来自：

- `gml:id`；
- 外部业务字段；
- generic attribute；
- 数据提供方的其他属性。

不要默认所有文件都把最终业务 ID 放在同一个位置。先与老师确认项目要使用哪个字段作为 `building_id`。

读取 `gml:id` 的示例：

```python
GML_ID = "{http://www.opengis.net/gml}id"

for element in root.iter():
    if local_name(element.tag) == "Building":
        building_id = element.attrib.get(GML_ID)
        print(building_id)
```

检查：

- ID 是否缺失；
- ID 是否重复；
- ID 是否稳定；
- ID 是否能与人工标注表对应。

## 5. LoD 是什么

LoD 表示模型采用的空间表达层级。不同版本的具体定义应以数据版本和 OGC 文档为准。

对本项目前期，可以这样理解：

- LoD0：二维或近二维表达；
- LoD1：建筑块体，通常由底面轮廓和高度形成；
- LoD2：通常包含更明确的屋顶和边界表面；
- 更高细节：可能包含更细建筑结构或室内表达。

项目主要使用 LoD1，因为它足以支持：

- 建筑候选范围；
- 建筑距离；
- 建筑大致高度；
- 近似立面；
- 简化视锥和遮挡判断。

LoD1 通常没有可靠纹理、窗户和门，因此不要期待直接与照片做精细特征匹配。

## 6. `srsName` 与坐标

几何元素可能带有：

```xml
srsName="..."
```

它说明坐标参考系。`srsName` 可能出现在上层几何对象而不是每个坐标元素上，解析时需要沿结构检查。

需要确认：

- CRS 标识能否映射到 EPSG；
- 坐标顺序；
- 单位；
- 是否包含 Z；
- 高程与 GPS altitude 是否使用相同基准。

不要在未确认 `srsName` 时对坐标进行平移、缩放或交换轴来“让图看起来对”。

## 7. `gml:pos` 与 `gml:posList`

坐标可能以单点或坐标列表出现：

```xml
<gml:pos>100.0 200.0 10.0</gml:pos>
```

```xml
<gml:posList>
  100.0 200.0 0.0
  120.0 200.0 0.0
  120.0 220.0 0.0
  100.0 220.0 0.0
  100.0 200.0 0.0
</gml:posList>
```

不能只根据数字数量猜二维还是三维，应同时检查 `srsDimension`、CRS 和数据说明。

一个仅用于检查三维 `posList` 的小函数：

```python
def parse_pos_list_3d(text: str) -> list[tuple[float, float, float]]:
    values = [float(value) for value in text.split()]
    if len(values) % 3 != 0:
        raise ValueError("坐标数量不能被 3 整除，无法按三维坐标解析")
    return [
        (values[index], values[index + 1], values[index + 2])
        for index in range(0, len(values), 3)
    ]
```

这不是通用 CityGML 解析器。只有确认当前数据是三维坐标后才能使用。

## 8. 从 LoD1 得到项目需要的表达

对于候选检索，建议将每栋建筑预处理为简化记录：

```text
building_id
footprint
centroid_x
centroid_y
min_z
max_z
height
bounds
source_crs
```

### footprint

footprint 可以来自明确的地面表面，也可以在确认数据结构后从 LoD1 块体的最低水平环提取。不要把任意面的投影都当作 footprint。

### 高度

如果没有可信的属性高度，可以从几何 Z 范围得到近似：

```text
height = max_z - min_z
```

必须在报告中说明这是几何估计，不一定等于建筑学意义上的高度。

### 近似立面

footprint 的相邻顶点形成底边，再从 `min_z` 延伸到 `max_z`，可以构造垂直矩形立面。

对每条 footprint 边：

```text
(x1, y1, min_z)
(x2, y2, min_z)
(x2, y2, max_z)
(x1, y1, max_z)
```

后续可计算立面法向、是否朝向相机，以及投影范围。

## 9. 常见几何问题

### 环没有闭合

首尾坐标可能不同。构造 Polygon 前检查并按数据规范处理。

### 坐标重复

连续重复点可能产生零长度边，影响法向计算。

### 环方向不同

顺时针和逆时针会影响法向方向。必须统一约定并用一个已知建筑检查。

### 多个部件

一栋建筑可能有多个几何部分，不能默认只有一个 Polygon。

### 无效几何

自相交、多边形为空或坐标数量错误时，应记录并拒绝，而不是静默跳过。

### 高程异常

如果高度为负、接近零或远超校园建筑范围，应回查 Z 轴、基准和几何选择。

## 10. 推荐的前期处理流程

```text
检查文件版本与命名空间
→ 确认 srsName、轴顺序和单位
→ 找到建筑元素及业务 ID
→ 提取 LoD1 几何坐标
→ 构造 footprint 和高度
→ 检查几何有效性
→ 导出 GeoJSON/CSV 供 QGIS 核验
→ 人工检查 10 栋建筑
```

先完成 10 栋，再扩大到全部建筑。

## 11. 项目练习

为 10 栋建筑输出：

```text
building_id,source_crs,centroid_x,centroid_y,
min_z,max_z,height,vertex_count,is_valid
```

同时生成 GeoJSON，在 QGIS 中检查：

- 建筑位置是否正确；
- footprint 是否与原模型一致；
- ID 是否正确；
- 建筑高度是否合理；
- 是否存在重复、空或异常建筑。

至少保存一个正常案例和一个异常案例的说明。

## 验收清单

- [ ] 能解释 CityGML 与 OBJ/GLB 的区别；
- [ ] 能确认项目文件的 CityGML 版本；
- [ ] 能找到命名空间和 `srsName`；
- [ ] 能提取建筑唯一 ID；
- [ ] 能解释项目为什么先使用 LoD1；
- [ ] 能从 10 栋建筑提取坐标和高度；
- [ ] 能在 QGIS 中核验 footprint；
- [ ] 能说出当前解析器明确不支持什么。

## 官方资料

- [OGC CityGML 3.0 概念模型标准](https://docs.ogc.org/is/20-010/20-010.html)
- [OGC CityGML 3.0 用户指南](https://docs.ogc.org/guides/20-066.html)
- [OGC CityGML 3.0 GML 编码标准](https://docs.ogc.org/is/21-006r2/21-006r2.html)
- [Python ElementTree](https://docs.python.org/zh-cn/3/library/xml.etree.elementtree.html)

