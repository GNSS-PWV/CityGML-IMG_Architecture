# CityGML 与 LoD1 建筑模型

## 学习目标

完成本教程后，你应该能够：

- 解释 CityGML 与普通三维模型的区别；
- 了解常用的 CityGML 查看、分析和编辑工具；
- 使用三维查看器和 GIS 软件检查 CityGML；
- 理解建筑对象、语义、几何和 LoD；
- 检查 CityGML 文件版本与 `srsName`；
- 理解 XML 命名空间和 `gml:id`；
- 从项目数据中提取建筑 ID 和坐标；
- 将 LoD1 建筑简化为 footprint、高度和近似立面；
- 进行简单的属性编辑，并理解属性与几何的区别；
- 将 CityGML 转换为适合 GIS 核验的中间数据；
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

可以把普通三维模型和 CityGML 粗略理解为：

```text
普通三维模型

Building
└── Mesh
    ├── Vertex
    ├── Face
    └── Texture
```

而 CityGML 更接近：

```text
CityModel
└── Building
    ├── gml:id
    ├── 属性
    ├── LoD
    └── Geometry
        ├── GroundSurface
        ├── WallSurface
        └── RoofSurface
```

因此，CityGML 不应简单理解成“另一种三维模型格式”，而应理解为：

> **带有城市语义、空间关系和多细节层级信息的三维 GIS 数据模型。**

## 2. 先确认数据版本

CityGML 2.0 和 3.0 的 XML 命名空间、模块结构和部分元素不同。不要在看到 `.gml` 后假定它使用某个版本。

用文本编辑器查看文件开头，重点找：

```xml
xmlns:core="..."
xmlns:bldg="..."
xmlns:gml="http://www.opengis.net/gml"
srsName="..."
```

也可以重点搜索：

```text
citygml
core
bldg
version
schemaLocation
srsName
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

注意：

- `.gml` 文件扩展名本身不能证明它一定是 CityGML；
- 即使都是 `.gml`，不同数据集也可能使用不同版本和不同模块；
- 不要因为某段代码适用于 CityGML 2.0，就假设它同样适用于 CityGML 3.0。

## 3. CityGML 的查看与基本操作

在编写 Python 解析程序之前，建议先使用可视化工具直接观察数据。这样可以先建立“文件中的 XML 对象”和“实际三维建筑”之间的对应关系。

### 3.1 常用工具

对于入门阶段，可以按照下面的层次使用工具：

| 工具 | 主要用途 | 推荐阶段 |
|---|---|---|
| FZK Viewer | 三维查看、建筑对象检查 | 入门 |
| QGIS | 二维 GIS、属性、CRS、空间核验 | 入门/分析 |
| Python | 批量读取和处理 | 开发 |
| 3DCityDB | 大规模 CityGML 数据管理、查询、导入导出 | 进阶 |

其中：

**FZK Viewer**

适合第一次打开 CityGML，重点观察：

- 建筑物是否正常显示；
- 建筑高度是否合理；
- 建筑轮廓是否正确；
- 是否能选择单栋建筑；
- 是否能够看到对象属性；
- 数据是否存在明显的错位、翻转或尺度异常。

**QGIS**

适合进行 GIS 层面的检查，例如：

- 建筑物位置是否正确；
- footprint 是否正确；
- CRS 是否正确；
- 属性是否能够正常读取；
- 是否能够与其他 GIS 数据叠加。

**Python**

适合进行批量处理，例如：

- 提取所有建筑；
- 提取建筑 ID；
- 提取坐标；
- 计算高度；
- 检查几何；
- 导出 GeoJSON 或 CSV。

**3DCityDB**

适用于数据量较大或需要数据库管理的情况，可以将 CityGML 导入数据库后进行查询、管理和导出。

### 3.2 第一次查看时应该关注什么

第一次打开一个 CityGML 文件时，可以按照以下顺序观察：

```text
建筑在哪里？
    ↓
能否选择单栋建筑？
    ↓
建筑 ID 是什么？
    ↓
建筑采用什么 LoD？
    ↓
建筑大约多高？
    ↓
有没有明显的屋顶、墙面、地面结构？
    ↓
属性信息在哪里？
    ↓
这些信息在 XML 中对应什么元素？
```

三维查看器解决的是：

> **“这个数据实际长什么样？”**

而 XML 和 Python 解决的是：

> **“这些信息在文件中是怎样存储和组织的？”**

二者应该结合起来使用。

### 3.3 CityGML 与查看器中的对象对应关系

可以建立下面的对应关系：

| 三维/GIS 中观察到的内容 | CityGML 中常见的对应元素 |
|---|---|
| 一栋建筑 | `bldg:Building` |
| 建筑唯一 ID | `gml:id` |
| 建筑高度属性 | `bldg:measuredHeight` |
| 墙面 | `bldg:WallSurface` |
| 屋顶 | `bldg:RoofSurface` |
| 地面 | `bldg:GroundSurface` |
| 建筑几何 | GML geometry |
| 坐标参考系 | `srsName` |

不同 CityGML 版本和数据集的实际结构可能不同，因此上表用于建立概念对应关系，实际解析时仍应以文件结构为准。

### 3.4 QGIS 中的作用

QGIS 不需要承担完整的 CityGML 三维编辑任务，更适合作为**空间核验工具**。

本项目中，可以将提取后的建筑 footprint 导出为 GeoJSON，然后在 QGIS 中检查：

```text
CityGML
   ↓
提取 Building
   ↓
提取 footprint
   ↓
GeoJSON
   ↓
QGIS
```

重点检查：

- 建筑位置是否正确；
- footprint 是否与原始模型一致；
- 坐标方向是否正确；
- 是否出现整体偏移；
- 是否存在异常建筑。

如果坐标看起来明显错误，不要马上交换 X/Y 或修改坐标。应回到 `srsName`、坐标顺序和数据说明进行检查。

## 4. XML 与命名空间

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

需要特别理解：

```text
<bldg:Building>
```

中的 `bldg` 只是前缀。

真正定义它身份的是类似：

```text
http://www.opengis.net/citygml/building/...
```

这样的命名空间 URI。

## 5. 建筑 ID

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

需要区分：

```text
gml:id
```

和：

```text
项目中的 building_id
```

它们有可能相同，也有可能不同。

因此，在正式建立数据处理流程时，应明确记录二者之间的对应关系。

## 6. LoD 是什么

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

一个很重要的理解是：

> LoD 不是“模型精度”的简单等级，而是城市对象空间表达内容和复杂程度的层级。

## 7. `srsName` 与坐标

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

对于本项目，至少需要明确：

```text
CRS：
EPSG：
X 含义：
Y 含义：
Z 含义：
水平单位：
垂直单位：
高程基准：
```

如果发现建筑整体偏移、旋转方向异常或者坐标数量级明显不合理，应优先检查 CRS 和坐标解释，而不是直接修改原始数据。

## 8. `gml:pos` 与 `gml:posList`

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

还需要注意，实际文件可能使用：

- `gml:pos`；
- `gml:posList`；
- `gml:coordinates`；
- `gml:LinearRing`；
- `gml:Polygon`；
- `gml:MultiSurface`；
- `gml:Solid`；

等不同结构，因此不能假设所有建筑都可以通过一次搜索 `posList` 得到完整几何。

## 9. 从 LoD1 得到项目需要的表达

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

## 10. 常见几何问题

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

### 属性与几何不一致

例如：

```text
measuredHeight = 30 m
```

但实际几何最高点只有：

```text
Z = 20 m
```

这种情况下不能简单认为 30 m 就是真实建筑高度，应检查属性定义和几何数据。

## 11. CityGML 的简单编辑与转换

CityGML 可以编辑，但需要区分**属性编辑**和**几何编辑**。

### 11.1 简单属性编辑

例如：

```xml
<bldg:Building gml:id="building_001">
    <bldg:measuredHeight>20.5</bldg:measuredHeight>
</bldg:Building>
```

将：

```text
20.5
```

修改为：

```text
25.0
```

修改的是建筑的**属性值**。

但需要注意：

> 修改 `measuredHeight` 不一定会修改建筑的实际三维几何。

如果几何最高点仍然只有 20.5 m，那么查看器显示的几何模型可能仍然只有原来的高度。

### 11.2 几何编辑

如果修改的是：

```xml
<gml:posList>
    ...
</gml:posList>
```

实际上是在修改几何坐标。

例如原本：

```text
Z = 0
Z = 20
```

修改为：

```text
Z = 0
Z = 30
```

才意味着几何本身发生了高度变化。

因此需要区分：

```text
属性
measuredHeight
       ↓
“这个建筑被描述为多高”

几何
坐标 Z
       ↓
“这个建筑实际被建模到了哪里”
```

对于数据质量检查，两者的一致性也值得关注。

### 11.3 不建议直接修改大型 CityGML

少量属性可以用 VS Code、Notepad++ 等文本编辑器进行学习性修改。

但是对于大型数据集，不建议直接手工修改几何。

原因包括：

- XML 层级复杂；
- 命名空间容易写错；
- 几何可能由多个对象组成；
- 对象之间可能存在引用关系；
- 修改坐标可能破坏几何结构；
- 属性与几何可能失去一致性。

大规模修改应优先使用：

- Python；
- 数据库工具；
- 专门的 CityGML 软件。

### 11.4 CityGML → GeoJSON

本项目中，更推荐把 GeoJSON 作为**分析和核验用的中间格式**，而不是把它看成 CityGML 的替代品。

典型流程：

```text
CityGML
   ↓
提取 Building
   ↓
提取 footprint
   ↓
提取 height
   ↓
GeoJSON
   ↓
QGIS
```

例如最终可以得到：

```text
building_id
footprint
height
centroid
```

然后在 QGIS 中检查建筑位置和轮廓。

这样做的意义是：

- CityGML 保留原始三维语义；
- GeoJSON 方便二维 GIS 检查；
- Python 负责两者之间的数据处理。

### 11.5 CityGML 与 3DCityDB

当数据从几十栋、几百栋扩大到整个城区时，可以进一步考虑 3DCityDB。

基本流程可以理解为：

```text
CityGML
   ↓
3DCityDB
   ↓
数据库查询 / 管理 / 分析
   ↓
CityGML / 其他输出
```

3DCityDB 更适合：

- 大规模城市模型；
- 建筑对象查询；
- 数据库化管理；
- 多次查询和批处理；
- CityGML 导入导出。

需要注意，不同 CityGML 版本之间进行转换时不一定能够做到完全无损，因此进行格式转换后应重新检查版本、模块、属性和几何。

## 12. 推荐的前期处理流程

```text
检查文件版本与命名空间
→ 确认 srsName、轴顺序和单位
→ 用三维查看器检查数据是否正常
→ 在 QGIS 中核验空间位置
→ 找到建筑元素及业务 ID
→ 提取 LoD1 几何坐标
→ 构造 footprint 和高度
→ 检查几何有效性
→ 导出 GeoJSON/CSV 供 QGIS 核验
→ 人工检查 10 栋建筑
```

先完成 10 栋，再扩大到全部建筑。

在遇到异常建筑时，建议保留原始数据和异常记录，而不是直接删除：

```text
building_id
error_type
description
possible_cause
action
```

例如：

```text
building_015
error_type = invalid_geometry
description = polygon self-intersection
possible_cause = duplicate vertex
action = manual inspection
```

这样方便后续调试和统计。

## 13. 项目练习

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
- [ ] 知道至少一种三维 CityGML 查看工具；
- [ ] 能使用查看器观察单栋建筑；
- [ ] 能在 QGIS 中核验建筑位置和 footprint；
- [ ] 能确认项目文件的 CityGML 版本；
- [ ] 能找到命名空间和 `srsName`；
- [ ] 能提取建筑唯一 ID；
- [ ] 能解释项目为什么先使用 LoD1；
- [ ] 能区分属性高度和几何高度；
- [ ] 能说明简单修改属性与修改几何的区别；
- [ ] 能从 10 栋建筑提取坐标和高度；
- [ ] 能说出当前解析器明确不支持什么。

## 官方资料

- [OGC CityGML 3.0 概念模型标准](https://docs.ogc.org/is/20-010/20-010.html)
- [OGC CityGML 3.0 用户指南](https://docs.ogc.org/guides/20-066.html)
- [OGC CityGML 3.0 GML 编码标准](https://docs.ogc.org/is/21-006r2/21-006r2.html)
- [Python ElementTree](https://docs.python.org/zh-cn/3/library/xml.etree.elementtree.html)
- [3DCityDB Documentation](https://docs.3dcitydb.org/)
