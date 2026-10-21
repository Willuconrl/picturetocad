# 坐标系契约（跨模块硬约定）

> 本文件是 **S2 / S3 / S4 / S7 的硬契约**。任何几何模块的改动都必须先读它；
> 违反约定会导致 0.5 px 级别的系统偏移，且**不会**被任何单元测试之外的手段发现。

## 两套坐标，只有一个换算边界

| 名称 | 原点 | 轴向 | 单位 | 谁在用 |
|---|---|---|---|---|
| 图像像素坐标（连续） | 图像左上角 | x 右、y **下** | px | S2 边缘、S3 骨架、S4 拟合输入 |
| CAD 坐标 | 图幅左下角 | x 右、y **上** | mm | S8 写出的 DXF、S5/S6 的图层与文字 |

换算只在 `src/coords.py` 的 `Transform2D` 里做：
`px_img →(×mm_per_px)→ mm_img →(flip Y)→ mm_cad →(配准)→ mm_world`。
**禁止**在其它模块里手写 `h - y` 或 `x * 25.4 / dpi`。

## 连续像素坐标：像素 i 覆盖 [i, i+1)，中心在 i+0.5

这一条是 S2 起就生效的约定（用户决策，M3 验收项 #2）：

- 像素 `(col, row)` 的**左上角**在 `(col, row)`，**中心**在 `(col+0.5, row+0.5)`；
- 图像边界：宽 `W` 像素的图，x 取值范围是 `[0, W]`，**不是** `[0, W-1]`；
- 一条从列边界 100.0 开始的竖线，其左边缘就是 `x = 100.0`；第一个被覆盖的像素
  索引是 100，它的中心是 `x = 100.5`；
- 轮廓点、骨架点、节点坐标**一律**用这套坐标写出，不做 ±0.5 的补偿。

### 为什么必须写死这条

OpenCV 的 `findContours` 返回的是**像素索引**（整数），而 `skimage` 的骨架、
`cv2.minAreaRect`、我们自己的拟合都在连续坐标里工作。两者混用会引入 **0.5 px**
的系统偏移——它比 S2 的亚像素精度指标（0.1 px）还大 5 倍，且在图上完全看不出来。
M3 实现时正是混用导致首版结果整体偏移 0.5 px，因此把它固化成契约。

### 单元测试要求

**所有几何模块的测试必须包含一条 `+0.5` 断言**，例如：

```python
# S2：解析覆盖率渲染的竖笔画，左边缘真值 100.37 → 轮廓点中位数 ≈ 100.37
assert abs(median(x for x in edge_points) - 100.37) <= 0.1

# S3：骨架像素 (59,150) 的中心 → 节点坐标必须是 (59.5, 150.5)
assert node.x == pytest.approx(59.5)
assert node.y == pytest.approx(150.5)
```

缺这条断言视为测试不完整（review 时直接打回）。

## 单位与精度

- 内部几何**一律 float64、毫米**；像素只在 S0–S4 的输入输出边界出现。
- DXF 写出保留 **4 位小数**（0.0001 mm），舍入 ≤0.01 mm，不得成为误差源。
- 误差预算的基准 DPI 是 300（`qc.budget_reference_dpi`）；低 DPI 图的 mm 预算按
  `300 / true_dpi` 放大（见 `src/s0_qc.py` 的 `budget_scale`）。

## Y 轴翻转只允许通过矩阵

```python
from src.coords import Transform2D
t = Transform2D.image_to_cad(dpi=300.0, height_px=625)   # px（连续）→ mm（CAD）
cad_points = t.apply(pixel_points)
pixel_points = t.inverse().apply(cad_points)             # 逆变换必须可用
```

`Transform2D` 的矩阵是 3×3 齐次、float64；缩放/平移/旋转/翻转全部由它承担，
并要求 `inverse()` 可逆（奇异矩阵会抛 `ValueError`）。
