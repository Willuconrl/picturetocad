# src/geom_utils.py
"""几何工具：解析求交（闭式解，无迭代）。

M9-2：S4 分段后，相邻 LINE / ARC 段的端点是各自独立拟合出来的，在相切过渡处会留下
~0.1 mm 的接缝间隙。用"直线方程 ∩ 圆方程"的闭式交点把两侧端点吸到同一点。

全部函数是纯函数、无副作用；退化情况返回空列表而不是抛异常。
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def line_circle_intersections(
    point: Sequence[float],
    direction: Sequence[float],
    center: Sequence[float],
    radius: float,
) -> list[np.ndarray]:
    """直线 `p = point + t·direction` 与圆 `|p−center| = radius` 的交点（闭式）。

    返回 0/1/2 个交点（相切时 1 个）。方向向量不必归一化；接近相切时用判别式
    的相对阈值判定为相切，避免浮点噪声造出两个几乎重合的点。
    """
    p = np.asarray(point, dtype=np.float64)
    d = np.asarray(direction, dtype=np.float64)
    norm = float(np.linalg.norm(d))
    if norm < 1e-12 or radius <= 0.0:
        return []
    d = d / norm
    c = np.asarray(center, dtype=np.float64)
    offset = p - c
    b = 2.0 * float(offset @ d)
    cc = float(offset @ offset) - radius * radius
    disc = b * b - 4.0 * cc
    scale = max(b * b, abs(4.0 * cc), 1e-12)
    if disc < -1e-12 * scale:
        return []
    if disc <= 1e-12 * scale:  # 相切
        return [p + (-b / 2.0) * d]
    root = math.sqrt(disc)
    return [p + ((-b - root) / 2.0) * d, p + ((-b + root) / 2.0) * d]


def line_line_intersection(
    point_a: Sequence[float],
    direction_a: Sequence[float],
    point_b: Sequence[float],
    direction_b: Sequence[float],
) -> np.ndarray | None:
    """两条直线的闭式交点；平行时返回 None。"""
    p1 = np.asarray(point_a, dtype=np.float64)
    d1 = np.asarray(direction_a, dtype=np.float64)
    p2 = np.asarray(point_b, dtype=np.float64)
    d2 = np.asarray(direction_b, dtype=np.float64)
    matrix = np.array([[d1[0], -d2[0]], [d1[1], -d2[1]]])
    if abs(float(np.linalg.det(matrix))) < 1e-12:
        return None
    t = np.linalg.solve(matrix, p2 - p1)
    return p1 + t[0] * d1


__all__ = ["line_circle_intersections", "line_line_intersection"]
