# tests/test_geom_utils.py
"""M9-2：闭式求交工具的单测（无迭代、无副作用）。"""

from __future__ import annotations

import numpy as np
import pytest

from src.geom_utils import line_circle_intersections, line_line_intersection


def test_secant_two_solutions() -> None:
    """割线：圆 x²+y²=25 与 y=3 交于 (±4, 3)。"""
    points = line_circle_intersections((0.0, 3.0), (1.0, 0.0), (0.0, 0.0), 5.0)
    assert len(points) == 2
    xs = sorted(float(p[0]) for p in points)
    assert xs == pytest.approx([-4.0, 4.0], abs=1e-9)
    assert all(p[1] == pytest.approx(3.0, abs=1e-9) for p in points)


def test_tangent_single_solution() -> None:
    """切线：圆 x²+y²=25 与 y=5 相切于 (0, 5)，只应返回一个点。"""
    points = line_circle_intersections((0.0, 5.0), (1.0, 0.0), (0.0, 0.0), 5.0)
    assert len(points) == 1
    assert float(points[0][0]) == pytest.approx(0.0, abs=1e-6)
    assert float(points[0][1]) == pytest.approx(5.0, abs=1e-6)


def test_no_intersection() -> None:
    """相离：y=6 与半径 5 的圆无交点。"""
    assert line_circle_intersections((0.0, 6.0), (1.0, 0.0), (0.0, 0.0), 5.0) == []


def test_near_tangent_is_stable() -> None:
    """接近相切（偏移 1e-9）时不因浮点噪声产生两个几乎重合的点。"""
    points = line_circle_intersections((0.0, 5.0 - 1e-9), (1.0, 0.0), (0.0, 0.0), 5.0)
    assert 1 <= len(points) <= 2
    if len(points) == 2:
        assert np.linalg.norm(points[0] - points[1]) < 1e-3


def test_line_line_intersection() -> None:
    hit = line_line_intersection((0.0, 0.0), (1.0, 0.0), (2.0, -1.0), (0.0, 1.0))
    assert hit is not None
    assert hit == pytest.approx(np.array([2.0, 0.0]), abs=1e-9)
    assert line_line_intersection((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 0.0)) is None
