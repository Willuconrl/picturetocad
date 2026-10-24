# tests/test_arc_snap.py
"""M9-3：ARC 端点吸附的**几何契约**（5 个用例必须全绿，改动 ARC 端点逻辑前必过）。"""

from __future__ import annotations

import math

import pytest

from src.arc_snap import image_angle_deg, snap_arc_endpoint, sweep_between

CENTER = (0.0, 0.0)
RADIUS = 10.0


def _point(angle_deg: float) -> tuple[float, float]:
    radians = math.radians(angle_deg)
    return (RADIUS * math.cos(radians), RADIUS * math.sin(radians))


def test_snap_preserves_ccw_sweep_sign() -> None:
    """正向（图像角度递增）弧：吸附起点后 sweep 仍为正。"""
    result = snap_arc_endpoint(
        center=CENTER,
        new_endpoint=_point(3.0),
        start_deg=0.0,
        end_deg=90.0,
        sweep_deg=90.0,
        snap_start=True,
    )
    assert result is not None
    assert result["sweep_deg"] > 0
    assert result["start_angle_deg"] == pytest.approx(3.0, abs=1e-3)
    assert result["sweep_deg"] == pytest.approx(87.0, abs=1e-3)


def test_snap_preserves_cw_sweep_sign() -> None:
    """反向（图像角度递减）弧：吸附终点后 sweep 仍为负。"""
    result = snap_arc_endpoint(
        center=CENTER,
        new_endpoint=_point(3.0),
        start_deg=90.0,
        end_deg=0.0,
        sweep_deg=-90.0,
        snap_start=False,
    )
    assert result is not None
    assert result["sweep_deg"] < 0
    assert result["end_angle_deg"] == pytest.approx(3.0, abs=1e-3)
    assert result["sweep_deg"] == pytest.approx(-87.0, abs=1e-3)


def test_snap_rejects_large_drift() -> None:
    """把端点挪到会翻转覆盖范围的位置：|Δsweep| > 5° → 拒绝（返回 None）。"""
    result = snap_arc_endpoint(
        center=CENTER,
        new_endpoint=_point(180.0),
        start_deg=0.0,
        end_deg=90.0,
        sweep_deg=90.0,
        snap_start=True,
    )
    assert result is None


def test_snap_near_tangent_stable() -> None:
    """端点只挪 0.05 px 级：扫角变化必须 < 0.1°（数值稳定）。"""
    tiny = _point(0.0)
    moved = (tiny[0] + 0.05, tiny[1])
    result = snap_arc_endpoint(
        center=CENTER,
        new_endpoint=moved,
        start_deg=0.0,
        end_deg=90.0,
        sweep_deg=90.0,
        snap_start=True,
    )
    assert result is not None
    assert abs(result["sweep_deg"] - 90.0) < 0.1


def test_snap_across_zero_angle() -> None:
    """跨 0°/360° 边界：符号保持、大小正确（不能变成 350°+）。"""
    result = snap_arc_endpoint(
        center=CENTER,
        new_endpoint=_point(359.0),
        start_deg=0.0,
        end_deg=3.0,
        sweep_deg=3.0,
        snap_start=True,
    )
    assert result is not None
    assert result["start_angle_deg"] == pytest.approx(359.0, abs=1e-3)
    assert result["sweep_deg"] == pytest.approx(4.0, abs=1e-3)
    assert result["sweep_deg"] > 0

    backward = snap_arc_endpoint(
        center=CENTER,
        new_endpoint=_point(357.5),
        start_deg=0.0,
        end_deg=357.0,
        sweep_deg=-3.0,
        snap_start=False,
    )
    assert backward is not None
    assert backward["sweep_deg"] < 0


def test_sweep_and_angle_helpers() -> None:
    assert image_angle_deg(CENTER, (1.0, 0.0)) == pytest.approx(0.0)
    assert image_angle_deg(CENTER, (0.0, -1.0)) == pytest.approx(270.0)
    assert sweep_between(350.0, 10.0, forward=True) == pytest.approx(20.0)
    assert sweep_between(10.0, 350.0, forward=False) == pytest.approx(-20.0)
