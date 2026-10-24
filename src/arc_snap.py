# src/arc_snap.py
"""ARC 端点吸附的角度/扫角换算（M9-3）。

S4 内部的角度约定（必须显式钉死，别再靠"猜"）：

- `start_angle_deg` / `end_angle_deg` 是**图像坐标系**（y 向下）下的 `atan2(dy, dx)`，范围 [0, 360)；
  与 S8 的 CAD 系（y 向上）之间的翻转由 S8 统一处理，本模块不跨越那条边界。
- `sweep_deg` 的**符号**表示沿圆周的前进方向：`+` = 图像角度递增（对应 CAD 的顺时针），
  `-` = 图像角度递减。S8/DXF 只认 CCW，方向翻转同样在 S8 处理。
- 因此吸附端点时只需保证：**保持 `sweep_deg` 的符号不变，只重算大小**。

任何人改动 ARC 端点逻辑，必须先过 `tests/test_arc_snap.py` 的 5 个契约用例。
"""

from __future__ import annotations

import math
import sys
from typing import Sequence

# 扫角漂移门限（度）：超过即认为吸附会翻转圆弧的覆盖范围，拒绝
MAX_SWEEP_DRIFT_DEG = 5.0


def image_angle_deg(center: Sequence[float], point: Sequence[float]) -> float:
    """图像坐标系下点相对圆心的角度（[0, 360)）。"""
    return math.degrees(math.atan2(point[1] - center[1], point[0] - center[0])) % 360.0


def sweep_between(start_deg: float, end_deg: float, *, forward: bool) -> float:
    """按给定方向计算带符号扫角（与 S4 现有的取模公式一致）。"""
    if forward:
        return (end_deg - start_deg) % 360.0
    return -((start_deg - end_deg) % 360.0)


def snap_arc_endpoint(
    *,
    center: Sequence[float],
    new_endpoint: Sequence[float],
    start_deg: float,
    end_deg: float,
    sweep_deg: float,
    snap_start: bool,
) -> dict[str, float] | None:
    """把弧的一端吸附到 `new_endpoint`，返回新的三个角度值；漂移过大时返回 None。

    `snap_start=True` 表示改弧的起点角度，否则改终点角度。圆心与半径不变
    （调用方保证 `new_endpoint` 在圆上或足够接近）。
    """
    forward = sweep_deg >= 0.0
    new_angle = image_angle_deg(center, new_endpoint)
    if snap_start:
        start_angle, end_angle = new_angle, float(end_deg)
    else:
        start_angle, end_angle = float(start_deg), new_angle
    new_sweep = sweep_between(start_angle, end_angle, forward=forward)
    if abs(new_sweep) > 360.0 or abs(new_sweep - sweep_deg) > MAX_SWEEP_DRIFT_DEG:
        return None
    return {
        "start_angle_deg": round(start_angle, 4),
        "end_angle_deg": round(end_angle, 4),
        "sweep_deg": round(new_sweep, 4),
    }


__all__ = ["MAX_SWEEP_DRIFT_DEG", "image_angle_deg", "snap_arc_endpoint", "sweep_between"]
