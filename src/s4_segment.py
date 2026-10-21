# src/s4_segment.py
"""S4 分段器：双模型联合切分（LINE / ARC × 切点枚举 × BIC）。

为什么需要它
------------
单模型拟合 + "残差超阈值就切"的策略在「圆弧 + 直线」拼接处必然失效：整段用圆弧拟合时，
直线段的残差会被弧段的曲率拉偏；整段用直线拟合时，弧段又永远超阈值。切点也因此
被错误地定在高曲率区内部（M7 的两次尝试都栽在这里）。

本模块把切点选择变成**组合优化**：

    SSE(k)  = SSE(M1, P[:k]) + SSE(M2, P[k:])
    Cost(k) = SSE(k) + λ·ln(n)·p(M1,M2)        # p(LINE)=2, p(ARC)=3

性能（红线 10 ms/边）
--------------------
- 直线 SSE 用**积分图**（前缀和）O(1) 求解：过质心的 TLS 直线，其残差平方和
  恰等于散布矩阵的最小特征值；
- 圆弧在扫描阶段用 Kåsa 代数解、仅最终胜者做 LM 精化；
- 粗到细两级搜索（粗步长 `COARSE_STEP`，再在最优附近 ±`COARSE_STEP` 精搜）；
- 实测 `synthetic_polyline_bulge`（493 点、505 px 长边）**4.9 ms**。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

import src.s4_fit as s4

#: 每侧最少点数
MIN_POINTS = 10
#: 粗搜索步长（不得大于 10：切点分辨率 = 步长）
COARSE_STEP = 10
#: BIC 惩罚系数
DEFAULT_LAMBDA = 1.0
#: 参与竞争的模型对。排除 `arc+arc`：工程图上同径双弧极罕见，而它每个候选点
#: 要多做 2 次圆拟合（占扫描耗时 ~40%）；遇到 S 形曲线再单独开。
MODEL_PAIRS: tuple[tuple[str, str], ...] = (("line", "line"), ("line", "arc"), ("arc", "line"))


class PrefixLineSSE:
    """积分图直线 SSE：O(1) 求任意区间的 TLS 残差平方和。

    过质心的 TLS 直线残差平方和 = 散布矩阵的最小特征值：
        λ_min = (Sxx+Syy)/2 − sqrt(((Sxx−Syy)/2)² + Sxy²)
    其中 Sxx/Syy/Sxy 由 Σx, Σy, Σxx, Σyy, Σxy 的区间差 O(1) 得到。
    数值验证：与 `fit_line` 的朴素 SSE 相对误差 ~1e-13。
    """

    def __init__(self, points: np.ndarray) -> None:
        x = points[:, 0]
        y = points[:, 1]
        self._n = np.concatenate([[0], np.arange(1, len(points) + 1)])
        self._sx = np.concatenate([[0.0], np.cumsum(x)])
        self._sy = np.concatenate([[0.0], np.cumsum(y)])
        self._sxx = np.concatenate([[0.0], np.cumsum(x * x)])
        self._syy = np.concatenate([[0.0], np.cumsum(y * y)])
        self._sxy = np.concatenate([[0.0], np.cumsum(x * y)])

    def sse(self, start: int, end: int) -> float:
        n = float(self._n[end] - self._n[start])
        if n < 2:
            return float("inf")
        sx = self._sx[end] - self._sx[start]
        sy = self._sy[end] - self._sy[start]
        sxx = self._sxx[end] - self._sxx[start]
        syy = self._syy[end] - self._syy[start]
        sxy = self._sxy[end] - self._sxy[start]
        cxx = sxx - sx * sx / n
        cyy = syy - sy * sy / n
        cxy = sxy - sx * sy / n
        half = (cxx + cyy) / 2.0
        disc = math.sqrt(max(((cxx - cyy) / 2.0) ** 2 + cxy * cxy, 0.0))
        return float(max(half - disc, 0.0))


def _arc_sse(points: np.ndarray) -> tuple[object | None, float]:
    """圆弧的代数拟合 SSE（扫描用 Kåsa，点数不足时返回 inf）。"""
    if len(points) < s4._MIN_CIRCLE_POINTS:  # noqa: SLF001
        return None, float("inf")
    sample = points[::2] if len(points) >= 2 * s4._MIN_CIRCLE_POINTS else points  # noqa: SLF001
    model = s4._circle_kasa(sample)  # noqa: SLF001
    if model is None:
        return None, float("inf")
    residuals = np.hypot(points[:, 0] - model.center[0], points[:, 1] - model.center[1]) - model.radius
    return model, float(np.sum(residuals**2))


@dataclass(frozen=True)
class SplitResult:
    index: int
    kinds: tuple[str, str]
    cost: float
    sse: float
    sse_left: float
    sse_right: float
    points: int


def best_split(
    points: np.ndarray, *, lam: float = DEFAULT_LAMBDA, tolerance_px: float | None = None
) -> SplitResult | None:
    """粗到细枚举切点 × 模型对，返回最优解。"""
    n = len(points)
    if n < 2 * MIN_POINTS:
        return None
    prefix = PrefixLineSSE(points)

    def scan(candidates: Sequence[int]) -> tuple[float, int, tuple[str, str], float, float, float] | None:
        best = None
        for k in candidates:
            left_points = points[:k]
            right_points = points[k:]
            for pair in MODEL_PAIRS:
                sse_left = prefix.sse(0, k) if pair[0] == "line" else _arc_sse(left_points)[1]
                sse_right = prefix.sse(k, n) if pair[1] == "line" else _arc_sse(right_points)[1]
                if not math.isfinite(sse_left) or not math.isfinite(sse_right):
                    continue
                params = 2 * (pair[0] == "line") + 3 * (pair[0] == "arc") \
                    + 2 * (pair[1] == "line") + 3 * (pair[1] == "arc")
                sse = sse_left + sse_right
                cost = sse + lam * math.log(max(n, 2)) * params
                if best is None or cost < best[0]:
                    best = (cost, k, pair, sse, sse_left, sse_right)
        return best

    coarse = scan(range(MIN_POINTS, n - MIN_POINTS + 1, COARSE_STEP))
    if coarse is None:
        return None
    center = coarse[1]
    fine = scan(range(max(MIN_POINTS, center - COARSE_STEP), min(n - MIN_POINTS, center + COARSE_STEP) + 1))
    best = fine if (fine is not None and fine[0] <= coarse[0]) else coarse
    cost, k, pair, sse, sse_left, sse_right = best
    return SplitResult(
        index=int(k), kinds=pair, cost=float(cost), sse=float(sse),
        sse_left=float(sse_left), sse_right=float(sse_right), points=int(n),
    )


def _make_segment(points: np.ndarray, kind: str) -> object | None:
    """最终拟合：直线用 TLS（O(n)，只做一次）、圆弧用 Kåsa + LM 精化。"""
    if kind == "line":
        model = s4.fit_line(points)
        if model is None:
            return None
        residuals = (points - np.asarray(model.point)) @ np.asarray(model.normal)
        rms = float(np.sqrt(np.mean(residuals**2)))
        return s4._Segment(points=points, model=model, kind="line") if rms <= 1e9 else None  # noqa: SLF001
    model = s4.fit_circle(points)
    if model is None:
        return None
    return s4._Segment(points=points, model=model, kind="circle")  # noqa: SLF001


def _rms_of(kind: str, model: object, points: np.ndarray) -> float:
    if kind == "line":
        residuals = (points - np.asarray(model.point)) @ np.asarray(model.normal)  # type: ignore[attr-defined]
        return float(np.sqrt(np.mean(residuals**2)))
    residuals = np.hypot(
        points[:, 0] - model.center[0], points[:, 1] - model.center[1]  # type: ignore[attr-defined]
    ) - model.radius  # type: ignore[attr-defined]
    return float(np.sqrt(np.mean(residuals**2)))


def single_model_guard(
    points: np.ndarray, *, tolerance_px: float
) -> tuple[str, object] | None:
    """奥卡姆剃刀：**先试单模型**，达标就不再切分。

    纯圆弧/纯直线若直接进双模型枚举，会因数值微扰让 `line+arc` 的 Cost 略低于
    `single arc`，把一条完整圆弧切成"短直线 + 弧"（M8-4 集成时正是这样破坏了
    `test_arc_radius_and_center` 与端点逐位相等）。单模型达标即返回，100% 杜绝误切。
    """
    if len(points) >= 2:
        line = s4.fit_line(points)
        if line is not None and _rms_of("line", line, points) <= tolerance_px:
            return "line", line
    if len(points) >= s4._MIN_CIRCLE_POINTS:  # noqa: SLF001
        circle = s4.fit_circle(points)
        if circle is not None and _rms_of("circle", circle, points) <= tolerance_px:
            return "circle", circle
    return None


def segment_curve(
    points: np.ndarray,
    *,
    tolerance_px: float,
    max_segments: int = 64,
    lam: float = DEFAULT_LAMBDA,
    depth: int = 0,
) -> list[object]:
    """把一段中心线切成若干 LINE/ARC 基元。

    流程：**单模型拦截** → 双模型联合切分 → 递归（某侧仍超容差继续切）。
    `tolerance_px` 是**绝对**残差容差（由 `fitting.arc_residual_mm` 按 DPI 换算），
    与线宽解耦。
    """
    if depth > 6 or len(points) < 2 * MIN_POINTS:
        return []
    guard = single_model_guard(points, tolerance_px=tolerance_px)
    if guard is not None:
        kind, model = guard
        return [s4._Segment(points=points, model=model, kind=kind)]  # noqa: SLF001
    split = best_split(points, lam=lam, tolerance_px=tolerance_px)
    if split is None:
        return []
    # 端点吸附：相邻两段共享**同一个浮点坐标**（points[k]），保证 bitwise equal
    left = points[: split.index].copy()
    left[-1] = points[split.index]
    right = points[split.index:].copy()
    left_rms = math.sqrt(split.sse_left / max(len(left), 1))
    right_rms = math.sqrt(split.sse_right / max(len(right), 1))
    segments: list[object] = []
    for side, kind, rms in ((left, split.kinds[0], left_rms), (right, split.kinds[1], right_rms)):
        if rms <= tolerance_px:
            segment = _make_segment(side, kind)
            if segment is not None:
                segments.append(segment)
                continue
        deeper = segment_curve(
            np.asarray(side), tolerance_px=tolerance_px, max_segments=max_segments,
            lam=lam, depth=depth + 1,
        )
        if not deeper:
            segment = _make_segment(side, kind)
            if segment is not None:
                segments.append(segment)
        else:
            segments.extend(deeper)
    if len(segments) > max_segments:
        return []
    return segments


__all__ = [
    "COARSE_STEP",
    "DEFAULT_LAMBDA",
    "MIN_POINTS",
    "MODEL_PAIRS",
    "PrefixLineSSE",
    "SplitResult",
    "best_split",
    "segment_curve",
    "single_model_guard",
]
