# src/s4_segment.py
"""S4 切分预言机（Split-Only Oracle）：**只决定在哪里切、每段是什么模型**。

设计变化（M8-4 Phase 4）
------------------------
Phase 3 的 `segment_curve` 直接返回拟合好的 `_Segment`，绕过了 `run_s4_fit` 里
一整套后处理（`select_model` 竞争、`_refit_circle`、`_line_extent`、节点吸附、
`_enforce_g1`），导致 8 个既有用例失败。本模块因此**降级为只输出切点与模型类型**：

    plan_segmentation(points, tolerance_px) -> Plan(split_indices=[220], model_types=["LINE", "ARC"])

拟合、端点延伸、节点吸附、G1 复检全部仍由 `run_s4_fit` 原有代码完成。

决策逻辑
--------
1. **单模型拦截（奥卡姆剃刀）**：整段能用一个 LINE（或一个 ARC）拟合到容差内，
   就直接返回空切点列表——避免把完美圆弧切成"短直线 + 弧"。
2. 否则用**双模型联合 SSE + BIC** 枚举切点，必要时递归细分。

性能（红线 10 ms/边）
---------------------
- 直线 SSE 走**积分图**（前缀和）O(1)：过质心的 TLS 残差平方和 = 散布矩阵最小特征值；
- 圆弧扫描用 Kåsa 代数解、隔点采样；
- 粗到细两级搜索（粗步长 `COARSE_STEP=10`）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

import src.s4_fit as s4

#: 每侧最少点数
MIN_POINTS = 10
#: 粗搜索步长（用户约束：不得大于 10）
COARSE_STEP = 10
#: BIC 惩罚系数
DEFAULT_LAMBDA = 1.0
#: 参与竞争的模型对。排除 `arc+arc`：工程图上同径双弧极罕见，而它每个候选点
#: 要多做 2 次圆拟合（占扫描 ~40%）；遇到 S 形曲线再单独开。
MODEL_PAIRS: tuple[tuple[str, str], ...] = (("line", "line"), ("line", "arc"), ("arc", "line"))
#: 递归细分上限
MAX_DEPTH = 6


@dataclass(frozen=True)
class Plan:
    """切分计划：`split_indices[i]` 是第 i+1 段的起点（相对 points 的下标）。"""

    split_indices: tuple[int, ...]
    model_types: tuple[str, ...]  # "LINE" / "ARC"，长度 = len(split_indices) + 1

    @property
    def segment_count(self) -> int:
        return len(self.model_types)


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


def _arc_sse(points: np.ndarray) -> float:
    """圆弧的代数拟合 SSE（扫描用 Kåsa，点数不足时返回 inf）。"""
    if len(points) < s4._MIN_CIRCLE_POINTS:  # noqa: SLF001
        return float("inf")
    sample = points[::2] if len(points) >= 2 * s4._MIN_CIRCLE_POINTS else points  # noqa: SLF001
    model = s4._circle_kasa(sample)  # noqa: SLF001
    if model is None:
        return float("inf")
    residuals = np.hypot(points[:, 0] - model.center[0], points[:, 1] - model.center[1]) - model.radius
    return float(np.sum(residuals**2))


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
    """粗到细枚举切点 × 模型对，返回最优解（只用于决策，不产出拟合对象）。"""
    n = len(points)
    if n < 2 * MIN_POINTS:
        return None
    prefix = PrefixLineSSE(points)

    def scan(candidates: Sequence[int]) -> tuple[float, int, tuple[str, str], float, float, float] | None:
        best = None
        for k in candidates:
            for pair in MODEL_PAIRS:
                sse_left = prefix.sse(0, k) if pair[0] == "line" else _arc_sse(points[:k])
                sse_right = prefix.sse(k, n) if pair[1] == "line" else _arc_sse(points[k:])
                if not math.isfinite(sse_left) or not math.isfinite(sse_right):
                    continue
                params = 6 - len(pair)  # 3 参数 + 3 参数；line 记 2
                params = sum(3 if kind == "arc" else 2 for kind in pair)
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


def _single_model_kind(points: np.ndarray, *, tolerance_px: float) -> str | None:
    """整段能否单模型拟合到容差内？返回 "LINE"/"ARC"/None。"""
    if len(points) < 2:
        return None
    line = s4.fit_line(points)
    if line is not None and line.rms <= tolerance_px:
        return "LINE"
    if len(points) >= s4._MIN_CIRCLE_POINTS:  # noqa: SLF001
        circle = s4.fit_circle(points)
        if circle is not None and float(circle.rms) <= tolerance_px:
            return "ARC"
    return None


def _split_recursive(
    points: np.ndarray, *, tolerance_px: float, lam: float, depth: int
) -> Plan | None:
    guard = _single_model_kind(points, tolerance_px=tolerance_px)
    if guard is not None:
        return Plan(split_indices=(), model_types=(guard,))
    if depth >= MAX_DEPTH:
        return None
    split = best_split(points, lam=lam, tolerance_px=tolerance_px)
    if split is None:
        return None
    left_plan = _split_recursive(
        points[: split.index], tolerance_px=tolerance_px, lam=lam, depth=depth + 1
    )
    right_plan = _split_recursive(
        points[split.index:], tolerance_px=tolerance_px, lam=lam, depth=depth + 1
    )
    if left_plan is None or right_plan is None:
        return None
    # 左段的切点原样保留；右段的切点整体平移 split.index
    indices = tuple(left_plan.split_indices) + (split.index,) + tuple(
        split.index + index for index in right_plan.split_indices
    )
    types = tuple(left_plan.model_types) + tuple(right_plan.model_types)
    return Plan(split_indices=indices, model_types=types)


def plan_segmentation(
    points: np.ndarray, *, tolerance_px: float, lam: float = DEFAULT_LAMBDA
) -> Plan:
    """返回切分计划。退化情况（点数不足/求解失败）返回"整段单模型"的兜底计划。"""
    if len(points) < 2 * MIN_POINTS:
        guard = _single_model_kind(points, tolerance_px=tolerance_px)
        return Plan(split_indices=(), model_types=(guard or "LINE",))
    plan = _split_recursive(np.asarray(points), tolerance_px=tolerance_px, lam=lam, depth=0)
    if plan is None:
        return Plan(split_indices=(), model_types=("LINE",))
    # 去掉过短段（< MIN_POINTS）造成的合并：确保相邻切点间距足够
    cleaned: list[int] = []
    for index in plan.split_indices:
        if MIN_POINTS <= index <= len(points) - MIN_POINTS and (
            not cleaned or index - cleaned[-1] >= MIN_POINTS
        ):
            cleaned.append(int(index))
    if len(cleaned) + 1 != len(plan.model_types):
        types = list(plan.model_types[: len(cleaned) + 1])
        while len(types) < len(cleaned) + 1:
            types.append(types[-1])
        return Plan(split_indices=tuple(cleaned), model_types=tuple(types))
    return Plan(split_indices=tuple(cleaned), model_types=tuple(plan.model_types))


__all__ = [
    "COARSE_STEP",
    "DEFAULT_LAMBDA",
    "MIN_POINTS",
    "MODEL_PAIRS",
    "Plan",
    "PrefixLineSSE",
    "SplitResult",
    "best_split",
    "plan_segmentation",
]
