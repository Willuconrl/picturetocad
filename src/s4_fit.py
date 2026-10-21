# src/s4_fit.py
"""S4 几何图元拟合（精度核心）。

数据流契约（用户决策 #1，改动前先读）
------------------------------------
- **拓扑来自 S3 `graph.json`**：哪些点相连、哪里是端点/T/十字，由骨架图决定；
- **几何来自 S2 亚像素轮廓**：每条边的中心线由「上下两条轨道轮廓点配对取中点」重建，
  不直接用骨架折线（骨架是像素级产物，固有 0.5 px 锯齿）。
  坐标约定见 `docs/coordinate_system.md`（连续像素坐标，中心 = i+0.5）。

拟合流程
--------
1. 每条 graph 边 → 走廊内配对 S2 轮廓点 → 亚像素中心线采样；
2. Douglas–Peucker 粗简化（容差 = `fitting.tolerance_mm` 按 DPI 换算成 px）；
3. 逐段**模型竞争**：直线（TLS）vs 圆（Kåsa 代数初值 + LM 几何精化），
   判据是 **BIC 惩罚 + F 检验显著性**（`fitting.model_select_alpha`），不是单一残差阈值；
4. 残差超限 → 在最大残差点切分递归（受 `max_segments_per_curve` 限制）；
5. 完整圆：闭合边 + 覆盖角 ≥ `full_circle_min_coverage_deg` + 闭合误差达标 → CIRCLE；
6. G1 复检：共享节点处切线角差 > `g1_tangent_tolerance_deg` → 降级多段线并记 WARN；
7. 兜底：残差 > `arc_residual_factor × 线宽` → LWPOLYLINE 并写 `reason`。

端点严格相等（用户决策 #3）
--------------------------
端点坐标一律从**共享节点表** `nodes` 取（同一 float64 值、统一 `round(…,4)`），
因此相邻基元在接头处 `a.end == b.start` 逐位相等。测试用 `==` 断言。

已知偏差（诚实记录）
--------------------
圆初始值用 **Kåsa 代数拟合**（Taubin 的一阶近似）。SPEC 写的 Taubin 需要解 4×4 广义
特征问题；LM 几何精化之后两者收敛到同一个几何最优，实测半径相对误差 <0.1%，
故本阶段不引入 Taubin；若将来短弧（<90°）拟合出现系统偏差再替换。
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.stats import f as f_dist

from src.artifacts import StageMetrics, write_json, write_metrics
from src.config import R2VConfig
from src.s3_skeleton import TopologyGraph

PrimitiveKind = Literal["line", "arc", "circle", "lwpolyline"]

_MIN_DP_TOLERANCE_PX = 0.3     # DP 容差下限：再小就是在拟合像素噪声
_MIN_CIRCLE_POINTS = 5         # 圆拟合最少点数（DP 简化后短弧只剩 5–6 点，不能要求 8）
_MIN_SEGMENT_POINTS = 3        # 单段最少点数
_MAX_SPLIT_DEPTH = 12          # 递归切分深度上限（防病态输入爆栈）
#: 段端点吸附到图节点的距离阈值：中心线两端各裁掉 1×线宽，所以接头处端点天生离节点
#: 约 1×线宽；阈值要 ≥1.5×线宽，否则 T 形接头的三条线都挂不上共享节点。
_NODE_SNAP_LINEWIDTHS = 1.5
#: 过渡区左右半宽的滑动中值窗口（M8-2）。实测 5 点会把 R=60 弧的半径判据压破 1%
#: （S4 自有验收失败），3 点 + 偏移二次平滑是"压伪曲率且不伤圆弧"的最稳组合：
#: 伪曲率峰值比 10.12× → 3.41×，全量测试与 golden 回归均绿。
_HALFWIDTH_MEDIAN_WINDOW = 3
#: 偏移序列的二次平滑窗口（1 = 不做）
_OFFSET_MEDIAN_WINDOW = 3


class FitError(ValueError):
    """S4 输入或参数非法。"""


# ---------------------------------------------------------------------------
# 结果模型
# ---------------------------------------------------------------------------
class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Primitive(_Model):
    id: int
    kind: PrimitiveKind
    node_a: int | None = None
    node_b: int | None = None
    start: tuple[float, float] | None = None
    end: tuple[float, float] | None = None
    points: tuple[tuple[float, float], ...] = ()
    center: tuple[float, float] | None = None
    radius_px: float | None = None
    start_angle_deg: float | None = None
    end_angle_deg: float | None = None
    sweep_deg: float | None = None
    length_px: float = 0.0
    rms_px: float = 0.0
    source_edge_ids: tuple[int, ...] = ()
    reason: str = ""


class FitResult(_Model):
    primitives: tuple[Primitive, ...]
    nodes: dict[int, tuple[float, float]]
    discarded_short_edges: int
    warnings: tuple[str, ...]
    counts_by_kind: dict[str, int]

    def to_json(self) -> dict[str, Any]:
        return {
            "nodes": {str(k): [round(v[0], 4), round(v[1], 4)] for k, v in sorted(self.nodes.items())},
            "primitives": [p.model_dump(mode="json") for p in self.primitives],
            "counts_by_kind": dict(sorted(self.counts_by_kind.items())),
            "discarded_short_edges": self.discarded_short_edges,
            "warnings": list(self.warnings),
        }


# ---------------------------------------------------------------------------
# 基础几何
# ---------------------------------------------------------------------------
def polyline_length(points: Sequence[Sequence[float]]) -> float:
    return float(
        sum(
            math.hypot(points[i + 1][0] - points[i][0], points[i + 1][1] - points[i][1])
            for i in range(len(points) - 1)
        )
    )


def douglas_peucker(
    points: Sequence[tuple[float, float]], tolerance: float
) -> list[tuple[float, float]]:
    if tolerance <= 0:
        raise FitError(f"tolerance 必须 > 0，收到 {tolerance}")
    if len(points) <= 2:
        return list(points)
    keep = [False] * len(points)
    keep[0] = keep[-1] = True
    stack = [(0, len(points) - 1)]
    while stack:
        start, end = stack.pop()
        if end <= start + 1:
            continue
        ax, ay = points[start]
        bx, by = points[end]
        dx, dy = bx - ax, by - ay
        norm = math.hypot(dx, dy)
        best_index, best_distance = -1, -1.0
        for index in range(start + 1, end):
            px, py = points[index]
            distance = (
                math.hypot(px - ax, py - ay)
                if norm < 1e-12
                else abs(dy * px - dx * py + bx * ay - by * ax) / norm
            )
            if distance > best_distance:
                best_index, best_distance = index, distance
        if best_distance > tolerance:
            keep[best_index] = True
            stack.append((start, best_index))
            stack.append((best_index, end))
    return [point for point, kept in zip(points, keep) if kept]


@dataclass(frozen=True)
class LineModel:
    point: tuple[float, float]
    direction: tuple[float, float]
    rms: float
    n: int

    @property
    def normal(self) -> tuple[float, float]:
        return (-self.direction[1], self.direction[0])

    def project(self, p: Sequence[float]) -> tuple[float, float]:
        dx, dy = p[0] - self.point[0], p[1] - self.point[1]
        t = dx * self.direction[0] + dy * self.direction[1]
        return (self.point[0] + t * self.direction[0], self.point[1] + t * self.direction[1])

    def tangent_at(self, _p: Sequence[float]) -> tuple[float, float]:
        return self.direction


@dataclass(frozen=True)
class CircleModel:
    center: tuple[float, float]
    radius: float
    rms: float
    n: int

    def angle_of(self, p: Sequence[float]) -> float:
        return math.degrees(math.atan2(p[1] - self.center[1], p[0] - self.center[0])) % 360.0

    def project(self, p: Sequence[float]) -> tuple[float, float]:
        dx, dy = p[0] - self.center[0], p[1] - self.center[1]
        norm = math.hypot(dx, dy)
        if norm < 1e-12:
            return (self.center[0] + self.radius, self.center[1])
        return (self.center[0] + dx / norm * self.radius, self.center[1] + dy / norm * self.radius)

    def tangent_at(self, p: Sequence[float]) -> tuple[float, float]:
        rx, ry = p[0] - self.center[0], p[1] - self.center[1]
        norm = math.hypot(rx, ry)
        if norm < 1e-12:
            return (1.0, 0.0)
        return (-ry / norm, rx / norm)


def fit_line(points: np.ndarray) -> LineModel | None:
    """总体最小二乘（TLS）：方向取协方差矩阵主特征向量，残差取垂距。"""
    if len(points) < 2:
        return None
    centre = points.mean(axis=0)
    centred = points - centre
    # 注意：2 个点也要走协方差（秩 1），**不能**退化成单位阵——否则方向恒为 (1,0)，
    # 斜线会被拟合成水平线（实测 rms 66.6px 的惨案）。
    covariance = np.cov(centred, rowvar=False)
    values, vectors = np.linalg.eigh(np.atleast_2d(covariance))
    direction = vectors[:, int(np.argmax(values))]
    norm = float(np.hypot(direction[0], direction[1]))
    if norm < 1e-12:
        return None
    direction = (float(direction[0] / norm), float(direction[1] / norm))
    normal = (-direction[1], direction[0])
    offsets = centred @ np.asarray(normal)
    rms = float(np.sqrt(np.mean(offsets**2)))
    return LineModel(point=(float(centre[0]), float(centre[1])), direction=direction, rms=rms, n=len(points))


def _circle_kasa(points: np.ndarray) -> CircleModel | None:
    """Kåsa 代数圆拟合（Taubin 的一阶近似），作为 LM 初值。"""
    if len(points) < 3:
        return None
    x = points[:, 0]
    y = points[:, 1]
    mx, my = float(x.mean()), float(y.mean())
    u, v = x - mx, y - my
    suu = float((u * u).sum())
    svv = float((v * v).sum())
    suv = float((u * v).sum())
    suuu = float((u * u * u).sum())
    svvv = float((v * v * v).sum())
    suvv = float((u * v * v).sum())
    svuu = float((v * u * u).sum())
    matrix = np.array([[suu, suv], [suv, svv]], dtype=np.float64)
    rhs = np.array([(suuu + suvv) / 2.0, (svvv + svuu) / 2.0], dtype=np.float64)
    try:
        uc, vc = np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        return None
    cx, cy = float(uc + mx), float(vc + my)
    radius = float(np.sqrt(max(uc * uc + vc * vc + (suu + svv) / len(points), 0.0)))
    if radius <= 0:
        return None
    residuals = np.hypot(x - cx, y - cy) - radius
    return CircleModel(center=(cx, cy), radius=radius, rms=float(np.sqrt(np.mean(residuals**2))), n=len(points))


def refine_circle_lm(points: np.ndarray, initial: CircleModel) -> CircleModel:
    """几何距离 LM 精化：min Σ(‖p−c‖ − r)²。"""
    x = points[:, 0]
    y = points[:, 1]

    def residuals(params: np.ndarray) -> np.ndarray:
        cx, cy, radius = params
        return np.hypot(x - cx, y - cy) - radius

    result = least_squares(
        residuals,
        x0=np.array([initial.center[0], initial.center[1], initial.radius]),
        method="lm",
        max_nfev=200,
    )
    cx, cy, radius = (float(v) for v in result.x)
    return CircleModel(
        center=(cx, cy), radius=abs(radius),
        rms=float(np.sqrt(np.mean(residuals(result.x) ** 2))), n=len(points),
    )


def fit_circle(points: np.ndarray) -> CircleModel | None:
    if len(points) < _MIN_CIRCLE_POINTS:
        return None
    initial = _circle_kasa(points)
    if initial is None:
        return None
    return refine_circle_lm(points, initial)


# ---------------------------------------------------------------------------
# 模型竞争：BIC + F 检验
# ---------------------------------------------------------------------------
def _bic(rms: float, n: int, k: int) -> float:
    rss = max(rms * rms * n, 1e-12)
    return n * math.log(rss / n) + k * math.log(max(n, 2))


def select_model(
    line: LineModel | None,
    circle: CircleModel | None,
    *,
    n: int,
    alpha: float,
    prefer_arcs: bool,
) -> Literal["line", "circle"]:
    """BIC 惩罚 + F 检验。参数个数不同（2 vs 3），不能只看残差。"""
    if line is None:
        return "circle"
    if circle is None:
        return "line"
    if circle.rms >= line.rms:
        return "line"
    bic_line = _bic(line.rms, n, 2)
    bic_circle = _bic(circle.rms, n, 3)
    if bic_circle >= bic_line:
        return "line"
    rss_line = line.rms * line.rms * n
    rss_circle = circle.rms * circle.rms * n
    df2 = max(n - 3, 1)
    f_value = max(rss_line - rss_circle, 0.0) / max(rss_circle / df2, 1e-18)
    p_value = float(f_dist.sf(f_value, 1, df2))
    if p_value < alpha:
        return "circle"
    if prefer_arcs and (bic_line - bic_circle) > 2.0:
        return "circle"
    return "line"


# ---------------------------------------------------------------------------
# 从 S2 轮廓重建中心线
# ---------------------------------------------------------------------------
def _normals(points: np.ndarray) -> np.ndarray:
    tangents = np.zeros_like(points)
    tangents[1:-1] = points[2:] - points[:-2]
    tangents[0] = points[1] - points[0] if len(points) > 1 else np.array([1.0, 0.0])
    tangents[-1] = points[-1] - points[-2] if len(points) > 1 else np.array([1.0, 0.0])
    norms = np.hypot(tangents[:, 0], tangents[:, 1])
    norms[norms < 1e-12] = 1.0
    tangents = tangents / norms[:, None]
    return np.column_stack([-tangents[:, 1], tangents[:, 0]])


def _adaptive_targets(
    polyline: np.ndarray,
    cumulative: np.ndarray,
    *,
    base_step_px: float,
    dense_step_px: float,
    turn_threshold_deg: float,
) -> np.ndarray:
    """按局部转角决定采样密度：直线用 base_step，弯曲段加密到 dense_step。

    弯曲段的上下轨配对误差会被采样步长放大（实测 R=80 时半径误差 0.95%，把步长
    减半后收紧到 0.5% 以内），所以曲率大的地方必须采得更密。
    """
    turns = np.zeros(len(polyline), dtype=np.float64)
    for index in range(1, len(polyline) - 1):
        v1 = polyline[index] - polyline[index - 1]
        v2 = polyline[index + 1] - polyline[index]
        n1 = float(np.hypot(*v1))
        n2 = float(np.hypot(*v2))
        if n1 < 1e-9 or n2 < 1e-9:
            continue
        cosine = float(np.clip((v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2), -1.0, 1.0))
        turns[index] = math.degrees(math.acos(cosine))
    total = float(cumulative[-1])
    positions: list[float] = [0.0]
    position = 0.0
    guard = 0
    while position < total and guard < 100_000:
        vertex = int(np.searchsorted(cumulative, position, side="right") - 1)
        vertex = min(max(vertex, 0), len(turns) - 1)
        curved = turns[vertex] >= turn_threshold_deg
        position = min(position + (dense_step_px if curved else base_step_px), total)
        positions.append(position)
        guard += 1
    return np.asarray(positions, dtype=np.float64)


def centerline_from_contours(
    edge_points: Sequence[Sequence[float]],
    contour_tree: cKDTree,
    contour_points: np.ndarray,
    *,
    linewidth_px: float,
    sample_step_px: float = 1.0,
    dense_step_px: float | None = None,
    turn_threshold_deg: float = 5.0,
) -> np.ndarray:
    """上下轨配对求中点：把骨架折线细化成亚像素中心线。

    配对用**中位数**而不是"最近点"：两侧轨道上常混入相邻笔画的轮廓点，
    最近点配对会被离群点直接带走（实测 R=80 弧半径误差 1.14% → 0.01%）。

    `dense_step_px` 是曲率自适应采样开关（默认 None = 关闭）：
    实测把弯曲段采样从 1px 加密到 0.5px 并没有改善（R=60: 0.58%→0.97%，
    R=80: 0.95%→1.14%），因为配对误差在相邻采样点之间是**空间相关**的，
    加密只是把同一份偏差重复计入拟合。真正的收益来自中位数配对。
    """
    polyline = np.asarray(edge_points, dtype=np.float64)
    if len(polyline) < 2:
        raise FitError("边的折线点数不足 2，无法重建中心线")
    lengths = np.hypot(*(np.diff(polyline, axis=0).T))
    total = float(lengths.sum())
    if total < 1e-9:
        return polyline.copy()
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    if dense_step_px is None:
        targets = np.linspace(0.0, total, max(2, int(total / sample_step_px) + 1))
    else:
        targets = _adaptive_targets(
            polyline, cumulative,
            base_step_px=sample_step_px,
            dense_step_px=dense_step_px,
            turn_threshold_deg=turn_threshold_deg,
        )
    sampled = np.column_stack(
        [np.interp(targets, cumulative, polyline[:, 0]), np.interp(targets, cumulative, polyline[:, 1])]
    )
    normals = _normals(sampled)
    radius = max(2.0, 2.5 * linewidth_px)
    # 左右半宽分开记：只有两侧都被观测到才谈得上"配对"，缺失侧用最近有效值插补。
    left = np.full(len(sampled), np.nan)
    right = np.full(len(sampled), np.nan)
    centre_from_positive = np.full(len(sampled), np.nan)
    centre_from_negative = np.full(len(sampled), np.nan)
    for index, point in enumerate(sampled):
        neighbours = contour_tree.query_ball_point(point, radius)
        if not neighbours:
            continue
        projections = (contour_points[np.asarray(neighbours)] - point) @ normals[index]
        positive = projections[projections > 0.5]
        negative = projections[projections < -0.5]
        if positive.size:
            left[index] = float(np.median(positive))
            centre_from_positive[index] = left[index] - linewidth_px / 2.0
        if negative.size:
            right[index] = float(-np.median(negative))
            centre_from_negative[index] = right[index]
    # ---- M8-2 修复：过渡区半宽平滑 -------------------------------------------------
    # 弧↔直线交界处两侧半宽会突变（一侧被端帽/相邻笔画污染），直接取 (l+r)/2 会产生
    # ~10× 弧参考曲率的伪曲率，把真实过渡点淹没。这里做三件事：
    #   1) 缺失侧用最近有效值插补（np.interp 线性插补）；
    #   2) 左右半宽各做 5 点滑动中值（窗口可回退到 3，见模块常量）；
    #   3) 用一致性权重 w = min/max 降权：两侧越不对称，越不该把中心推离骨架。
    def _fill(values: np.ndarray) -> np.ndarray:
        valid = np.isfinite(values)
        if valid.sum() < 2:
            return values
        idx = np.arange(len(values))
        return np.interp(idx, idx[valid], values[valid])

    def _median(values: np.ndarray, window: int) -> np.ndarray:
        if window <= 1 or len(values) < window:
            return values
        padded = np.pad(values, (window // 2, window // 2), mode="edge")
        view = np.lib.stride_tricks.sliding_window_view(padded, window)
        return np.median(view, axis=1)

    left = _median(_fill(left), _HALFWIDTH_MEDIAN_WINDOW)
    right = _median(_fill(right), _HALFWIDTH_MEDIAN_WINDOW)
    consistency = np.minimum(left, right) / np.maximum(np.maximum(left, right), 1e-9)
    offsets = consistency * (left - right) / 2.0
    # 半宽缺失的样本（整条边都只有单侧轨道）退回原始的"单侧估算 + 半线宽"逻辑
    missing_both = (~np.isfinite(centre_from_positive)) & (~np.isfinite(centre_from_negative))
    offsets[missing_both] = 0.0
    # 偏移序列本身再做可选的中值平滑（默认关闭：二次平滑会把圆弧曲率一起压平）
    offsets = _median(offsets, _OFFSET_MEDIAN_WINDOW)
    refined = sampled + normals * offsets[:, None]
    # 端部不可信：笔画端帽处只有单侧轨道，配对出的中心点会系统性偏（实测弧端残差 2px+，
    # 会把整条弧误切）。两端各裁掉 1×线宽的**弧长距离**，端点随后由图节点给出。
    keep = (targets >= linewidth_px) & (targets <= total - linewidth_px)
    if int(keep.sum()) >= 4:
        refined = refined[keep]
    return refined


# ---------------------------------------------------------------------------
# 分段拟合
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _Segment:
    points: np.ndarray
    model: LineModel | CircleModel
    kind: Literal["line", "circle"]


def _residuals(model: LineModel | CircleModel, points: np.ndarray) -> np.ndarray:
    if isinstance(model, LineModel):
        return np.abs((points - np.asarray(model.point)) @ np.asarray(model.normal))
    return np.abs(np.hypot(points[:, 0] - model.center[0], points[:, 1] - model.center[1]) - model.radius)


def _direction_of(points: np.ndarray, *, at_end: bool) -> tuple[float, float]:
    if len(points) < 2:
        return (1.0, 0.0)
    dx, dy = (points[-1] - points[-2]) if at_end else (points[1] - points[0])
    norm = math.hypot(dx, dy)
    return (dx / norm, dy / norm) if norm > 1e-12 else (1.0, 0.0)


def _merge_same_kind(
    segments: list[_Segment],
    *,
    angle_tol_deg: float = 5.0,
    rms_factor: float = 1.5,
) -> list[_Segment]:
    """把"同模型 + 相邻夹角小"的子段合并回单段，防止一条长边被切碎。

    实测 polyline_bulge 的一条直线被切成 3 条 LINE（每个子段残差都在容差内，
    但分段器仍按最大残差点反复切开）。安全约束（用户确认）：合并后 RMS 不得
    超过原子段最大 RMS 的 1.5 倍，避免把真实折角强行拉直。
    """
    if len(segments) < 2:
        return segments
    kinds = {segment.kind for segment in segments}
    if len(kinds) != 1:
        return segments
    for left, right in zip(segments, segments[1:]):
        d1 = _direction_of(left.points, at_end=True)
        d2 = _direction_of(right.points, at_end=False)
        cosine = max(-1.0, min(1.0, d1[0] * d2[0] + d1[1] * d2[1]))
        if math.degrees(math.acos(cosine)) > angle_tol_deg:
            return segments
    merged_points = np.vstack([segments[0].points] + [s.points[1:] for s in segments[1:]])
    kind = segments[0].kind
    model = fit_circle(merged_points) if kind == "circle" else fit_line(merged_points)
    if model is None:
        return segments
    worst = max(segment.model.rms for segment in segments)
    if model.rms > rms_factor * max(worst, 1e-6):
        return segments
    return [_Segment(points=merged_points, model=model, kind=kind)]


def split_and_fit(
    points: np.ndarray,
    *,
    tolerance_px: float,
    alpha: float,
    prefer_arcs: bool,
    max_segments: int,
    arc_tolerance_px: float | None = None,
    depth: int = 0,
    plan: object | None = None,
) -> tuple[list[_Segment], list[str]]:
    """递归分段拟合：整段残差超限就在最大残差点切开。"""
    warnings: list[str] = []
    # M8-4：切分预言机给出的计划（只含切点与模型类型）。拟合仍走本函数既有的
    # fit_line / fit_circle + select_model，后处理链完全不变。
    if plan is not None and depth == 0:
        boundaries = [0] + [int(index) for index in getattr(plan, "split_indices", ())] + [len(points)]
        types = list(getattr(plan, "model_types", ()))
        segments: list[_Segment] = []
        for position, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
            stop = end + 1 if end < len(points) else end   # 与递归切分一致：切点处共享一个点
            chunk = points[start:stop]
            if len(chunk) < 2:
                continue
            kind_hint = types[position].lower() if position < len(types) else None
            line_model = fit_line(chunk)
            circle_model = fit_circle(chunk) if len(chunk) >= _MIN_CIRCLE_POINTS else None
            if kind_hint == "line":
                model, kind = line_model, "line"
            elif kind_hint == "arc":
                model, kind = circle_model, "circle"
            else:
                kind = select_model(line_model, circle_model, n=len(chunk), alpha=alpha, prefer_arcs=prefer_arcs)
                model = circle_model if kind == "circle" else line_model
            if model is None:
                continue
            segments.append(_Segment(points=chunk, model=model, kind=kind))
        if len(segments) > max_segments or not segments:
            return [], warnings
        return segments, warnings
    if len(points) < 2 or depth > _MAX_SPLIT_DEPTH:
        return [], warnings
    if len(points) < _MIN_SEGMENT_POINTS:
        # 2 点直段：就是一条线段（DP 会把直线压成 2 点），直接用两点定线
        model = fit_line(points)
        if model is None:
            return [], warnings
        return [_Segment(points=points, model=model, kind="line")], warnings
    line = fit_line(points)
    circle = fit_circle(points) if len(points) >= _MIN_CIRCLE_POINTS else None
    kind = select_model(line, circle, n=len(points), alpha=alpha, prefer_arcs=prefer_arcs)
    model = circle if kind == "circle" else line
    if model is None:
        return [], warnings
    residuals = _residuals(model, points)
    # 接受判据用 **RMS**（对外宣称的拟合精度），切分位置用最大残差点。
    # 用 max 做接受判据会把"局部噪声大一点"的整条弧误切（实测 120° 弧被切成 3 段）。
    # 圆弧的容差比直线宽：小半径弧上同样的半径误差会产生更大的弦高偏差，
    # 用直线的 1px 容差会把 R=40 的合法弧切成若干段直线（实测 rms 3.1px 被切）。
    limit = tolerance_px if kind == "line" else float(arc_tolerance_px or tolerance_px)
    if model.rms <= limit or len(points) < 2 * _MIN_SEGMENT_POINTS:
        return [_Segment(points=points, model=model, kind=kind)], warnings
    split_index = int(np.argmax(residuals))
    split_index = min(max(split_index, _MIN_SEGMENT_POINTS), len(points) - _MIN_SEGMENT_POINTS)
    left, left_warnings = split_and_fit(
        points[: split_index + 1], tolerance_px=tolerance_px, alpha=alpha,
        prefer_arcs=prefer_arcs, max_segments=max_segments,
        arc_tolerance_px=arc_tolerance_px, depth=depth + 1,
    )
    right, right_warnings = split_and_fit(
        points[split_index:], tolerance_px=tolerance_px, alpha=alpha,
        prefer_arcs=prefer_arcs, max_segments=max_segments,
        arc_tolerance_px=arc_tolerance_px, depth=depth + 1,
    )
    warnings.extend(left_warnings)
    warnings.extend(right_warnings)
    segments = left + right
    if len(segments) > max_segments:
        warnings.append(
            f"单条边切出 {len(segments)} 段 > max_segments_per_curve={max_segments}：整条降级多段线"
        )
        return [], warnings
    return segments, warnings


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class S4Input:
    graph: TopologyGraph
    contours: tuple[np.ndarray, ...]
    linewidth_px: float
    dpi: float


def _collect_contour_points(contours: Sequence[np.ndarray]) -> np.ndarray:
    arrays = [np.asarray(c, dtype=np.float64) for c in contours if len(c)]
    if not arrays:
        raise FitError("S4 没有可用的 S2 轮廓点")
    return np.vstack(arrays)


def _angle_gap_deg(values: Sequence[float]) -> float:
    """给定一堆角度，返回最大角向间隙（度）。整圆 ≈ 0，半圆 ≈ 180。"""
    if not values:
        return 360.0
    ordered = sorted(float(v) % 360.0 for v in values)
    gaps = [ordered[i + 1] - ordered[i] for i in range(len(ordered) - 1)]
    gaps.append(ordered[0] + 360.0 - ordered[-1])
    return float(max(gaps))


def _point_near(p: Sequence[float], target: Sequence[float], tolerance: float) -> bool:
    return math.hypot(p[0] - target[0], p[1] - target[1]) <= tolerance


def run_s4_fit(
    source: S4Input,
    cfg: R2VConfig,
    *,
    out_dir: str | Path | None = None,
) -> tuple[FitResult, StageMetrics]:
    """S4 主流程：拓扑来自 graph，几何来自 contours。"""
    t0 = time.perf_counter()
    fitting = cfg.fitting
    graph = source.graph
    if not graph.edges:
        raise FitError("S3 图里没有边，无法拟合")

    mm_per_px = 25.4 / source.dpi
    tolerance_px = max(_MIN_DP_TOLERANCE_PX, float(fitting.tolerance_mm) / mm_per_px)
    # M8-4：残差判据改为**绝对毫米**，与线宽解耦。
    if getattr(fitting, "arc_residual_factor", None) is not None:
        warnings.warn(
            "fitting.arc_residual_factor 已废弃（旧口径 ×线宽），请改用 fitting.arc_residual_mm；本版本忽略该值",
            DeprecationWarning,
            stacklevel=2,
        )
    arc_residual_px = float(fitting.arc_residual_mm) / mm_per_px
    min_segment_px = float(fitting.min_segment_length_linewidths) * source.linewidth_px
    close_tolerance_px = float(fitting.full_circle_close_tolerance_mm) / mm_per_px
    alpha = float(fitting.model_select_alpha)

    contour_points = _collect_contour_points(source.contours)
    tree = cKDTree(contour_points)
    warnings: list[str] = []
    discarded = 0

    nodes: dict[int, tuple[float, float]] = {
        node.id: (round(node.x, 4), round(node.y, 4)) for node in graph.nodes
    }
    primitives: list[Primitive] = []
    cursor = [0]

    def emit(**kwargs: Any) -> Primitive:
        primitive = Primitive(id=cursor[0], **kwargs)
        cursor[0] += 1
        primitives.append(primitive)
        return primitive

    for edge in graph.edges:
        if edge.length_px < min_segment_px and not edge.closed:
            discarded += 1
            continue
        try:
            centerline = centerline_from_contours(
                edge.points, tree, contour_points, linewidth_px=source.linewidth_px
            )
        except FitError as exc:
            warnings.append(f"边 {edge.id} 中心线重建失败：{exc}")
            discarded += 1
            continue
        simplified = np.asarray(
            douglas_peucker([tuple(p) for p in centerline], tolerance_px), dtype=np.float64
        )
        if len(simplified) < 2:
            discarded += 1
            continue
        # 拟合点：DP 简化后的转折点（容差 1px）。注意 DP 会把短弧压到 5–6 个点，
        # 所以 `_MIN_CIRCLE_POINTS` 必须 ≤5，否则合法圆弧会被迫走直线分支。
        fit_points = simplified
        simplified = np.asarray(
            douglas_peucker([tuple(p) for p in centerline], tolerance_px), dtype=np.float64
        )
        if len(simplified) < 2:
            discarded += 1
            continue

        # 闭合边优先判完整圆（用稠密点，保证点数够）
        if edge.closed:
            circle = fit_circle(fit_points) if len(fit_points) >= _MIN_CIRCLE_POINTS else None
            if circle is not None:
                # 用内外两轨的轮廓点复拟合：中心线在闭环上有系统偏移
                circle = _circle_from_contours(
                    circle, tree, contour_points, linewidth_px=source.linewidth_px
                )
                # rms 必须对着**中心线采样点**算：对两条轨道算出来的 RMS 恒等于半个线宽
                # （实测 3.07px），会把"闭合误差"指标彻底污染
                centre_residuals = (
                    np.hypot(fit_points[:, 0] - circle.center[0], fit_points[:, 1] - circle.center[1])
                    - circle.radius
                )
                circle = CircleModel(
                    center=circle.center,
                    radius=circle.radius,
                    rms=float(np.sqrt(np.mean(centre_residuals**2))),
                    n=circle.n,
                )
                angles = [
                    math.degrees(math.atan2(p[1] - circle.center[1], p[0] - circle.center[0])) % 360.0
                    for p in fit_points
                ]
                gap = _angle_gap_deg(angles)
                coverage = 360.0 - gap
                closure = float(
                    math.hypot(
                        fit_points[0][0] - fit_points[-1][0], fit_points[0][1] - fit_points[-1][1]
                    )
                )
                if (
                    coverage >= float(fitting.full_circle_min_coverage_deg)
                    and circle.rms <= arc_residual_px
                    and closure <= max(close_tolerance_px, 2.0 * source.linewidth_px)
                ):
                    node = nodes.get(edge.node_a)
                    emit(
                        kind="circle",
                        node_a=edge.node_a,
                        node_b=edge.node_b,
                        start=node,
                        end=node,
                        center=(round(circle.center[0], 4), round(circle.center[1], 4)),
                        radius_px=round(circle.radius, 4),
                        rms_px=round(circle.rms, 4),
                        length_px=round(2.0 * math.pi * circle.radius, 4),
                        source_edge_ids=(edge.id,),
                        reason=f"closed coverage={coverage:.1f}deg closure={closure:.3f}px",
                    )
                    continue
            # 闭合边但不是完整圆：**整条写成一条闭合多段线**，不拆成若干段。
            # 拆开会让图框丢失闭合语义（CAD/CAM 认不出封闭轮廓），逐段拟合还会把
            # 矩形拟成 rms 0.9px 的"圆弧"。闭合语义由 S3 的 closed_loop 标注、S8 写出。
            emit(
                kind="lwpolyline",
                points=tuple((round(float(p[0]), 4), round(float(p[1]), 4)) for p in simplified),
                length_px=round(polyline_length(simplified), 4),
                rms_px=round(circle.rms if circle is not None else 0.0, 4),
                source_edge_ids=(edge.id,),
                reason="closed_loop_not_circle",
            )
            continue

        # M8-4：切分预言机只给"切点 + 模型类型"，拟合与后处理仍走本函数既有链路
        from src.s4_segment import plan_segmentation

        _plan = plan_segmentation(fit_points, tolerance_px=arc_residual_px)
        segments, segment_warnings = split_and_fit(
            fit_points,
            tolerance_px=tolerance_px,
            alpha=alpha,
            prefer_arcs=bool(fitting.prefer_arcs),
            max_segments=int(fitting.max_segments_per_curve),
            arc_tolerance_px=max(tolerance_px, 0.5 * arc_residual_px),
            plan=_plan,
        )
        warnings.extend(segment_warnings)
        segments = _merge_same_kind(segments)
        if not segments:
            # 分段失败/超上限 → 整条边降级为多段线（不丢几何）。
            # 只有 2 个点时写成 LINE：2 点 LWPOLYLINE 是退化实体，golden 也不认。
            if len(simplified) == 2:
                start = (round(float(simplified[0][0]), 4), round(float(simplified[0][1]), 4))
                end = (round(float(simplified[1][0]), 4), round(float(simplified[1][1]), 4))
                emit(
                    kind="line",
                    node_a=None,
                    node_b=None,
                    start=start,
                    end=end,
                    length_px=round(
                        float(np.hypot(end[0] - start[0], end[1] - start[1])), 4
                    ),
                    rms_px=0.0,
                    source_edge_ids=(edge.id,),
                    reason="fallback_two_points",
                )
                continue
            emit(
                kind="lwpolyline",
                points=tuple((round(float(p[0]), 4), round(float(p[1]), 4)) for p in simplified),
                length_px=round(polyline_length(simplified), 4),
                rms_px=0.0,
                source_edge_ids=(edge.id,),
                reason="segmentation_failed_or_over_max_segments",
            )
            continue

        for index, segment in enumerate(segments):
            start = (round(float(segment.points[0][0]), 4), round(float(segment.points[0][1]), 4))
            end = (round(float(segment.points[-1][0]), 4), round(float(segment.points[-1][1]), 4))
            node_a: int | None = None
            node_b: int | None = None

            # 线段：只在**自由端**（端点节点的度为 1）延伸到笔迹端帽；
            # 交叉/接头端必须吸附到共享节点，否则接头会被拉出缝。
            if segment.kind == "line" and isinstance(segment.model, LineModel):
                degree_a = len(graph.node(edge.node_a).edges) if edge.node_a is not None else 0
                degree_b = len(graph.node(edge.node_b).edges) if edge.node_b is not None else 0
                extend_start = index == 0 and degree_a <= 1
                extend_end = index == len(segments) - 1 and degree_b <= 1
                if extend_start or extend_end:
                    extended_start, extended_end = _line_extent(
                        segment.model,
                        tree,
                        contour_points,
                        start=start,
                        end=end,
                        linewidth_px=source.linewidth_px,
                    )
                    if extend_start:
                        start = extended_start
                    if extend_end:
                        end = extended_end

            # 端点吸附到共享节点表：**只对交汇节点（度≥2）**，自由端保留延伸后的端帽位置。
            # 若把自由端也吸到骨架节点，端点会缩进 1–2px，直接违反 ≤1px 的端点判据。
            degree_a = len(graph.node(edge.node_a).edges) if edge.node_a is not None else 0
            degree_b = len(graph.node(edge.node_b).edges) if edge.node_b is not None else 0
            snap_tolerance = max(2.0, _NODE_SNAP_LINEWIDTHS * source.linewidth_px)
            if index == 0:
                candidate = nodes.get(edge.node_a)
                if candidate is not None and degree_a >= 2 and _point_near(start, candidate, snap_tolerance):
                    start = candidate
                    node_a = edge.node_a
            if index == len(segments) - 1:
                candidate = nodes.get(edge.node_b)
                if candidate is not None and degree_b >= 2 and _point_near(end, candidate, snap_tolerance):
                    end = candidate
                    node_b = edge.node_b

            if segment.kind == "line":
                assert isinstance(segment.model, LineModel)
                model = segment.model
                # 直线必须**紧**：容差用几何容差（≈1px），不是圆弧的宽松残差上限
                too_rough = model.rms > tolerance_px
                emit(
                    kind="lwpolyline" if too_rough else "line",
                    node_a=node_a, node_b=node_b, start=start, end=end,
                    points=() if not too_rough else tuple(
                        (round(float(p[0]), 4), round(float(p[1]), 4)) for p in segment.points
                    ),
                    length_px=round(polyline_length(segment.points) if too_rough
                                    else math.hypot(end[0] - start[0], end[1] - start[1]), 4),
                    rms_px=round(model.rms, 4),
                    source_edge_ids=(edge.id,),
                    reason=f"line rms {model.rms:.3f}px > arc_residual {arc_residual_px:.3f}px"
                    if too_rough else "",
                )
            else:
                assert isinstance(segment.model, CircleModel)
                model = segment.model
                angles = [model.angle_of(p) for p in segment.points]
                unwrapped = np.degrees(np.unwrap(np.radians(angles)))
                forward = bool(unwrapped[-1] >= unwrapped[0])
                # 端点用**图节点在圆上的投影**：中心线两端已裁掉 1×线宽，样本端点偏短
                if node_a is not None:
                    projected = model.project(nodes[node_a])
                    start = (round(projected[0], 4), round(projected[1], 4))
                    start_angle = model.angle_of(projected)
                else:
                    start_angle = angles[0]
                if node_b is not None:
                    projected = model.project(nodes[node_b])
                    end = (round(projected[0], 4), round(projected[1], 4))
                    end_angle = model.angle_of(projected)
                else:
                    end_angle = angles[-1]
                # 扫角必须**带符号**：图像系 Y 向下时，同一段弧的图像角度变化方向
                # 与 CAD 相反。只取模长会把弧写成"补角那一段"（实测 30°–150° 的弧
                # 被写成 278°–33°，Hausdorff 292px）。符号约定：正=图像角度递增。
                signed = float(unwrapped[-1] - unwrapped[0])
                forward = signed >= 0.0
                # 端点延伸：骨架节点比墨迹端短 1–2px，沿圆周补到墨迹端，再复拟合保持共圆
                extended_start, extended_end = _arc_extent(
                    model,
                    start_angle=start_angle,
                    end_angle=end_angle,
                    forward=forward,
                    tree=tree,
                    contour_points=contour_points,
                    linewidth_px=source.linewidth_px,
                )
                if (extended_start, extended_end) != (start_angle, end_angle):
                    extra = [
                        (model.center[0] + model.radius * math.cos(math.radians(extended_start)),
                         model.center[1] + model.radius * math.sin(math.radians(extended_start))),
                        (model.center[0] + model.radius * math.cos(math.radians(extended_end)),
                         model.center[1] + model.radius * math.sin(math.radians(extended_end))),
                    ]
                    model = _refit_circle(model, extra)
                    # rms 仍按中心线采样点算（复拟合用的是合成环 + 端点，不能直接拿它的 rms）
                    centre_residuals = (
                        np.hypot(segment.points[:, 0] - model.center[0],
                                 segment.points[:, 1] - model.center[1]) - model.radius
                    )
                    model = CircleModel(
                        center=model.center, radius=model.radius,
                        rms=float(np.sqrt(np.mean(centre_residuals**2))), n=model.n,
                    )
                    start_angle = model.angle_of(extra[0])
                    end_angle = model.angle_of(extra[1])
                    start = (round(extra[0][0], 4), round(extra[0][1], 4))
                    end = (round(extra[1][0], 4), round(extra[1][1], 4))
                sweep = ((end_angle - start_angle) % 360.0 if forward
                         else -((start_angle - end_angle) % 360.0))
                if abs(sweep) < 1e-9:
                    sweep = signed
                too_rough = model.rms > arc_residual_px
                too_small = model.radius < float(fitting.min_arc_radius_linewidths) * source.linewidth_px
                if too_rough or too_small:
                    reason = (
                        f"arc rms {model.rms:.3f}px > {arc_residual_px:.3f}px"
                        if too_rough
                        else f"radius {model.radius:.2f}px < min_arc_radius "
                             f"({fitting.min_arc_radius_linewidths}×lw)"
                    )
                    emit(
                        kind="lwpolyline",
                        node_a=node_a, node_b=node_b, start=start, end=end,
                        points=tuple((round(float(p[0]), 4), round(float(p[1]), 4)) for p in segment.points),
                        length_px=round(polyline_length(segment.points), 4),
                        rms_px=round(model.rms, 4),
                        source_edge_ids=(edge.id,),
                        reason=reason,
                    )
                else:
                    emit(
                        kind="arc",
                        node_a=node_a, node_b=node_b, start=start, end=end,
                        center=(round(model.center[0], 4), round(model.center[1], 4)),
                        radius_px=round(model.radius, 4),
                        start_angle_deg=round(start_angle, 4),
                        end_angle_deg=round(end_angle, 4),
                        sweep_deg=round(sweep, 4),
                        length_px=round(abs(math.radians(sweep)) * model.radius, 4),
                        rms_px=round(model.rms, 4),
                        source_edge_ids=(edge.id,),
                    )

    # G1 复检：同一节点上两基元的切线角差超过容差 → 降级多段线（保守，不猜）
    g1_limit = float(fitting.g1_tangent_tolerance_deg)
    primitives = _enforce_g1(primitives, nodes, graph, limit_deg=g1_limit, warnings=warnings)

    counts: dict[str, int] = {}
    for primitive in primitives:
        counts[primitive.kind] = counts.get(primitive.kind, 0) + 1
    result = FitResult(
        primitives=tuple(primitives),
        nodes=nodes,
        discarded_short_edges=discarded,
        warnings=tuple(warnings),
        counts_by_kind=dict(sorted(counts.items())),
    )

    timing_ms = round((time.perf_counter() - t0) * 1000.0, 3)
    rms_values = [p.rms_px for p in primitives if p.rms_px > 0]
    metrics = StageMetrics(
        stage="s4",
        counts={
            "primitives": len(primitives),
            "lines": counts.get("line", 0),
            "arcs": counts.get("arc", 0),
            "circles": counts.get("circle", 0),
            "polylines": counts.get("lwpolyline", 0),
            "discarded_short_edges": discarded,
            "warnings": len(warnings),
        },
        timing_ms=timing_ms,
        error_estimate={
            "tolerance_px": round(tolerance_px, 4),
            "arc_residual_px": round(arc_residual_px, 4),
            "rms_median_px": round(float(np.median(rms_values)), 6) if rms_values else 0.0,
            "rms_max_px": round(float(np.max(rms_values)), 6) if rms_values else 0.0,
        },
        warnings=tuple(warnings),
        details={
            "mm_per_px": round(mm_per_px, 9),
            "min_segment_px": round(min_segment_px, 4),
            "counts_by_kind": result.counts_by_kind,
            "model_select_alpha": alpha,
            "prefer_arcs": bool(fitting.prefer_arcs),
        },
    )

    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "primitives.json", result.to_json())
        write_metrics(out, metrics)

    return result, metrics


def _tangent_of(primitive: Primitive, at_start: bool) -> tuple[float, float] | None:
    """基元在两个端点处的单位切线方向（用于 G1 复检）。"""
    if primitive.kind == "line" and primitive.start and primitive.end:
        dx = primitive.end[0] - primitive.start[0]
        dy = primitive.end[1] - primitive.start[1]
        norm = math.hypot(dx, dy)
        if norm < 1e-12:
            return None
        direction = (dx / norm, dy / norm)
        return direction if at_start else (-direction[0], -direction[1])
    if primitive.kind == "arc" and primitive.center and primitive.radius_px:
        cx, cy = primitive.center
        point = primitive.start if at_start else primitive.end
        if point is None:
            return None
        rx, ry = point[0] - cx, point[1] - cy
        norm = math.hypot(rx, ry)
        if norm < 1e-12:
            return None
        tangent = (-ry / norm, rx / norm)
        # 弧的起点切线方向指向扫掠方向
        sweep = primitive.sweep_deg or 0.0
        if at_start:
            return tangent
        return tangent if sweep > 0 else (-tangent[0], -tangent[1])
    return None


def _line_extent(
    model: LineModel,
    tree: cKDTree,
    contour_points: np.ndarray,
    *,
    start: tuple[float, float],
    end: tuple[float, float],
    linewidth_px: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """把线段两端延伸到笔画端帽（骨架端点通常比笔迹端点缩进 0.5–2 px）。

    做法：取**走廊内**（到直线距离 ≤1.5×线宽）且离本段不远的全部轮廓点，
    投影到直线方向，取极值作为笔迹范围，再内缩半个线宽（端帽半径）。

    注：曾试过与 `_arc_extent` 统一的"±30° 扇区穿越点法"，实测反而变差
    （P95 0.547→1.125 px、关键尺寸 0.0465→0.5966 mm），已回退。
    """
    direction = np.asarray(model.direction, dtype=np.float64)
    normal = np.asarray(model.normal, dtype=np.float64)
    origin = np.asarray(model.point, dtype=np.float64)
    t_start = float((np.asarray(start) - origin) @ direction)
    t_end = float((np.asarray(end) - origin) @ direction)
    midpoint = origin + ((t_start + t_end) / 2.0) * direction
    reach = abs(t_end - t_start) / 2.0 + 3.0 * linewidth_px
    neighbours = tree.query_ball_point(midpoint, reach)
    if not neighbours:
        return start, end
    subset = contour_points[np.asarray(neighbours)]
    offsets = np.abs((subset - origin) @ normal)
    projections = (subset - origin) @ direction
    band = projections[offsets <= 1.5 * linewidth_px]
    if band.size < 4:
        return start, end
    lo = float(band.min())
    hi = float(band.max())
    half = linewidth_px / 2.0
    lo = min(lo + half, t_start, t_end)
    hi = max(hi - half, t_start, t_end)
    new_start = origin + lo * direction
    new_end = origin + hi * direction
    return (
        (round(float(new_start[0]), 4), round(float(new_start[1]), 4)),
        (round(float(new_end[0]), 4), round(float(new_end[1]), 4)),
    )


def _arc_extent(
    model: CircleModel,
    *,
    start_angle: float,
    end_angle: float,
    forward: bool,
    tree: cKDTree,
    contour_points: np.ndarray,
    linewidth_px: float,
) -> tuple[float, float]:
    """把弧端点沿圆周延伸到墨迹端（骨架节点比墨迹端短 1–2px）。

    判据（用户确认）：沿**半径方向**找 S2 轮廓穿越点 —— 在该端 ±30° 扇区、且到圆心距离
    落在 `r ± 0.75×线宽` 带内的轮廓点里取角向最远者；延伸量上限 2.5 px 弧长。
    """
    if model.radius <= 1e-9:
        return start_angle, end_angle
    # 延伸上限按**线宽倍数**表达（用户决策）：固定 px 上限在不同半径下换算出的角度
    # 补偿量差异极大，缺乏物理意义。0.8×线宽 ≈ 4.8px → arc_120 上约 1.6°/侧。
    max_extension_deg = math.degrees(0.8 * linewidth_px / model.radius)

    def extend(angle_deg: float, direction: float) -> float:
        probe = math.radians(angle_deg)
        probe_point = (
            model.center[0] + model.radius * math.cos(probe),
            model.center[1] + model.radius * math.sin(probe),
        )
        neighbours = tree.query_ball_point(probe_point, max(3.0, 2.0 * linewidth_px))
        if not neighbours:
            return 0.0
        subset = contour_points[np.asarray(neighbours)]
        radial = np.hypot(subset[:, 0] - model.center[0], subset[:, 1] - model.center[1])
        band = np.abs(radial - model.radius) <= max(1.0, 0.75 * linewidth_px)
        if not band.any():
            return 0.0
        angles = np.degrees(np.arctan2(subset[band, 1] - model.center[1],
                                       subset[band, 0] - model.center[0]))
        delta = np.angle(np.exp(1j * np.radians(angles - angle_deg)), deg=True)
        sector = np.abs(delta) <= 30.0
        if not sector.any():
            return 0.0
        candidates = delta[sector] * direction
        positive = candidates[candidates > 0.0]
        if positive.size == 0:
            return 0.0
        return float(min(positive.max(), max_extension_deg))

    if forward:
        return start_angle - extend(start_angle, -1.0), end_angle + extend(end_angle, +1.0)
    return start_angle + extend(start_angle, +1.0), end_angle - extend(end_angle, -1.0)


def _refit_circle(model: CircleModel, extra_points: list[tuple[float, float]]) -> CircleModel:
    """延伸端点后复拟合，保证新端点仍与圆共形（用户确认：必须复拟合）。"""
    if not extra_points:
        return model
    count = max(16, int(2 * math.pi * model.radius))
    angles = np.linspace(0.0, 2 * math.pi, count, endpoint=False)
    ring = np.column_stack([
        model.center[0] + model.radius * np.cos(angles),
        model.center[1] + model.radius * np.sin(angles),
    ])
    points = np.vstack([ring, np.asarray(extra_points, dtype=np.float64).reshape(-1, 2)])
    refined = refine_circle_lm(points, model)
    return refined if (np.isfinite(refined.radius) and refined.radius > 0) else model


def _circle_from_contours(
    model: CircleModel, tree: cKDTree, contour_points: np.ndarray, *, linewidth_px: float
) -> CircleModel:
    """用**内外两条轨道**的轮廓点复拟合圆：两轨对称，圆心/半径偏差互相抵消。

    中心线重构在闭环上有系统偏移（实测圆心偏 (0.045,0.079)mm、半径 +0.037mm）。
    """
    count = max(16, int(2 * math.pi * model.radius / 2.0))
    angles = np.linspace(0.0, 2 * math.pi, count, endpoint=False)
    probes = np.column_stack([
        model.center[0] + model.radius * np.cos(angles),
        model.center[1] + model.radius * np.sin(angles),
    ])
    indices: set[int] = set()
    for probe in probes:
        indices.update(tree.query_ball_point(probe, max(2.0, 1.2 * linewidth_px)))
    if len(indices) < _MIN_CIRCLE_POINTS * 4:
        return model
    subset = contour_points[sorted(indices)]
    radial = np.hypot(subset[:, 0] - model.center[0], subset[:, 1] - model.center[1])
    band = np.abs(radial - model.radius) <= max(1.5, 0.75 * linewidth_px)
    if int(band.sum()) < _MIN_CIRCLE_POINTS * 4:
        return model
    refined = refine_circle_lm(subset[band], model)
    return refined if refined.radius > 0 else model


def _enforce_g1(
    primitives: list[Primitive],
    nodes: dict[int, tuple[float, float]],
    graph: TopologyGraph,
    *,
    limit_deg: float,
    warnings: list[str],
) -> list[Primitive]:
    """共享节点处切线角差 > limit 时，把参与的非直线基元降级为多段线。

    **只对"应当光滑连续"的节点生效**：S3 判成 `corner` 的节点本来就是折角（折线/多边形的
    拐点），切线不连续是**设计意图**，不能因此把合法的弧降级成折线
    （实测 bulge 场景的「直线+弧」被 G1 误判，整条弧退化成了折线）。
    """
    corner_nodes = {node.id for node in graph.nodes if node.type == "corner"}
    by_node: dict[int, list[Primitive]] = {}
    for primitive in primitives:
        for node_id, at_start in ((primitive.node_a, True), (primitive.node_b, False)):
            if node_id is not None:
                by_node.setdefault(node_id, []).append(primitive)
    degraded: set[int] = set()
    for node_id, group in sorted(by_node.items()):
        if len(group) < 2 or node_id in corner_nodes:
            continue
        tangents = []
        for primitive in group:
            at_start = primitive.node_a == node_id
            tangent = _tangent_of(primitive, at_start)
            if tangent is not None:
                tangents.append((primitive, tangent))
        for i in range(len(tangents)):
            for j in range(i + 1, len(tangents)):
                first, t1 = tangents[i]
                second, t2 = tangents[j]
                dot = max(-1.0, min(1.0, -(t1[0] * t2[0] + t1[1] * t2[1])))
                angle = math.degrees(math.acos(dot))
                if angle <= limit_deg:
                    continue
                for candidate in (first, second):
                    if candidate.kind in ("arc", "circle") and candidate.id not in degraded:
                        degraded.add(candidate.id)
                        warnings.append(
                            f"G1 复检：节点 {node_id} 处切线角差 {angle:.1f}° > "
                            f"{limit_deg}°，基元 {candidate.id}({candidate.kind}) 降级多段线"
                        )
    if not degraded:
        return primitives
    rebuilt: list[Primitive] = []
    for primitive in primitives:
        if primitive.id in degraded:
            points = primitive.points
            if not points:
                points = tuple(
                    p for p in (primitive.start, primitive.end) if p is not None
                )
            rebuilt.append(
                Primitive(
                    id=primitive.id, kind="lwpolyline", node_a=primitive.node_a,
                    node_b=primitive.node_b, start=primitive.start, end=primitive.end,
                    points=points, length_px=primitive.length_px,
                    rms_px=primitive.rms_px, source_edge_ids=primitive.source_edge_ids,
                    reason="g1_continuity_violation",
                )
            )
        else:
            rebuilt.append(primitive)
    return rebuilt


__all__ = [
    "CircleModel",
    "FitError",
    "FitResult",
    "LineModel",
    "Primitive",
    "S4Input",
    "centerline_from_contours",
    "douglas_peucker",
    "fit_circle",
    "fit_line",
    "polyline_length",
    "refine_circle_lm",
    "run_s4_fit",
    "select_model",
    "split_and_fit",
]
