# src/s5_layer.py
"""S5 线宽恢复与分层（决定"像不像原图"，SPEC §4 S5）。

做什么
------
1. **逐基元线宽**：沿基元几何采样，取到最近 S2 轮廓点的距离中位数 ×2。
   不用骨架的距离变换：栅格上 `2×DT` 恒比真实墨宽多 1px（实测 5.9px 笔画量成 6.9px），
   而轮廓点到中心线的距离没有这个偏置。
2. **聚类成档**：一维按 `linewidth.cluster_eps_mm` 切分，档数超出
   [`cluster_levels_min`, `cluster_levels_max`] 时合并最近的档并 WARN。
3. **吸附标准笔宽**：每档吸附到 `linewidth.standard_widths_mm` 最近值，
   偏移超过笔宽 20% 时 WARN（说明档位本身可疑）。
4. **图层**：`LINES_W{宽度}`（小数位 = `layer_name_decimals`），并校验目标图层在
   `dxf.layers` 里存在；不存在则退到最近的 LINES_* 图层并 WARN。
5. **图框检测**：找出靠近页边、能围成矩形的四条正交长线；矩形面积占比落在
   [0.3, `border.max_rect_area_ratio`] 内才判为图框，归入 `BORDER`。
6. **正交化**（可选，`dxf.orthogonalize`）：只把偏离主轴 ≤
   `linewidth.orthogonalize_tolerance_deg` 的**直线**拉正，并记录角度修正量与端点位移。

输出
----
`LayerResult`：每个基元的图层/笔宽/档位/是否图框；聚类摘要；正交化修正清单。
`metrics_s5.json` 记录档数、各层基元数、正交化条数。
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy.spatial import cKDTree

from src.artifacts import StageMetrics, write_json, write_metrics
from src.config import R2VConfig
from src.s4_fit import FitResult, Primitive

#: 图框矩形面积占页面比例的下限（低于此认为不是图框，而是零散线条）
_BORDER_AREA_RATIO_MIN = 0.30

#: 吸附偏移超过笔宽的这个比例就 WARN
_SNAP_WARN_RATIO = 0.20

#: 判定"正交方向"的角度容差（度）
_AXIS_TOLERANCE_DEG = 5.0


class LayerError(ValueError):
    """S5 输入或参数非法。"""


# ---------------------------------------------------------------------------
# 结果模型
# ---------------------------------------------------------------------------
class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class LinewidthCluster(_Model):
    id: int
    width_mm: float
    standard_mm: float
    count: int


class LayerAssignment(_Model):
    primitive_id: int
    kind: str
    layer: str
    width_mm: float
    cluster_id: int | None
    snapped: bool
    snap_delta_mm: float
    is_border: bool


class OrthogonalCorrection(_Model):
    primitive_id: int
    original_start: tuple[float, float]
    original_end: tuple[float, float]
    corrected_start: tuple[float, float]
    corrected_end: tuple[float, float]
    angle_deg: float
    displacement_px: float


class LayerResult(_Model):
    assignments: tuple[LayerAssignment, ...]
    clusters: tuple[LinewidthCluster, ...]
    standard_widths_mm: tuple[float, ...]
    border_primitive_ids: tuple[int, ...]
    orthogonalized: tuple[OrthogonalCorrection, ...]
    warnings: tuple[str, ...]
    counts_by_layer: dict[str, int]

    def to_json(self) -> dict[str, Any]:
        return {
            "assignments": [a.model_dump(mode="json") for a in self.assignments],
            "clusters": [c.model_dump(mode="json") for c in self.clusters],
            "standard_widths_mm": list(self.standard_widths_mm),
            "border_primitive_ids": list(self.border_primitive_ids),
            "orthogonalized": [o.model_dump(mode="json") for o in self.orthogonalized],
            "warnings": list(self.warnings),
            "counts_by_layer": dict(sorted(self.counts_by_layer.items())),
        }


# ---------------------------------------------------------------------------
# 线宽测量
# ---------------------------------------------------------------------------
def _samples_and_normals(primitive: Primitive) -> tuple[np.ndarray, np.ndarray]:
    """基元的等距采样点及其单位法线。"""
    if primitive.kind == "circle" and primitive.center and primitive.radius_px:
        cx, cy = primitive.center
        count = max(24, int(2 * math.pi * primitive.radius_px / 2.0))
        angles = np.linspace(0.0, 2 * math.pi, count, endpoint=False)
        points = np.column_stack([cx + primitive.radius_px * np.cos(angles),
                                  cy + primitive.radius_px * np.sin(angles)])
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        return points, normals
    if primitive.kind == "arc" and primitive.center and primitive.radius_px:
        cx, cy = primitive.center
        start = math.radians(primitive.start_angle_deg or 0.0)
        sweep = math.radians(primitive.sweep_deg or 0.0)
        count = max(8, int(abs(sweep) * primitive.radius_px / 2.0))
        angles = start + sweep * np.linspace(0.0, 1.0, count)
        points = np.column_stack([cx + primitive.radius_px * np.cos(angles),
                                  cy + primitive.radius_px * np.sin(angles)])
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        return points, normals
    if primitive.points:
        points = np.asarray(primitive.points, dtype=np.float64)
    elif primitive.start and primitive.end:
        x0, y0 = primitive.start
        x1, y1 = primitive.end
        count = max(2, int(math.hypot(x1 - x0, y1 - y0) / 2.0) + 1)
        points = np.column_stack([np.linspace(x0, x1, count), np.linspace(y0, y1, count)])
    else:
        raise LayerError(f"基元 {primitive.id} 没有可用于测宽的几何")
    if len(points) < 2:
        raise LayerError(f"基元 {primitive.id} 采样点不足 2")
    tangents = np.zeros_like(points)
    tangents[1:-1] = points[2:] - points[:-2]
    tangents[0] = points[1] - points[0]
    tangents[-1] = points[-1] - points[-2]
    norms = np.hypot(tangents[:, 0], tangents[:, 1])
    norms[norms < 1e-12] = 1.0
    tangents = tangents / norms[:, None]
    normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])
    return points, normals


def primitive_width_px(
    primitive: Primitive,
    tree: cKDTree,
    contour_points: np.ndarray,
    *,
    search_radius_px: float = 15.0,
) -> float:
    """基元的墨宽（px）：两侧轨道沿**法线投影**的距离，取中位数。

    不能用"到最近轮廓点的距离×2"：轮廓是多边形顶点序列（间距≈1px），
    最近点是顶点而不是边，细笔画上会系统性偏大（实测 2.13px 笔画量成 2.73px）。
    投影到法线方向可以把切向的采样误差消掉。
    """
    points, normals = _samples_and_normals(primitive)
    widths: list[float] = []
    for point, normal in zip(points, normals):
        neighbours = tree.query_ball_point(point, search_radius_px)
        if not neighbours:
            continue
        projections = (contour_points[np.asarray(neighbours)] - point) @ normal
        positive = projections[projections > 0.5]
        negative = projections[projections < -0.5]
        if positive.size and negative.size:
            widths.append(float(positive.min() - negative.max()))
    if not widths:
        # 单侧轨道（例如贴着图幅边缘）：退回最近点法，并如实记录在调用方 WARN
        distances = tree.query(points, k=1)[0]
        return float(np.median(distances)) * 2.0
    return float(np.median(widths))


def cluster_widths(values: Sequence[float], *, eps_mm: float) -> list[list[int]]:
    """一维聚类：排序后在相邻间距 > eps 处切开，返回每档的索引列表。"""
    if eps_mm <= 0:
        raise LayerError(f"cluster_eps_mm 必须 > 0，收到 {eps_mm}")
    if not values:
        return []
    order = sorted(range(len(values)), key=lambda i: (values[i], i))
    groups: list[list[int]] = [[order[0]]]
    for previous, current in zip(order, order[1:]):
        if values[current] - values[previous] > eps_mm:
            groups.append([current])
        else:
            groups[-1].append(current)
    return groups


def _merge_to_range(
    groups: list[list[int]], values: Sequence[float], *, minimum: int, maximum: int
) -> tuple[list[list[int]], list[str]]:
    """把档数压到 [minimum, maximum]：超出上限合并间距最小的两档。"""
    warnings: list[str] = []
    groups = [sorted(group) for group in groups]

    def centre(group: list[int]) -> float:
        return float(np.median([values[i] for i in group]))

    while len(groups) > maximum and len(groups) > 1:
        centres = [centre(group) for group in groups]
        gaps = [(centres[i + 1] - centres[i], i) for i in range(len(centres) - 1)]
        gaps.sort()
        _, index = gaps[0]
        merged = sorted(groups[index] + groups[index + 1])
        groups = groups[:index] + [merged] + groups[index + 2:]
        warnings.append(
            f"线宽档数超过 cluster_levels_max={maximum}：合并最近两档（间距 {gaps[0][0]:.4f}mm）"
        )
    if len(groups) < minimum:
        warnings.append(
            f"线宽只聚出 {len(groups)} 档（cluster_levels_min={minimum}）："
            "图中笔宽种类本就少，不强行拆分"
        )
    return groups, warnings


# ---------------------------------------------------------------------------
# 图层
# ---------------------------------------------------------------------------
def _layer_name(width_mm: float, decimals: int) -> str:
    return f"LINES_W{width_mm:.{decimals}f}"


def _resolve_layer(
    desired: str, configured: Sequence[str], *, preferred_width: float
) -> tuple[str, str | None]:
    """返回 (实际图层名, 警告)。`configured` 来自 config.dxf.layers。"""
    if not configured:
        raise LayerError("config.dxf.layers 为空")
    if desired in configured:
        return desired, None
    candidates = [name for name in configured if name.startswith("LINES_")]
    if not candidates:
        return configured[0], f"想要的图层 {desired} 不在 dxf.layers 里，也没有 LINES_* 可用：退到 {configured[0]}"
    best = min(candidates, key=lambda name: abs(_parse_layer_width(name) - preferred_width))
    return best, f"想要的图层 {desired} 不在 dxf.layers 里：退到最近的 {best}"


def _parse_layer_width(name: str) -> float:
    try:
        return float(name.split("_W")[-1])
    except (IndexError, ValueError):
        return 0.0


# ---------------------------------------------------------------------------
# 图框与正交化
# ---------------------------------------------------------------------------
def detect_border(
    primitives: Sequence[Primitive],
    *,
    page_size_px: tuple[int, int],
    max_area_ratio: float,
) -> tuple[list[int], list[str]]:
    """找页边矩形：四条正交长线围出的矩形，面积占比在 [0.3, max_area_ratio]。"""
    width, height = page_size_px
    if width <= 0 or height <= 0:
        raise LayerError(f"page_size_px 非法：{page_size_px}")
    warnings: list[str] = []
    horizontals: list[tuple[float, Primitive]] = []
    verticals: list[tuple[float, Primitive]] = []
    for primitive in primitives:
        if primitive.kind != "line" or not primitive.start or not primitive.end:
            continue
        (x0, y0), (x1, y1) = primitive.start, primitive.end
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        length = math.hypot(dx, dy)
        if length < 0.4 * max(width, height):
            continue
        if dy <= _AXIS_TOLERANCE_DEG * length * math.pi / 180.0:
            horizontals.append(((y0 + y1) / 2.0, primitive))
        elif dx <= _AXIS_TOLERANCE_DEG * length * math.pi / 180.0:
            verticals.append(((x0 + x1) / 2.0, primitive))
    if len(horizontals) < 2 or len(verticals) < 2:
        return [], warnings
    horizontals.sort(key=lambda item: item[0])
    verticals.sort(key=lambda item: item[0])
    top_y, top = horizontals[0]
    bottom_y, bottom = horizontals[-1]
    left_x, left = verticals[0]
    right_x, right = verticals[-1]
    area_ratio = (abs(right_x - left_x) * abs(bottom_y - top_y)) / float(width * height)
    if not (_BORDER_AREA_RATIO_MIN <= area_ratio <= max_area_ratio):
        warnings.append(
            f"候选图框面积占比 {area_ratio:.3f} 不在 [{_BORDER_AREA_RATIO_MIN}, {max_area_ratio}]：不判为图框"
        )
        return [], warnings
    return [top.id, bottom.id, left.id, right.id], warnings


def orthogonalize_lines(
    primitives: Sequence[Primitive],
    *,
    tolerance_deg: float,
) -> tuple[list[OrthogonalCorrection], list[str]]:
    """把偏离主轴 ≤tolerance 的直线拉正，返回修正清单。"""
    corrections: list[OrthogonalCorrection] = []
    warnings: list[str] = []
    if tolerance_deg <= 0:
        return corrections, warnings
    for primitive in primitives:
        if primitive.kind != "line" or not primitive.start or not primitive.end:
            continue
        (x0, y0), (x1, y1) = primitive.start, primitive.end
        dx, dy = x1 - x0, y1 - y0
        angle = math.degrees(math.atan2(dy, dx))
        # 折叠到 [-45, 45)：同时覆盖水平（0°）与竖直（90°）
        folded = (angle + 45.0) % 90.0 - 45.0
        if abs(folded) > tolerance_deg:
            continue
        if abs(folded) <= 45.0 - _AXIS_TOLERANCE_DEG and abs(folded) < 45.0:
            horizontal = abs(folded) <= tolerance_deg and abs(angle) <= 45.0 or abs(abs(angle) - 180.0) <= 45.0
        else:
            horizontal = False
        # 用折叠角判断主轴：|folded| 接近 0 表示接近水平或竖直，用原始角区分
        near_vertical = abs(abs(folded) - 45.0) < 1e-9
        if near_vertical:
            continue
        axis_vertical = abs(abs(angle) - 90.0) <= tolerance_deg
        if axis_vertical:
            new_x0 = new_x1 = (x0 + x1) / 2.0
            new_y0, new_y1 = y0, y1
        else:
            new_y0 = new_y1 = (y0 + y1) / 2.0
            new_x0, new_x1 = x0, x1
        displacement = max(
            math.hypot(new_x0 - x0, new_y0 - y0), math.hypot(new_x1 - x1, new_y1 - y1)
        )
        corrections.append(
            OrthogonalCorrection(
                primitive_id=primitive.id,
                original_start=(round(x0, 4), round(y0, 4)),
                original_end=(round(x1, 4), round(y1, 4)),
                corrected_start=(round(new_x0, 4), round(new_y0, 4)),
                corrected_end=(round(new_x1, 4), round(new_y1, 4)),
                angle_deg=round(folded, 4),
                displacement_px=round(displacement, 4),
            )
        )
    return corrections, warnings


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def run_s5_layer(
    fit: FitResult,
    cfg: R2VConfig,
    *,
    contours: Sequence[np.ndarray],
    mm_per_px: float,
    page_size_px: tuple[int, int] | None = None,
    out_dir: str | Path | None = None,
) -> tuple[LayerResult, StageMetrics]:
    """S5 主流程：测宽 → 聚类 → 吸附 → 分层 → 图框 → 正交化。"""
    t0 = time.perf_counter()
    linewidth_cfg = cfg.linewidth
    if mm_per_px <= 0:
        raise LayerError(f"mm_per_px 必须 > 0，收到 {mm_per_px}")
    arrays = [np.asarray(c, dtype=np.float64) for c in contours if len(c)]
    if not arrays:
        raise LayerError("S5 需要 S2 轮廓点来测笔宽")
    contour_points = np.vstack(arrays)
    tree = cKDTree(contour_points)
    warnings: list[str] = []

    widths_px: dict[int, float] = {}
    for primitive in fit.primitives:
        widths_px[primitive.id] = primitive_width_px(primitive, tree, contour_points)
    widths_mm = {pid: value * mm_per_px for pid, value in widths_px.items()}

    groups = cluster_widths(list(widths_mm.values()), eps_mm=float(linewidth_cfg.cluster_eps_mm))
    groups, merge_warnings = _merge_to_range(
        groups,
        list(widths_mm.values()),
        minimum=int(linewidth_cfg.cluster_levels_min),
        maximum=int(linewidth_cfg.cluster_levels_max),
    )
    warnings.extend(merge_warnings)

    standard = tuple(float(v) for v in linewidth_cfg.standard_widths_mm)
    primitive_ids = list(widths_mm)
    clusters: list[LinewidthCluster] = []
    cluster_of: dict[int, int] = {}
    cluster_standard: dict[int, float] = {}
    for cluster_id, group in enumerate(groups):
        members = [primitive_ids[i] for i in group]
        centre = float(np.median([widths_mm[pid] for pid in members]))
        snapped_to = min(standard, key=lambda value: abs(value - centre))
        if abs(snapped_to - centre) > _SNAP_WARN_RATIO * max(centre, 1e-9):
            warnings.append(
                f"档 {cluster_id} 中心 {centre:.4f}mm 距最近标准笔宽 {snapped_to:.4f}mm 超过 20%：档位可疑"
            )
        clusters.append(
            LinewidthCluster(
                id=cluster_id,
                width_mm=round(centre, 6),
                standard_mm=round(snapped_to, 6),
                count=len(members),
            )
        )
        cluster_standard[cluster_id] = snapped_to
        for pid in members:
            cluster_of[pid] = cluster_id

    configured_layers = tuple(layer.name for layer in cfg.dxf.layers)
    decimals = int(linewidth_cfg.layer_name_decimals)
    assignments: list[LayerAssignment] = []
    border_ids: tuple[int, ...] = ()
    if page_size_px is not None:
        border_ids, border_warnings = detect_border(
            fit.primitives,
            page_size_px=page_size_px,
            max_area_ratio=float(linewidth_cfg.border.max_rect_area_ratio),
        )
        warnings.extend(border_warnings)

    for primitive in fit.primitives:
        cluster_id = cluster_of[primitive.id]
        width = cluster_standard[cluster_id]
        if primitive.id in border_ids:
            layer = "BORDER"
            if "BORDER" not in configured_layers:
                raise LayerError("dxf.layers 里没有 BORDER 图层")
            snapped = True
            delta = width - widths_mm[primitive.id]
        else:
            snapped = bool(linewidth_cfg.snap_to_standard)
            effective = width if snapped else widths_mm[primitive.id]
            delta = width - widths_mm[primitive.id]
            if cfg.dxf.layer_by_linewidth:
                desired = _layer_name(effective, decimals)
            else:
                desired = "LINES_W0.35" if "LINES_W0.35" in configured_layers else configured_layers[0]
            layer, layer_warning = _resolve_layer(desired, configured_layers, preferred_width=effective)
            if layer_warning:
                warnings.append(layer_warning)
        assignments.append(
            LayerAssignment(
                primitive_id=primitive.id,
                kind=primitive.kind,
                layer=layer,
                width_mm=round(width, 6),
                cluster_id=cluster_id,
                snapped=snapped,
                snap_delta_mm=round(delta, 6),
                is_border=primitive.id in border_ids,
            )
        )

    corrections: list[OrthogonalCorrection] = []
    if cfg.dxf.orthogonalize:
        corrections, ortho_warnings = orthogonalize_lines(
            fit.primitives, tolerance_deg=float(linewidth_cfg.orthogonalize_tolerance_deg)
        )
        warnings.extend(ortho_warnings)

    counts_by_layer: dict[str, int] = {}
    for assignment in assignments:
        counts_by_layer[assignment.layer] = counts_by_layer.get(assignment.layer, 0) + 1

    result = LayerResult(
        assignments=tuple(assignments),
        clusters=tuple(clusters),
        standard_widths_mm=standard,
        border_primitive_ids=tuple(border_ids),
        orthogonalized=tuple(corrections),
        warnings=tuple(warnings),
        counts_by_layer=dict(sorted(counts_by_layer.items())),
    )

    timing_ms = round((time.perf_counter() - t0) * 1000.0, 3)
    metrics = StageMetrics(
        stage="s5",
        counts={
            "primitives": len(assignments),
            "clusters": len(clusters),
            "border_primitives": len(border_ids),
            "orthogonalized": len(corrections),
            "warnings": len(warnings),
        },
        timing_ms=timing_ms,
        error_estimate={
            "mm_per_px": round(float(mm_per_px), 12),
            "max_snap_delta_mm": round(
                max((abs(a.snap_delta_mm) for a in assignments), default=0.0), 6
            ),
        },
        warnings=tuple(warnings),
        details={
            "cluster_centres_mm": [c.width_mm for c in clusters],
            "counts_by_layer": result.counts_by_layer,
            "orthogonalize_enabled": bool(cfg.dxf.orthogonalize),
        },
    )

    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "layers.json", result.to_json())
        write_metrics(out, metrics)
    return result, metrics


__all__ = [
    "LayerAssignment",
    "LayerError",
    "LayerResult",
    "LinewidthCluster",
    "OrthogonalCorrection",
    "cluster_widths",
    "detect_border",
    "orthogonalize_lines",
    "primitive_width_px",
    "run_s5_layer",
]
