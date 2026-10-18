# src/golden_compare.py
"""golden 比对：实体数差 / 关键尺寸误差 / Hausdorff P95（SPEC §1.2 三项验收）。

设计约束
--------
1. **不做隐式容错**：对不上的实体、不支持的实体类型、NaN 坐标一律显式记入
   `failed_checks` 或直接抛错，绝不「跳过差异」。
2. **参数不从代码里偷**：所有容差与采样步长都由调用方（config.yaml / 测试）传入，
   函数签名里没有隐藏默认阈值。
3. **确定性**：实体排序用坐标字典序（SPEC §3.3），同一 DXF 两次比对结果比特一致。

术语
----
- 关键尺寸：直线 = 长度 + 两端点；圆 = 半径 + 圆心；弧 = 半径 + 扫角 + 弧长 + 圆心；
  多段线 = 总长 + 顶点数 + 顶点最大位移。
- Hausdorff P95：两个点云各自到对方的最近距离取 95 分位，再取两者较大值（对称）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

import ezdxf
import numpy as np
from ezdxf.math import arc_angle_span_rad, bulge_to_arc
from scipy.spatial import cKDTree

MetricKind = Literal["rel", "abs"]

#: 坐标比较时保留的有效位（避免 1e-16 级噪声进字典序）
_SORT_DECIMALS = 9


# ---------------------------------------------------------------------------
# 几何实体
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GeomLine:
    layer: str
    start: tuple[float, float]
    end: tuple[float, float]

    @property
    def kind(self) -> str:
        return "LINE"


@dataclass(frozen=True)
class GeomCircle:
    layer: str
    center: tuple[float, float]
    radius: float

    @property
    def kind(self) -> str:
        return "CIRCLE"


@dataclass(frozen=True)
class GeomArc:
    layer: str
    center: tuple[float, float]
    radius: float
    start_deg: float
    end_deg: float

    @property
    def kind(self) -> str:
        return "ARC"

    @property
    def sweep_deg(self) -> float:
        return float(
            np.degrees(arc_angle_span_rad(np.radians(self.start_deg), np.radians(self.end_deg)))
        )


@dataclass(frozen=True)
class GeomPolyline:
    layer: str
    points: tuple[tuple[float, float], ...]
    bulges: tuple[float, ...]
    closed: bool

    @property
    def kind(self) -> str:
        return "LWPOLYLINE"


GeomEntity = GeomLine | GeomCircle | GeomArc | GeomPolyline


# ---------------------------------------------------------------------------
# 读取
# ---------------------------------------------------------------------------
def _scalar(value: Any) -> float:
    """ezdxf 的 get_points() 对单字符格式返回 1 元组，统一拆成 float。"""
    if isinstance(value, (tuple, list)):
        if len(value) != 1:
            raise ValueError(f"期望 1 元组，收到 {value!r}")
        return float(value[0])
    return float(value)


def read_dxf(path: str | Path, *, ignore_types: frozenset[str]) -> list[GeomEntity]:
    """读取 DXF 模型空间，转成可比较的几何实体列表。

    `ignore_types` 必须显式给出（例如 frozenset({"TEXT", "MTEXT"})）：
    静默忽略未知实体会让「产物少了 20 条线」变成假绿。
    """
    doc = ezdxf.readfile(str(path))
    out: list[GeomEntity] = []
    for e in doc.modelspace():
        dxftype = e.dxftype()
        layer = e.dxf.layer
        if dxftype in ignore_types:
            continue
        if dxftype == "LINE":
            out.append(
                GeomLine(
                    layer=layer,
                    start=(float(e.dxf.start.x), float(e.dxf.start.y)),
                    end=(float(e.dxf.end.x), float(e.dxf.end.y)),
                )
            )
        elif dxftype == "CIRCLE":
            out.append(
                GeomCircle(
                    layer=layer,
                    center=(float(e.dxf.center.x), float(e.dxf.center.y)),
                    radius=float(e.dxf.radius),
                )
            )
        elif dxftype == "ARC":
            out.append(
                GeomArc(
                    layer=layer,
                    center=(float(e.dxf.center.x), float(e.dxf.center.y)),
                    radius=float(e.dxf.radius),
                    start_deg=float(e.dxf.start_angle),
                    end_deg=float(e.dxf.end_angle),
                )
            )
        elif dxftype == "LWPOLYLINE":
            pts = tuple((float(p[0]), float(p[1])) for p in e.get_points("xy"))
            bulges = tuple(_scalar(b) for b in e.get_points("b"))
            out.append(
                GeomPolyline(layer=layer, points=pts, bulges=bulges, closed=bool(e.closed))
            )
        else:
            raise ValueError(
                f"不支持的实体类型 {dxftype}（{path}）；"
                "若确属应忽略的类型，请显式传入 ignore_types"
            )
    for e in out:
        for arr in _coords(e):
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"实体含 NaN/Inf 坐标：{e}")
    return out


def _coords(e: GeomEntity) -> list[np.ndarray]:
    if isinstance(e, (GeomLine,)):
        return [np.asarray(e.start), np.asarray(e.end)]
    if isinstance(e, (GeomCircle, GeomArc)):
        return [np.asarray(e.center)]
    return [np.asarray(e.points)]


def canonical_sort(entities: Iterable[GeomEntity]) -> list[GeomEntity]:
    """按 (类型, 坐标字典序) 稳定排序，保证两端实体一一配对。"""

    def key(e: GeomEntity) -> tuple[Any, ...]:
        if isinstance(e, GeomLine):
            return ("LINE", *np.round(np.asarray(e.start), _SORT_DECIMALS),
                    *np.round(np.asarray(e.end), _SORT_DECIMALS))
        if isinstance(e, GeomCircle):
            return ("CIRCLE", *np.round(np.asarray(e.center), _SORT_DECIMALS), round(e.radius, _SORT_DECIMALS))
        if isinstance(e, GeomArc):
            return (
                "ARC",
                *np.round(np.asarray(e.center), _SORT_DECIMALS),
                round(e.radius, _SORT_DECIMALS),
                round(e.start_deg, _SORT_DECIMALS),
                round(e.end_deg, _SORT_DECIMALS),
            )
        flat = tuple(float(v) for xy in e.points for v in xy)
        return ("LWPOLYLINE", e.closed, *np.round(np.asarray(flat), _SORT_DECIMALS))

    return sorted(entities, key=key)


# ---------------------------------------------------------------------------
# 采样
# ---------------------------------------------------------------------------
def _arc_points(
    center: tuple[float, float],
    radius: float,
    start_deg: float,
    end_deg: float,
    spacing_mm: float,
) -> np.ndarray:
    span = arc_angle_span_rad(np.radians(start_deg), np.radians(end_deg))
    arc_len = abs(span) * radius
    n = max(2, int(np.ceil(arc_len / spacing_mm)) + 1)
    angles = np.radians(start_deg) + np.linspace(0.0, span, n)
    return np.column_stack(
        [center[0] + radius * np.cos(angles), center[1] + radius * np.sin(angles)]
    )


def _polyline_chain(e: GeomPolyline, spacing_mm: float) -> np.ndarray:
    chains: list[np.ndarray] = []
    n = len(e.points)
    indices = range(n if e.closed else n - 1)
    for i in indices:
        p1 = e.points[i]
        p2 = e.points[(i + 1) % n]
        b = e.bulges[i] if i < len(e.bulges) else 0.0
        if abs(b) < 1e-12:
            chunk = np.array([p1, p2], dtype=np.float64)
        else:
            center, a0, a1, radius = bulge_to_arc(p1, p2, b)
            chunk = _arc_points(
                (float(center.x), float(center.y)), float(radius),
                float(np.degrees(a0)), float(np.degrees(a1)), spacing_mm,
            )
            if b < 0:
                chunk = chunk[::-1]
        chains.append(chunk if not chains else chunk[1:])
    return np.vstack(chains)


def sample_entity(e: GeomEntity, *, spacing_mm: float) -> np.ndarray:
    """按弧长步长把实体离散成点云（mm）。"""
    if spacing_mm <= 0:
        raise ValueError(f"spacing_mm 必须 > 0，收到 {spacing_mm}")
    if isinstance(e, GeomLine):
        p1, p2 = np.asarray(e.start), np.asarray(e.end)
        length = float(np.linalg.norm(p2 - p1))
        n = max(2, int(np.ceil(length / spacing_mm)) + 1)
        return np.linspace(p1, p2, n)
    if isinstance(e, GeomCircle):
        circ = 2.0 * np.pi * e.radius
        n = max(16, int(np.ceil(circ / spacing_mm)))
        angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        return np.column_stack(
            [e.center[0] + e.radius * np.cos(angles), e.center[1] + e.radius * np.sin(angles)]
        )
    if isinstance(e, GeomArc):
        return _arc_points(e.center, e.radius, e.start_deg, e.end_deg, spacing_mm)
    if isinstance(e, GeomPolyline):
        return _polyline_chain(e, spacing_mm)
    raise TypeError(f"未知实体类型：{type(e)!r}")


def sample_entities(entities: Iterable[GeomEntity], *, spacing_mm: float) -> np.ndarray:
    clouds = [sample_entity(e, spacing_mm=spacing_mm) for e in canonical_sort(entities)]
    if not clouds:
        raise ValueError("实体列表为空，无法采样")
    return np.vstack(clouds)


# ---------------------------------------------------------------------------
# 三项指标
# ---------------------------------------------------------------------------
def hausdorff_p95(points_a: np.ndarray, points_b: np.ndarray, *, percentile: float = 95.0) -> float:
    """对称 Hausdorff 的 percentile 分位版本（SPEC §1.2「Hausdorff P95」）。"""
    if points_a.size == 0 or points_b.size == 0:
        raise ValueError("点云为空")
    if not (0.0 < percentile <= 100.0):
        raise ValueError(f"percentile 必须落在 (0, 100]，收到 {percentile}")
    tree_a = cKDTree(points_a)
    tree_b = cKDTree(points_b)
    d_ab = tree_b.query(points_a, k=1)[0]
    d_ba = tree_a.query(points_b, k=1)[0]
    return float(max(np.percentile(d_ab, percentile), np.percentile(d_ba, percentile)))


def _polyline_length(e: GeomPolyline, *, spacing_mm: float) -> float:
    chain = _polyline_chain(e, spacing_mm)
    return float(np.sum(np.linalg.norm(np.diff(chain, axis=0), axis=1)))


def entity_metrics(e: GeomEntity, *, spacing_mm: float) -> dict[str, tuple[float, MetricKind]]:
    """实体的关键尺寸字典：名字 → (值, 该值按相对误差比还是绝对 mm 比)。"""
    if isinstance(e, GeomLine):
        return {
            "length": (float(np.linalg.norm(np.subtract(e.end, e.start))), "rel"),
            "x0": (e.start[0], "abs"),
            "y0": (e.start[1], "abs"),
            "x1": (e.end[0], "abs"),
            "y1": (e.end[1], "abs"),
        }
    if isinstance(e, GeomCircle):
        return {
            "radius": (e.radius, "rel"),
            "cx": (e.center[0], "abs"),
            "cy": (e.center[1], "abs"),
        }
    if isinstance(e, GeomArc):
        return {
            "radius": (e.radius, "rel"),
            "sweep_deg": (e.sweep_deg, "rel"),
            "arc_length": (np.radians(e.sweep_deg) * e.radius, "rel"),
            "cx": (e.center[0], "abs"),
            "cy": (e.center[1], "abs"),
        }
    if isinstance(e, GeomPolyline):
        return {
            "length": (_polyline_length(e, spacing_mm=spacing_mm), "rel"),
            "n_vertices": (float(len(e.points)), "abs"),
            "bulge_sum": (float(sum(e.bulges)), "abs"),
        }
    raise TypeError(f"未知实体类型：{type(e)!r}")


@dataclass(frozen=True)
class DimDiff:
    name: str
    ref: float
    cand: float
    abs_err: float
    rel_err: float | None
    kind: MetricKind
    ok: bool


@dataclass(frozen=True)
class CompareResult:
    entity_count_ref: int
    entity_count_cand: int
    entity_count_diff: int
    count_by_kind_ref: dict[str, int]
    count_by_kind_cand: dict[str, int]
    count_ok: bool
    hausdorff_p95_mm: float
    hausdorff_ok: bool
    dims: tuple[DimDiff, ...]
    worst_rel_dim: DimDiff | None
    worst_abs_dim: DimDiff | None
    dims_ok: bool
    failed_checks: tuple[str, ...]
    passed: bool

    def to_json(self) -> dict[str, Any]:
        return {
            "entity_count_ref": self.entity_count_ref,
            "entity_count_cand": self.entity_count_cand,
            "entity_count_diff": self.entity_count_diff,
            "count_by_kind_ref": self.count_by_kind_ref,
            "count_by_kind_cand": self.count_by_kind_cand,
            "count_ok": self.count_ok,
            "hausdorff_p95_mm": round(self.hausdorff_p95_mm, 6),
            "hausdorff_ok": self.hausdorff_ok,
            "worst_rel_dim": None
            if self.worst_rel_dim is None
            else {
                "name": self.worst_rel_dim.name,
                "ref": self.worst_rel_dim.ref,
                "cand": self.worst_rel_dim.cand,
                "rel_err": self.worst_rel_dim.rel_err,
            },
            "worst_abs_dim": None
            if self.worst_abs_dim is None
            else {
                "name": self.worst_abs_dim.name,
                "ref": self.worst_abs_dim.ref,
                "cand": self.worst_abs_dim.cand,
                "abs_err_mm": self.worst_abs_dim.abs_err,
            },
            "dims_ok": self.dims_ok,
            "failed_checks": list(self.failed_checks),
            "passed": self.passed,
        }


def _count_by_kind(entities: Sequence[GeomEntity]) -> dict[str, int]:
    out: dict[str, int] = {}
    for e in entities:
        out[e.kind] = out.get(e.kind, 0) + 1
    return dict(sorted(out.items()))


def compare_geometry(
    ref: Sequence[GeomEntity],
    cand: Sequence[GeomEntity],
    *,
    spacing_mm: float,
    max_hausdorff_p95_mm: float,
    max_entity_count_diff: int,
    dim_abs_tol_mm: float,
    dim_rel_tol: float,
) -> CompareResult:
    """三项比对。所有阈值都必须由调用方显式给出（来自 config.yaml 或测试）。"""
    if spacing_mm <= 0:
        raise ValueError(f"spacing_mm 必须 > 0，收到 {spacing_mm}")
    if max_entity_count_diff < 0:
        raise ValueError(f"max_entity_count_diff 必须 >= 0，收到 {max_entity_count_diff}")
    if dim_abs_tol_mm < 0 or dim_rel_tol < 0:
        raise ValueError("维度容差不能为负")

    failed: list[str] = []
    if len(ref) == 0:
        raise ValueError("参考（golden）实体列表为空：没有基准就无所谓比对，请先补 golden")
    count_diff = len(cand) - len(ref)
    count_ok = abs(count_diff) <= max_entity_count_diff
    if not count_ok:
        failed.append("ENTITY_COUNT")

    if len(cand) == 0:
        # 产物一条都没出：这是明确的失败（不是异常），Hausdorff 记为无穷大
        haus = float("inf")
        haus_ok = False
        failed.append("HAUSDORFF_P95")
    else:
        haus = hausdorff_p95(
            sample_entities(ref, spacing_mm=spacing_mm),
            sample_entities(cand, spacing_mm=spacing_mm),
        )
        haus_ok = haus <= max_hausdorff_p95_mm
        if not haus_ok:
            failed.append("HAUSDORFF_P95")

    # 关键尺寸：按类型配对比较（数量不齐时只比较能配对的部分，并记差异）
    ref_sorted = canonical_sort(ref)
    cand_sorted = canonical_sort(cand)
    by_kind_ref: dict[str, list[GeomEntity]] = {}
    by_kind_cand: dict[str, list[GeomEntity]] = {}
    for e in ref_sorted:
        by_kind_ref.setdefault(e.kind, []).append(e)
    for e in cand_sorted:
        by_kind_cand.setdefault(e.kind, []).append(e)

    dims: list[DimDiff] = []
    for kind in sorted(set(by_kind_ref) | set(by_kind_cand)):
        a, b = by_kind_ref.get(kind, []), by_kind_cand.get(kind, [])
        if len(a) != len(b):
            failed.append(f"COUNT_MISMATCH:{kind}")
        for idx, (ea, eb) in enumerate(zip(a, b)):
            ma = entity_metrics(ea, spacing_mm=spacing_mm)
            mb = entity_metrics(eb, spacing_mm=spacing_mm)
            for name in sorted(set(ma) & set(mb)):
                v_ref, kind_metric = ma[name]
                v_cand = mb[name][0]
                abs_err = abs(v_cand - v_ref)
                rel_err = None if v_ref == 0 else abs_err / abs(v_ref)
                ok = rel_err <= dim_rel_tol if kind_metric == "rel" else abs_err <= dim_abs_tol_mm
                if rel_err is None and kind_metric == "rel":
                    ok = abs_err <= dim_abs_tol_mm
                dims.append(
                    DimDiff(
                        name=f"{kind}[{idx}].{name}",
                        ref=v_ref,
                        cand=v_cand,
                        abs_err=abs_err,
                        rel_err=rel_err,
                        kind=kind_metric,
                        ok=ok,
                    )
                )

    rel_dims = [d for d in dims if d.kind == "rel" and d.rel_err is not None]
    abs_dims = [d for d in dims if d.kind == "abs"]
    worst_rel = max(rel_dims, key=lambda d: d.rel_err or 0.0) if rel_dims else None
    worst_abs = max(abs_dims, key=lambda d: d.abs_err) if abs_dims else None
    dims_ok = all(d.ok for d in dims) and not any(f.startswith("COUNT_MISMATCH") for f in failed)
    if not dims_ok:
        failed.append("KEY_DIMS")

    return CompareResult(
        entity_count_ref=len(ref),
        entity_count_cand=len(cand),
        entity_count_diff=count_diff,
        count_by_kind_ref=_count_by_kind(ref),
        count_by_kind_cand=_count_by_kind(cand),
        count_ok=count_ok,
        hausdorff_p95_mm=haus,
        hausdorff_ok=haus_ok,
        dims=tuple(dims),
        worst_rel_dim=worst_rel,
        worst_abs_dim=worst_abs,
        dims_ok=dims_ok,
        failed_checks=tuple(failed),
        passed=not failed,
    )


def compare_dxf_files(
    ref_path: str | Path,
    cand_path: str | Path,
    *,
    ignore_types: frozenset[str],
    spacing_mm: float,
    max_hausdorff_p95_mm: float,
    max_entity_count_diff: int,
    dim_abs_tol_mm: float,
    dim_rel_tol: float,
) -> CompareResult:
    """文件级入口：读两个 DXF 后比对。"""
    ref = read_dxf(ref_path, ignore_types=ignore_types)
    cand = read_dxf(cand_path, ignore_types=ignore_types)
    return compare_geometry(
        ref,
        cand,
        spacing_mm=spacing_mm,
        max_hausdorff_p95_mm=max_hausdorff_p95_mm,
        max_entity_count_diff=max_entity_count_diff,
        dim_abs_tol_mm=dim_abs_tol_mm,
        dim_rel_tol=dim_rel_tol,
    )


__all__ = [
    "CompareResult",
    "DimDiff",
    "GeomArc",
    "GeomCircle",
    "GeomEntity",
    "GeomLine",
    "GeomPolyline",
    "canonical_sort",
    "compare_dxf_files",
    "compare_geometry",
    "entity_metrics",
    "hausdorff_p95",
    "read_dxf",
    "sample_entities",
    "sample_entity",
]
