# src/s8_dxf.py
"""S8 DXF 组装：把 S4 基元 + S5 图层 + S7 变换写成 DXF，并回读自检（SPEC §4 S8）。

输入契约
--------
- 几何：`FitResult.primitives`（图像连续像素坐标，Y 向下，见 `docs/coordinate_system.md`）
- 图层：`LayerResult.assignments`（每个基元一条；图框是 `BORDER`）
- 拓扑：`TopologyGraph.edges[*].closed_loop` —— 闭合环（图框/窗洞）在 DXF 里必须写成
  **单条 `closed=True` 的 LWPOLYLINE**，拆成 4 条 LINE 会丢失闭合语义，下游 CAM 认不出封闭轮廓。
- 尺度：`RegistrationResult`（未配准时是 identity，只做纸面 mm 换算）

输出
----
- `outputs/<run>/drawing.dxf`（比特级可复现，见 `src/dxf_repro.py`）
- `metrics_s8.json`：实体数、图层分布、回读自检结果、待 OCR 区域数、WARN

未配准水印（用户决策 #4）
------------------------
`registration.is_identity` 为真时：
① `HEADER.$PROJECTNAME = "R2V-UNCALIBRATED"`；
② `metrics_s8.warnings` 追加"图纸未配准，尺寸不可信"。
不写 `$COMMENT`（部分 CAD 不显示）、不加 TEXT 实体（不污染图纸内容）。
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict

from src.artifacts import StageMetrics, write_json, write_metrics
from src.config import R2VConfig
from src.dxf_writer import DXFWriter, LayerConfig
from src.s3_skeleton import TopologyGraph
from src.s4_fit import FitResult, Primitive
from src.s5_layer import LayerResult
from src.s7_register import RegistrationResult

#: 未配准水印（用户决策 #4）
UNCALIBRATED_PROJECT_NAME = "R2V-UNCALIBRATED"
UNCALIBRATED_WARNING = "图纸未配准，尺寸不可信（HEADER.$PROJECTNAME=R2V-UNCALIBRATED）"

#: 弧离散成多段线时每 90° 的采样数（仅用于闭合环拼接与回退）
_ARC_SEGMENTS_PER_90 = 8


class DxfAssemblyError(ValueError):
    """S8 输入非法（缺图层、缺几何、矩阵退化等）。"""


class SelfCheckReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    entities_expected: int
    entities_written: int
    entities_read_back: int
    max_coord_delta_mm: float
    layers_read_back: tuple[str, ...]
    closed_polylines: int
    text_written: int = 0
    text_skipped_low_confidence: int = 0
    passed: bool
    problems: tuple[str, ...]


class S8Result(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    dxf_path: str
    entities: dict[str, int]
    layers: dict[str, int]
    self_check: SelfCheckReport
    uncalibrated: bool
    pending_ocr_regions: int
    warnings: tuple[str, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "dxf_path": self.dxf_path,
            "entities": dict(sorted(self.entities.items())),
            "layers": dict(sorted(self.layers.items())),
            "self_check": self.self_check.model_dump(mode="json"),
            "uncalibrated": self.uncalibrated,
            "pending_ocr_regions": self.pending_ocr_regions,
            "warnings": list(self.warnings),
        }


def _scale_of(matrix: np.ndarray) -> float:
    a, b = matrix[0, 0], matrix[0, 1]
    c, d = matrix[1, 0], matrix[1, 1]
    det = abs(a * d - b * c)
    if det < 1e-18:
        raise DxfAssemblyError(f"变换矩阵退化（|det|={det:.3e}），无法确定像素尺度")
    return float(math.sqrt(det))


def build_world_matrix(
    registration: RegistrationResult,
    *,
    px_to_mm: float,
    height_px: int,
    points_px: np.ndarray,
) -> tuple[np.ndarray, float]:
    """返回 (mm_img→CAD 世界矩阵, 输出用 mm/px)。

    `DXFWriter._to_cad` 先做 Y 翻转再乘世界矩阵，所以这里要乘回一个"mm_img→px"的矩阵，
    保证最终结果恰好等于 `M_px @ px`。
    """
    if registration.is_identity:
        # 未配准：纸面坐标，无需世界矩阵（DXFWriter 的翻转已经给出左下角原点、Y 向上）
        return np.eye(3, dtype=np.float64), px_to_mm

    scale = _scale_of(np.asarray(registration.matrix) @ np.diag([px_to_mm, px_to_mm, 1.0]))
    m_px_cad = np.asarray(registration.matrix, dtype=np.float64).copy()
    # 平移到非负象限（DXF 原点习惯放在图幅左下角）
    if points_px.size:
        homogeneous = np.column_stack([points_px, np.ones(len(points_px))])
        world = (homogeneous @ m_px_cad.T)[:, :2]
        shift = world.min(axis=0)
        m_px_cad[0, 2] -= float(shift[0])
        m_px_cad[1, 2] -= float(shift[1])
    # mm_img(已翻转) → px 的矩阵：x/s、(h_px − y/s)
    mm_per_px = scale
    p = np.array(
        [[1.0 / mm_per_px, 0.0, 0.0],
         [0.0, -1.0 / mm_per_px, float(height_px)],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return m_px_cad @ p, mm_per_px


def _arc_points(center: tuple[float, float], radius: float, start_deg: float, sweep_deg: float) -> list[tuple[float, float]]:
    count = max(4, int(abs(sweep_deg) / 90.0 * _ARC_SEGMENTS_PER_90) + 1)
    angles = np.radians(np.linspace(start_deg, start_deg + sweep_deg, count))
    return [(center[0] + radius * math.cos(a), center[1] + radius * math.sin(a)) for a in angles]


def _primitive_points(primitive: Primitive) -> list[tuple[float, float]]:
    if primitive.kind == "line":
        if primitive.start is None or primitive.end is None:
            raise DxfAssemblyError(f"LINE 基元 {primitive.id} 缺少端点")
        return [primitive.start, primitive.end]
    if primitive.kind == "arc":
        if primitive.center is None or primitive.radius_px is None:
            raise DxfAssemblyError(f"ARC 基元 {primitive.id} 缺少圆心/半径")
        sweep = primitive.sweep_deg or 0.0
        if primitive.start_angle_deg is None or abs(sweep) < 1e-9:
            raise DxfAssemblyError(f"ARC 基元 {primitive.id} 扫角为 0，无法写出")
        return _arc_points(primitive.center, primitive.radius_px, primitive.start_angle_deg, sweep)
    if primitive.points:
        return list(primitive.points)
    if primitive.start and primitive.end:
        return [primitive.start, primitive.end]
    raise DxfAssemblyError(f"基元 {primitive.id}({primitive.kind}) 没有可用几何")


def _angle_deg(point: tuple[float, float], center: tuple[float, float]) -> float:
    return math.degrees(math.atan2(point[1] - center[1], point[0] - center[0])) % 360.0


def _arc_point(primitive: Primitive, *, at_start: bool | None) -> tuple[float, float]:
    """取弧的起点 / 终点 / 中点（图像像素系）。"""
    assert primitive.center is not None and primitive.radius_px is not None
    center = primitive.center
    radius = primitive.radius_px
    start_deg = primitive.start_angle_deg or 0.0
    sweep = primitive.sweep_deg or 0.0
    if at_start is None:
        angle = math.radians(start_deg + sweep / 2.0)
    elif at_start:
        angle = math.radians(start_deg)
    else:
        angle = math.radians(start_deg + sweep)
    return (center[0] + radius * math.cos(angle), center[1] + radius * math.sin(angle))


def _to_cad_points(
    points: Sequence[tuple[float, float]], mm_per_px: float, writer: DXFWriter
) -> list[tuple[float, float]]:
    """把图像像素点换算成**最终 CAD 坐标**（走 writer 的翻转 + 世界矩阵）。"""
    array = np.asarray(points, dtype=np.float64) * mm_per_px
    cad = writer._to_cad(array)  # noqa: SLF001 - 与写出端共用同一套变换，避免两套口径
    return [(float(row[0]), float(row[1])) for row in cad]


def _chain_closed_loop(primitives: Sequence[Primitive], *, tolerance_px: float) -> list[tuple[float, float]]:
    """把同一条闭合环上的若干基元按首尾相接拼成闭合折线。"""
    ordered = sorted(primitives, key=lambda p: p.id)
    points: list[tuple[float, float]] = []
    for primitive in ordered:
        piece = _primitive_points(primitive)
        if not points:
            points.extend(piece)
            continue
        head = piece[0]
        tail = points[-1]
        if math.hypot(head[0] - tail[0], head[1] - tail[1]) > tolerance_px:
            # 顺序对不上：尝试反转
            if math.hypot(piece[-1][0] - tail[0], piece[-1][1] - tail[1]) <= tolerance_px:
                piece = list(reversed(piece))
            else:
                raise DxfAssemblyError(
                    f"闭合环拼接失败：基元 {primitive.id} 与已拼片段端点相距 "
                    f"{math.hypot(head[0] - tail[0], head[1] - tail[1]):.3f}px > {tolerance_px}px"
                )
        points.extend(piece[1:])
    if len(points) >= 3:
        first, last = points[0], points[-1]
        if math.hypot(first[0] - last[0], first[1] - last[1]) <= tolerance_px:
            points = points[:-1]
    return points


def run_s8_dxf(
    fit: FitResult,
    layers: LayerResult,
    graph: TopologyGraph,
    registration: RegistrationResult,
    cfg: R2VConfig,
    *,
    dpi: float,
    image_size_px: tuple[int, int],
    out_dir: str | Path | None = None,
    file_name: str = "drawing.dxf",
    pending_ocr_regions: Sequence[Sequence[float]] = (),
    text_items: Sequence[dict[str, Any]] = (),
) -> tuple[S8Result, StageMetrics]:
    """S8 主流程：几何 + 图层 + 尺度 → DXF（含回读自检）。"""
    t0 = time.perf_counter()
    if not fit.primitives:
        raise DxfAssemblyError("S8 没有可写出的基元")
    width_px, height_px = image_size_px
    if width_px <= 0 or height_px <= 0:
        raise DxfAssemblyError(f"image_size_px 非法：{image_size_px}")
    if dpi <= 0:
        raise DxfAssemblyError(f"dpi 必须 > 0，收到 {dpi}")

    assignment_by_id = {a.primitive_id: a for a in layers.assignments}
    missing = [p.id for p in fit.primitives if p.id not in assignment_by_id]
    if missing:
        raise DxfAssemblyError(f"以下基元没有图层分配（S5 与 S4 不一致）：{missing[:8]}")

    px_to_mm = 25.4 / dpi
    all_points = np.asarray(
        [
            point
            for primitive in fit.primitives
            for point in (_primitive_points(primitive)[:1] + _primitive_points(primitive)[-1:])
        ],
        dtype=np.float64,
    )
    world_matrix, mm_per_px = build_world_matrix(
        registration, px_to_mm=px_to_mm, height_px=height_px, points_px=all_points
    )
    image_height_mm = height_px * px_to_mm

    layer_names = tuple(layer.name for layer in cfg.dxf.layers)
    # LayerConfig 要求 len(names)=len(bins)+1；S8 的图层来自 S5 显式分配，
    # 不靠宽度归档，所以只给它一个占位映射，其余图层构造后补建。
    writer = DXFWriter(
        version=cfg.dxf.version,
        layer_config=LayerConfig(bins=(), names=(layer_names[0],)),
        image_height_mm=image_height_mm,
        transform_mm_img_to_world=world_matrix,
    )
    for name in layer_names:
        if name not in writer.doc.layers:
            writer.doc.layers.add(name)
    warnings: list[str] = list(writer.warnings)
    if registration.is_identity:
        writer.doc.header["$PROJECTNAME"] = UNCALIBRATED_PROJECT_NAME
        warnings.append(UNCALIBRATED_WARNING)

    closed_loop_edges = {
        edge.id: edge for edge in graph.edges if edge.closed_loop
    }
    consumed: set[int] = set()
    closed_polylines = 0
    entity_counts: dict[str, int] = {}
    layer_counts: dict[str, int] = {}

    def record(kind: str, layer: str) -> None:
        entity_counts[kind] = entity_counts.get(kind, 0) + 1
        layer_counts[layer] = layer_counts.get(layer, 0) + 1

    # 1) 闭合环优先：整条边写成一条 closed=True 的 LWPOLYLINE（圆环仍写 CIRCLE）
    for edge_id, edge in sorted(closed_loop_edges.items()):
        members = [p for p in fit.primitives if edge_id in p.source_edge_ids]
        if not members:
            continue
        circles = [p for p in members if p.kind == "circle"]
        assignment = assignment_by_id[members[0].id]
        layer = assignment.layer
        if circles:
            circle = circles[0]
            assert circle.center is not None and circle.radius_px is not None
            writer.add_circle(
                (circle.center[0] * mm_per_px, circle.center[1] * mm_per_px),
                circle.radius_px * mm_per_px, layer=layer, linewidth_mm=assignment.width_mm,
            )
            record("CIRCLE", layer)
            consumed.update(p.id for p in members)
            continue
        try:
            chain = _chain_closed_loop(members, tolerance_px=max(2.0, 0.5 * edge.linewidth_px))
        except DxfAssemblyError as exc:
            warnings.append(f"边 {edge_id} 闭合环拼接失败，退回各自独立写出：{exc}")
            continue
        if len(chain) < 3:
            warnings.append(f"边 {edge_id} 闭合环点数 < 3，跳过")
            continue
        writer.add_lwpolyline(
            [(x * mm_per_px, y * mm_per_px) for x, y in chain],
            layer=layer, closed=True, linewidth_mm=assignment.width_mm,
        )
        record("LWPOLYLINE", layer)
        closed_polylines += 1
        consumed.update(p.id for p in members)

    # 2) 其余基元逐条写出
    for primitive in fit.primitives:
        if primitive.id in consumed:
            continue
        assignment = assignment_by_id[primitive.id]
        layer = assignment.layer
        width = assignment.width_mm
        if primitive.kind == "line":
            assert primitive.start is not None and primitive.end is not None
            writer.add_line(
                (primitive.start[0] * mm_per_px, primitive.start[1] * mm_per_px),
                (primitive.end[0] * mm_per_px, primitive.end[1] * mm_per_px),
                layer=layer, linewidth_mm=width,
            )
            record("LINE", layer)
        elif primitive.kind == "arc":
            assert primitive.center is not None and primitive.radius_px is not None
            # 角度必须先在 **CAD 系**里定出来，再换算成 DXFWriter 期望的"图像系原始角"。
            # DXFWriter 会做 sa=-ea_raw、ea=-sa_raw 的镜像；直接传图像系角度时，
            # 弧的方向约定（图像 Y 向下时 CCW 与 CAD 相反）会让扫角落到错误的象限
            # （实测 30°–150° 的弧被写成 278°–33°，Hausdorff 冲到 292px）。
            center_px = primitive.center
            start_pt = _arc_point(primitive, at_start=True)
            end_pt = _arc_point(primitive, at_start=False)
            mid_pt = _arc_point(primitive, at_start=None)
            cad = _to_cad_points([center_px, start_pt, end_pt, mid_pt], mm_per_px, writer)
            a_start = _angle_deg(cad[1], cad[0])
            a_end = _angle_deg(cad[2], cad[0])
            a_mid = _angle_deg(cad[3], cad[0])
            # 选包含中点的那个 CCW 区间（DXF 的 ARC 只能逆时针）
            if (a_mid - a_start) % 360.0 <= (a_end - a_start) % 360.0:
                ccw_start, ccw_end = a_start, a_end
            else:
                ccw_start, ccw_end = a_end, a_start
            writer.add_arc(
                (center_px[0] * mm_per_px, center_px[1] * mm_per_px),
                primitive.radius_px * mm_per_px,
                (-ccw_end) % 360.0 if writer._reflected else ccw_start,  # noqa: SLF001
                (-ccw_start) % 360.0 if writer._reflected else ccw_end,  # noqa: SLF001
                layer=layer, linewidth_mm=width,
            )
            record("ARC", layer)
        elif primitive.kind == "circle":
            assert primitive.center is not None and primitive.radius_px is not None
            writer.add_circle(
                (primitive.center[0] * mm_per_px, primitive.center[1] * mm_per_px),
                primitive.radius_px * mm_per_px, layer=layer, linewidth_mm=width,
            )
            record("CIRCLE", layer)
        else:  # lwpolyline / 回退
            points = _primitive_points(primitive)
            writer.add_lwpolyline(
                [(x * mm_per_px, y * mm_per_px) for x, y in points],
                layer=layer, closed=False, linewidth_mm=width,
            )
            record("LWPOLYLINE", layer)

    out = Path(out_dir) if out_dir is not None else Path.cwd()
    out.mkdir(parents=True, exist_ok=True)
    # 3) TEXT：置信度达标的写实体，占位条目只写掩膜（灰色虚线框）
    confidence_min = float(cfg.text.ocr.text_confidence_min)
    written_texts = 0
    skipped_texts = 0
    placeholder_boxes: list[tuple[float, float, float, float]] = []
    for item in text_items:
        confidence = float(item.get("confidence", 0.0))
        is_placeholder = bool(item.get("is_placeholder", True))
        insertion = item.get("insertion_point") or [0.0, 0.0]
        x_mm, y_mm = float(insertion[0]), float(insertion[1])
        height_mm = float(item.get("height_mm", 0.0))
        text_value = str(item.get("text", ""))
        if is_placeholder or confidence < confidence_min or not text_value:
            skipped_texts += 1
            width_mm = height_mm * max(1.0, float(len(text_value) or 1)) * 0.6
            placeholder_boxes.append((x_mm / mm_per_px, y_mm / mm_per_px,
                                      width_mm / mm_per_px, height_mm / mm_per_px))
            continue
        writer.add_text(
            text_value,
            (x_mm / mm_per_px, y_mm / mm_per_px),
            height_mm,
            layer="TEXT",
            xdata={
                "line_id": item.get("line_id", ""),
                "engine": item.get("engine", "none"),
                "confidence": f"{confidence:.6f}",
            },
        )
        record("TEXT", "TEXT")
        written_texts += 1

    dxf_path = out / file_name
    report = writer.save(
        dxf_path,
        rms_mm=None if registration.is_identity else float(registration.rms_mm),
        project_name=UNCALIBRATED_PROJECT_NAME if registration.is_identity else None,
        write_metrics=False,   # metrics_s8.json 由本模块以 StageMetrics 格式写
    )
    warnings.extend(writer.warnings)
    problems = tuple(str(item) for item in report.get("mismatches", []))

    self_check = SelfCheckReport(
        entities_expected=int(report.get("expected_total", len(writer._written))),  # noqa: SLF001
        entities_written=int(report.get("total", 0)),
        entities_read_back=int(report.get("checked_entities", 0)),
        max_coord_delta_mm=float(report.get("max_geometry_deviation_mm", 0.0)),
        layers_read_back=tuple(sorted({entity.dxf.layer for entity in writer.msp})),
        closed_polylines=closed_polylines,
        passed=bool(report.get("ok", False)),
        problems=problems,
        text_written=written_texts,
        text_skipped_low_confidence=skipped_texts,
    )

    result = S8Result(
        dxf_path=str(dxf_path),
        entities=dict(sorted(entity_counts.items())),
        layers=dict(sorted(layer_counts.items())),
        self_check=self_check,
        uncalibrated=bool(registration.is_identity),
        pending_ocr_regions=len(pending_ocr_regions),
        warnings=tuple(warnings),
    )

    timing_ms = round((time.perf_counter() - t0) * 1000.0, 3)
    metrics = StageMetrics(
        stage="s8",
        counts={
            "entities": sum(entity_counts.values()),
            "lines": entity_counts.get("LINE", 0),
            "arcs": entity_counts.get("ARC", 0),
            "circles": entity_counts.get("CIRCLE", 0),
            "polylines": entity_counts.get("LWPOLYLINE", 0),
            "closed_polylines": closed_polylines,
            "text_written": written_texts,
            "text_skipped_low_confidence": skipped_texts,
            "pending_ocr_regions": len(pending_ocr_regions),
            "warnings": len(warnings),
        },
        timing_ms=timing_ms,
        error_estimate={
            "max_coord_delta_mm": self_check.max_coord_delta_mm,
            "mm_per_px": round(mm_per_px, 12),
            "image_height_mm": round(image_height_mm, 6),
        },
        warnings=tuple(warnings),
        details={
            "dxf_path": str(dxf_path),
            "layers": result.layers,
            "uncalibrated": result.uncalibrated,
            "project_name": writer.doc.header.get("$PROJECTNAME", ""),
            "pending_ocr_bboxes": [list(map(float, box)) for box in pending_ocr_regions],
            "self_check_passed": self_check.passed,
            "text_confidence_min": confidence_min,
        },
    )

    if out_dir is not None:
        write_json(out / "s8_report.json", result.to_json())
        write_metrics(out, metrics)
        if placeholder_boxes:
            _write_debug_overlay(out / "debug" / "overlay_s8.png", width_px, height_px, placeholder_boxes)

    return result, metrics


def _write_debug_overlay(
    path: Path, width_px: int, height_px: int, boxes: Sequence[tuple[float, float, float, float]]
) -> None:
    """被跳过的文字条目：灰色虚线框标注位置（便于人工排查低置信度原因）。"""
    import cv2

    canvas = np.full((height_px, width_px, 3), 255, np.uint8)
    for x, y, w, h in boxes:
        p1 = (int(round(x)), int(round(y)))
        p2 = (int(round(x + w)), int(round(y + h)))
        # 虚线：按 6px 画、4px 空
        for offset in range(0, max(1, p2[0] - p1[0]), 10):
            cv2.line(canvas, (p1[0] + offset, p1[1]), (min(p1[0] + offset + 6, p2[0]), p1[1]), (128, 128, 128), 1)
            cv2.line(canvas, (p1[0] + offset, p2[1]), (min(p1[0] + offset + 6, p2[0]), p2[1]), (128, 128, 128), 1)
        for offset in range(0, max(1, p2[1] - p1[1]), 10):
            cv2.line(canvas, (p1[0], p1[1] + offset), (p1[0], min(p1[1] + offset + 6, p2[1])), (128, 128, 128), 1)
            cv2.line(canvas, (p2[0], p1[1] + offset), (p2[0], min(p1[1] + offset + 6, p2[1])), (128, 128, 128), 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), canvas)


__all__ = [
    "DxfAssemblyError",
    "S8Result",
    "SelfCheckReport",
    "UNCALIBRATED_PROJECT_NAME",
    "UNCALIBRATED_WARNING",
    "build_world_matrix",
    "run_s8_dxf",
]
