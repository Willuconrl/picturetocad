# src/s6_text.py
"""S6 文字分离（第一轮）：只做**文字候选区提取**，不接任何 OCR 引擎。

思路
----
已识别的几何基元覆盖的像素一定不是文字。把基元按"膨胀半径 = 线宽/2 + 2px"画成掩膜，
从二值图里扣掉，剩下的连通域再按形状筛选，就是文字候选区。

筛选条件（参数全部来自 `config.yaml` 的 `text.separation`）
-----------------------------------------------------------
- 面积 ≥ `min_area_px`（噪点）
- 面积 ≤ `area_max_mm2`（大块是图形不是文字）
- 宽高比 ≤ `aspect_ratio_max`（长条是线条残留）
- 与基元膨胀掩膜不重叠（扣完自然满足）

输出
----
- `pending_ocr_regions.json`：每个候选区的 bbox / 面积 / 宽高比 / 到最近基元的距离，
  供 S8 的 `pending_ocr_regions` 与下一轮 OCR 消费；
- `debug/overlay_s6.png`：原图灰底 + 红色半透明已识别基元 + 绿色候选框 + 蓝色被排除连通域。

**不写任何 TEXT/MTEXT**：本轮没有识别，编造文字比缺字更危险。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict

from src.artifacts import StageMetrics, write_json, write_metrics
from src.config import R2VConfig
from src.s4_fit import FitResult, Primitive


class S6Error(ValueError):
    """S6 输入非法。"""


class TextCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: int
    bbox: tuple[int, int, int, int]          # x, y, w, h（整数像素，左上角）
    area_px: int
    aspect_ratio: float
    distance_to_nearest_primitive_px: float
    height_mm: float


class ExcludedRegion(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    bbox: tuple[int, int, int, int]
    area_px: int
    reason: str


class S6Result(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    candidates: tuple[TextCandidate, ...]
    excluded: tuple[ExcludedRegion, ...]
    debug_path: str | None
    counts: dict[str, int]
    warnings: tuple[str, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "candidates": [c.model_dump(mode="json") for c in self.candidates],
            "excluded": [e.model_dump(mode="json") for e in self.excluded],
            "counts": dict(sorted(self.counts.items())),
            "debug_path": self.debug_path,
            "warnings": list(self.warnings),
        }


# ---------------------------------------------------------------------------
# 基元掩膜
# ---------------------------------------------------------------------------
def _arc_points(center: tuple[float, float], radius: float, start_deg: float, sweep_deg: float) -> np.ndarray:
    count = max(8, int(abs(sweep_deg) / 6.0) + 1)
    angles = np.radians(np.linspace(start_deg, start_deg + sweep_deg, count))
    return np.column_stack([
        center[0] + radius * np.cos(angles),
        center[1] + radius * np.sin(angles),
    ])


def _primitive_polyline(primitive: Primitive) -> np.ndarray | None:
    if primitive.kind == "line" and primitive.start and primitive.end:
        return np.asarray([primitive.start, primitive.end], dtype=np.float64)
    if primitive.kind == "arc" and primitive.center and primitive.radius_px is not None \
            and primitive.start_angle_deg is not None:
        return _arc_points(
            primitive.center, primitive.radius_px,
            primitive.start_angle_deg, primitive.sweep_deg or 0.0,
        )
    if primitive.kind == "circle" and primitive.center and primitive.radius_px is not None:
        return _arc_points(primitive.center, primitive.radius_px, 0.0, 360.0)
    if primitive.points:
        return np.asarray(primitive.points, dtype=np.float64)
    if primitive.start and primitive.end:
        return np.asarray([primitive.start, primitive.end], dtype=np.float64)
    return None


def rasterize_primitives(
    shape: tuple[int, int], primitives: Sequence[Primitive], *, thickness_px: int
) -> np.ndarray:
    """把基元画成掩膜（用于从二值图里扣掉已识别几何）。"""
    height, width = shape
    mask = np.zeros((height, width), dtype=np.uint8)
    thickness = max(1, int(round(thickness_px)))
    for primitive in primitives:
        points = _primitive_polyline(primitive)
        if points is None or len(points) < 2:
            continue
        # 裁到图内再取整（cv2 对超界坐标会静默截断）
        clipped = np.clip(np.round(points), [-1e4, -1e4], [1e4, 1e4]).astype(np.int32)
        cv2.polylines(mask, [clipped], False, 255, thickness, lineType=cv2.LINE_8)
        if primitive.kind == "circle":
            centre = tuple(int(round(v)) for v in primitive.center)  # type: ignore[arg-type]
            cv2.circle(mask, centre, int(round(primitive.radius_px or 0)), 255, thickness)
    return mask


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def structural_primitives(
    primitives: Sequence[Primitive], *, linewidth_px: float, min_height_mm: float, mm_per_px: float
) -> list[Primitive]:
    """挑出"结构性"基元（长笔画），用于从二值图里扣掉已识别几何。

    为什么不能把所有基元都扣掉：**字符笔画本身也会被 S3/S4 当成短基元拟合**
    （实测三个字母在二值图里几乎全被基元覆盖，候选区只剩 1–2 px 碎片）。
    真正的"图线"是长笔画，用长度阈值把它们区分开：
    `max(10×线宽, 4×最小字高)` —— 短于此的基元当作"可能是字符笔画"，不参与扣除。
    根治办法是 S6 在 S4 之前做文字/线条分离（M8 范围），本轮用这条工程折中。
    """
    threshold = max(10.0 * linewidth_px, 4.0 * float(min_height_mm) / max(mm_per_px, 1e-9))
    structural: list[Primitive] = []
    for primitive in primitives:
        if primitive.kind in ("arc", "circle") and primitive.radius_px is not None:
            structural.append(primitive)
            continue
        if primitive.length_px >= threshold:
            structural.append(primitive)
    return structural


def run_s6_text(
    binary: np.ndarray,
    fit: FitResult,
    cfg: R2VConfig,
    *,
    dpi: float,
    out_dir: str | Path | None = None,
    gray: np.ndarray | None = None,
    linewidth_px: float | None = None,
) -> tuple[S6Result, StageMetrics]:
    """文字候选区提取（不接 OCR）。"""
    t0 = time.perf_counter()
    text_cfg = cfg.text
    if not text_cfg.enable:
        raise S6Error("config.text.enable=false：S6 未启用")
    mask = np.asarray(binary)
    if mask.ndim != 2:
        raise S6Error(f"S6 需要二维二值图，收到形状 {mask.shape}")
    if dpi <= 0:
        raise S6Error(f"dpi 必须 > 0，收到 {dpi}")
    separation = text_cfg.separation
    mm_per_px = 25.4 / dpi
    linewidth = float(linewidth_px or 3.0)
    dilate_px = linewidth * float(separation.dilate_linewidth_factor) + float(separation.dilate_extra_px)

    primitive_mask = rasterize_primitives(
        mask.shape,
        structural_primitives(
            fit.primitives, linewidth_px=linewidth,
            min_height_mm=float(text_cfg.min_height_mm), mm_per_px=mm_per_px,
        ),
        thickness_px=linewidth + 2.0 * dilate_px,
    )
    free = ((mask > 0) & (primitive_mask == 0)).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(free, connectivity=8)

    # 到最近基元的距离（用未膨胀掩膜的倒距离场）
    thin_mask = rasterize_primitives(
        mask.shape,
        structural_primitives(
            fit.primitives, linewidth_px=linewidth,
            min_height_mm=float(text_cfg.min_height_mm), mm_per_px=mm_per_px,
        ),
        thickness_px=max(1, int(round(linewidth))),
    )
    distance_to_primitive = cv2.distanceTransform((thin_mask == 0).astype(np.uint8), cv2.DIST_L2, 5)

    candidates: list[TextCandidate] = []
    excluded: list[ExcludedRegion] = []
    warnings: list[str] = []
    for index in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[index])
        if area <= 0:
            continue
        bbox = (x, y, w, h)
        aspect = float(max(w, h) / max(1, min(w, h)))
        height_mm = h * mm_per_px
        if area < int(separation.min_area_px):
            excluded.append(ExcludedRegion(bbox=bbox, area_px=area, reason="area_too_small"))
            continue
        if aspect > float(separation.aspect_ratio_max):
            excluded.append(ExcludedRegion(bbox=bbox, area_px=area, reason="aspect_too_long"))
            continue
        area_mm2 = area * mm_per_px * mm_per_px
        if area_mm2 > float(separation.area_max_mm2):
            excluded.append(ExcludedRegion(bbox=bbox, area_px=area, reason="area_too_large_mm2"))
            continue
        if height_mm < float(text_cfg.min_height_mm) or height_mm > float(text_cfg.max_height_mm):
            excluded.append(ExcludedRegion(bbox=bbox, area_px=area, reason="height_out_of_range"))
            continue
        cx, cy = centroids[index]
        distance = float(distance_to_primitive[int(round(cy)), int(round(cx))]) if (
            0 <= int(round(cy)) < mask.shape[0] and 0 <= int(round(cx)) < mask.shape[1]
        ) else float("nan")
        candidates.append(
            TextCandidate(
                id=len(candidates),
                bbox=bbox,
                area_px=area,
                aspect_ratio=round(aspect, 4),
                distance_to_nearest_primitive_px=round(distance, 4),
                height_mm=round(height_mm, 4),
            )
        )

    debug_path: str | None = None
    if out_dir is not None:
        out = Path(out_dir)
        debug_dir = out / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        debug_path = str(debug_dir / "overlay_s6.png")
        _write_debug(debug_path, mask, primitive_mask, gray, candidates, excluded)

    counts = {
        "candidates": len(candidates),
        "excluded": len(excluded),
        "components": len(candidates) + len(excluded),
    }
    result = S6Result(
        candidates=tuple(candidates),
        excluded=tuple(excluded),
        debug_path=debug_path,
        counts=counts,
        warnings=tuple(warnings),
    )
    timing_ms = round((time.perf_counter() - t0) * 1000.0, 3)
    metrics = StageMetrics(
        stage="s6",
        counts={**counts, "warnings": len(warnings)},
        timing_ms=timing_ms,
        error_estimate={
            "dilate_px": round(dilate_px, 4),
            "mm_per_px": round(mm_per_px, 12),
            "exclusion_ratio": round(
                len(excluded) / max(1, len(candidates) + len(excluded)), 6
            ),
        },
        warnings=tuple(warnings),
        details={
            "min_area_px": int(separation.min_area_px),
            "aspect_ratio_max": float(separation.aspect_ratio_max),
            "area_max_mm2": float(separation.area_max_mm2),
            "ocr_engine": "none（本轮只提取候选区）",
            "debug_path": debug_path,
        },
    )
    if out_dir is not None:
        out = Path(out_dir)
        write_json(out / "pending_ocr_regions.json", result.to_json())
        write_metrics(out, metrics)
    return result, metrics


def _write_debug(
    path: str,
    binary: np.ndarray,
    primitive_mask: np.ndarray,
    gray: np.ndarray | None,
    candidates: Sequence[TextCandidate],
    excluded: Sequence[ExcludedRegion],
) -> None:
    """原图灰底 + 红色半透明基元 + 绿色候选框 + 蓝色被排除连通域。"""
    if gray is not None:
        base = np.asarray(gray)
        if base.max() <= 1.0:
            base = base * 255.0
        canvas = cv2.cvtColor(base.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    else:
        canvas = cv2.cvtColor(binary.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    overlay = canvas.copy()
    overlay[primitive_mask > 0] = (0, 0, 255)
    canvas = cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0.0)
    for candidate in candidates:
        x, y, w, h = candidate.bbox
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 0), 1)
    for item in excluded:
        x, y, w, h = item.bbox
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (255, 128, 0), 1)
        cv2.putText(canvas, item.reason[:12], (x, max(8, y - 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 128, 0), 1, cv2.LINE_AA)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), canvas):
        raise S6Error(f"debug 叠加图写出失败：{path}")


__all__ = [
    "ExcludedRegion",
    "S6Error",
    "S6Result",
    "TextCandidate",
    "rasterize_primitives",
    "run_s6_text",
]
