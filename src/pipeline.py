# src/pipeline.py
"""端到端管线 S0 → S8（CLI `r2v convert` 与 golden 回归共用同一条路径）。

只用一条实现，避免"测试里手工串的管线"与"用户跑的管线"分叉。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.artifacts import StageMetrics
from src.config import R2VConfig
from src.s0_qc import run_s0_qc
from src.s1_preprocess import PreprocessResult, run_s1_preprocess
from src.s2_binarize import S2Input, binarize_image
from src.s3_skeleton import TopologyGraph, run_s3_skeleton
from src.s4_fit import FitResult, S4Input, run_s4_fit
from src.s5_layer import LayerResult, run_s5_layer
from src.s6_text import run_s6_text
from src.s7_register import RegistrationResult, run_s7_register
from src.s8_dxf import S8Result, run_s8_dxf


@dataclass(frozen=True)
class PipelineResult:
    image: str
    dpi: float
    page_size_px: tuple[int, int]
    graph: TopologyGraph
    fit: FitResult
    layers: LayerResult
    registration: RegistrationResult
    dxf: S8Result
    metrics: dict[str, StageMetrics]
    timing_ms: float


def run_pipeline(
    image_path: str | Path,
    cfg: R2VConfig,
    *,
    out_dir: str | Path | None = None,
    rotation_deg: float = 0.0,
    control_points: tuple = (),
    pending_ocr_regions: tuple = (),
) -> PipelineResult:
    """跑完整条链。`out_dir` 为 None 时只算不落盘（仍返回内存结果）。"""
    t0 = time.perf_counter()
    path = str(image_path)
    out = Path(out_dir) if out_dir is not None else None
    metrics: dict[str, StageMetrics] = {}

    qc, qc_metrics = run_s0_qc(path, cfg.qc, out_dir=out)
    metrics["s0"] = qc_metrics

    pre: PreprocessResult
    pre, s1_metrics = run_s1_preprocess(path, cfg, out_dir=out, rotation_deg=rotation_deg)
    metrics["s1"] = s1_metrics

    s2, s2_metrics = binarize_image(S2Input.from_preprocess(pre), cfg, out_dir=out)
    metrics["s2"] = s2_metrics

    s3, s3_metrics = run_s3_skeleton(
        s2.binary, cfg, dpi=pre.dpi, linewidth_px=s2.linewidth_px, out_dir=out
    )
    metrics["s3"] = s3_metrics

    fit, s4_metrics = run_s4_fit(
        S4Input(graph=s3.graph, contours=s2.edges.contours, linewidth_px=s2.linewidth_px, dpi=pre.dpi),
        cfg,
        out_dir=out,
    )
    metrics["s4"] = s4_metrics

    page = (int(s2.binary.shape[1]), int(s2.binary.shape[0]))
    registration, s7_metrics = run_s7_register(
        cfg, control_points=control_points, px_to_mm=25.4 / pre.dpi, out_dir=out, image_size_px=page
    )
    metrics["s7"] = s7_metrics

    layers, s5_metrics = run_s5_layer(
        fit,
        cfg,
        contours=s2.edges.contours,
        # registration.scale = 每像素对应的世界毫米数（未配准时就是 25.4/dpi）
        mm_per_px=registration.scale,
        page_size_px=page,
        graph=s3.graph,
        out_dir=out,
    )
    metrics["s5"] = s5_metrics

    # S6 首轮：只提取文字候选区（不接 OCR），候选区回填给 S8 的 pending_ocr_regions
    s6, s6_metrics = run_s6_text(
        s2.binary,
        fit,
        cfg,
        dpi=pre.dpi,
        out_dir=out,
        gray=pre.image,
        linewidth_px=s2.linewidth_px,
    )
    metrics["s6"] = s6_metrics
    ocr_regions = tuple(candidate.bbox for candidate in s6.candidates)

    dxf, s8_metrics = run_s8_dxf(
        fit,
        layers,
        s3.graph,
        registration,
        cfg,
        dpi=pre.dpi,
        image_size_px=page,
        out_dir=out,
        pending_ocr_regions=ocr_regions,
    )
    metrics["s8"] = s8_metrics

    return PipelineResult(
        image=path,
        dpi=float(pre.dpi),
        page_size_px=page,
        graph=s3.graph,
        fit=fit,
        layers=layers,
        registration=registration,
        dxf=dxf,
        metrics=metrics,
        timing_ms=round((time.perf_counter() - t0) * 1000.0, 3),
    )


def pipeline_summary(result: PipelineResult) -> dict[str, Any]:
    """给 CLI / metrics 用的紧凑摘要。"""
    return {
        "image": result.image,
        "dpi": result.dpi,
        "page_size_px": list(result.page_size_px),
        "primitives": len(result.fit.primitives),
        "clusters": len(result.layers.clusters),
        "entities": result.dxf.entities,
        "layers": result.dxf.layers,
        "self_check_passed": result.dxf.self_check.passed,
        "uncalibrated": result.dxf.uncalibrated,
        "timing_ms": result.timing_ms,
        "warnings": list(result.dxf.warnings),
    }


__all__ = ["PipelineResult", "pipeline_summary", "run_pipeline"]
