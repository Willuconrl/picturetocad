# tests/test_s4_segment.py
"""M8-4 切分预言机单测：积分图 SSE 数值一致性、单模型拦截、切分计划正确性、性能红线。

预言机只输出 `split_indices` 与 `model_types`（不做拟合、不吸附端点），
拟合与后处理全部由 `run_s4_fit` 既有链路完成。
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import cKDTree

from src.config import R2VConfig, load_config
from src.s1_preprocess import run_s1_preprocess
from src.s2_binarize import S2Input, binarize_image
from src.s3_skeleton import run_s3_skeleton
from src.s4_segment import Plan, PrefixLineSSE, best_split, plan_segmentation
import src.s4_fit as s4

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config.yaml"
SCENE = ROOT / "test_images" / "synthetic_polyline_bulge.png"


@pytest.fixture(scope="module")
def cfg() -> R2VConfig:
    return load_config(CONFIG)


@pytest.fixture(scope="module")
def bulge_points(cfg: R2VConfig) -> np.ndarray:
    pre, _ = run_s1_preprocess(str(SCENE), cfg)
    s2, _ = binarize_image(S2Input.from_preprocess(pre), cfg)
    s3, _ = run_s3_skeleton(s2.binary, cfg, dpi=pre.dpi, linewidth_px=s2.linewidth_px)
    contours = np.vstack([np.asarray(c, dtype=np.float64) for c in s2.edges.contours])
    tree = cKDTree(contours)
    edge = max(s3.graph.edges, key=lambda e: e.length_px)
    return s4.centerline_from_contours(
        edge.points, tree, contours, linewidth_px=s2.linewidth_px
    )


def test_integral_sse_matches_naive_fit() -> None:
    """积分图 SSE 必须与朴素 fit_line 的 SSE 数值一致（rtol=1e-5）。"""
    rng = np.random.default_rng(7)
    points = np.column_stack([
        np.linspace(0.0, 100.0, 120),
        0.4 * np.sin(np.linspace(0.0, 3.0, 120)),
    ]) + rng.normal(0.0, 0.15, (120, 2))
    prefix = PrefixLineSSE(points)
    for start, end in ((0, 120), (5, 90), (20, 40), (3, 9)):
        model = s4.fit_line(points[start:end])
        assert model is not None
        residuals = (points[start:end] - np.asarray(model.point)) @ np.asarray(model.normal)
        naive = float(np.sum(residuals**2))
        assert prefix.sse(start, end) == pytest.approx(naive, rel=1e-5), (start, end)


def test_bulge_plan_is_line_then_arc(bulge_points: np.ndarray) -> None:
    """bulge 边的计划：1 个切点、两段 LINE→ARC，切点使弧段 ≈273 点。"""
    plan = plan_segmentation(bulge_points, tolerance_px=1.77)
    assert plan.segment_count == 2, f"应切成 2 段，实际 {plan.model_types}"
    assert len(plan.split_indices) == 1
    assert plan.model_types == ("LINE", "ARC"), plan.model_types
    arc_points = len(bulge_points) - plan.split_indices[0]
    assert arc_points == pytest.approx(273, abs=10), f"弧段点数 {arc_points}，期望 273±10"


def test_perfect_arc_is_single_arc(bulge_points: np.ndarray) -> None:
    """奥卡姆剃刀：完美圆弧必须返回 1 段 ARC、零切点。"""
    angles = np.linspace(0.0, np.deg2rad(120.0), 120)
    points = np.column_stack([50.0 + 80.0 * np.cos(angles), 60.0 + 80.0 * np.sin(angles)])
    plan = plan_segmentation(points, tolerance_px=1.77)
    assert plan.split_indices == ()
    assert plan.model_types == ("ARC",)


def test_perfect_line_is_single_line() -> None:
    points = np.column_stack([np.linspace(0.0, 200.0, 200), np.full(200, 30.0)])
    plan = plan_segmentation(points, tolerance_px=1.77)
    assert plan.split_indices == ()
    assert plan.model_types == ("LINE",)


def test_plan_only_returns_indices_and_types(bulge_points: np.ndarray) -> None:
    """接口契约：预言机只输出切点与模型类型，不返回任何拟合对象。"""
    plan = plan_segmentation(bulge_points, tolerance_px=1.77)
    assert isinstance(plan, Plan)
    assert all(isinstance(index, int) for index in plan.split_indices)
    assert all(kind in ("LINE", "ARC") for kind in plan.model_types)
    assert len(plan.model_types) == len(plan.split_indices) + 1


def test_single_edge_latency_under_red_line(bulge_points: np.ndarray) -> None:
    """性能红线：单条边规划 < 10 ms。"""
    plan_segmentation(bulge_points, tolerance_px=1.77)  # 预热
    t0 = time.perf_counter()
    plan_segmentation(bulge_points, tolerance_px=1.77)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert elapsed_ms < 10.0, f"单边规划 {elapsed_ms:.2f} ms 超过 10 ms 红线"


def test_best_split_still_exposed(bulge_points: np.ndarray) -> None:
    result = best_split(bulge_points, tolerance_px=1.77)
    assert result is not None and result.kinds in (("line", "arc"), ("arc", "line"))
