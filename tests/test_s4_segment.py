# tests/test_s4_segment.py
"""M8-4 分段器单测：积分图 SSE 的数值一致性 + bulge 边切分正确性与耗时红线。

判据（用户确认）：
1. 积分图直线 SSE 与朴素 `fit_line` SSE 数值一致（`rtol=1e-5`）；
2. `synthetic_polyline_bulge` 的那条边切成 **LINE + ARC**，弧段长度与 golden 的 273 px 吻合；
3. 单条边切分耗时 **< 10 ms**（性能红线）。
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import cKDTree

from src.config import R2VConfig, load_config
from src.s1_preprocess import run_s1_preprocess
from src.s2_binarize import S2Input, binarize_image
from src.s3_skeleton import run_s3_skeleton
from src.s4_segment import PrefixLineSSE, best_split, segment_curve
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


def test_bulge_split_is_line_plus_arc(bulge_points: np.ndarray) -> None:
    """切点必须落在弧↔直线交界：弧段 ≈273 px、两侧 RMS 远低于 0.15 mm。"""
    result = best_split(bulge_points, tolerance_px=1.77)
    assert result is not None
    kinds = sorted(result.kinds)
    assert kinds == ["arc", "line"], f"模型组合应为 LINE+ARC，实际 {result.kinds}"
    arc_points = result.index if result.kinds[0] == "arc" else result.points - result.index
    assert arc_points == pytest.approx(273, abs=10), f"弧段点数 {arc_points}，期望 273±10"
    rms_left = math.sqrt(result.sse_left / max(result.index, 1))
    rms_right = math.sqrt(result.sse_right / max(result.points - result.index, 1))
    assert rms_left <= 1.77 and rms_right <= 1.77, (rms_left, rms_right)


def test_segment_curve_returns_two_primitives(bulge_points: np.ndarray) -> None:
    segments = segment_curve(bulge_points, tolerance_px=1.77)
    kinds = sorted(segment.kind for segment in segments)
    assert kinds == ["circle", "line"], f"应为一条直线 + 一段圆弧，实际 {kinds}"


def test_single_edge_latency_under_red_line(bulge_points: np.ndarray) -> None:
    """性能红线：单条边切分 < 10 ms（含 LM 精化）。"""
    # 预热，排除首次 import/缓存开销
    segment_curve(bulge_points, tolerance_px=1.77)
    t0 = time.perf_counter()
    segment_curve(bulge_points, tolerance_px=1.77)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert elapsed_ms < 10.0, f"单边切分 {elapsed_ms:.2f} ms 超过 10 ms 红线"


def test_perfect_arc_is_not_split() -> None:
    """奥卡姆剃刀：完美圆弧必须原样返回 **1 个 ARC**，绝不能被切成 line+arc。"""
    angles = np.linspace(0.0, np.deg2rad(120.0), 120)
    points = np.column_stack([50.0 + 80.0 * np.cos(angles), 60.0 + 80.0 * np.sin(angles)])
    segments = segment_curve(points, tolerance_px=1.77)
    assert len(segments) == 1, f"完美圆弧被切成 {len(segments)} 段"
    assert segments[0].kind == "circle"


def test_perfect_line_is_not_split() -> None:
    points = np.column_stack([np.linspace(0.0, 200.0, 200), np.full(200, 30.0)])
    segments = segment_curve(points, tolerance_px=1.77)
    assert len(segments) == 1 and segments[0].kind == "line"


def test_adjacent_segments_share_bitwise_identical_endpoint(bulge_points: np.ndarray) -> None:
    """相邻基元的连接点必须**逐位相等**（`==`，不允许近似相等）。"""
    segments = segment_curve(bulge_points, tolerance_px=1.77)
    assert len(segments) >= 2, f"bulge 边应切成多段，实际 {len(segments)}"
    for previous, current in zip(segments, segments[1:]):
        end = previous.points[-1]
        start = current.points[0]
        assert end[0] == start[0] and end[1] == start[1], (
            f"连接点不相等：{end} vs {start}"
        )
