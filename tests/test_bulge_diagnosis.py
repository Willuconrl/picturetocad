# tests/test_bulge_diagnosis.py
"""M8-2 验收锚点：bulge 过渡区的三个诊断指标（防修复后回退）。

锚点（用户确认，留裕量）：
1. `contour_halfwidth_ratio >= 0.85` —— S2 轮廓贴在墨迹边界（当前 0.955）；
2. `curvature_peak_ratio <= 8.0` —— 中心线伪曲率与弧参考曲率的比值（修复前 10.12，修复后 3.41）；
3. `linewidth_overestimate <= 1.30` —— S2 全局线宽估计的偏高倍数（当前 1.20，为 S2 修复留空间）。

指标定义与 `scripts/diagnose_bulge.py` 完全一致，但这里只算这三个数，运行时间 <2s。
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import cKDTree

from src.config import R2VConfig, load_config
from src.s1_preprocess import run_s1_preprocess
from src.s2_binarize import S2Input, binarize_image
from src.s3_skeleton import run_s3_skeleton
import src.s4_fit as s4

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config.yaml"
SCENE = ROOT / "test_images" / "synthetic_polyline_bulge.png"


@pytest.fixture(scope="module")
def cfg() -> R2VConfig:
    return load_config(CONFIG)


@pytest.fixture(scope="module")
def metrics(cfg: R2VConfig) -> dict[str, float]:
    """跑一次 S1→S3 + S4 中心线重建，取三个诊断量。"""
    pre, _ = run_s1_preprocess(str(SCENE), cfg)
    s2, _ = binarize_image(S2Input.from_preprocess(pre), cfg)
    s3, _ = run_s3_skeleton(s2.binary, cfg, dpi=pre.dpi, linewidth_px=s2.linewidth_px)
    edge = max(s3.graph.edges, key=lambda e: e.length_px)
    contours = np.vstack([np.asarray(c, dtype=np.float64) for c in s2.edges.contours])
    tree = cKDTree(contours)

    # 真实墨宽：二值面积 / 骨架长
    area = int((s2.binary > 0).sum())
    true_width = area / max(edge.length_px, 1e-9)

    # 轮廓到骨架的距离中位数（等距采样，避免逐点全图查询）
    polyline = np.asarray(edge.points, dtype=np.float64)
    lengths = np.hypot(*(np.diff(polyline, axis=0).T))
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    targets = np.linspace(0.0, float(lengths.sum()), max(2, int(lengths.sum()) + 1))
    skeleton = np.column_stack([
        np.interp(targets, cumulative, polyline[:, 0]),
        np.interp(targets, cumulative, polyline[:, 1]),
    ])
    contour_median = float(np.median(tree.query(skeleton, k=1)[0]))

    # 曲率峰值比：S4 重建中心线 vs 远离过渡点的中段
    refined = s4.centerline_from_contours(
        edge.points, tree, contours, linewidth_px=float(s2.linewidth_px)
    )
    curvature = _curvature(refined)
    peak = float(np.max(curvature))
    interior = max(3, len(curvature) // 10)
    transition = int(np.argmax(curvature[interior: len(curvature) - interior])) + interior
    margin = max(5, len(curvature) // 10)
    far = np.concatenate([
        curvature[interior: max(interior + 1, transition - margin)],
        curvature[min(len(curvature) - 1, transition + margin): len(curvature) - interior],
    ])
    far = far[np.isfinite(far) & (far > 0)]
    arc_reference = float(np.median(far)) if far.size else float("nan")

    return {
        "contour_halfwidth_ratio": contour_median / max(true_width / 2.0, 1e-9),
        "curvature_peak_ratio": peak / max(arc_reference, 1e-9),
        "linewidth_overestimate": float(s2.linewidth_px) / max(true_width, 1e-9),
    }


def _curvature(points: np.ndarray) -> np.ndarray:
    n = len(points)
    if n < 3:
        return np.zeros(n)
    tangents = np.zeros((n, 2))
    tangents[1:-1] = points[2:] - points[:-2]
    tangents[0] = points[1] - points[0]
    tangents[-1] = points[-1] - points[-2]
    norms = np.hypot(tangents[:, 0], tangents[:, 1])
    norms[norms < 1e-12] = 1.0
    unit = tangents / norms[:, None]
    turning = np.zeros(n)
    for i in range(1, n - 1):
        cosine = float(np.clip(unit[i - 1] @ unit[i + 1], -1.0, 1.0))
        turning[i] = math.degrees(math.acos(cosine))
    step = np.hypot(*np.diff(points, axis=0).T)
    step = np.concatenate([step, step[-1:]])
    step[step < 1e-9] = 1e-9
    return turning / step


def test_contour_sits_on_ink_boundary(metrics: dict[str, float]) -> None:
    """S2 轮廓必须贴在墨迹边界：轮廓到骨架的距离 ≈ 半宽。"""
    assert metrics["contour_halfwidth_ratio"] >= 0.85, (
        f"轮廓/半宽比 {metrics['contour_halfwidth_ratio']:.3f} < 0.85：S2 轮廓偏离墨迹边界"
    )


def test_spurious_curvature_is_suppressed(metrics: dict[str, float]) -> None:
    """过渡区伪曲率不得超过弧参考的 8 倍（修复前 10.12×，修复后 3.41×）。"""
    assert metrics["curvature_peak_ratio"] <= 8.0, (
        f"伪曲率峰值比 {metrics['curvature_peak_ratio']:.3f} > 8.0：半宽平滑被回退或失效"
    )


def test_linewidth_estimate_within_tolerance(metrics: dict[str, float]) -> None:
    """S2 全局线宽高估不得超过 1.05×（M8-3 EDT 改造后实测 1.0147×）。"""
    assert metrics["linewidth_overestimate"] <= 1.05, (
        f"线宽高估 {metrics['linewidth_overestimate']:.3f}× > 1.05：S2 全局线宽估计退化"
    )
