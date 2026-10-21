# tests/test_s4_fit.py
"""S4 几何拟合：六条验收判据（SPEC §4 S4）。

验收判据（用户确认）：
1. 线段端点 ≤1.0 px、长度相对误差 ≤0.5%；
2. 圆弧半径相对误差 ≤1.0%、圆心 ≤1.5 px；
3. 完整圆闭合 ≤1.0 px；
4. 噪声折线不被 BIC 判成圆弧（负例）；
5. 接头端点严格相等（`==` 断言，禁止 allclose）；
6. metrics_s4 往返 + 同输入两次逐字节一致。

数据流契约：拓扑来自 S3 graph，几何来自 S2 亚像素轮廓。
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.artifacts import read_metrics
from src.config import R2VConfig, load_config
from src.s2_binarize import S2Input, binarize_image
from src.s3_skeleton import run_s3_skeleton
from src.s4_fit import (
    CircleModel,
    FitError,
    LineModel,
    S4Input,
    douglas_peucker,
    fit_circle,
    fit_line,
    run_s4_fit,
    select_model,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config.yaml"
LW = 6.0


@pytest.fixture(scope="module")
def cfg() -> R2VConfig:
    return load_config(CONFIG)


# ---------------------------------------------------------------------------
# 合成夹具：8× 超采样 + INTER_AREA 降采样 → 亚像素抗锯齿
# ---------------------------------------------------------------------------
def _render(draw, *, width: int = 400, height: int = 300, ss: int = 8) -> np.ndarray:
    big = np.full((height * ss, width * ss), 255, np.uint8)
    draw(big, ss)
    return cv2.resize(big, (width, height), interpolation=cv2.INTER_AREA)


def _pipeline(image: np.ndarray, cfg: R2VConfig, *, tmp: Path | None = None):
    gray = image.astype(np.float32) / 255.0
    s2, _ = binarize_image(S2Input.from_array(gray, dpi=300.0), cfg)
    s3, _ = run_s3_skeleton(s2.binary, cfg, dpi=300.0, linewidth_px=s2.linewidth_px)
    source = S4Input(
        graph=s3.graph, contours=s2.edges.contours, linewidth_px=s2.linewidth_px, dpi=300.0
    )
    return run_s4_fit(source, cfg, out_dir=tmp)


def _of_kind(fit, kind: str):
    return [p for p in fit.primitives if p.kind == kind]


# ---------------------------------------------------------------------------
# 验收 1：线段端点 ≤1.0 px、长度相对误差 ≤0.5%
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("a", "b"),
    [
        ((100.37, 150.0), (300.37, 150.0)),
        ((100.5, 80.2), (300.5, 220.3)),
    ],
    ids=["horizontal", "diagonal"],
)
def test_line_endpoint_and_length_accuracy(
    cfg: R2VConfig, a: tuple[float, float], b: tuple[float, float]
) -> None:
    image = _render(
        lambda big, s: cv2.line(
            big, (int(a[0] * s), int(a[1] * s)), (int(b[0] * s), int(b[1] * s)), 0, int(LW * s)
        )
    )
    fit, _ = _pipeline(image, cfg)
    lines = _of_kind(fit, "line")
    assert len(lines) == 1, f"应拟出 1 条直线，实际 {fit.counts_by_kind}"
    line = lines[0]
    true_length = float(np.hypot(b[0] - a[0], b[1] - a[1]))
    relative = abs(line.length_px - true_length) / true_length
    assert relative <= 0.005, f"长度 {line.length_px:.3f} vs {true_length:.3f}（{relative:.2%}）"
    for point in (line.start, line.end):
        assert point is not None
        nearest = min(np.hypot(point[0] - a[0], point[1] - a[1]), np.hypot(point[0] - b[0], point[1] - b[1]))
        assert nearest <= 1.0, f"端点 {point} 距真值 {nearest:.3f} px > 1.0"


# ---------------------------------------------------------------------------
# 验收 2：圆弧半径 ≤1%、圆心 ≤1.5 px
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("radius", [60.0, 80.0])
def test_arc_radius_and_center(cfg: R2VConfig, radius: float) -> None:
    centre = (200.0, 150.0)
    image = _render(
        lambda big, s: cv2.ellipse(
            big,
            (int(centre[0] * s), int(centre[1] * s)),
            (int(radius * s), int(radius * s)),
            0, 25, 155, 0, int(LW * s),
        )
    )
    fit, _ = _pipeline(image, cfg)
    arcs = _of_kind(fit, "arc")
    assert arcs, f"没有拟出圆弧：{fit.counts_by_kind}"
    arc = arcs[0]
    assert arc.radius_px is not None and arc.center is not None
    relative = abs(arc.radius_px - radius) / radius
    assert relative <= 0.01, f"半径 {arc.radius_px:.3f} vs {radius}（{relative:.2%}）"
    offset = float(np.hypot(arc.center[0] - centre[0], arc.center[1] - centre[1]))
    assert offset <= 1.5, f"圆心偏移 {offset:.3f} px"


# ---------------------------------------------------------------------------
# 验收 3：完整圆闭合 ≤1.0 px
# ---------------------------------------------------------------------------
def test_full_circle_closure(cfg: R2VConfig) -> None:
    radius = 80.0
    image = _render(
        lambda big, s: cv2.circle(big, (int(200 * s), int(150 * s)), int(radius * s), 0, int(LW * s))
    )
    fit, _ = _pipeline(image, cfg)
    circles = _of_kind(fit, "circle")
    assert len(circles) == 1, f"应拟出 1 个 CIRCLE，实际 {fit.counts_by_kind}"
    circle = circles[0]
    assert circle.radius_px is not None
    # 闭合误差 = 拟合残差；半径另受 1% 约束
    assert circle.rms_px <= 1.0, f"闭合误差 {circle.rms_px:.3f} px > 1.0"
    assert abs(circle.radius_px - radius) / radius <= 0.01
    assert circle.start == circle.end, "闭合圆的起止点必须是同一个共享节点坐标"


# ---------------------------------------------------------------------------
# 验收 4：噪声折线不得被判成圆弧
# ---------------------------------------------------------------------------
def test_noisy_polyline_is_not_an_arc(cfg: R2VConfig) -> None:
    rng = np.random.default_rng(11)
    vertices = [(100 + i * 12.0, 220.0 + (8.0 if i % 2 else -8.0)) for i in range(16)]
    vertices = [(x + rng.normal(0, 0.35), y + rng.normal(0, 0.35)) for x, y in vertices]

    image = np.full((300, 400), 255, np.uint8)
    points = np.array([[(int(round(x)), int(round(y))) for x, y in vertices]], np.int32)
    cv2.polylines(image, points, False, 0, int(LW))
    fit, _ = _pipeline(image, cfg)
    assert not _of_kind(fit, "arc"), f"噪声折线被误判为圆弧：{fit.counts_by_kind}"
    assert not _of_kind(fit, "circle")


def test_select_model_prefers_line_for_collinear_points() -> None:
    rng = np.random.default_rng(3)
    xs = np.linspace(0.0, 100.0, 40)
    points = np.column_stack([xs, rng.normal(0, 0.2, xs.size)])
    line = fit_line(points)
    circle = fit_circle(points)
    assert line is not None
    assert select_model(line, circle, n=len(points), alpha=0.05, prefer_arcs=True) == "line"


def test_select_model_picks_circle_for_arc_points() -> None:
    angles = np.radians(np.linspace(20.0, 160.0, 60))
    points = np.column_stack([200 + 80 * np.cos(angles), 150 + 80 * np.sin(angles)])
    line = fit_line(points)
    circle = fit_circle(points)
    assert circle is not None and circle.radius == pytest.approx(80.0, rel=1e-6)
    assert select_model(line, circle, n=len(points), alpha=0.05, prefer_arcs=False) == "circle"


# ---------------------------------------------------------------------------
# 验收 5：接头端点严格相等（==）
# ---------------------------------------------------------------------------
def test_shared_endpoints_are_bitwise_equal(cfg: R2VConfig) -> None:
    image = _render(
        lambda big, s: (
            cv2.line(big, (int(60 * s), int(150 * s)), (int(340 * s), int(150 * s)), 0, int(LW * s)),
            cv2.line(big, (int(200 * s), int(150 * s)), (int(200 * s), int(60 * s)), 0, int(LW * s)),
        )
    )
    fit, _ = _pipeline(image, cfg)
    anchored = [p for p in fit.primitives if p.node_a is not None or p.node_b is not None]
    assert len(anchored) >= 2, f"T 形接头应至少有两个基元挂在共享节点上：{fit.counts_by_kind}"

    by_node: dict[int, set[tuple[float, float]]] = {}
    for primitive in anchored:
        for node_id, point in ((primitive.node_a, primitive.start), (primitive.node_b, primitive.end)):
            if node_id is None or point is None:
                continue
            by_node.setdefault(node_id, set()).add(point)

    shared = {node: points for node, points in by_node.items() if len(points) > 0}
    assert shared, "没有任何基元挂到共享节点"
    for node_id, points in shared.items():
        assert len(points) == 1, f"节点 {node_id} 上出现多个不同的端点坐标：{points}"
        assert points == {fit.nodes[node_id]}, f"节点 {node_id} 的端点不等于共享节点坐标"


# ---------------------------------------------------------------------------
# 验收 6：metrics 往返 + 逐字节确定性
# ---------------------------------------------------------------------------
def test_metrics_roundtrip(cfg: R2VConfig, tmp_path: Path) -> None:
    image = _render(
        lambda big, s: cv2.line(big, (int(80 * s), int(150 * s)), (int(320 * s), int(150 * s)), 0, int(LW * s))
    )
    fit, metrics = _pipeline(image, cfg, tmp=tmp_path)
    assert (tmp_path / "primitives.json").is_file()
    loaded = read_metrics(tmp_path / "metrics_s4.json")
    assert loaded.stage == "s4"
    assert loaded.counts["lines"] == fit.counts_by_kind.get("line", 0)
    assert loaded.details["counts_by_kind"] == fit.counts_by_kind
    assert loaded.error_estimate["tolerance_px"] > 0

    payload = json.loads((tmp_path / "primitives.json").read_text(encoding="utf-8"))
    assert len(payload["primitives"]) == len(fit.primitives)
    assert payload["counts_by_kind"] == fit.counts_by_kind


def test_results_are_byte_identical(cfg: R2VConfig, tmp_path: Path) -> None:
    image = _render(
        lambda big, s: (
            cv2.line(big, (int(60 * s), int(150 * s)), (int(340 * s), int(150 * s)), 0, int(LW * s)),
            cv2.line(big, (int(200 * s), int(60 * s)), (int(200 * s), int(150 * s)), 0, int(LW * s)),
        )
    )
    first, _ = _pipeline(image, cfg, tmp=tmp_path / "a")
    second, _ = _pipeline(image, cfg, tmp=tmp_path / "b")
    assert (tmp_path / "a" / "primitives.json").read_bytes() == (tmp_path / "b" / "primitives.json").read_bytes()
    assert first.to_json() == second.to_json()


# ---------------------------------------------------------------------------
# 其他：单元与边界
# ---------------------------------------------------------------------------
def test_fit_line_recovers_direction_for_two_points() -> None:
    """2 点也必须给出正确方向（曾经的 bug：用了单位阵，斜线被拟成水平线）。"""
    line = fit_line(np.array([[0.0, 0.0], [10.0, 10.0]]))
    assert line is not None
    assert abs(abs(line.direction[0]) - abs(line.direction[1])) < 1e-9


def test_fit_circle_on_synthetic_points() -> None:
    angles = np.linspace(0.0, 2 * np.pi, 60, endpoint=False)
    points = np.column_stack([50 + 30 * np.cos(angles), 40 + 30 * np.sin(angles)])
    circle = fit_circle(points)
    assert isinstance(circle, CircleModel)
    assert circle.radius == pytest.approx(30.0, rel=1e-6)
    assert circle.center == pytest.approx((50.0, 40.0), abs=1e-6)


def test_douglas_peucker_keeps_endpoints_and_corners() -> None:
    points = [(0.0, 0.0), (5.0, 0.0), (10.0, 0.0), (10.0, 5.0), (10.0, 10.0)]
    simplified = douglas_peucker(points, 0.5)
    assert simplified[0] == points[0]
    assert simplified[-1] == points[-1]
    assert (10.0, 0.0) in simplified


def test_douglas_peucker_rejects_bad_tolerance() -> None:
    with pytest.raises(FitError, match="tolerance"):
        douglas_peucker([(0.0, 0.0), (1.0, 1.0)], 0.0)


def test_short_edges_are_discarded_and_counted(cfg: R2VConfig) -> None:
    """短边丢弃必须记账：discarded_short_edges 在 metrics 与结果里一致。"""
    image = _render(
        lambda big, s: cv2.line(big, (int(60 * s), int(150 * s)), (int(340 * s), int(150 * s)), 0, int(LW * s))
    )
    fit, metrics = _pipeline(image, cfg)
    assert metrics.counts["discarded_short_edges"] == fit.discarded_short_edges


def test_empty_graph_raises(cfg: R2VConfig) -> None:
    from src.s3_skeleton import TopologyGraph

    empty = TopologyGraph(nodes=(), edges=(), skeleton_path="thin", width_cv={"thin": 0.0})
    with pytest.raises(FitError, match="没有边"):
        run_s4_fit(S4Input(graph=empty, contours=(np.zeros((4, 2)),), linewidth_px=5.0, dpi=300.0), cfg)


def test_missing_contours_raises(cfg: R2VConfig) -> None:
    image = _render(
        lambda big, s: cv2.line(big, (int(60 * s), int(150 * s)), (int(340 * s), int(150 * s)), 0, int(LW * s))
    )
    gray = image.astype(np.float32) / 255.0
    s2, _ = binarize_image(S2Input.from_array(gray, dpi=300.0), cfg)
    s3, _ = run_s3_skeleton(s2.binary, cfg, dpi=300.0, linewidth_px=s2.linewidth_px)
    with pytest.raises(FitError, match="轮廓"):
        run_s4_fit(S4Input(graph=s3.graph, contours=(), linewidth_px=5.0, dpi=300.0), cfg)
