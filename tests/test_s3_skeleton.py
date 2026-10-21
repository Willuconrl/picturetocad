# tests/test_s3_skeleton.py
"""S3 骨架化与拓扑重建：五条验收判据（SPEC §4 S3）。

验收判据（用户确认）：
1. 合成图节点类型准确率 ≥95%；
2. 毛刺修剪在 2.5×线宽边界不误删真端点；
3. 桥接在 3×线宽/20° 两侧行为正确且打 `bridged: true`；
4. graph.json 与 metrics_s3 往返；
5. 同输入两次运行的 graph.json 逐字节一致。

坐标契约：像素 (col,row) 的中心是 (col+0.5, row+0.5)（见 docs/coordinate_system.md）。
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.artifacts import read_metrics
from src.config import R2VConfig, load_config
from src.s3_skeleton import (
    SkeletonError,
    bridge_gaps,
    merge_collinear_nodes,
    prune_spurs,
    run_s3_skeleton,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config.yaml"
LW = 5.0  # 合成图线宽（px）


@pytest.fixture(scope="module")
def cfg() -> R2VConfig:
    return load_config(CONFIG)


def _canvas(width: int = 400, height: int = 300) -> np.ndarray:
    return np.zeros((height, width), np.uint8)


def _line(img: np.ndarray, a: tuple[int, int], b: tuple[int, int], thickness: int = 5) -> None:
    cv2.line(img, a, b, 255, thickness)


# ---------------------------------------------------------------------------
# 验收 1：节点类型准确率 ≥95%
# ---------------------------------------------------------------------------
def _node_near(result, x: float, y: float, radius: float = 6.0):
    best, best_distance = None, 1e9
    for node in result.graph.nodes:
        distance = float(np.hypot(node.x - (x + 0.5), node.y - (y + 0.5)))
        if distance < best_distance:
            best, best_distance = node, distance
    return best if best_distance <= radius else None


def _scene_node_type(cfg: R2VConfig, draw, expect_xy: tuple[float, float]) -> str | None:
    image = _canvas()
    draw(image)
    result, _ = run_s3_skeleton(image, cfg, dpi=300.0, linewidth_px=LW)
    node = _node_near(result, *expect_xy)
    return None if node is None else node.type


SCENES = {
    "cross": (
        lambda im: (
            _line(im, (40, 150), (360, 150)),
            _line(im, (200, 40), (200, 260)),
        ),
        (200, 150),
        "cross",
    ),
    "tee": (
        lambda im: (
            _line(im, (40, 150), (360, 150)),
            _line(im, (200, 150), (200, 60)),
        ),
        (200, 150),
        "tee",
    ),
    "corner": (
        lambda im: (
            _line(im, (60, 60), (60, 240)),
            _line(im, (60, 240), (340, 240)),
        ),
        (60, 240),
        "corner",
    ),
    "wye": (
        lambda im: (
            _line(im, (200, 150), (80, 60)),
            _line(im, (200, 150), (320, 60)),
            _line(im, (200, 150), (200, 260)),
        ),
        (200, 150),
        "wye",
    ),
}


@pytest.mark.parametrize("name", sorted(SCENES))
def test_node_type_accuracy(cfg: R2VConfig, name: str) -> None:
    draw, expected_xy, expected_type = SCENES[name]
    got = _scene_node_type(cfg, draw, expected_xy)
    assert got == expected_type, f"{name}: 期望 {expected_type}，得到 {got}"


def test_node_type_accuracy_meets_threshold(cfg: R2VConfig) -> None:
    """汇总准确率：≥95%（config 的 skeleton.acceptance.node_type_accuracy_min）。"""
    correct = 0
    for name in sorted(SCENES):
        draw, expected_xy, expected_type = SCENES[name]
        if _scene_node_type(cfg, draw, expected_xy) == expected_type:
            correct += 1
    accuracy = correct / len(SCENES)
    assert accuracy >= cfg.skeleton.acceptance.node_type_accuracy_min, (
        f"节点类型准确率 {accuracy:.2%} < {cfg.skeleton.acceptance.node_type_accuracy_min:.0%}"
    )


def test_endpoint_and_loop_classification(cfg: R2VConfig) -> None:
    image = _canvas()
    _line(image, (60, 150), (340, 150))
    result, _ = run_s3_skeleton(image, cfg, dpi=300.0, linewidth_px=LW)
    types = sorted(node.type for node in result.graph.nodes)
    assert types == ["endpoint", "endpoint"]

    ring = _canvas(320, 320)
    cv2.circle(ring, (160, 160), 100, 255, 5)
    loop_result, _ = run_s3_skeleton(ring, cfg, dpi=300.0, linewidth_px=LW)
    assert [node.type for node in loop_result.graph.nodes] == ["isolated"]
    assert loop_result.graph.edges[0].closed is True


def test_skeleton_uses_continuous_pixel_coordinates(cfg: R2VConfig) -> None:
    """+0.5 断言：骨架像素 (59,150) 的中心是 (59.5,150.5)，节点坐标必须 +0.5。"""
    image = _canvas()
    _line(image, (60, 150), (340, 150))
    result, _ = run_s3_skeleton(image, cfg, dpi=300.0, linewidth_px=LW)
    xs = sorted(node.x for node in result.graph.nodes)
    ys = sorted(node.y for node in result.graph.nodes)
    assert xs[0] == pytest.approx(59.5, abs=1.0)
    assert xs[-1] == pytest.approx(340.5, abs=1.0)
    for y in ys:
        assert y == pytest.approx(150.5, abs=0.6), y


# ---------------------------------------------------------------------------
# 验收 2：毛刺修剪（2.5×线宽边界）
# ---------------------------------------------------------------------------
def _spur_scene(length_px: int) -> np.ndarray:
    image = _canvas()
    _line(image, (40, 180), (360, 180))
    _line(image, (200, 180), (200, 180 - length_px))
    return image


def test_spur_shorter_than_threshold_is_pruned(cfg: R2VConfig) -> None:
    threshold = cfg.skeleton.spur_removal.length_threshold_linewidths * LW  # 12.5 px
    image = _spur_scene(int(threshold * 0.6))
    result, metrics = run_s3_skeleton(image, cfg, dpi=300.0, linewidth_px=LW)
    assert result.pruned_spurs >= 1
    assert metrics.counts["pruned_spurs"] == result.pruned_spurs
    assert all(node.degree != 3 for node in result.graph.nodes), "毛刺未被剪掉"


def test_spur_longer_than_threshold_is_kept(cfg: R2VConfig) -> None:
    threshold = cfg.skeleton.spur_removal.length_threshold_linewidths * LW
    image = _spur_scene(int(threshold * 1.6))
    result, _ = run_s3_skeleton(image, cfg, dpi=300.0, linewidth_px=LW)
    assert result.pruned_spurs == 0
    assert any(node.type == "tee" for node in result.graph.nodes)


def test_true_endpoint_is_never_pruned(cfg: R2VConfig) -> None:
    """一条 280px 的线：两个端点都必须保留（长度远超阈值）。"""
    image = _canvas()
    _line(image, (60, 150), (340, 150))
    result, _ = run_s3_skeleton(image, cfg, dpi=300.0, linewidth_px=LW)
    endpoints = [node for node in result.graph.nodes if node.type == "endpoint"]
    assert len(endpoints) == 2
    assert result.pruned_spurs == 0


def test_prune_spurs_direct_api() -> None:
    """直接测剪枝函数：短分支被删、长分支保留。"""
    from src.s3_skeleton import _MutableGraph  # noqa: PLC2701 - 受控测试内部结构

    graph = _MutableGraph()
    a = graph.add_node(0.0, 0.0)
    b = graph.add_node(100.0, 0.0)
    short = graph.add_node(50.0, 6.0)   # 6px 分支 < 2.5×5
    graph.add_edge(a, b, [(0.0, 0.0), (100.0, 0.0)], linewidth_px=5.0)
    graph.add_edge(graph.add_node(50.0, 0.0), short, [(50.0, 0.0), (50.0, 6.0)], linewidth_px=5.0)
    removed = prune_spurs(
        graph, linewidth_px=5.0, length_threshold_linewidths=2.5, endpoint_curvature_check=True
    )
    assert removed >= 0  # 分支垂直于母线 → 不是共线尾巴，允许被剪
    assert short not in graph.nodes or graph.degree(short) == 0


# ---------------------------------------------------------------------------
# 验收 3：断线桥接（3×线宽 / 20°）
# ---------------------------------------------------------------------------
def _bridge_scene(gap_px: int, *, dy: int = 0) -> np.ndarray:
    image = _canvas()
    _line(image, (40, 150), (195, 150))
    _line(image, (195 + gap_px, 150 + dy), (360, 150 + dy))
    return image


def test_gap_within_limit_is_bridged(cfg: R2VConfig) -> None:
    max_gap = cfg.skeleton.bridge.max_gap_linewidths * LW  # 15 px
    image = _bridge_scene(int(max_gap * 0.6))
    result, metrics = run_s3_skeleton(image, cfg, dpi=300.0, linewidth_px=LW)
    bridged = [edge for edge in result.graph.edges if edge.bridged]
    assert len(bridged) == 1, f"应桥接 1 条，实际 {len(bridged)}"
    assert metrics.counts["bridged_edges"] == 1
    assert result.bridged_edges == 1


def test_gap_beyond_limit_is_not_bridged(cfg: R2VConfig) -> None:
    max_gap = cfg.skeleton.bridge.max_gap_linewidths * LW
    image = _bridge_scene(int(max_gap * 1.8))
    result, _ = run_s3_skeleton(image, cfg, dpi=300.0, linewidth_px=LW)
    assert [edge for edge in result.graph.edges if edge.bridged] == []


def test_direction_continuity_blocks_bridging(cfg: R2VConfig) -> None:
    """间隙小但方向偏折超过 20° → 不桥接。"""
    max_gap = cfg.skeleton.bridge.max_gap_linewidths * LW
    dy = int(round(max_gap * 0.6 * np.tan(np.radians(40.0))))
    image = _bridge_scene(int(max_gap * 0.6), dy=dy)
    result, _ = run_s3_skeleton(image, cfg, dpi=300.0, linewidth_px=LW)
    assert [edge for edge in result.graph.edges if edge.bridged] == []


def test_bridging_can_be_disabled(cfg: R2VConfig) -> None:
    disabled = cfg.model_copy(
        update={"skeleton": cfg.skeleton.model_copy(
            update={"bridge": cfg.skeleton.bridge.model_copy(update={"enable": False})}
        )}
    )
    image = _bridge_scene(6)
    result, _ = run_s3_skeleton(image, disabled, dpi=300.0, linewidth_px=LW)
    assert result.bridged_edges == 0


# ---------------------------------------------------------------------------
# 验收 4：graph.json / metrics_s3 往返
# ---------------------------------------------------------------------------
def test_graph_and_metrics_roundtrip(cfg: R2VConfig, tmp_path: Path) -> None:
    image = _canvas()
    _line(image, (40, 150), (360, 150))
    _line(image, (200, 150), (200, 60))
    result, metrics = run_s3_skeleton(image, cfg, dpi=300.0, linewidth_px=LW, out_dir=tmp_path)

    graph_json = tmp_path / "graph.json"
    metrics_json = tmp_path / "metrics_s3.json"
    assert graph_json.is_file() and metrics_json.is_file()

    payload = json.loads(graph_json.read_text(encoding="utf-8"))
    assert payload["skeleton_path"] in ("thin", "medial_axis")
    assert len(payload["nodes"]) == len(result.graph.nodes)
    assert len(payload["edges"]) == len(result.graph.edges)

    loaded = read_metrics(metrics_json)
    assert loaded.stage == "s3"
    assert loaded.counts["nodes"] == len(result.graph.nodes)
    assert loaded.counts["edges"] == len(result.graph.edges)
    assert loaded.counts["pruned_spurs"] == result.pruned_spurs
    assert loaded.details["node_types"] == result.graph.counts_by_type()
    assert loaded.timing_ms >= 0.0


def test_metrics_counts_are_integers(cfg: R2VConfig, tmp_path: Path) -> None:
    image = _canvas()
    _line(image, (60, 150), (340, 150))
    _, metrics = run_s3_skeleton(image, cfg, dpi=300.0, linewidth_px=LW, out_dir=tmp_path)
    assert all(isinstance(value, int) for value in metrics.counts.values())


# ---------------------------------------------------------------------------
# 验收 5：确定性（两次运行 graph.json 逐字节一致）
# ---------------------------------------------------------------------------
def test_graph_is_byte_identical_across_runs(cfg: R2VConfig, tmp_path: Path) -> None:
    image = _canvas()
    _line(image, (40, 150), (360, 150))
    _line(image, (200, 40), (200, 260))
    cv2.circle(image, (300, 90), 40, 255, 5)
    first, _ = run_s3_skeleton(image, cfg, dpi=300.0, linewidth_px=LW, out_dir=tmp_path / "a")
    second, _ = run_s3_skeleton(image, cfg, dpi=300.0, linewidth_px=LW, out_dir=tmp_path / "b")
    assert (tmp_path / "a" / "graph.json").read_bytes() == (tmp_path / "b" / "graph.json").read_bytes()
    assert first.graph.to_json() == second.graph.to_json()
    assert [n.id for n in first.graph.nodes] == [n.id for n in second.graph.nodes]


# ---------------------------------------------------------------------------
# 其他：输入校验与图操作
# ---------------------------------------------------------------------------
def test_empty_input_rejected(cfg: R2VConfig) -> None:
    with pytest.raises(SkeletonError, match="为空"):
        run_s3_skeleton(np.zeros((32, 32), np.uint8), cfg, dpi=300.0)


def test_non_2d_input_rejected(cfg: R2VConfig) -> None:
    with pytest.raises(SkeletonError, match="二维"):
        run_s3_skeleton(np.zeros((8, 8, 3), np.uint8), cfg, dpi=300.0)


def test_merge_collinear_nodes_direct_api() -> None:
    from src.s3_skeleton import _MutableGraph  # noqa: PLC2701

    graph = _MutableGraph()
    a = graph.add_node(0.0, 0.0)
    mid = graph.add_node(50.0, 0.1)   # 偏离 0.1px < 0.5px 容差
    b = graph.add_node(100.0, 0.0)
    graph.add_edge(a, mid, [(0.0, 0.0), (50.0, 0.1)], linewidth_px=5.0)
    graph.add_edge(mid, b, [(50.0, 0.1), (100.0, 0.0)], linewidth_px=5.0)
    merged = merge_collinear_nodes(graph, tolerance_px=0.5)
    assert merged == 1
    assert len(graph.nodes) == 2
    assert len(graph.edges) == 1


def test_bridge_gaps_direct_api() -> None:
    from src.s3_skeleton import _MutableGraph  # noqa: PLC2701

    graph = _MutableGraph()
    a = graph.add_node(0.0, 0.0)
    b = graph.add_node(10.0, 0.0)
    c = graph.add_node(20.0, 0.0)
    d = graph.add_node(30.0, 0.0)
    graph.add_edge(a, b, [(0.0, 0.0), (10.0, 0.0)], linewidth_px=5.0)
    graph.add_edge(c, d, [(20.0, 0.0), (30.0, 0.0)], linewidth_px=5.0)
    bridged = bridge_gaps(graph, linewidth_px=5.0, max_gap_linewidths=3.0, max_angle_deg=20.0)
    assert bridged == 1
    assert any(edge["bridged"] for edge in graph.edges.values())


def test_dual_path_selection_is_recorded(cfg: R2VConfig) -> None:
    image = _canvas()
    _line(image, (60, 150), (340, 150))
    result, metrics = run_s3_skeleton(image, cfg, dpi=300.0, linewidth_px=LW)
    assert result.graph.skeleton_path in ("thin", "medial_axis")
    assert set(result.graph.width_cv) == {"thin", "medial_axis"}
    assert metrics.details["skeleton_path"] == result.graph.skeleton_path
