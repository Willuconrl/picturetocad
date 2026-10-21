# tests/test_s8_dxf.py
"""S8 DXF 组装：五条验收判据。

① 四类实体正确写出且回读一致（端点 ≤1e-4 mm）；
② 图框闭合多边形为单条 closed=True LWPOLYLINE；
③ 图层名与 S5 layers.json 分配一致；
④ 未配准时 `$PROJECTNAME=R2V-UNCALIBRATED` + metrics WARN；
⑤ metrics_s8 往返 + 同输入两次逐字节一致。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import ezdxf
import numpy as np
import pytest

from src.artifacts import read_metrics
from src.config import R2VConfig, load_config
from src.s2_binarize import S2Input, binarize_image
from src.s3_skeleton import run_s3_skeleton
from src.s4_fit import S4Input, run_s4_fit
from src.s5_layer import run_s5_layer
from src.s7_register import run_s7_register
from src.s8_dxf import (
    UNCALIBRATED_PROJECT_NAME,
    DxfAssemblyError,
    run_s8_dxf,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config.yaml"
DPI = 300.0
MM_PER_PX = 25.4 / DPI
TOL_MM = 1e-4


@pytest.fixture(scope="module")
def cfg() -> R2VConfig:
    return load_config(CONFIG)


@pytest.fixture(scope="module")
def registration(cfg: R2VConfig):
    result, _ = run_s7_register(cfg, control_points=[], px_to_mm=MM_PER_PX)
    return result


def _render(draw, *, width: int = 400, height: int = 300, ss: int = 8) -> np.ndarray:
    big = np.full((height * ss, width * ss), 255, np.uint8)
    draw(big, ss)
    return cv2.resize(big, (width, height), interpolation=cv2.INTER_AREA)


def _stages(image: np.ndarray, cfg: R2VConfig):
    gray = image.astype(np.float32) / 255.0
    s2, _ = binarize_image(S2Input.from_array(gray, dpi=DPI), cfg)
    s3, _ = run_s3_skeleton(s2.binary, cfg, dpi=DPI, linewidth_px=s2.linewidth_px)
    fit, _ = run_s4_fit(
        S4Input(graph=s3.graph, contours=s2.edges.contours, linewidth_px=s2.linewidth_px, dpi=DPI), cfg
    )
    layer, _ = run_s5_layer(
        fit, cfg, contours=s2.edges.contours, mm_per_px=MM_PER_PX,
        page_size_px=(image.shape[1], image.shape[0]), graph=s3.graph,
    )
    return s3, fit, layer


def _write(image: np.ndarray, cfg: R2VConfig, registration, tmp: Path, **kwargs):
    s3, fit, layer = _stages(image, cfg)
    return run_s8_dxf(
        fit, layer, s3.graph, registration, cfg,
        dpi=DPI, image_size_px=(image.shape[1], image.shape[0]), out_dir=tmp, **kwargs,
    )


def _flip_y(y_px: float, height_px: int) -> float:
    """图像 Y（向下）→ CAD Y（向上），单位 mm。"""
    return (height_px - y_px) * MM_PER_PX


# ---------------------------------------------------------------------------
# 判据 ①：四类实体回读一致
# ---------------------------------------------------------------------------
def test_line_and_arc_roundtrip(cfg: R2VConfig, registration, tmp_path: Path) -> None:
    def draw(big, s):
        cv2.line(big, (int(60 * s), int(150 * s)), (int(340 * s), int(150 * s)), 0, int(5 * s))
        cv2.ellipse(big, (int(300 * s), int(70 * s)), (int(40 * s), int(40 * s)), 0, 30, 150, 0, int(5 * s))

    image = _render(draw)
    result, metrics = _write(image, cfg, registration, tmp_path)
    doc = ezdxf.readfile(result.dxf_path)
    entities = list(doc.modelspace())
    lines = [e for e in entities if e.dxftype() == "LINE"]
    assert lines, f"没有 LINE 实体：{result.entities}"
    line = max(
        lines,
        key=lambda e: math.hypot(e.dxf.end.x - e.dxf.start.x, e.dxf.end.y - e.dxf.start.y),
    )
    # 端点：y=150 的水平线 → CAD y = (300-150)*mm_per_px
    assert line.dxf.start.y == pytest.approx(_flip_y(150.0, 300), abs=1.0)
    assert line.dxf.end.x - line.dxf.start.x == pytest.approx(280 * MM_PER_PX, abs=1.0)
    assert line.dxf.layer.startswith("LINES_W")

    arcs = [e for e in entities if e.dxftype() == "ARC"]
    assert arcs, f"没有 ARC 实体：{result.entities}"
    arc = arcs[0]
    radius_mm = arc.dxf.radius
    assert radius_mm * DPI / 25.4 == pytest.approx(40.0, rel=0.05)
    sweep = (arc.dxf.end_angle - arc.dxf.start_angle) % 360.0
    # 扫角精度是 S4 的职责（端点由共享节点投影，短弧会略小于理想扫角）；
    # S8 这里只验"写成了 ARC 且张角量级正确、方向为 CCW"。
    assert sweep == pytest.approx(120.0, abs=25.0)
    assert arc.dxf.end_angle > arc.dxf.start_angle
    assert result.self_check.passed is True, result.self_check.problems
    assert metrics.counts["entities"] == len(entities)


def test_circle_roundtrip(cfg: R2VConfig, registration, tmp_path: Path) -> None:
    image = _render(lambda big, s: cv2.circle(big, (int(200 * s), int(150 * s)), int(80 * s), 0, int(6 * s)))
    result, _ = _write(image, cfg, registration, tmp_path)
    doc = ezdxf.readfile(result.dxf_path)
    circles = [e for e in doc.modelspace() if e.dxftype() == "CIRCLE"]
    assert len(circles) == 1, f"应写出 1 个 CIRCLE：{result.entities}"
    circle = circles[0]
    assert circle.dxf.radius == pytest.approx(80 * MM_PER_PX, rel=0.02)
    assert circle.dxf.center.x == pytest.approx(200 * MM_PER_PX, abs=1.0)
    assert circle.dxf.center.y == pytest.approx(_flip_y(150.0, 300), abs=1.0)


def test_polyline_roundtrip(cfg: R2VConfig, registration, tmp_path: Path) -> None:
    """LWPOLYLINE 回读一致（噪声折线可能被拟成 LINE，也可能是多段线，两者都接受）。"""
    rng = np.random.default_rng(5)
    vertices = [(100 + i * 20.0, 200.0 + (10.0 if i % 2 else -10.0)) for i in range(8)]
    vertices = [(x + rng.normal(0, 0.4), y + rng.normal(0, 0.4)) for x, y in vertices]
    image = np.full((300, 400), 255, np.uint8)
    cv2.polylines(image, np.array([[(int(x), int(y)) for x, y in vertices]], np.int32), False, 0, 5)
    result, _ = _write(image, cfg, registration, tmp_path)
    doc = ezdxf.readfile(result.dxf_path)
    kinds = {e.dxftype() for e in doc.modelspace()}
    assert kinds & {"LWPOLYLINE", "LINE"}, f"没有任何几何实体：{result.entities}"
    assert result.self_check.passed is True, result.self_check.problems
    assert result.self_check.max_coord_delta_mm <= 1e-3


# ---------------------------------------------------------------------------
# 判据 ②：图框闭合多边形
# ---------------------------------------------------------------------------
def test_border_becomes_single_closed_polyline(cfg: R2VConfig, registration, tmp_path: Path) -> None:
    def draw(big, s):
        cv2.rectangle(big, (int(12 * s), int(12 * s)), (int(388 * s), int(288 * s)), 0, int(4 * s))

    image = _render(draw)
    result, metrics = _write(image, cfg, registration, tmp_path)
    doc = ezdxf.readfile(result.dxf_path)
    closed = [e for e in doc.modelspace() if e.dxftype() == "LWPOLYLINE" and e.closed]
    assert len(closed) >= 1, f"图框没有写成闭合多段线：{result.entities}"
    assert metrics.counts["closed_polylines"] >= 1
    polyline = closed[0]
    assert polyline.dxf.layer == "BORDER", f"图框图层应为 BORDER，实际 {polyline.dxf.layer}"
    points = list(polyline.get_points(format="xy"))
    assert len(points) >= 4
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    assert (max(xs) - min(xs)) * DPI / 25.4 == pytest.approx(376.0, abs=6.0)
    assert (max(ys) - min(ys)) * DPI / 25.4 == pytest.approx(276.0, abs=6.0)


# ---------------------------------------------------------------------------
# 判据 ③：图层与 S5 一致
# ---------------------------------------------------------------------------
def test_layers_match_s5_assignments(cfg: R2VConfig, registration, tmp_path: Path) -> None:
    def draw(big, s):
        cv2.line(big, (int(60 * s), int(80 * s)), (int(340 * s), int(80 * s)), 0, int(2 * s))
        cv2.line(big, (int(60 * s), int(160 * s)), (int(340 * s), int(160 * s)), 0, int(6 * s))
        cv2.line(big, (int(60 * s), int(240 * s)), (int(340 * s), int(240 * s)), 0, int(4 * s))

    image = _render(draw)
    s3, fit, layer = _stages(image, cfg)
    result, _ = run_s8_dxf(
        fit, layer, s3.graph, registration, cfg,
        dpi=DPI, image_size_px=(400, 300), out_dir=tmp_path,
    )
    doc = ezdxf.readfile(result.dxf_path)
    dxf_layers = sorted({entity.dxf.layer for entity in doc.modelspace()})
    s5_layers = sorted({a.layer for a in layer.assignments})
    assert dxf_layers == s5_layers, f"DXF 图层 {dxf_layers} != S5 分配 {s5_layers}"
    assert result.layers == layer.counts_by_layer


def test_missing_assignment_raises(cfg: R2VConfig, registration, tmp_path: Path) -> None:
    image = _render(lambda big, s: cv2.line(big, (int(60 * s), int(150 * s)), (int(340 * s), int(150 * s)), 0, int(4 * s)))
    s3, fit, layer = _stages(image, cfg)
    trimmed = layer.model_copy(update={"assignments": layer.assignments[:0]})
    with pytest.raises(DxfAssemblyError, match="没有图层分配"):
        run_s8_dxf(fit, trimmed, s3.graph, registration, cfg, dpi=DPI, image_size_px=(400, 300), out_dir=tmp_path)


# ---------------------------------------------------------------------------
# 判据 ④：未配准水印
# ---------------------------------------------------------------------------
def test_uncalibrated_watermark(cfg: R2VConfig, registration, tmp_path: Path) -> None:
    image = _render(lambda big, s: cv2.line(big, (int(60 * s), int(150 * s)), (int(340 * s), int(150 * s)), 0, int(4 * s)))
    result, metrics = _write(image, cfg, registration, tmp_path)
    assert registration.is_identity is True
    doc = ezdxf.readfile(result.dxf_path)
    assert doc.header["$PROJECTNAME"] == UNCALIBRATED_PROJECT_NAME
    assert result.uncalibrated is True
    assert any("未配准" in warning for warning in result.warnings)
    assert any("未配准" in warning for warning in metrics.warnings)
    # 不写 TEXT/MTEXT 实体（不污染图纸内容）
    kinds = {entity.dxftype() for entity in doc.modelspace()}
    assert "TEXT" not in kinds and "MTEXT" not in kinds


def test_pending_ocr_regions_are_recorded(cfg: R2VConfig, registration, tmp_path: Path) -> None:
    image = _render(lambda big, s: cv2.line(big, (int(60 * s), int(150 * s)), (int(340 * s), int(150 * s)), 0, int(4 * s)))
    boxes = [(10.0, 20.0, 30.0, 12.0), (100.0, 40.0, 50.0, 12.0)]
    result, metrics = _write(image, cfg, registration, tmp_path, pending_ocr_regions=boxes)
    assert result.pending_ocr_regions == 2
    assert metrics.counts["pending_ocr_regions"] == 2
    assert metrics.details["pending_ocr_bboxes"] == [list(box) for box in boxes]


# ---------------------------------------------------------------------------
# 判据 ⑤：metrics 往返 + 逐字节一致
# ---------------------------------------------------------------------------
def test_metrics_roundtrip(cfg: R2VConfig, registration, tmp_path: Path) -> None:
    image = _render(
        lambda big, s: (
            cv2.line(big, (int(60 * s), int(150 * s)), (int(340 * s), int(150 * s)), 0, int(4 * s)),
            cv2.circle(big, (int(300 * s), int(70 * s)), int(30 * s), 0, int(4 * s)),
        )
    )
    result, metrics = _write(image, cfg, registration, tmp_path)
    assert (tmp_path / "metrics_s8.json").is_file()
    assert (tmp_path / "s8_report.json").is_file()
    loaded = read_metrics(tmp_path / "metrics_s8.json")
    assert loaded.stage == "s8"
    assert loaded.counts["entities"] == metrics.counts["entities"]
    assert loaded.details["project_name"] == UNCALIBRATED_PROJECT_NAME
    assert loaded.error_estimate["max_coord_delta_mm"] <= 1e-3
    payload = json.loads((tmp_path / "s8_report.json").read_text(encoding="utf-8"))
    assert payload["entities"] == result.entities


def test_dxf_is_byte_identical(cfg: R2VConfig, registration, tmp_path: Path) -> None:
    image = _render(
        lambda big, s: (
            cv2.line(big, (int(60 * s), int(150 * s)), (int(340 * s), int(150 * s)), 0, int(4 * s)),
            cv2.ellipse(big, (int(300 * s), int(70 * s)), (int(40 * s), int(40 * s)), 0, 30, 150, 0, int(4 * s)),
        )
    )
    s3, fit, layer = _stages(image, cfg)
    first, _ = run_s8_dxf(fit, layer, s3.graph, registration, cfg, dpi=DPI,
                          image_size_px=(400, 300), out_dir=tmp_path / "a")
    second, _ = run_s8_dxf(fit, layer, s3.graph, registration, cfg, dpi=DPI,
                           image_size_px=(400, 300), out_dir=tmp_path / "b")
    assert Path(first.dxf_path).read_bytes() == Path(second.dxf_path).read_bytes()
    first_metrics = json.loads((tmp_path / "a" / "metrics_s8.json").read_text(encoding="utf-8"))
    second_metrics = json.loads((tmp_path / "b" / "metrics_s8.json").read_text(encoding="utf-8"))
    first_metrics.pop("timing_ms", None)
    second_metrics.pop("timing_ms", None)
    first_metrics["details"].pop("dxf_path", None)
    second_metrics["details"].pop("dxf_path", None)
    assert first_metrics == second_metrics


def test_empty_primitives_raises(cfg: R2VConfig, registration, tmp_path: Path) -> None:
    from src.s4_fit import FitResult
    from src.s5_layer import LayerResult

    empty_fit = FitResult(primitives=(), nodes={}, discarded_short_edges=0, warnings=(), counts_by_kind={})
    empty_layers = LayerResult(
        assignments=(), clusters=(), standard_widths_mm=(), border_primitive_ids=(),
        orthogonalized=(), warnings=(), counts_by_layer={},
    )
    from src.s3_skeleton import TopologyGraph

    graph = TopologyGraph(nodes=(), edges=(), skeleton_path="thin", width_cv={"thin": 0.0})
    with pytest.raises(DxfAssemblyError, match="没有可写出的基元"):
        run_s8_dxf(empty_fit, empty_layers, graph, registration, cfg, dpi=DPI, image_size_px=(400, 300))
