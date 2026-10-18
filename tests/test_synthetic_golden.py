# tests/test_synthetic_golden.py
"""合成图与 golden DXF 的自洽性（用户决策 #2）。"""

from __future__ import annotations

import re
from pathlib import Path

import ezdxf
import numpy as np
import pytest

from src.config import load_manifest
from src.golden_compare import (
    GeomArc,
    GeomCircle,
    GeomEntity,
    GeomLine,
    GeomPolyline,
    canonical_sort,
    read_dxf,
)
from tests.synthetic_golden import (
    ArcSpec,
    CircleSpec,
    LineSpec,
    PolylineSpec,
    Scene,
    all_scenes,
    build_dxf,
    check,
    dxf_bytes,
    render_png_bytes,
    render_scene,
)

ROOT = Path(__file__).resolve().parents[1]


def _expected_entities(scene: Scene) -> list[GeomEntity]:
    out: list[GeomEntity] = []
    for spec in scene.specs:
        if isinstance(spec, LineSpec):
            out.append(GeomLine(layer=spec.layer, start=spec.start, end=spec.end))
        elif isinstance(spec, CircleSpec):
            out.append(GeomCircle(layer=spec.layer, center=spec.center, radius=spec.radius))
        elif isinstance(spec, ArcSpec):
            out.append(
                GeomArc(
                    layer=spec.layer,
                    center=spec.center,
                    radius=spec.radius,
                    start_deg=spec.start_deg,
                    end_deg=spec.end_deg,
                )
            )
        elif isinstance(spec, PolylineSpec):
            out.append(
                GeomPolyline(
                    layer=spec.layer,
                    points=spec.points,
                    bulges=spec.resolved_bulges,
                    closed=spec.closed,
                )
            )
        else:  # pragma: no cover
            raise TypeError(spec)
    return out


def _assert_entity_close(a: GeomEntity, b: GeomEntity, atol: float) -> None:
    assert a.kind == b.kind
    assert a.layer == b.layer
    if isinstance(a, GeomLine) and isinstance(b, GeomLine):
        assert a.start == pytest.approx(b.start, abs=atol)
        assert a.end == pytest.approx(b.end, abs=atol)
    elif isinstance(a, GeomCircle) and isinstance(b, GeomCircle):
        assert a.center == pytest.approx(b.center, abs=atol)
        assert a.radius == pytest.approx(b.radius, abs=atol)
    elif isinstance(a, GeomArc) and isinstance(b, GeomArc):
        assert a.center == pytest.approx(b.center, abs=atol)
        assert a.radius == pytest.approx(b.radius, abs=atol)
        assert a.start_deg == pytest.approx(b.start_deg, abs=atol)
        assert a.end_deg == pytest.approx(b.end_deg, abs=atol)
    elif isinstance(a, GeomPolyline) and isinstance(b, GeomPolyline):
        assert a.closed == b.closed
        assert len(a.points) == len(b.points)
        for pa, pb in zip(a.points, b.points):
            assert pa == pytest.approx(pb, abs=atol)
        for ba, bb in zip(a.bulges, b.bulges):
            assert ba == pytest.approx(bb, abs=atol)
    else:  # pragma: no cover
        raise AssertionError(f"类型不匹配：{type(a)} vs {type(b)}")


def test_scene_inventory_covers_required_primitives() -> None:
    scenes = all_scenes()
    names = {s.name for s in scenes}
    assert names == {"circle_full", "line_single", "arc_120", "polyline_bulge"}
    kinds = {spec.__class__.__name__ for s in scenes for spec in s.specs}
    assert kinds == {"CircleSpec", "LineSpec", "ArcSpec", "PolylineSpec"}


def test_arc_scene_is_120_degrees() -> None:
    scene = next(s for s in all_scenes() if s.name == "arc_120")
    spec = scene.specs[0]
    assert isinstance(spec, ArcSpec)
    assert spec.sweep_deg == pytest.approx(120.0, abs=1e-9)


@pytest.mark.parametrize("scene", all_scenes(), ids=lambda s: s.name)
def test_dxf_roundtrip_matches_specs(scene: Scene, tmp_path: Path) -> None:
    path = tmp_path / scene.dxf_name
    build_dxf(scene).saveas(path)
    got = read_dxf(path, ignore_types=frozenset())
    expected = canonical_sort(_expected_entities(scene))
    assert len(got) == len(expected)
    for a, b in zip(expected, canonical_sort(got)):
        _assert_entity_close(a, b, atol=1e-9)


@pytest.mark.parametrize("scene", all_scenes(), ids=lambda s: s.name)
def test_render_is_deterministic(scene: Scene) -> None:
    first = render_png_bytes(scene)
    second = render_png_bytes(scene)
    assert first == second
    assert len(first) > 0


@pytest.mark.parametrize("scene", all_scenes(), ids=lambda s: s.name)
def test_dxf_bytes_are_deterministic(scene: Scene) -> None:
    """golden DXF 必须字节稳定，否则每次重新生成都会污染 git 历史。"""
    assert dxf_bytes(scene) == dxf_bytes(scene)
    assert b"\r\n" not in dxf_bytes(scene)  # 只写 LF，避免 Windows 换行转换


@pytest.mark.parametrize("scene", all_scenes(), ids=lambda s: s.name)
def test_class_section_is_sorted(scene: Scene) -> None:
    """CLASSES 段必须按名字排序（ezdxf 的 REQUIRED_CLASSES 是 set，跨进程顺序会变）。"""
    names = re.findall(r"\n  0\nCLASS\n  1\n([^\n]+)\n", dxf_bytes(scene).decode("utf-8"))
    assert names, "CLASSES 段为空"
    assert names == sorted(names)


@pytest.mark.parametrize("scene", all_scenes(), ids=lambda s: s.name)
def test_render_size_matches_declared_canvas(scene: Scene) -> None:
    img = render_scene(scene)
    w, h = scene.pixel_size()
    assert img.shape == (h, w)
    assert img.dtype == np.uint8
    # 有线条（不是空白图）
    assert int((img < 128).sum()) > 0


def test_circle_raster_is_centered_on_true_center() -> None:
    """栅格化的圆，其暗像素质心必须落在真实圆心 1 px 内（Y 翻转/定点写反了就会露馅）。"""
    scene = next(s for s in all_scenes() if s.name == "circle_full")
    img = render_scene(scene)
    ys, xs = np.nonzero(img < 128)
    assert len(xs) > 1000
    cx_px = xs.mean()
    cy_px = ys.mean()
    true_cx = 20.0 / scene.mm_per_px
    true_cy = (scene.height_mm - 20.0) / scene.mm_per_px
    assert abs(cx_px - true_cx) < 1.0
    assert abs(cy_px - true_cy) < 1.0


def test_disk_assets_match_code() -> None:
    problems = check(ROOT / "test_images", ROOT / "golden")
    assert problems == []


def test_manifest_registers_every_scene() -> None:
    manifest = load_manifest(
        ROOT / "test_images" / "manifest.csv",
        images_dir=ROOT / "test_images",
        golden_dir=ROOT / "golden",
    )
    for scene in all_scenes():
        row = manifest.row_for(scene.image_name)
        assert row.has_golden is True
        assert row.golden_dxf == scene.dxf_name
        assert row.true_dpi == pytest.approx(scene.dpi)


def test_golden_dxf_present_and_parsable() -> None:
    for scene in all_scenes():
        path = ROOT / "golden" / scene.dxf_name
        assert path.is_file(), f"缺 golden：{path}"
        doc = ezdxf.readfile(path)
        assert doc.dxfversion == "AC1032"  # R2018
        assert len(list(doc.modelspace())) == len(scene.specs)
