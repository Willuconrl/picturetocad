# tests/test_golden_compare.py
"""三项 golden 比对（实体数差 / 关键尺寸 / Hausdorff P95）能否抓住真实退化。"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import ezdxf
import numpy as np
import pytest

from src.golden_compare import (
    GeomCircle,
    GeomLine,
    compare_dxf_files,
    compare_geometry,
    hausdorff_p95,
    read_dxf,
)
from tests.synthetic_golden import all_scenes, build_dxf

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "golden"

# 全部阈值显式给：函数签名里没有隐藏默认值
PARAMS = dict(
    spacing_mm=0.05,
    max_hausdorff_p95_mm=0.1,
    max_entity_count_diff=0,
    dim_abs_tol_mm=0.05,
    dim_rel_tol=1e-3,
)


def _golden_entities(scene_name: str):
    path = GOLDEN / f"synthetic_{scene_name}.dxf"
    return read_dxf(path, ignore_types=frozenset())


def test_golden_against_itself_passes() -> None:
    for scene in all_scenes():
        ents = _golden_entities(scene.name)
        result = compare_geometry(ents, ents, **PARAMS)
        assert result.passed, f"{scene.name}: {result.failed_checks}"
        assert result.hausdorff_p95_mm == pytest.approx(0.0, abs=1e-12)
        assert result.entity_count_diff == 0


def test_dxf_file_level_compare() -> None:
    path = GOLDEN / "synthetic_circle_full.dxf"
    result = compare_dxf_files(path, path, ignore_types=frozenset(), **PARAMS)
    assert result.passed
    assert result.count_by_kind_ref == {"CIRCLE": 1}


def test_radius_error_of_half_percent_is_caught() -> None:
    ref = _golden_entities("circle_full")
    circle = ref[0]
    assert isinstance(circle, GeomCircle)
    cand = [replace(circle, radius=circle.radius * 1.005)]
    result = compare_geometry(ref, cand, **PARAMS)
    assert not result.passed
    assert "KEY_DIMS" in result.failed_checks
    # 0.06 mm 的半径偏差没有超过 0.1 mm 的 Hausdorff 阈值，但被关键尺寸抓到
    assert "HAUSDORFF_P95" not in result.failed_checks
    assert result.hausdorff_p95_mm == pytest.approx(0.06, abs=0.01)
    assert result.worst_rel_dim is not None
    assert result.worst_rel_dim.name.endswith(".radius")
    assert result.worst_rel_dim.rel_err == pytest.approx(0.005, abs=1e-6)


def test_shifted_line_is_caught_by_hausdorff() -> None:
    ref = _golden_entities("line_single")
    line = ref[0]
    assert isinstance(line, GeomLine)
    dy = 0.3
    cand = [
        replace(
            line,
            start=(line.start[0], line.start[1] + dy),
            end=(line.end[0], line.end[1] + dy),
        )
    ]
    result = compare_geometry(ref, cand, **PARAMS)
    assert not result.passed
    assert "HAUSDORFF_P95" in result.failed_checks
    assert "KEY_DIMS" in result.failed_checks
    assert result.hausdorff_p95_mm == pytest.approx(dy, abs=0.01)


def test_missing_entity_is_caught_by_count() -> None:
    ref = _golden_entities("polyline_bulge")
    result = compare_geometry(ref, [], **PARAMS)
    assert not result.passed
    assert "ENTITY_COUNT" in result.failed_checks
    assert result.entity_count_diff == -result.entity_count_ref
    assert result.count_by_kind_cand == {}


def test_hausdorff_p95_unit_behaviour() -> None:
    line_a = np.column_stack([np.linspace(0, 10, 101), np.zeros(101)])
    line_b = np.column_stack([np.linspace(0, 10, 101), np.full(101, 0.5)])
    assert hausdorff_p95(line_a, line_b) == pytest.approx(0.5, abs=1e-9)
    assert hausdorff_p95(line_a, line_a) == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(ValueError, match="percentile"):
        hausdorff_p95(line_a, line_b, percentile=0.0)


def test_unsupported_entity_type_must_not_be_silently_skipped(tmp_path: Path) -> None:
    scene = all_scenes()[0]
    doc = build_dxf(scene)
    doc.modelspace().add_text("R6", dxfattribs={"height": 2.5, "insert": (5.0, 5.0)})
    path = tmp_path / "with_text.dxf"
    doc.saveas(path)

    with pytest.raises(ValueError, match="不支持的实体类型 TEXT"):
        read_dxf(path, ignore_types=frozenset())

    ents = read_dxf(path, ignore_types=frozenset({"TEXT"}))
    assert len(ents) == len(scene.specs)


def test_unknown_threshold_arguments_are_rejected() -> None:
    ents = _golden_entities("circle_full")
    with pytest.raises(ValueError, match="spacing_mm"):
        compare_geometry(ents, ents, **{**PARAMS, "spacing_mm": 0.0})
    with pytest.raises(ValueError, match="max_entity_count_diff"):
        compare_geometry(ents, ents, **{**PARAMS, "max_entity_count_diff": -1})


def test_nan_coordinates_rejected(tmp_path: Path) -> None:
    doc = ezdxf.new("R2018")
    doc.modelspace().add_line((0.0, 0.0), (float("nan"), 1.0))
    path = tmp_path / "nan.dxf"
    doc.saveas(path)
    with pytest.raises(ValueError, match="NaN/Inf"):
        read_dxf(path, ignore_types=frozenset())
