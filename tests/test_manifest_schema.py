# tests/test_manifest_schema.py
"""manifest.csv schema 的正/负用例（用户决策 #1）。"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import (
    MANIFEST_COLUMNS,
    ManifestError,
    ManifestRow,
    dpi_budget_scale,
    load_manifest,
    mm_per_px,
)

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "test_images"
GOLDEN = ROOT / "golden"
MANIFEST = IMAGES / "manifest.csv"


def _write_csv(tmp_path: Path, header: str, rows: list[str]) -> Path:
    path = tmp_path / "manifest.csv"
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
    return path


def test_real_manifest_loads() -> None:
    manifest = load_manifest(MANIFEST, images_dir=IMAGES, golden_dir=GOLDEN)
    assert len(manifest.rows) == 5
    assert {r.page_class for r in manifest.rows} == {"mechanical_part", "synthetic"}
    assert len(manifest.with_golden) == 4


def test_real_screenshot_row_is_low_dpi_and_has_no_golden() -> None:
    manifest = load_manifest(MANIFEST, images_dir=IMAGES, golden_dir=GOLDEN)
    row = manifest.row_for("QDIZE9403.JPG")
    assert row.true_dpi == 72.0
    assert row.preset == "mechanical"
    assert row.has_golden is False
    assert row.golden_dxf == "QDIZE9403.dxf"  # 预留目标名，但文件尚未描
    assert "截图" in row.notes


def test_column_order_is_contract(tmp_path: Path) -> None:
    bad = _write_csv(tmp_path, "file,true_dpi,preset,page_class,golden_dxf,notes,has_golden", [])
    with pytest.raises(ManifestError, match="表头不符"):
        load_manifest(bad, images_dir=IMAGES, require_files=False)
    assert MANIFEST_COLUMNS[1] == "preset"


def test_has_golden_requires_golden_dxf() -> None:
    with pytest.raises(ValueError, match="has_golden=true 时必须填 golden_dxf"):
        ManifestRow(
            file="a.png", true_dpi=300, page_class="synthetic", has_golden=True, golden_dxf=None
        )


def test_point_zero_dpi_rejected() -> None:
    with pytest.raises(ValueError, match="greater than 0"):
        ManifestRow(file="a.png", true_dpi=0, page_class="synthetic")


def test_blank_preset_means_base() -> None:
    row = ManifestRow(file="a.png", preset="", true_dpi=300, page_class="synthetic")
    assert row.preset is None


def test_path_separator_in_file_rejected() -> None:
    with pytest.raises(ValueError, match="不许带路径"):
        ManifestRow(file="sub/a.png", true_dpi=300, page_class="synthetic")


def test_golden_must_be_dxf() -> None:
    with pytest.raises(ValueError, match="必须是 .dxf"):
        ManifestRow(
            file="a.png", true_dpi=300, page_class="synthetic", golden_dxf="a.dwg", has_golden=True
        )


def test_duplicate_file_rejected(tmp_path: Path) -> None:
    header = ",".join(MANIFEST_COLUMNS)
    rows = [
        "a.png,,300,synthetic,,x,false",
        "A.PNG,,300,synthetic,,x,false",
    ]
    path = _write_csv(tmp_path, header, rows)
    with pytest.raises(ManifestError, match="file 重复"):
        load_manifest(path, images_dir=IMAGES, require_files=False)


def test_missing_image_detected(tmp_path: Path) -> None:
    header = ",".join(MANIFEST_COLUMNS)
    path = _write_csv(tmp_path, header, ["no_such_file.png,,300,synthetic,,x,false"])
    with pytest.raises(ManifestError, match="图像不存在"):
        load_manifest(path, images_dir=IMAGES, require_files=True)


def test_missing_golden_detected(tmp_path: Path) -> None:
    header = ",".join(MANIFEST_COLUMNS)
    path = _write_csv(
        tmp_path,
        header,
        ["synthetic_circle_full.png,,300,synthetic,not_drawn.dxf,x,true"],
    )
    with pytest.raises(ManifestError, match="golden 不存在"):
        load_manifest(path, images_dir=IMAGES, golden_dir=GOLDEN, require_files=True)


def test_row_error_carries_line_number(tmp_path: Path) -> None:
    header = ",".join(MANIFEST_COLUMNS)
    path = _write_csv(
        tmp_path,
        header,
        [
            "synthetic_circle_full.png,,300,synthetic,synthetic_circle_full.dxf,note,true",
            "synthetic_arc_120.png,,not-a-number,synthetic,,note,false",
        ],
    )
    with pytest.raises(ManifestError, match=":3 行非法"):
        load_manifest(path, images_dir=IMAGES, golden_dir=GOLDEN, require_files=False)


def test_budget_scale_for_screenshot() -> None:
    """72 DPI 截图的 mm 预算放大 300/72 ≈ 4.1667 倍（用户决策 #3）。"""
    assert dpi_budget_scale(72.0, 300.0) == pytest.approx(4.1666666, rel=1e-6)
    assert dpi_budget_scale(300.0, 300.0) == 1.0
    with pytest.raises(ValueError, match="true_dpi"):
        dpi_budget_scale(0.0, 300.0)


def test_mm_per_px_spec_table() -> None:
    """SPEC §1.1 的换算表必须与代码一致。"""
    assert mm_per_px(300.0) == pytest.approx(0.0846667, rel=1e-6)
    assert mm_per_px(400.0) == pytest.approx(0.0635, rel=1e-9)
    assert mm_per_px(600.0) == pytest.approx(0.0423333, rel=1e-6)
    with pytest.raises(ValueError, match="dpi"):
        mm_per_px(-1.0)
