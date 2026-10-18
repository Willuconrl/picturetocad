# tests/test_dxf_writer.py
"""DXF 写出契约：比特级可复现（SPEC §3.3）+ 回读自检（用户决策 #1）。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.artifacts import sha256_file
from src.dxf_repro import FIXED_EZDXF_MARKER
from src.dxf_writer import DXFSelfCheckError, DXFWriter, LayerConfig


def _new_writer() -> DXFWriter:
    return DXFWriter(
        image_height_mm=100.0,
        layer_config=LayerConfig(
            bins=(0.25, 0.42),
            names=("LINES_W0.18", "LINES_W0.35", "LINES_W0.50"),
        ),
    )


def _populate(writer: DXFWriter) -> None:
    writer.add_line((0.0, 90.0), (10.0, 90.0), "LINES_W0.35", linewidth_mm=0.35)
    writer.add_circle((20.0, 80.0), 5.0, linewidth_mm=0.18)
    writer.add_arc((40.0, 80.0), 5.0, 0.0, 90.0, linewidth_mm=0.18)
    writer.add_lwpolyline(
        [(0.0, 60.0), (10.0, 60.0), (10.0, 70.0)], [0.5, 0.0, 0.0],
        layer="ARCS", linewidth_mm=0.5,
    )
    writer.add_text("中文 ABC", (0.0, 50.0), 2.5)


def _save(writer: DXFWriter, path: Path) -> dict:
    return writer.save(path, manifest_id="TEST", config_hash="cfg", rms_mm=None)


def test_two_saves_are_bit_identical(tmp_path: Path) -> None:
    """同一输入两次 save() 的 SHA256 必须相同（用户决策 #1 的验收断言）。"""
    digests = []
    for index in range(2):
        writer = _new_writer()
        _populate(writer)
        out = tmp_path / f"run{index}" / "out.dxf"
        out.parent.mkdir(parents=True, exist_ok=True)
        report = _save(writer, out)
        assert report["ok"] is True
        digests.append(sha256_file(out))
    assert digests[0] == digests[1], digests


def test_output_uses_lf_only(tmp_path: Path) -> None:
    writer = _new_writer()
    _populate(writer)
    out = tmp_path / "out.dxf"
    _save(writer, out)
    data = out.read_bytes()
    assert b"\r\n" not in data, "DXF 里出现 CRLF：写出路径没有走字节写入"
    assert data.endswith(b"EOF\n")


def test_fixed_marker_replaces_timestamp(tmp_path: Path) -> None:
    writer = _new_writer()
    _populate(writer)
    out = tmp_path / "out.dxf"
    _save(writer, out)
    data = out.read_bytes()
    assert FIXED_EZDXF_MARKER.encode() in data
    assert b" @ 20" not in data, "DXF 里仍有 ezdxf 的保存时间戳"


def test_self_check_reports_zero_deviation(tmp_path: Path) -> None:
    writer = _new_writer()
    _populate(writer)
    report = _save(writer, tmp_path / "out.dxf")
    assert report["ok"] is True
    assert report["mismatches"] == []
    assert report["error_estimate"]["coordinate_rounding_mm"] == pytest.approx(0.00005)
    assert report["count_by_kind"] == {
        "ARC": 1,
        "CIRCLE": 1,
        "LINE": 1,
        "LWPOLYLINE": 1,
        "TEXT": 1,
    }


def test_metrics_s8_written_next_to_output(tmp_path: Path) -> None:
    writer = _new_writer()
    _populate(writer)
    out = tmp_path / "out.dxf"
    _save(writer, out)
    metrics = out.with_name("metrics_s8.json")
    assert metrics.is_file()
    payload = json.loads(metrics.read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert "error_estimate" in payload


def test_self_check_failure_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """自检必须真的在比对：把容差压到 0 且坐标带噪声时应抛错。"""
    writer = _new_writer()
    _populate(writer)
    monkeypatch.setattr("src.dxf_writer.SELF_CHECK_TOL", 0.0, raising=True)
    # 篡改期望几何快照，制造"写出的与期望的不一致"（start 是 (x, y) 元组）
    want = writer._written_geom[0]  # noqa: SLF001 - 故意破坏以验证自检
    want["start"] = (want["start"][0] + 0.5, want["start"][1])
    with pytest.raises(DXFSelfCheckError):
        writer.save(tmp_path / "bad.dxf", manifest_id="T", config_hash="c")


def test_reproducible_across_layer_counts(tmp_path: Path) -> None:
    """两次写出即使写入顺序一致，也不能因为 CLASS 顺序/时间戳产生差异。"""
    digests = set()
    for index in range(3):
        writer = _new_writer()
        _populate(writer)
        out = tmp_path / f"r{index}" / "out.dxf"
        out.parent.mkdir(parents=True, exist_ok=True)
        _save(writer, out)
        digests.add(sha256_file(out))
    assert len(digests) == 1
