# tests/test_config_schema.py
"""config.yaml 的 pydantic-settings schema：合法值通过、非法值启动即报错（SPEC §5）。"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from src.config import (
    PRESET_NAMES,
    ConfigError,
    load_config,
    load_yaml_mapping,
    make_config,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config.yaml"


@pytest.fixture(scope="module")
def base_data() -> dict:
    return load_yaml_mapping(CONFIG)


def _mutate(base: dict, dotted: str, value) -> dict:  # noqa: ANN001
    data = copy.deepcopy(base)
    parts = dotted.split(".")
    cursor = data
    for part in parts[:-1]:
        cursor = cursor[part]
    cursor[parts[-1]] = value
    return data


def test_real_config_loads() -> None:
    cfg = load_config(CONFIG)
    assert cfg.meta.schema_version == "0.2"
    assert cfg.qc.dpi_mode == "degrade"
    assert cfg.qc.budget_reference_dpi == 300.0
    assert len(cfg.dxf.layers) == 10
    assert cfg.dxf.insunits == 4
    assert cfg.dxf.coordinate_decimals == 4
    assert cfg.fitting.tolerance_mm == pytest.approx(0.085)
    assert cfg.presets.default is None
    assert len(PRESET_NAMES) == 4


def test_section_count_is_twelve() -> None:
    cfg = load_config(CONFIG)
    assert len(type(cfg).model_fields) == 12


@pytest.mark.parametrize(
    ("dotted", "bad_value", "expect"),
    [
        ("qc.dpi_mode", "bogus", "qc.dpi_mode"),
        ("fitting.tolerance_mm", -1.0, "fitting.tolerance_mm"),
        ("dxf.coordinate_decimals", 2, "dxf.coordinate_decimals"),
        ("skeleton.spur_removal.length_threshold_px", 10.0, "spur_removal.length_threshold_px"),
        ("registration.mode", "scale", "scale_denominator"),
        ("preprocess.upsample_if_thin.factor", 4, "upsample_if_thin.factor"),
    ],
    ids=[
        "bad-dpi-mode",
        "negative-tolerance",
        "too-few-decimals",
        "hand-written-spur-px",
        "scale-without-denominator",
        "upsample-factor-4",
    ],
)
def test_six_illegal_values_fail_with_field_path(
    base_data: dict, dotted: str, bad_value, expect: str
) -> None:  # noqa: ANN001
    """6 个非法值必须启动即失败，且错误信息指出字段路径。"""
    data = _mutate(base_data, dotted, bad_value)
    with pytest.raises(ConfigError) as excinfo:
        make_config(data, source="test")
    message = str(excinfo.value)
    assert expect in message, f"错误信息没有指出字段 {expect}：{message}"


def test_unknown_key_is_rejected(base_data: dict) -> None:
    data = copy.deepcopy(base_data)
    data["fitting"]["tolerence_mm"] = 0.1  # 故意拼错
    with pytest.raises(ConfigError, match="tolerence_mm"):
        make_config(data)


def test_duplicate_layer_names_rejected(base_data: dict) -> None:
    data = copy.deepcopy(base_data)
    data["dxf"]["layers"][1]["name"] = data["dxf"]["layers"][0]["name"]
    with pytest.raises(ConfigError, match="图层名重复"):
        make_config(data)


def test_min_max_height_ordering_enforced(base_data: dict) -> None:
    data = _mutate(base_data, "text.min_height_mm", 25.0)
    with pytest.raises(ConfigError, match="min_height_mm"):
        make_config(data)


def test_standard_widths_must_be_ascending(base_data: dict) -> None:
    data = _mutate(base_data, "linewidth.standard_widths_mm", [0.35, 0.18])
    with pytest.raises(ConfigError, match="升序"):
        make_config(data)


def test_rms_warn_must_be_below_max(base_data: dict) -> None:
    data = _mutate(base_data, "registration.rms_warn_mm", 0.9)
    with pytest.raises(ConfigError, match="rms_warn_mm"):
        make_config(data)


def test_missing_section_fails(base_data: dict) -> None:
    data = copy.deepcopy(base_data)
    del data["registration"]
    with pytest.raises(ConfigError, match="registration"):
        make_config(data)


def test_non_mapping_yaml_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="顶层必须是映射"):
        make_config(load_yaml_mapping(path))


def test_empty_config_file_rejected(tmp_path: Path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ConfigError, match="顶层必须是映射"):
        make_config(load_yaml_mapping(path))


def test_missing_config_file_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="不存在"):
        load_config(tmp_path / "nope.yaml")


def test_config_is_frozen(base_data: dict) -> None:
    cfg = make_config(base_data)
    with pytest.raises(Exception):  # pydantic ValidationError
        cfg.qc.dpi_mode = "warn"  # type: ignore[misc]
