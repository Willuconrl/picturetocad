# tests/test_presets.py
"""预设覆盖：4 套 preset 全过、未知键硬失败、哈希可复现（用户验收判据）。"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from src.config import PRESET_NAMES, ConfigError, load_config, load_yaml_mapping
from src.presets import (
    PresetError,
    apply_overrides,
    build_effective,
    config_hash,
    flatten_paths,
    load_presets,
    resolve_presets_path,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config.yaml"


@pytest.fixture(scope="module")
def base_data() -> dict:
    return load_yaml_mapping(CONFIG)


def test_resolve_presets_path_from_config() -> None:
    """预设文件路径由 config.yaml 声明（M9-4 起 vendored 进仓库 config/presets.yaml）。"""
    declared = load_config(CONFIG).presets.file
    path = resolve_presets_path(CONFIG, declared)
    assert path.is_file()
    assert path.name == "presets.yaml"


def test_resolve_presets_path_missing() -> None:
    with pytest.raises(PresetError, match="找不到 presets.file"):
        resolve_presets_path(CONFIG, "no/such/presets.yaml")


def test_presets_file_declares_four_presets() -> None:
    declared = load_config(CONFIG).presets.file
    path = resolve_presets_path(CONFIG, declared)
    preset_file = load_presets(path)
    assert preset_file.version == "0.2"
    assert set(preset_file.presets) == set(PRESET_NAMES)


@pytest.mark.parametrize("name", sorted(PRESET_NAMES))
def test_each_preset_builds_and_validates(name: str) -> None:
    effective = build_effective(CONFIG, name)
    assert effective.preset == name
    assert effective.overridden_keys, "overrides 不应为空"
    assert len(effective.config_hash) == 64
    assert effective.config is not None


def test_preset_hash_is_stable() -> None:
    a = build_effective(CONFIG, "blueprint")
    b = build_effective(CONFIG, "blueprint")
    assert a.config_hash == b.config_hash
    assert a.overridden_keys == b.overridden_keys


def test_different_presets_have_different_hashes() -> None:
    hashes = {name: build_effective(CONFIG, name).config_hash for name in PRESET_NAMES}
    assert len(set(hashes.values())) == len(PRESET_NAMES)


def test_blueprint_actually_overrides_orthogonalize() -> None:
    base = build_effective(CONFIG, None)
    bp = build_effective(CONFIG, "blueprint")
    assert base.config.dxf.orthogonalize is False
    assert bp.config.dxf.orthogonalize is True
    assert "dxf.orthogonalize" in bp.overridden_keys


def test_unknown_preset_rejected() -> None:
    with pytest.raises(PresetError, match="未知 preset"):
        build_effective(CONFIG, "no-such-preset")


def test_unknown_override_key_rejected_with_suggestion(base_data: dict) -> None:
    with pytest.raises(PresetError) as excinfo:
        apply_overrides(
            base_data,
            {"fitting.tolerence_mm": 0.1},
            allow_unknown_keys=False,
            source="test",
        )
    message = str(excinfo.value)
    assert "tolerence_mm" in message
    assert "tolerance_mm" in message, f"没有给出最接近的候选：{message}"


def test_intermediate_path_must_exist(base_data: dict) -> None:
    with pytest.raises(PresetError, match="中间段"):
        apply_overrides(base_data, {"nope.deep.key": 1}, allow_unknown_keys=False)


def test_type_mismatch_after_patch_fails(base_data: dict) -> None:
    """`0.15` 被写成 `"0.15"` 必须在覆盖阶段就失败，不能被 pydantic 宽松转换吞掉。"""
    with pytest.raises(PresetError) as excinfo:
        apply_overrides(base_data, {"fitting.tolerance_mm": "0.15"}, allow_unknown_keys=False)
    message = str(excinfo.value)
    assert "fitting.tolerance_mm" in message
    assert "该字段接受" in message and "float" in message


def test_bool_override_with_number_rejected(base_data: dict) -> None:
    with pytest.raises(PresetError) as excinfo:
        apply_overrides(base_data, {"fitting.prefer_arcs": 1}, allow_unknown_keys=False)
    assert "该字段接受" in str(excinfo.value)


def test_int_field_with_float_rejected(base_data: dict) -> None:
    with pytest.raises(PresetError) as excinfo:
        apply_overrides(base_data, {"dxf.coordinate_decimals": 4.5}, allow_unknown_keys=False)
    assert "该字段接受" in str(excinfo.value)


def test_bool_union_field_accepts_bool(base_data: dict) -> None:
    """`background_flatten.enable` 的 base 是 "auto"（str），但字段接受 bool —— 不应误报。"""
    data = apply_overrides(
        base_data, {"preprocess.background_flatten.enable": False}, allow_unknown_keys=False
    )
    assert data["preprocess"]["background_flatten"]["enable"] is False


def test_int_accepted_where_float_expected(base_data: dict) -> None:
    data = apply_overrides(
        base_data, {"fitting.tolerance_mm": 1}, allow_unknown_keys=False
    )
    assert data["fitting"]["tolerance_mm"] == 1


def test_apply_overrides_does_not_mutate_input(base_data: dict) -> None:
    snapshot = copy.deepcopy(base_data)
    apply_overrides(base_data, {"fitting.tolerance_mm": 0.2}, allow_unknown_keys=False)
    assert base_data == snapshot


def test_apply_overrides_can_replace_lists(base_data: dict) -> None:
    data = apply_overrides(
        base_data, {"binarize.methods": ["otsu"]}, allow_unknown_keys=False
    )
    assert data["binarize"]["methods"] == ["otsu"]
    assert base_data["binarize"]["methods"] != ["otsu"]


def test_flatten_paths_reaches_leaf_keys(base_data: dict) -> None:
    paths = flatten_paths(base_data)
    assert "fitting.tolerance_mm" in paths
    assert "dxf.layers" in paths
    assert len(paths) > 100


def test_config_hash_ignores_key_order() -> None:
    a = {"b": 1, "a": {"y": 2, "x": [1, 2]}}
    b = {"a": {"x": [1, 2], "y": 2}, "b": 1}
    assert config_hash(a) == config_hash(b)
    assert config_hash(a) != config_hash({"b": 2, "a": {"y": 2, "x": [1, 2]}})
