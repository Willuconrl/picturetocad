# tests/test_cli.py
"""`r2v` CLI 骨架：退出码、有效参数哈希稳定性、能力边界提示（M0-b 验收判据）。"""

from __future__ import annotations

from pathlib import Path

import pytest

from src import CAPABILITY_NOTICE
from src.cli import EXIT_CONFIG_ERROR, EXIT_OK, main

ROOT = Path(__file__).resolve().parents[1]
CONFIG = str(ROOT / "config.yaml")


def test_check_config_ok(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["--config", CONFIG, "--check-config"])
    out = capsys.readouterr().out
    assert code == EXIT_OK
    assert out.startswith("OK config_hash=")
    assert "sections=12" in out
    assert "layers=10" in out


@pytest.mark.parametrize("preset", ["topo-map", "blueprint", "mechanical", "sketch"])
def test_check_config_with_each_preset(preset: str, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["--config", CONFIG, "--check-config", "--preset", preset])
    out = capsys.readouterr().out
    assert code == EXIT_OK
    assert f"preset={preset}" in out


def test_print_effective_is_byte_identical_across_runs(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["--config", CONFIG, "--print-effective", "--preset", "mechanical"]) == EXIT_OK
    first = capsys.readouterr().out
    assert main(["--config", CONFIG, "--print-effective", "--preset", "mechanical"]) == EXIT_OK
    second = capsys.readouterr().out
    assert first == second
    assert '"config_hash"' in first
    assert '"preset": "mechanical"' in first


def test_print_effective_differs_between_presets(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--config", CONFIG, "--print-effective", "--preset", "blueprint"]) == EXIT_OK
    blueprint = capsys.readouterr().out
    assert main(["--config", CONFIG, "--print-effective", "--preset", "mechanical"]) == EXIT_OK
    mechanical = capsys.readouterr().out
    assert blueprint != mechanical


def test_unknown_preset_exits_with_config_error(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["--config", CONFIG, "--check-config", "--preset", "does-not-exist"])
    captured = capsys.readouterr()
    assert code == EXIT_CONFIG_ERROR
    assert "ERROR PresetError" in captured.err


def test_missing_config_exits_with_config_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["--config", str(tmp_path / "nope.yaml"), "--check-config"])
    captured = capsys.readouterr()
    assert code == EXIT_CONFIG_ERROR
    assert "ERROR ConfigError" in captured.err


def test_no_action_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--config", CONFIG]) == EXIT_CONFIG_ERROR
    out = capsys.readouterr().out
    assert "usage: r2v" in out


def test_help_contains_capability_notice(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "能力边界" in out
    assert CAPABILITY_NOTICE.split("。")[0] in out


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    assert "r2v-tool" in capsys.readouterr().out


def test_list_presets(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--config", CONFIG, "--list-presets"]) == EXIT_OK
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
    names = {ln.split("\t")[0] for ln in lines}
    assert names == {"topo-map", "blueprint", "mechanical", "sketch"}
