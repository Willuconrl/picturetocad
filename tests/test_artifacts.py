# tests/test_artifacts.py
"""产物契约：metrics_<stage>.json / manifest.json 的确定性与诚实性（SPEC §3.3）。"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.artifacts import (
    ArtifactError,
    StageMetrics,
    build_manifest,
    content_id,
    environment_info,
    git_commit,
    new_run_id,
    read_json,
    read_metrics,
    run_directory,
    sha256_file,
    utc_now_iso,
    write_json,
    write_manifest,
    write_metrics,
)

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_INPUT = ROOT / "test_images" / "QDIZE9403.JPG"
CONFIG = ROOT / "config.yaml"


def test_write_json_is_deterministic(tmp_path: Path) -> None:
    payload = {"b": [1, 2], "a": {"z": "中文", "y": None}}
    first = write_json(tmp_path / "a.json", payload)
    bytes_a = first.read_bytes()
    second = write_json(tmp_path / "a.json", payload)
    assert bytes_a == second.read_bytes()
    assert bytes_a.endswith(b"\n")
    assert b"\r\n" not in bytes_a
    # 键排序：a 在 b 之前
    assert bytes_a.index(b'"a"') < bytes_a.index(b'"b"')


def test_metrics_roundtrip(tmp_path: Path) -> None:
    metrics = StageMetrics(
        stage="s3",
        counts={"nodes": 12, "edges": 15},
        timing_ms=1.23456,
        error_estimate={"endpoint_px": 0.4, "rms_mm": None},
        warnings=("WARN demo",),
        details={"seed": 42},
    )
    path = write_metrics(tmp_path, metrics)
    assert path.name == "metrics_s3.json"
    loaded = read_metrics(path)
    assert loaded.counts == {"nodes": 12, "edges": 15}
    assert loaded.timing_ms == pytest.approx(1.235)  # 保留 3 位小数
    assert loaded.warnings == ("WARN demo",)


def test_metrics_rejects_bad_stage() -> None:
    with pytest.raises(ValueError):
        StageMetrics(stage="s9", timing_ms=1.0)
    with pytest.raises(ValueError):
        StageMetrics(stage="stage3", timing_ms=1.0)


def test_metrics_rejects_negative_timing() -> None:
    with pytest.raises(ValueError):
        StageMetrics(stage="s2", timing_ms=-1.0)


def test_read_metrics_missing_field(tmp_path: Path) -> None:
    path = tmp_path / "metrics_s2.json"
    path.write_text('{"stage": "s2"}', encoding="utf-8")
    with pytest.raises(ArtifactError, match="缺少必需字段"):
        read_metrics(path)


def test_sha256_file_matches_manual(tmp_path: Path) -> None:
    import hashlib

    path = tmp_path / "x.bin"
    path.write_bytes(b"abc")
    assert sha256_file(path) == hashlib.sha256(b"abc").hexdigest()
    with pytest.raises(ArtifactError, match="不存在"):
        sha256_file(tmp_path / "nope.bin")


def test_content_id_lengths() -> None:
    assert len(content_id(b"data", length=16)) == 16
    assert content_id(b"data", length=16) != content_id(b"other", length=16)
    with pytest.raises(ArtifactError, match="length"):
        content_id(b"data", length=4)


def test_git_commit_reads_repository_head() -> None:
    commit = git_commit(ROOT)
    assert commit is not None
    assert len(commit) == 40
    assert all(c in "0123456789abcdef" for c in commit)


def test_git_commit_none_outside_repo(tmp_path: Path) -> None:
    assert git_commit(tmp_path) is None


def test_build_manifest_records_hashes_and_commit() -> None:
    manifest = build_manifest(
        input_path=SAMPLE_INPUT,
        config_path=CONFIG,
        config_hash="deadbeef",
        preset="mechanical",
        run_id=new_run_id(),
        started_at=utc_now_iso(),
        finished_at=utc_now_iso(),
        exit_code=0,
        stages={"s0": "abc123"},
    )
    assert manifest.input_sha256 == sha256_file(SAMPLE_INPUT)
    assert len(manifest.input_sha256) == 64
    assert manifest.config_hash == "deadbeef"
    assert manifest.preset == "mechanical"
    assert manifest.git_commit is not None
    assert manifest.stages == {"s0": "abc123"}
    assert manifest.warnings == ()


def test_manifest_warns_when_git_missing(tmp_path: Path) -> None:
    fake_config = tmp_path / "config.yaml"
    fake_config.write_text("meta: {}\n", encoding="utf-8")
    manifest = build_manifest(
        input_path=SAMPLE_INPUT,
        config_path=fake_config,
        config_hash="x",
        preset=None,
        run_id="run-test",
        started_at=utc_now_iso(),
    )
    assert manifest.git_commit is None
    assert any("git" in w for w in manifest.warnings), manifest.warnings


def test_manifest_missing_input_rejected(tmp_path: Path) -> None:
    with pytest.raises(ArtifactError, match="输入文件不存在"):
        build_manifest(
            input_path=tmp_path / "nope.png",
            config_path=CONFIG,
            config_hash="x",
            preset=None,
            run_id="r",
            started_at=utc_now_iso(),
        )


def test_manifest_roundtrip(tmp_path: Path) -> None:
    manifest = build_manifest(
        input_path=SAMPLE_INPUT,
        config_path=CONFIG,
        config_hash="cafe",
        preset=None,
        run_id="run-1",
        started_at=utc_now_iso(),
        exit_code=0,
    )
    path = write_manifest(tmp_path, manifest)
    data = read_json(path)
    assert data["run_id"] == "run-1"
    assert data["tool"] == "r2v-tool"
    assert data["config_hash"] == "cafe"
    assert data["environment"]["python"]


def test_environment_info_fields() -> None:
    env = environment_info()
    assert env.python.count(".") >= 1
    assert env.platform
    assert env.machine


def test_run_directory_creates_debug(tmp_path: Path) -> None:
    run_dir = run_directory(tmp_path, "run-abc")
    assert (run_dir / "debug").is_dir()
    with pytest.raises(ArtifactError, match="run_id"):
        run_directory(tmp_path, "bad/../id", create=False)
