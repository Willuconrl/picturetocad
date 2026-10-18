# src/artifacts.py
"""阶段产物契约：`metrics_<stage>.json` 与全局 `manifest.json`（SPEC §3.3）。

铁律
----
1. **可复现**：JSON 一律 `sort_keys=True` + `ensure_ascii=False` + LF + 末尾换行；
   同一份数据两次写出比特一致。
2. **不许编造**：拿不到的信息写 `null` 并**追加 WARN**，绝不填假值
   （例如没有 git 仓库时 `git_commit=null` + 一条警告）。
3. 时间戳是唯一允许非确定的东西（SPEC 明确要求记录起止时间），因此 manifest 不参与
   比特级比对；参与比对的是阶段产物与 DXF。
4. 阶段产物按**内容哈希**命名，支持增量重跑（`content_id`）。
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field

from src import __version__

STAGE_RE = re.compile(r"^s[0-8]$")


class ArtifactError(ValueError):
    """产物写出/读取失败。"""


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    path = Path(path)
    if not path.is_file():
        raise ArtifactError(f"文件不存在，无法计算哈希：{path}")
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def content_id(data: bytes, *, length: int = 16) -> str:
    """阶段产物命名用的内容哈希（默认取前 16 位十六进制）。"""
    if length < 8 or length > 64:
        raise ArtifactError(f"length 必须落在 [8, 64]，收到 {length}")
    return sha256_bytes(data)[:length]


def utc_now_iso() -> str:
    """带时区的 ISO-8601（秒级）。"""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    """确定性 JSON 写出（排序键、UTF-8、LF、末尾换行）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    path.write_bytes(text.encode("utf-8"))
    return path


def read_json(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise ArtifactError(f"JSON 不存在：{path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ArtifactError(f"{path} 不是合法 JSON：{exc}") from exc
    if not isinstance(data, dict):
        raise ArtifactError(f"{path} 顶层必须是对象，收到 {type(data).__name__}")
    return data


# ---------------------------------------------------------------------------
# git
# ---------------------------------------------------------------------------
def find_git_dir(start: Path) -> Path | None:
    """向上查找 `.git` 目录（不调用 git 子进程：沙箱下子进程管道不可用）。"""
    current = Path(start).resolve()
    for candidate in [current, *current.parents]:
        git = candidate / ".git"
        if git.is_dir():
            return git
        if candidate == candidate.parent:
            break
    return None


def git_commit(start: Path) -> str | None:
    """读取当前 HEAD 的提交号；不是 git 仓库/读不到时返回 None（调用方必须记 WARN）。"""
    git_dir = find_git_dir(start)
    if git_dir is None:
        return None
    head_file = git_dir / "HEAD"
    if not head_file.is_file():
        return None
    head = head_file.read_text(encoding="utf-8").strip()
    if head.startswith("ref:"):
        ref = head.split(" ", 1)[1].strip()
        ref_file = git_dir / ref
        if ref_file.is_file():
            value = ref_file.read_text(encoding="utf-8").strip()
            return value if re.fullmatch(r"[0-9a-f]{40}", value) else None
        packed = git_dir / "packed-refs"
        if packed.is_file():
            for line in packed.read_text(encoding="utf-8").splitlines():
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) == 2 and parts[1] == ref and re.fullmatch(r"[0-9a-f]{40}", parts[0]):
                    return parts[0]
        return None
    return head if re.fullmatch(r"[0-9a-f]{40}", head) else None


# ---------------------------------------------------------------------------
# metrics_<stage>.json
# ---------------------------------------------------------------------------
class StageMetrics(BaseModel):
    """单阶段指标。字段命名与 SPEC §3.3 对齐：counts / timing_ms / error_estimate / warnings。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    stage: str = Field(..., pattern=r"^s[0-8]$")
    counts: dict[str, int] = Field(default_factory=dict)
    timing_ms: float = Field(..., ge=0)
    error_estimate: dict[str, float | None] = Field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    details: dict[str, Any] = Field(default_factory=dict)

    @property
    def filename(self) -> str:
        return f"metrics_{self.stage}.json"

    def to_json(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "counts": dict(sorted(self.counts.items())),
            "timing_ms": round(float(self.timing_ms), 3),
            "error_estimate": dict(sorted(self.error_estimate.items())),
            "warnings": list(self.warnings),
            "details": self.details,
        }


def write_metrics(directory: Path, metrics: StageMetrics) -> Path:
    """写 `metrics_<stage>.json`（覆盖同名文件）。"""
    return write_json(Path(directory) / metrics.filename, metrics.to_json())


def read_metrics(path: Path) -> StageMetrics:
    data = read_json(path)
    missing = {"stage", "timing_ms"} - set(data)
    if missing:
        raise ArtifactError(f"{path} 缺少必需字段：{sorted(missing)}")
    return StageMetrics.model_validate(data)


# ---------------------------------------------------------------------------
# manifest.json
# ---------------------------------------------------------------------------
class EnvironmentInfo(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    python: str
    platform: str
    machine: str


class RunManifest(BaseModel):
    """一次运行的全局清单。`warnings` 必须如实记录缺失信息（如无 git 仓库）。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tool: str = "r2v-tool"
    version: str
    run_id: str
    started_at: str
    finished_at: str | None
    exit_code: int | None
    input_path: str
    input_sha256: str
    config_path: str
    config_hash: str
    preset: str | None
    git_commit: str | None
    environment: EnvironmentInfo
    stages: dict[str, str] = Field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "version": self.version,
            "run_id": self.run_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "exit_code": self.exit_code,
            "input_path": self.input_path,
            "input_sha256": self.input_sha256,
            "config_path": self.config_path,
            "config_hash": self.config_hash,
            "preset": self.preset,
            "git_commit": self.git_commit,
            "environment": self.environment.model_dump(),
            "stages": dict(sorted(self.stages.items())),
            "warnings": list(self.warnings),
        }


def environment_info() -> EnvironmentInfo:
    return EnvironmentInfo(
        python=platform.python_version(),
        platform=platform.platform(),
        machine=platform.machine(),
    )


def build_manifest(
    *,
    input_path: Path,
    config_path: Path,
    config_hash: str,
    preset: str | None,
    run_id: str,
    started_at: str,
    finished_at: str | None = None,
    exit_code: int | None = None,
    stages: Mapping[str, str] | None = None,
    extra_warnings: tuple[str, ...] = (),
) -> RunManifest:
    """构造 manifest；缺失的信息变成警告，不编造。"""
    warnings: list[str] = list(extra_warnings)
    commit = git_commit(Path(config_path).parent)
    if commit is None:
        warnings.append("未找到 git 提交号（不是 git 仓库或 HEAD 不可读）：git_commit=null")
    input_path = Path(input_path)
    if not input_path.is_file():
        raise ArtifactError(f"输入文件不存在：{input_path}")
    return RunManifest(
        version=__version__,
        run_id=run_id,
        started_at=started_at,
        finished_at=finished_at,
        exit_code=exit_code,
        input_path=str(input_path),
        input_sha256=sha256_file(input_path),
        config_path=str(config_path),
        config_hash=config_hash,
        preset=preset,
        git_commit=commit,
        environment=environment_info(),
        stages=dict(stages or {}),
        warnings=tuple(warnings),
    )


def write_manifest(run_dir: Path, manifest: RunManifest) -> Path:
    return write_json(Path(run_dir) / "manifest.json", manifest.to_json())


# ---------------------------------------------------------------------------
# 运行目录
# ---------------------------------------------------------------------------
def new_run_id(*, seed_prefix: str = "run") -> str:
    """运行 ID：UTC 时间戳 + 进程号，避免两次运行互相覆盖。"""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{seed_prefix}-{stamp}-{os.getpid()}"


def run_directory(outputs_root: Path, run_id: str, *, create: bool = True) -> Path:
    outputs_root = Path(outputs_root)
    if not re.fullmatch(r"[A-Za-z0-9._\-]+", run_id):
        raise ArtifactError(f"run_id 只允许字母数字与 ._-：{run_id!r}")
    path = outputs_root / run_id
    if create:
        (path / "debug").mkdir(parents=True, exist_ok=True)
    return path


__all__ = [
    "ArtifactError",
    "EnvironmentInfo",
    "RunManifest",
    "STAGE_RE",
    "StageMetrics",
    "build_manifest",
    "content_id",
    "environment_info",
    "find_git_dir",
    "git_commit",
    "new_run_id",
    "read_json",
    "read_metrics",
    "run_directory",
    "sha256_bytes",
    "sha256_file",
    "utc_now_iso",
    "write_json",
    "write_manifest",
    "write_metrics",
]
