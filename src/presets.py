# src/presets.py
"""图纸预设（`.dsh/skills/r2v-domain/presets.yaml`）的读取与点路径覆盖。

契约（与 presets.yaml 头部说明一致）
------------------------------------
1. 预设只写**差量覆盖**：`overrides` 的键是点路径（如 `fitting.tolerance_mm`），
   值必须是 base 里同名路径的类型兼容值；未出现的键一律沿用 base。
2. 覆盖后**必须重新校验**（`presets.validate_after_apply`）：类型不兼容、越界、
   跨字段约束冲突都要在启动时炸掉，不许带着非法参数跑。
3. 覆盖到 base 不存在的键 = 硬失败（`presets.allow_unknown_keys=false`），
   并给出最接近的候选，防止 `tolerance_mm` 被拼成 `tolerence_mm` 后静默失效。
4. 预设文件路径由 `config.yaml: presets.file` 给出；相对路径先相对 config 所在目录找，
   找不到再向上逐级找（仓库把 skill 文件放在 `.dsh/skills/` 下，与 config 不同级）。
5. 有效参数必须可哈希、可复现：`config_hash` 用规范化 JSON（键排序、无空白）算 SHA-256。
"""

from __future__ import annotations

import copy
import difflib
import hashlib
import json
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Union, get_args, get_origin

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.config import ConfigError, R2VConfig, load_yaml_mapping, make_config

MAX_PARENT_SEARCH = 6


class PresetError(ValueError):
    """presets.yaml 读取/覆盖失败。"""


class PresetEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    description: str = Field(..., min_length=1)
    overrides: dict[str, Any] = Field(..., min_length=1)


class PresetFile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(..., min_length=1)
    version: str = Field(..., pattern=r"^\d+\.\d+$")
    base: str = Field(..., min_length=1)
    presets: dict[str, PresetEntry] = Field(..., min_length=1)


# ---------------------------------------------------------------------------
# 读取与解析
# ---------------------------------------------------------------------------
def load_presets(path: Path) -> PresetFile:
    path = Path(path)
    if not path.is_file():
        raise PresetError(f"presets 文件不存在：{path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise PresetError(f"{path.name} YAML 解析失败：{exc}") from exc
    if not isinstance(data, dict):
        raise PresetError(f"{path.name} 顶层必须是映射，收到 {type(data).__name__}")
    try:
        return PresetFile.model_validate(data)
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(str(p) for p in err['loc']) or '<root>'}: {err['msg']}"
            for err in exc.errors()
        )
        raise PresetError(f"{path.name} 校验失败：{details}") from exc


def resolve_presets_path(config_path: Path, presets_file: str) -> Path:
    """把 `presets.file` 解析成真实路径：先相对 config 目录，再逐级向父目录找。"""
    config_path = Path(config_path).resolve()
    candidate = Path(presets_file)
    if candidate.is_absolute():
        if not candidate.is_file():
            raise PresetError(f"presets.file 指向的文件不存在：{candidate}")
        return candidate

    tried: list[Path] = []
    for base in [config_path.parent, *config_path.parents]:
        path = (base / candidate)
        tried.append(path)
        if path.is_file():
            return path.resolve()
        if base == base.parent:  # 到根就停
            break
    raise PresetError(
        f"找不到 presets.file={presets_file!r}；已尝试：\n  "
        + "\n  ".join(str(p) for p in tried)
    )


# ---------------------------------------------------------------------------
# 点路径覆盖
# ---------------------------------------------------------------------------
def flatten_paths(data: Mapping[str, Any], prefix: str = "") -> list[str]:
    """把嵌套 mapping 展平成点路径列表（只展开 dict）。"""
    out: list[str] = []
    for key, value in data.items():
        path = f"{prefix}{key}"
        if isinstance(value, Mapping):
            out.extend(flatten_paths(value, prefix=f"{path}."))
        else:
            out.append(path)
    return out


def _suggestion(path: str, known: list[str]) -> str:
    close = difflib.get_close_matches(path, known, n=3, cutoff=0.6)
    return f"；最接近的候选：{close}" if close else ""


_TYPE_NAMES = {
    bool: "bool",
    int: "int",
    float: "float",
    str: "str",
    list: "list",
    dict: "dict",
}

_UNION_ORIGINS = {Union, getattr(types, "UnionType", object())}


def _type_name(value: object) -> str:
    if value is None:
        return "none"
    for py_type in (bool, int, float, str, list, dict):
        if isinstance(value, py_type):
            return _TYPE_NAMES[py_type]
    return type(value).__name__


def _model_from_annotation(ann: object) -> type[BaseModel] | None:
    """从注解里取出嵌套的 pydantic 模型（穿过 Optional/Union）。"""
    origin = get_origin(ann)
    if origin in _UNION_ORIGINS:
        for arg in get_args(ann):
            if isinstance(arg, type) and issubclass(arg, BaseModel):
                return arg
        return None
    if isinstance(ann, type) and issubclass(ann, BaseModel):
        return ann
    return None


def _annotation_type_names(ann: object) -> set[str] | None:
    """把字段注解答成可接受的 Python 类型名集合；解不出返回 None。"""
    if ann is None:
        return None
    origin = get_origin(ann)
    if origin in _UNION_ORIGINS:
        out: set[str] = set()
        for arg in get_args(ann):
            sub = _annotation_type_names(arg)
            if sub is None:
                return None
            out |= sub
        return out
    if origin is Literal:
        return {_type_name(v) for v in get_args(ann)} or None
    if origin in (list, set, frozenset, tuple):
        return {"list"}
    if origin is dict:
        return {"dict"}
    if ann is type(None):
        return {"none"}
    if isinstance(ann, type):
        if issubclass(ann, BaseModel):
            return {"dict"}
        for py_type in (bool, int, float, str, list, tuple, dict):
            if issubclass(ann, py_type):
                return {"list" if py_type is tuple else _TYPE_NAMES[py_type]}
    return None


def resolve_annotation(dotted: str) -> object | None:
    """按点路径从 `R2VConfig` 里取出字段注解；路径不存在返回 None。"""
    model: type[BaseModel] | None = R2VConfig
    parts = dotted.split(".")
    for index, part in enumerate(parts):
        if model is None or part not in model.model_fields:
            return None
        ann = model.model_fields[part].annotation
        if index == len(parts) - 1:
            return ann
        model = _model_from_annotation(ann)
    return None


def check_type_compat(dotted: str, value: object, base_value: object = None) -> None:
    """覆盖值必须与**字段注解**兼容（防 `0.15` 被写成 `"0.15"`）。

    优先用 schema 注解判断，因为注解比"当前取值"更准确：
    `background_flatten.enable` 的 base 是 `"auto"`（str），但字段接受 `bool | "auto"`，
    机械图 preset 里写 `false` 是合法的。
    pydantic 默认宽松模式会把 `"0.15"` 悄悄转成 0.15，这类"看起来成功"的漂移必须拦。
    """
    allowed = _annotation_type_names(resolve_annotation(dotted))
    got = _type_name(value)
    if allowed is not None:
        if got not in allowed and not (got == "int" and "float" in allowed):
            raise PresetError(
                f"{dotted}: 覆盖值类型是 {got}（{value!r}），但该字段接受 {sorted(allowed)}；"
                "不要用字符串代替数字/布尔"
            )
        return
    # 注解解析不出来时退回"与 base 当前值同类型"的保守判断
    if base_value is None:
        return
    if _type_name(base_value) != got and not (_type_name(base_value) == "float" and got == "int"):
        raise PresetError(
            f"{dotted}: base 是 {_type_name(base_value)}，覆盖值却是 {got}（{value!r}）"
        )


def apply_overrides(
    base: Mapping[str, Any],
    overrides: Mapping[str, Any],
    *,
    allow_unknown_keys: bool,
    source: str = "presets",
) -> dict[str, Any]:
    """把点路径覆盖 patch 到 base 的深拷贝上；返回新 dict（不改入参）。"""
    if not isinstance(base, Mapping):
        raise PresetError(f"{source}: base 必须是映射，收到 {type(base).__name__}")
    merged = copy.deepcopy(dict(base))
    known = flatten_paths(merged)

    for dotted, value in overrides.items():
        if not isinstance(dotted, str) or "." not in dotted:
            raise PresetError(
                f"{source}: 覆盖键必须是点路径（如 fitting.tolerance_mm），收到 {dotted!r}"
            )
        parts = dotted.split(".")
        cursor: Any = merged
        for i, part in enumerate(parts[:-1]):
            if not isinstance(cursor, dict):
                raise PresetError(
                    f"{source}: 路径 {dotted!r} 在第 {i + 1} 段 {part!r} 处遇到非映射值"
                )
            if part not in cursor:
                msg = f"{source}: 路径 {dotted!r} 的中间段 {part!r} 在 base 中不存在"
                if not allow_unknown_keys:
                    msg += _suggestion(dotted, known)
                raise PresetError(msg)
            cursor = cursor[part]
        leaf = parts[-1]
        if not isinstance(cursor, dict):
            raise PresetError(f"{source}: 路径 {dotted!r} 的父级不是映射")
        if leaf not in cursor and not allow_unknown_keys:
            raise PresetError(
                f"{source}: 覆盖键 {dotted!r} 在 base 中不存在"
                f"（allow_unknown_keys=false，禁止拼写漂移）{_suggestion(dotted, known)}"
            )
        if leaf in cursor:
            check_type_compat(dotted, value, cursor[leaf])
        cursor[leaf] = copy.deepcopy(value)
    return merged


# ---------------------------------------------------------------------------
# 有效参数
# ---------------------------------------------------------------------------
def config_hash(data: Mapping[str, Any]) -> str:
    """规范化 JSON 的 SHA-256：键排序、紧凑分隔符、保留非 ASCII（可复现）。"""
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class EffectiveConfig:
    """base ⊕ preset 之后的最终参数。"""

    config: R2VConfig
    data: dict[str, Any]
    preset: str | None
    config_hash: str
    overridden_keys: tuple[str, ...]
    presets_path: Path | None

    def to_json(self) -> dict[str, Any]:
        return {
            "preset": self.preset,
            "config_hash": self.config_hash,
            "overridden_keys": list(self.overridden_keys),
            "presets_path": None if self.presets_path is None else str(self.presets_path),
        }


def build_effective(
    config_path: Path,
    preset: str | None = None,
    *,
    presets_path: Path | None = None,
) -> EffectiveConfig:
    """读 config.yaml → （可选）套 preset → 重新校验 → 算有效参数哈希。

    `preset=None` 表示用 `presets.default`（若也是 null 则纯 base）。
    """
    config_path = Path(config_path)
    base_data = load_yaml_mapping(config_path)
    base_config = make_config(base_data, source=str(config_path))

    resolved_presets: Path | None = None
    name = preset if preset is not None else base_config.presets.default

    if name is None:
        merged = copy.deepcopy(base_data)
        effective_config = base_config
        overridden: tuple[str, ...] = ()
    else:
        resolved_presets = (
            Path(presets_path)
            if presets_path is not None
            else resolve_presets_path(config_path, base_config.presets.file)
        )
        preset_file = load_presets(resolved_presets)
        if name not in preset_file.presets:
            raise PresetError(
                f"未知 preset {name!r}；{resolved_presets.name} 里有 "
                f"{sorted(preset_file.presets)}"
            )
        entry = preset_file.presets[name]
        merged = apply_overrides(
            base_data,
            entry.overrides,
            allow_unknown_keys=base_config.presets.allow_unknown_keys,
            source=f"{resolved_presets.name}:{name}",
        )
        overridden = tuple(sorted(entry.overrides))
        if base_config.presets.validate_after_apply:
            try:
                effective_config = make_config(merged, source=f"{config_path.name} + preset {name}")
            except ConfigError as exc:
                raise PresetError(f"preset {name!r} 覆盖后的参数非法：\n{exc}") from exc
        else:
            effective_config = base_config

    return EffectiveConfig(
        config=effective_config,
        data=merged,
        preset=name,
        config_hash=config_hash(merged),
        overridden_keys=overridden,
        presets_path=resolved_presets,
    )


__all__ = [
    "EffectiveConfig",
    "PresetEntry",
    "PresetError",
    "PresetFile",
    "apply_overrides",
    "build_effective",
    "check_type_compat",
    "config_hash",
    "flatten_paths",
    "load_presets",
    "resolve_annotation",
    "resolve_presets_path",
]
