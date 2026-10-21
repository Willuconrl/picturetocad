#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R2V 文档 / 配置契约静态校验（零依赖）。

设计目的：把「数值漂移」和「文档格式漂移」变成可自动发现的失败，而不是靠人记得同步。

检查项：
  1. config.yaml 与 presets.yaml 可解析（自带受限 YAML 子集解析器；装了 PyYAML 则优先用它）
  2. 每个 preset 的 overrides 键路径在 config.yaml base 中存在（防拼写漂移）
  3. 覆盖值与 base 的类型兼容（防 0.15 写成 "0.15"）
  4. schema 版本号一致（config.meta.schema_version == presets.version）
  5. dxf.layers 与 dxf_rules.md 的图层表逐名对齐
  6. 实体类型表与 config 的完整圆覆盖角约定自洽
  7. 文档不得残留 Markdown 围栏、不得有裸数字容差、不得缺尾换行、不得写到一半
  8. 文档引用的**仓库内文件路径**（含 `/` 的引用）必须存在；运行时产物（metrics_*.json 等）不检查
  9. SKILL.md 与 SPEC.md 的误差预算条目数一致

用法：
    python scripts/check_contracts.py [--root <r2v-tool 根目录>]
退出码：0 = 全部通过（可能有 WARN）；1 = 存在 ERROR。
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

SKILL_REL = Path(".dsh/skills/r2v-domain")
DOCS = ("SKILL.md", "coords.md", "dxf_rules.md")
# 只校验带路径分隔符的引用：纯文件名多为运行时产物（metrics_s3.json 等）
REF_RE = re.compile(r"`([A-Za-z0-9_.-]+/[A-Za-z0-9_./-]+\.(?:yaml|yml|json|md|py|dxf|png|txt))`")
BARE_TOL_RE = re.compile(r"容差[^\n|]{0,20}?\d+(?:\.\d+)?\s*px")
# 已被规范要求、但仓库里还没写的文件：缺失算 WARN 不算 ERROR（src/ 目前为空）
PLANNED_PATHS = {"r2v-tool/src/config.py", "src/config.py"}

problems: list[tuple[str, str]] = []


def err(check: str, message: str) -> None:
    problems.append(("ERROR", f"[{check}] {message}"))


def warn(check: str, message: str) -> None:
    problems.append(("WARN", f"[{check}] {message}"))


# --------------------------------------------------------------------------- #
# 受限 YAML 解析器：支持本仓库用到的子集
#   - 注释、空行、`---` 文档起点
#   - 块映射 + 同级 `- ` 序列项
#   - 标量：int / float / true / false / null / '...' / "..." / 裸串
#   - 行内流式 [a, b] 与 {k: v, k: v}
# 不支持锚点、多行标量、复杂键等——本仓库不需要；遇到无法归约的行会显式报错。
# --------------------------------------------------------------------------- #
class YamlSubsetError(ValueError):
    pass


def _parse_scalar(text: str):
    value = text.strip()
    if not value:
        return None
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    lowered = value.lower()
    if lowered in ("true", "yes", "on"):
        return True
    if lowered in ("false", "no", "off"):
        return False
    if lowered in ("null", "~", ""):
        return None
    if value.startswith("[") and value.endswith("]"):
        body = value[1:-1].strip()
        return [] if not body else [_parse_scalar(part) for part in _split_inline(body)]
    if value.startswith("{") and value.endswith("}"):
        body = value[1:-1].strip()
        out = {}
        if body:
            for part in _split_inline(body):
                key, _, raw = part.partition(":")
                out[key.strip().strip("\"'")] = _parse_scalar(raw)
        return out
    if re.fullmatch(r"[+-]?\d+", value):
        return int(value)
    if re.fullmatch(r"[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?", value):
        return float(value)
    return value


def _split_inline(body: str) -> list[str]:
    parts, depth, current = [], 0, []
    for char in body:
        if char in "[{":
            depth += 1
        elif char in "]}":
            depth -= 1
        if char == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(char)
    if current:
        parts.append("".join(current))
    return [part.strip() for part in parts if part.strip()]


def _indent_of(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _load_block(lines: list[tuple[int, str]], index: int, indent: int):
    """从 (缩进, 去注释文本) 列表的第 index 行开始，按 indent 解析一个块。"""
    if index >= len(lines):
        return {}, index
    if lines[index][1].startswith("- "):
        seq = []
        while index < len(lines) and lines[index][0] == indent and lines[index][1].startswith("- "):
            item = lines[index][1][2:].strip()
            index += 1
            if item == "":
                child, index = _load_block(lines, index, indent + 2)
                seq.append(child)
            elif ":" in item and not item.startswith(("[", "{")):
                key, _, raw = item.partition(":")
                mapping = {key.strip(): _parse_scalar(raw)}
                child, index = _load_block(lines, index, indent + 2)
                if isinstance(child, dict):
                    mapping.update(child)
                seq.append(mapping)
            else:
                seq.append(_parse_scalar(item))
        return seq, index

    mapping: dict = {}
    while index < len(lines) and lines[index][0] == indent:
        text = lines[index][1]
        if text.startswith("- "):
            break
        key, sep, raw = text.partition(":")
        if not sep:
            raise YamlSubsetError(f"第 {index + 1} 行不是 key: value —— {text!r}")
        key = key.strip().strip("\"'")
        index += 1
        if raw.strip():
            mapping[key] = _parse_scalar(raw)
        elif index < len(lines) and lines[index][0] > indent:
            # 子块缩进量由下一行决定（YAML 允许任意缩进宽度，含跳级）
            child, index = _load_block(lines, index, lines[index][0])
            mapping[key] = child
        else:
            mapping[key] = {}
    return mapping, index


def parse_yaml_subset(text: str) -> dict:
    lines: list[tuple[int, str]] = []
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        stripped = raw.rstrip()
        if stripped.strip() == "---":
            continue
        indent = _indent_of(stripped)
        lines.append((indent, _strip_comment(stripped.strip())))
    if not lines:
        return {}
    base_indent = lines[0][0]
    data, consumed = _load_block(lines, 0, base_indent)
    if consumed != len(lines):
        raise YamlSubsetError(f"有 {len(lines) - consumed} 行未被解析（缩进不一致？）")
    if not isinstance(data, dict):
        raise YamlSubsetError("顶层不是映射")
    return data


def _strip_comment(text: str) -> str:
    out, quote = [], None
    for index, char in enumerate(text):
        if quote:
            out.append(char)
            if char == quote:
                quote = None
            continue
        if char in "\"'":
            quote = char
            out.append(char)
            continue
        if char == "#" and (index == 0 or text[index - 1] in " \t"):
            break
        out.append(char)
    return "".join(out).rstrip()


def load_config(path: Path, check: str) -> dict | None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        err(check, f"无法读取 {path}: {exc}")
        return None
    try:
        import yaml  # noqa: PLC0415

        data = yaml.safe_load(text)
    except ImportError:
        try:
            data = parse_yaml_subset(text)
        except YamlSubsetError as exc:
            err(check, f"{path} 无法解析（受限解析器）: {exc}")
            return None
    except Exception as exc:  # yaml.YAMLError
        err(check, f"{path} 不是合法 YAML: {str(exc).splitlines()[0]}")
        return None
    if not isinstance(data, dict):
        err(check, f"{path} 顶层必须是映射，实际是 {type(data).__name__}")
        return None
    return data


def flatten(node, prefix: str = "") -> dict:
    out: dict = {}
    if isinstance(node, dict):
        for key, value in node.items():
            out.update(flatten(value, f"{prefix}.{key}" if prefix else str(key)))
    else:
        out[prefix] = node
    return out


def type_ok(base, override) -> bool:
    # 三态开关（true/false/"auto"）的 base 是字符串，允许 preset 覆盖为布尔
    if isinstance(base, str) and base == "auto" and isinstance(override, bool):
        return True
    if isinstance(base, bool) or isinstance(override, bool):
        return isinstance(base, bool) and isinstance(override, bool)
    if isinstance(base, (int, float)):
        return isinstance(override, (int, float))
    if isinstance(base, list):
        return isinstance(override, list)
    if isinstance(base, str):
        return isinstance(override, str)
    return type(base) is type(override)


def check_presets(base_cfg: dict, presets_cfg: dict) -> None:
    base = flatten(base_cfg)
    presets = presets_cfg.get("presets")
    if not isinstance(presets, dict) or not presets:
        err("presets", "presets.yaml 缺少非空的 presets 映射")
        return
    for preset_name, preset in presets.items():
        where = f"preset '{preset_name}'"
        if not isinstance(preset, dict):
            err("presets", f"{where} 必须是映射")
            continue
        if "overrides" not in preset:
            err("presets", f"{where} 缺少 overrides（未覆盖项一律沿用 base，不重复声明）")
            continue
        if not isinstance(preset.get("description"), str):
            warn("presets", f"{where} 缺少 description")
        overrides = preset["overrides"] or {}
        if not isinstance(overrides, dict):
            err("presets", f"{where}.overrides 必须是映射")
            continue
        for key, value in overrides.items():
            key = str(key)
            if key not in base:
                err("presets", f"{where} 的覆盖键 {key!r} 在 config.yaml base 中不存在")
                continue
            if not type_ok(base[key], value):
                err(
                    "presets",
                    f"{where} 的 {key} 类型不符：base={type(base[key]).__name__} "
                    f"override={type(value).__name__}",
                )


def check_versions(base_cfg: dict, presets_cfg: dict) -> None:
    cfg_version = str((base_cfg.get("meta") or {}).get("schema_version", "")).strip()
    pre_version = str(presets_cfg.get("version", "")).strip()
    if not cfg_version:
        err("version", "config.yaml 缺少 meta.schema_version")
    if cfg_version != pre_version:
        err(
            "version",
            f"版本号不一致：config.meta.schema_version={cfg_version!r} "
            f"vs presets.version={pre_version!r}",
        )


def check_layers(base_cfg: dict, dxf_doc: Path) -> None:
    layers = (base_cfg.get("dxf") or {}).get("layers")
    if not isinstance(layers, list) or not layers:
        err("layers", "config.yaml 缺少 dxf.layers")
        return
    names = [str(layer.get("name", "")) for layer in layers if isinstance(layer, dict)]
    text = dxf_doc.read_text(encoding="utf-8")
    for name in names:
        if f"`{name}`" not in text:
            err("layers", f"图层 {name!r} 在 dxf_rules.md 的图层表中缺失")
    table_names = set(re.findall(r"^\|\s*`([A-Z]+(?:_[A-Z]+)*(?:\.\d+)?)`\s*\|", text, re.M))
    # 只把「看起来像图层名」的残留项当成错误：图层是复数/带线宽后缀，实体是单数
    for name in sorted(table_names - set(names) - {"DEBUG"}):
        if name.endswith("S") or re.search(r"\d", name):
            err("layers", f"dxf_rules.md 列出的图层 {name!r} 在 config.yaml 的 dxf.layers 中不存在")


def _parse_layer_table(text: str) -> dict:
    """解析 Markdown 图层表：| `NAME` | 内容 | 颜色索引 | 线型 |"""
    out: dict = {}
    for line in text.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 4 or not re.fullmatch(r"`[A-Z][A-Z_]*`", cells[0]):
            continue
        color = re.search(r"\d+", cells[2])
        if color is None:
            continue
        out[cells[0].strip("`")] = {"color": int(color.group()), "linetype": cells[3]}
    return out


def check_code_layers(base_cfg: dict, writer: Path, dxf_doc: Path) -> None:
    """代码 / 文档里的图层表必须与 config.yaml 的 dxf.layers 一致。

    代码里硬编码图层表本身就是漂移源（config 才是真相源）；这里至少保证
    名字、颜色索引、线型三者在「代码 / 文档 / config」之间不打架。
    """
    layers = (base_cfg.get("dxf") or {}).get("layers") or []
    if not isinstance(layers, list) or not layers:
        return
    cfg = {
        str(layer.get("name")): {
            "color": layer.get("color"),
            "linetype": layer.get("linetype", "CONTINUOUS"),
        }
        for layer in layers
        if isinstance(layer, dict)
    }
    compare = []

    if writer.is_file():
        text = writer.read_text(encoding="utf-8")
        code = {}
        for match in re.finditer(
            r'\(\s*"([A-Z][A-Z_]*(?:\.\d+)?)"\s*,\s*(\d+)\s*,\s*"([A-Z]+)"', text
        ):
            code[match.group(1)] = {"color": int(match.group(2)), "linetype": match.group(3)}
        if code:
            compare.append(("dxf_writer.py 的 _LAYER_DEFS", code))
        decimals_pattern = r"COORD_DECIMALS\s*=\s*(\d+)"
        found = re.search(decimals_pattern, text)
        if not found:
            warn("code", f"{writer.name} 未声明 COORD_DECIMALS，坐标精度无法核对")
        else:
            expect = (base_cfg.get("dxf") or {}).get("coordinate_decimals")
            if expect is not None and int(found.group(1)) != int(expect):
                err(
                    "code",
                    f"{writer.name} 的 COORD_DECIMALS={found.group(1)} 与 "
                    f"config 的 dxf.coordinate_decimals={expect} 不一致",
                )

    if dxf_doc.is_file():
        table = _parse_layer_table(dxf_doc.read_text(encoding="utf-8"))
        if table:
            compare.append(("dxf_rules.md 的图层表", table))

    for source, table in compare:
        for name, spec in sorted(table.items()):
            if name == "DEBUG" or name not in cfg:
                continue  # DEBUG 的颜色由 dxf.debug_layer_color 控制，单独核对
            if spec["color"] != cfg[name]["color"]:
                err(
                    "layers",
                    f"{source} 的 {name} 颜色={spec['color']} 与 config 的 {cfg[name]['color']} 不一致",
                )
            if spec["linetype"] != cfg[name]["linetype"]:
                err(
                    "layers",
                    f"{source} 的 {name} 线型={spec['linetype']} 与 config 的 "
                    f"{cfg[name]['linetype']} 不一致",
                )
        for name in sorted(set(table) - set(cfg) - {"DEBUG"}):
            err("layers", f"{source} 多出 config.yaml 没有的图层 {name!r}")
    debug_color = (base_cfg.get("dxf") or {}).get("debug_layer_color")
    for source, table in compare:
        if source.startswith("dxf_writer") and "DEBUG" in table and debug_color is not None:
            if table["DEBUG"]["color"] != int(debug_color):
                err(
                    "layers",
                    f"{source} 的 DEBUG 颜色={table['DEBUG']['color']} 与 "
                    f"dxf.debug_layer_color={debug_color} 不一致",
                )


def check_entities(base_cfg: dict, dxf_doc: Path) -> None:
    text = dxf_doc.read_text(encoding="utf-8")
    for entity in ("LINE", "LWPOLYLINE", "ARC", "CIRCLE", "TEXT", "MTEXT", "POLYLINE"):
        if f"`{entity}`" not in text:
            warn("entities", f"实体 {entity} 未在 dxf_rules.md 中说明")
    coverage = (base_cfg.get("fitting") or {}).get("full_circle_min_coverage_deg")
    if coverage is None:
        err("entities", "config.yaml 缺少 fitting.full_circle_min_coverage_deg")
    elif str(coverage) not in text and "`fitting.full_circle_min_coverage_deg`" not in text:
        warn(
            "entities",
            f"dxf_rules.md 未引用 config 的完整圆覆盖角阈值 {coverage}"
            "（应按键名 `fitting.full_circle_min_coverage_deg` 引用，避免数字漂移）",
        )


def check_doc_hygiene(skill_dir: Path) -> None:
    for name in DOCS:
        path = skill_dir / name
        if not path.is_file():
            err("format", f"缺少文档 {name}")
            continue
        text = path.read_text(encoding="utf-8")
        # 只打击「整篇被围栏包裹」的粘贴稿；正文里的代码块是正常内容
        lines = text.splitlines()
        without_fences = [ln for ln in lines if not ln.strip().startswith("```")]
        if lines and without_fences and lines[0].strip().startswith("```"):
            err("format", f"{name} 以 Markdown 围栏开头——疑似整篇粘贴稿，应去掉外层围栏")
        if len(without_fences) < len(lines) - 4 and len(lines) > 0:
            warn("format", f"{name} 含 {len(lines) - len(without_fences)} 行围栏标记，确认不是粘贴稿")
        if text.startswith("\n"):
            warn("format", f"{name} 以空行开头")
        if not text.endswith("\n"):
            err("format", f"{name} 末尾缺少换行")
        content = [ln for ln in text.rstrip("\n").splitlines() if ln.strip()]
        if content and re.search(r"(：|\*\*|:)$", content[-1].strip()):
            err("format", f"{name} 以 {content[-1].strip()[:40]!r} 结尾，疑似写到一半")
        for lineno, line in enumerate(text.splitlines(), 1):
            if BARE_TOL_RE.search(line):
                warn(
                    "units",
                    f"{name}:{lineno} 出现形如容差的 px 数字；确认它是在定义容差"
                    "（应改为 mm 或线宽倍数）而不是在举例说明反例",
                )


def check_referenced_paths(root: Path, skill_dir: Path) -> None:
    """引用路径按「相对 skill、相对仓库根、相对仓库根的各级祖先」解析。

    skill 与文档可能位于仓库之外（如 project3/.dsh/skills 与 project3/r2v-tool），
    文档里的 `r2v-tool/config.yaml` 是相对共同祖先写的，因此必须向上找。
    """
    ancestors = [skill_dir, *skill_dir.parents, root, *root.parents]
    for name in DOCS:
        path = skill_dir / name
        if not path.is_file():
            continue
        for match in REF_RE.finditer(path.read_text(encoding="utf-8")):
            ref = match.group(1)
            found = any((base / ref).exists() for base in ancestors)
            if found:
                continue
            if ref in PLANNED_PATHS:
                warn("refs", f"{name} 引用的 {ref!r} 尚未实现（规范要求，仓库里还没有）")
            else:
                err("refs", f"{name} 引用的仓库路径 {ref!r} 不存在")


def table_rows(text: str, heading_re: str, stop_re: str) -> int:
    lines = text.splitlines()
    start = next((i for i, ln in enumerate(lines) if re.search(heading_re, ln)), None)
    if start is None:
        return -1
    rows = 0
    for line in lines[start + 1:]:
        if re.search(stop_re, line):
            break
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            if set(stripped) <= set("|-: "):
                continue
            rows += 1
    return max(rows - 1, 0)


def check_budget_rows(root: Path, skill_dir: Path) -> None:
    spec_path = root / "specs" / "SPEC.md"
    skill_path = skill_dir / "SKILL.md"
    if not spec_path.is_file() or not skill_path.is_file():
        warn("budget", "SPEC.md 或 SKILL.md 缺失，跳过误差预算条目比对")
        return
    spec_rows = table_rows(spec_path.read_text(encoding="utf-8"), r"###\s*1\.2", r"###\s*1\.3")
    skill_rows = table_rows(skill_path.read_text(encoding="utf-8"), r"##\s*误差预算", r"^##\s")
    if spec_rows < 0:
        err("budget", "SPEC.md 未找到 §1.2 误差预算表")
    elif skill_rows != spec_rows:
        err(
            "budget",
            f"误差预算条目数不一致：SPEC.md §1.2 有 {spec_rows} 行，SKILL.md 有 {skill_rows} 行",
        )


def check_licenses(root: Path) -> None:
    """docs/LICENSES.md 必须在、必须覆盖 requirements.txt 的每个依赖、UNKNOWN 必须进清单。"""
    check = "licenses"
    licenses = root / "docs" / "LICENSES.md"
    if not licenses.is_file():
        err(check, f"缺少 {licenses.relative_to(root).as_posix()}（SPEC §2 交付物：依赖名/版本/许可证/来源）")
        return

    text = licenses.read_text(encoding="utf-8")
    rows: dict[str, list[str]] = {}
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 4:
            continue
        name = cells[0].strip("`")
        if name in ("包", "") or set(name) <= {"-", ":"}:
            continue
        if name in rows:
            err(check, f"{name} 在许可证表里重复出现")
        rows[name] = cells
        if not cells[2] or cells[2] not in ("UNKNOWN",) and " " not in cells[2] and cells[2] == "":
            err(check, f"{name} 的 SPDX 列为空")

    req = root / "requirements.txt"
    if req.is_file():
        declared: list[str] = []
        for raw in req.read_text(encoding="utf-8").splitlines():
            line = raw.split("#", 1)[0].strip()
            if "==" in line:
                declared.append(line.split("==", 1)[0].strip())
        missing = sorted({n for n in declared if n not in rows})
        if missing:
            err(check, f"LICENSES.md 缺少这些依赖（requirements.txt 里有）：{missing}")

    # 可选依赖小节（OCR）：用户装 requirements-ocr.txt 前必须能查到许可证
    if "可选依赖" not in text:
        err(check, "LICENSES.md 缺少『可选依赖』小节（requirements-ocr.txt 里的包必须可查许可证）")
    else:
        optional = text.split("可选依赖", 1)[1]
        for name in ("paddleocr", "paddlepaddle", "pytesseract"):
            if name not in optional:
                err(check, f"LICENSES.md 的『可选依赖』小节缺少 {name}")

    unknown = sorted(n for n, c in rows.items() if len(c) > 2 and c[2] == "UNKNOWN")
    confirm = text.split("待人工确认", 1)[1] if "待人工确认" in text else ""
    for name in unknown:
        if name not in confirm:
            err(check, f"{name} 标为 UNKNOWN 但未出现在「待人工确认」清单里")
    if unknown:
        warn(check, f"{len(unknown)} 个依赖 SPDX 未知，已列入待人工确认：{unknown}")
    if not rows:
        err(check, "许可证表里没有任何依赖行")


def report(root: Path) -> int:
    errors = [msg for level, msg in problems if level == "ERROR"]
    warnings = [msg for level, msg in problems if level == "WARN"]
    print(f"root = {root}")
    for msg in errors:
        print(f"ERROR {msg}")
    for msg in warnings:
        print(f"WARN  {msg}")
    print(f"SUMMARY errors={len(errors)} warnings={len(warnings)}")
    return 1 if errors else 0


def resolve_skill_dir(root: Path, explicit: str | None) -> Path | None:
    """定位 skill 目录。

    DSH 只扫描「项目根」下的 `.dsh/skills`（项目根 = 最近的存在 .git 的祖先，
    没有 .git 则用 cwd）。因此 skill 可能位于仓库根，也可能位于仓库根的上一级
    —— 即当前 project3/.dsh/skills/r2v-domain 的布局。

    两处同时存在副本时必须报错并指出哪一份是被扫描的（仓库内那份在
    r2v-tool/.dsh 下不会被 DSH 发现，属于陈旧副本），避免校验了错误的文件。
    """
    if explicit:
        return Path(explicit).resolve()
    in_repo = root / SKILL_REL
    beside_repo = root.parent / SKILL_REL
    if in_repo.is_dir() and beside_repo.is_dir():
        err(
            "layout",
            f"存在两份 skill 副本：被 DSH 扫描的是 {beside_repo}，"
            f"陈旧副本 {in_repo} 不会被发现——请删除陈旧副本",
        )
    for candidate in (beside_repo, in_repo):
        if candidate.is_dir():
            return candidate
    return None


def selftest() -> int:
    """受限 YAML 解析器自检：证明它在缺 PyYAML 时给出的结论可信。"""
    cases = {
        "缩进跳级（presets 用 2 空格、子键用 6 空格）": (
            "meta:\n  schema_version: \"0.2\"\npresets:\n  topo-map:\n    overrides:\n"
            "      fitting.tolerance_mm: 0.15\n",
            lambda d: d["presets"]["topo-map"]["overrides"]["fitting.tolerance_mm"] == 0.15,
        ),
        "行内映射与行内列表": (
            "dxf:\n  layers:\n    - {name: \"LINES_W0.18\", color: 7, linetype: CONTINUOUS}\n"
            "  comments_include: [version, rms]\n",
            lambda d: d["dxf"]["layers"][0]["name"] == "LINES_W0.18"
            and d["dxf"]["layers"][0]["color"] == 7
            and d["dxf"]["comments_include"] == ["version", "rms"],
        ),
        "注释、空行、文档起点、引号内井号": (
            "---\n# 注释\n\nkey: \"a # b\"   # 尾注释\nother: null\nflag: true\n",
            lambda d: d["key"] == "a # b" and d["other"] is None and d["flag"] is True,
        ),
        "三态开关与列表序列": (
            "background_flatten:\n  enable: auto\nmethods:\n  - sauvola\n  - otsu\n",
            lambda d: d["background_flatten"]["enable"] == "auto"
            and d["methods"] == ["sauvola", "otsu"],
        ),
        "嵌套序列（- key: value + 子键）": (
            "layer_priority:\n  - name: linewidth\n    weight: 3\n",
            lambda d: d["layer_priority"][0] == {"name": "linewidth", "weight": 3},
        ),
    }
    failures = 0
    for title, (text, predicate) in cases.items():
        try:
            parsed = parse_yaml_subset(text)
            ok = bool(predicate(parsed))
        except Exception as exc:  # noqa: BLE001
            ok, parsed = False, f"{type(exc).__name__}: {exc}"
        print(f"{'PASS' if ok else 'FAIL'}  {title}" + ("" if ok else f"  -> {parsed}"))
        failures += 0 if ok else 1
    print(f"SELFTEST failures={failures}/{len(cases)}")
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="R2V 契约静态校验（零依赖）")
    parser.add_argument("--root", default=None, help="r2v-tool 仓库根目录（默认脚本上级目录）")
    parser.add_argument("--skill-dir", default=None, help="显式指定 skill 目录")
    parser.add_argument("--selftest", action="store_true", help="只跑受限 YAML 解析器自检")
    args = parser.parse_args()

    if args.selftest:
        return selftest()

    root = Path(args.root).resolve() if args.root else Path(__file__).resolve().parents[1]
    skill_dir = resolve_skill_dir(root, args.skill_dir)
    if skill_dir is None:
        err(
            "layout",
            f"skill 目录不存在：试过 {root / SKILL_REL} 与 {root.parent / SKILL_REL}"
            "（可加 --skill-dir 指定）",
        )
        return report(root)
    print(f"skill_dir = {skill_dir}")

    base_cfg = load_config(root / "config.yaml", "config")
    presets_cfg = load_config(skill_dir / "presets.yaml", "presets")
    if base_cfg is not None and presets_cfg is not None:
        check_versions(base_cfg, presets_cfg)
        check_presets(base_cfg, presets_cfg)
        check_layers(base_cfg, skill_dir / "dxf_rules.md")
        check_entities(base_cfg, skill_dir / "dxf_rules.md")
        check_code_layers(base_cfg, root / "src" / "dxf_writer.py", skill_dir / "dxf_rules.md")
        print(f"presets = {', '.join(sorted((presets_cfg.get('presets') or {}).keys()))}")

    decimals = ((base_cfg or {}).get("dxf") or {}).get("coordinate_decimals")
    if decimals is not None and int(decimals) < 4:
        err("precision", f"dxf.coordinate_decimals={decimals} 少于 4 位小数（0.0001 mm 契约）")

    check_doc_hygiene(skill_dir)
    check_referenced_paths(root, skill_dir)
    check_budget_rows(root, skill_dir)
    check_licenses(root)
    return report(root)


if __name__ == "__main__":
    sys.exit(main())
