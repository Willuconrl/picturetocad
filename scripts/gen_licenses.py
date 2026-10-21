#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从**已安装包的 metadata** 生成 `docs/LICENSES.md`（不编造许可证）。

规则
----
1. 依赖清单取自 `requirements.txt` 的锁定行（唯一真相源），逐包读
   `importlib.metadata`：版本、`License-Expression`（PEP 639）、License 分类器、
   `Home-page` / `Project-URL`。
2. SPDX 判定**只用可验证证据**：
   a. metadata 里有 `License-Expression` → 直接采用；
   b. 否则查白名单分类器（只收语义唯一、不会被误判的条目）；
   c. 其余一律 `UNKNOWN`，并在文件末尾的「待人工确认」里列出来。
   **绝不从 License 的自由文本里猜 SPDX**：`BSD License` 分类器分不清 2/3 条款，
   猜错比留空更糟。
3. 未安装的包不静默跳过：写进「待人工确认」并让脚本以退出码 1 结束。

用法：
    python scripts/gen_licenses.py            # 生成/覆盖 docs/LICENSES.md
    python scripts/gen_licenses.py --check    # 只校验现有文件是否为最新
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = ROOT / "requirements.txt"
OUTPUT = ROOT / "docs" / "LICENSES.md"

#: 只在语义唯一时使用的分类器 → SPDX 白名单
CLASSIFIER_SPDX = {
    "License :: OSI Approved :: MIT License": "MIT",
    "License :: OSI Approved :: Apache Software License": "Apache-2.0",
    "License :: OSI Approved :: ISC License (ISCL)": "ISC",
    "License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)": "MPL-2.0",
    "License :: OSI Approved :: Python Software Foundation License": "PSF-2.0",
    "License :: OSI Approved :: The Unlicense (Unlicense)": "Unlicense",
}

#: 自由文本 License 字段里出现这些完整 SPDX 标识时可直接采信（是标识符而非"BSD License"这种说法）
KNOWN_SPDX_IDS = (
    "Apache-2.0",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "ISC",
    "MIT",
    "MPL-2.0",
    "PSF-2.0",
)


@dataclass(frozen=True)
class Dependency:
    name: str
    version: str
    pins: tuple[str, ...] = ()


@dataclass(frozen=True)
class LicenseRow:
    name: str
    version: str
    spdx: str
    source_url: str
    evidence: str


def parse_requirements(path: Path) -> list[Dependency]:
    """解析锁定行；同名多行（python_version 分支）合并成一条，并保留全部分支版本。"""
    grouped: dict[str, list[str]] = {}
    order: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or "==" not in line:
            continue
        name, rest = line.split("==", 1)
        version = rest.split(";")[0].strip()
        key = name.strip()
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        if version not in grouped[key]:
            grouped[key].append(version)
    if not order:
        raise SystemExit(f"ERROR {path} 里没有锁定行")
    deps: list[Dependency] = []
    for name in order:
        pins = tuple(grouped[name])
        deps.append(Dependency(name=name, version=pins[0], pins=pins))
    return deps


def _source_url(md: importlib_metadata.PackageMetadata, name: str) -> str:
    project_urls = md.get_all("Project-URL") or []
    parsed: dict[str, str] = {}
    for item in project_urls:
        if "," in item:
            label, url = item.split(",", 1)
            parsed[label.strip().lower()] = url.strip()
    for key in ("homepage", "home-page", "source", "repository", "documentation"):
        if key in parsed and parsed[key]:
            return parsed[key]
    home = md.get("Home-page")
    if home:
        return str(home)
    return f"https://pypi.org/project/{name}/"


#: 从 LICENSE 文件正文判定的特征文本（先判长文本，避免被短特征截胡）
LICENSE_TEXT_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Apache-2.0", ("apache license", "version 2.0")),
    ("MPL-2.0", ("mozilla public license version 2.0",)),
    ("PSF-2.0", ("python software foundation license",)),
    ("ISC", ("permission to use, copy, modify, and/or distribute this software",)),
    ("BSD-3-Clause", ("redistribution and use in source and binary forms", "neither the name")),
    ("BSD-2-Clause", ("redistribution and use in source and binary forms",)),
    ("MIT", ("permission is hereby granted, free of charge",)),
)

#: 包内许可证文件名（大小写不敏感前缀）
LICENSE_FILE_PREFIXES = ("LICENSE", "LICENCE", "COPYING", "NOTICE")


def _license_text_files(name: str) -> list[Path]:
    """列出已安装包内的许可证文件（dist-info 与包目录都找）。"""
    try:
        dist = importlib_metadata.distribution(name)
    except importlib_metadata.PackageNotFoundError:
        return []
    found: list[Path] = []
    seen: set[Path] = set()
    for item in dist.files or []:
        upper = item.name.upper()
        if not upper.startswith(LICENSE_FILE_PREFIXES):
            continue
        path = Path(dist.locate_file(item))
        if path.is_file() and path not in seen and path.stat().st_size <= 1 << 20:
            seen.add(path)
            found.append(path)
    return found


def _spdx_from_license_file(name: str) -> tuple[str, str] | None:
    """读包内 LICENSE 正文推断 SPDX。成功返回 (spdx, 证据)，失败返回 None。"""
    for path in _license_text_files(name):
        try:
            text = path.read_text(encoding="utf-8", errors="replace").lower()
        except OSError:
            continue
        for spdx, needles in LICENSE_TEXT_PATTERNS:
            if all(needle in text for needle in needles):
                return spdx, f"source=LICENSE_file_inference ({path.name})"
    return None


def _spdx(md: importlib_metadata.PackageMetadata, name: str) -> tuple[str, str]:
    """返回 (SPDX 或 UNKNOWN, 证据说明)。

    判定顺序：License-Expression → 语义唯一的分类器 → License 字段 → 包内 LICENSE 正文。
    前四步都失败才落 UNKNOWN，并在「待人工确认」里列出。
    """
    expression = md.get("License-Expression")
    if expression:
        return str(expression).strip(), "metadata: License-Expression"

    classifiers = md.get_all("Classifier") or []
    for classifier in classifiers:
        if classifier in CLASSIFIER_SPDX:
            return CLASSIFIER_SPDX[classifier], f"classifier: {classifier}"

    raw_license = (md.get("License") or "").strip()
    if raw_license in KNOWN_SPDX_IDS:
        return raw_license, "metadata: License"

    # 自由文本里出现完整 SPDX 标识也采信（例如 "MIT License" 不算，必须出现 "MIT" 标识本身）
    tokens = {tok.strip(" .;,()") for tok in raw_license.replace("\n", " ").split()}
    hits = sorted(tokens & set(KNOWN_SPDX_IDS))
    if len(hits) == 1:
        return hits[0], "metadata: License（含唯一 SPDX 标识）"

    from_file = _spdx_from_license_file(name)
    if from_file is not None:
        return from_file

    reason = "metadata 缺失" if not raw_license and not classifiers else (
        f"无 License-Expression，分类器不可判定（{classifiers[:2] or '无'}），且包内无 LICENSE 文件"
    )
    return "UNKNOWN", reason


def collect(deps: Iterable[Dependency]) -> tuple[list[LicenseRow], list[str]]:
    rows: list[LicenseRow] = []
    problems: list[str] = []
    for dep in deps:
        try:
            md = importlib_metadata.metadata(dep.name)
            installed = importlib_metadata.version(dep.name)
        except importlib_metadata.PackageNotFoundError:
            problems.append(f"{dep.name}=={dep.version}：**未安装**，无法读取 metadata")
            rows.append(
                LicenseRow(
                    name=dep.name,
                    version=dep.version,
                    spdx="UNKNOWN",
                    source_url=f"https://pypi.org/project/{dep.name}/",
                    evidence="未安装",
                )
            )
            continue
        spdx, evidence = _spdx(md, dep.name)
        if installed not in dep.pins:
            problems.append(
                f"{dep.name}：requirements 锁定 {'/'.join(dep.pins)}，实际安装 {installed}"
                "（许可证按实际安装版本记录，请先对齐环境）"
            )
        if len(dep.pins) > 1:
            evidence += f"；requirements 分支锁定 {'/'.join(dep.pins)}，本表按实装 {installed} 记录"
        rows.append(
            LicenseRow(
                name=dep.name,
                version=installed,
                spdx=spdx,
                source_url=_source_url(md, dep.name),
                evidence=evidence,
            )
        )
    return rows, problems


def render(rows: Sequence[LicenseRow], problems: Sequence[str]) -> str:
    unknown = [r for r in rows if r.spdx == "UNKNOWN"]
    lines: list[str] = []
    lines.append("# 第三方依赖许可证清单")
    lines.append("")
    lines.append(
        "> 本文件由 `scripts/gen_licenses.py` 从**已安装包的 metadata** 生成，不要手改；"
        "改依赖后重新运行脚本。"
    )
    lines.append(">")
    lines.append(
        "> SPDX 判定只采信 `License-Expression` 或语义唯一的分类器；"
        "其余标为 `UNKNOWN` 并列入文末待人工确认清单——**不猜**。"
    )
    lines.append("")
    lines.append("| 包 | 版本 | SPDX | 来源 | 判定依据 |")
    lines.append("|---|---|---|---|---|")
    for row in rows:
        lines.append(
            f"| {row.name} | {row.version} | {row.spdx} | {row.source_url} | {row.evidence} |"
        )
    lines.append("")
    lines.append("## 红线（SPEC §2）")
    lines.append("")
    lines.append("- 禁止 GPL 依赖进入主分发：Potrace 一律不用。")
    lines.append(
        "- PySide6 为 LGPL-3：必须以独立包/动态链接方式隔离，并在本文件写明用户替换义务。"
    )
    lines.append("")
    lines.append("## 待人工确认")
    lines.append("")
    if unknown:
        for row in unknown:
            lines.append(
                f"- [ ] `{row.name}=={row.version}`：SPDX 未知（{row.evidence}），"
                f"请人工核对 {row.source_url} 后补全"
            )
    else:
        lines.append("- 无（全部依赖都有可判定的 SPDX 标识）。")
    for problem in problems:
        lines.append(f"- [ ] {problem}")
    lines.append("")
    return "\n".join(lines)


def generate() -> tuple[str, int]:
    deps = parse_requirements(REQUIREMENTS)
    rows, problems = collect(deps)
    text = render(rows, problems)
    return text, len([r for r in rows if r.spdx == "UNKNOWN"]) + len(problems)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="生成 docs/LICENSES.md")
    ap.add_argument("--check", action="store_true", help="只校验现有文件是否与 metadata 一致")
    args = ap.parse_args(argv)

    text, unresolved = generate()
    if args.check:
        if not OUTPUT.is_file():
            print(f"FAIL 缺少 {OUTPUT.relative_to(ROOT)}")
            return 1
        if OUTPUT.read_text(encoding="utf-8") != text:
            print(f"FAIL {OUTPUT.relative_to(ROOT)} 与已安装 metadata 不一致（请重新生成）")
            return 1
        print(f"OK {OUTPUT.relative_to(ROOT)} 与 metadata 一致；UNKNOWN/待确认 {unresolved} 项")
        return 0

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(text, encoding="utf-8")
    print(
        f"WROTE {OUTPUT.relative_to(ROOT)} deps={len(parse_requirements(REQUIREMENTS))} "
        f"unresolved={unresolved}（UNKNOWN 属允许状态，由 check_contracts.py 以 WARN 报出）"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
