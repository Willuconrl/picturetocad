#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""依赖 wheel 可用性验证（零第三方依赖，只用标准库）。

为什么需要这个脚本
------------------
验收基线是 **Linux + Python 3.10**，而本机是 Windows + Python 3.14。
`pip install` 成功只证明「当前解释器能装」，**不证明**基线环境有 wheel。
本脚本直接查 PyPI 元数据，回答两个问题：

1. `--check`     锁定在 requirements.txt 里的每个版本，在目标平台
                 （默认 cp310 + manylinux_x86_64）和本机解释器上是否都有可用 wheel。
2. `--recommend` 给定包名，列出「同时满足 cp310/manylinux 与本机解释器」的最高版本，
                 供升级依赖时选版本（不许凭印象写版本号）。

退出码：0 = 全部满足；1 = 有缺失；2 = 用法/网络错误。

用法：
    python scripts/verify_wheels.py --check
    python scripts/verify_wheels.py --check --requirements requirements.txt
    python scripts/verify_wheels.py --recommend numpy opencv-python-headless

诚实性说明：
- requires_python 只实现 PEP 440 的常用子集（>=, >, <=, <, ==, !=, ==X.Y.*）；
  遇到不认识的运算符直接报错退出，**不做猜测性放行**。
- 本脚本不下载 wheel，只读 PyPI JSON 元数据；能否真正 import 由 pytest 负责。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

PYPI_JSON = "https://pypi.org/pypi/{name}/json"
PYPI_RELEASE_JSON = "https://pypi.org/pypi/{name}/{version}/json"
TIMEOUT_S = 30

# 基线（验收环境）与本机（开发环境）的 wheel 目标
BASELINE_TAG = "cp310-manylinux_x86_64"


# ---------------------------------------------------------------------------
# PEP 440 版本号（只取 release 段做比较，够用且不引入 packaging 依赖）
# ---------------------------------------------------------------------------
_VERSION_RE = re.compile(r"^\s*v?(\d+(?:\.\d+)*)(.*)$")


def release_tuple(text: str) -> tuple[int, ...]:
    """把 '2.3.1rc1' / '1.4.4' 解析成可比较的 release 元组。"""
    m = _VERSION_RE.match(text or "")
    if not m:
        raise ValueError(f"无法解析版本号: {text!r}")
    return tuple(int(p) for p in m.group(1).split("."))


def _pad(a: tuple[int, ...], b: tuple[int, ...]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    n = max(len(a), len(b))
    return a + (0,) * (n - len(a)), b + (0,) * (n - len(b))


def _cmp(a: tuple[int, ...], b: tuple[int, ...]) -> int:
    a, b = _pad(a, b)
    return (a > b) - (a < b)


class SpecifierError(ValueError):
    """代码里出现了本脚本不支持的 requires_python 写法——必须报错，不许猜。"""


def spec_allows(spec: str | None, version: str) -> bool:
    """判断 requires_python 是否允许某个解释器版本（如 '3.10.14'）。"""
    if spec is None or not spec.strip():
        return True
    target = release_tuple(version)
    for raw in spec.split(","):
        clause = raw.strip()
        if not clause:
            continue
        m = re.match(r"^(>=|<=|==|!=|>|<)\s*(.+)$", clause)
        if not m:
            raise SpecifierError(f"不支持的 requires_python 子句: {clause!r} (spec={spec!r})")
        op, want = m.group(1), m.group(2).strip()
        if want.endswith(".*"):
            want = want[:-2]
            base = release_tuple(want)
            same = _cmp(target[: len(base)], base) == 0
            if op == "==" and not same:
                return False
            if op == "!=" and same:
                return False
            if op not in ("==", "!="):
                raise SpecifierError(f"通配符只允许配 == / !=: {clause!r}")
            continue
        rhs = release_tuple(want)
        c = _cmp(target, rhs)
        ok = {
            ">=": c >= 0,
            ">": c > 0,
            "<=": c <= 0,
            "<": c < 0,
            "==": c == 0,
            "!=": c != 0,
        }[op]
        if not ok:
            return False
    return True


# ---------------------------------------------------------------------------
# wheel 标签解析
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Wheel:
    filename: str
    python_tag: str
    abi_tag: str
    platform_tag: str

    @property
    def is_pure(self) -> bool:
        return self.abi_tag == "none" and self.platform_tag == "any"


def parse_wheel(filename: str) -> Wheel | None:
    if not filename.endswith(".whl"):
        return None
    parts = filename[: -len(".whl")].split("-")
    if len(parts) < 5:
        return None
    return Wheel(filename=filename, python_tag=parts[-3], abi_tag=parts[-2], platform_tag=parts[-1])


def wheel_suits(wheel: Wheel, python_tag: str, platform_prefix: str) -> bool:
    """目标解释器标签（cp310）+ 平台前缀（manylinux / win_amd64 / any）。"""
    if wheel.is_pure:
        return True
    want_py = re.fullmatch(r"cp(\d)(\d+)", python_tag)
    if not want_py:
        raise ValueError(f"未知解释器标签: {python_tag!r}")
    want_minor = int(want_py.group(2))

    # cp310-cp310-*, cp310-abi3-*, cp36-abi3-*（abi3 向后兼容）
    if wheel.abi_tag == "abi3":
        m = re.fullmatch(r"cp(\d)(\d+)", wheel.python_tag)
        if not m:
            return False
        return int(m.group(2)) <= want_minor
    if wheel.python_tag != python_tag:
        return False
    if platform_prefix == "any":
        return wheel.platform_tag == "any"
    if platform_prefix.startswith("manylinux"):
        return wheel.platform_tag.startswith("manylinux") and wheel.platform_tag.endswith("x86_64")
    return wheel.platform_tag.startswith(platform_prefix)


# ---------------------------------------------------------------------------
# PyPI 访问
# ---------------------------------------------------------------------------
_cache: dict[str, dict] = {}
RETRIES = 3
RETRY_SLEEP_S = 2.0


def pypi_json(url: str) -> dict:
    """带重试的 PyPI 读取。PyPI 偶发 RST，重试 3 次仍失败则硬退出（不许静默跳过包）。"""
    if url in _cache:
        return _cache[url]
    last: Exception | None = None
    for attempt in range(1, RETRIES + 1):
        req = urllib.request.Request(url, headers={"User-Agent": "r2v-tool-wheel-check/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            _cache[url] = data
            return data
        except (urllib.error.URLError, ConnectionResetError, TimeoutError, json.JSONDecodeError) as exc:
            last = exc
            if attempt < RETRIES:
                time.sleep(RETRY_SLEEP_S * attempt)
    raise SystemExit(f"ERROR 无法读取 {url}（重试 {RETRIES} 次后仍失败）: {last}")


def package_data(name: str) -> dict:
    """一次请求拿到该包全部版本的元数据（`releases` 里每个文件自带 requires_python）。"""
    return pypi_json(PYPI_JSON.format(name=name))


def release_entries(name: str, version: str) -> list[dict]:
    return list(package_data(name).get("releases", {}).get(version, []))


def release_files(name: str, version: str) -> list[str]:
    return [f["filename"] for f in release_entries(name, version)]


def all_versions(name: str) -> list[str]:
    return list(package_data(name).get("releases", {}).keys())


def requires_python_of(name: str, version: str) -> str | None:
    for f in release_entries(name, version):
        rp = f.get("requires_python")
        if rp:
            return rp
    return None


# ---------------------------------------------------------------------------
# 模式
# ---------------------------------------------------------------------------
def local_python_tag() -> tuple[str, str]:
    py = f"cp{sys.version_info.major}{sys.version_info.minor}"
    if sys.platform.startswith("win"):
        plat = "win_amd64"
    elif sys.platform == "darwin":
        plat = "macosx"
    else:
        plat = "manylinux"
    return py, plat


def supports(wheel_files: Sequence[str], python_tag: str, platform_prefix: str) -> str | None:
    for name in wheel_files:
        w = parse_wheel(name)
        if w and wheel_suits(w, python_tag, platform_prefix):
            return w.filename
    return None


_REQ_RE = re.compile(r"^\s*([A-Za-z0-9_.\-]+)\s*==\s*([^\s;#]+)\s*(?:;\s*(.*))?$")
_MARKER_RE = re.compile(r"""^\s*python_version\s*(>=|<=|==|!=|>|<)\s*["']([^"']+)["']\s*$""")


def marker_allows(marker: str | None, python_version: str) -> bool:
    """环境标记是否适用于目标解释器。

    只支持 `python_version <运算符> "X.Y"` 这一种写法；遇到别的写法直接报错——
    静默放行会让基线校验变成假绿。
    """
    if marker is None:
        return True
    m = _MARKER_RE.match(marker)
    if not m:
        raise SpecifierError(f"不支持的环境标记: {marker!r}（只支持 python_version 比较）")
    op, want = m.group(1), m.group(2)
    c = _cmp(release_tuple(python_version), release_tuple(want))
    return {
        ">=": c >= 0,
        ">": c > 0,
        "<=": c <= 0,
        "<": c < 0,
        "==": c == 0,
        "!=": c != 0,
    }[op]


@dataclass
class Pin:
    name: str
    version: str
    marker: str | None


def parse_requirements(path: Path) -> list[Pin]:
    pins: list[Pin] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        m = _REQ_RE.match(line)
        if not m:
            raise SystemExit(f"ERROR {path}:{lineno} 不是锁定行（必须 name==version）: {raw!r}")
        pins.append(Pin(name=m.group(1), version=m.group(2), marker=(m.group(3) or None)))
    if not pins:
        raise SystemExit(f"ERROR {path} 里没有任何锁定行")
    return pins


def cmd_check(path: Path) -> int:
    pins = parse_requirements(path)
    local_tag, local_plat = local_python_tag()
    local_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    print(f"baseline target : {BASELINE_TAG} / Python 3.10")
    print(f"local  target   : {local_tag} / Python {local_ver} / {local_plat}")
    print(f"{'package':28s} {'version':12s} {'baseline':10s} {'local':10s} requires_python")
    failures = 0
    for pin in pins:
        base_applies = marker_allows(pin.marker, "3.10")
        local_applies = marker_allows(pin.marker, local_ver)
        if not base_applies and not local_applies:
            print(f"{pin.name:28s} {pin.version:12s} {'N/A':10s} {'N/A':10s} marker={pin.marker}")
            failures += 1
            continue
        files = release_files(pin.name, pin.version)
        if not files:
            print(f"{pin.name:28s} {pin.version:12s} {'NO-FILES':10s} {'-':10s} -")
            failures += 1
            continue
        base_ok = supports(files, "cp310", "manylinux") if base_applies else None
        local_ok = supports(files, local_tag, local_plat) if local_applies else None
        rp = requires_python_of(pin.name, pin.version) or "-"
        if base_applies and not spec_allows(rp, "3.10.14"):
            base_ok = False
        if local_applies and not spec_allows(rp, local_ver + ".0"):
            local_ok = False
        base_s = "SKIP" if base_ok is None else ("OK" if base_ok else "MISSING")
        local_s = "SKIP" if local_ok is None else ("OK" if local_ok else "MISSING")
        if base_ok is False or local_ok is False:
            failures += 1
        print(f"{pin.name:28s} {pin.version:12s} {base_s:10s} {local_s:10s} {rp}")
    distinct = {(p.name, p.version) for p in pins}
    print(f"SUMMARY pins={len(pins)} distinct_versions={len(distinct)} failures={failures}")
    return 0 if failures == 0 else 1


def cmd_recommend(names: Iterable[str]) -> int:
    local_tag, local_plat = local_python_tag()
    for name in names:
        versions = all_versions(name)
        cands = []
        for v in versions:
            try:
                rt = release_tuple(v)
            except ValueError:
                continue
            if re.search(r"(a|b|rc|dev|post)", v):
                continue
            cands.append((rt, v))
        cands.sort(reverse=True)
        picked = None
        for _, v in cands:
            try:
                rp = requires_python_of(name, v)
                files = release_files(name, v)
            except SystemExit:
                continue
            if not spec_allows(rp, "3.10.14") or not spec_allows(rp, sys.version.split()[0]):
                continue
            if not supports(files, "cp310", "manylinux"):
                continue
            if not supports(files, local_tag, local_plat):
                continue
            picked = (v, rp)
            break
        if picked is None:
            print(f"{name:28s} NO CANDIDATE (需要人工判断：可能要拆 python_version 标记)")
            continue
        print(f"{name:28s} {picked[0]:12s} requires_python={picked[1] or '-'}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="验证依赖 wheel 在基线平台与本机是否可用")
    ap.add_argument("--check", action="store_true", help="校验 requirements.txt 的锁定行")
    ap.add_argument("--recommend", nargs="+", metavar="PKG", help="为这些包推荐最高可用版本")
    ap.add_argument("--requirements", default="requirements.txt", help="锁定文件路径")
    args = ap.parse_args(argv)

    if not args.check and not args.recommend:
        ap.print_help()
        return 2
    rc = 0
    if args.check:
        rc |= cmd_check(Path(args.requirements))
    if args.recommend:
        rc |= cmd_recommend(args.recommend)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
