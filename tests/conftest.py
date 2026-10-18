# tests/conftest.py
"""pytest 全局夹具。

两件事：
1. 把仓库根放进 `sys.path`，供 `import src.*` 与 `import tests.*` 使用；
2. 提供自管的 `tmp_path`（见下）。

关于 `tmp_path` 覆盖
--------------------
本项目的开发会话运行在 Windows 的受限沙箱里，pytest 自带的 `tmp_path` /
`tmp_path_factory` 在 `tempfile.gettempdir()` 下的旧目录上会 `PermissionError`
（[WinError 5]），并且会在 `pytest_sessionfinish` 里把整个会话拖挂。
这里用自管临时目录替代：优先 `R2V_TEST_TMP_ROOT`，否则退化为系统临时目录下的
一次性目录。行为与 pytest 内置版本一致（每个用例一个独立目录、结束后尽力清理），
但在受限环境下也能跑完。
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Iterator

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _safe_name(nodeid: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.\-]+", "_", nodeid)
    return name[-80:] or "test"


@pytest.fixture
def tmp_path(request: pytest.FixtureRequest) -> Iterator[Path]:
    """每个用例一个独立可写目录（替代 pytest 内置 tmp_path）。"""
    root_env = os.environ.get("R2V_TEST_TMP_ROOT")
    if root_env:
        base = Path(root_env)
        base.mkdir(parents=True, exist_ok=True)
    else:
        base = Path(tempfile.mkdtemp(prefix="r2v-tests-"))

    path = base / _safe_name(request.node.nodeid)
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    try:
        yield path
    finally:
        if not root_env:  # 系统临时目录：整体删掉
            shutil.rmtree(base, ignore_errors=True)
        else:
            shutil.rmtree(path, ignore_errors=True)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: 需要真实文件资产或大图的用例")
