# tests/conftest.py
"""pytest 全局夹具。

两件事：
1. 把仓库根放进 `sys.path`，供 `import src.*` 与 `import tests.*` 使用；
2. **条件性**覆写 `tmp_path`（见下）。

`tmp_path` 覆写逻辑（用户决策 #2）
---------------------------------
CI 与普通开发机**完全不受影响**：只有在环境变量 `R2V_TEST_TMP_ROOT` 存在时才定义
自定义 `tmp_path` 并覆盖 pytest 内置实现；否则本文件不定义该夹具，pytest 用自带的。

为什么需要这个后门：本项目的受限沙箱会话里，pytest 内置 `tmp_path` 会在旧临时目录上
抛 `PermissionError [WinError 5]`，并且 `pytest_sessionfinish` 会把整轮会话拖挂。
受限环境运行时这样用：

    R2V_TEST_TMP_ROOT=D:\\some\\writable\\dir  python -m pytest -p no:tmpdir -p no:cacheprovider

行为与内置版本一致：每个用例一个独立目录，结束后尽力清理。
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

#: 只有显式设置了这个变量才覆写 tmp_path（CI/普通环境走 pytest 内置实现）
_TMP_ROOT_ENV = "R2V_TEST_TMP_ROOT"


def _safe_name(nodeid: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.\-]+", "_", nodeid)
    return name[-80:] or "test"


if os.environ.get(_TMP_ROOT_ENV):

    @pytest.fixture
    def tmp_path(request: pytest.FixtureRequest) -> Iterator[Path]:
        """受限环境下的 tmp_path 替代实现（每个用例一个独立目录）。"""
        base = Path(os.environ[_TMP_ROOT_ENV])
        base.mkdir(parents=True, exist_ok=True)
        path = base / _safe_name(request.node.nodeid)
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
        path.mkdir(parents=True, exist_ok=True)
        try:
            yield path
        finally:
            shutil.rmtree(path, ignore_errors=True)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: 需要真实文件资产或大图的用例")
