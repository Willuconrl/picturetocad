# tests/conftest.py
"""pytest 全局夹具：把仓库根放进 sys.path，供 `import src.*` 与 `import tests.*` 使用。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pytest_configure(config) -> None:  # noqa: ANN001 - pytest 钩子签名
    """打印一次仓库根，方便定位测试实际跑在哪份代码上。"""
    config.addinivalue_line("markers", "slow: 需要真实文件资产的用例")
