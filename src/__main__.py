# src/__main__.py
"""`python -m src` 入口。"""

from __future__ import annotations

import sys

from src.cli import main

if __name__ == "__main__":
    sys.exit(main())
