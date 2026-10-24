# scripts/ci_report_status.py
"""CI 可观测性：把 pytest 失败摘要写进 **commit status**（公开可读，无需日志权限）。

背景：GitHub Actions 的步骤日志需要鉴权才能读取，而 commit status 是公开的。
CI 里在 pytest 之后调用本脚本，就能把"哪几个用例失败"塞进 status description
（≤140 字符），外部（包括 AI 代理）用 `GET /repos/{repo}/commits/{sha}/statuses`
即可读到，无需任何 token。

用法（CI 内）：
    python -m pytest -q --tb=no > pytest.log 2>&1; rc=$?
    python scripts/ci_report_status.py pytest.log "$rc"
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

MAX_DESCRIPTION = 140


def summarize(log_path: Path, returncode: str, pattern: str = "FAILED ") -> str:
    failures: list[str] = []
    if log_path.is_file():
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith(pattern):
                failures.append(line[len(pattern):].split(" - ")[0].strip())
    prefix = f"rc={returncode} failures={len(failures)}"
    if not failures:
        return prefix[:MAX_DESCRIPTION]
    body = " | ".join(failures[:3])
    text = f"{prefix}: {body}"
    if len(text) > MAX_DESCRIPTION:
        text = text[: MAX_DESCRIPTION - 1] + "…"
    return text


def post_status(state: str, description: str, context: str = "pytest-detail") -> None:
    token = os.environ.get("GITHUB_TOKEN")
    repo = os.environ.get("GITHUB_REPOSITORY")
    sha = os.environ.get("GITHUB_SHA")
    if not (token and repo and sha):
        print("[ci_report_status] 缺少 GITHUB_TOKEN/GITHUB_REPOSITORY/GITHUB_SHA，跳过上报")
        return
    payload = json.dumps(
        {"state": state, "context": context, "description": description}
    ).encode("utf-8")
    request = urllib.request.Request(
        f"https://api.github.com/repos/{repo}/statuses/{sha}",
        data=payload,
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            response.read()
        print(f"[ci_report_status] 已上报：{description}")
    except Exception as exc:  # noqa: BLE001 - 上报失败不该影响测试结论
        print(f"[ci_report_status] 上报失败（忽略）：{exc}")


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(
            "用法：ci_report_status.py <log> <returncode> [--context X] [--pattern PREFIX]",
            file=sys.stderr,
        )
        return 2
    log_path = Path(argv[1])
    returncode = argv[2]
    context = "pytest-detail"
    pattern = "FAILED "
    rest = argv[3:]
    for index, item in enumerate(rest):
        if item == "--context" and index + 1 < len(rest):
            context = rest[index + 1]
        if item == "--pattern" and index + 1 < len(rest):
            pattern = rest[index + 1]
    description = summarize(log_path, returncode, pattern)
    state = "success" if returncode.strip() == "0" else "failure"
    post_status(state, description, context)
    print(f"[ci_report_status] {description}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
