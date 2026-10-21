#!/usr/bin/env bash
# 本地等效 CI：在开发机上跑一遍 .github/workflows/ci.yml 的同一串命令。
#
# 定位（用户决策 M8-1 #3）：
#   本地负责**脚本正确性**（命令拼写、步骤编排、退出码），
#   Python 3.10 解释器行为与 wheel 兼容性由 GitHub Actions 的 runner 兜底。
#   本脚本开头会打印解释器版本警告，避免"本地全绿、CI 飘红"的误判。
set -euo pipefail

cd "$(dirname "$0")/.."

PY_VERSION="$(python -V 2>&1 | awk '{print $2}')"
echo "== local CI =================================================="
echo "Python: ${PY_VERSION}"
if [[ "${PY_VERSION}" != 3.10.* && "${PY_VERSION}" != 3.12.* ]]; then
  echo "Warning: Running on ${PY_VERSION}; CI baseline is 3.10 and dev group is 3.12."
  echo "         本机结果只能验证脚本正确性，解释器兼容性以 GitHub Actions 为准。"
fi
echo "==============================================================="

step() { echo; echo "--> $*"; }

step "Install dependencies"
python -m pip install --upgrade pip >/dev/null
python -m pip install -r requirements.txt

step "Verify pinned wheels"
python scripts/verify_wheels.py --check

step "Run tests"
python -m pytest -q

step "Check contracts"
python scripts/check_contracts.py

step "Smoke convert (end-to-end)"
SMOKE_DIR="$(mktemp -d)"
python -m src convert --image test_images/synthetic_line_single.png --out "${SMOKE_DIR}"
test -f "${SMOKE_DIR}/drawing.dxf"
test -f "${SMOKE_DIR}/metrics_s8.json"
echo "smoke outputs: ${SMOKE_DIR}"

echo
echo "本地 CI 全部步骤通过。注意：CI 还要求 3.10 解释器，本地结果不代表 CI 一定绿。"
