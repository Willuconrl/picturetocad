# CI 说明

## 徽章

README 顶部徽章的 URL 目前是占位形式：

```
https://github.com/OWNER/r2v-tool/actions/workflows/ci.yml/badge.svg
```

首次 `git push` 到 GitHub 之后，把 `OWNER` 替换成实际账号/组织名即可。**不影响 CI 运行**，
只影响徽章能否显示（占位状态徽章会 404）。

替换命令（任选其一，在仓库根目录执行）：

```bash
# Linux / macOS / Git Bash
sed -i 's#github.com/OWNER/r2v-tool#github.com/<你的账号>/r2v-tool#g' README.md docs/CI.md
```

```powershell
# Windows PowerShell
(Get-Content README.md -Raw) -replace 'github.com/OWNER/r2v-tool','github.com/<你的账号>/r2v-tool' |
  Set-Content README.md -NoNewline
```

仓库改名时同样只需替换这一段，`ci.yml` 里的 `on:` 触发器按分支名匹配，与仓库名无关。

## 工作流

`.github/workflows/ci.yml` 有两个 job：

| job | Runner | Python | 说明 |
|---|---|---|---|
| `test` | `ubuntu-latest` | 3.10、3.12 | 验收基线 + 开发组依赖；跑全流程 |
| `test-windows` | `windows-latest` | 3.10 | 提前暴露路径/编码问题；`pytest` 与 `check_contracts` 硬通过，端到端冒烟允许 `continue-on-error` |

两个 job 都是 `fail-fast: false`：Linux 与 Windows 各自跑完，互不打断。

**不在 Linux 上跑 3.11**：`requirements.txt` 里 3.11 与 3.10 同属 `python_version < 3.12`
分组，覆盖率等价；若将来依赖分组改成三段，再补 3.11 并同步更新 `ci.yml` 注释。

**CI 不装 `requirements-ocr.txt`**：保证 S6 的降级路径（无引擎 → `text=""` / `confidence=0` /
`engine="none"` + WARN）在每次提交上都被覆盖。

## 本地等效运行

```bash
bash scripts/ci_local.sh
```

脚本会先打印本机解释器版本；不是 3.10/3.12 时输出 Warning——本地只保证"脚本正确性"，
解释器与 wheel 兼容性以 CI runner 为准。
