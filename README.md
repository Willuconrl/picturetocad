# r2v-tool

光栅图纸 → CAD（DXF）矢量化工具。精度优先：宁可少输出，也不输出无法标注误差的东西。

> **能力边界（硬性）**：输出是对输入图像的**几何近似**，不是对现实物体的测量还原。
> **未配准（无控制点、无比例尺）的输出，其尺寸无任何可信度**，不得用于加工、报建、算量、结算。
> 输入下限：≥300 DPI、线条与背景对比度可辨、单页单图、无明显透视畸变。
> 所有自动结果默认视为**草稿**，人工校核是流程的一部分。

## 安装

验收基线是 **Linux + Python 3.10**；本机开发环境为 Windows + Python 3.14。

```bash
python -m pip install -r requirements.txt
python scripts/verify_wheels.py --check      # 校验锁定版本在 cp310-manylinux 与本机都有 wheel
```

`requirements.txt` 里 numpy / scipy / scikit-image 按 `python_version` 分流：这三者**没有任何
单个版本**同时提供 cp310 与 cp314 的 wheel，故 `<3.12` 用验收基线组、`>=3.12` 用开发组。
**验收与回归以 `<3.12` 组为准**，详见文件头部注释。

两个依赖选择需要记住（别改回去）：

- **opencv-python-headless** 而不是 opencv-python：本工具是 Linux 服务器上的批处理 CLI，
  没有 GUI；完整版会引入 libGL/X11 依赖，在无显卡容器里直接 import 失败。headless 只去掉
  highgui 窗口，图像算法完全相同，debug 叠加图走 PNG 落盘。
- **Pillow**：DPI 元数据解析（`SPEC §3.2` 要求精确 DPI），cv2 读 EXIF/pHYs 不可靠。

## 用法

```bash
python -m src --check-config                            # 校验 config.yaml 后退出（非法值启动即失败）
python -m src --check-config --preset blueprint         # 套用预设后校验
python -m src --print-effective --preset mechanical     # 打印合并后的有效参数 + config_hash
python -m src --list-presets                            # 列出四套预设
```

退出码：`0` 成功；`2` 配置/预设非法。

## 目录

| 目录 | 内容 |
|---|---|
| `src/` | 算法与核心逻辑，按阶段分模块（`s0_qc.py` … `s8_dxf.py`，见 `specs/SPEC.md`） |
| `tests/` | pytest 用例 + hypothesis 性质测试 + 合成图夹具 |
| `test_images/` | 输入图与 `manifest.csv`（七列：`file, preset, true_dpi, page_class, golden_dxf, notes, has_golden`） |
| `golden/` | 手工描/程序生成的正确 DXF，用于回归比对 |
| `outputs/` | 运行时产物：`metrics_<stage>.json`、`manifest.json`、`debug/overlay_*.png` |
| `specs/` | `SPEC.md`：误差预算与逐阶段验收标准（真相源） |

配置参数只写 `config.yaml`（数值真相源），由 `src/config.py` 的 pydantic schema 校验；
预设只写差量覆盖，见 `config.yaml` 的 `presets` 段。

## 当前进度

- **M0-a 完成**：`manifest.csv` schema、合成图 + golden 回归框架、依赖锁定与 wheel 校验。
- **M0-b 完成**：`config.yaml` 的 pydantic-settings schema、预设覆盖与重校验、`src/coords.py`
  坐标变换、`src/artifacts.py` 产物契约、`r2v` CLI 骨架、DXF 比特级可复现（`src/dxf_repro.py`）。
- 后续：S0 质检 → S1 预处理 → … → S8 组装（见 `specs/SPEC.md` §4）。

## TODO

- [ ] **M4 完成后补 py3.10 CI**：当前 3.10 分支只做了 wheel 元数据校验，未在 3.10 解释器上实跑
      （开发机只有 3.14）。等几何拟合 + 拓扑（M4）代码稳定后加 GitHub Actions `py3.10` job，
      避免现在把时间花在环境搭建上而阻塞核心算法开发。
