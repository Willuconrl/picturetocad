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
r2v --check-config                            # 校验 config.yaml 后退出（非法值启动即失败）
r2v --check-config --preset blueprint         # 套用预设后校验
r2v --print-effective --preset mechanical     # 打印合并后的有效参数 + config_hash
r2v --list-presets                            # 列出四套预设
```

未安装时等价写法：`python -m src --check-config`（开发模式 `pip install -e .` 后 `r2v` 直接可用）。

退出码：`0` 成功；`2` 配置/预设非法。

## 目录

| 目录 | 内容 |
|---|---|
| `src/` | 算法与核心逻辑，按阶段分模块（`s0_qc.py` … `s8_dxf.py`，见 `specs/SPEC.md`） |
| `tests/` | pytest 用例 + hypothesis 性质测试 + 合成图夹具 |
| `test_images/` | 输入图与 `manifest.csv`（七列：`file, preset, true_dpi, page_class, golden_dxf, notes, has_golden`） |
| `golden/` | 手工描/程序生成的正确 DXF，用于回归比对 |
| `docs/` | `LICENSES.md`：依赖许可证清单（由 `scripts/gen_licenses.py` 从已安装 metadata 生成） |
| `outputs/` | 运行时产物：`metrics_<stage>.json`、`manifest.json`、`debug/overlay_*.png` |
| `specs/` | `SPEC.md`：误差预算与逐阶段验收标准（真相源） |
| `docs/` | `LICENSES.md`（依赖许可证）、`coordinate_system.md`（跨模块坐标契约） |

配置参数只写 `config.yaml`（数值真相源），由 `src/config.py` 的 pydantic schema 校验；
预设只写差量覆盖，见 `config.yaml` 的 `presets` 段。

## 当前进度

- **M0-a 完成**：`manifest.csv` schema、合成图 + golden 回归框架、依赖锁定与 wheel 校验。
- **M0-b 完成**：`config.yaml` 的 pydantic-settings schema、预设覆盖与重校验、`src/coords.py`
  坐标变换、`src/artifacts.py` 产物契约、`r2v` CLI 骨架、DXF 比特级可复现（`src/dxf_repro.py`）。
- **M1 完成**：`src/s0_qc.py`（DPI/锐度/倾斜/光照 + `qc_report.json` + degrade 预算放大）、
  可安装包与控制台入口 `r2v`、`docs/LICENSES.md`。
- **M2 完成**：`src/s1_preprocess.py`（BT.709 灰度/黑边裁切/反白/背景平坦化/去噪/超采样 +
  `metrics_s1.json`）；S0 补充 JPEG 块效应与半调网点检测、`skew_correction_applied` 字段；
  锐度阈值收紧到 0.8；许可证改为「metadata → 包内 LICENSE 文件」两级判定。
- **M3 完成**：`src/s2_binarize.py`（四路二值化 + foreground_stability 选优 + 形态学清理 +
  沿法线的 50% 电平亚像素边缘 + 逐点线宽 + `metrics_s2.json`）。
- **M4 完成**：`src/s3_skeleton.py`（细化/中轴双路径 + 毛刺修剪 + 共线合并 + 拐点切分 +
  节点分类 + 断线桥接 + `graph.json`/`metrics_s3.json`）；坐标契约固化在 `docs/coordinate_system.md`。
- **M5 完成**：`src/s4_fit.py`（S2 轮廓重建中心线 → DP 分段 → 直线 TLS / 圆 Kåsa+LM →
  BIC+F 检验竞争 → 完整圆判定 → G1 复检 → LWPOLYLINE 回退 + `primitives.json`/`metrics_s4.json`）。
- **M6 部分完成**：`src/s7_register.py`（四种配准模式 + 稳健离群剔除 + `transform.json`/`metrics_s7.json`）
  四条判据全过；`src/s5_layer.py`（逐基元测宽 → 聚类 → 标准笔宽吸附 → 分层 → 图框 → 正交化）
  五条判据中 ②④⑤ 通过，①③ 见下"已知限制"。
- 后续：S6 OCR + S8 DXF 组装 + Golden 回归（M7，见 `specs/SPEC.md` §4）。

## 已知限制

- **JPEG 块效应**检测是启发式（块边界/内部梯度比），只在 JPEG 输入上运行；指标值会写进
  `qc_report.json` 且标 `confidence=heuristic`，阈值 2.0 只拦强块效应。
- **莫尔纹**用「小圆点连通域密度」判定（干净线条图 0–0.4 / 1e4px²，6px 网点图 279），
  对极细网点（<2px）可能漏检。
- S0 的 `perspective` 与 `stain_mask` 仍未实现，报告里以 `not_evaluated` 明示。
- **S2 亚像素边缘**在"完美抗锯齿阶跃"上存在 ≤1/12 px 的固有偏置（50% 电平判据 + 双线性
  重建的几何后果），实测 ≤0.09 px；真实扫描（边缘本身模糊）上 ≤0.05 px。
- **S2 局部阈值**（sauvola/niblack）在笔画边缘会系统性多判 1 列像素，选优环节用"同分取紧"
  规则压制；`break_touching_text` 与 `steger` 中心线尚未实现，会在 `warnings` 里明示。
- **细笔画的墨宽偏高**（S5 判据①）：0.18 mm ≈ 2.1 px @300 DPI 的笔画实测 2.82 px（+0.69 px =
  0.058 mm > 0.02 mm）；4.1 px / 5.9 px 档仅 +0.11 / +0.16 px。根因在 S2：细笔画的二值掩膜
  把半墨像素也判进来，轮廓点落在掩膜边界而不是 50% 电平处。修复点在 `_edge_offset` 的
  双向搜索，会影响已验收的 S2 指标，需单独决策。
- **闭合多边形（图框）的拐点切分**未完成（S5 判据③）：8 连通环的追踪可能退化成螺旋，
  强行在拐点切开会把圆切成 4 段（实测 S4 因此测不出 CIRCLE）。当前策略是"闭合环不切"，
  代价是图框矩形被当成一条曲线，S5 凑不出 4 条正交线。计划在 M7 与 S8 一起处理。

## TODO

- [ ] **M4 完成后补 py3.10 CI**：当前 3.10 分支只做了 wheel 元数据校验，未在 3.10 解释器上实跑
      （开发机只有 3.14）。等几何拟合 + 拓扑（M4）代码稳定后加 GitHub Actions `py3.10` job，
      避免现在把时间花在环境搭建上而阻塞核心算法开发。
