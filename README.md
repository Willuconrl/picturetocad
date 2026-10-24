# r2v-tool

[![ci](https://github.com/Willuconrl/picturetocad/actions/workflows/ci.yml/badge.svg)](https://github.com/Willuconrl/picturetocad/actions/workflows/ci.yml)

> 徽章 URL 需在首次推送到 GitHub 后，把 `OWNER` 换成实际账号/组织名（本仓库当前未推送，
> 徽章不会亮起，这是预期状态）。

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

### 端到端转换

```bash
r2v convert --image test_images/synthetic_line_single.png --out outputs/run1
```

产物：`drawing.dxf` + 各阶段 `metrics_*.json` + `pending_ocr_regions.json` /
`pending_ocr_lines.json` + `debug/overlay_s6.png`。**OCR 引擎缺失时 stderr 会打印
`OCR skipped: pending implementation` 之类的提示，不会静默跳过。**

### OCR 可选依赖

文字识别是**可选能力**：不装也能跑完整几何链路，只是不产生 TEXT 实体。

```bash
python -m pip install -r requirements-ocr.txt      # 需要联网，约数百 MB（含 paddle 基座）
```

| 引擎 | 包 | 版本约束 | 说明 |
|---|---|---|---|
| paddle（首选） | `paddleocr` + `paddlepaddle` | `>=3.7.0,<3.8.0` / `>=3.0.0` | Apache-2.0；纯 Python wheel；传递依赖 `paddlex` 上限 `<3.8.0` 必须同步 |
| tesseract（降级） | `pytesseract` | `>=0.3.13,<0.4` | Apache-2.0；**还需系统装 `tesseract` 可执行文件**，只装 Python 包会调用失败 |
| 无 | — | — | 走降级路径：`text=""` / `confidence=0` / `engine="none"` + WARN |

引擎探测规则：`paddle` 需要 `paddleocr` 与 `paddle` 同时可导入；`tesseract` 需要
`pytesseract` 可导入**且** `tesseract` 在 `PATH` 里（只装 Python 包不算可用）。
详见 `src/ocr_engine.py` 与 `docs/LICENSES.md` 的「可选依赖」小节。

### 低置信度文本处理

S6 会输出 `text` / `confidence` / `engine` / `is_placeholder` 四个字段；S8 只在
**`confidence >= text.confidence_min` 且 `is_placeholder=false`** 时写 TEXT 实体。

- **灰框（`debug/overlay_s8.png` 里的灰色虚线框）** = 被判为低置信度或占位的行，
  只有掩膜、没有 TEXT 实体；数量记在 `metrics_s8.json` 的
  `counts.text_skipped_low_confidence`。
- **阈值位置**：`config.yaml` → `text.ocr.text_confidence_min`（默认 `0.6`）。
- **排查方向**：① 行高是否 < `text.min_height_mm`（默认 1.5 mm，≈8 pt@300DPI）；
  ② 原图该区域是否与图线粘连（被「结构性基元扣除」吃掉）；③ 引擎是否真的可用
  （看 `metrics_s6.json` 的 `details.ocr_engine` 与 WARN）；④ 语言包是否覆盖
  （`text.ocr.lang` 默认 `["ch","en"]`）。
- **绝不编造**：置信度不足时宁可缺字，也不写猜测内容——缺字可人工补，错字会污染下游。

## CI 保证

`.github/workflows/ci.yml` 在每次 push / PR 上跑以下步骤（`ubuntu-latest`）：

| 步骤 | 命令 | 保证什么 |
|---|---|---|
| 依赖安装 | `pip install -r requirements.txt` | 锁定的 17 个版本可安装 |
| wheel 校验 | `scripts/verify_wheels.py --check` | 锁定版本在 cp310-manylinux 与当前解释器都有 wheel |
| 测试 | `pytest -q` | 333 passed + 1 xfailed 基线（`polyline_bulge` 为已知限制） |
| 契约校验 | `scripts/check_contracts.py` | 配置/坐标契约/许可证清单/文档链接一致 |
| 端到端冒烟 | `python -m src convert ...` | `drawing.dxf` 与 `metrics_s8.json` 真的能产出 |

**matrix 只跑 3.10 与 3.12**：3.10 是验收基线（生产/交付），3.12 是开发组依赖（numpy 2.5.4 等）。
**故意不跑 3.11**——它在 `requirements.txt` 里与 3.10 同属 `python_version < 3.12` 分组，
覆盖率完全等价，加节点只是浪费算力；若将来依赖分组改成三段，再补 3.11。

**CI 刻意不装 `requirements-ocr.txt`**：保证 S6 的降级路径（无引擎 → `text=""` /
`confidence=0` / `engine="none"` + WARN）持续被覆盖。

本地等效运行（脚本正确性本地验、解释器兼容性以 CI 为准）：

```bash
bash scripts/ci_local.sh        # 会先打印 Python 版本；非 3.10/3.12 时给出 Warning
```

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
  五条判据全部通过。
- **M7-S8 完成**：`src/s8_dxf.py`（LINE/ARC/CIRCLE/LWPOLYLINE + 闭合环映射 + 回读自检 +
  未配准水印 + `metrics_s8.json`）；S3 的边新增 `closed_loop` 标注；S5 图框改按闭合边整体判定。
- **M7-Golden 完成（框架）**：`src/golden_compare.py` 扩到**五维**（Hausdorff P95 / 原子基元计数 /
  关键尺寸 / 图层分布 / 闭合标志）、`scripts/golden_compare.py` CLI、`r2v convert` 子命令、
  `src/pipeline.py`（S0→S8 单一实现）、`tests/test_golden_regression.py`（4 合成场景 + 反向用例）。
  **4 个场景当前均未达标**（xfail，实测值见测试文件）——框架本身已验证能检出偏差。
- **M7 收口（3/4 绿 + 已知限制）**：`src/s8_dxf.py`、`src/golden_compare.py`（五维）、
  `scripts/golden_compare.py`、`r2v convert`、`src/pipeline.py`；line/circle/arc 三场景
  五维判据全过，`polyline_bulge` 见「已知限制」。
- **S6 首轮**：`src/s6_text.py` —— 文字候选区提取（结构性基元扣除 + 连通域筛选），
  产出 `pending_ocr_regions.json` 与 `debug/overlay_s6.png`；**未接任何 OCR 引擎**。
- 后续：S6 OCR 引擎接入 → M8（bulge 过渡区根治 + 架构优化）。

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
- **闭合多边形（图框）的处理**：S3 把闭合环标成 `closed_loop`，S4 整条输出一条闭合多段线，
  S5 按整条边取外接框判图框，S8 写成单条 `closed=True` 的 LWPOLYLINE。代价是闭合多边形内部
  不再拆成独立 LINE/ARC——图框这类对象本来就应该保持闭合语义。
- ~~**`synthetic_polyline_bulge` 未达标**~~ → **已在 M9-3 解决（P95 0.242 px，1 LINE + 1 ARC）**。
  历史根因链（记录下来供后人避坑）：
  ① 切点判据多次失败（枚举切点+最小 SSE → 直线侧被判成巨弧；曲率突变点阈值与线宽绑定 → 永不触发）；
  ② 真正的伪影来源是 **S4 接缝写回槽位碰撞**——直线与弧两个 `model_copy` 用了可能重合的索引
  表达式，在「弧在前、线在后」时把 LINE 覆盖成第二条 ARC（`counts={'ARC': 2}`、P95 185.9 px）；
  ③ 修法：`_align_line_arc_seams` 用**角度语义**判定接缝端 + 直线∩圆闭式交点对齐两侧端点
  （2° 跨度门禁 / 2.0 mm 距离门禁 / 5° 扫角漂移门禁），并给被改写的弧打 `seam_aligned`
  标记、由 `_enforce_g1` 豁免（G1 用坐标判切线，会误降级 0.6° 的合法吸附）。
- **S6 目前只提候选区**：字符笔画会被 S3/S4 拟合成短基元，因此 S6 用「结构性基元」
  （长度 > max(10×线宽, 4×最小字高)）做扣除，短笔画不扣。根治办法是在 S4 之前做
  文字/线条分离（M8）。

## TODO

- [ ] **M4 完成后补 py3.10 CI**：当前 3.10 分支只做了 wheel 元数据校验，未在 3.10 解释器上实跑
      （开发机只有 3.14）。等几何拟合 + 拓扑（M4）代码稳定后加 GitHub Actions `py3.10` job，
      避免现在把时间花在环境搭建上而阻塞核心算法开发。
