# M8-4 PoC：双模型联合切分（LINE/ARC 枚举 + BIC）

> 生成/复现：`python scripts/poc_dual_model_segment.py`（可加 `--tolerance-mm/--lam/--json-out`）。
> 背景：M7 的切点判据在 `synthetic_polyline_bulge` 上失效——"圆弧 + 直线"拼接处，
> 任何单模型残差阈值都会被高曲率区拉偏（伪曲率 4.31×，见 `docs/m8-3_edt_poc.md`）。

## 算法

对切点 k 与模型对 (M1, M2) ∈ {LINE, ARC}²：

```
SSE(k)  = SSE(M1, P[:k]) + SSE(M2, P[k:])
Cost(k) = SSE(k) + λ · ln(n) · p(M1,M2)        # p(LINE)=2, p(ARC)=3
```

- 搜索：**粗到细**两级（粗步长 10，再在最优附近 ±10 精搜），把 O(n) 次圆拟合降到 O(n/10)；
- 扫描阶段的圆拟合用 **Kåsa 代数解**（不做 LM），只有最终胜者做 LM 精化；
- 残差判据与线宽**解耦**：绝对阈值 `--tolerance-mm`（默认 0.15 mm，300 DPI ≈ 1.77 px）；
- 参与竞争的模型对：`line+line`、`line+arc`、`arc+line`。**排除 `arc+arc`**——
  工程图上"两段同径圆弧相接"极罕见，而它每个候选点多做 2 次圆拟合（占扫描 ~40%）。

## bulge 场景实测（493 个中心线采样点，边长 505 px）

| 项 | 值 | 判定 |
|---|---|---|
| 切点索引 | **220** | 剩余 273 点给 ARC 段 —— 与 golden 的弧长 273 px **完全吻合** ✓ |
| 模型组合 | **line(220) + arc(273)** | 与 golden 的「直线 + 圆弧」一致 ✓ |
| 左段 RMS | **0.146 px** | ≤ 1.77 px（0.15 mm）✓ |
| 右段 RMS | **0.178 px** | ≤ 1.77 px ✓ |
| 是否通过容差 | **true** | ✓ |
| 扫描耗时 | **13.62 ms** | ✗ 超过 10 ms 红线 |
| 精化耗时 | 0.67 ms | — |
| **总耗时** | **14.29 ms** | ✗ **超红线 43%** |

## 结论

1. **切点识别成功**：220/493 → 弧段 273 点，与 golden 的 273 px 弧长完全对上；
   两侧 RMS 均 <0.2 px（远低于 0.15 mm 判据）。算法本身是有效的。
2. **性能未达标**：14.29 ms vs 10 ms 红线。已做的优化：粗到细搜索（-48%）、
   Kåsa 替代 LM 扫描（-89%）、排除 arc+arc（-40%）、圆拟合隔点采样（无效，成本不在圆）。
3. **剩余瓶颈是直线拟合**：每个候选点 2 次 `fit_line`（`np.cov` + `eigh` 全量数组），
   约占总耗时 60%。可行的下一步优化：
   - **积分图（prefix sums）算直线 SSE**：`Σx, Σy, Σx², Σy², Σxy` 可 O(1) 求任意区间的
     协方差矩阵与残差平方和，把直线拟合从 O(n) 降到 O(1)，预计总耗时 ≤5 ms；
   - 或把候选点再抽稀到步长 20（精度风险：切点分辨率降到 20 px）。

## 集成前置条件（M8-4 第二阶段）

- 必须先做「积分图直线 SSE」优化并实测 <10 ms，**否则不得合入主分支**（用户红线）；
- 集成时 `arc_residual` 改为**绝对毫米**口径（配置与日志统一 mm，内部按 DPI 换算 px）；
- 集成后需重跑 `synthetic_polyline_bulge` 的 golden 回归，目标 P95 ≤0.5 px。

## Phase 2：积分图 SSE 优化（性能红线已满足）

### 实现

`PrefixLineSSE`：过质心的 TLS 直线残差平方和 = 散布矩阵最小特征值
`λ_min = (Sxx+Syy)/2 − sqrt(((Sxx−Syy)/2)² + Sxy²)`，
而 `Sxx/Syy/Sxy` 由 `Σx, Σy, Σxx, Σyy, Σxy` 的区间前缀和 O(1) 求出。
扫描阶段的直线 SSE **全部**改走积分图（不再有朴素 `fit_line` 回退路径）；
圆弧仍走 Kåsa（隔点采样），仅最终胜者做 LM 精化。

### 数值一致性（`rtol=1e-5` 判据）

| 区间 | 朴素 SSE | 积分图 SSE | 相对误差 |
|---|---|---|---|
| [0,200) | 9.4675568175 | 9.4675568175 | 7.5e-13 |
| [10,150) | 6.0426375814 | 6.0426375814 | 5.0e-13 |
| [50,90) | 1.2486247837 | 1.2486247837 | 4.3e-14 |
| [3,7) | 0.0972301240 | 0.0972301240 | 0 |

**通过**（最大相对误差 7.5e-13 ≪ 1e-5），已固化为 `tests/test_s4_segment.py::test_integral_sse_matches_naive_fit`。

### 性能（bulge 边：493 点 / 505 px）

| 阶段 | 优化前 | 优化后 |
|---|---|---|
| 扫描 | 13.62 ms | **4.11 ms** |
| LM 精化 | 0.67 ms | 0.78 ms |
| **合计** | **14.29 ms** | **4.89 ms** |

**红线（<10 ms）已满足**，余量 51%。已固化为
`tests/test_s4_segment.py::test_single_edge_latency_under_red_line`。

### 切分结果（优化后不变）

`line(220) + arc(273)`，弧段 273 点 = golden 弧长 273 px，两侧 RMS 0.146 / 0.178 px。

### 管线集成：本轮**未合入**（回滚）

把 `run_s4_fit` 的分段调用换成 `segment_curve` 后，全量测试出现 **4 个失败**：

| 失败用例 | 现象 |
|---|---|
| `test_arc_radius_and_center[60.0]` | 半径/圆心判据超差 |
| `test_arc_radius_and_center[80.0]` | 同上 |
| `test_shared_endpoints_are_bitwise_equal` | 相邻基元端点不再逐位相等 |
| `test_s8_dxf.py::test_line_and_arc_roundtrip` | ARC 半径断言 22.6906 != 期望值 |

原因（初判）：新分段器对**纯弧边**也可能选出 `line+arc` 组合（BIC 惩罚不足以排除），
把一条完整圆弧切成"短直线 + 弧"，进而改变端点与半径。按"数值验证不通过不得集成"的
硬约束，本轮**回滚了 `run_s4_fit` 与配置改动**，只提交经单测验证的模块与 PoC。

### 下一轮（集成）必做

1. 纯弧边保护：`best_split` 之前先做整段圆弧拟合，若整段 RMS ≤ 容差且圆心/半径稳定，
   直接判为单 ARC（不做切分）；
2. `_make_segment` 的端点必须由相邻段共享（复用 S3 节点），保证逐位相等；
3. `arc_residual_mm` + `arc_residual_factor` 废弃告警的配置迁移（本轮已写好补丁但随集成回滚）；
4. 集成后再跑 golden：目标 bulge P95 ≤1.0 px、原子基元恰为 1 LINE + 1 ARC。

## Phase 3：单模型拦截 + 端点吸附（模块内已验证，**集成仍未合入**）

### 已实现并验证

| 修复 | 实现 | 验证 |
|---|---|---|
| 纯弧/纯线被误切 | `single_model_guard`：先试单 LINE（RMS ≤ 容差 → 直接返回），再试单 ARC | `test_perfect_arc_is_not_split`（完美 120° 弧 → 恰 1 个 ARC）、`test_perfect_line_is_not_split` ✓ |
| 端点不逐位相等 | 切分时令 `left[-1] = right[0] = points[k]`（同一浮点坐标） | `test_adjacent_segments_share_bitwise_identical_endpoint`（`==` 断言）✓ |
| 性能 | 积分图直线 SSE + 粗到细（步长 10） | 单边 **4.89 ms** < 10 ms ✓ |

模块级 7 个用例全绿，全量 `pytest` 仍 **340 passed, 1 xfailed**（集成回滚状态）。

### 集成仍然失败的精确原因（比 Phase 2 的记录更细）

再次把 `run_s4_fit` 接到 `segment_curve`（并把 `arc_residual_mm` 接上）后，失败面从 4 个
**扩大到 8 个**，新增 `test_line_endpoint_and_length_accuracy[horizontal/diagonal]` 与
两个 golden 场景。根因不是切分本身，而是**分段之后管线还做了一大串后处理**，
而 `segment_curve` 给出的模型绕过了它们：

| 后处理（`run_s4_fit` 内） | 作用 | 被绕过后的后果 |
|---|---|---|
| `_refit_circle` | 用中心线样本重算圆心/半径/rms | 圆弧半径 22.6906 ≠ 期望（`test_s8_dxf`） |
| `_merge_same_kind` | 同模型相邻段合并 | 段数变化 → 端点/长度判据漂移 |
| `_line_extent` | 自由端延伸到墨迹端帽 | 线段端点缩短 1–2 px → 端点判据失败 |
| 节点吸附 `nodes[...]` | 交汇点吸到共享节点 | 端点不逐位相等 |
| `_enforce_g1` | G1 连续性复检 | 切点处切向跳变未被修正 |

也就是说：**"切分器"与"后处理链"必须一起换**，只换前者等于让后处理作用在语义不同的输入上。

### M8-4 收口建议

1. 把 `segment_curve` 的产物**转成旧 `_Segment` 语义**（`points` 必须是原折线的连续切片、
   不含重复端点），再交给现有后处理链；
2. 或用"适配层"：`segment_curve` 只决定 **切点索引**，模型仍由 `_refit_circle` /
   `fit_line` + `_line_extent` / 节点吸附在 `run_s4_fit` 内完成（推荐，改动最小）。
3. 集成后重跑：目标 bulge P95 ≤1.0 px、原子基元恰为 1 LINE + 1 ARC，且 8 个失败用例转绿。

## Phase 4：Split-Only Oracle（切分预言机）——测试全绿，但 bulge 仍被降级为多段线

### 已完成

1. `src/s4_segment.py` 重构为**只输出切点 + 模型类型**的预言机：
   `plan_segmentation(points, tolerance_px) -> Plan(split_indices=(220,), model_types=("LINE","ARC"))`，
   不再返回任何拟合对象、不做端点吸附。
2. `split_and_fit` 新增 `plan` 参数：按计划切片后，仍用**既有** `fit_line` / `fit_circle` +
   `select_model` 拟合；后续 `_merge_same_kind`、`_line_extent`、节点吸附、`_enforce_g1`
   全部原样执行。
3. 配置迁移：`fitting.arc_residual_mm: 0.15`；`arc_residual_factor` 置空并在设值时抛 `DeprecationWarning`。
4. 单模型拦截在预言机内部生效（完美正弦弧 → 1 段 ARC、完美直线 → 1 段 LINE）。

### 验证

| 项 | 结果 |
|---|---|
| 全量 `pytest` | **343 passed, 1 xfailed** —— **Phase 3 的 8 个失败用例全部转绿** ✓ |
| `check_contracts` | `errors=0 warnings=1` |
| 单边规划耗时 | **< 10 ms**（`test_single_edge_latency_under_red_line` 通过） |
| line_single / arc_120 / circle_full | P95 0.219 / 0.302 / 0.126 px，五维全过 ✓ |
| **bulge** | **1 条 LWPOLYLINE（原子计数 11 个 LINE），P95 = 13.700 px** ✗ |

### 新问题（本轮未能定位完）

直接调用 `split_and_fit(..., plan=plan)` 能得到**正确**的两段：
`[('line', 221), ('circle', 273)]`，弧段 Kåsa rms 仅 0.178 px。
但走完整 `run_pipeline` 后，`fit.primitives` 只有**一条 lwpolyline（474.8 px）**、且 `warnings` 为空。
说明在 `split_and_fit` 返回之后、`primitives` 生成之前的某一步（`_merge_same_kind` 合并、
`_enforce_g1` 复检、或"圆弧张角/半径合理性"检查）把 ARC 段判掉并整条降级成了多段线。
下一轮只需在这一小段里插桩定位，不应再改动预言机与 `split_and_fit`。

### 未执行的步骤（因验收不达标）

- **未**把 `test_bulge_diagnosis` 的 P95 锚点收紧到 ≤1.0（当前实测 13.7，会直接红）。
- **未**合并到 `master`：bulge 的原子基元不是 1 LINE + 1 ARC，P95 反而比 M8-3 的 10.975 更差。

## Phase 5：静默降级的根因定位（已修复根因，但 ARC 朝向仍错）

### 根因（第 3 项问题的答案）

**位置**：`run_s4_fit` 里调用预言机的那一行 + `split_and_fit(plan=...)` 的输入。

**条件**：`plan_segmentation(fit_points, ...)` 吃的是 **DP 简化后**的拟合点
（`fit_points = simplified`，本例只有 **12 个点**）。预言机内部要求
`len(points) >= 2*MIN_POINTS = 20` 才有意义，12 点直接走兜底分支返回
`Plan(split_indices=(), model_types=("LINE",))` —— 整条边被当成一段 LINE。
随后在圆弧/直线降级判断处（`too_rough = model.rms > tolerance_px`，
实测 **rms 42.753 px** vs 1.772 px）被写成 `lwpolyline`，而**只写进了
`primitive.reason`，没有 append 到 `warnings`** —— 这就是"静默降级"。

### 已修复

1. **预言机改用稠密中心线**：`plan_segmentation(centerline, ...)`；
   计划切点多于 1 段时，`split_and_fit` 也改用 `centerline` 拟合（直线/圆弧都更准）。
2. **零静默失败**：两条降级路径现在都会 append 结构化 warning：
   - `LINE_DEGRADED_TO_POLYLINE edge={id} seg={i} rms_px=… tolerance_px=… points=…`
   - `ARC_DEGRADED_TO_POLYLINE edge={id} seg={i} rms_px=… limit_px=… radius_px=… min_radius_px=… reason=ARC_RMS_EXCEEDS_TOLERANCE|ARC_RADIUS_TOO_SMALL`

### 效果

| 项 | 修复前 | 修复后 |
|---|---|---|
| bulge 基元 | 1×lwpolyline（原子 11 LINE） | **1 LINE + 1 ARC** ✓ |
| LINE rms / 长度 | 42.753 px（降级） | **0.146 px** / 214.5 px ✓ |
| ARC rms / r / sweep | — | **0.178 px** / 146.94 px / 107.52°（真值 r=148）✓ |
| **bulge P95** | 13.700 px | **94.137 px** ✗ |

**新问题**：基元数量、类型、半径、残差全部正确，但几何位置与 golden 呈**镜像**
（P95 94 px ≈ 2×矢高），即圆弧凸向与 golden 相反。属于 ARC 朝向/端点角度的
符号问题，与切分器无关。**未合并到 master**（P95 远未达标）。

### 测试状态

`343 passed, 1 xfailed`；`check_contracts` `errors=0 warnings=1`。
三张全绿场景未受影响（0.219 / 0.302 / 0.126 px）。


## Phase 6：ARC 朝向用叉积判定（确定性）

把 `forward` 的判定从"角度展开差值的符号"换成**首/中/末三点的叉积符号**
（`cross_z = (mid−first) × (last−mid)`，图像坐标 y 向下时 θ 递增对应 `cross_z > 0`）。
不做切向投票，点的先后顺序直接来自 S3 轮廓，即拓扑真值。

实测：

| 场景 | 改前 P95 | 改后 P95 |
|---|---|---|
| synthetic_arc_120 | 0.302 px | **0.302 px**（未回退） |
| synthetic_polyline_bulge | 94.137 px | **94.137 px**（未改善） |

**未解决**：bulge 的 ARC 几何仍是镜像。诊断数据：

- 我方 ARC：center=(236.16, 266.23) px、r=146.94 px、sweep=107.52°；
- golden：LWPOLYLINE `(10,10,b=0.5) → (30,10) → (40,25)`，即 R=12.5 mm=147.6 px、
  张角 106.26°，圆心应位于弦**下方** (20, 2.5) mm、顶点 (20, 15) mm；
- 我方圆心换算后 ≈ (20.0, 17.4) mm —— 在弦**上方**，顶点 (20, 5) mm，正好镜像。

由于 `fit_circle` 的圆心由中心线点集决定，而叉积只影响 sweep 方向、不影响圆心，
说明**镜子是在拟合输入那侧形成的**（中心线切片/端点延伸），而不是 sweep 符号。
下一步应核对 S3 该边的点序方向（`edge.points` 从哪一端开始），以及
`_arc_extent` 是否把端点沿错误的圆周方向延伸。

**未合并到 master**（P95 远未达标）。
