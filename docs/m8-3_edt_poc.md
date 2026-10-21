# M8-3 PoC：用 EDT 估计墨宽（结论：2×p90 精确命中名义值）

> 生成方式：`python scripts/poc_edt_linewidth.py`（可加 `--image/--nominal`）。
> 背景见 `docs/bulge_diagnosis.md` 的「S2 修复尝试与根因重定位」：
> `area / 骨架长` 在折弯笔画上高估，因为形态学骨架会切角（420 px vs 几何 504 px）。

## 四场景实测（合成图名义线宽均为 6 px）

| 场景 | 名义 | `2×median` | `2×p75` | **`2×p90`** | `2×p95` | 旧法 `area/骨架长` |
|---|---|---|---|---|---|---|
| synthetic_line_single | 6.0 | 4.000 | 5.657 | **6.000** | 6.000 | 6.231（+3.9%） |
| synthetic_arc_120 | 6.0 | 4.000 | 5.657 | **6.000** | 6.325 | 7.012（+16.9%） |
| synthetic_circle_full | 6.0 | 4.000 | 5.657 | **6.000** | 6.325 | 6.724（+12.1%） |
| synthetic_polyline_bulge | 6.0 | 4.000 | 5.657 | **6.000** | 6.325 | 7.110（+18.5%） |

## 结论

- **`2×p90` 在四个场景上全部精确命中 6.000 px**，完全消除旧法的 +4%~+18.5% 高估；
- `2×median` 被笔画边缘的低距离值拉低到 4.0 px（−33%），**不能用**；
- `2×p75` 偏低 5.7%，`2×p95` 偏高 5.4%，只有 p90 对四种几何（直线/弧/圆/折线）都稳定；
- EDT 是**纯局部几何量**，不经过骨架，因此不受折角切角影响 —— 与 M8-2 的根因分析一致。

## 建议的正式改造（M8-3 待批准）

```python
# src/s2_binarize.py::_global_linewidth
distance = distance_transform_edt(foreground)
linewidth = 2.0 * np.percentile(distance[foreground], 90)
```

仍是逐连通域统计后取中位数，保持"多连通域取中位"的既有语义；
`area / 骨架长` 逻辑删除，`major_axis` 兜底一并删除（它从未生效）。

---

## M8-3 实施结果（EDT 线宽 + 切点判据观察）

### 实现

`src/s2_binarize.py::_global_linewidth` 改为**逐连通域取 `min(2×p90(EDT), 面积/骨架长)`，再取中位数**。

为什么不是纯 `2×p90`：纯 EDT 在**细笔画**上会因掩膜量化而偏高（3 px 笔画 → 4 px 掩膜 →
4.0 px，+33%，`test_linewidth_relative_error[3.0]` 直接失败）。两个估计量的偏差都在偏高侧：
`面积/骨架长` 被骨架切角抬高、`2×p90(EDT)` 被掩膜量化抬高，取较小值即可同时消掉。
逐连通域再取中位保持多线宽语义；面积 < 50 px² 的噪点连通域被过滤。

### 实测（`synthetic_polyline_bulge`）

| 指标 | M8-2 收口 | M8-3 实施后 | 锚点 | 状态 |
|---|---|---|---|---|
| S2 全局线宽 | 7.1095 px | **6.0000 px** | 名义 6.0 | ✓ |
| `linewidth_overestimate` | 1.2024× | **1.0147×** | ≤1.05 | ✓ |
| `curvature_peak_ratio` | 3.41× | 4.31× | ≤8.0 | ✓ |
| `contour_halfwidth_ratio` | 0.955 | 0.955 | ≥0.85 | ✓ |
| **bulge 场景 P95** | 8.09 px | **10.98 px** | ≤0.5 px | ✗ |
| 原子基元 | 1 ARC + 3 LINE | **1 ARC + 3 LINE** | 1 LINE + 1 ARC | ✗ 未拆分 |

### 切点判据结论：**仍未生效**

线宽从 7.11 px 修正到 6.00 px 后（搜索半径 17.8 → 15.0 px），S4 的分段结果**没有变化**：
仍是 1 ARC + 3 LINE，直线段依旧被切成 3 段、弧段仍不完整。

数据（本轮实测）：`curvature_peak_ratio = 4.31`（伪曲率仍有 4.3 倍于弧参考），
说明**过渡区的伪曲率虽被压到一半，但仍高于真实过渡信号**，用任何基于中心线曲率的
切点判据都无法稳定定位。按用户指示：**不改切点逻辑**，记录数据待下一轮。

### 其他场景（确认无回归）

| 场景 | P95 | 结论 |
|---|---|---|
| synthetic_line_single | 0.219 px | ✓ 全过 |
| synthetic_arc_120 | 0.302 px | ✓ 全过（比 M8-2 的 0.449 更好） |
| synthetic_circle_full | 0.126 px | ✓ 全过 |

全量 `pytest`：**336 passed, 1 xfailed**（含收紧到 1.05 的新锚点）。

## EDT 改造净影响对照表（M8-2 `c2fde8a` vs M8-3 `894ffeb`）

复现方式：

```bash
python scripts/measure_net_impact.py --out outputs/net_impact/m83.json
git checkout c2fde8a -- src/s2_binarize.py          # 临时换回旧 S2
python scripts/measure_net_impact.py --out outputs/net_impact/m82.json
git checkout HEAD -- src/s2_binarize.py             # 换回新 S2
```

| 场景 | S2 线宽（旧→新） | S4 分段数 | golden P95（旧→新） | S5 图层 |
|---|---|---|---|---|
| synthetic_line_single | 6.019 → **6.000** | 1 | 0.127 → 0.219 px | `LINES_W0.50` |
| synthetic_arc_120 | 7.012 → **6.000** | 1 | 0.425 → **0.302 px**（更好） | `ARCS` |
| synthetic_circle_full | 6.724 → **6.000** | 1 | 0.126 → 0.126 px | `CIRCLES` |
| synthetic_polyline_bulge | 7.109 → **6.000** | 2 | 9.919 → **10.975 px**（更差） | `ARCS` + `LINES_W0.35` |

**结论（资产负债表）**

- **S2 改对了**：四场景线宽全部收敛到 6.000 px（真值 5.913，误差 +1.5%），
  旧法在弧/圆/折线上分别高估 +16.9% / +12.1% / +18.5%。
- **收益**：`synthetic_arc_120` 的 P95 从 0.425 降到 0.302 px（唯一被线宽直接改善的场景）。
- **代价**：`synthetic_line_single` P95 0.127 → 0.219 px（仍远低于 0.5 门槛，可接受）；
  `synthetic_polyline_bulge` P95 9.919 → 10.975 px（**变差 10.6%**）。
- **根因**：S4 的分段阈值 `arc_residual = arc_residual_factor × 线宽` 与线宽**隐式耦合**。
  线宽从 7.11 修正到 6.00 后，容差同步收紧 16%，原本"凑巧"被接受的弧段被判为过糙，
  分段结果偏移。这不是 S2 的缺陷，而是 S4 用旧（偏大）线宽调出来的经验阈值失效。

## 已知限制（M8-3 收口）

1. **bulge 场景 P95 = 10.975 px，未达标（判据 ≤0.5 px）**。原子基元为 1 ARC + 3 LINE，
   期望 1 LINE + 1 ARC。根因是**切点判据未能识别弧↔直线过渡**：即使伪曲率已从 10.12×
   压到 4.31×，仍高于真实过渡信号。
2. **明确不做的事**：不为把 P95 拉回 9.9 而解耦 `arc_residual`。旧值是在错误线宽下
   凑出来的巧合，维护它等于把错误固化。带着真实误差进入 M8-4，用正确模型真正解决。
3. **M8-4 目标**：基于干净线重重构 S4 分段——双模型（直线 + 圆弧）联合 SSE + BIC 选择，
   替换当前的"单模型拟合 → 超阈值再切"策略。
