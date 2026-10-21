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
