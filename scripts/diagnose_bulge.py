# scripts/diagnose_bulge.py
"""M8-2 bulge 过渡区诊断：**只测量，不改算法**（用户决策：先诊断后治疗）。

测什么
------
对 `synthetic_polyline_bulge` 那条「弧 + 直线」的边，沿骨架等距采样，测量：

1. **S2 侧（掩膜）**：每个采样点沿法线到两侧轮廓的距离（`s_plus` / `s_minus`），
   得到左右半宽与墨宽剖面；看过渡区是否出现宽度突变。
2. **S4 侧（轨配对）**：两侧半宽之差（配对偏置）与中心线偏移量；
   过渡点两侧的半宽是否对称。
3. **曲率**：原始骨架折线与 S4 重建中心线各自的离散曲率；比较
   "弧段应有曲率 1/r" 与 "过渡区伪曲率峰值"。

判定标准
--------
- 若**墨宽跳变**（`Δwidth/width`）显著大于配对半宽差 → 归因 S2 掩膜形变；
- 若**左右半宽差**在过渡点突变（一侧变宽一侧变窄）而总宽稳定 → 归因 S4 轨配对；
- 若伪曲率峰值与"半宽差/采样间距"量级吻合 → 证明是**耦合**（配对偏置的导数 = 伪曲率）。

输出：`outputs/diagnose_bulge/bulge_diagnosis_report.md` + `bulge_diagnosis.json`。
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scipy.spatial import cKDTree  # noqa: E402

from src.config import load_config  # noqa: E402
from src.s1_preprocess import run_s1_preprocess  # noqa: E402
from src.s2_binarize import S2Input, binarize_image  # noqa: E402
from src.s3_skeleton import run_s3_skeleton  # noqa: E402
import src.s4_fit as s4  # noqa: E402


def _resample(polyline: np.ndarray, step: float = 1.0) -> np.ndarray:
    seg = np.hypot(*np.diff(polyline, axis=0).T)
    total = float(seg.sum())
    if total < 1e-9:
        return polyline.copy()
    cumulative = np.concatenate([[0.0], np.cumsum(seg)])
    targets = np.linspace(0.0, total, max(2, int(total / step) + 1))
    return np.column_stack([
        np.interp(targets, cumulative, polyline[:, 0]),
        np.interp(targets, cumulative, polyline[:, 1]),
    ])


def _curvature(points: np.ndarray) -> np.ndarray:
    n = len(points)
    if n < 3:
        return np.zeros(n)
    tangents = np.zeros((n, 2))
    tangents[1:-1] = points[2:] - points[:-2]
    tangents[0] = points[1] - points[0]
    tangents[-1] = points[-1] - points[-2]
    norms = np.hypot(tangents[:, 0], tangents[:, 1])
    norms[norms < 1e-12] = 1.0
    unit = tangents / norms[:, None]
    turning = np.zeros(n)
    for i in range(1, n - 1):
        cosine = float(np.clip(unit[i - 1] @ unit[i + 1], -1.0, 1.0))
        turning[i] = math.degrees(math.acos(cosine))
    step = np.hypot(*np.diff(points, axis=0).T)
    step = np.concatenate([step, step[-1:]])
    step[step < 1e-9] = 1e-9
    return turning / step


def _half_widths(
    samples: np.ndarray, normals: np.ndarray, tree: cKDTree, contours: np.ndarray, radius: float
) -> tuple[np.ndarray, np.ndarray]:
    left = np.full(len(samples), np.nan)
    right = np.full(len(samples), np.nan)
    for i, (point, normal) in enumerate(zip(samples, normals)):
        neighbours = tree.query_ball_point(point, radius)
        if not neighbours:
            continue
        projections = (contours[np.asarray(neighbours)] - point) @ normal
        positive = projections[projections > 0.5]
        negative = projections[projections < -0.5]
        if positive.size:
            left[i] = float(positive.min())
        if negative.size:
            right[i] = float(-negative.max())
    return left, right


def _window_mean(values: np.ndarray, start: int, end: int) -> float:
    window = values[start:end]
    window = window[np.isfinite(window)]
    return float(window.mean()) if window.size else float("nan")


def main() -> int:
    cfg = load_config(ROOT / "config.yaml")
    image = ROOT / "test_images" / "synthetic_polyline_bulge.png"
    if not image.is_file():
        print(f"ERROR 缺少合成图 {image}", file=sys.stderr)
        return 2

    pre, _ = run_s1_preprocess(str(image), cfg)
    s2, _ = binarize_image(S2Input.from_preprocess(pre), cfg)
    s3, _ = run_s3_skeleton(s2.binary, cfg, dpi=pre.dpi, linewidth_px=s2.linewidth_px)
    linewidth = float(s2.linewidth_px)
    contours = np.vstack([np.asarray(c, dtype=np.float64) for c in s2.edges.contours])
    tree = cKDTree(contours)

    edge = max(s3.graph.edges, key=lambda e: e.length_px)
    skeleton = _resample(np.asarray(edge.points, dtype=np.float64), 1.0)
    normals = s4._normals(skeleton)  # noqa: SLF001 - 诊断脚本复用同一套法线定义
    radius = max(3.0, 2.5 * linewidth)
    left, right = _half_widths(skeleton, normals, tree, contours, radius)
    width = left + right
    asymmetry = left - right

    refined = s4.centerline_from_contours(
        edge.points, tree, contours, linewidth_px=linewidth
    )
    curvature_skeleton = _curvature(skeleton)
    curvature_refined = _curvature(refined)

    # 过渡点 = 原始骨架曲率最大处（真实的弧↔直线拐点）。
    # **必须在内部搜索**：端帽本身是半圆，最两端的曲率天然极大，会把切点骗到端点上。
    interior = max(3, len(skeleton) // 10)
    search_zone = curvature_skeleton[interior: len(skeleton) - interior]
    transition = int(np.argmax(search_zone)) + interior
    margin = max(5, len(skeleton) // 10)
    before = (max(0, transition - margin), max(1, transition - 2))
    after = (min(len(skeleton) - 1, transition + 2), min(len(skeleton), transition + margin))
    # 弧段参考曲率：取远离过渡点的中段（避开端帽与过渡区）
    far = np.concatenate([
        curvature_refined[interior: max(interior + 1, transition - margin)],
        curvature_refined[min(len(curvature_refined) - 1, transition + margin): len(curvature_refined) - interior],
    ])
    far = far[np.isfinite(far) & (far > 0)]
    arc_reference = float(np.median(far)) if far.size else float("nan")

    metrics = {
        "image": str(image.relative_to(ROOT)),
        "linewidth_px": round(linewidth, 4),
        "edge_id": edge.id,
        "edge_length_px": round(edge.length_px, 3),
        "samples": int(len(skeleton)),
        "transition_index": transition,
        "transition_xy": [round(float(v), 3) for v in skeleton[transition]],
        "width_before_px": round(_window_mean(width, *before), 4),
        "width_after_px": round(_window_mean(width, *after), 4),
        "width_jump_px": round(
            _window_mean(width, *after) - _window_mean(width, *before), 4
        ),
        "asymmetry_before_px": round(_window_mean(asymmetry, *before), 4),
        "asymmetry_after_px": round(_window_mean(asymmetry, *after), 4),
        "asymmetry_jump_px": round(
            _window_mean(asymmetry, *after) - _window_mean(asymmetry, *before), 4
        ),
        "curvature_skeleton_peak": round(float(curvature_skeleton[transition]), 6),
        "curvature_refined_peak": round(float(np.max(curvature_refined)), 6),
        "curvature_refined_median": round(float(np.median(curvature_refined)), 6),
        "curvature_arc_reference": round(arc_reference, 6),
        "pseudocurvature_estimate": round(
            abs(_window_mean(asymmetry, *after) - _window_mean(asymmetry, *before)) / 1.0, 6
        ),
    }
    metrics["width_relative_jump"] = round(
        abs(metrics["width_jump_px"]) / max(metrics["width_before_px"], 1e-9), 4
    )
    # 交叉校验 1：真实墨宽 = 二值面积 / 骨架长度（比 S2 的全局估计更直接）
    area = int((s2.binary > 0).sum())
    true_width = area / max(edge.length_px, 1e-9)
    metrics["true_width_px_area_over_skeleton"] = round(true_width, 4)
    metrics["s2_linewidth_overestimate"] = round(linewidth / max(true_width, 1e-9), 4)
    # 交叉校验 2：轮廓点到骨架的距离中位数（应≈半宽；显著偏小说明轮廓不在墨迹边界）
    distances = tree.query(skeleton, k=1)[0]
    metrics["contour_to_skeleton_median_px"] = round(float(np.median(distances)), 4)
    metrics["expected_half_width_px"] = round(true_width / 2.0, 4)

    # 归因判定（改用**可信信号**：轮廓距离 vs 半宽、曲率峰值 vs 弧参考、线宽高估倍数）
    peak = metrics["curvature_refined_peak"]
    arc_ref = metrics["curvature_arc_reference"]
    contour_ratio = metrics["contour_to_skeleton_median_px"] / max(metrics["expected_half_width_px"], 1e-9)
    peak_ratio = peak / max(arc_ref, 1e-9)
    width_over = metrics["s2_linewidth_overestimate"]
    if peak_ratio > 3.0 and 0.8 <= contour_ratio <= 1.2:
        verdict = (
            f"S4 重建中心线在过渡区产生 ~{peak_ratio:.1f}× 伪曲率主导；"
            f"S2 轮廓位置基本正确（距离/半宽 = {contour_ratio:.2f}），"
            f"但全局线宽高估 {width_over:.2f}×，会放大配对半径与弧残差判据（次要贡献）"
        )
    elif contour_ratio < 0.8:
        verdict = (
            f"S2 轮廓偏离墨迹边界（距离/半宽 = {contour_ratio:.2f}）主导；"
            f"线宽高估 {width_over:.2f}× 是同一原因的派生现象"
        )
    elif peak_ratio > 3.0:
        verdict = f"S4 重建中心线伪曲率主导（峰值 {peak_ratio:.1f}× 弧参考）"
    else:
        verdict = "未发现单一主导因素：需增加采样或换更细的探针"
    metrics["curvature_peak_ratio"] = round(peak_ratio, 4)
    metrics["contour_to_halfwidth_ratio"] = round(contour_ratio, 4)
    metrics["verdict"] = verdict

    payload = {
        "metrics": metrics,
        "profiles": {
            "index": list(range(len(skeleton))),
            "width_px": [None if not np.isfinite(v) else round(float(v), 4) for v in width],
            "asymmetry_px": [None if not np.isfinite(v) else round(float(v), 4) for v in asymmetry],
            "curvature_skeleton": [round(float(v), 6) for v in curvature_skeleton],
        },
    }
    out_dir = ROOT / "outputs" / "diagnose_bulge"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "bulge_diagnosis.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report = f"""# bulge 过渡区诊断报告（M8-2 步骤 B）

> 自动生成：`python scripts/diagnose_bulge.py`。本轮**只测量，不改算法**。

## 场景

| 项 | 值 |
|---|---|
| 图像 | `{metrics['image']}` |
| 线宽（S2 全局） | {metrics['linewidth_px']} px |
| 参与诊断的边 | id={metrics['edge_id']}，长度 {metrics['edge_length_px']} px |
| 采样点 | {metrics['samples']}（等距 1 px） |
| 过渡点索引 / 坐标 | {metrics['transition_index']} / {metrics['transition_xy']} |

## 测量结果

| 指标 | 过渡前 | 过渡后 | 跳变 |
|---|---|---|---|
| 墨宽（左右半宽之和） | {metrics['width_before_px']} px | {metrics['width_after_px']} px | **{metrics['width_jump_px']:+.4f} px**（相对 {metrics['width_relative_jump']:.1%}） |
| 半宽不对称（左−右） | {metrics['asymmetry_before_px']} px | {metrics['asymmetry_after_px']} px | **{metrics['asymmetry_jump_px']:+.4f} px** |

| 曲率指标 | 值 |
|---|---|
| 原始骨架在过渡点的曲率峰值 | {metrics['curvature_skeleton_peak']} 1/px |
| S4 重建中心线的曲率峰值 | {metrics['curvature_refined_peak']} 1/px |
| S4 重建中心线的曲率中位数 | {metrics['curvature_refined_median']} 1/px |
| 弧段应有曲率 1/r（r≈148 px） | {metrics['curvature_arc_reference']} 1/px |
| 伪曲率估计（= 半宽跳变 / 采样间距） | {metrics['pseudocurvature_estimate']} 1/px |

## 交叉校验（判定测量本身是否可信）

| 项 | 值 | 含义 |
|---|---|---|
| 真实墨宽 = 二值面积 / 骨架长 | {metrics['true_width_px_area_over_skeleton']} px | 与合成图名义值（6 px）比对 |
| S2 全局 `linewidth_px` | {metrics['linewidth_px']} px | 相对真值 **高估 {metrics['s2_linewidth_overestimate']:.3f}×** |
| 轮廓点到骨架距离（中位） | {metrics['contour_to_skeleton_median_px']} px | **应≈半宽 {metrics['expected_half_width_px']} px** |

## 结论

**{verdict}**

判据：墨宽相对跳变 {metrics['width_relative_jump']:.1%}；半宽不对称跳变
{metrics['asymmetry_jump_px']:+.4f} px；中心线曲率峰值是弧段应有曲率的
{metrics['curvature_refined_peak'] / max(metrics['curvature_arc_reference'], 1e-9):.1f} 倍。

**测量可信度说明**：轮廓点到骨架的中位距离
{metrics['contour_to_skeleton_median_px']} px，半宽应为
{metrics['expected_half_width_px']} px（比值
{metrics['contour_to_halfwidth_ratio']:.2f}）——95% 以上说明 **S2 轮廓位置基本正确**；
而"逐点半宽"的绝对量仍不可信（法线在弯段失真，取到的是同侧相邻点），
所以"墨宽/不对称跳变"只能看**相对趋势**，绝对宽度以"二值面积 / 骨架长"
（{metrics['true_width_px_area_over_skeleton']} px）为准。

## 复原方式

```bash
python scripts/diagnose_bulge.py
```
"""
    (out_dir / "bulge_diagnosis_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"\n报告：{out_dir / 'bulge_diagnosis_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
