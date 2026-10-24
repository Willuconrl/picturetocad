# scripts/probe_seam_bias.py
"""M9-2 Phase 2 探针：量测 bulge 场景 LINE↔ARC 接缝处的**模型偏差结构**（只读）。

不做任何修改，只输出：
- 原始骨架顶点（oracle 切点，中心线在切点处的坐标）
- 两侧独立拟合后的端点
- 直线方程 ∩ 圆方程的两个候选交点
- 切点邻域的局部曲率极小点（过渡点的近似观测）
- 各候选点到上述参考点的距离（mm）、以及每个候选点处的切向夹角（°）

用法：
    python scripts/probe_seam_bias.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scipy.spatial import cKDTree  # noqa: E402

from src.config import load_config  # noqa: E402
from src.geom_utils import line_circle_intersections  # noqa: E402
from src.s1_preprocess import run_s1_preprocess  # noqa: E402
from src.s2_binarize import S2Input, binarize_image  # noqa: E402
from src.s3_skeleton import run_s3_skeleton  # noqa: E402
from src.s4_segment import plan_segmentation  # noqa: E402
import src.s4_fit as s4  # noqa: E402

MM_PER_PX = 25.4 / 300.0


def _curvature(points: np.ndarray) -> np.ndarray:
    n = len(points)
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


def main() -> int:
    cfg = load_config(ROOT / "config.yaml")
    pre, _ = run_s1_preprocess(str(ROOT / "test_images" / "synthetic_polyline_bulge.png"), cfg)
    s2, _ = binarize_image(S2Input.from_preprocess(pre), cfg)
    s3, _ = run_s3_skeleton(s2.binary, cfg, dpi=pre.dpi, linewidth_px=s2.linewidth_px)
    contours = np.vstack([np.asarray(c, dtype=np.float64) for c in s2.edges.contours])
    tree = cKDTree(contours)
    edge = max(s3.graph.edges, key=lambda e: e.length_px)
    centerline = s4.centerline_from_contours(
        edge.points, tree, contours, linewidth_px=s2.linewidth_px
    )
    tolerance_px = 0.15 / MM_PER_PX
    plan = plan_segmentation(centerline, tolerance_px=tolerance_px)
    k = plan.split_indices[0]

    left_pts = centerline[: k + 1]
    right_pts = centerline[k:]
    shared = np.asarray(centerline[k], dtype=np.float64)
    line_model = s4.fit_line(left_pts)
    circle_model = s4.fit_circle(right_pts)

    curvature = _curvature(centerline)
    window = 10
    lo = max(1, k - window)
    hi = min(len(centerline) - 1, k + window)
    extremum_index = lo + int(np.argmin(curvature[lo:hi]))
    extremum = np.asarray(centerline[extremum_index], dtype=np.float64)

    normal = np.asarray(line_model.normal, dtype=np.float64)
    hits = line_circle_intersections(
        tuple(np.asarray(line_model.point, dtype=np.float64)),
        (-normal[1], normal[0]),
        tuple(np.asarray(circle_model.center, dtype=np.float64)),
        float(circle_model.radius),
    )

    def mm(point: np.ndarray) -> float:
        return float(np.hypot(point[0] - shared[0], point[1] - shared[1])) * MM_PER_PX

    print(f"seam index           : {k} / {len(centerline)}")
    print(f"original skeleton    : ({shared[0]:.3f}, {shared[1]:.3f}) px")
    print(f"fitted LINE endpoint : ({left_pts[-1][0]:.3f}, {left_pts[-1][1]:.3f}) px  "
          f"dist_to_shared={mm(np.asarray(left_pts[-1])):.4f} mm")
    print(f"fitted ARC endpoint  : ({right_pts[0][0]:.3f}, {right_pts[0][1]:.3f}) px  "
          f"dist_to_shared={mm(np.asarray(right_pts[0])):.4f} mm")
    print(f"local curvature min  : idx={extremum_index} ({extremum[0]:.3f}, {extremum[1]:.3f}) px  "
          f"dist_to_shared={mm(extremum):.4f} mm")
    print(f"line rms={line_model.rms:.4f}px  circle rms={circle_model.rms:.4f}px  "
          f"r={circle_model.radius:.3f}px")
    tangent = np.array([-normal[1], normal[0]])
    for index, point in enumerate(hits):
        radial = point - np.asarray(circle_model.center, dtype=np.float64)
        arc_tangent = np.array([-radial[1], radial[0]])
        arc_tangent = arc_tangent / max(float(np.linalg.norm(arc_tangent)), 1e-9)
        cosine = float(np.clip(abs(arc_tangent @ tangent), -1.0, 1.0))
        angle = math.degrees(math.acos(cosine))
        print(
            f"intersection[{index}] : ({point[0]:.3f}, {point[1]:.3f}) px  "
            f"dist_to_shared={mm(point):.4f} mm  "
            f"dist_to_curv={float(np.hypot(*(point - extremum))) * MM_PER_PX:.4f} mm  "
            f"tangent_angle={angle:.2f}°"
        )
    if not hits:
        print("intersection         : none (0 solutions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
