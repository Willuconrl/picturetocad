# scripts/diagnose_corner.py
"""M9-1 诊断：骨架在 line↔line 拐点处的内缩（chamfer）随拐角角度的变化。

做法：对每个角度画一条"两段直线成角"的合成图 → S1/S2/S3 → 找到拐点节点 →
量它到**解析交点**（两侧骨架末段直线拟合的交点）的距离。输出表格。

用法：
    python scripts/diagnose_corner.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.s2_binarize import S2Input, binarize_image  # noqa: E402
from src.s3_skeleton import run_s3_skeleton  # noqa: E402

MM_PER_PX = 25.4 / 300.0
WINDOW_PX = 18.0


def _fit_line(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = points.mean(axis=0)
    centred = points - center
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    direction = vt[0]
    return center, direction / np.linalg.norm(direction)


def _intersect(line_a, line_b) -> np.ndarray:
    (p1, d1), (p2, d2) = line_a, line_b
    matrix = np.array([[d1[0], -d2[0]], [d1[1], -d2[1]]])
    rhs = p2 - p1
    t = np.linalg.solve(matrix, rhs)
    return p1 + t[0] * d1


def _corner_offset(angle_deg: float, cfg) -> tuple[float, float, int]:
    """返回 (chamfer_offset_px, vertex_angle_deg, node_count)。"""
    width, height = 480, 360
    vertex = np.array([240.0, 180.0])
    arm = 150.0
    half = math.radians(angle_deg) / 2.0
    # 两条臂关于竖直方向对称：一条朝向 -x，一条朝向 +x 方向下压
    p1 = vertex + arm * np.array([-math.sin(half), -math.cos(half)])
    p2 = vertex + arm * np.array([math.sin(half), -math.cos(half)])
    canvas = np.full((height, width), 255, np.uint8)
    cv2.line(canvas, tuple(np.round(p1).astype(int)), tuple(np.round(vertex).astype(int)), 0, 5, cv2.LINE_AA)
    cv2.line(canvas, tuple(np.round(vertex).astype(int)), tuple(np.round(p2).astype(int)), 0, 5, cv2.LINE_AA)
    s2, _ = binarize_image(
        S2Input.from_array(canvas.astype(np.float32) / 255.0, dpi=300.0), cfg
    )
    s3, _ = run_s3_skeleton(s2.binary, cfg, dpi=300.0, linewidth_px=s2.linewidth_px)
    graph = s3.graph
    corners = [n for n in graph.nodes if n.type == "corner"]
    if len(corners) != 1:
        # 没有拐点时退回"离真交点最近的节点"
        corners = sorted(graph.nodes, key=lambda n: math.hypot(n.x - vertex[0], n.y - vertex[1]))[:1]
    corner = corners[0]
    # 两侧直线拟合：取拐点相邻的边，各自靠近拐点的一段
    sides = []
    for edge in graph.edges:
        pts = np.asarray(edge.points, dtype=np.float64)
        if len(pts) < 3:
            continue
        d0 = math.hypot(pts[0][0] - corner.x, pts[0][1] - corner.y)
        d1 = math.hypot(pts[-1][0] - corner.x, pts[-1][1] - corner.y)
        near = pts[: len(pts) // 2] if d0 < d1 else pts[len(pts) // 2:]
        near = np.asarray([p for p in near if math.hypot(p[0] - corner.x, p[1] - corner.y) >= WINDOW_PX * 0.5])
        if len(near) >= 3:
            sides.append(_fit_line(near))
    if len(sides) < 2:
        return float("nan"), angle_deg, len(graph.nodes)
    intersection = _intersect(sides[0], sides[1])
    offset = float(np.hypot(intersection[0] - corner.x, intersection[1] - corner.y))
    return offset, angle_deg, len(graph.nodes)


def main() -> int:
    cfg = load_config(ROOT / "config.yaml")
    print(f"{'angle_deg':>9} | {'chamfer_offset_mm':>18} | {'chamfer_offset_px':>17}")
    print("-" * 52)
    rows = []
    for angle in (30.0, 60.0, 90.0, 120.0, 150.0):
        offset_px, angle_used, _ = _corner_offset(angle, cfg)
        rows.append((angle_used, offset_px * MM_PER_PX, offset_px))
        print(f"{angle_used:9.1f} | {offset_px * MM_PER_PX:18.4f} | {offset_px:17.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
