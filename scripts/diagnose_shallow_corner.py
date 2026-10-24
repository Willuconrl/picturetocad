# scripts/diagnose_shallow_corner.py
"""M9-4.1 诊断：S3 `split_corners` 对**浅角折线**（大内角）是否分裂，以及小半径弧的风险。

对两组几何各跑一遍 S1→S3：
  A. 折线：内角 120/135/150/160/170/175°（真值应有拐点）
  B. 圆弧：半径 8/12/20/40 px（**不应**被切成拐点）

输出每个 case 的节点数、边数、corner 节点数，用于确定阈值与"邻域对比"判据。
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

WIDTH, HEIGHT = 520, 400
LINE_WIDTH = 5


def _run(canvas: np.ndarray, cfg) -> tuple[int, int, int]:
    s2, _ = binarize_image(S2Input.from_array(canvas.astype(np.float32) / 255.0, dpi=300.0), cfg)
    s3, _ = run_s3_skeleton(s2.binary, cfg, dpi=300.0, linewidth_px=s2.linewidth_px)
    corners = sum(1 for node in s3.graph.nodes if node.type == "corner")
    return len(s3.graph.nodes), len(s3.graph.edges), corners


def polyline_canvas(interior_deg: float) -> np.ndarray:
    canvas = np.full((HEIGHT, WIDTH), 255, np.uint8)
    vertex = np.array([WIDTH / 2, HEIGHT * 0.75])
    arm = 160.0
    half = math.radians(interior_deg) / 2.0
    p1 = vertex + arm * np.array([-math.sin(half), -math.cos(half)])
    p2 = vertex + arm * np.array([math.sin(half), -math.cos(half)])
    for point in (p1, p2):
        cv2.line(canvas, tuple(np.round(vertex).astype(int)), tuple(np.round(point).astype(int)), 0, LINE_WIDTH, cv2.LINE_AA)
    return canvas


def arc_canvas(radius: float) -> np.ndarray:
    canvas = np.full((HEIGHT, WIDTH), 255, np.uint8)
    center = (int(WIDTH / 2), int(HEIGHT / 2))
    cv2.ellipse(canvas, center, (int(radius), int(radius)), 0.0, 200.0, 340.0, 0, LINE_WIDTH, cv2.LINE_AA)
    return canvas


def main() -> int:
    cfg = load_config(ROOT / "config.yaml")
    print(f"{'case':>10} | {'nodes':>5} | {'edges':>5} | {'corners':>7}")
    print("-" * 38)
    for interior in (120.0, 135.0, 150.0, 160.0, 170.0, 175.0):
        nodes, edges, corners = _run(polyline_canvas(interior), cfg)
        print(f"poly{int(interior):>5}° | {nodes:5d} | {edges:5d} | {corners:7d}")
    for radius in (8.0, 12.0, 20.0, 40.0):
        nodes, edges, corners = _run(arc_canvas(radius), cfg)
        print(f"arc r={int(radius):>3} | {nodes:5d} | {edges:5d} | {corners:7d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
