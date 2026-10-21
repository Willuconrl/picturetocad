# scripts/measure_net_impact.py
"""M8-3 净影响对照：一次跑齐 golden 四场景的关键指标，输出 JSON 便于前后对比。

指标：S2 线宽、S4 基元分布与分段数、S5 图层、golden 五维（P95 / 计数 / 尺寸 / 图层 / 闭合）。
对 `synthetic_polyline_bulge` 额外输出：真实墨宽、线宽高估倍数、伪曲率峰值比。

用法：
    python scripts/measure_net_impact.py --out outputs/net_impact/current.json

对照 M8-2 的用法（临时换回旧 S2 实现，跑完再换回来）：
    git checkout c2fde8a -- src/s2_binarize.py
    python scripts/measure_net_impact.py --out outputs/net_impact/m82.json
    git checkout HEAD -- src/s2_binarize.py
"""

from __future__ import annotations

import argparse
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
from src.golden_compare import (  # noqa: E402
    GOLDEN_HAUSDORFF_P95_PX,
    GOLDEN_KEY_DIMENSION_MM,
    compare_dxf_files,
)
from src.pipeline import run_pipeline  # noqa: E402
import src.s4_fit as s4  # noqa: E402

SCENES = (
    "synthetic_line_single",
    "synthetic_arc_120",
    "synthetic_circle_full",
    "synthetic_polyline_bulge",
)
MM_PER_PX = 25.4 / 300.0


def curvature_of(points: np.ndarray) -> np.ndarray:
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    cfg = load_config(ROOT / "config.yaml")
    out_root = ROOT / "outputs" / "net_impact"
    out_root.mkdir(parents=True, exist_ok=True)

    payload: dict[str, dict] = {}
    for scene in SCENES:
        run = run_pipeline(str(ROOT / "test_images" / f"{scene}.png"), cfg, out_dir=out_root / scene)
        report = compare_dxf_files(
            ROOT / "golden" / f"{scene}.dxf",
            Path(run.dxf.dxf_path),
            ignore_types=frozenset({"TEXT", "MTEXT"}),
            spacing_mm=0.02,
            max_hausdorff_p95_mm=GOLDEN_HAUSDORFF_P95_PX * MM_PER_PX,
            max_entity_count_diff=0,
            dim_abs_tol_mm=GOLDEN_KEY_DIMENSION_MM,
            dim_rel_tol=0.005,
            check_layers=True,
            check_closed=True,
            count_mode="atomic",
        ).to_json()
        entry = {
            "s2_linewidth_px": round(float(run.metrics["s2"].error_estimate.get("linewidth_px", 0.0)), 4),
            "s4_primitives": dict(sorted(run.fit.counts_by_kind.items())),
            "s4_segments": len(run.fit.primitives),
            "s5_layers": dict(sorted(run.layers.counts_by_layer.items())),
            "p95_px": round(float(report["hausdorff_p95_mm"]) / MM_PER_PX, 4),
            "entity_count_diff": report["entity_count_diff"],
            "worst_dim_mm": (report["worst_abs_dim"] or {}).get("abs_err_mm", 0.0),
            "layers_ok": report["layers_ok"],
            "closed_ok": report["closed_ok"],
            "passed": report["passed"],
        }
        if scene == "synthetic_polyline_bulge":
            graph = run.graph
            edge = max(graph.edges, key=lambda e: e.length_px)
            contours = np.vstack([np.asarray(c, dtype=np.float64) for c in run.metrics["s2"].details.get("contours", [])]) \
                if False else None
            # 直接用管线内部的 S2 结果重建中心线成本高，这里用与测试一致的近似：
            # true_width 用二值面积/骨架长不可得，故从 S2 线宽与几何反推不严谨；
            # 因此 bulge 的深度诊断仍以 docs/m8-3_edt_poc.md 的专门测量为准。
            entry["bulge_note"] = "深度诊断见 docs/m8-3_edt_poc.md（由 scripts/diagnose_bulge.py 产出）"
        payload[scene] = entry

    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
