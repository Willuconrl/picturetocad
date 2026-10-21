# scripts/poc_edt_linewidth.py
"""M8-3 PoC：用**欧氏距离变换（EDT）**估计墨宽，验证它能否替代骨架长度法。

背景（M8-2 结论）
-----------------
`area / 骨架长` 在折弯笔画上高估 ~20%：6 px 宽的笔画在 ≥90° 折弯处，形态学骨架会
"切角"，骨架长由 504 px 掉到 420 px（`skeletonize` 与 `thin` 都一样）。
EDT 是**纯局部几何属性**，不依赖骨架长度，理论上不会切角。

统计量的选择
------------
对一个宽 `w` 的笔画，横截面上的 EDT 值近似是 `0.5, 1.5, …, w/2`（像素中心到边界）：
- `2×median` 会被边缘的低值拉低（≈`w/2`）；
- `2×p75` 接近 `w`；
- `2×p90/p95` 更接近 `w` 但受笔画端帽与折角影响。
本脚本把四个统计量都打出来，与合成图名义线宽（6 px）对比，选最稳的那个。

用法：
    python scripts/poc_edt_linewidth.py
    python scripts/poc_edt_linewidth.py --image test_images/synthetic_arc_120.png --nominal 6
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scipy.ndimage import distance_transform_edt  # noqa: E402
from skimage.morphology import skeletonize  # noqa: E402

from src.config import load_config  # noqa: E402
from src.s1_preprocess import run_s1_preprocess  # noqa: E402
from src.s2_binarize import S2Input, binarize_image  # noqa: E402


def edt_statistics(binary: np.ndarray) -> dict[str, float]:
    """对前景像素的 EDT 做多分位统计（距离×2 = 宽度估计）。"""
    foreground = binary > 0
    distance = distance_transform_edt(foreground)
    values = distance[foreground]
    if values.size == 0:
        raise ValueError("二值图为空，无法估计线宽")
    result = {
        "foreground_px": int(values.size),
        "edt_max": float(values.max()),
        "width_median": float(2.0 * np.median(values)),
        "width_p75": float(2.0 * np.percentile(values, 75)),
        "width_p90": float(2.0 * np.percentile(values, 90)),
        "width_p95": float(2.0 * np.percentile(values, 95)),
        "width_max": float(2.0 * values.max()),
    }
    # 参考：旧法（面积 / 骨架长），用来说明新法是否真的更准
    skeleton = skeletonize(foreground)
    skeleton_length = int(skeleton.sum())
    if skeleton_length:
        result["old_area_over_skeleton"] = float(values.size / skeleton_length)
        result["skeleton_length_px"] = skeleton_length
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default=str(ROOT / "test_images" / "synthetic_polyline_bulge.png"))
    parser.add_argument("--nominal", type=float, default=6.0, help="合成图名义线宽（px）")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args(argv)

    cfg = load_config(ROOT / "config.yaml")
    pre, _ = run_s1_preprocess(args.image, cfg)
    s2, _ = binarize_image(S2Input.from_preprocess(pre), cfg)
    stats = edt_statistics(s2.binary)
    stats["image"] = args.image.replace("\\", "/")
    stats["nominal_px"] = args.nominal
    stats["s2_current_linewidth_px"] = round(float(s2.linewidth_px), 4)
    for key in ("width_median", "width_p75", "width_p90", "width_p95", "width_max", "old_area_over_skeleton"):
        if key in stats:
            stats[f"{key}_ratio"] = round(stats[key] / args.nominal, 4)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
