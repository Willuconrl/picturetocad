# scripts/golden_compare.py
"""golden 回归 CLI：拿管线输出的 DXF 与解析 golden 比五维指标。

用法：
    python scripts/golden_compare.py --actual outputs/run1/drawing.dxf \
        --golden golden/synthetic_line_single.dxf --dpi 300
    python scripts/golden_compare.py --scene-dir outputs/golden --golden-dir golden --dpi 300

退出码：0 全部通过；1 有判据未通过；2 输入/参数错误。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.golden_compare import (  # noqa: E402
    GOLDEN_HAUSDORFF_P95_PX,
    GOLDEN_KEY_DIMENSION_MM,
    GoldenError,
    compare_dxf_files,
)

#: 与 SPEC 的关键尺寸判据一致：绝对 0.05mm + 相对 0.5%
DIM_REL_TOL = 0.005

#: 采样步长：比 1px（@300DPI≈0.085mm）细一个量级
SPACING_MM = 0.02


def compare_one(actual: Path, golden: Path, dpi: float, *, strict: bool) -> dict:
    mm_per_px = 25.4 / dpi
    hausdorff_mm = (GOLDEN_HAUSDORFF_P95_PX if strict else 1.0) * mm_per_px
    dim_tol = GOLDEN_KEY_DIMENSION_MM if strict else 0.1
    result = compare_dxf_files(
        golden,
        actual,
        ignore_types=frozenset({"TEXT", "MTEXT"}),
        spacing_mm=SPACING_MM,
        max_hausdorff_p95_mm=hausdorff_mm,
        max_entity_count_diff=0,
        dim_abs_tol_mm=dim_tol,
        dim_rel_tol=DIM_REL_TOL,
        check_layers=True,
        check_closed=True,
        count_mode="atomic",
    )
    payload = result.to_json()
    payload["scene"] = golden.stem
    payload["actual"] = str(actual)
    payload["golden"] = str(golden)
    payload["hausdorff_p95_px"] = round(result.hausdorff_p95_mm / mm_per_px, 6)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actual", help="管线输出的 DXF")
    parser.add_argument("--golden", help="golden DXF")
    parser.add_argument("--scene-dir", help="批量模式：管线输出目录（含 *.dxf）")
    parser.add_argument("--golden-dir", help="批量模式：golden 目录")
    parser.add_argument("--dpi", type=float, default=300.0, help="输入图像 DPI（默认 300）")
    parser.add_argument("--loose", action="store_true", help="用 SPEC 原值（真实图）而非合成图收紧值")
    parser.add_argument("--json-out", default=None, help="把结果写到该 JSON 文件")
    args = parser.parse_args(argv)

    strict = not args.loose
    try:
        if args.scene_dir:
            scene_dir = Path(args.scene_dir)
            golden_dir = Path(args.golden_dir or "golden")
            pairs = []
            for actual in sorted(scene_dir.glob("*.dxf")):
                golden = golden_dir / actual.name
                if golden.is_file():
                    pairs.append((actual, golden))
            if not pairs:
                raise GoldenError(f"批量模式没有找到配对：{scene_dir} vs {golden_dir}")
            payloads = [compare_one(a, g, args.dpi, strict=strict) for a, g in pairs]
            failed = [p["scene"] for p in payloads if not p["passed"]]
            document = {"scenes": payloads, "failed": failed, "passed": not failed}
        else:
            if not (args.actual and args.golden):
                parser.print_help()
                return 2
            payloads = [compare_one(Path(args.actual), Path(args.golden), args.dpi, strict=strict)]
            document = {"scenes": payloads, "failed": [] if payloads[0]["passed"] else [payloads[0]["scene"]],
                        "passed": payloads[0]["passed"]}
    except GoldenError as exc:
        print(f"ERROR {exc}", file=sys.stderr)
        return 2

    text = json.dumps(document, indent=2, ensure_ascii=False, sort_keys=True)
    print(text)
    if args.json_out:
        Path(args.json_out).write_text(text + "\n", encoding="utf-8")

    for payload in payloads:
        status = "PASS" if payload["passed"] else "FAIL"
        print(
            f"{status} {payload['scene']}: hausdorff_p95={payload['hausdorff_p95_px']:.3f}px "
            f"count_diff={payload['entity_count_diff']} "
            f"dim_err={payload['worst_abs_dim']['abs_err_mm'] if payload['worst_abs_dim'] else 0:.4f}mm "
            f"layers_ok={payload['layers_ok']} closed_ok={payload['closed_ok']}",
            file=sys.stderr,
        )
    return 0 if document["passed"] else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
