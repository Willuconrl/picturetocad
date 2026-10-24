# scripts/perf_baseline.py
"""M9-4.2 全链路性能基线：各阶段耗时、端到端墙钟、峰值内存、规模扫描。

用法：
    python scripts/perf_baseline.py                 # 现有测试图
    python scripts/perf_baseline.py --scale-sweep    # 额外做规模扫描（放大图像）

输出：控制台表格 + `docs/perf_baseline.md`（可 --md 指定路径）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import tracemalloc
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.pipeline import run_pipeline  # noqa: E402

STAGES = ("s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8")


def _run_once(image: Path, cfg, out_dir: Path) -> dict[str, object]:
    tracemalloc.start()
    started = time.perf_counter()
    result = run_pipeline(str(image), cfg, out_dir=out_dir)
    wall_ms = (time.perf_counter() - started) * 1000.0
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    timings: dict[str, float] = {}
    for stage in STAGES:
        metrics = result.metrics.get(stage)
        value = getattr(metrics, "timing_ms", None) if metrics is not None else None
        if value is not None:
            timings[stage] = round(float(value), 2)
    height, width = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE).shape[:2]
    return {
        "image": image.name,
        "size_px": f"{width}x{height}",
        "megapixels": round(width * height / 1e6, 3),
        "wall_ms": round(wall_ms, 1),
        "peak_mem_mb": round(peak / (1024 * 1024), 1),
        "stage_ms": timings,
        "sum_stage_ms": round(sum(timings.values()), 1),
        "primitives": len(result.fit.primitives),
        "entities": len(result.dxf.entities),
    }


def _write_markdown(rows: list[dict[str, object]], sweep: list[dict[str, object]], path: Path) -> None:
    lines = [
        "# 全链路性能基线（M9-4.2）",
        "",
        "> 生成：`python scripts/perf_baseline.py`。所有耗时为单次运行（开发机、Python 3.14、Windows），",
        "> 仅用于量级判断与瓶颈定位，不作为跨机器基准。",
        "",
        "## 逐图耗时",
        "",
        "| 图像 | 尺寸 | MP | 端到端 ms | 各阶段合计 ms | 峰值内存 MB | 基元数 | 实体数 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['image']} | {row['size_px']} | {row['megapixels']} | {row['wall_ms']} | "
            f"{row['sum_stage_ms']} | {row['peak_mem_mb']} | {row['primitives']} | {row['entities']} |"
        )
    lines += ["", "## 阶段耗时明细（ms）", "", "| 图像 | " + " | ".join(STAGES) + " |", "|---|" + "---|" * len(STAGES)]
    for row in rows:
        stage_ms = row["stage_ms"]  # type: ignore[assignment]
        cells = [f"{stage_ms.get(stage, '-')}" for stage in STAGES]  # type: ignore[union-attr]
        lines.append(f"| {row['image']} | " + " | ".join(cells) + " |")
    if sweep:
        lines += [
            "",
            "## 规模扫描（同一场景放大）",
            "",
            "| 尺寸 | MP | 端到端 ms | 峰值内存 MB | 每 MP 耗时 ms |",
            "|---|---|---|---|---|",
        ]
        for row in sweep:
            lines.append(
                f"| {row['size_px']} | {row['megapixels']} | {row['wall_ms']} | "
                f"{row['peak_mem_mb']} | {round(float(row['wall_ms']) / float(row['megapixels']), 1)} |"
            )
    lines += ["", "## 瓶颈结论", ""]
    if rows:
        slowest_stage = max(STAGES, key=lambda s: sum(float(r["stage_ms"].get(s, 0)) for r in rows))  # type: ignore[union-attr]
        total = sum(float(r["sum_stage_ms"]) for r in rows)
        share = sum(float(r["stage_ms"].get(slowest_stage, 0)) for r in rows) / max(total, 1e-9)
        lines.append(f"- 最耗时阶段：**{slowest_stage.upper()}**，占各阶段合计的 {share:.1%}。")
        slowest_image = max(rows, key=lambda r: float(r["wall_ms"]))
        lines.append(f"- 最慢图像：`{slowest_image['image']}`（{slowest_image['wall_ms']} ms）。")
        lines.append(
            "- 端到端墙钟与阶段合计的差值即 IO/管线开销（读图、写 DXF、序列化 metrics），"
            "通常 <10%，若显著偏大说明 IO 成为瓶颈。"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--md", default=str(ROOT / "docs" / "perf_baseline.md"))
    parser.add_argument("--scale-sweep", action="store_true")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args(argv)

    cfg = load_config(ROOT / "config.yaml")
    out_root = ROOT / "outputs" / "perf"
    out_root.mkdir(parents=True, exist_ok=True)

    images = [
        ROOT / "test_images" / "synthetic_line_single.png",
        ROOT / "test_images" / "synthetic_arc_120.png",
        ROOT / "test_images" / "synthetic_circle_full.png",
        ROOT / "test_images" / "synthetic_polyline_bulge.png",
    ]
    real = ROOT / "test_images" / "QDIZE9403.JPG"
    if real.is_file():
        images.append(real)

    rows = [_run_once(image, cfg, out_root / image.stem) for image in images if image.is_file()]

    sweep: list[dict[str, object]] = []
    if args.scale_sweep:
        base = cv2.imread(str(ROOT / "test_images" / "synthetic_polyline_bulge.png"), cv2.IMREAD_GRAYSCALE)
        for factor in (1, 2, 4):
            scaled = cv2.resize(base, None, fx=factor, fy=factor, interpolation=cv2.INTER_NEAREST)
            path = out_root / f"scaled_x{factor}.png"
            cv2.imwrite(str(path), scaled)
            row = _run_once(path, cfg, out_root / f"scaled_x{factor}")
            row["image"] = f"bulge x{factor}"
            sweep.append(row)

    md_path = Path(args.md)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    _write_markdown(rows, sweep, md_path)

    payload = {"images": rows, "sweep": sweep}
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"报告：{md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
