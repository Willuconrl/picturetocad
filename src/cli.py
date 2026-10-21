# src/cli.py
"""`r2v` 命令行骨架（M0 阶段只做参数与配置校验，不含管线）。

用法：
    python -m src --check-config
    python -m src --print-effective --preset blueprint
    python -m src --list-presets

退出码：
    0  成功
    2  配置/preset 非法（SPEC §5：非法值启动即失败）
    其他见各阶段约定
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from src import CAPABILITY_NOTICE, __version__
from src.config import ConfigError, load_yaml_mapping, make_config
from src.presets import (
    EffectiveConfig,
    PresetError,
    build_effective,
    load_presets,
    resolve_presets_path,
)

EXIT_OK = 0
EXIT_CONFIG_ERROR = 2

DEFAULT_CONFIG = "config.yaml"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="r2v",
        description=(
            "光栅图纸转 CAD（DXF）矢量化工具。\n\n能力边界：" + CAPABILITY_NOTICE
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  python -m src --check-config\n"
            "  python -m src --print-effective --preset mechanical\n"
        ),
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="config.yaml 路径")
    parser.add_argument(
        "command", nargs="?", choices=["convert"], help="convert：端到端转换一张图（S0→S8）"
    )
    parser.add_argument("--image", default=None, help="convert 的输入图像路径")
    parser.add_argument("--out", default=None, help="convert 的输出目录")
    parser.add_argument("--preset", default=None, help="套用哪个预设（缺省用 config 的 presets.default）")
    parser.add_argument("--presets", default=None, help="显式指定 presets.yaml 路径（默认按 config 解析）")
    parser.add_argument("--check-config", action="store_true", help="校验配置后退出")
    parser.add_argument("--print-effective", action="store_true", help="打印合并后的有效参数（JSON）")
    parser.add_argument("--list-presets", action="store_true", help="列出可用预设后退出")
    parser.add_argument("--version", action="version", version=f"r2v-tool {__version__}")
    return parser


def _load(config_path: Path, preset: str | None, presets_path: str | None) -> EffectiveConfig:
    return build_effective(
        config_path,
        preset,
        presets_path=Path(presets_path) if presets_path else None,
    )


def cmd_check_config(effective: EffectiveConfig) -> int:
    cfg = effective.config
    sections = len(type(cfg).model_fields)
    print(
        "OK "
        f"config_hash={effective.config_hash[:16]} "
        f"preset={effective.preset or '-'} "
        f"overrides={len(effective.overridden_keys)} "
        f"sections={sections} "
        f"layers={len(cfg.dxf.layers)} "
        f"dpi_mode={cfg.qc.dpi_mode}"
    )
    return EXIT_OK


def cmd_print_effective(effective: EffectiveConfig) -> int:
    document = {
        "config_hash": effective.config_hash,
        "preset": effective.preset,
        "overridden_keys": list(effective.overridden_keys),
        "effective": effective.data,
    }
    sys.stdout.write(
        json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )
    return EXIT_OK


def cmd_list_presets(config_path: Path, presets_path: str | None) -> int:
    cfg = make_config(load_yaml_mapping(config_path), source=str(config_path))
    resolved = (
        Path(presets_path) if presets_path else resolve_presets_path(config_path, cfg.presets.file)
    )
    preset_file = load_presets(resolved)
    for name in sorted(preset_file.presets):
        print(f"{name}\t{preset_file.presets[name].description}")
    return EXIT_OK


def cmd_convert(effective: EffectiveConfig, image: str, out: str) -> int:
    """端到端转换：S0→S8，产出 DXF 与各阶段 metrics。"""
    from src.pipeline import pipeline_summary, run_pipeline

    result = run_pipeline(image, effective.config, out_dir=Path(out))
    summary = pipeline_summary(result)
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    # OCR 尚未实现：显式告知，不写任何 TEXT 实体（编造文字比缺字更危险）
    print("OCR skipped: pending implementation", file=sys.stderr)
    if not result.dxf.self_check.passed:
        print(
            "ERROR DXF 自检未通过：" + "; ".join(result.dxf.self_check.problems[:3]),
            file=sys.stderr,
        )
        return EXIT_CONFIG_ERROR
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "convert":
        if not args.image or not args.out:
            parser.print_help()
            return EXIT_CONFIG_ERROR
        config_path = Path(args.config)
        try:
            effective = _load(config_path, args.preset, args.presets)
        except (ConfigError, PresetError) as exc:
            print(f"ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
            return EXIT_CONFIG_ERROR
        return cmd_convert(effective, args.image, args.out)

    if not (args.check_config or args.print_effective or args.list_presets):
        parser.print_help()
        return EXIT_CONFIG_ERROR

    config_path = Path(args.config)
    try:
        if args.list_presets:
            return cmd_list_presets(config_path, args.presets)
        effective = _load(config_path, args.preset, args.presets)
        if args.check_config:
            return cmd_check_config(effective)
        return cmd_print_effective(effective)
    except (ConfigError, PresetError) as exc:
        print(f"ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
