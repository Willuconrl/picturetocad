# tests/test_s6_text.py
"""S6 首轮：文字候选区提取（不接 OCR 引擎）。

判据：
① 已识别基元覆盖的像素不出现在候选区里；
② 明显文字区域（合成图上的字符块）被检出、不被误排除；
③ `pending_ocr_regions.json` 与 `metrics_s6.json` 往返一致，debug 图落盘；
④ 噪点/长条/超大块被正确排除并记录原因。
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.artifacts import read_metrics
from src.config import R2VConfig, load_config
from src.s3_skeleton import run_s3_skeleton
from src.s4_fit import S4Input, run_s4_fit
from src.s6_text import S6Error, rasterize_primitives, run_s6_text

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config.yaml"
DPI = 300.0
MM_PER_PX = 25.4 / DPI


@pytest.fixture(scope="module")
def cfg() -> R2VConfig:
    return load_config(CONFIG)


def _line_scene(cfg: R2VConfig, *, with_text: bool = True, with_noise: bool = False):
    """一条水平线 + 若干字符块，返回 (二进制图, S4 拟合结果, 灰度图)。"""
    gray = np.full((300, 500), 255, np.uint8)
    cv2.line(gray, (60, 220), (440, 220), 0, 6)
    if with_text:
        for i, char in enumerate("R2V"):
            cv2.putText(gray, char, (80 + i * 30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 0, 2)
    if with_noise:
        gray[40, 40] = 0                       # 单个椒盐噪点（面积 1 < 20）
        cv2.line(gray, (300, 60), (460, 60), 0, 2)   # 长条（aspect > 8）——但会被基元扣掉
    from src.s2_binarize import S2Input, binarize_image

    s2, _ = binarize_image(S2Input.from_array(gray.astype(np.float32) / 255.0, dpi=DPI), cfg)
    s3, _ = run_s3_skeleton(s2.binary, cfg, dpi=DPI, linewidth_px=s2.linewidth_px)
    fit, _ = run_s4_fit(
        S4Input(graph=s3.graph, contours=s2.edges.contours, linewidth_px=s2.linewidth_px, dpi=DPI),
        cfg,
    )
    return s2.binary, fit, gray, s2.linewidth_px


def test_text_candidates_found_and_primitives_excluded(cfg: R2VConfig, tmp_path: Path) -> None:
    binary, fit, gray, linewidth = _line_scene(cfg)
    result, metrics = run_s6_text(
        binary, fit, cfg, dpi=DPI, out_dir=tmp_path, gray=gray, linewidth_px=linewidth
    )
    assert result.candidates, f"没有检出文字候选区：{result.counts}"
    assert metrics.counts["candidates"] == len(result.candidates)

    # ① 候选区不能落在**结构性基元**（图线）上：字符笔画本身也会被拟合成短基元，
    #    所以只对长笔画做重叠检查（见 src/s6_text.structural_primitives 的说明）
    from src.s6_text import structural_primitives

    linewidth = max(1, int(round(linewidth)))
    structural = structural_primitives(
        fit.primitives, linewidth_px=float(linewidth),
        min_height_mm=float(cfg.text.min_height_mm), mm_per_px=MM_PER_PX,
    )
    primitive_mask = rasterize_primitives(binary.shape, structural, thickness_px=linewidth)
    for candidate in result.candidates:
        x, y, w, h = candidate.bbox
        overlap = int((primitive_mask[y:y + h, x:x + w] > 0).sum())
        assert overlap == 0, f"候选区 {candidate.bbox} 与图线重叠 {overlap} px"

    # ② 文字在 y≈95..125 一带，候选区应落在这一带
    ys = [c.bbox[1] for c in result.candidates]
    assert min(ys) < 130, f"候选区没有覆盖文字带：{ys}"


def test_noise_and_long_regions_are_excluded(cfg: R2VConfig, tmp_path: Path) -> None:
    """筛选逻辑单测：无基元干扰时，噪点/长条/超大块都必须被排除并记原因。

    说明：在真实管线里，孤立长条会被 S3/S4 拟合成线段并从二值图里扣掉，
    轮不到 S6 排除；这里用一个没有基元的合成二值图直接检验筛选器本身。
    """
    from src.s4_fit import FitResult

    empty_fit = FitResult(
        primitives=(), nodes={}, discarded_short_edges=0, warnings=(), counts_by_kind={}
    )
    binary = np.zeros((200, 300), np.uint8)
    binary[20, 20] = 255                                     # 1 px 噪点
    cv2.rectangle(binary, (40, 40), (44, 44), 255, -1)       # 5×5 = 25 px，够 20 但高只有 0.4mm
    cv2.line(binary, (60, 80), (260, 80), 255, 2)            # 200×2 长条（aspect 100）
    cv2.rectangle(binary, (40, 120), (140, 190), 255, -1)    # 100×70 大块（7 mm² 超大）
    result, _ = run_s6_text(binary, empty_fit, cfg, dpi=DPI, out_dir=tmp_path, linewidth_px=4.0)
    reasons = {item.reason for item in result.excluded}
    assert "area_too_small" in reasons, reasons
    assert "aspect_too_long" in reasons, reasons
    assert "area_too_large_mm2" in reasons, reasons
    # 5×5 的块高度 0.42mm < min_height_mm(1.5) → 高度越界
    assert "height_out_of_range" in reasons, reasons
    assert result.candidates == ()


def test_outputs_roundtrip(cfg: R2VConfig, tmp_path: Path) -> None:
    binary, fit, gray, linewidth = _line_scene(cfg)
    result, metrics = run_s6_text(
        binary, fit, cfg, dpi=DPI, out_dir=tmp_path, gray=gray, linewidth_px=linewidth
    )
    payload = json.loads((tmp_path / "pending_ocr_regions.json").read_text(encoding="utf-8"))
    assert len(payload["candidates"]) == len(result.candidates)
    first = payload["candidates"][0]
    assert set(first) == {
        "id", "bbox", "area_px", "aspect_ratio",
        "distance_to_nearest_primitive_px", "height_mm",
    }
    loaded = read_metrics(tmp_path / "metrics_s6.json")
    assert loaded.stage == "s6"
    assert loaded.counts["candidates"] == len(result.candidates)
    assert loaded.details["ocr_engine"].startswith("none")
    debug_path = Path(result.debug_path or "")
    assert debug_path.is_file(), "debug 叠加图没有落盘"
    image = cv2.imread(str(debug_path))
    assert image is not None and image.shape[0] == binary.shape[0]


def test_letters_are_grouped_into_one_line(cfg: R2VConfig, tmp_path: Path) -> None:
    """R2V 三个字母必须合并成 **一行**，且行框紧密包裹三个字母。"""
    binary, fit, gray, linewidth = _line_scene(cfg)
    result, metrics = run_s6_text(
        binary, fit, cfg, dpi=DPI, out_dir=tmp_path, gray=gray, linewidth_px=linewidth
    )
    assert len(result.candidates) == 3, f"应检出 3 个字母，实际 {len(result.candidates)}"
    assert len(result.lines) == 1, f"应合并成 1 行，实际 {len(result.lines)} 行"
    line = result.lines[0]
    assert len(line.member_region_ids) == 3
    # 行框应覆盖全部成员且不超出太多（紧密包裹）
    xs = [c.bbox[0] for c in result.candidates]
    xe = [c.bbox[0] + c.bbox[2] for c in result.candidates]
    ys = [c.bbox[1] for c in result.candidates]
    ye = [c.bbox[1] + c.bbox[3] for c in result.candidates]
    assert line.bbox[0] == min(xs) and line.bbox[0] + line.bbox[2] == max(xe)
    assert line.bbox[1] == min(ys) and line.bbox[1] + line.bbox[3] == max(ye)
    assert abs(line.baseline_angle) <= 5.0, f"基线角 {line.baseline_angle}° 不合理"
    assert line.char_count_estimate == 3
    assert metrics.counts["lines"] == 1


def test_pending_ocr_lines_written(cfg: R2VConfig, tmp_path: Path) -> None:
    binary, fit, gray, linewidth = _line_scene(cfg)
    result, _ = run_s6_text(
        binary, fit, cfg, dpi=DPI, out_dir=tmp_path, gray=gray, linewidth_px=linewidth
    )
    payload = json.loads((tmp_path / "pending_ocr_lines.json").read_text(encoding="utf-8"))
    assert len(payload["lines"]) == len(result.lines)
    first = payload["lines"][0]
    assert set(first) == {
        "id", "bbox", "member_region_ids", "char_count_estimate", "baseline_angle",
        "baseline_y_mm", "height_mm", "length_mm",
        "text", "confidence", "engine", "is_placeholder",
    }
    # regions 文件必须保留（溯源依据）
    assert (tmp_path / "pending_ocr_regions.json").is_file()


def test_text_free_scenes_have_no_lines(cfg: R2VConfig, tmp_path: Path) -> None:
    """没有文字的合成场景：既不能有候选区，也不能有虚假行。"""
    from src.s4_fit import FitResult

    empty_fit = FitResult(
        primitives=(), nodes={}, discarded_short_edges=0, warnings=(), counts_by_kind={}
    )
    for name in ("synthetic_line_single", "synthetic_arc_120", "synthetic_circle_full"):
        image = ROOT / "test_images" / f"{name}.png"
        from src.s2_binarize import S2Input, binarize_image

        gray = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
        s2, _ = binarize_image(S2Input.from_array(gray.astype(np.float32) / 255.0, dpi=DPI), cfg)
        s3, _ = run_s3_skeleton(s2.binary, cfg, dpi=DPI, linewidth_px=s2.linewidth_px)
        fit, _ = run_s4_fit(
            S4Input(graph=s3.graph, contours=s2.edges.contours, linewidth_px=s2.linewidth_px, dpi=DPI),
            cfg,
        )
        result, _ = run_s6_text(
            s2.binary, fit, cfg, dpi=DPI, out_dir=tmp_path / name,
            gray=gray, linewidth_px=s2.linewidth_px,
        )
        assert result.candidates == (), f"{name} 出现候选区"
        assert result.lines == (), f"{name} 出现虚假行"


def test_baseline_y_mm_formula(cfg: R2VConfig, tmp_path: Path) -> None:
    """基线 = 行框下沿 − descender_ratio × 行高（工程图标注的物理锚点）。"""
    binary, fit, gray, linewidth = _line_scene(cfg)
    result, _ = run_s6_text(
        binary, fit, cfg, dpi=DPI, out_dir=tmp_path, gray=gray, linewidth_px=linewidth
    )
    line = result.lines[0]
    x, y, w, h = line.bbox
    expected_mm = (y + h - cfg.text.ocr.descender_ratio * h) * MM_PER_PX
    assert line.baseline_y_mm == pytest.approx(expected_mm, abs=1e-3)
    assert line.baseline_y_mm < line.height_mm + expected_mm  # 基线在行框内


def test_ocr_degrades_without_engine(cfg: R2VConfig, tmp_path: Path) -> None:
    """引擎缺失：不抛异常，字段齐全，WARN 带安装指引。"""
    from src.ocr_engine import detect_engines

    if detect_engines():
        pytest.skip("本机装了 OCR 引擎，降级路径不适用")
    binary, fit, gray, linewidth = _line_scene(cfg)
    result, metrics = run_s6_text(
        binary, fit, cfg, dpi=DPI, out_dir=tmp_path, gray=gray, linewidth_px=linewidth
    )
    for line in result.lines:
        assert line.text == ""
        assert line.confidence == 0.0
        assert line.engine == "none"
        assert line.is_placeholder is True
    assert any("requirements-ocr.txt" in warning for warning in result.warnings)
    assert metrics.counts["lines"] == len(result.lines)


def test_text_items_structure(cfg: R2VConfig, tmp_path: Path) -> None:
    """S8 的 text_items 入参结构（本轮不写 TEXT，只校验结构）。"""
    from src.s6_text import collect_text_items

    binary, fit, gray, linewidth = _line_scene(cfg)
    result, _ = run_s6_text(
        binary, fit, cfg, dpi=DPI, out_dir=tmp_path, gray=gray, linewidth_px=linewidth
    )
    items = collect_text_items(
        result.lines, mm_per_px=MM_PER_PX,
        confidence_min=cfg.text.ocr.text_confidence_min,
    )
    assert len(items) == len(result.lines)
    first = items[0]
    assert set(first) == {
        "line_id", "insertion_point", "text", "height_mm", "confidence", "is_placeholder",
    }
    assert first["line_id"].startswith("s6-line-")
    assert len(first["insertion_point"]) == 2
    # 置信度 0 < 阈值 0.6 → 占位，S8 不应写 TEXT
    assert first["is_placeholder"] is True


def test_blank_image_has_no_candidates(cfg: R2VConfig, tmp_path: Path) -> None:
    """没有基元也没有文字：候选区必须为空，不能凭空造字。"""
    from src.s4_fit import FitResult

    empty_fit = FitResult(
        primitives=(), nodes={}, discarded_short_edges=0, warnings=(), counts_by_kind={}
    )
    blank = np.zeros((120, 160), np.uint8)
    result, _ = run_s6_text(blank, empty_fit, cfg, dpi=DPI, out_dir=tmp_path, linewidth_px=4.0)
    assert result.candidates == ()


def test_rejects_bad_input(cfg: R2VConfig) -> None:
    from src.s4_fit import FitResult

    empty_fit = FitResult(
        primitives=(), nodes={}, discarded_short_edges=0, warnings=(), counts_by_kind={}
    )
    with pytest.raises(S6Error, match="二维"):
        run_s6_text(np.zeros((8, 8, 3), np.uint8), empty_fit, cfg, dpi=DPI)
    with pytest.raises(S6Error, match="dpi"):
        run_s6_text(np.zeros((8, 8), np.uint8), empty_fit, cfg, dpi=0.0)
