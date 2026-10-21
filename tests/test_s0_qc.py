# tests/test_s0_qc.py
"""S0 摄入质检：四条验收判据 + 边界用例（SPEC §4 S0）。

验收判据（用户确认）：
1. QDIZE9403.JPG 输出完整 qc_report，72 DPI 命中 degrade 且预算按 300/72 放大；
2. 合成 2.0° 倾斜图，测角误差 ≤0.2°；
3. 锐度阶梯图判定与真值一致（配置阈值 0.25 下：清晰通过、重度模糊不通过）；
4. metrics_s0.json 通过 `read_metrics` 往返。
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from src.artifacts import read_metrics
from src.config import QcConfig, load_config
from src.s0_qc import (
    QcError,
    estimate_skew,
    halftone_dot_density,
    jpeg_blockiness,
    lighting_blocks,
    run_s0_qc,
    run_s0_qc_timed,
    sharpness_metric,
)

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "test_images"
SCREENSHOT = IMAGES / "QDIZE9403.JPG"


@pytest.fixture(scope="module")
def qc() -> QcConfig:
    return load_config(ROOT / "config.yaml").qc


# ---------------------------------------------------------------------------
# 夹具：可控的合成质检图
# ---------------------------------------------------------------------------
def _grid(angle_deg: float = 0.0, *, side: int = 472, step: int = 40) -> np.ndarray:
    w = h = side
    canvas = np.full((h, w), 255, np.uint8)
    for x in range(0, w, step):
        cv2.line(canvas, (x, 0), (x, h), 0, 1)
    for y in range(0, h, step):
        cv2.line(canvas, (0, y), (w, y), 0, 1)
    if angle_deg:
        matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
        canvas = cv2.warpAffine(canvas, matrix, (w, h), flags=cv2.INTER_LINEAR, borderValue=255)
    return canvas


def _line_image() -> np.ndarray:
    canvas = np.full((354, 472), 255, np.uint8)
    cv2.line(canvas, (20, 177), (452, 177), 0, 3)
    return canvas


def _save_png(path: Path, array: np.ndarray, dpi: tuple[float, float] | None) -> Path:
    kwargs = {"dpi": dpi} if dpi is not None else {}
    Image.fromarray(array).save(path, format="PNG", **kwargs)
    return path


# ---------------------------------------------------------------------------
# 验收 1：低 DPI 截图 → degrade + 预算放大
# ---------------------------------------------------------------------------
def test_screenshot_report_has_full_qc_fields(qc: QcConfig) -> None:
    report, _ = run_s0_qc(SCREENSHOT, qc)
    payload = report.to_json()
    for key in (
        "image_path", "image_sha256", "width_px", "height_px", "dpi", "sharpness",
        "skew", "lighting", "artifacts", "budget_scale", "mm_per_px", "status",
        "skew_correction_applied", "warnings", "not_evaluated",
    ):
        assert key in payload, f"qc_report 缺少 {key}"
    assert payload["skew_correction_applied"] is False, "S0 只判定，不得声称已校正"
    assert (report.width_px, report.height_px) == (736, 625)
    assert len(report.image_sha256) == 64


def test_screenshot_dpi_is_72_from_metadata(qc: QcConfig) -> None:
    report, _ = run_s0_qc(SCREENSHOT, qc)
    assert report.dpi.source == "metadata"
    assert report.dpi.applied == pytest.approx(72.0)
    assert report.dpi_below_min is True


def test_screenshot_degrades_and_scales_budget(qc: QcConfig) -> None:
    report, metrics = run_s0_qc(SCREENSHOT, qc)
    assert qc.dpi_mode == "degrade"
    assert report.status == "degraded"
    # 300/72 = 4.166666...
    assert report.budget_scale == pytest.approx(300.0 / 72.0, rel=1e-6)
    assert metrics.error_estimate["budget_scale"] == pytest.approx(report.budget_scale)
    assert any("降级" in w and "4.1667" in w for w in report.warnings), report.warnings


def test_screenshot_mm_per_px_matches_spec(qc: QcConfig) -> None:
    report, _ = run_s0_qc(SCREENSHOT, qc)
    assert report.mm_per_px == pytest.approx(25.4 / 72.0, rel=1e-9)


def test_reject_mode_rejects_low_dpi(qc: QcConfig) -> None:
    strict = qc.model_copy(update={"dpi_mode": "reject"})
    report, _ = run_s0_qc(SCREENSHOT, strict)
    assert report.status == "rejected"
    assert report.budget_scale == 1.0


def test_warn_mode_keeps_budget_at_one(qc: QcConfig) -> None:
    loose = qc.model_copy(update={"dpi_mode": "warn"})
    report, _ = run_s0_qc(SCREENSHOT, loose)
    assert report.status == "degraded"
    assert report.budget_scale == 1.0


# ---------------------------------------------------------------------------
# 验收 2：倾斜检测
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("angle", [0.0, 1.0, 2.0, 3.0, -2.0])
def test_skew_measurement_accuracy(qc: QcConfig, tmp_path: Path, angle: float) -> None:
    """合成倾斜网格：测角误差 ≤0.2°（用户验收判据 2）。"""
    path = _save_png(tmp_path / f"grid_{angle}.png", _grid(angle), (300.0, 300.0))
    report, _ = run_s0_qc(path, qc)
    assert report.skew.confident is True, report.skew
    assert abs(report.skew.angle_deg - angle) <= 0.2, (
        f"倾斜 {angle}° 测得 {report.skew.angle_deg}°，误差超过 0.2°"
    )
    assert report.skew.method == "projection_profile"


def test_skew_above_limit_is_flagged_with_correction(qc: QcConfig, tmp_path: Path) -> None:
    path = _save_png(tmp_path / "grid_4.png", _grid(4.0), (300.0, 300.0))
    report, _ = run_s0_qc(path, qc)
    assert report.skew.confident is True
    assert report.skew.exceeds_limit is True
    assert report.skew.corrective_rotation_deg == pytest.approx(-report.skew.angle_deg, abs=1e-4)
    assert any("旋转校正" in w for w in report.warnings)


def test_skew_not_confident_without_principal_lines(qc: QcConfig, tmp_path: Path) -> None:
    """单条线/圆没有主向：必须判不可信并记 0，不能凭空给出倾斜角去旋转。"""
    circle = np.full((400, 400), 255, np.uint8)
    cv2.circle(circle, (200, 200), 150, 0, 3)
    for name, img in (("circle.png", circle), ("line.png", _line_image())):
        path = _save_png(tmp_path / name, img, (300.0, 300.0))
        report, _ = run_s0_qc(path, qc)
        assert report.skew.confident is False
        assert report.skew.angle_deg == 0.0
        assert any("投影峰显著性" in w for w in report.warnings)


def test_estimate_skew_rejects_bad_search_range() -> None:
    with pytest.raises(QcError, match="search_deg"):
        estimate_skew(_grid(), search_deg=0.0)


# ---------------------------------------------------------------------------
# 验收 3：锐度阶梯
# ---------------------------------------------------------------------------
def test_sharpness_ladder_matches_threshold(qc: QcConfig, tmp_path: Path) -> None:
    """阈值 0.8 的语义：清晰通过；σ=2 起的模糊不通过（保护 S2 边缘提取）；单调下降。"""
    base = _line_image()
    values: list[float] = []
    for sigma in (0.0, 2.0, 4.0, 6.0):
        img = base if sigma == 0 else cv2.GaussianBlur(base, (0, 0), sigma)
        path = _save_png(tmp_path / f"sharp_{sigma}.png", img, (300.0, 300.0))
        report, _ = run_s0_qc(path, qc)
        values.append(report.sharpness.value)
        if sigma == 0.0:
            assert report.sharpness.passed is True
            assert report.sharpness.value > qc.sharpness_min
        if sigma >= 2.0:
            assert report.sharpness.passed is False, (sigma, report.sharpness)
            assert report.sharpness.value < qc.sharpness_min
    assert values == sorted(values, reverse=True), f"锐度未随模糊单调下降：{values}"


def test_sharpness_threshold_comes_from_config(qc: QcConfig, tmp_path: Path) -> None:
    path = _save_png(tmp_path / "sharp.png", _line_image(), (300.0, 300.0))
    baseline, _ = run_s0_qc(path, qc)
    strict = qc.model_copy(update={"sharpness_min": baseline.sharpness.value + 0.5})
    report, _ = run_s0_qc(path, strict)
    assert report.sharpness.passed is False
    assert report.sharpness.threshold == pytest.approx(strict.sharpness_min)


def test_sharpness_of_flat_image_is_zero() -> None:
    assert sharpness_metric(np.full((64, 64), 255, np.uint8)) == 0.0


def test_sharpness_requires_grayscale() -> None:
    with pytest.raises(QcError, match="灰度图"):
        sharpness_metric(np.zeros((4, 4, 3), np.uint8))


# ---------------------------------------------------------------------------
# 验收 4：metrics_s0 往返
# ---------------------------------------------------------------------------
def test_metrics_s0_roundtrip(qc: QcConfig, tmp_path: Path) -> None:
    report, metrics = run_s0_qc_timed(SCREENSHOT, qc, out_dir=tmp_path)
    assert (tmp_path / "qc_report.json").is_file()
    assert (tmp_path / "metrics_s0.json").is_file()

    loaded = read_metrics(tmp_path / "metrics_s0.json")
    assert loaded.stage == "s0"
    assert loaded.counts == metrics.counts
    assert loaded.timing_ms >= 0.0
    assert loaded.error_estimate["budget_scale"] == pytest.approx(report.budget_scale)
    assert loaded.details["status"] == report.status

    report_json = json.loads((tmp_path / "qc_report.json").read_text(encoding="utf-8"))
    assert report_json["image_sha256"] == report.image_sha256
    assert report_json["dpi"]["applied"] == pytest.approx(72.0)


def test_qc_report_is_deterministic_except_timing(qc: QcConfig, tmp_path: Path) -> None:
    first, _ = run_s0_qc(SCREENSHOT, qc, out_dir=tmp_path / "a")
    second, _ = run_s0_qc(SCREENSHOT, qc, out_dir=tmp_path / "b")
    assert first.to_json() == second.to_json()


# ---------------------------------------------------------------------------
# DPI / 光照 边界
# ---------------------------------------------------------------------------
def test_missing_dpi_falls_back_and_warns(qc: QcConfig, tmp_path: Path) -> None:
    path = _save_png(tmp_path / "nodpi.png", _line_image(), None)
    report, _ = run_s0_qc(path, qc)
    assert report.dpi.source == "fallback"
    assert report.dpi.applied == pytest.approx(qc.fallback_dpi)
    assert report.status == "degraded"
    assert any("没有可用的 DPI 元数据" in w for w in report.warnings)


def test_anisotropic_dpi_warns(qc: QcConfig, tmp_path: Path) -> None:
    path = _save_png(tmp_path / "aniso.png", _line_image(), (300.0, 600.0))
    report, _ = run_s0_qc(path, qc)
    assert report.dpi.anisotropic is True
    assert report.dpi.applied == pytest.approx(300.0)  # 按 X 处理
    assert any("各向异性" in w for w in report.warnings)


def test_lighting_gradient_triggers_background_flatten(qc: QcConfig, tmp_path: Path) -> None:
    h, w = 354, 472
    gradient = np.linspace(40, 250, w, dtype=np.float32)[None, :].repeat(h, 0)
    canvas = np.full((h, w), 255, np.uint8)
    cv2.line(canvas, (20, h // 2), (w - 20, h // 2), 0, 2)
    mixed = np.clip(canvas.astype(np.float32) * (gradient / 255.0), 0, 255).astype(np.uint8)
    path = _save_png(tmp_path / "grad.png", mixed, (300.0, 300.0))
    report, _ = run_s0_qc(path, qc)
    assert report.lighting.uneven is True
    assert report.lighting.recommend_background_flatten is True
    assert any("光照不均" in w for w in report.warnings)


def test_flat_lighting_is_not_flagged(qc: QcConfig, tmp_path: Path) -> None:
    path = _save_png(tmp_path / "flat.png", _line_image(), (300.0, 300.0))
    report, _ = run_s0_qc(path, qc)
    assert report.lighting.uneven is False


def test_lighting_blocks_rejects_bad_ratio() -> None:
    with pytest.raises(QcError, match="block_ratio"):
        lighting_blocks(np.full((32, 32), 255, np.uint8), block_ratio=0.0)


def test_missing_image_raises(tmp_path: Path, qc: QcConfig) -> None:
    with pytest.raises(QcError, match="图像不存在"):
        run_s0_qc(tmp_path / "nope.png", qc)


def test_corrupt_image_raises(tmp_path: Path, qc: QcConfig) -> None:
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image")
    with pytest.raises(QcError, match="无法解码"):
        run_s0_qc(bad, qc)


def test_not_evaluated_checks_are_declared(qc: QcConfig) -> None:
    report, _ = run_s0_qc(SCREENSHOT, qc)
    names = {c.name for c in report.not_evaluated}
    assert names == {"perspective", "stain_mask"}
    assert all(c.reason for c in report.not_evaluated)


# ---------------------------------------------------------------------------
# 伪影：JPEG 块效应（仅 JPEG 输入）与半调网点/莫尔纹
# ---------------------------------------------------------------------------
def _halftone(pitch: int = 6, *, width: int = 400, height: int = 300) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    dots = (np.sin(2 * np.pi * xx / pitch) > 0) & (np.sin(2 * np.pi * yy / pitch) > 0)
    return (255 - dots.astype(np.uint8) * 255)


def test_moire_detected_on_halftone(qc: QcConfig, tmp_path: Path) -> None:
    path = _save_png(tmp_path / "halftone.png", _halftone(6), (300.0, 300.0))
    report, _ = run_s0_qc(path, qc)
    artifact = next(a for a in report.artifacts if a.name == "moire")
    assert artifact.evaluated is True
    assert artifact.value is not None and artifact.value > qc.moire_dot_density_max
    assert artifact.triggered is True
    assert artifact.confidence == "measured"
    assert report.status == "degraded"
    assert any("网点" in w for w in report.warnings)


def test_moire_not_triggered_on_clean_lines(qc: QcConfig, tmp_path: Path) -> None:
    path = _save_png(tmp_path / "clean.png", _grid(), (300.0, 300.0))
    report, _ = run_s0_qc(path, qc)
    artifact = next(a for a in report.artifacts if a.name == "moire")
    assert artifact.triggered is False
    assert artifact.value is not None and artifact.value < qc.moire_dot_density_max


def test_jpeg_blockiness_skipped_for_png(qc: QcConfig, tmp_path: Path) -> None:
    path = _save_png(tmp_path / "clean.png", _grid(), (300.0, 300.0))
    report, _ = run_s0_qc(path, qc)
    artifact = next(a for a in report.artifacts if a.name == "jpeg_blockiness")
    assert artifact.evaluated is False
    assert artifact.value is None
    assert "不是 JPEG" in artifact.note


def test_jpeg_blockiness_evaluated_for_jpeg(qc: QcConfig) -> None:
    assert SCREENSHOT.suffix.upper() == ".JPG"
    report, _ = run_s0_qc(SCREENSHOT, qc)
    artifact = next(a for a in report.artifacts if a.name == "jpeg_blockiness")
    assert artifact.evaluated is True
    assert artifact.confidence == "heuristic"
    assert artifact.value is not None and artifact.value > 0.0


def test_artifact_detection_can_be_disabled(qc: QcConfig, tmp_path: Path) -> None:
    disabled = qc.model_copy(update={"moire_detect": False, "jpeg_ring_artifact_detect": False})
    path = _save_png(tmp_path / "halftone.png", _halftone(4), (300.0, 300.0))
    report, _ = run_s0_qc(path, disabled)
    assert report.artifacts == ()


def test_halftone_dot_density_rejects_colour_input() -> None:
    with pytest.raises(QcError, match="灰度图"):
        halftone_dot_density(np.zeros((8, 8, 3), np.uint8))


def test_jpeg_blockiness_rejects_colour_input() -> None:
    with pytest.raises(QcError, match="灰度图"):
        jpeg_blockiness(np.zeros((8, 8, 3), np.uint8))
