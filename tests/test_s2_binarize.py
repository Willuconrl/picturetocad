# tests/test_s2_binarize.py
"""S2 二值化 + 亚像素边缘：五条验收判据 + 各算子边界（SPEC §4 S2）。

验收判据（用户确认）：
1. 亚像素边缘定位误差 ≤0.1 px；
2. 前景 IoU ≥0.98；
3. 四路选优两次运行结果一致；
4. 线宽估计相对误差 ≤3%；
5. metrics_s2 往返。

合成真值用**解析覆盖率渲染**（列方向按面积覆盖率算灰度），不做 8× 超采样再降采样——
后者把边缘量化到 1/8 px，会把"测试精度"卡在 0.125 px，根本量不出 0.1 px 的实现差异。
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.artifacts import read_metrics
from src.config import R2VConfig, load_config
from src.s2_binarize import (
    BinarizeError,
    S2Input,
    binarize_image,
    binarize_local_gaussian,
    binarize_niblack,
    binarize_otsu,
    binarize_sauvola,
    foreground_stability,
    postprocess,
    subpixel_edges,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config.yaml"


@pytest.fixture(scope="module")
def cfg() -> R2VConfig:
    return load_config(CONFIG)


# ---------------------------------------------------------------------------
# 夹具：解析覆盖率的亚像素真值
# ---------------------------------------------------------------------------
def _coverage_columns(strokes: list[tuple[float, float]], width: int) -> np.ndarray:
    """返回每列被笔画覆盖的比例（0..1），并集取最大值。"""
    idx = np.arange(width, dtype=np.float64)
    coverage = np.zeros(width, dtype=np.float64)
    for x0, x1 in strokes:
        piece = np.clip(np.minimum(idx + 1.0, x1) - np.maximum(idx, x0), 0.0, 1.0)
        coverage = np.maximum(coverage, piece)
    return coverage


def _render(strokes: list[tuple[float, float]], *, width: int = 400, height: int = 300) -> tuple[np.ndarray, np.ndarray]:
    coverage = _coverage_columns(strokes, width)
    column = (1.0 - coverage) * 255.0
    image = np.repeat(column[None, :], height, axis=0).astype(np.uint8)
    truth = np.repeat((coverage > 0.5)[None, :], height, axis=0)
    return image, truth


def _edge_positions(
    result, *, x_split: float, y_lo: int = 80, y_hi: int = 220, left: bool
) -> float:
    candidates = [
        p[0]
        for contour in result.edges.contours
        for p in contour
        if y_lo <= p[1] <= y_hi and ((p[0] < x_split) if left else (p[0] > x_split))
    ]
    assert candidates, "没有找到轮廓点"
    return float(np.median(candidates))


# ---------------------------------------------------------------------------
# 验收 1：亚像素边缘定位误差 ≤0.1 px
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("offset", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
def test_subpixel_edge_accuracy_on_analytic_stroke(
    cfg: R2VConfig, offset: float
) -> None:
    x0 = 100.0 + offset
    image, _ = _render([(x0, x0 + 6.0)])
    result, _ = binarize_image(
        S2Input.from_array(image.astype(np.float32) / 255.0, dpi=300.0), cfg
    )
    left = _edge_positions(result, x_split=x0 + 3.0, left=True)
    right = _edge_positions(result, x_split=x0 + 3.0, left=False)
    assert abs(left - x0) <= 0.1, f"左边缘真值 {x0:.3f}，测得 {left:.3f}"
    assert abs(right - (x0 + 6.0)) <= 0.1, f"右边缘真值 {x0 + 6:.3f}，测得 {right:.3f}"


@pytest.mark.parametrize("sigma", [0.6, 0.8])
def test_subpixel_edge_accuracy_on_blurred_edge(
    cfg: R2VConfig, sigma: float
) -> None:
    """真实扫描的边缘是模糊的；对称模糊下 50% 电平判据是无偏的。"""
    x0 = 100.37
    image, _ = _render([(x0, x0 + 6.0)])
    blurred = cv2.GaussianBlur(image.astype(np.float32) / 255.0, (0, 0), sigma)
    result, _ = binarize_image(S2Input.from_array(blurred, dpi=300.0), cfg)
    left = _edge_positions(result, x_split=x0 + 3.0, left=True)
    assert abs(left - x0) <= 0.1, f"σ={sigma}: 真值 {x0:.3f}，测得 {left:.3f}"


def test_subpixel_edge_accuracy_on_wide_blurred_stroke(cfg: R2VConfig) -> None:
    """宽笔画 + 中等模糊（两侧过渡带不重叠）：仍须 ≤0.1 px。"""
    x0 = 100.37
    image, _ = _render([(x0, x0 + 16.0)])
    blurred = cv2.GaussianBlur(image.astype(np.float32) / 255.0, (0, 0), 0.9)
    result, _ = binarize_image(S2Input.from_array(blurred, dpi=300.0), cfg)
    left = _edge_positions(result, x_split=x0 + 8.0, left=True)
    assert abs(left - x0) <= 0.1, f"真值 {x0:.3f}，测得 {left:.3f}"


def test_subpixel_requires_contrast(cfg: R2VConfig) -> None:
    flat = np.full((64, 64), 0.5, dtype=np.float32)
    binary = np.zeros((64, 64), np.uint8)
    binary[20:40, 20:40] = 255
    edges = subpixel_edges(
        flat,
        binary,
        gradient_sigma_px=cfg.binarize.subpixel.gradient_sigma_px,
        min_gradient_mag=cfg.binarize.subpixel.min_gradient_mag,
        corner_preserve=True,
    )
    assert edges.point_count >= 0  # 不崩即可；无对比度时不做修正
    assert edges.mean_abs_shift_px == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 验收 2：前景 IoU ≥0.98
# ---------------------------------------------------------------------------
def test_foreground_iou_against_truth(cfg: R2VConfig) -> None:
    strokes = [(60.3, 66.3), (150.7, 156.7), (240.1, 246.1), (330.9, 336.9)]
    image, truth = _render(strokes, width=400, height=300)
    result, _ = binarize_image(
        S2Input.from_array(image.astype(np.float32) / 255.0, dpi=300.0), cfg
    )
    predicted = result.binary > 0
    iou = float(np.logical_and(truth, predicted).sum() / np.logical_or(truth, predicted).sum())
    assert iou >= 0.98, f"IoU={iou:.4f} < 0.98"


def test_iou_after_postprocess_removes_specks(cfg: R2VConfig) -> None:
    """salt-and-pepper 噪声在去小对象后不应破坏 IoU。"""
    image, truth = _render([(100.4, 106.4)], width=400, height=300)
    rng = np.random.default_rng(7)
    noisy = image.copy()
    speck = rng.random(image.shape) < 0.002
    noisy[speck] = 0
    result, metrics = binarize_image(
        S2Input.from_array(noisy.astype(np.float32) / 255.0, dpi=300.0), cfg
    )
    predicted = result.binary > 0
    iou = float(np.logical_and(truth, predicted).sum() / np.logical_or(truth, predicted).sum())
    assert iou >= 0.98, f"去噪后 IoU={iou:.4f}"
    assert metrics.counts["components"] >= 1


# ---------------------------------------------------------------------------
# 验收 3：四路选优两次运行一致
# ---------------------------------------------------------------------------
def test_method_selection_is_deterministic(cfg: R2VConfig) -> None:
    image, _ = _render([(80.25, 86.25), (200.75, 206.75)], width=320, height=240)
    source = S2Input.from_array(image.astype(np.float32) / 255.0, dpi=300.0)
    first, metrics_first = binarize_image(source, cfg)
    second, metrics_second = binarize_image(source, cfg)
    assert first.method == second.method
    assert [c.score for c in first.candidates] == [c.score for c in second.candidates]
    assert metrics_first.details["scores"] == metrics_second.details["scores"]
    assert {c.method for c in first.candidates} == set(cfg.binarize.methods)


def test_selection_prefers_fewer_specks() -> None:
    clean = np.zeros((100, 100), np.uint8)
    clean[40:60, 40:60] = 255
    speckled = clean.copy()
    speckled[5, 5] = 255
    speckled[90, 90] = 255
    speck_clean, _, score_clean = foreground_stability(clean)
    speck_noisy, _, score_noisy = foreground_stability(speckled)
    assert score_clean > score_noisy
    assert speck_noisy > speck_clean


# ---------------------------------------------------------------------------
# 验收 4：线宽相对误差 ≤3%
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("width_px", [3.0, 6.0, 10.0])
def test_linewidth_relative_error(cfg: R2VConfig, width_px: float) -> None:
    image, _ = _render([(100.3, 100.3 + width_px)], width=300, height=200)
    result, metrics = binarize_image(
        S2Input.from_array(image.astype(np.float32) / 255.0, dpi=300.0), cfg
    )
    relative = abs(result.linewidth_px - width_px) / width_px
    assert relative <= 0.03, f"线宽真值 {width_px}，估计 {result.linewidth_px}（{relative:.2%}）"
    assert metrics.error_estimate["linewidth_px"] == pytest.approx(result.linewidth_px)


# ---------------------------------------------------------------------------
# 验收 5：metrics_s2 往返
# ---------------------------------------------------------------------------
def test_metrics_s2_roundtrip(cfg: R2VConfig, tmp_path: Path) -> None:
    image, _ = _render([(120.4, 126.4)], width=320, height=240)
    result, metrics = binarize_image(
        S2Input.from_array(image.astype(np.float32) / 255.0, dpi=300.0), cfg, out_dir=tmp_path
    )
    assert (tmp_path / "binary.png").is_file()
    loaded = read_metrics(tmp_path / "metrics_s2.json")
    assert loaded.stage == "s2"
    assert loaded.counts == metrics.counts
    assert loaded.timing_ms >= 0.0
    assert loaded.details["method"] == result.method
    assert loaded.error_estimate["linewidth_px"] == pytest.approx(result.linewidth_px)

    report = json.loads((tmp_path / "s2_report.json").read_text(encoding="utf-8"))
    assert report["method"] == result.method
    assert report["subpixel_points"] == result.edges.point_count
    assert len(report["scores"]) == len(cfg.binarize.methods)


# ---------------------------------------------------------------------------
# 算子级用例
# ---------------------------------------------------------------------------
def test_each_binarizer_returns_binary_mask() -> None:
    image, truth = _render([(100.4, 106.4)], width=300, height=200)
    gray = image.astype(np.float32) / 255.0
    binaries = {
        "otsu": binarize_otsu(gray),
        "sauvola": binarize_sauvola(gray, window=9, k=0.2, r=128.0),
        "niblack": binarize_niblack(gray, window=9, k=-0.2),
        "local_gaussian": binarize_local_gaussian(gray, window=9, offset=5.0),
    }
    for name, binary in binaries.items():
        assert binary.dtype == np.uint8
        assert set(np.unique(binary)).issubset({0, 255}), name
        overlap = float(np.logical_and(truth, binary > 0).sum() / np.logical_or(truth, binary > 0).sum())
        # 局部阈值方法（niblack k=-0.2）在锐利合成图上会略微过分割，这是设计取舍；
        # 选优环节负责挑出更稳的一路，这里只做"确实产出了合理掩膜"的冒烟检查。
        assert overlap > 0.75, f"{name} 与真值重合仅 {overlap:.3f}"


def test_nan_input_rejected(cfg: R2VConfig) -> None:
    bad = np.full((32, 32), np.nan, np.float32)
    with pytest.raises(BinarizeError, match="NaN/Inf"):
        binarize_image(S2Input.from_array(bad, dpi=300.0), cfg)


def test_non_gray_input_rejected() -> None:
    with pytest.raises(BinarizeError, match="灰度图"):
        S2Input.from_array(np.zeros((8, 8, 3), np.float32), dpi=300.0)


def test_empty_image_yields_empty_edges(cfg: R2VConfig) -> None:
    blank = np.ones((64, 64), np.float32)
    result, _ = binarize_image(S2Input.from_array(blank, dpi=300.0), cfg)
    assert result.foreground_px == 0
    assert result.edges.point_count == 0
    assert any("轮廓" in w for w in result.warnings)


def test_postprocess_removes_small_objects(cfg: R2VConfig) -> None:
    binary = np.zeros((100, 100), np.uint8)
    binary[10:20, 10:20] = 255      # 100 px 大块
    binary[80:81, 80:81] = 255      # 1 px 噪点
    warnings: list[str] = []
    tuned = cfg.model_copy(
        update={"binarize": cfg.binarize.model_copy(
            update={"post": cfg.binarize.post.model_copy(update={"fill_holes": False})}
        )}
    )
    out = postprocess(
        binary, cfg=tuned.binarize, dpi=300.0, linewidth_px=4.0, warnings=warnings
    )
    assert out[85, 85] == 0
    assert out[15, 15] == 255
    assert any("去小对象" in w for w in warnings)


def test_subpixel_disabled_is_reported(cfg: R2VConfig) -> None:
    image, _ = _render([(100.4, 106.4)], width=200, height=120)
    tuned = cfg.model_copy(
        update={"binarize": cfg.binarize.model_copy(
            update={"subpixel": cfg.binarize.subpixel.model_copy(update={"enable": False})}
        )}
    )
    result, _ = binarize_image(
        S2Input.from_array(image.astype(np.float32) / 255.0, dpi=300.0), tuned
    )
    assert result.edges.point_count == 0
    assert any("subpixel.enable=false" in w for w in result.warnings)


def test_unimplemented_options_are_reported(cfg: R2VConfig) -> None:
    image, _ = _render([(100.4, 106.4)], width=200, height=120)
    tuned = cfg.model_copy(
        update={"binarize": cfg.binarize.model_copy(
            update={
                "subpixel": cfg.binarize.subpixel.model_copy(
                    update={"interpolation": "parabolic_fit"}
                ),
                "linewidth": cfg.binarize.linewidth.model_copy(update={"method": "steger"}),
            }
        )}
    )
    result, _ = binarize_image(
        S2Input.from_array(image.astype(np.float32) / 255.0, dpi=300.0), tuned
    )
    assert any("parabolic_fit" in w for w in result.warnings)
    assert any("steger" in w for w in result.warnings)


def test_input_from_preprocess_result(cfg: R2VConfig, tmp_path: Path) -> None:
    from src.s1_preprocess import run_s1_preprocess

    from PIL import Image

    image, _ = _render([(100.4, 106.4)], width=240, height=160)
    path = tmp_path / "src.png"
    Image.fromarray(image).save(path, dpi=(300.0, 300.0))
    preprocessed, _ = run_s1_preprocess(path, cfg)
    result, metrics = binarize_image(S2Input.from_preprocess(preprocessed), cfg)
    assert result.foreground_px > 0
    assert metrics.details["mm_per_px"] == pytest.approx(25.4 / 300.0, rel=1e-6)
