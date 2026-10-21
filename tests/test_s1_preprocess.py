# tests/test_s1_preprocess.py
"""S1 预处理：四条验收判据 + 各算子边界（SPEC §4 S1）。

验收判据（用户确认）：
1. 去噪/平坦化后**边缘位移 ≤0.3 px**；
2. 合成梯度光照图平坦化后分块 CV < 0.05（原图约 0.42）；
3. 反白检测正片/负片 100% 正确；
4. 超采样 factor=2：尺寸翻倍、mm/px 减半、mm 坐标往返误差 <1e-9。
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw

from src.artifacts import read_metrics
from src.config import R2VConfig, load_config
from src.coords import Transform2D
from src.s0_qc import lighting_blocks
from src.s1_preprocess import (
    PreprocessError,
    bt709_gray,
    crop_dark_border,
    denoise,
    detect_inverted,
    estimate_linewidth_px,
    flatten_background,
    load_image,
    run_s1_preprocess,
    upsample,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config.yaml"


@pytest.fixture(scope="module")
def cfg() -> R2VConfig:
    return load_config(CONFIG)


# ---------------------------------------------------------------------------
# 夹具
# ---------------------------------------------------------------------------
def _save_png(path: Path, array: np.ndarray, dpi: tuple[float, float] | None = (300.0, 300.0)) -> Path:
    kwargs = {"dpi": dpi} if dpi is not None else {}
    Image.fromarray(np.asarray(array, dtype=np.uint8)).save(path, format="PNG", **kwargs)
    return path


def _lines(width: int = 472, height: int = 354, *, thickness: int = 3) -> np.ndarray:
    """线间距 ~120px 的工程图式网格（间距必须明显大于背景核 20×线宽）。"""
    canvas = np.full((height, width), 255, np.uint8)
    for x in range(50, width - 20, 120):
        cv2.line(canvas, (x, 10), (x, height - 10), 0, thickness)
    for y in range(35, height - 20, 100):
        cv2.line(canvas, (10, y), (width - 10, y), 0, thickness)
    return canvas


def _edge_image(
    width: int = 400, height: int = 200, *, x0: float = 140.37, band: int = 60
) -> np.ndarray:
    """白底上一条 60px 宽的暗带，左边界有 1px 软过渡（50% 阈值恰在 x0+0.5）。

    为什么不做「左半黑右半白」：那种图会被 S1 的黑边自动检测当成扫描暗边裁掉，
    测出来的位移是裁切量而不是滤波位移。窄暗带四周都是白边，自动裁切不触发。
    """
    xx = np.arange(width, dtype=np.float32)[None, :].repeat(height, 0)
    ramp = np.minimum(xx - x0, (x0 + band) - xx) / 1.0
    return (np.clip(ramp, 0.0, 1.0) * 255.0).astype(np.uint8)


def _edge_x_subpixel(gray: np.ndarray, *, level: float = 0.5) -> float:
    """用行剖面中位数测边缘的亚像素位置（先归一化，消除对比度差异）。"""
    profile = gray.astype(np.float64).mean(axis=0)
    lo, hi = float(np.percentile(profile, 1)), float(np.percentile(profile, 99))
    if hi - lo < 1e-6:
        raise AssertionError("图像没有对比度，无法测边缘")
    norm = (profile - lo) / (hi - lo)
    target = level
    for index in range(len(norm) - 1):
        a, b = norm[index], norm[index + 1]
        if (a - target) * (b - target) <= 0 and a != b:
            return float(index + (target - a) / (b - a))
    raise AssertionError("找不到边缘穿越点")


# ---------------------------------------------------------------------------
# 验收 1：边缘位移 ≤0.3 px
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("method", ["nlmeans", "bilateral"])
def test_edge_displacement_within_budget(
    cfg: R2VConfig, tmp_path: Path, method: str
) -> None:
    """只开去噪（关掉平坦化），隔离出滤波本身造成的边缘位移。"""
    image = _edge_image()
    path = _save_png(tmp_path / f"edge_{method}.png", image)
    before = _edge_x_subpixel(image.astype(np.float32) / 255.0)

    tuned = cfg.model_copy(
        update={"preprocess": cfg.preprocess.model_copy(
            update={
                "denoise": cfg.preprocess.denoise.model_copy(update={"method": method}),
                "background_flatten": cfg.preprocess.background_flatten.model_copy(
                    update={"enable": False}
                ),
            }
        )}
    )
    result, _ = run_s1_preprocess(path, tuned)
    after = _edge_x_subpixel(result.image)
    assert abs(after - before) <= 0.3, f"{method}: 边缘位移 {after - before:+.3f} px 超过 0.3 px"


def _line_centre_x(gray: np.ndarray, *, near: int = 50, half: int = 12) -> float:
    """测某条竖线的亚像素质心（对对比度变化稳健）。"""
    columns = np.arange(max(0, near - half), min(gray.shape[1], near + half))
    weights = (1.0 - gray.astype(np.float64)[:, columns]).sum(axis=0)
    total = weights.sum()
    if total <= 1e-9:
        raise AssertionError("窗口内没有线条，无法测中心")
    return float((columns * weights).sum() / total)


def test_edge_displacement_survives_flatten(cfg: R2VConfig, tmp_path: Path) -> None:
    """背景平坦化（除法归一）不得移动线条中心。"""
    image = _lines()
    path = _save_png(tmp_path / "edge_flat.png", image)
    before = _line_centre_x(image.astype(np.float32) / 255.0)

    tuned = cfg.model_copy(
        update={"preprocess": cfg.preprocess.model_copy(
            update={"background_flatten": cfg.preprocess.background_flatten.model_copy(
                update={"enable": True}
            )}
        )}
    )
    result, _ = run_s1_preprocess(path, tuned)
    after = _line_centre_x(result.image)
    assert result.background_flattened is True
    assert abs(after - before) <= 0.3, f"平坦化后线心位移 {after - before:+.3f} px"


# ---------------------------------------------------------------------------
# 验收 2：光照不均被压平
# ---------------------------------------------------------------------------
def test_background_flatten_reduces_block_cv(cfg: R2VConfig, tmp_path: Path) -> None:
    h, w = 354, 472
    lines = _lines(w, h)
    gradient = np.linspace(40, 250, w, dtype=np.float32)[None, :].repeat(h, 0) / 255.0
    shaded = np.clip(lines.astype(np.float32) * gradient, 0, 255).astype(np.uint8)
    path = _save_png(tmp_path / "shaded.png", shaded)

    before_cv, _ = lighting_blocks(shaded, block_ratio=cfg.qc.shadow_block_size_ratio)
    assert before_cv > cfg.qc.shadow_var_threshold, before_cv

    tuned = cfg.model_copy(
        update={"preprocess": cfg.preprocess.model_copy(
            update={"background_flatten": cfg.preprocess.background_flatten.model_copy(
                update={"enable": True, "method": "morph_open"}
            )}
        )}
    )
    result, metrics = run_s1_preprocess(path, tuned)
    after_cv, _ = lighting_blocks(
        (result.image * 255).astype(np.uint8), block_ratio=cfg.qc.shadow_block_size_ratio
    )
    assert after_cv < 0.05, f"平坦化后分块 CV={after_cv:.4f} 未降到 0.05 以下（原 {before_cv:.4f}）"
    assert metrics.details["background_flattened"] is True


def test_auto_flatten_follows_lighting_detection(cfg: R2VConfig, tmp_path: Path) -> None:
    """enable=auto：平坦光照不启用，梯度光照启用。"""
    flat = _save_png(tmp_path / "flat.png", _lines())
    result_flat, _ = run_s1_preprocess(flat, cfg)
    assert result_flat.background_flattened is False

    h, w = 354, 472
    gradient = np.linspace(40, 250, w, dtype=np.float32)[None, :].repeat(h, 0) / 255.0
    shaded = np.clip(_lines(w, h).astype(np.float32) * gradient, 0, 255).astype(np.uint8)
    result_shaded, _ = run_s1_preprocess(_save_png(tmp_path / "shaded.png", shaded), cfg)
    assert result_shaded.background_flattened is True


# ---------------------------------------------------------------------------
# 验收 3：反白检测
# ---------------------------------------------------------------------------
def test_invert_detection_positive_and_negative(cfg: R2VConfig, tmp_path: Path) -> None:
    positive = _lines()
    negative = 255 - positive

    pos_path = _save_png(tmp_path / "positive.png", positive)
    neg_path = _save_png(tmp_path / "negative.png", negative)

    pos_result, _ = run_s1_preprocess(pos_path, cfg)
    neg_result, _ = run_s1_preprocess(neg_path, cfg)

    assert pos_result.inverted is False
    assert neg_result.inverted is True
    # 两者的产物都应是"深线浅底"
    for result in (pos_result, neg_result):
        assert float(result.image.mean()) > 0.5, "反转后背景应变亮"
        assert float(result.image.min()) < 0.3, "反转后线条应变暗"


def test_invert_flag_matches_helper(cfg: R2VConfig) -> None:
    gray = _lines().astype(np.float32) / 255.0
    assert detect_inverted(gray, max_foreground_ratio=cfg.preprocess.invert_foreground_ratio) is False
    assert detect_inverted(1.0 - gray, max_foreground_ratio=cfg.preprocess.invert_foreground_ratio) is True


# ---------------------------------------------------------------------------
# 验收 4：超采样
# ---------------------------------------------------------------------------
def test_upsample_doubles_size_and_halves_mm_per_px(cfg: R2VConfig, tmp_path: Path) -> None:
    source = _save_png(tmp_path / "thin.png", _lines(thickness=2))
    tuned = cfg.model_copy(
        update={"preprocess": cfg.preprocess.model_copy(
            update={"upsample_if_thin": cfg.preprocess.upsample_if_thin.model_copy(
                update={"enable": True, "min_linewidth_px": 10.0, "factor": 2}
            )}
        )}
    )
    result, metrics = run_s1_preprocess(source, tuned)
    assert result.upsample_factor == 2
    assert result.width_px == 472 * 2
    assert result.height_px == 354 * 2
    assert result.mm_per_px == pytest.approx((25.4 / 300.0) / 2, rel=1e-12)
    assert metrics.counts["upsample_factor"] == 2


def test_upsample_mm_roundtrip_below_1e9(cfg: R2VConfig, tmp_path: Path) -> None:
    """同一物理点在原图与超采样图中的 mm 坐标必须一致（误差 <1e-9）。"""
    source = _save_png(tmp_path / "thin.png", _lines(thickness=2))
    tuned = cfg.model_copy(
        update={"preprocess": cfg.preprocess.model_copy(
            update={"upsample_if_thin": cfg.preprocess.upsample_if_thin.model_copy(
                update={"enable": True, "min_linewidth_px": 10.0, "factor": 2}
            )}
        )}
    )
    result, _ = run_s1_preprocess(source, tuned)

    dpi = result.dpi
    before = Transform2D.image_to_cad(dpi=dpi, height_px=354)
    # 超采样后等效 DPI 翻倍：mm/px 减半，故 (x,y) 与 (2x,2y) 必须映射到同一 mm 点
    after = Transform2D.image_to_cad(dpi=dpi * result.upsample_factor, height_px=result.height_px)
    points_before = np.array([[100.0, 50.0], [0.0, 0.0], [471.0, 353.0]])
    points_after = points_before * result.upsample_factor
    np.testing.assert_allclose(
        before.apply(points_before), after.apply(points_after), atol=1e-9, rtol=0
    )


def test_upsample_disabled_keeps_size(cfg: R2VConfig, tmp_path: Path) -> None:
    source = _save_png(tmp_path / "thick.png", _lines(thickness=4))
    result, _ = run_s1_preprocess(source, cfg)
    assert result.upsample_factor == 1
    assert (result.width_px, result.height_px) == (472, 354)


def test_upsample_rejects_illegal_factor() -> None:
    with pytest.raises(PreprocessError, match="只允许 2 或 3"):
        upsample(np.zeros((8, 8), np.float32), factor=4, kernel="lanczos")


# ---------------------------------------------------------------------------
# 算子级用例
# ---------------------------------------------------------------------------
def test_bt709_weights() -> None:
    pixels = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255]]], np.uint8)
    gray = bt709_gray(pixels)[0]
    assert gray[0] == pytest.approx(0.2126, abs=1e-4)
    assert gray[1] == pytest.approx(0.7152, abs=1e-4)
    assert gray[2] == pytest.approx(0.0722, abs=1e-4)
    assert gray[3] == pytest.approx(1.0, abs=1e-6)


def test_bt709_rejects_bad_shape() -> None:
    with pytest.raises(PreprocessError, match="需要 2D 或 RGB"):
        bt709_gray(np.zeros((4, 4, 2), np.uint8))


def test_estimate_linewidth_close_to_truth() -> None:
    """墨迹宽度 = 前景面积 / 骨架长度：5 行宽的实心带必须给 ≈5。"""
    canvas = np.full((200, 200), 255, np.uint8)
    cv2.rectangle(canvas, (20, 100), (180, 104), 0, -1)  # 恰好 5 行
    width = estimate_linewidth_px(canvas.astype(np.float32) / 255.0)
    assert 4.5 <= width <= 5.5, width

    thin = np.full((200, 200), 255, np.uint8)
    cv2.line(thin, (20, 100), (180, 100), 0, 1)
    assert estimate_linewidth_px(thin.astype(np.float32) / 255.0) == pytest.approx(1.0, abs=0.1)


def test_estimate_linewidth_of_blank_is_zero() -> None:
    assert estimate_linewidth_px(np.ones((32, 32), np.float32)) == 0.0


def test_flatten_background_methods_reduce_gradient() -> None:
    h, w = 120, 160
    gradient = np.linspace(0.2, 0.95, w, dtype=np.float32)[None, :].repeat(h, 0)
    for method in ("morph_open", "polynomial", "gaussian_ilr"):
        flat, background = flatten_background(
            gradient, method=method, kernel_px=31, polynomial_degree=2
        )
        assert flat.shape == gradient.shape
        assert background.min() > 0
        cv_after, _ = lighting_blocks((flat * 255).astype(np.uint8), block_ratio=4)
        assert cv_after < 0.05, f"{method}: CV={cv_after:.4f}"


def test_flatten_background_rejects_unknown_method() -> None:
    with pytest.raises(PreprocessError, match="未知的背景平坦化方法"):
        flatten_background(
            np.zeros((8, 8), np.float32), method="magic", kernel_px=5, polynomial_degree=2
        )


def test_denoise_none_is_identity() -> None:
    gray = np.random.default_rng(0).random((16, 16)).astype(np.float32)
    out = denoise(gray, method="none", h_normalized=0.04, patch_px=5, search_px=21)
    np.testing.assert_array_equal(out, gray)


def test_crop_dark_border_explicit_and_auto() -> None:
    canvas = np.full((100, 100), 255, np.uint8)
    canvas[:6, :] = 0
    canvas[-4:, :] = 0
    cropped, box = crop_dark_border(canvas.astype(np.float32) / 255.0, clip_px=0)
    assert box[0] > 0 and box[1] > 0
    assert cropped.shape[0] < 100

    cropped2, box2 = crop_dark_border(canvas.astype(np.float32) / 255.0, clip_px=5)
    assert box2 == (5, 5, 5, 5)
    assert cropped2.shape == (90, 90)


def test_load_image_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(PreprocessError, match="图像不存在"):
        load_image(tmp_path / "nope.png")


def test_load_image_corrupt_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image")
    with pytest.raises(PreprocessError, match="无法解码"):
        load_image(bad)


# ---------------------------------------------------------------------------
# 产物：metrics_s1 + preprocessed.png
# ---------------------------------------------------------------------------
def test_metrics_s1_roundtrip(cfg: R2VConfig, tmp_path: Path) -> None:
    source = _save_png(tmp_path / "src.png", _lines())
    out = tmp_path / "out"
    result, metrics = run_s1_preprocess(source, cfg, out_dir=out)

    assert (out / "preprocessed.png").is_file()
    assert (out / "metrics_s1.json").is_file()
    loaded = read_metrics(out / "metrics_s1.json")
    assert loaded.stage == "s1"
    assert loaded.counts == metrics.counts
    assert loaded.timing_ms >= 0.0
    assert loaded.error_estimate["mm_per_px"] == pytest.approx(result.mm_per_px, rel=1e-6)
    assert loaded.details["source_sha256"] == result.source_sha256

    report = json.loads((out / "preprocess_report.json").read_text(encoding="utf-8"))
    assert report["upsample_factor"] == result.upsample_factor
    assert report["mm_per_px"] == pytest.approx(result.mm_per_px, rel=1e-12)


def test_skew_correction_is_recorded(cfg: R2VConfig, tmp_path: Path) -> None:
    source = _save_png(tmp_path / "src.png", _lines())
    result, metrics = run_s1_preprocess(source, cfg, rotation_deg=-2.0)
    assert result.skew_correction_applied is True
    assert metrics.details["skew_correction_applied"] is True
    assert result.rotation_deg == pytest.approx(-2.0)

    plain, metrics_plain = run_s1_preprocess(source, cfg)
    assert plain.skew_correction_applied is False
    assert metrics_plain.details["skew_correction_applied"] is False


def test_output_is_float32_unit_range(cfg: R2VConfig, tmp_path: Path) -> None:
    source = _save_png(tmp_path / "src.png", _lines())
    result, _ = run_s1_preprocess(source, cfg)
    assert result.image.dtype == np.float32
    assert result.image.min() >= 0.0 and result.image.max() <= 1.0


def test_dpi_fallback_used_when_missing(cfg: R2VConfig, tmp_path: Path) -> None:
    source = _save_png(tmp_path / "nodpi.png", _lines(), dpi=None)
    result, _ = run_s1_preprocess(source, cfg)
    assert result.dpi == pytest.approx(cfg.qc.fallback_dpi)
    assert any("fallback_dpi" in w for w in result.warnings)
