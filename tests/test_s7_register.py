# tests/test_s7_register.py
"""S7 配准：四条验收判据。

① 合成平移/旋转/缩放图配准误差 ≤0.5 px；
② mm 坐标换算往返误差 <1e-9；
③ 控制点不足时降级为 identity 并写 WARN；
④ metrics_s7 往返 + 同输入两次逐字节一致。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from src.artifacts import read_metrics
from src.config import R2VConfig, load_config
from src.coords import Transform2D
from src.s7_register import (
    ControlPoint,
    RegistrationError,
    fit_affine,
    fit_projective,
    fit_similarity,
    run_s7_register,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config.yaml"
PX_TO_MM = 25.4 / 300.0


@pytest.fixture(scope="module")
def cfg() -> R2VConfig:
    return load_config(CONFIG)


def _similarity_matrix(*, scale: float, rotation_deg: float, tx: float, ty: float) -> np.ndarray:
    angle = math.radians(rotation_deg)
    matrix = np.eye(3)
    matrix[0, 0] = scale * math.cos(angle)
    matrix[0, 1] = -scale * math.sin(angle)
    matrix[1, 0] = scale * math.sin(angle)
    matrix[1, 1] = scale * math.cos(angle)
    matrix[0, 2] = tx
    matrix[1, 2] = ty
    return matrix


def _apply(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack([points, np.ones(len(points))])
    mapped = homogeneous @ matrix.T
    return mapped[:, :2] / mapped[:, 2:3]


# ---------------------------------------------------------------------------
# 判据 ①：平移 + 旋转 + 缩放
# ---------------------------------------------------------------------------
def test_two_point_registration_within_half_pixel(cfg: R2VConfig) -> None:
    truth = _similarity_matrix(scale=1.7, rotation_deg=23.0, tx=120.0, ty=-45.0)
    source = np.array([[50.0, 60.0], [320.0, 140.0], [180.0, 260.0]])
    target = _apply(truth, source)
    rng = np.random.default_rng(7)
    noisy = target + rng.normal(0.0, 0.08, target.shape)  # mm 级噪声
    pairs = [
        ControlPoint(source=tuple(source[i]), target=tuple(noisy[i]), label=f"p{i}")
        for i in range(len(source))
    ]
    result, metrics = run_s7_register(cfg, control_points=pairs, px_to_mm=PX_TO_MM)
    assert result.mode == "two-point"
    # 误差换算成像素：rms_mm / (px_to_mm × 配准缩放)
    rms_px = result.rms_mm / (PX_TO_MM * result.scale)
    assert rms_px <= 0.5, f"配准误差 {rms_px:.3f} px > 0.5"
    assert abs(result.scale - 1.7) / 1.7 <= 0.01, f"缩放 {result.scale:.4f}"
    assert abs(result.rotation_deg - 23.0) <= 0.5, f"旋转 {result.rotation_deg:.3f}°"
    assert metrics.error_estimate["rms_mm"] == result.rms_mm


def test_affine_registration_recovers_matrix(cfg: R2VConfig) -> None:
    truth = np.array([[1.3, 0.15, 12.0], [-0.08, 1.25, 30.0], [0.0, 0.0, 1.0]])
    source = np.array([[0.0, 0.0], [300.0, 0.0], [0.0, 200.0], [300.0, 200.0], [150.0, 90.0]])
    target = _apply(truth, source)
    pairs = [ControlPoint(source=tuple(s), target=tuple(t)) for s, t in zip(source, target)]
    affine_cfg = cfg.model_copy(
        update={"registration": cfg.registration.model_copy(update={"mode": "affine"})}
    )
    result, _ = run_s7_register(affine_cfg, control_points=pairs, px_to_mm=PX_TO_MM)
    mapped = _apply(np.asarray(result.matrix), source)
    assert np.allclose(mapped, target, atol=1e-9)
    assert result.rms_mm < 1e-9


def test_projective_requires_explicit_acceptance(cfg: R2VConfig) -> None:
    pairs = [
        ControlPoint(source=(0.0, 0.0), target=(0.0, 0.0)),
        ControlPoint(source=(100.0, 0.0), target=(100.0, 0.0)),
        ControlPoint(source=(0.0, 100.0), target=(0.0, 100.0)),
        ControlPoint(source=(100.0, 100.0), target=(100.0, 100.0)),
    ]
    projective_cfg = cfg.model_copy(
        update={"registration": cfg.registration.model_copy(update={"mode": "projective"})}
    )
    result, _ = run_s7_register(projective_cfg, control_points=pairs, px_to_mm=PX_TO_MM)
    assert result.is_identity is True
    assert any("projective" in warning for warning in result.warnings)
    accepted, _ = run_s7_register(
        projective_cfg, control_points=pairs, px_to_mm=PX_TO_MM, accept_projective=True
    )
    assert accepted.is_identity is False


# ---------------------------------------------------------------------------
# 判据 ②：mm 坐标往返误差 < 1e-9
# ---------------------------------------------------------------------------
def test_mm_roundtrip_below_1e9(cfg: R2VConfig) -> None:
    truth = _similarity_matrix(scale=2.5, rotation_deg=-17.0, tx=88.0, ty=42.0)
    source = np.array([[10.0, 20.0], [400.0, 30.0]])
    target = _apply(truth, source)
    pairs = [ControlPoint(source=tuple(s), target=tuple(t)) for s, t in zip(source, target)]
    result, _ = run_s7_register(cfg, control_points=pairs, px_to_mm=PX_TO_MM)
    transform = result.transform()
    mm = transform.apply([(12.34, 56.78), (321.0, 87.5)])
    back = transform.inverse().apply(mm)
    error = float(np.max(np.abs(np.asarray(back) - np.array([[12.34, 56.78], [321.0, 87.5]]))))
    assert error < 1e-9, f"往返误差 {error:.3e}"


# ---------------------------------------------------------------------------
# 判据 ③：控制点不足 → identity + WARN
# ---------------------------------------------------------------------------
def test_insufficient_control_points_degrade_to_identity(cfg: R2VConfig) -> None:
    affine_cfg = cfg.model_copy(
        update={"registration": cfg.registration.model_copy(update={"mode": "affine"})}
    )
    pairs = [ControlPoint(source=(10.0, 10.0), target=(10.0, 10.0))]
    result, metrics = run_s7_register(affine_cfg, control_points=pairs, px_to_mm=PX_TO_MM)
    assert result.is_identity is True
    assert any("降级为 identity" in warning for warning in result.warnings)
    assert metrics.counts["warnings"] >= 1
    matrix = np.asarray(result.matrix)
    assert matrix[0, 0] == pytest.approx(PX_TO_MM)
    assert matrix[1, 1] == pytest.approx(PX_TO_MM)


def test_two_point_too_close_degrades(cfg: R2VConfig) -> None:
    pairs = [
        ControlPoint(source=(10.0, 10.0), target=(0.0, 0.0)),
        ControlPoint(source=(12.0, 10.0), target=(1.0, 0.0)),  # 间距 1mm < 20mm
    ]
    result, _ = run_s7_register(cfg, control_points=pairs, px_to_mm=PX_TO_MM)
    assert result.is_identity is True
    assert any("two-point" in warning for warning in result.warnings)


def test_rms_over_max_raises(cfg: R2VConfig) -> None:
    """系统性错配（不是单个离群点）：稳健剔除救不回来，必须拒绝输出。"""
    source = np.array([[0.0, 0.0], [200.0, 0.0], [200.0, 150.0], [0.0, 150.0], [100.0, 75.0]])
    offsets = np.array([[6.0, 0.0], [-6.0, 0.0], [0.0, 6.0], [0.0, -6.0], [4.0, 4.0]])
    target = source + np.array([5.0, 5.0]) + offsets
    pairs = [ControlPoint(source=tuple(s), target=tuple(t)) for s, t in zip(source, target)]
    affine_cfg = cfg.model_copy(
        update={"registration": cfg.registration.model_copy(update={"mode": "affine"})}
    )
    with pytest.raises(RegistrationError, match="rms_max_mm"):
        run_s7_register(affine_cfg, control_points=pairs, px_to_mm=PX_TO_MM)


def test_outlier_is_rejected(cfg: R2VConfig) -> None:
    truth = _similarity_matrix(scale=1.0, rotation_deg=0.0, tx=5.0, ty=5.0)
    source = np.array([[0.0, 0.0], [200.0, 0.0], [200.0, 150.0], [0.0, 150.0], [100.0, 75.0]])
    target = _apply(truth, source)
    target[2] = target[2] + np.array([9.0, -7.0])  # 一个离群点
    pairs = [ControlPoint(source=tuple(s), target=tuple(t)) for s, t in zip(source, target)]
    affine_cfg = cfg.model_copy(
        update={"registration": cfg.registration.model_copy(update={"mode": "affine"})}
    )
    result, _ = run_s7_register(affine_cfg, control_points=pairs, px_to_mm=PX_TO_MM)
    assert 2 in result.rejected_indices
    assert any("离群" in warning for warning in result.warnings)


# ---------------------------------------------------------------------------
# 判据 ④：metrics 往返 + 逐字节一致
# ---------------------------------------------------------------------------
def test_metrics_roundtrip(cfg: R2VConfig, tmp_path: Path) -> None:
    truth = _similarity_matrix(scale=1.2, rotation_deg=8.0, tx=15.0, ty=-9.0)
    source = np.array([[40.0, 50.0], [300.0, 120.0], [140.0, 240.0]])
    target = _apply(truth, source)
    pairs = [ControlPoint(source=tuple(s), target=tuple(t), label=f"c{i}") for i, (s, t) in enumerate(zip(source, target))]
    result, metrics = run_s7_register(cfg, control_points=pairs, px_to_mm=PX_TO_MM, out_dir=tmp_path)
    assert (tmp_path / "transform.json").is_file()
    loaded = read_metrics(tmp_path / "metrics_s7.json")
    assert loaded.stage == "s7"
    assert loaded.counts["control_points"] == len(pairs)
    assert loaded.error_estimate["rms_mm"] == result.rms_mm
    payload = json.loads((tmp_path / "transform.json").read_text(encoding="utf-8"))
    assert payload["mode"] == result.mode
    assert len(payload["control_points"]) == len(pairs)
    assert payload["matrix"] == [[float(v) for v in row] for row in result.matrix]


def test_results_are_byte_identical(cfg: R2VConfig, tmp_path: Path) -> None:
    truth = _similarity_matrix(scale=1.4, rotation_deg=-12.0, tx=70.0, ty=33.0)
    source = np.array([[20.0, 30.0], [280.0, 90.0], [90.0, 210.0]])
    target = _apply(truth, source)
    pairs = [ControlPoint(source=tuple(s), target=tuple(t)) for s, t in zip(source, target)]
    first, _ = run_s7_register(cfg, control_points=pairs, px_to_mm=PX_TO_MM, out_dir=tmp_path / "a")
    second, _ = run_s7_register(cfg, control_points=pairs, px_to_mm=PX_TO_MM, out_dir=tmp_path / "b")
    assert (tmp_path / "a" / "transform.json").read_bytes() == (tmp_path / "b" / "transform.json").read_bytes()
    assert first.to_json() == second.to_json()
    # metrics 里的 timing_ms 天生逐次不同，只比对数据字段
    first_metrics = json.loads((tmp_path / "a" / "metrics_s7.json").read_text(encoding="utf-8"))
    second_metrics = json.loads((tmp_path / "b" / "metrics_s7.json").read_text(encoding="utf-8"))
    first_metrics.pop("timing_ms", None)
    second_metrics.pop("timing_ms", None)
    assert first_metrics == second_metrics


# ---------------------------------------------------------------------------
# 算子单元
# ---------------------------------------------------------------------------
def test_fit_similarity_is_exact_on_synthetic_data() -> None:
    truth = _similarity_matrix(scale=0.75, rotation_deg=41.0, tx=-13.0, ty=64.0)
    source = np.array([[0.0, 0.0], [100.0, 0.0], [0.0, 80.0]])
    target = _apply(truth, source)
    matrix = fit_similarity(source, target)
    assert np.allclose(matrix, truth, atol=1e-9)


def test_fit_affine_needs_three_points() -> None:
    with pytest.raises(RegistrationError, match="至少需要 3"):
        fit_affine(np.zeros((2, 2)), np.zeros((2, 2)))


def test_fit_projective_needs_four_points() -> None:
    with pytest.raises(RegistrationError, match="至少需要 4"):
        fit_projective(np.zeros((3, 2)), np.zeros((3, 2)))


def test_transform_is_3x3_float64(cfg: R2VConfig) -> None:
    result, _ = run_s7_register(cfg, control_points=[], px_to_mm=PX_TO_MM)
    transform = Transform2D(np.asarray(result.matrix))
    assert transform.matrix.shape == (3, 3)
    assert transform.matrix.dtype == np.float64
