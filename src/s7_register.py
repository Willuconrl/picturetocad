# src/s7_register.py
"""S7 配准与尺度：把像素坐标映射到真实毫米（SPEC §4 S7）。

模式（`registration.mode`）
--------------------------
- `none`      ：只按 DPI 把 px 换成 mm（纸面尺寸），**不建立**世界坐标 → 必写 WARN；
- `scale`     ：按 `scale_denominator`（1:N）放大纸面尺寸；
- `two-point` ：两点法（平移 + 旋转 + 等比缩放，Helmert/相似变换）；
- `affine`    ：≥`affine_min_control_points` 个控制点，最小二乘（允许各向异性与剪切）；
- `projective`：≥`projective_min_control_points` 个控制点，DLT 求单应（角度/面积不再保真）。

质量闸门
--------
- `rms_warn_mm`：超过只 WARN；
- `rms_max_mm` ：超过**抛错**（`RegistrationError`），不输出不可信结果（SPEC §4 S7）；
- `outlier_sigma`：残差超过 mean+kσ 的控制点判为离群，剔除后重算，并在结果里列出。

坐标契约
--------
输入控制点是**连续像素坐标**（像素中心 = i+0.5，见 `docs/coordinate_system.md`），
输出是 3×3 float64 齐次矩阵（px → mm），与 `src/coords.py` 的 `Transform2D` 同构。
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from src.artifacts import StageMetrics, write_json, write_metrics
from src.config import R2VConfig
from src.coords import Transform2D

RegistrationModeName = Literal["none", "scale", "two-point", "affine", "projective"]


class RegistrationError(ValueError):
    """配准失败（控制点不足以外的原因：残差超限、点退化等）。"""


class ControlPoint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source: tuple[float, float] = Field(..., description="图像连续像素坐标 (x, y)")
    target: tuple[float, float] = Field(..., description="世界坐标 mm (x, y)")
    label: str = ""


class RegistrationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: RegistrationModeName
    matrix: tuple[tuple[float, float, float], ...]
    is_identity: bool
    scale: float
    rotation_deg: float
    rms_mm: float
    max_residual_mm: float
    residuals_mm: tuple[float, ...]
    rejected_indices: tuple[int, ...]
    control_points_used: int
    warnings: tuple[str, ...]

    def transform(self) -> Transform2D:
        return Transform2D(np.asarray(self.matrix, dtype=np.float64))

    def to_json(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "matrix": [[float(v) for v in row] for row in self.matrix],
            "is_identity": self.is_identity,
            "scale": self.scale,
            "rotation_deg": self.rotation_deg,
            "rms_mm": self.rms_mm,
            "max_residual_mm": self.max_residual_mm,
            "residuals_mm": [round(v, 9) for v in self.residuals_mm],
            "rejected_indices": list(self.rejected_indices),
            "control_points_used": self.control_points_used,
            "warnings": list(self.warnings),
        }


# ---------------------------------------------------------------------------
# 变换求解
# ---------------------------------------------------------------------------
def _as_arrays(pairs: Sequence[ControlPoint]) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray([p.source for p in pairs], dtype=np.float64)
    target = np.asarray([p.target for p in pairs], dtype=np.float64)
    if source.ndim != 2 or source.shape[1] != 2:
        raise RegistrationError("控制点 source 必须是 (N,2)")
    if not np.all(np.isfinite(source)) or not np.all(np.isfinite(target)):
        raise RegistrationError("控制点包含 NaN/Inf")
    return source, target


def fit_similarity(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Helmert 相似变换（Umeyama）：平移 + 旋转 + 等比缩放。"""
    if len(source) < 2:
        raise RegistrationError("two-point 模式需要 2 个控制点")
    src_mean = source.mean(axis=0)
    dst_mean = target.mean(axis=0)
    src_centered = source - src_mean
    dst_centered = target - dst_mean
    covariance = dst_centered.T @ src_centered / len(source)
    u, s, vt = np.linalg.svd(covariance)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    variance = float((src_centered**2).sum() / len(source))
    if variance < 1e-12:
        raise RegistrationError("控制点退化（source 全重合），无法求相似变换")
    scale = float(s.sum() / variance)
    translation = dst_mean - scale * (rotation @ src_mean)
    matrix = np.eye(3)
    matrix[:2, :2] = scale * rotation
    matrix[:2, 2] = translation
    return matrix


def fit_affine(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """最小二乘仿射（≥3 点）。"""
    if len(source) < 3:
        raise RegistrationError("affine 模式至少需要 3 个控制点")
    design = np.column_stack([source, np.ones(len(source))])
    solution, *_ = np.linalg.lstsq(design, target, rcond=None)
    matrix = np.eye(3)
    matrix[:2, :] = solution.T
    return matrix


def fit_projective(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """DLT 单应（≥4 点）。"""
    if len(source) < 4:
        raise RegistrationError("projective 模式至少需要 4 个控制点")
    rows = []
    for (x, y), (u, v) in zip(source, target):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, vt = np.linalg.svd(np.asarray(rows, dtype=np.float64))
    matrix = vt[-1].reshape(3, 3)
    if abs(matrix[2, 2]) < 1e-12:
        raise RegistrationError("单应矩阵退化（h33≈0）")
    return matrix / matrix[2, 2]


def _apply(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack([points, np.ones(len(points))])
    mapped = homogeneous @ matrix.T
    w = mapped[:, 2:3]
    w[np.abs(w) < 1e-12] = 1e-12
    return mapped[:, :2] / w


def _decompose(matrix: np.ndarray) -> tuple[float, float]:
    """返回 (平均缩放, 旋转角°)。"""
    a, b = matrix[0, 0], matrix[0, 1]
    c, d = matrix[1, 0], matrix[1, 1]
    scale = math.sqrt(abs(a * d - b * c))
    rotation = math.degrees(math.atan2(c, a))
    return scale, rotation


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def _trimmed_solve(
    mode: RegistrationModeName,
    source: np.ndarray,
    target: np.ndarray,
    required: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """小样本情形下的稳健求解：枚举最小子集，取"去掉最大残差后误差最小"的那组。

    最小二乘会把单个错点的误差摊到所有点上（实测 5 点仿射：错点残差 3.42mm、
    好点 2.28mm，mean+3σ / MAD 都判不出来）。枚举最小子集（C(n,3) 很小）可以
    做到"少一个点也要拟合得好"，从而把错点暴露出来。
    """
    from itertools import combinations

    n = len(source)
    if n > 8 or n <= required:
        return None
    best: tuple[float, np.ndarray] | None = None
    for subset in combinations(range(n), required):
        index = list(subset)
        try:
            matrix = _solve(mode, source[index], target[index])
        except (RegistrationError, np.linalg.LinAlgError):
            continue
        residuals = np.hypot(*(_apply(matrix, source) - target).T)
        cost = float(np.sort(residuals)[: n - 1].sum())
        if best is None or cost < best[0]:
            best = (cost, matrix)
    if best is None:
        return None
    matrix = best[1]
    return matrix, np.hypot(*(_apply(matrix, source) - target).T)


def run_s7_register(
    cfg: R2VConfig,
    *,
    control_points: Sequence[ControlPoint] = (),
    px_to_mm: float,
    out_dir: str | Path | None = None,
    image_size_px: tuple[int, int] | None = None,
    accept_projective: bool = False,
) -> tuple[RegistrationResult, StageMetrics]:
    """执行配准。`px_to_mm` 是无配准时 1 px 对应的纸面毫米数（25.4/dpi）。"""
    t0 = time.perf_counter()
    registration = cfg.registration
    mode = registration.mode
    warnings: list[str] = []
    degraded = False

    def identity_scale(value: float, why: str) -> np.ndarray:
        nonlocal degraded
        degraded = True
        warnings.append(why)
        matrix = np.eye(3)
        matrix[0, 0] = matrix[1, 1] = value
        return matrix

    if mode == "none":
        matrix = identity_scale(px_to_mm, "registration.mode=none：只做纸面尺寸换算，未建立世界坐标，尺寸不可信")
    elif mode == "scale":
        denominator = registration.scale_denominator
        if denominator is None:  # schema 已保证，这里兜底
            raise RegistrationError("mode=scale 但缺少 scale_denominator")
        matrix = identity_scale(
            px_to_mm * float(denominator),
            f"registration.mode=scale：按 1:{denominator:g} 放大纸面尺寸（假定图纸无旋转/畸变）",
        )
    elif mode == "projective" and registration.projective_requires_perspective_accept and not accept_projective:
        matrix = identity_scale(
            px_to_mm,
            "registration.mode=projective 但未显式接受透视校正（projective_requires_perspective_accept）："
            "降级为 identity，尺寸不可信",
        )
    else:
        required = 2 if mode == "two-point" else (
            registration.affine_min_control_points if mode == "affine"
            else registration.projective_min_control_points
        )
        if len(control_points) < required:
            matrix = identity_scale(
                px_to_mm,
                f"registration.mode={mode} 需要 ≥{required} 个控制点，实际 {len(control_points)}："
                "降级为 identity（仅纸面换算），尺寸不可信",
            )
        else:
            source, target = _as_arrays(control_points)
            matrix = _solve(mode, source, target)
            if mode == "two-point":
                span = float(np.hypot(*(target[1] - target[0])))
                if span < float(registration.two_point_min_distance_mm):
                    matrix = identity_scale(
                        px_to_mm,
                        f"two-point 两控制点间距 {span:.2f}mm < two_point_min_distance_mm "
                        f"{registration.two_point_min_distance_mm:g}mm：旋转/缩放不可信，降级 identity",
                    )

    # 残差与离群剔除
    residuals = np.zeros(len(control_points))
    rejected: list[int] = []
    if len(control_points) >= 2:
        source, target = _as_arrays(control_points)
        residuals = np.hypot(*(_apply(matrix, source) - target).T)
        if len(control_points) >= 4:
            required = 2 if mode == "two-point" else (
                registration.affine_min_control_points if mode == "affine"
                else registration.projective_min_control_points
            )
            # 先用"最小子集枚举"探一遍：LS 会把错点误差摊薄，MAD 判不出来；
            # 探出候选后再用**全部保留点**重做 LS，兼顾鲁棒与精度。
            probe = _trimmed_solve(mode, source, target, required)
            probe_residuals = probe[1] if probe is not None else residuals
            median = float(np.median(probe_residuals))
            mad = float(np.median(np.abs(probe_residuals - median)))
            sigma = max(1.4826 * mad, 1e-9)
            threshold = median + registration.outlier_sigma * sigma
            candidates = [
                i for i, value in enumerate(probe_residuals)
                if value > threshold and value > 1e-6
            ]
            if probe is not None and candidates and len(candidates) < len(control_points) - 1:
                kept = [p for i, p in enumerate(control_points) if i not in candidates]
                # 保留点必须**超定**（≥ 参数个数+1），否则剔除后能精确穿过剩余点，
                # RMS 恒为 0，等于用"删点"来伪造精度（实测 5 点删到 3 点、RMS 0.0）
                if _enough(mode, len(kept), registration) and len(kept) >= required + 1:
                    source2, target2 = _as_arrays(kept)
                    matrix = _solve(mode, source2, target2)
                    residuals = np.hypot(*(_apply(matrix, source) - target).T)
                    rejected = candidates
                    warnings.append(
                        f"剔除 {len(candidates)} 个离群控制点（残差 > median+"
                        f"{registration.outlier_sigma:g}σ）：{candidates}"
                    )

    # 质量闸门只看**内点**：被剔除的离群点残差本来就应该大，
    # 把它算进 RMS 会让"已经修好"的结果仍然被拒（实测 11.4mm 错点把 RMS 拉到 5.1mm）
    inlier_mask = np.ones(len(residuals), dtype=bool)
    if rejected:
        inlier_mask[list(rejected)] = False
    gated = residuals[inlier_mask] if inlier_mask.any() else residuals
    rms = float(np.sqrt(np.mean(gated**2))) if gated.size else 0.0
    max_residual = float(gated.max()) if gated.size else 0.0
    if rms > registration.rms_max_mm and not degraded:
        raise RegistrationError(
            f"配准 RMS {rms:.3f}mm > rms_max_mm {registration.rms_max_mm:g}mm：拒绝输出（SPEC §4 S7）"
        )
    if degraded and rms > registration.rms_max_mm:
        warnings.append(
            f"已降级为 identity，但控制点残差 RMS {rms:.3f}mm 仍 > rms_max_mm "
            f"{registration.rms_max_mm:g}mm：控制点本身可能不匹配，尺寸不可信"
        )
    elif rms > registration.rms_warn_mm:
        warnings.append(f"配准 RMS {rms:.3f}mm > rms_warn_mm {registration.rms_warn_mm:g}mm：精度已受影响")

    scale, rotation = _decompose(matrix)
    is_identity = bool(
        np.allclose(matrix[:2, :2], np.eye(2) * px_to_mm, atol=1e-12)
        and np.allclose(matrix[:2, 2], 0.0, atol=1e-12)
    )
    result = RegistrationResult(
        mode=mode,
        matrix=tuple(tuple(float(v) for v in row) for row in matrix),
        is_identity=bool(is_identity),
        scale=round(scale, 9),
        rotation_deg=round(rotation, 6),
        rms_mm=round(rms, 6),
        max_residual_mm=round(max_residual, 6),
        residuals_mm=tuple(round(float(v), 9) for v in residuals),
        rejected_indices=tuple(rejected),
        control_points_used=len(control_points) - len(rejected),
        warnings=tuple(warnings),
    )

    timing_ms = round((time.perf_counter() - t0) * 1000.0, 3)
    metrics = StageMetrics(
        stage="s7",
        counts={
            "control_points": len(control_points),
            "rejected": len(rejected),
            "used": result.control_points_used,
            "warnings": len(warnings),
        },
        timing_ms=timing_ms,
        error_estimate={
            "rms_mm": result.rms_mm,
            "max_residual_mm": result.max_residual_mm,
            "scale": result.scale,
            "rotation_deg": result.rotation_deg,
        },
        warnings=tuple(warnings),
        details={
            "mode": mode,
            "is_identity": result.is_identity,
            "rejected_indices": list(rejected),
            "image_size_px": list(image_size_px) if image_size_px else None,
            "px_to_mm": round(px_to_mm, 12),
        },
    )

    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "transform.json", {
            **result.to_json(),
            "control_points": [
                {"source": list(p.source), "target": list(p.target), "label": p.label}
                for p in control_points
            ],
        })
        write_metrics(out, metrics)

    return result, metrics


def _enough(mode: RegistrationModeName, count: int, registration: Any) -> bool:
    if mode == "two-point":
        return count >= 2
    if mode == "affine":
        return count >= registration.affine_min_control_points
    if mode == "projective":
        return count >= registration.projective_min_control_points
    return False


def _solve(mode: RegistrationModeName, source: np.ndarray, target: np.ndarray) -> np.ndarray:
    if mode == "two-point":
        return fit_similarity(source, target)
    if mode == "affine":
        return fit_affine(source, target)
    if mode == "projective":
        return fit_projective(source, target)
    raise RegistrationError(f"未知配准模式：{mode}")


__all__ = [
    "ControlPoint",
    "RegistrationError",
    "RegistrationResult",
    "fit_affine",
    "fit_projective",
    "fit_similarity",
    "run_s7_register",
]
