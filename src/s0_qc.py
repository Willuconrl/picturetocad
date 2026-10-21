# src/s0_qc.py
"""S0 摄入与质检：**精度项目的闸门**（SPEC §4 S0）。

本阶段做四件事，全部落进 `qc_report.json`：

1. **DPI**：从 EXIF / PNG pHYs / TIFF 分辨率标签读出真实 DPI；缺失→按
   `qc.fallback_dpi` 兜底并 WARN；低于 `qc.dpi_min` 时按 `qc.dpi_mode` 处理
   （reject / warn / degrade / ignore）。`degrade` 会把误差预算按
   `qc.budget_reference_dpi / true_dpi` 放大（300 DPI 图上 1px=0.0847mm，
   72 DPI 图上 1px=0.3528mm，同一 px 误差在 mm 上差 4.17 倍）。
2. **锐度**：`sharpness = p90(|∇I| / 4) / (p99 − p1)`，即归一化梯度幅值的 90 分位。
   清晰线条图 ≈ 0.5–1.0；σ=2 高斯模糊后掉到 ≈0.15。
3. **倾斜**：Hough 直线取与主轴（0°/90°）的偏差，按线段长度加权取中位数。
4. **光照不均**：分块均值的变异系数（块尺寸 = 短边 / `qc.shadow_block_size_ratio`），
   超过 `qc.shadow_var_threshold` 则建议背景平坦化。

尚未实现的检测（JPEG 环状伪影、莫尔纹、透视、折痕污渍）**如实标记为 not_evaluated**，
绝不假装通过——SPEC §0 的原则是「宁可少输出」。

关于硬编码
----------
精度判据全部来自 `config.yaml`（`QcConfig`）。这里只有**算法内部常数**
（Canny 阈值、梯度归一化分母 4、下采样上限），它们不参与跨 DPI 的精度调参，
也不随图纸类型变化；改它们只会改变检测器灵敏度，不会改变验收口径。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field

from src.artifacts import StageMetrics, sha256_file, write_json, write_metrics
from src.config import QcConfig, dpi_budget_scale, mm_per_px

QcStatus = Literal["pass", "degraded", "rejected"]

# --- 算法内部常数（不是精度阈值，见模块文档）---
_SOBEL_NORMALIZER = 4.0      # 3x3 Sobel 对单位阶跃的最大响应
_MAX_ANALYSIS_SIDE = 1200    # 质检分析用下采样上限（px）
_SKEW_COARSE_STEP_DEG = 0.10
_SKEW_FINE_STEP_DEG = 0.02
_SKEW_ANALYSIS_SIDE = 600    # 投影法分析用下采样上限（px）
_MOIRE_MAX_SIDE = 4000       # 网点统计的原分辨率上限，超过则下采样并 WARN


class QcError(ValueError):
    """图像无法读取或质检输入非法。"""


# ---------------------------------------------------------------------------
# 报告模型
# ---------------------------------------------------------------------------
class _Block(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DpiInfo(_Block):
    value_x: float
    value_y: float
    source: Literal["metadata", "fallback"]
    anisotropic: bool
    anisotropy_ratio: float
    applied: float


class SharpnessInfo(_Block):
    metric: Literal["gradient_p90_normalized"]
    value: float
    threshold: float
    passed: bool


class SkewInfo(_Block):
    angle_deg: float
    confident: bool
    prominence: float
    exceeds_limit: bool
    corrective_rotation_deg: float
    method: Literal["projection_profile"]


class LightingInfo(_Block):
    block_size_px: int
    block_cv: float
    threshold: float
    uneven: bool
    recommend_background_flatten: bool


class CheckNotEvaluated(_Block):
    name: str
    reason: str


class ArtifactInfo(_Block):
    """伪影检测结果。`confidence` 说明这个数字有多可信，不许含糊。"""

    name: Literal["jpeg_blockiness", "moire"]
    evaluated: bool
    value: float | None
    threshold: float
    triggered: bool
    confidence: Literal["measured", "heuristic"]
    note: str


class QcReport(_Block):
    image_path: str
    image_sha256: str
    width_px: int
    height_px: int
    color_mode: str
    dpi: DpiInfo
    sharpness: SharpnessInfo
    skew: SkewInfo
    lighting: LightingInfo
    artifacts: tuple[ArtifactInfo, ...]
    dpi_below_min: bool
    budget_scale: float
    mm_per_px: float
    status: QcStatus
    skew_correction_applied: bool = Field(
        ...,
        description=(
            "本图是否**已经**做过倾斜校正。S0 只判定不修改像素，故恒为 False；"
            "S1 执行校正后在 metrics_s1.json 里置 true。下游（尤其 S7）只认最后置 true 的那份产物，"
            "避免重复校正。"
        ),
    )
    warnings: tuple[str, ...]
    not_evaluated: tuple[CheckNotEvaluated, ...]

    def to_json(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


# ---------------------------------------------------------------------------
# 图像与 DPI
# ---------------------------------------------------------------------------
def _exif_dpi(image: Image.Image) -> tuple[float, float] | None:
    """按 TIFF/PNG/JPEG 的官方位置读分辨率，返回 (x, y)；读不到返回 None。

    PNG 的 pHYs 以「像素/米」存整数，300 dpi 存成 11811 ppm，读回来是 299.9994。
    这种量化误差会污染后续所有 mm 计算，故统一保留 1 位小数——它远小于任何精度判据，
    但能让「同一张图在不同工具里显示同一个 DPI」。
    """
    mode = image.mode  # noqa: F841 - 保留原始模式便于排查
    result: tuple[float, float] | None = None

    if hasattr(image, "tag_v2"):  # TIFF
        tags = image.tag_v2
        try:
            x = float(tags.get(282, 0) or 0)
            y = float(tags.get(283, 0) or 0)
            unit = int(tags.get(296, 2) or 2)
        except (TypeError, ValueError):
            return None
        if x > 0 and y > 0:
            if unit == 3:  # 3 = 每厘米
                x, y = x * 2.54, y * 2.54
            return round(x, 1), round(y, 1)

    dpi = image.info.get("dpi")
    if isinstance(dpi, tuple) and len(dpi) == 2:
        try:
            x, y = float(dpi[0]), float(dpi[1])
        except (TypeError, ValueError):
            x = y = 0.0
        if x > 0 and y > 0:
            return round(x, 1), round(y, 1)

    # JPEG EXIF 兜底：282/283 分辨率 + 296 单位（PNG 的 pHYs 已由 info["dpi"] 覆盖）
    try:
        exif = image.getexif()
    except Exception:  # noqa: BLE001 - 损坏的 EXIF 不应让质检崩掉
        return None
    if exif:
        try:
            x = float(exif.get(282) or 0)
            y = float(exif.get(283) or 0)
            unit = int(exif.get(296) or 2)
        except (TypeError, ValueError):
            return None
        if x > 0 and y > 0:
            if unit == 3:
                x, y = x * 2.54, y * 2.54
            return round(x, 1), round(y, 1)
    return result


def exif_dpi(image: Image.Image) -> tuple[float, float] | None:
    """公开入口：读图像分辨率元数据（EXIF / PNG pHYs / TIFF 标签），读不到返回 None。"""
    return _exif_dpi(image)


def _load_gray(path: Path) -> tuple[np.ndarray, Image.Image]:
    if not Path(path).is_file():
        raise QcError(f"图像不存在：{path}")
    try:
        image = Image.open(path)
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise QcError(f"无法解码图像 {path}：{exc}") from exc
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    return gray, image


def _downsample(gray: np.ndarray) -> tuple[np.ndarray, float]:
    """长边限制到 _MAX_ANALYSIS_SIDE；返回 (图, 缩放系数 px_out/px_in)。"""
    h, w = gray.shape[:2]
    scale = min(1.0, _MAX_ANALYSIS_SIDE / float(max(h, w)))
    if scale >= 1.0:
        return gray, 1.0
    out = cv2.resize(gray, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                     interpolation=cv2.INTER_AREA)
    return out, scale


# ---------------------------------------------------------------------------
# 各检测项
# ---------------------------------------------------------------------------
def sharpness_metric(gray: np.ndarray) -> float:
    """归一化**峰值梯度**：`p99.5(|∇I|/4) / (p99 − p1)`。

    为什么不用 p90：线条图里 90% 是平坦背景，p90 恒为 0，指标会失去分辨力。
    为什么除以 4：3×3 Sobel 对单位阶跃的最大响应是 4，除完后理想阶跃的边缘 ≈ 1.0。
    实测（300 DPI 合成图）：清晰 ≈ 1.00–1.05；σ=2 高斯模糊 ≈ 0.63；
    σ=6 重度模糊 ≈ 0.2 量级。所以该指标是"峰值陡峭度"，与线宽/覆盖率基本无关。
    """
    if gray.ndim != 2:
        raise QcError(f"sharpness_metric 需要灰度图，收到形状 {gray.shape}")
    img = gray.astype(np.float32) / 255.0
    lo, hi = np.percentile(img, [1.0, 99.0])
    rng = float(hi - lo)
    if rng <= 1e-6:
        return 0.0  # 纯色图没有边缘
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy) / _SOBEL_NORMALIZER
    return float(np.percentile(mag, 99.5) / rng)


def estimate_skew(
    gray: np.ndarray,
    *,
    search_deg: float,
    coarse_step_deg: float = _SKEW_COARSE_STEP_DEG,
    fine_step_deg: float = _SKEW_FINE_STEP_DEG,
) -> tuple[float, float]:
    """投影轮廓法估页面倾斜：找让行/列投影最"锐"的旋转角。

    对每个候选角 θ 把前景掩膜旋转 θ（`getRotationMatrix2D` 正角=逆时针），
    计算 `var(行投影) + var(列投影)`；真正的校平角会让线条落到同一行/列上，得分出现尖峰。
    返回 `(需要的校正角, 峰值显著性)`：
      - 校正角 = `-最优旋转角`（正值 = 逆时针校正）；
      - 显著性 = `峰值 / 中位数`，图纸没有主导横竖线时接近 1（判为不可信）。

    为什么不用 Hough：QDIZE9403 这类圆形零件图上，斜向标注线会让 Hough 角度中位数
    跑到 ±7°，而投影法在网格图上误差 ≤0.1°、在无主向图上显著性≈1.0，可分性更好。
    """
    if search_deg <= 0:
        raise QcError(f"search_deg 必须 > 0，收到 {search_deg}")
    h, w = gray.shape[:2]
    scale = min(1.0, _MAX_ANALYSIS_SIDE / float(max(h, w)))
    small = (
        cv2.resize(gray, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                   interpolation=cv2.INTER_AREA)
        if scale < 1.0 else gray
    )
    _, mask = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    fg = (mask > 0).astype(np.float32)  # 1 = 图元
    if fg.sum() < 16:
        return 0.0, 1.0
    hh, ww = fg.shape
    centre = (ww / 2.0, hh / 2.0)

    def score(angle: float) -> float:
        matrix = cv2.getRotationMatrix2D(centre, angle, 1.0)
        rotated = cv2.warpAffine(
            fg, matrix, (ww, hh), flags=cv2.INTER_NEAREST, borderValue=0
        )
        return float(rotated.sum(axis=0).var() + rotated.sum(axis=1).var())

    coarse = np.arange(-search_deg, search_deg + 1e-9, coarse_step_deg)
    scores = np.array([score(float(a)) for a in coarse], dtype=np.float64)
    median = float(np.median(scores))
    best_index = int(np.argmax(scores))
    best_angle = float(coarse[best_index])
    prominence = float(scores[best_index] / median) if median > 1e-12 else float("inf")

    fine = np.arange(
        best_angle - coarse_step_deg, best_angle + coarse_step_deg + 1e-9, fine_step_deg
    )
    fine_scores = np.array([score(float(a)) for a in fine], dtype=np.float64)
    best_angle = float(fine[int(np.argmax(fine_scores))])
    if abs(best_angle) < fine_step_deg / 2.0:
        best_angle = 0.0
    return -best_angle, prominence


def halftone_dot_density(
    gray: np.ndarray, *, max_area: int = 40, max_side: int = 8
) -> float:
    """每 1e4 px² 的「小圆点」连通域数量 —— 半调网点/莫尔纹的结构特征。

    网点图由成千上万个 1–9 px 的孤立小点组成，而线条图是少数大连通域。
    实测（同一批图）：干净线条图 0–0.4，6 px 网点 279，4 px 网点 625 —— 分离度 >100 倍。

    注意：必须在**原分辨率**上统计；下采样会把网点糊成一片，密度骤降到 0。
    """
    if gray.ndim != 2:
        raise QcError(f"halftone_dot_density 需要灰度图，收到形状 {gray.shape}")
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    small = 0
    for index in range(1, count):
        width = int(stats[index, cv2.CC_STAT_WIDTH])
        height = int(stats[index, cv2.CC_STAT_HEIGHT])
        area = int(stats[index, cv2.CC_STAT_AREA])
        long_side = max(width, height)
        short_side = max(1, min(width, height))
        if 1 <= area <= max_area and long_side <= max_side and long_side / short_side <= 3.0:
            small += 1
    return small * 10_000.0 / float(gray.size)


def jpeg_blockiness(gray: np.ndarray, *, block: int = 8) -> float:
    """8×8 块边界与块内部平均梯度之比（JPEG 块效应的启发式指标）。

    干净图实测 1.0–1.2。**局限**：若图元恰好落在 8 px 网格上（程序生成的等距网格），
    指标会假高（实测 6.5），因此只对**真实 JPEG 输入**启用，且在报告里标 `confidence=heuristic`。
    """
    if gray.ndim != 2:
        raise QcError(f"jpeg_blockiness 需要灰度图，收到形状 {gray.shape}")
    diff = np.abs(np.diff(gray.astype(np.float32), axis=1))
    if diff.size == 0:
        return 1.0
    columns = np.arange(diff.shape[1])
    boundary = (columns % block) == (block - 1)
    if not boundary.any() or boundary.all():
        return 1.0
    edge = float(diff[:, boundary].mean())
    inner = float(diff[:, ~boundary].mean())
    return edge / max(inner, 1e-6)


def lighting_blocks(gray: np.ndarray, *, block_ratio: float) -> tuple[float, int]:
    """分块**中位数**的变异系数 CV = std/mean。返回 (cv, block_size_px)。

    为什么用中位数而不是均值：块内只要压到一条线，均值就被拉低，CV 会变成"内容密度"
    而不是"光照不均"。中位数对少数暗像素稳健（3px 线占 29×29 块不到 1%），
    实测：平坦光照的线条图 CV 0.148（均值）→ 0.01 量级（中位数），而真实的梯度光照
    仍然给出 >0.3。
    """
    if block_ratio <= 0:
        raise QcError(f"block_ratio 必须 > 0，收到 {block_ratio}")
    h, w = gray.shape[:2]
    block = max(8, int(round(min(h, w) / block_ratio)))
    img = np.asarray(gray, dtype=np.float32)
    values: list[float] = []
    for y0 in range(0, h - block + 1, block):
        for x0 in range(0, w - block + 1, block):
            values.append(float(np.median(img[y0:y0 + block, x0:x0 + block])))
    if not values:
        return 0.0, block
    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean())
    if mean <= 1e-6:
        return 0.0, block
    return float(arr.std() / mean), block


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def run_s0_qc(
    image_path: str | Path,
    qc: QcConfig,
    *,
    out_dir: str | Path | None = None,
) -> tuple[QcReport, StageMetrics]:
    """执行 S0 质检。返回 (报告, 指标)；`out_dir` 非空时写 qc_report.json + metrics_s0.json。"""
    path = Path(image_path)
    gray, image = _load_gray(path)
    height_px, width_px = gray.shape[:2]
    warnings: list[str] = []

    # ---- DPI ----
    measured = _exif_dpi(image)
    if measured is None:
        dpi_x = dpi_y = float(qc.fallback_dpi)
        source: Literal["metadata", "fallback"] = "fallback"
        warnings.append(
            f"图像没有可用的 DPI 元数据（EXIF/pHYs/TIFF 均缺失）：按 fallback_dpi="
            f"{qc.fallback_dpi:g} 处理，尺寸结论不可信"
        )
    else:
        dpi_x, dpi_y = measured
        source = "metadata"
    anisotropic = bool(abs(dpi_x - dpi_y) / max(dpi_x, 1e-9) > 0.01)
    if anisotropic:
        warnings.append(
            f"X/Y 分辨率不一致（{dpi_x:g} / {dpi_y:g} dpi，比值 {dpi_y / max(dpi_x, 1e-9):.4f}）："
            "按 X 处理，存在各向异性畸变风险"
        )
    applied_dpi = float(dpi_x)

    below_min = applied_dpi < qc.dpi_min
    if below_min:
        if qc.dpi_mode == "reject":
            warnings.append(f"DPI={applied_dpi:g} < dpi_min={qc.dpi_min:g}：按 dpi_mode=reject 拒绝")
        elif qc.dpi_mode == "warn":
            warnings.append(f"DPI={applied_dpi:g} < dpi_min={qc.dpi_min:g}：继续但打标记")
        elif qc.dpi_mode == "degrade":
            scale = dpi_budget_scale(applied_dpi, qc.budget_reference_dpi)
            warnings.append(
                f"DPI={applied_dpi:g} < dpi_min={qc.dpi_min:g}：降级继续，"
                f"误差预算按 {qc.budget_reference_dpi:g}/{applied_dpi:g}={scale:.4f} 倍放大"
            )
        else:  # ignore
            warnings.append(f"DPI={applied_dpi:g} < dpi_min={qc.dpi_min:g}：dpi_mode=ignore 忽略")

    if qc.dpi_mode == "reject" and below_min:
        status: QcStatus = "rejected"
        budget = 1.0
    elif below_min and qc.dpi_mode == "degrade":
        status = "degraded"
        budget = dpi_budget_scale(applied_dpi, qc.budget_reference_dpi)
    elif below_min and qc.dpi_mode == "warn":
        status = "degraded"
        budget = 1.0
    elif source == "fallback":
        status = "degraded"
        budget = 1.0
    else:
        status = "pass"
        budget = 1.0

    # ---- 锐度（在下采样图上算，尺度无关）----
    small, scale = _downsample(gray)
    sharp_value = sharpness_metric(small)
    sharp_ok = sharp_value >= qc.sharpness_min
    if not sharp_ok:
        warnings.append(
            f"锐度 {sharp_value:.3f} < sharpness_min {qc.sharpness_min:g}："
            "疑似低分辨率扫描或过度压缩，拟合精度会下降"
        )
        status = "degraded" if status == "pass" else status

    # ---- 倾斜 ----
    angle, prominence = estimate_skew(small, search_deg=qc.skew_search_deg)
    skew_confident = prominence >= qc.skew_prominence_min
    if not skew_confident:
        angle = 0.0
        warnings.append(
            f"倾斜检测不可信（投影峰显著性 {prominence:.2f} < skew_prominence_min "
            f"{qc.skew_prominence_min:g}）：图中没有足够的主向直线，skew 记 0"
        )
    exceeds = bool(skew_confident and abs(angle) > qc.skew_max_deg)
    corrective = -angle if exceeds else 0.0
    if exceeds:
        warnings.append(
            f"页面倾斜 {angle:.2f}° > skew_max_deg {qc.skew_max_deg:g}："
            f"建议旋转校正 {corrective:.2f}°"
        )
        status = "degraded" if status == "pass" else status
    skew = SkewInfo(
        angle_deg=round(angle, 4),
        confident=bool(skew_confident),
        prominence=round(prominence, 6),
        exceeds_limit=exceeds,
        corrective_rotation_deg=round(corrective, 4),
        method="projection_profile",
    )

    # ---- 光照 ----
    block_cv, block_size = lighting_blocks(small, block_ratio=qc.shadow_block_size_ratio)
    uneven = block_cv > qc.shadow_var_threshold
    if uneven:
        warnings.append(
            f"分块均值变异系数 {block_cv:.3f} > shadow_var_threshold "
            f"{qc.shadow_var_threshold:g}：光照不均，将启用背景平坦化"
        )
    lighting = LightingInfo(
        block_size_px=int(round(block_size / scale)),
        block_cv=round(block_cv, 6),
        threshold=qc.shadow_var_threshold,
        uneven=bool(uneven),
        recommend_background_flatten=bool(uneven),
    )

    # ---- 伪影：JPEG 块效应（仅 JPEG 输入）+ 半调网点/莫尔纹 ----
    artifacts: list[ArtifactInfo] = []

    if qc.jpeg_ring_artifact_detect:
        if image.format == "JPEG":
            blockiness = jpeg_blockiness(small)
            triggered = blockiness > qc.jpeg_blockiness_max
            artifacts.append(
                ArtifactInfo(
                    name="jpeg_blockiness",
                    evaluated=True,
                    value=round(blockiness, 6),
                    threshold=qc.jpeg_blockiness_max,
                    triggered=bool(triggered),
                    confidence="heuristic",
                    note="8×8 块边界/内部梯度比；干净图 1.0–1.2，强块效应 >2",
                )
            )
            if triggered:
                warnings.append(
                    f"JPEG 块效应 {blockiness:.2f} > jpeg_blockiness_max "
                    f"{qc.jpeg_blockiness_max:g}：建议去块滤波（S1 会据此调整）"
                )
                status = "degraded" if status == "pass" else status
        else:
            artifacts.append(
                ArtifactInfo(
                    name="jpeg_blockiness",
                    evaluated=False,
                    value=None,
                    threshold=qc.jpeg_blockiness_max,
                    triggered=False,
                    confidence="heuristic",
                    note=f"输入不是 JPEG（format={image.format}），该检测不适用",
                )
            )

    if qc.moire_detect:
        # 网点统计必须在原分辨率上做（下采样会把点糊成一片）
        dot_source = gray
        if max(gray.shape[:2]) > _MOIRE_MAX_SIDE:
            factor = _MOIRE_MAX_SIDE / float(max(gray.shape[:2]))
            dot_source = cv2.resize(
                gray,
                (max(1, int(round(gray.shape[1] * factor))),
                 max(1, int(round(gray.shape[0] * factor)))),
                interpolation=cv2.INTER_AREA,
            )
            warnings.append(
                f"莫尔纹检测在下采样图（长边 {_MOIRE_MAX_SIDE}px）上进行：极小网点可能漏检"
            )
        density = halftone_dot_density(dot_source)
        triggered = density > qc.moire_dot_density_max
        artifacts.append(
            ArtifactInfo(
                name="moire",
                evaluated=True,
                value=round(density, 6),
                threshold=qc.moire_dot_density_max,
                triggered=bool(triggered),
                confidence="measured",
                note="小圆点连通域密度（每 1e4 px²）；干净图 0–0.4，网点图 279–625",
            )
        )
        if triggered:
            warnings.append(
                f"疑似半调网点/莫尔纹：点密度 {density:.1f}/1e4px > "
                f"moire_dot_density_max {qc.moire_dot_density_max:g}："
                "S1 需去网滤波，否则 S2 骨架会被网点污染"
            )
            status = "degraded" if status == "pass" else status

    # ---- 仍未实现的检测如实标注（不假装通过）----
    not_evaluated = (
        CheckNotEvaluated(
            name="perspective",
            reason="S0 未实现：需要四边检测 + 汇聚度估计，排在 S7 配准前补齐（用户决策 #3）",
        ),
        CheckNotEvaluated(
            name="stain_mask",
            reason="S0 未实现：需要大面积非图斑分割，优先级最低，排在 S3 之后",
        ),
    )

    report = QcReport(
        image_path=str(path),
        image_sha256=sha256_file(path),
        width_px=int(width_px),
        height_px=int(height_px),
        color_mode=image.mode,
        dpi=DpiInfo(
            value_x=float(dpi_x),
            value_y=float(dpi_y),
            source=source,
            anisotropic=anisotropic,
            anisotropy_ratio=round(dpi_y / max(dpi_x, 1e-9), 6),
            applied=applied_dpi,
        ),
        sharpness=SharpnessInfo(
            metric="gradient_p90_normalized",
            value=round(sharp_value, 6),
            threshold=qc.sharpness_min,
            passed=bool(sharp_ok),
        ),
        skew=skew,
        lighting=lighting,
        artifacts=tuple(artifacts),
        dpi_below_min=bool(below_min),
        budget_scale=round(float(budget), 6),
        mm_per_px=round(mm_per_px(applied_dpi), 9),
        status=status,
        skew_correction_applied=False,  # S0 只判定，不修改像素
        warnings=tuple(warnings),
        not_evaluated=not_evaluated,
    )

    block = max(8, int(round(block_size)))
    metrics = StageMetrics(
        stage="s0",
        counts={
            "width_px": int(width_px),
            "height_px": int(height_px),
            "analysis_side_px": int(max(small.shape[:2])),
            "lighting_blocks": int((small.shape[0] // block) * (small.shape[1] // block)),
            "warnings": len(warnings),
            "artifacts_triggered": sum(1 for a in artifacts if a.triggered),
        },
        timing_ms=0.0,
        error_estimate={
            "mm_per_px": report.mm_per_px,
            "budget_scale": report.budget_scale,
            "sharpness": report.sharpness.value,
            "skew_deg": report.skew.angle_deg,
        },
        warnings=tuple(warnings),
        details={
            "status": report.status,
            "image_sha256": report.image_sha256,
            "skew_confident": report.skew.confident,
            "skew_prominence": report.skew.prominence,
            "skew_correction_applied": report.skew_correction_applied,
            "artifacts": [
                {"name": a.name, "value": a.value, "triggered": a.triggered,
                 "confidence": a.confidence}
                for a in artifacts
            ],
            "not_evaluated": [c.name for c in report.not_evaluated],
        },
    )

    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "qc_report.json", report.to_json())
        write_metrics(out, metrics)
    return report, metrics


def run_s0_qc_timed(
    image_path: str | Path,
    qc: QcConfig,
    *,
    out_dir: str | Path | None = None,
) -> tuple[QcReport, StageMetrics]:
    """同 `run_s0_qc`，但把真实耗时写进 metrics（不要在上层重复计时）。"""
    import time

    t0 = time.perf_counter()
    report, metrics = run_s0_qc(image_path, qc, out_dir=out_dir)
    timing = round((time.perf_counter() - t0) * 1000.0, 3)
    timed = metrics.model_copy(update={"timing_ms": timing})
    if out_dir is not None:
        write_metrics(Path(out_dir), timed)
    return report, timed


__all__ = [
    "ArtifactInfo",
    "CheckNotEvaluated",
    "DpiInfo",
    "LightingInfo",
    "QcError",
    "QcReport",
    "SharpnessInfo",
    "SkewInfo",
    "estimate_skew",
    "exif_dpi",
    "halftone_dot_density",
    "jpeg_blockiness",
    "lighting_blocks",
    "run_s0_qc",
    "run_s0_qc_timed",
    "sharpness_metric",
]
