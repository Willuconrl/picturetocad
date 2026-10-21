# src/s1_preprocess.py
"""S1 预处理：**不动边缘位置**是唯一目标（SPEC §4 S1）。

流水线（顺序即契约）：

1. 灰度化：彩色图按 **ITU-R BT.709** 加权（`0.2126R + 0.7152G + 0.0722B`），
   禁止等权平均——等权会让深蓝/深红的细线偏亮而丢线。
2. 黑边裁切：`clip_border_px>0` 按像素裁，`=0` 自动检测四周明显偏暗的行/列。
3. 反白检测与反转：白底黑线 → 不反转；黑底白线（蓝图/负片）→ 反转成正片。
4. 背景平坦化：morph_open / polynomial / gaussian_ilr 三种，按 `background_flatten.enable`
   （`auto` 时用 S0 的光照判据决定）。
5. 去噪：nlmeans / bilateral / none。**禁止大核高斯**（会把边缘摊开、位移）。
6. 超采样：线条太细（`< upsample_if_thin.min_linewidth_px`）时按 `factor` 放大再进 S2，
   否则骨架必然偏移。

精度约束
--------
- 内部一律 float32、灰度、[0,1]；只在写出 PNG 时转为 uint8。
- 去噪与平坦化后，**边缘位置位移必须 ≤0.3 px**（config 注释里的硬指标，由
  `tests/test_s1_preprocess.py` 用合成斜面实测）。
- 超采样后 `mm_per_px` 必须同步除以 `factor`，否则所有几何量差一个倍数。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from PIL import Image
from skimage.morphology import skeletonize

from src.artifacts import StageMetrics, sha256_file, write_json, write_metrics
from src.config import R2VConfig
from src.s0_qc import exif_dpi, lighting_blocks

#: Otsu 判前景：灰度低于该值视为图元
_EPS = 1e-6

#: 自适应阈值的常数偏移（灰度级）。实测 5–30 对线宽估计无影响，取 10 居中
_ADAPTIVE_C = 10


class PreprocessError(ValueError):
    """预处理失败（图像不可读、参数非法等）。"""


@dataclass(frozen=True)
class PreprocessResult:
    """预处理产物。`image` 是 float32、[0,1]、灰度（已反转成正片）。"""

    image: np.ndarray
    width_px: int
    height_px: int
    source_path: str
    source_sha256: str
    dpi: float
    mm_per_px: float
    upsample_factor: int
    inverted: bool
    background_flattened: bool
    background_method: str | None
    denoise_method: str
    estimated_linewidth_px: float
    rotation_deg: float
    skew_correction_applied: bool
    crop_px: tuple[int, int, int, int]
    warnings: tuple[str, ...]

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height_px, self.width_px)


# ---------------------------------------------------------------------------
# 基础算子
# ---------------------------------------------------------------------------
def bt709_gray(rgb: np.ndarray) -> np.ndarray:
    """RGB(A) → 灰度，BT.709 权重。输入 uint8/float，输出 float32 [0,1]。"""
    array = np.asarray(rgb)
    if array.ndim == 2:
        gray = array.astype(np.float32)
        scale = 255.0 if gray.max() > 1.0 else 1.0
        return np.clip(gray / scale, 0.0, 1.0)
    if array.ndim != 3 or array.shape[2] not in (3, 4):
        raise PreprocessError(f"bt709_gray 需要 2D 或 RGB/RGBA，收到形状 {array.shape}")
    channels = array[:, :, :3].astype(np.float32)
    scale = 255.0 if channels.max() > 1.0 else 1.0
    channels /= scale
    weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    return np.clip(channels @ weights, 0.0, 1.0)


def _to_u8(gray: np.ndarray) -> np.ndarray:
    return np.clip(gray * 255.0 + 0.5, 0, 255).astype(np.uint8)


def _to_float(gray: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(gray, dtype=np.float32) / 255.0, 0.0, 1.0)


def estimate_linewidth_px(gray: np.ndarray) -> float:
    """估计**墨迹宽度**（像素）：`前景面积 / 骨架长度`。

    二值化用**自适应阈值**而不是 Otsu：光照不均时 Otsu 会把暗侧背景整片判成前景，
    实测把 5px 线的宽度估成 24px（并进而让背景核彻底跑偏）。自适应阈值只看局部对比，
    实测在「干净图」与「梯度光照图」上给出同一个值（4.91 vs 4.91）。

    为什么不用 `2×距离变换最大值`：那条式子对粗线系统性偏高，且交叉处会被放大。
    实测与已知栅格一致：1 行线 → 1.00，5 行线 → 5.09，梯度光照网格 → 4.91。
    """
    u8 = _to_u8(gray) if float(gray.max()) <= 1.0 else np.asarray(gray, dtype=np.uint8)
    height, width = u8.shape[:2]
    block = max(11, int(min(height, width) // 8)) | 1  # 自适应窗口，奇数
    binary = cv2.adaptiveThreshold(
        u8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, _ADAPTIVE_C
    )
    foreground = binary > 0
    area = int(foreground.sum())
    if area == 0:
        return 0.0
    skeleton = skeletonize(foreground)
    skeleton_length = int(skeleton.sum())
    if skeleton_length >= 16:
        return float(area) / float(skeleton_length)
    # 骨架太短（小碎块）时退回距离变换：2×最大内切半径
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    return 2.0 * float(distance.max())


def crop_dark_border(gray: np.ndarray, *, clip_px: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """裁掉扫描仪黑边。`clip_px>0` 直接裁；`=0` 自动检测明显偏暗的边缘行/列。

    返回 (裁剪后图像, (top, bottom, left, right))。自动检测上限为每边短边的 10%，
    避免把正常图框当成黑边裁掉。
    """
    height, width = gray.shape[:2]
    if clip_px > 0:
        n = min(clip_px, height // 4, width // 4)
        return gray[n:height - n, n:width - n], (n, n, n, n)

    limit_h = max(1, int(height * 0.10))
    limit_w = max(1, int(width * 0.10))
    median = float(np.median(gray))
    dark_level = 0.5 * median
    if dark_level <= _EPS:
        return gray, (0, 0, 0, 0)

    top = 0
    while top < limit_h and float(gray[top].mean()) < dark_level:
        top += 1
    bottom = 0
    while bottom < limit_h and float(gray[height - 1 - bottom].mean()) < dark_level:
        bottom += 1
    left = 0
    while left < limit_w and float(gray[:, left].mean()) < dark_level:
        left += 1
    right = 0
    while right < limit_w and float(gray[:, width - 1 - right].mean()) < dark_level:
        right += 1
    if top == bottom == left == right == 0:
        return gray, (0, 0, 0, 0)
    cropped = gray[top:height - bottom, left:width - right]
    if cropped.size < 16:
        raise PreprocessError("黑边裁切后图像过小：请检查 clip_border_px 或原图")
    return cropped, (top, bottom, left, right)


def detect_inverted(gray: np.ndarray, *, max_foreground_ratio: float) -> bool:
    """反白检测：亮像素占比 < `max_foreground_ratio` 且整体偏暗 → 判为负片。

    注意：`config.yaml` 里该字段的注释写的是「前景占比 <0.15 且均值>128 则反转」，
    但「均值>128」与「图像是负片」互相矛盾（负片就是整体偏暗）。这里按物理意义实现：
    **亮像素（纸面）占比过小** 才是负片；均值条件作为二次确认。已在 M2 报告中提请确认。
    """
    u8 = _to_u8(gray)
    threshold, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright_ratio = float((u8 > threshold).mean())
    mean = float(gray.mean())
    return bool(bright_ratio < max_foreground_ratio and mean < 0.5)


def flatten_background(
    gray: np.ndarray,
    *,
    method: Literal["morph_open", "polynomial", "gaussian_ilr"],
    kernel_px: int,
    polynomial_degree: int,
) -> tuple[np.ndarray, np.ndarray]:
    """估计并除以背景。返回 (平坦化图像, 背景图)。

    三种方法都只做**除法归一**，不做减法——除法保留亮度比例，减法会把细线一起压掉。
    """
    height, width = gray.shape[:2]
    kernel = max(3, int(kernel_px))
    if kernel % 2 == 0:
        kernel += 1

    if method == "morph_open":
        shape = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
        background = cv2.morphologyEx(gray, cv2.MORPH_OPEN, shape)
    elif method == "gaussian_ilr":
        sigma = max(1.0, kernel / 6.0)
        background = cv2.GaussianBlur(gray, (0, 0), sigma)
    elif method == "polynomial":
        grid = 16
        small = cv2.resize(gray, (max(2, width // grid), max(2, height // grid)),
                           interpolation=cv2.INTER_AREA)
        yy, xx = np.mgrid[0:small.shape[0], 0:small.shape[1]]
        xs = (xx / max(1, small.shape[1] - 1)).ravel()
        ys = (yy / max(1, small.shape[0] - 1)).ravel()
        zs = small.ravel().astype(np.float64)
        terms = [(i, j) for i in range(polynomial_degree + 1)
                 for j in range(polynomial_degree + 1 - i)]
        design = np.column_stack([(xs ** i) * (ys ** j) for i, j in terms])
        coeffs, *_ = np.linalg.lstsq(design, zs, rcond=None)
        yy2, xx2 = np.mgrid[0:height, 0:width]
        xs2 = (xx2 / max(1, width - 1)).ravel()
        ys2 = (yy2 / max(1, height - 1)).ravel()
        design2 = np.column_stack([(xs2 ** i) * (ys2 ** j) for i, j in terms])
        background = (design2 @ coeffs).reshape(height, width).astype(np.float32)
    else:  # pragma: no cover - Literal 已限死
        raise PreprocessError(f"未知的背景平坦化方法：{method}")

    background = np.clip(background, _EPS, None)
    flat = gray / background
    # 归一到 [0,1]：用亮背景（≈1.0）为参照，保留原对比度比例
    flat = flat / max(float(np.median(flat)), _EPS)
    return np.clip(flat, 0.0, 1.0).astype(np.float32), background.astype(np.float32)


def denoise(
    gray: np.ndarray,
    *,
    method: Literal["nlmeans", "bilateral", "none"],
    h_normalized: float,
    patch_px: int,
    search_px: int,
) -> np.ndarray:
    """去噪。参数由 config 派生：`h = nlmeans_h × 255`（归一化强度 → 灰度级）。"""
    if method == "none":
        return gray
    u8 = _to_u8(gray)
    h = max(1.0, float(h_normalized) * 255.0)
    patch = max(3, int(patch_px) | 1)  # 必须奇数
    if method == "nlmeans":
        out = cv2.fastNlMeansDenoising(
            u8, None, h=h, templateWindowSize=patch, searchWindowSize=max(patch, int(search_px))
        )
        return _to_float(out)
    if method == "bilateral":
        # config 没有单独的 bilateral 字段：d 复用 patch，sigmaColor 用同一 h，sigmaSpace 用 patch
        out = cv2.bilateralFilter(u8, d=patch, sigmaColor=h, sigmaSpace=float(patch))
        return _to_float(out)
    raise PreprocessError(f"未知的去噪方法：{method}")


def upsample(gray: np.ndarray, *, factor: int, kernel: Literal["lanczos"]) -> np.ndarray:
    """超采样。只允许 2/3 倍（config 约束），kernal 目前只实现 lanczos。"""
    if factor not in (2, 3):
        raise PreprocessError(f"超采样倍数只允许 2 或 3，收到 {factor}")
    if kernel != "lanczos":  # pragma: no cover - Literal 已限死
        raise PreprocessError(f"未知的超采样核：{kernel}")
    height, width = gray.shape[:2]
    return cv2.resize(
        gray, (width * factor, height * factor), interpolation=cv2.INTER_LANCZOS4
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def load_image(path: str | Path) -> tuple[np.ndarray, str, str]:
    """读图 → (灰度 float32 [0,1], Pillow 模式, format)。"""
    path = Path(path)
    if not path.is_file():
        raise PreprocessError(f"图像不存在：{path}")
    try:
        with Image.open(path) as image:
            image.load()
            mode = image.mode
            fmt = image.format or ""
            array = np.asarray(image)
    except Exception as exc:  # noqa: BLE001 - 统一转成 PreprocessError 并保留原因
        raise PreprocessError(f"无法解码图像 {path}：{exc}") from exc
    return bt709_gray(array), mode, fmt


def run_s1_preprocess(
    image_path: str | Path,
    cfg: R2VConfig,
    *,
    out_dir: str | Path | None = None,
    rotation_deg: float = 0.0,
) -> tuple[PreprocessResult, StageMetrics]:
    """执行 S1。`rotation_deg` 非 0 时执行倾斜校正，并在产物里标记已校正。"""
    t0 = time.perf_counter()
    path = Path(image_path)
    prep = cfg.preprocess
    qc = cfg.qc
    warnings: list[str] = []

    gray, mode, fmt = load_image(path)
    source_sha256 = sha256_file(path)

    # ---- DPI ----
    with Image.open(path) as image:
        measured = exif_dpi(image)
    if measured is None:
        dpi = float(qc.fallback_dpi)
        warnings.append(f"没有可用 DPI 元数据：按 fallback_dpi={qc.fallback_dpi:g} 计算 mm/px")
    else:
        dpi = float(measured[0])
        if abs(measured[0] - measured[1]) / max(measured[0], 1e-9) > 0.01:
            warnings.append(f"X/Y DPI 不一致（{measured[0]:g}/{measured[1]:g}）：按 X 计算")

    # ---- 黑边裁切 ----
    gray, crop = crop_dark_border(gray, clip_px=int(prep.clip_border_px))
    if crop != (0, 0, 0, 0):
        warnings.append(f"已裁切黑边（上/下/左/右 = {crop}）")

    # ---- 反白 ----
    inverted = detect_inverted(gray, max_foreground_ratio=prep.invert_foreground_ratio)
    if inverted:
        gray = 1.0 - gray
        warnings.append("检测到反白（负片）：已反转成正片")
    elif not prep.invert_auto:
        warnings.append("invert_auto=false：跳过反白检测（若原图是负片，后续必然失败）")

    # ---- 线宽估计（供背景核尺寸与超采样判据）----
    linewidth_px = estimate_linewidth_px(gray)
    if linewidth_px <= 0:
        warnings.append("线宽估计为 0（图中没有可辨识的图元）：后续阶段会拒绝处理")

    # ---- 背景平坦化 ----
    enable = prep.background_flatten.enable
    if enable == "auto":
        block_cv, _ = lighting_blocks(gray, block_ratio=qc.shadow_block_size_ratio)
        do_flatten = block_cv > qc.shadow_var_threshold
        reason = f"auto：分块 CV={block_cv:.3f} {'>' if do_flatten else '<='} {qc.shadow_var_threshold:g}"
    else:
        do_flatten = bool(enable)
        reason = f"配置强制 {'开启' if do_flatten else '关闭'}"

    background_method: str | None = None
    if do_flatten:
        base = max(linewidth_px, 1.0)
        wanted = int(round(base * prep.background_flatten.kernel_scale_linewidths))
        # 安全上限：结构元一旦大于线间距，开运算会把线条本身当成背景吃掉（实测核 98px、
        # 线间距 100px 时背景均值从 0.5 掉到 0.1，平坦化后 CV 反而升到 0.63）。
        cap = max(15, min(gray.shape[:2]) // 6)
        kernel = min(wanted, cap)
        if kernel < wanted:
            warnings.append(
                f"背景核由 {wanted}px 收到 {kernel}px（安全上限 min(h,w)/6）："
                "核大于线间距时开运算会吃掉线条"
            )
        gray, _background = flatten_background(
            gray,
            method=prep.background_flatten.method,
            kernel_px=kernel,
            polynomial_degree=prep.background_flatten.polynomial_degree,
        )
        background_method = prep.background_flatten.method
        warnings.append(f"背景平坦化：{background_method}（核 {kernel}px，{reason}）")

    # ---- 去噪 ----
    gray = denoise(
        gray,
        method=prep.denoise.method,
        h_normalized=prep.denoise.nlmeans_h,
        patch_px=prep.denoise.nlmeans_patch_px,
        search_px=prep.denoise.nlmeans_search_px,
    )

    # ---- 倾斜校正 ----
    skew_applied = False
    if abs(rotation_deg) > 1e-9:
        height, width = gray.shape[:2]
        matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), rotation_deg, 1.0)
        gray = cv2.warpAffine(
            gray, matrix, (width, height),
            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
        )
        skew_applied = True
        warnings.append(f"已执行倾斜校正 {rotation_deg:+.3f}°（metrics_s1 记录 skew_correction_applied=true）")

    # ---- 超采样 ----
    factor = 1
    if prep.upsample_if_thin.enable and 0 < linewidth_px < prep.upsample_if_thin.min_linewidth_px:
        factor = int(prep.upsample_if_thin.factor)
        gray = upsample(gray, factor=factor, kernel=prep.upsample_if_thin.kernel)
        warnings.append(
            f"线宽 {linewidth_px:.2f}px < {prep.upsample_if_thin.min_linewidth_px:g}px："
            f"按 {factor}× 超采样（mm/px 同步除以 {factor}）"
        )

    height_px, width_px = gray.shape[:2]
    result = PreprocessResult(
        image=gray.astype(np.float32),
        width_px=int(width_px),
        height_px=int(height_px),
        source_path=str(path),
        source_sha256=source_sha256,
        dpi=dpi,
        mm_per_px=(25.4 / dpi) / factor,
        upsample_factor=factor,
        inverted=inverted,
        background_flattened=do_flatten,
        background_method=background_method,
        denoise_method=prep.denoise.method,
        estimated_linewidth_px=round(linewidth_px, 4),
        rotation_deg=float(rotation_deg),
        skew_correction_applied=skew_applied,
        crop_px=crop,
        warnings=tuple(warnings),
    )

    timing_ms = round((time.perf_counter() - t0) * 1000.0, 3)
    metrics = StageMetrics(
        stage="s1",
        counts={
            "width_px": int(width_px),
            "height_px": int(height_px),
            "upsample_factor": int(factor),
            "warnings": len(warnings),
        },
        timing_ms=timing_ms,
        error_estimate={
            "mm_per_px": round(result.mm_per_px, 9),
            "estimated_linewidth_px": result.estimated_linewidth_px,
        },
        warnings=tuple(warnings),
        details={
            "inverted": result.inverted,
            "background_flattened": result.background_flattened,
            "background_method": result.background_method,
            "denoise_method": result.denoise_method,
            "skew_correction_applied": result.skew_correction_applied,
            "rotation_deg": result.rotation_deg,
            "crop_px": list(result.crop_px),
            "source_sha256": result.source_sha256,
            "source_format": fmt,
            "source_mode": mode,
        },
    )

    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        Image.fromarray(_to_u8(gray)).save(out / "preprocessed.png")
        write_metrics(out, metrics)
        write_json(out / "preprocess_report.json", _report_json(result))
    return result, metrics


def _report_json(result: PreprocessResult) -> dict[str, Any]:
    return {
        "source_path": result.source_path,
        "source_sha256": result.source_sha256,
        "width_px": result.width_px,
        "height_px": result.height_px,
        "dpi": result.dpi,
        "mm_per_px": result.mm_per_px,
        "upsample_factor": result.upsample_factor,
        "inverted": result.inverted,
        "background_flattened": result.background_flattened,
        "background_method": result.background_method,
        "denoise_method": result.denoise_method,
        "estimated_linewidth_px": result.estimated_linewidth_px,
        "rotation_deg": result.rotation_deg,
        "skew_correction_applied": result.skew_correction_applied,
        "crop_px": list(result.crop_px),
        "warnings": list(result.warnings),
    }


__all__ = [
    "PreprocessError",
    "PreprocessResult",
    "bt709_gray",
    "crop_dark_border",
    "denoise",
    "detect_inverted",
    "estimate_linewidth_px",
    "flatten_background",
    "load_image",
    "run_s1_preprocess",
    "upsample",
]
