# src/s2_binarize.py
"""S2 二值化与亚像素边缘（精度第一来源：这里错 1px，后面全错）。

四路二值化并行 + 判据选优
------------------------
`sauvola` / `otsu` / `niblack` / `local_gaussian` 各出一张前景掩膜，用
**foreground_stability** 打分选优（SPEC §4 S2：连通域尺度稳定性 + 局部对比度一致性）：

    score = 1 / ((1 + speck_fraction) * (1 + width_cv))

- `speck_fraction`：area < 4 px 的碎点数占前景的比例（噪点越多越差）；
- `width_cv`：各连通域笔画宽度（2×距离变换中位数）的变异系数（粗细越不一致越差）。

选优是**确定性**的（同输入同分），不掺随机、不依赖字典序。

亚像素边缘
----------
在二值轮廓上沿**梯度法线**采样 |∇I|，用三点抛物线求峰位（`gradient_normal_quad`），
把轮廓点推到亚像素位置。角点邻域（相邻法线夹角 > 60°）跳过修正，避免把直角磨圆。

逐点线宽
--------
每个亚像素点取最近像素的距离变换值 ×2；全局线宽用中位数（`median_mad` 稳健统计）。

硬约束
------
- 内部一律 float32/float64，px 单位；mm 换算由调用方用 `mm_per_px` 完成。
- 不吞异常：无法解码/参数非法一律抛 `BinarizeError`；未实现的选项（如 `steger` 中心线、
  `break_touching_text`）写 WARN 并说明退化为哪种行为，不假装成功。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

from src.artifacts import StageMetrics, write_json, write_metrics
from src.config import BinarizeConfig, R2VConfig
from src.s1_preprocess import PreprocessResult, estimate_linewidth_px

_EPS = 1e-9


def _to_u8(gray: np.ndarray) -> np.ndarray:
    """float32 [0,1] 或任意灰度数组 → uint8（四路二值化统一入口）。"""
    array = np.asarray(gray, dtype=np.float32)
    if array.max() <= 1.0:
        return np.clip(array * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return np.clip(array + 0.5, 0, 255).astype(np.uint8)

#: 选优判据里的碎点面积上限（px²）。算法内部常数：只影响"什么算噪点"，不参与精度调参
_SPECK_AREA_PX = 4
#: 线宽统计时忽略的极小连通域（p90 在小样本上没有统计意义，M8-3）
_MIN_AREA_FOR_WIDTH_PX = 50

#: 漏墨惩罚的容差：AA 边缘的半亮像素允许落在掩膜外，超过这个比例才开始扣分
_MISSED_TOLERANCE = 0.20

#: 选优分数的并列窗口：同桶内视为并列，再按前景占比/碎点决定
_SCORE_TIE_EPS = 0.02

#: 亚像素采样步长（px）、搜索半宽（px）与最小边缘对比度。
#: 半宽必须 ≥2px：轮廓点在第一列覆盖率≥50% 的像素中心，距真实边缘可能已有 1px，
#: 窗口太窄会把「50% 灰像素」当成亮平台，导致电平判错（实测 1.5px → 误差 0.23px）。
_SUBPIX_STEP_PX = 0.1
_SUBPIX_SPAN_PX = 2.0
_MIN_EDGE_CONTRAST = 0.05

#: 角点保护阈值：相邻法线夹角超过此值就不做亚像素修正
_CORNER_ANGLE_DEG = 60.0


class BinarizeError(ValueError):
    """S2 输入或参数非法。"""


@dataclass(frozen=True)
class CandidateScore:
    method: str
    speck_fraction: float
    width_cv: float
    foreground_ratio: float
    score: float


@dataclass(frozen=True)
class SubpixelEdges:
    """亚像素轮廓。`contours[i]` 是 (N,2) float64 像素坐标。"""

    contours: tuple[np.ndarray, ...]
    widths: tuple[np.ndarray, ...]
    mean_abs_shift_px: float
    corner_points_skipped: int
    weak_gradient_points: int

    @property
    def point_count(self) -> int:
        return int(sum(len(c) for c in self.contours))


@dataclass(frozen=True)
class BinarizeResult:
    binary: np.ndarray
    method: str
    candidates: tuple[CandidateScore, ...]
    edges: SubpixelEdges
    linewidth_px: float
    linewidth_mad_px: float
    components: int
    foreground_px: int
    warnings: tuple[str, ...]


# ---------------------------------------------------------------------------
# 四路二值化（统一约定：返回 uint8，255 = 墨迹/前景）
# ---------------------------------------------------------------------------
def _odd(value: float, *, minimum: int = 3) -> int:
    n = max(minimum, int(round(value)))
    return n if n % 2 == 1 else n + 1


def _window_px(ratio: float, shape: tuple[int, int], *, minimum: int = 3) -> int:
    short_side = min(shape)
    return _odd(short_side * ratio, minimum=minimum)


def binarize_otsu(gray: np.ndarray) -> np.ndarray:
    u8 = _to_u8(gray)
    _, binary = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def _local_mean_std(u8: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    kernel = (window, window)
    img = u8.astype(np.float32)
    mean = cv2.boxFilter(img, cv2.CV_32F, kernel, normalize=True)
    mean_sq = cv2.boxFilter(img * img, cv2.CV_32F, kernel, normalize=True)
    variance = np.clip(mean_sq - mean * mean, 0.0, None)
    return mean, np.sqrt(variance)


def binarize_sauvola(gray: np.ndarray, *, window: int, k: float, r: float) -> np.ndarray:
    u8 = _to_u8(gray)
    mean, std = _local_mean_std(u8, window)
    threshold = mean * (1.0 + k * (std / max(r, _EPS) - 1.0))
    return np.where(u8.astype(np.float32) < threshold, 255, 0).astype(np.uint8)


def binarize_niblack(gray: np.ndarray, *, window: int, k: float) -> np.ndarray:
    u8 = _to_u8(gray)
    mean, std = _local_mean_std(u8, window)
    threshold = mean + k * std
    return np.where(u8.astype(np.float32) < threshold, 255, 0).astype(np.uint8)


def binarize_local_gaussian(
    gray: np.ndarray, *, window: int, offset: float
) -> np.ndarray:
    u8 = _to_u8(gray)
    sigma = max(1.0, window / 6.0)
    background = cv2.GaussianBlur(u8.astype(np.float32), (0, 0), sigma)
    threshold = background - float(offset)
    return np.where(u8.astype(np.float32) < threshold, 255, 0).astype(np.uint8)


# ---------------------------------------------------------------------------
# 选优判据
# ---------------------------------------------------------------------------
def foreground_stability(
    binary: np.ndarray, *, gray: np.ndarray | None = None, speck_area: int = _SPECK_AREA_PX
) -> tuple[float, float, float]:
    """返回 (speck_fraction, width_cv, score)；score 越大越好。

    若给了 `gray`，再加一项**漏墨惩罚**：
    `missed = Σ(1-灰度)·(1-掩膜) / Σ(1-灰度)` —— 图里真正暗的像素有多少没被掩膜覆盖。
    这一项专治局部阈值把粗笔画"掏空"（内部被判成背景，只剩边框）：掏空时 missed 很大。
    只惩罚"漏"，不惩罚"多"：亚像素边缘本来就该落在半亮像素上，多圈 1px 不算错。
    """
    foreground = binary > 0
    total = int(foreground.sum())
    if total == 0:
        return 1.0, 0.0, 0.0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    speck_px = 0
    widths: list[float] = []
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area < speck_area:
            speck_px += area
            continue
        mask = labels == index
        widths.append(2.0 * float(np.median(distance[mask])))
    speck_fraction = speck_px / max(total, 1)
    if len(widths) >= 2:
        arr = np.asarray(widths, dtype=np.float64)
        mean = float(arr.mean())
        width_cv = float(arr.std() / mean) if mean > _EPS else 0.0
    else:
        width_cv = 0.0

    missed = 0.0
    if gray is not None:
        image = np.asarray(gray, dtype=np.float32)
        if image.max() > 1.0:
            image = image / 255.0
        darkness = np.clip(1.0 - image, 0.0, 1.0)
        total_dark = float(darkness.sum())
        if total_dark > _EPS:
            missed = float((darkness * (~foreground)).sum()) / total_dark
    # 只惩罚"大面积漏墨"（掏空），容差 0.2：AA 边缘的半亮像素本来就会落在掩膜外，
    # 若零容忍，正确掩膜会被误伤，选优反而偏向"越粗越好"。
    penalty = max(0.0, missed - _MISSED_TOLERANCE)
    score = 1.0 / ((1.0 + speck_fraction) * (1.0 + width_cv) * (1.0 + penalty))
    return speck_fraction, width_cv, score


# ---------------------------------------------------------------------------
# 二值后清理
# ---------------------------------------------------------------------------
def postprocess(
    binary: np.ndarray,
    *,
    cfg: BinarizeConfig,
    dpi: float,
    linewidth_px: float,
    warnings: list[str],
) -> np.ndarray:
    """按 `binarize.post` 清理：填孔 → 去小对象 → 细线修复。顺序固定（SPEC §4 S2）。"""
    mm_per_px = 25.4 / dpi
    px_per_mm2 = 1.0 / (mm_per_px * mm_per_px)
    out = binary.copy()

    if cfg.post.fill_holes:
        # 只填面积 ≤ hole_max_mm2 的孔：整片背景会通过边界判定为“外部”，不会被填
        max_hole_px = float(cfg.post.hole_max_mm2) * px_per_mm2
        inverted = cv2.bitwise_not(out)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(inverted, 8)
        height, width = out.shape[:2]
        for index in range(1, count):
            x, y, w, h, area = stats[index]
            touches_border = x == 0 or y == 0 or x + w >= width or y + h >= height
            if touches_border or float(area) > max_hole_px:
                continue
            out[labels == index] = 255

    min_object_px = float(cfg.post.remove_small_objects_mm2) * px_per_mm2
    if min_object_px > 0:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(out, 8)
        removed = 0
        for index in range(1, count):
            if float(stats[index, cv2.CC_STAT_AREA]) < min_object_px:
                out[labels == index] = 0
                removed += 1
        if removed:
            warnings.append(f"去小对象：移除 {removed} 个面积 < {cfg.post.remove_small_objects_mm2:g}mm² 的连通域")

    if cfg.post.thin_line_repair:
        size = _odd(max(3.0, linewidth_px * 0.8), minimum=3)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel)

    if cfg.post.break_touching_text:
        warnings.append("break_touching_text=true：该选项尚未实现（会偏置线宽），已跳过")
    return out


# ---------------------------------------------------------------------------
# 亚像素边缘
# ---------------------------------------------------------------------------
def _bilinear(image: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """双线性采样。**坐标是连续像素坐标**：像素 i 覆盖 [i, i+1)，中心在 i+0.5。"""
    height, width = image.shape[:2]
    x = np.clip(xs - 0.5, 0.0, width - 1.001)
    y = np.clip(ys - 0.5, 0.0, height - 1.001)
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    fx = (x - x0).astype(np.float32)
    fy = (y - y0).astype(np.float32)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    top = image[y0, x0] * (1 - fx) + image[y0, x1] * fx
    bottom = image[y1, x0] * (1 - fx) + image[y1, x1] * fx
    return top * (1 - fy) + bottom * fy


def _parabola_crossing(ts: np.ndarray, vals: np.ndarray, target: float) -> float | None:
    """在三点（等距）上拟合抛物线，解出 v(t)=target 的位置；失败返回 None。

    为什么不像常见做法那样对 |∇| 求峰：AA 渲染的阶跃边上 |∇| 是一段平台（相邻两像素
    等值），抛物线顶点在平台上完全不可信——实测偏移会错到 0.4–1.5 px。
    对**灰度**做 50% 电平穿越，在同样数据上稳定到 0.01 px 量级。
    """
    if len(ts) != 3 or len(vals) != 3:
        return None
    coefficients = np.polyfit(ts, vals, 2)
    coefficients[-1] -= target
    roots = np.roots(coefficients)
    real = [float(r.real) for r in roots if abs(r.imag) < 1e-9]
    real = [r for r in real if ts[0] - 1e-9 <= r <= ts[-1] + 1e-9]
    if not real:
        return None
    return min(real, key=lambda r: abs(r - float(np.mean(ts))))


def _edge_offset(
    gray: np.ndarray, cx: float, cy: float, nx: float, ny: float
) -> float | None:
    """沿法线找 50% 电平穿越（相对轮廓点的亚像素偏移）。

    **双向搜索、取离 0 最近的穿越**：轮廓点来自二值掩膜边界，而掩膜会把"半墨像素"
    （例如墨量 0.75）也算成前景，于是真正的 50% 电平落在轮廓点**内侧**。旧实现按
    t 从 −span 到 +span 扫描、返回**第一个**穿越，在细笔画上会抓到笔画另一侧的边缘
    （±2px 窗口已经跨过整条 2px 笔画），实测把 0.18mm 笔画的墨宽抬高了 0.69px。
    """
    ts = np.arange(-_SUBPIX_SPAN_PX, _SUBPIX_SPAN_PX + 1e-9, _SUBPIX_STEP_PX)
    values = _bilinear(gray, cx + ts * nx, cy + ts * ny)
    low, high = float(values.min()), float(values.max())
    if high - low < _MIN_EDGE_CONTRAST:
        return None
    target = 0.5 * (low + high)
    centered = values - target
    crossings: list[float] = []
    for index in range(len(ts) - 1):
        a, b = centered[index], centered[index + 1]
        if a == 0.0:
            crossings.append(float(ts[index]))
            continue
        if a * b > 0:
            continue
        if index == 0:
            triple = slice(0, 3)
        elif index + 2 >= len(ts):
            triple = slice(len(ts) - 3, len(ts))
        else:
            triple = slice(index - 1, index + 2)
        crossing = _parabola_crossing(ts[triple], values[triple], target)
        if crossing is None:
            # 抛物线解不出时退回线性插值
            crossing = float(ts[index] + (ts[index + 1] - ts[index]) * (a / (a - b)))
        crossings.append(float(crossing))
    if not crossings:
        return None
    return min(crossings, key=abs)


def _width_at(
    x: float,
    y: float,
    distance: np.ndarray,
    tree: "cKDTree | None",
    skeleton_widths: np.ndarray,
) -> float:
    """某轮廓点（连续坐标）所属笔画的局部宽度。

    像素栅格上「宽 w 的笔画，脊线处 DT=(w+1)/2」，所以 `2×DT` 恒比真值大 1；
    这里减 1 并夹到 ≥1。**全局线宽**另用 area/skeleton 精确计算，见 `_global_linewidth`。
    """
    if tree is not None and skeleton_widths.size:
        # 树建在像素索引上，连续坐标要减去 0.5
        _, index = tree.query([float(x) - 0.5, float(y) - 0.5], k=1)
        return max(1.0, float(skeleton_widths[int(index)]) - 1.0)
    height, width = distance.shape[:2]
    ix = int(np.clip(round(x - 0.5), 0, width - 1))
    iy = int(np.clip(round(y - 0.5), 0, height - 1))
    return max(1.0, 2.0 * float(distance[iy, ix]) - 1.0)


def _global_linewidth(binary: np.ndarray) -> float:
    """全局线宽：**逐连通域取 `2 × p90(EDT)`，再对各连通域取中位数**（M8-3）。

    为什么换掉 `area / max(骨架长, 主轴向长度)`：
    - 形态学骨架在 ≥90° 折弯处会**切角**，骨架长被低估（实测弯折笔画 420 px vs 几何
      504 px，−17%），于是"面积/骨架长"系统性高估（四场景 +4% ~ +18.5%）；
    - `major_axis` 兜底从未生效（骨架长总是更大），属于死代码；
    - EDT 是**纯局部几何量**，不经过骨架，折角处不会切角。

    为什么是 p90：横截面上的 EDT 近似为 `0.5, 1.5, …, w/2`，中位数会被边缘低值拉到
    `w/2`（实测 4.0 px vs 名义 6.0）；p90 在直线/弧/圆/折线四种几何上实测全部精确命中
    6.000 px（见 `docs/m8-3_edt_poc.md`）。

    为什么逐域再取中位：工程图常有多档线宽，整图统计会被面积最大的连通域主导；
    逐域算再取中位，保留"多线宽图纸取主基线宽"的原有语义。
    """
    foreground = binary > 0
    if not foreground.any():
        return 0.0
    distance = distance_transform_edt(foreground)
    skeleton = skeletonize(foreground)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    widths: list[float] = []
    for index in range(1, count):
        component_area = int(stats[index, cv2.CC_STAT_AREA])
        # 小连通域的 p90 没有统计意义（噪声极大），与去噪阈值取更严者
        if component_area < max(_SPECK_AREA_PX, _MIN_AREA_FOR_WIDTH_PX):
            continue
        mask = labels == index
        values = distance[mask]
        if values.size == 0:
            continue
        edt_width = 2.0 * float(np.percentile(values, 90))
        skeleton_length = int(skeleton[mask].sum())
        if skeleton_length <= 0:
            widths.append(edt_width)
            continue
        area_width = component_area / float(skeleton_length)
        # 两个估计量的偏差**都只出现在偏高侧**：
        # - `面积/骨架长` 因骨架切角而被抬高（弯折笔画 +4%~+18.5%）；
        # - `2×p90(EDT)` 因掩膜量化而被抬高（3 px 笔画 → 4 px 掩膜 → 4.0）。
        # 取二者较小值即可同时消掉两种高估：弯折场景 7.11→6.00，细笔画 4.0→3.02。
        widths.append(min(edt_width, area_width))
    if not widths:
        return 0.0
    return float(np.median(np.asarray(widths, dtype=np.float64)))


def subpixel_edges(
    gray: np.ndarray,
    binary: np.ndarray,
    *,
    gradient_sigma_px: float,
    min_gradient_mag: float,
    corner_preserve: bool,
    warnings: list[str] | None = None,
) -> SubpixelEdges:
    """在二值轮廓上做沿法线的抛物线亚像素修正。"""
    warnings = [] if warnings is None else warnings
    image = np.asarray(gray, dtype=np.float32)
    if image.max() > 1.0:
        image = image / 255.0
    blurred = cv2.GaussianBlur(image, (0, 0), max(0.1, gradient_sigma_px))
    gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    magnitude = np.hypot(gx, gy)

    contours, _ = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

    # 逐点线宽：轮廓点在**笔画的边**上，那里的距离变换≈1（2×DT=2），不是线宽。
    # 正确做法是取最近的脊线（骨架）像素，用它到边的距离×2。
    skeleton = skeletonize(binary > 0)
    if skeleton.any():
        skel_y, skel_x = np.nonzero(skeleton)
        skel_widths = 2.0 * distance[skel_y, skel_x]
        skel_tree = cKDTree(np.column_stack([skel_x, skel_y]).astype(np.float64))
    else:  # 理论上不会发生（没有前景就没有轮廓）
        skel_tree = None
        skel_widths = np.array([])

    out_contours: list[np.ndarray] = []
    out_widths: list[np.ndarray] = []
    shifts: list[float] = []
    corner_skipped = 0
    weak_skipped = 0

    for contour in contours:
        points = contour.reshape(-1, 2).astype(np.float64)
        if len(points) < 8:
            continue
        refined = np.empty_like(points)
        widths = np.empty(len(points), dtype=np.float64)
        count = len(points)

        for index in range(count):
            # OpenCV 轮廓点是**像素索引**；换算成连续坐标（像素中心 = i+0.5）
            px, py = points[index]
            cx, cy = px + 0.5, py + 0.5
            ix, iy = int(round(px)), int(round(py))
            ix = int(np.clip(ix, 0, magnitude.shape[1] - 1))
            iy = int(np.clip(iy, 0, magnitude.shape[0] - 1))
            gxi = float(gx[iy, ix])
            gyi = float(gy[iy, ix])
            grad = float(np.hypot(gxi, gyi))

            keep = False
            if corner_preserve:
                prev = points[(index - 3) % count]
                nxt = points[(index + 3) % count]
                v1 = np.array([px - prev[0], py - prev[1]])
                v2 = np.array([nxt[0] - px, nxt[1] - py])
                norm = float(np.linalg.norm(v1) * np.linalg.norm(v2))
                if norm > _EPS:
                    cosine = float(np.clip(np.dot(v1, v2) / norm, -1.0, 1.0))
                    if np.degrees(np.arccos(cosine)) > _CORNER_ANGLE_DEG:
                        keep = True
                        corner_skipped += 1

            if grad < min_gradient_mag:
                weak_skipped += 1
                refined[index] = (cx, cy)
                widths[index] = _width_at(cx, cy, distance, skel_tree, skel_widths)
                continue

            nx, ny = gxi / grad, gyi / grad
            offset = _edge_offset(image, cx, cy, nx, ny)
            if offset is None:
                weak_skipped += 1
                offset = 0.0
            if keep:  # 角点：只保留整数位置
                offset = 0.0
            refined[index] = (cx + offset * nx, cy + offset * ny)
            shifts.append(abs(offset))
            widths[index] = _width_at(cx, cy, distance, skel_tree, skel_widths)

        out_contours.append(refined)
        out_widths.append(widths)

    if not out_contours:
        warnings.append("S2：没有找到任何前景轮廓（二值图为空）")
    return SubpixelEdges(
        contours=tuple(out_contours),
        widths=tuple(out_widths),
        mean_abs_shift_px=float(np.mean(shifts)) if shifts else 0.0,
        corner_points_skipped=corner_skipped,
        weak_gradient_points=weak_skipped,
    )


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class S2Input:
    """S2 的输入：S1 产物，或裸灰度图（测试用）。"""

    image: np.ndarray
    dpi: float

    @classmethod
    def from_preprocess(cls, result: PreprocessResult) -> "S2Input":
        return cls(image=result.image, dpi=result.dpi)

    @classmethod
    def from_array(cls, image: np.ndarray, *, dpi: float) -> "S2Input":
        arr = np.asarray(image)
        if arr.ndim != 2:
            raise BinarizeError(f"S2 需要灰度图，收到形状 {arr.shape}")
        return cls(image=arr.astype(np.float32), dpi=float(dpi))


def _run_candidates(
    gray: np.ndarray, cfg: BinarizeConfig
) -> tuple[list[tuple[str, np.ndarray, CandidateScore]], float, list[str]]:
    shape = gray.shape[:2]
    warnings: list[str] = []
    # 自适应窗口必须明显大于笔画宽度，否则局部阈值会把粗笔画"掏空"（内部被判成背景）：
    # 实测 10px 笔画配 5px 窗口时，二值图只剩一圈边框，面积/骨架直接掉到 2.5。
    estimated_lw = estimate_linewidth_px(gray)
    floor_window = _odd(max(3.0, 4.0 * estimated_lw)) if estimated_lw > 0 else 3
    results: list[tuple[str, np.ndarray, CandidateScore]] = []

    def choose(ratio: float) -> int:
        wanted = _window_px(ratio, shape)
        window = max(wanted, floor_window)
        if window > wanted:
            warnings.append(
                f"自适应窗口由 {wanted}px 提到 {window}px（≥4×线宽 {estimated_lw:.1f}px）："
                "窗口小于笔画宽度时局部阈值会把粗线掏空"
            )
        return window

    for method in cfg.methods:
        if method == "otsu":
            binary = binarize_otsu(gray)
        elif method == "sauvola":
            binary = binarize_sauvola(
                gray,
                window=choose(cfg.sauvola.window_ratio),
                k=cfg.sauvola.k,
                r=cfg.sauvola.r,
            )
        elif method == "niblack":
            binary = binarize_niblack(
                gray,
                window=choose(cfg.niblack.window_ratio),
                k=cfg.niblack.k,
            )
        elif method == "local_gaussian":
            binary = binarize_local_gaussian(
                gray,
                window=choose(cfg.local_gaussian.window_ratio),
                offset=cfg.local_gaussian.offset,
            )
        else:  # pragma: no cover - Literal 已限死
            raise BinarizeError(f"未知二值化方法：{method}")
        speck, width_cv, score = foreground_stability(binary, gray=gray)
        foreground_ratio = float((binary > 0).mean())
        results.append(
            (
                method,
                binary,
                CandidateScore(
                    method=method,
                    speck_fraction=speck,
                    width_cv=width_cv,
                    foreground_ratio=foreground_ratio,
                    score=score,
                ),
            )
        )
    return results, estimated_lw, warnings


def binarize_image(
    source: S2Input,
    cfg: R2VConfig,
    *,
    out_dir: str | Path | None = None,
) -> tuple[BinarizeResult, StageMetrics]:
    """S2 主流程：四路选优 → 清理 → 亚像素边缘 → 逐点线宽。"""
    t0 = time.perf_counter()
    gray = np.asarray(source.image, dtype=np.float32)
    if gray.ndim != 2:
        raise BinarizeError(f"S2 需要灰度图，收到形状 {gray.shape}")
    if not np.isfinite(gray).all():
        raise BinarizeError("S2 输入包含 NaN/Inf")
    warnings: list[str] = []

    candidates, estimated_lw, window_warnings = _run_candidates(gray, cfg.binarize)
    warnings.extend(window_warnings)
    if not candidates:
        raise BinarizeError("binarize.methods 为空：至少需要一种二值化方法")
    # 确定性选优：分数分桶（同桶视为并列）→ 前景占比最小（更紧的掩膜）→ 碎点最少 → 方法名。
    # 为什么允许"并列"再取紧：局部阈值在笔画边缘会系统性多判 1 列（实测 6px 笔画出 7 列、
    # IoU 0.86），但它的碎点更少，单看分数总会赢；漏墨容差已保证紧掩膜不会漏掉真实墨迹。
    best_method, best_binary, best_score = sorted(
        candidates,
        key=lambda item: (
            -round(item[2].score / _SCORE_TIE_EPS),
            item[2].foreground_ratio,
            round(item[2].speck_fraction, 4),
            item[0],
        ),
    )[0]

    # 先估一次线宽用于清理核（用未清理的二值图，避免清理把细线吃掉后再估宽）
    _, _, _ = foreground_stability(best_binary)
    rough_distance = cv2.distanceTransform(best_binary, cv2.DIST_L2, 5)
    rough_width = (
        2.0 * float(np.median(rough_distance[rough_distance > 0]))
        if (rough_distance > 0).any()
        else 0.0
    )
    cleaned = postprocess(
        best_binary, cfg=cfg.binarize, dpi=source.dpi, linewidth_px=rough_width, warnings=warnings
    )

    subpixel_cfg = cfg.binarize.subpixel
    edges = SubpixelEdges(contours=(), widths=(), mean_abs_shift_px=0.0,
                          corner_points_skipped=0, weak_gradient_points=0)
    if subpixel_cfg.enable:
        if subpixel_cfg.method != "gradient_normal_quad":
            warnings.append(
                f"subpixel.method={subpixel_cfg.method} 尚未实现：退化为 gradient_normal_quad"
            )
        if subpixel_cfg.interpolation != "quadratic":
            warnings.append(
                f"subpixel.interpolation={subpixel_cfg.interpolation} 尚未实现：退化为 quadratic"
            )
        edges = subpixel_edges(
            gray,
            cleaned,
            gradient_sigma_px=subpixel_cfg.gradient_sigma_px,
            min_gradient_mag=subpixel_cfg.min_gradient_mag,
            corner_preserve=subpixel_cfg.corner_preserve,
            warnings=warnings,
        )
    else:
        warnings.append("subpixel.enable=false：只输出像素级二值图，后续拟合精度会下降")

    if cfg.binarize.linewidth.method != "distance_transform_ridge":
        warnings.append(
            f"binarize.linewidth.method={cfg.binarize.linewidth.method} 尚未实现："
            "退化为距离变换脊线"
        )

    all_widths = np.concatenate([w for w in edges.widths if len(w)]) if edges.widths else np.array([])
    point_width_px = float(np.median(all_widths)) if all_widths.size else 0.0
    # 线宽用**清理前**的二值图算：thin_line_repair 的闭运算会把线段两端各撑长约半个核，
    # 面积变大而骨架不变，面积/骨架比会被虚高（实测 10px 笔画量成 10.42）。
    linewidth_px = _global_linewidth(best_binary)
    if linewidth_px <= 0.0:
        linewidth_px = point_width_px
    mad_px = (
        float(np.median(np.abs(all_widths - point_width_px))) if all_widths.size else 0.0
    )
    components = int(cv2.connectedComponents(cleaned, 8)[0] - 1)
    foreground_px = int((cleaned > 0).sum())

    result = BinarizeResult(
        binary=cleaned,
        method=best_method,
        candidates=tuple(score for _, _, score in candidates),
        edges=edges,
        linewidth_px=round(linewidth_px, 4),
        linewidth_mad_px=round(mad_px, 4),
        components=components,
        foreground_px=foreground_px,
        warnings=tuple(warnings),
    )

    timing_ms = round((time.perf_counter() - t0) * 1000.0, 3)
    metrics = StageMetrics(
        stage="s2",
        counts={
            "foreground_px": foreground_px,
            "components": components,
            "contours": len(edges.contours),
            "subpixel_points": edges.point_count,
            "warnings": len(warnings),
        },
        timing_ms=timing_ms,
        error_estimate={
            "linewidth_px": result.linewidth_px,
            "linewidth_point_median_px": round(point_width_px, 4),
            "linewidth_mad_px": result.linewidth_mad_px,
            "mean_abs_subpixel_shift_px": round(edges.mean_abs_shift_px, 6),
        },
        warnings=tuple(warnings),
        details={
            "method": best_method,
            "estimated_input_linewidth_px": round(float(estimated_lw), 4),
            "scores": {
                c.method: {
                    "score": round(c.score, 6),
                    "speck_fraction": round(c.speck_fraction, 6),
                    "width_cv": round(c.width_cv, 6),
                    "foreground_ratio": round(c.foreground_ratio, 6),
                }
                for c in result.candidates
            },
            "corner_points_skipped": edges.corner_points_skipped,
            "weak_gradient_points": edges.weak_gradient_points,
            "mm_per_px": round(25.4 / source.dpi, 9),
        },
    )

    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out / "binary.png"), cleaned)
        write_metrics(out, metrics)
        write_json(out / "s2_report.json", _report_json(result, source))
    return result, metrics


def _report_json(result: BinarizeResult, source: S2Input) -> dict[str, Any]:
    return {
        "method": result.method,
        "linewidth_px": result.linewidth_px,
        "linewidth_mad_px": result.linewidth_mad_px,
        "components": result.components,
        "foreground_px": result.foreground_px,
        "contours": len(result.edges.contours),
        "subpixel_points": result.edges.point_count,
        "mean_abs_shift_px": result.edges.mean_abs_shift_px,
        "dpi": source.dpi,
        "scores": [
            {
                "method": c.method,
                "score": c.score,
                "speck_fraction": c.speck_fraction,
                "width_cv": c.width_cv,
                "foreground_ratio": c.foreground_ratio,
            }
            for c in result.candidates
        ],
        "warnings": list(result.warnings),
    }


__all__ = [
    "BinarizeError",
    "BinarizeResult",
    "CandidateScore",
    "S2Input",
    "SubpixelEdges",
    "binarize_image",
    "binarize_local_gaussian",
    "binarize_niblack",
    "binarize_otsu",
    "binarize_sauvola",
    "foreground_stability",
    "postprocess",
    "subpixel_edges",
]
