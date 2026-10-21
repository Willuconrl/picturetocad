# tests/synthetic_golden.py
"""合成图 + 程序生成 golden DXF 的回归夹具（用户决策 #2）。

为什么要有它
------------
`golden/` 里没有手工描的 DXF 之前，Hausdorff / 实体数 / 关键尺寸三项回归**无从比对**。
本模块用「已知参数」造图：几何参数就是真相（golden），位图与 DXF 都从同一份参数派生，
因此可以精确断言「管线输出的端点/半径/弧长是否偏离真相」。

契约
----
1. 每个 `Scene` 是**自洽的**：`specs` 是唯一真相，PNG 与 DXF 都不得引入额外信息。
2. 坐标一律 CAD 系：原点左下、Y 向上、单位 mm（SPEC §3.1）。
3. 渲染与写出必须**确定性**（同参数两次生成 PNG 字节一致），否则回归无从谈起。
4. 栅格化用 1/16 px 定点（`cv2` 的 `shift`）+ `LINE_AA`，避免整像素截断污染端点精度。

用法：
    python tests/synthetic_golden.py --images test_images --golden golden    # 生成资产
    python tests/synthetic_golden.py --check                                 # CI：资产是否与代码一致
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

# 直接 `python tests/synthetic_golden.py` 运行时，把仓库根放进 sys.path（供 import src.*）
if __package__ in (None, ""):
    _ROOT = Path(__file__).resolve().parents[1]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

import cv2
import ezdxf
import numpy as np
from ezdxf.math import arc_angle_span_rad, bulge_to_arc

from src.dxf_repro import dxf_bytes as dxf_bytes_for_drawing

# 物理常数（不是可调参数）：1 英寸 = 25.4 mm
MM_PER_INCH = 25.4

#: 定点位数：坐标量化到 1/16 px
SHIFT_BITS = 4

#: 曲线采样密度：弦长 ≤ 0.25 px（12 mm 圆弧上弦高误差 < 5e-6 mm，可忽略）
CHORD_FRACTION_PX = 0.25

Point = tuple[float, float]


# ---------------------------------------------------------------------------
# 几何规格（唯一真相）
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LineSpec:
    start: Point
    end: Point
    layer: str = "LINES_W0.50"

    def __post_init__(self) -> None:
        if self.start == self.end:
            raise ValueError("LineSpec 起止点不能相同")


@dataclass(frozen=True)
class CircleSpec:
    center: Point
    radius: float
    layer: str = "CIRCLES"

    def __post_init__(self) -> None:
        if self.radius <= 0:
            raise ValueError(f"CircleSpec.radius 必须 > 0，收到 {self.radius}")


@dataclass(frozen=True)
class ArcSpec:
    center: Point
    radius: float
    start_deg: float
    end_deg: float
    layer: str = "ARCS"

    def __post_init__(self) -> None:
        if self.radius <= 0:
            raise ValueError(f"ArcSpec.radius 必须 > 0，收到 {self.radius}")
        if not (0.0 <= self.start_deg < 360.0 and 0.0 <= self.end_deg < 360.0):
            raise ValueError("ArcSpec 的角度必须落在 [0, 360)")

    @property
    def sweep_deg(self) -> float:
        return float(np.degrees(arc_angle_span_rad(np.radians(self.start_deg), np.radians(self.end_deg))))


@dataclass(frozen=True)
class PolylineSpec:
    points: tuple[Point, ...]
    bulges: tuple[float, ...] = ()
    closed: bool = False
    layer: str = "LINES_W0.50"

    def __post_init__(self) -> None:
        if len(self.points) < 2:
            raise ValueError("PolylineSpec 至少 2 个顶点")
        if self.bulges and len(self.bulges) != len(self.points):
            raise ValueError(
                f"bulges 长度必须等于顶点数（末位对应闭合段）：{len(self.bulges)} != {len(self.points)}"
            )

    @property
    def resolved_bulges(self) -> tuple[float, ...]:
        if self.bulges:
            return self.bulges
        return (0.0,) * len(self.points)


Spec = LineSpec | CircleSpec | ArcSpec | PolylineSpec


@dataclass(frozen=True)
class Scene:
    """一张合成图：画布尺寸 + DPI + 线宽 + 全部图元。"""

    name: str
    width_mm: float
    height_mm: float
    dpi: float
    line_width_px: int
    specs: tuple[Spec, ...]
    note: str = ""

    def __post_init__(self) -> None:
        if self.width_mm <= 0 or self.height_mm <= 0:
            raise ValueError("画布尺寸必须 > 0")
        if self.dpi <= 0:
            raise ValueError("dpi 必须 > 0")
        if self.line_width_px < 1:
            raise ValueError("line_width_px 必须 >= 1")
        if not self.specs:
            raise ValueError("Scene 至少要有一个图元")
        w, h = self.pixel_size()
        if w < 16 or h < 16:
            raise ValueError(f"画布太小：{w}x{h} px")

    @property
    def mm_per_px(self) -> float:
        return MM_PER_INCH / self.dpi

    def pixel_size(self) -> tuple[int, int]:
        w = int(round(self.width_mm / self.mm_per_px))
        h = int(round(self.height_mm / self.mm_per_px))
        return w, h

    @property
    def physical_size_mm(self) -> tuple[float, float]:
        """**实际**物理尺寸 = 像素数 × 物理分辨率（用户决策 A）。

        声明的 `width_mm/height_mm` 与像素栅格往往差 0.02–0.03 mm（30mm→354px=29.98mm），
        而 golden 若用声明值、管线用实际像素高做 Y 翻转，就会引入
        `声明值 − 实际值` 的系统性 y 偏置（实测 ≈0.55px）。这里统一到实际栅格。
        """
        w, h = self.pixel_size()
        return w * self.mm_per_px, h * self.mm_per_px

    @property
    def image_name(self) -> str:
        return f"synthetic_{self.name}.png"

    @property
    def dxf_name(self) -> str:
        return f"synthetic_{self.name}.dxf"


# ---------------------------------------------------------------------------
# 采样（mm，CAD 系）
# ---------------------------------------------------------------------------
def _max_step(scene: Scene) -> float:
    return scene.mm_per_px * CHORD_FRACTION_PX


def _arc_points(center: Point, radius: float, a0_deg: float, a1_deg: float, step_mm: float) -> np.ndarray:
    """从 a0 逆时针到 a1（a1 已保证在 a0 之后）的采样点。"""
    span_rad = arc_angle_span_rad(np.radians(a0_deg), np.radians(a1_deg))
    arc_len = abs(span_rad) * radius
    n = max(2, int(np.ceil(arc_len / step_mm)) + 1)
    angles = np.radians(a0_deg) + np.linspace(0.0, span_rad, n)
    return np.column_stack(
        [center[0] + radius * np.cos(angles), center[1] + radius * np.sin(angles)]
    )


def spec_chain(spec: Spec, scene: Scene) -> tuple[np.ndarray, bool]:
    """把图元转成一条 mm 坐标折线链，并返回是否闭合。"""
    step = _max_step(scene)
    if isinstance(spec, LineSpec):
        return np.array([spec.start, spec.end], dtype=np.float64), False
    if isinstance(spec, CircleSpec):
        n = max(16, int(np.ceil(2.0 * np.pi * spec.radius / step)))
        angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        pts = np.column_stack(
            [spec.center[0] + spec.radius * np.cos(angles), spec.center[1] + spec.radius * np.sin(angles)]
        )
        return pts, True
    if isinstance(spec, ArcSpec):
        return _arc_points(spec.center, spec.radius, spec.start_deg, spec.end_deg, step), False
    if isinstance(spec, PolylineSpec):
        chains: list[np.ndarray] = []
        bulges = spec.resolved_bulges
        indices = range(len(spec.points) if spec.closed else len(spec.points) - 1)
        for i in indices:
            p1 = spec.points[i]
            p2 = spec.points[(i + 1) % len(spec.points)]
            b = bulges[i]
            if abs(b) < 1e-12:
                chunk = np.array([p1, p2], dtype=np.float64)
            else:
                center, a0, a1, radius = bulge_to_arc(p1, p2, b)
                chunk = _arc_points(
                    (center.x, center.y), radius, np.degrees(a0), np.degrees(a1), step
                )
                # bulge<0 时 bulge_to_arc 返回的是 CCW 弧且首尾互换，需要翻回顶点顺序
                if b < 0:
                    chunk = chunk[::-1]
            chains.append(chunk if not chains else chunk[1:])
        return np.vstack(chains), spec.closed
    raise TypeError(f"未知的 Spec 类型：{type(spec)!r}")


def scene_chain_mm(spec: Spec, scene: Scene) -> np.ndarray:
    return spec_chain(spec, scene)[0]


# ---------------------------------------------------------------------------
# 栅格化（mm → px，Y 翻转，1/16 px 定点）
# ---------------------------------------------------------------------------
def to_pixel_coords(points_mm: np.ndarray, scene: Scene) -> np.ndarray:
    """CAD 系（左下原点、Y 上）→ 图像系（左上原点、Y 下）。

    Y 翻转用**实际物理高度**（像素数 × mm/px），不用声明高度：管线是按像素高翻转的，
    两边口径必须一致，否则会留下 `声明值 − 实际值` 的系统性 y 偏置。
    """
    mpp = scene.mm_per_px
    _, height_mm = scene.physical_size_mm
    out = np.empty_like(points_mm)
    out[:, 0] = points_mm[:, 0] / mpp
    out[:, 1] = (height_mm - points_mm[:, 1]) / mpp
    return out


def _distance_to_segment(
    gx: np.ndarray, gy: np.ndarray, x0: float, y0: float, x1: float, y1: float
) -> np.ndarray:
    """点集到线段的距离（向量化）。"""
    dx, dy = x1 - x0, y1 - y0
    length_sq = dx * dx + dy * dy
    if length_sq < 1e-12:
        return np.hypot(gx - x0, gy - y0)
    t = np.clip(((gx - x0) * dx + (gy - y0) * dy) / length_sq, 0.0, 1.0)
    return np.hypot(gx - (x0 + t * dx), gy - (y0 + t * dy))


def render_scene(scene: Scene) -> np.ndarray:
    """把 Scene 渲染成 uint8 灰度图（背景 255、线 0）。纯确定性，不含随机。

    **解析覆盖率光栅化**（M7 fixture v2）：像素覆盖率 =
    `clamp(半径 + 0.5 − 到线段的距离, 0, 1)`，与 golden DXF 共用同一条解析链，
    因此"图—矢量"两侧真值一致。

    为什么不用 `cv2.polylines(lineType=cv2.LINE_AA)`：实测其 50% 墨宽恒为
    `thickness + 3`（3→5.98px、6→7.98px），名义线宽与物理墨宽差 0.13mm，
    会直接把 golden 的图层标签带错。
    """
    width_px, height_px = scene.pixel_size()
    ink = np.zeros((height_px, width_px), dtype=np.float64)
    radius = scene.line_width_px / 2.0
    for spec in scene.specs:
        chain_mm, closed = spec_chain(spec, scene)
        points = to_pixel_coords(np.asarray(chain_mm, dtype=np.float64), scene)
        segments = list(zip(points[:-1], points[1:]))
        if closed and len(points) > 2:
            segments.append((points[-1], points[0]))
        for (x0, y0), (x1, y1) in segments:
            pad = radius + 1.0
            col0 = max(0, int(np.floor(min(x0, x1) - pad)))
            col1 = min(width_px, int(np.ceil(max(x0, x1) + pad)) + 1)
            row0 = max(0, int(np.floor(min(y0, y1) - pad)))
            row1 = min(height_px, int(np.ceil(max(y0, y1) + pad)) + 1)
            if col1 <= col0 or row1 <= row0:
                continue
            ys, xs = np.mgrid[row0:row1, col0:col1]
            gx = xs + 0.5      # 像素中心（连续坐标契约：像素 i 覆盖 [i, i+1)）
            gy = ys + 0.5
            distance = _distance_to_segment(gx, gy, x0, y0, x1, y1)
            coverage = np.clip(radius + 0.5 - distance, 0.0, 1.0)
            ink[row0:row1, col0:col1] = np.maximum(ink[row0:row1, col0:col1], coverage)
    return np.round(255.0 * (1.0 - ink)).astype(np.uint8)


def render_png_bytes(scene: Scene) -> bytes:
    """PNG 字节。用 Pillow 写，**带上 DPI 元数据**（pHYs）——真实扫描件都有，
    且 S0 质检从这里读 DPI；cv2.imencode 不写 pHYs。
    """
    import io

    from PIL import Image

    image = Image.fromarray(render_scene(scene))
    buf = io.BytesIO()
    image.save(buf, format="PNG", dpi=(scene.dpi, scene.dpi))
    return buf.getvalue()


# ---------------------------------------------------------------------------
# DXF 写出（与 PNG 同源）
# ---------------------------------------------------------------------------
def build_dxf(scene: Scene) -> ezdxf.document.Drawing:
    doc = ezdxf.new("R2018", setup=False)
    doc.header["$INSUNITS"] = 4  # mm
    doc.header["$MEASUREMENT"] = 1
    layers = sorted({spec.layer for spec in scene.specs})
    for layer in layers:
        if layer not in doc.layers:
            doc.layers.add(layer)
    msp = doc.modelspace()
    for spec in scene.specs:
        attribs = {"layer": spec.layer}
        if isinstance(spec, LineSpec):
            msp.add_line(spec.start, spec.end, dxfattribs=attribs)
        elif isinstance(spec, CircleSpec):
            msp.add_circle(spec.center, spec.radius, dxfattribs=attribs)
        elif isinstance(spec, ArcSpec):
            msp.add_arc(
                spec.center, spec.radius, spec.start_deg, spec.end_deg, dxfattribs=attribs
            )
        elif isinstance(spec, PolylineSpec):
            pts = [
                (x, y, 0.0, 0.0, float(b))
                for (x, y), b in zip(spec.points, spec.resolved_bulges)
            ]
            msp.add_lwpolyline(pts, format="xyseb", close=spec.closed, dxfattribs=attribs)
        else:  # pragma: no cover - 类型已穷举
            raise TypeError(f"未知的 Spec 类型：{type(spec)!r}")
    return doc


def dxf_bytes(scene: Scene) -> bytes:
    """确定性 DXF 字节。

    实现已统一到 `src/dxf_repro.py`（与 `src/dxf_writer.py` 共用同一套固化逻辑：
    固定时间戳/GUID/标记串、CLASSES 段排序、只写 LF 的字节流），避免两处漂移。
    """
    return dxf_bytes_for_drawing(build_dxf(scene))


# ---------------------------------------------------------------------------
# 场景清单
# ---------------------------------------------------------------------------
def all_scenes() -> tuple[Scene, ...]:
    """首批合成图（用户决策 #2 要求的最小覆盖）。"""
    return (
        Scene(
            name="circle_full",
            width_mm=40.0,
            height_mm=40.0,
            dpi=300.0,
            line_width_px=6,
            specs=(CircleSpec(center=(20.0, 20.0), radius=12.0),),
            note="完整圆 r=12mm，用于 CIRCLE 判定与半径相对误差",
        ),
        Scene(
            name="line_single",
            width_mm=60.0,
            height_mm=30.0,
            dpi=300.0,
            line_width_px=6,
            specs=(LineSpec(start=(8.0, 10.0), end=(52.0, 20.0)),),
            note="单条斜线 (8,10)-(52,20)，用于端点误差与长度相对误差",
        ),
        Scene(
            name="arc_120",
            width_mm=40.0,
            height_mm=40.0,
            dpi=300.0,
            line_width_px=6,
            specs=(
                ArcSpec(center=(20.0, 20.0), radius=15.0, start_deg=30.0, end_deg=150.0),
            ),
            note="120° 圆弧 r=15mm（30°→150°），用于 ARC 半径与扫角",
        ),
        Scene(
            name="polyline_bulge",
            width_mm=50.0,
            height_mm=40.0,
            dpi=300.0,
            line_width_px=6,
            specs=(
                PolylineSpec(
                    points=((10.0, 10.0), (30.0, 10.0), (40.0, 25.0)),
                    bulges=(0.5, 0.0, 0.0),
                ),
            ),
            note="LWPOLYLINE 含 bulge=0.5 首段（106.26° 弧），用于 bulge 写读一致",
        ),
    )


# ---------------------------------------------------------------------------
# 生成 / 校验
# ---------------------------------------------------------------------------
def generate(images_dir: Path, golden_dir: Path) -> list[str]:
    """生成全部合成图与 golden DXF，返回写出的文件名列表。

    直接 `write_bytes`（不经过平台换行转换），保证同一参数两次生成**字节一致**。
    """
    images_dir.mkdir(parents=True, exist_ok=True)
    golden_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for scene in all_scenes():
        png = images_dir / scene.image_name
        png.write_bytes(render_png_bytes(scene))
        written.append(str(png))
        dxf = golden_dir / scene.dxf_name
        dxf.write_bytes(dxf_bytes(scene))
        written.append(str(dxf))
    return written


def check(images_dir: Path, golden_dir: Path) -> list[str]:
    """校验磁盘资产与代码定义一致（逐字节）；返回问题清单（空 = 通过）。"""
    problems: list[str] = []
    for scene in all_scenes():
        png = images_dir / scene.image_name
        if not png.is_file():
            problems.append(f"缺图：{png}")
        elif png.read_bytes() != render_png_bytes(scene):
            problems.append(f"图与代码不一致（像素有差异）：{png}")
        dxf = golden_dir / scene.dxf_name
        if not dxf.is_file():
            problems.append(f"缺 golden：{dxf}")
            continue
        if dxf.read_bytes() != dxf_bytes(scene):
            problems.append(f"golden 与代码不一致（字节有差异）：{dxf}")
        doc = ezdxf.readfile(dxf)
        actual = len(list(doc.modelspace()))
        if actual != len(scene.specs):
            problems.append(f"实体数不符：{dxf} 有 {actual} 个，期望 {len(scene.specs)}")
    return problems


def _main(argv: Sequence[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description="生成/校验合成图与 golden DXF")
    ap.add_argument("--images", default=str(root / "test_images"))
    ap.add_argument("--golden", default=str(root / "golden"))
    ap.add_argument("--check", action="store_true", help="只校验，不写文件")
    args = ap.parse_args(argv)

    images, golden = Path(args.images), Path(args.golden)
    if args.check:
        problems = check(images, golden)
        for p in problems:
            print(f"FAIL {p}")
        print(f"CHECK scenes={len(all_scenes())} problems={len(problems)}")
        return 0 if not problems else 1

    for name in generate(images, golden):
        print(f"WROTE {name}")
    print(f"GENERATE scenes={len(all_scenes())}")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
