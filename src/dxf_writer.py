# src/dxf_writer.py
"""
R2V DXF 写出模块 —— 唯一允许触碰 ezdxf 的地方。

契约（违反即失败）：
1. 输入坐标必须是 float64、单位毫米、CAD 系（原点在左下，Y 向上）。
2. 所有坐标写出前统一 round4(0.0001mm)，舍入不得成为误差源。
3. 线型必须在图层之前注册；图层必须在实体之前创建。
4. 完整圆必须用 CIRCLE 实体，禁止用折线拼圆。
5. 写出后必须回读自检（逐实体比对几何，不是只数实体个数），
   不一致则抛 DXFSelfCheckError 并退出码 3。
6. 任何异常不得编造几何。预期内的坏输入记 WARN 并跳过；
   **非预期异常必须向上抛**，不得吞掉——静默失败会产出看起来成功的错误图纸。

ezdxf 1.4.4 实测约束（这些坑会让写出"成功"但图元是错的）：
- `LWPolyline.set_bulge()` 不存在；bulge 只能用
  `add_lwpolyline(..., format="xyseb")` + 每顶点 5 元组 `(x, y, sw, ew, bulge)` 写入。
- `add_lwpolyline()` 的关键字是 `close`（不是 `closed`）；`format="xyb"` 非法，bulge 会被丢弃。
- `add_text()` 没有 `insert` 参数，位置必须走 `dxfattribs={"insert": (x, y)}`。
- `MText.height` / `MText.width` 不是 DXF 属性（赋值静默无效），须用
  `mt.dxf.char_height` / `mt.dxf.width`；`add_mtext()` 的合法属性名是 `insert`。
- `$COMMENTS` 不是合法头变量；元数据写 `$LASTSAVEDBY` / `$PROJECTNAME` /
  `header.custom_vars`（三者均实测可落盘并回读）。`Drawing.comments` 是幽灵属性。
- 图层可以引用不存在的线型而不报错，必须自己校验 `_register_linetypes` 的结果。
"""

from __future__ import annotations

import json
import logging
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

import ezdxf
import numpy as np
from ezdxf.document import Drawing
from ezdxf.entities import LWPolyline

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 常量与类型
# ---------------------------------------------------------------------------

DxfVersion = Literal["R2018", "R2010", "R2007", "R2000"]
EntityKind = Literal["LINE", "LWPOLYLINE", "ARC", "CIRCLE", "TEXT", "MTEXT"]

EXIT_SELF_CHECK_FAILED = 3   # 契约第 5 条：自检失败退出码

COORD_DECIMALS = 4          # 0.0001 mm
SELF_CHECK_TOL = 1e-3       # mm，自检坐标容差（全量逐实体比对，不再抽样）

# 线宽档（mm）-> DXF 标准线宽枚举（百分之一毫米）。
# DXF 只允许有限档位，吸附到就近档位；写不进档位时退回 BYLAYER，
# 并由 LWPOLYLINE 的 const_width 承担真实毫米线宽。
_LINEWEIGHT_ENUM = (
    0, 5, 9, 13, 15, 18, 20, 25, 30, 35, 40, 50, 53, 60, 70, 80,
    90, 100, 106, 120, 140, 158, 200, 211,
)


def lineweight_enum(linewidth_mm: float | None) -> int:
    """毫米线宽 -> DXF 线宽枚举值（1/100 mm 档位）；None -> BYLAYER(-1)。"""
    if linewidth_mm is None:
        return -1  # LINEWEIGHT_BYLAYER
    hundredths = float(linewidth_mm) * 100.0
    return min(_LINEWEIGHT_ENUM, key=lambda v: abs(v - hundredths))

# 图层定义：名称 / 颜色索引(ACI) / 线型
# 注意：dxf.layers 的 config.yaml 才是图层表的真相源；这里是 ezdxf 侧默认值，
# scripts/check_contracts.py 会核对二者一致（当前 CENTER 与 HIDDEN 存在已知差异）。
_LAYER_DEFS: tuple[tuple[str, int, str], ...] = (
    ("LINES_W0.18",   7, "CONTINUOUS"),
    ("LINES_W0.35",   4, "CONTINUOUS"),
    ("LINES_W0.50",   1, "CONTINUOUS"),
    ("ARCS",          3, "CONTINUOUS"),
    ("CIRCLES",       5, "CONTINUOUS"),
    ("TEXT",          6, "CONTINUOUS"),
    ("BORDER",        8, "CONTINUOUS"),
    ("CENTER",        2, "DASHED"),      # config.yaml 要求 2/DASHED
    ("HIDDEN",        9, "HIDDEN"),
    ("DEBUG",       252, "CONTINUOUS"),
)

# 自定义线型：名称 / 图案(mm) / 描述
_LINE_TYPES: tuple[tuple[str, Sequence[float], str], ...] = (
    ("CENTER",  [12.0, -3.0, 3.0, -3.0],           "Center line"),
    ("DASHED",  [6.0, -3.0],                       "Dashed line"),
    ("HIDDEN",  [6.0, -3.0, 1.5, -3.0],            "Hidden line"),
    ("PHANTOM", [18.0, -3.0, 3.0, -3.0, 3.0, -3.0],"Phantom line"),
)


class DXFSelfCheckError(RuntimeError):
    """回读自检未通过。按契约第 5 条，调用方应以退出码 3 终止。"""

    exit_code = EXIT_SELF_CHECK_FAILED


@dataclass(frozen=True)
class LayerConfig:
    """线宽档位 -> 图层名映射（由 S5 聚类结果生成）"""
    bins: tuple[float, ...]          # 升序，mm，例如 (0.25, 0.42)
    names: tuple[str, ...]           # len = len(bins)+1，例如 ("LINES_W0.18","LINES_W0.35","LINES_W0.50")

    def __post_init__(self) -> None:
        # 用 raise 而非 assert：assert 在 python -O 下被剥离，非法配置会静默放行
        if len(self.names) != len(self.bins) + 1:
            raise ValueError(
                f"图层名数必须 = 档位数+1：names={len(self.names)} bins={len(self.bins)}"
            )
        if list(self.bins) != sorted(self.bins):
            raise ValueError(f"bins 必须升序：{self.bins}")

    def resolve(self, linewidth_mm: float | None) -> str:
        if linewidth_mm is None or not self.bins:
            return self.names[0]
        for i, b in enumerate(self.bins):
            if linewidth_mm < b:
                return self.names[i]
        return self.names[-1]


@dataclass
class Entity:
    """管线 S4/S6 产出的几何实体，DXF 写出的唯一输入形态。"""
    kind: EntityKind
    layer: str
    # LINE: 2点; ARC: 圆心+半径+起止角(度); CIRCLE: 圆心+半径;
    # LWPOLYLINE: N点; TEXT/MTEXT: 插入点+文字
    data: dict[str, Any]
    linewidth_mm: float | None = None
    bulges: Sequence[float] | None = None   # 仅 LWPOLYLINE，len = len(points)-1（闭合时最后一段单独处理）
    meta: dict[str, Any] = field(default_factory=dict)  # 来源边id等，用于追溯


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def round4(v: float) -> float:
    """坐标统一舍入到 0.0001 mm，避免文本化时的尾数噪声。"""
    return round(float(v), COORD_DECIMALS)


def flip_y(pts: np.ndarray, height_mm: float) -> np.ndarray:
    """图像物理坐标(mm_img) -> CAD坐标(mm_cad)：Y 翻转。禁止分散写死 h-y。"""
    out = np.asarray(pts, dtype=np.float64).reshape(-1, 2).copy()
    out[:, 1] = height_mm - out[:, 1]
    return out


def apply_affine(pts: np.ndarray, m: np.ndarray) -> np.ndarray:
    """齐次 3x3 仿射变换，float64。m 为 S7 配准输出的 world 矩阵。"""
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    h = np.column_stack([p, np.ones(len(p), dtype=np.float64)])
    return (h @ m.T)[:, :2]


def _bulge_ok(b: float) -> bool:
    return isinstance(b, (int, float)) and np.isfinite(b)


# ---------------------------------------------------------------------------
# 主类
# ---------------------------------------------------------------------------

class DXFWriter:
    """
    DXF 写出器。生命周期：
      构造(_register_linetypes -> _setup_layers -> _setup_styles)
      -> add_* / write_entities
      -> save()（含回读自检 + metrics + 元数据注释）
    """

    def __init__(
        self,
        version: DxfVersion = "R2018",
        layer_config: LayerConfig | None = None,
        image_height_mm: float | None = None,
        transform_mm_img_to_world: np.ndarray | None = None,
        warnings: list[str] | None = None,
    ) -> None:
        self.version = version
        self.layer_config = layer_config or LayerConfig(bins=(), names=("LINES_W0.18",))

        # 契约第 1 条：输入必须是 CAD 系（Y 向上）。翻转所需的图幅高度不可缺省——
        # 缺省时若静默跳过翻转，全图上下颠倒却毫无提示，属于"编造几何"。
        if image_height_mm is None:
            raise ValueError(
                "image_height_mm 必须提供：内部坐标是 Y 向下的图像物理系(mm_img)，"
                "DXF 需要 Y 向上的 CAD 系(mm_cad)。缺少图幅高度将无法翻转，"
                "禁止静默输出未翻转的坐标。"
            )
        self.height_mm = float(image_height_mm)
        self._m_world = (
            np.asarray(transform_mm_img_to_world, dtype=np.float64)
            if transform_mm_img_to_world is not None else np.eye(3, dtype=np.float64)
        )
        if self._m_world.shape != (3, 3):
            raise ValueError(f"transform_mm_img_to_world 必须是 3x3 齐次矩阵，实际 {self._m_world.shape}")

        self.warnings = list(warnings or [])

        # 反射检测：_to_cad 里的 Y 翻转本身是一次镜像（定向翻转）。
        # 世界矩阵若也含反射（det<0），两次镜像相抵、定向不变；
        # 故"输出相对输入是否被镜像" = (Y翻转的 −1) × (世界矩阵 det 的符号) 是否为正号取反。
        world_det = float(np.linalg.det(self._m_world))
        if abs(world_det) < np.finfo(np.float64).eps:
            raise ValueError("transform_mm_img_to_world 退化（det≈0），无法写出")
        self._reflected = world_det > 0.0
        if abs(abs(world_det) - 1.0) > 1e-9:
            self.warnings.append(
                f"世界矩阵 det={world_det:.6g} 非单位缩放：CIRCLE 会退化为椭圆，"
                "DXF 无法表达，请在上游改为拟合椭圆或多段线"
            )

        self.doc: Drawing = ezdxf.new(version)
        self.msp = self.doc.modelspace()
        self._entities_written = 0
        self._written: list[Entity] = []           # 用于自检与报告
        # 期望几何快照：自检逐条比对"我写了什么"与"文件里读到了什么"
        self._written_geom: list[dict[str, Any]] = []

        self._setup_headers()
        self._register_linetypes()   # 必须先于图层
        self._setup_layers()
        self._setup_styles()

    # ----- 内部初始化 --------------------------------------------------------

    def _setup_headers(self) -> None:
        h = self.doc.header
        h["$INSUNITS"] = 4        # 毫米
        h["$LUNITS"] = 2          # 十进制
        h["$MEASUREMENT"] = 1     # 公制
        h["$ANGBASE"] = 0
        h["$ANGDIR"] = 0          # 逆时针为正（与 bulge 符号约定一致）

    def _register_linetypes(self) -> None:
        """注册自定义线型。必须在 _setup_layers 之前调用，否则图层引用失效。

        ezdxf 允许图层引用不存在的线型而不报错（audit() 也通过），所以这里显式校验。
        """
        table = self.doc.linetypes
        # 迭代 LinetypeTable 得到的是 Linetype 对象（不是名字）；`in` 才按名字判断。
        # DXF 线型名大小写不敏感（内置项是 "Continuous"，图层表里常写 "CONTINUOUS"），
        # 故统一按大写归一做存在性校验；写回时沿用文档里已有的拼写。
        self._linetype_names: set[str] = {e.dxf.name.upper() for e in table}
        for name, pattern, desc in _LINE_TYPES:
            if name.upper() in self._linetype_names:
                logger.debug("linetype %s 已存在，沿用文档原有定义", name)
                continue
            try:
                table.add(name, pattern=list(pattern), description=desc)
                self._linetype_names.add(name.upper())
            except Exception as exc:
                # 同名已存在（例如某些模板自带 CENTER）不视为失败，但必须记录
                self.warnings.append(f"linetype {name} 注册跳过: {exc}")
                self._linetype_names.add(name.upper())

    def _setup_layers(self) -> None:
        self._layer_names: set[str] = set()
        for name, color, ltype in _LAYER_DEFS:
            if ltype.upper() not in self._linetype_names:
                # 绝不引用不存在的线型：静默的无效引用会让线型在 CAD 里变成实线
                raise ValueError(
                    f"图层 {name} 引用的线型 {ltype!r} 未注册；"
                    f"已注册: {sorted(self._linetype_names)}"
                )
            self.doc.layers.add(name, color=color, linetype=ltype)
            self._layer_names.add(name)
        # 线宽档位图层兜底：若 LayerConfig 给了额外档位名，动态补上
        for name in self.layer_config.names:
            if name not in self._layer_names:
                self.doc.layers.add(name, color=7, linetype="CONTINUOUS")
                self._layer_names.add(name)

    def _require_layer(self, layer: str) -> str:
        """写出前校验图层存在：图层名拼错时 DXF 会静默自建一个默认图层。"""
        if layer not in self._layer_names:
            raise ValueError(f"图层 {layer!r} 不存在；已定义: {sorted(self._layer_names)}")
        return layer

    def _setup_styles(self) -> None:
        """中文支持：Standard 样式挂 gbcbig.shx 大字体。

        gbcbig.shx 是 CAD 侧字体文件名，本机不必存在；但设置本身失败就必须暴露，
        否则中文会变成问号而无人知晓。
        """
        st = self.doc.styles.get("Standard")
        st.dxf.font = "txt.shx"
        st.dxf.bigfont = "gbcbig.shx"

    # ----- 坐标归一化 --------------------------------------------------------

    def _to_cad(self, pts: np.ndarray) -> np.ndarray:
        p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        if not np.all(np.isfinite(p)):
            raise ValueError(f"坐标含 NaN/Inf，拒绝写出: {p!r}")
        # height_mm 在 __init__ 已强制提供，翻转不再可跳过
        p = flip_y(p, self.height_mm)
        if not np.allclose(self._m_world, np.eye(3)):
            p = apply_affine(p, self._m_world)
        return p

    @staticmethod
    def _pt(xy) -> tuple[float, float]:
        x, y = float(xy[0]), float(xy[1])
        return round4(x), round4(y)

    def _track(self, kind: str, entity, *, linewidth_mm: float | None = None,
               **geom: Any) -> None:
        """登记期望几何，供 save() 回读逐条比对。

        直接调用 add_* 与经 write_entities 两条路径都经此处，
        因此 _written（原始 Entity）与 _written_geom（写出后几何）始终同频。
        """
        self._written.append(Entity(kind=kind, layer=entity.dxf.layer,
                                    data=dict(geom), linewidth_mm=linewidth_mm,
                                    meta={"handle": entity.dxf.handle}))
        self._written_geom.append({
            "kind": kind,
            "handle": entity.dxf.handle,
            "layer": entity.dxf.layer,
            "linewidth_mm": linewidth_mm,
            **geom,
        })

    # ----- 实体写出 ----------------------------------------------------------

    def add_line(self, p1, p2, layer: str, linewidth_mm: float | None = None) -> None:
        self._require_layer(layer)
        cad = self._to_cad(np.vstack([p1, p2]))
        a, b = self._pt(cad[0]), self._pt(cad[1])
        ent = self.msp.add_line(a, b, dxfattribs={
            "layer": layer, "lineweight": lineweight_enum(linewidth_mm)})
        self._entities_written += 1
        self._track("LINE", ent, linewidth_mm=linewidth_mm, start=a, end=b)

    def add_circle(self, center, radius: float, layer: str = "CIRCLES",
                   linewidth_mm: float | None = None) -> None:
        self._require_layer(layer)
        c = self._to_cad(np.asarray(center, dtype=np.float64).reshape(1, 2))[0]
        cx, cy = self._pt(c)
        r = round4(float(radius))
        if not np.isfinite(r) or r <= 0.0:
            self.warnings.append(f"CIRCLE 半径<=0 或非有限，已丢弃: center={center} r={radius}")
            return
        ent = self.msp.add_circle((cx, cy), r, dxfattribs={
            "layer": layer, "lineweight": lineweight_enum(linewidth_mm)})
        self._entities_written += 1
        self._track("CIRCLE", ent, linewidth_mm=linewidth_mm, center=(cx, cy), radius=r)

    def add_arc(self, center, radius: float, start_angle: float, end_angle: float,
                layer: str = "ARCS", linewidth_mm: float | None = None) -> None:
        self._require_layer(layer)
        c = self._to_cad(np.asarray(center, dtype=np.float64).reshape(1, 2))[0]
        cx, cy = self._pt(c)
        r = round4(float(radius))
        if not np.isfinite(r) or r <= 0.0:
            self.warnings.append(f"ARC 半径<=0 或非有限，已丢弃: center={center} r={radius}")
            return
        # Y 翻转 (x,y)->(x,H-y) 把极角 θ 映射为 -θ，并反转扫掠方向。
        # DXF 的 ARC 只能表示逆时针弧，所以必须"取反 + 交换"才能描述镜像后的同一段弧：
        #   新 start = -原 end，新 end = -原 start
        # 只翻圆心不翻角度会把 90° 的弧写成 270° 的弧（张角完全变了）。
        # 世界矩阵若也含反射，两次镜像相抵，_reflected 已按 det 符号合并。
        sa_raw, ea_raw = float(start_angle), float(end_angle)
        if self._reflected:
            sa, ea = (-ea_raw) % 360.0, (-sa_raw) % 360.0
        else:
            sa, ea = sa_raw % 360.0, ea_raw % 360.0
        if abs((ea - sa) % 360.0) < 1e-9:
            self.warnings.append(f"ARC 起止角相同（扫掠角≈0），已丢弃: center={center}")
            return
        sa, ea = round4(sa), round4(ea)
        ent = self.msp.add_arc((cx, cy), r, sa, ea, dxfattribs={
            "layer": layer, "lineweight": lineweight_enum(linewidth_mm)})
        self._entities_written += 1
        self._track("ARC", ent, linewidth_mm=linewidth_mm,
                    center=(cx, cy), radius=r, start_angle=sa, end_angle=ea)

    def add_lwpolyline(self, points: Sequence, bulges: Sequence[float] | None = None,
                       layer: str = "LINES_W0.35", closed: bool = False,
                       linewidth_mm: float | None = None) -> None:
        """
        多段线写出。bulge 有符号：正=凸向行进方向左侧（逆时针弧），负=右侧。

        反射约定：bulge 的符号是相对行进方向的，形状必须保持"像素上看到的凸向"。
        坐标系含镜像（纯 Y 翻转）时行进方向反转，故 bulge 必须整体取反，
        否则每段弧都向反方向鼓出（外凸变内凹）。两次镜像相抵时不取反。
        调用方传入的 bulges 一律按**图像系**约定（与 points 同一坐标系），符号由本方法统一处理。

        ezdxf 1.4.4 没有 LWPolyline.set_bulge()，bulge 只能经 format="xyseb"
        的 5 元组顶点写入；用 format="xyb" 会被静默丢弃。
        """
        self._require_layer(layer)
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        cad = self._to_cad(pts)
        xy = [self._pt(cad[i]) for i in range(len(cad))]
        if len(xy) < 2:
            self.warnings.append("LWPOLYLINE 点数<2，已丢弃")
            return

        n_seg = len(xy) - 1 + (1 if closed and len(xy) > 2 else 0)
        bl = list(bulges or [])
        if len(bl) > n_seg:
            self.warnings.append(
                f"LWPOLYLINE bulge 数量 {len(bl)} > 段数 {n_seg}，多余部分被忽略"
            )
        sign = -1.0 if self._reflected else 1.0
        eff: list[float] = []
        for i in range(len(xy)):
            raw = bl[i] if i < len(bl) and _bulge_ok(bl[i]) else 0.0
            eff.append(round(float(raw) * sign, 12))

        verts = [(xy[i][0], xy[i][1], 0.0, 0.0, eff[i]) for i in range(len(xy))]
        attribs: dict[str, Any] = {"layer": layer}
        if linewidth_mm is not None:
            # LWPOLYLINE 没有受限的线宽枚举，用 const_width 直接写真实毫米线宽
            attribs["const_width"] = round4(float(linewidth_mm))
        else:
            attribs["lineweight"] = -1  # BYLAYER
        pl: LWPolyline = self.msp.add_lwpolyline(verts, format="xyseb", close=closed,
                                                 dxfattribs=attribs)
        self._entities_written += 1
        self._track("LWPOLYLINE", pl, linewidth_mm=linewidth_mm, points=xy,
                    bulges=eff, closed=bool(closed))

    def add_text(self, text: str, insert, height: float,
                 layer: str = "TEXT", rotation: float = 0.0) -> None:
        self._require_layer(layer)
        p = self._to_cad(np.asarray(insert, dtype=np.float64).reshape(1, 2))[0]
        x, y = self._pt(p)
        h = round4(float(height))
        rot = round4(float(rotation))
        # add_text() 没有 insert 参数，位置必须走 dxfattribs
        ent = self.msp.add_text(
            str(text),
            height=h,
            dxfattribs={"layer": layer, "rotation": rot, "insert": (x, y)},
        )
        self._entities_written += 1
        self._track("TEXT", ent, insert=(x, y), height=h, rotation=rot, text=str(text))

    def add_mtext(self, text: str, insert, height: float,
                  layer: str = "TEXT", width: float | None = None) -> None:
        self._require_layer(layer)
        p = self._to_cad(np.asarray(insert, dtype=np.float64).reshape(1, 2))[0]
        x, y = self._pt(p)
        h = round4(float(height))
        # 合法属性名是 insert；height/width 不是 DXF 属性，必须写 dxf.char_height / dxf.width
        mt = self.msp.add_mtext(str(text), dxfattribs={"layer": layer, "insert": (x, y)})
        mt.dxf.char_height = h
        w = None
        if width is not None and float(width) > 0.0:
            w = round4(float(width))
            mt.dxf.width = w
        self._entities_written += 1
        self._track("MTEXT", mt, insert=(x, y), char_height=h, width=w, text=str(text))

    def write_entities(self, entities: Iterable[Entity]) -> None:
        """批量写出，S8 阶段的唯一入口。

        错误分级（关键）：
        - KeyError/TypeError/ValueError/AttributeError/DXFAttributeError/DXFValueError
          = 数据问题，记 WARN 并跳过该实体（不产出错误几何，也不中断整批）；
        - 其它异常 = 程序缺陷（例如 API 用错），一律向上抛。
          绝不能把编程错误降级成 warning——那会让"少了一半图元"的 DXF 看起来是成功的。
        """
        expected_errors = (
            KeyError, TypeError, ValueError, AttributeError,
            ezdxf.lldxf.const.DXFAttributeError, ezdxf.lldxf.const.DXFValueError,
        )
        lc = self.layer_config
        for e in entities:
            layer = e.layer if e.layer != "__auto__" else lc.resolve(e.linewidth_mm)
            try:
                if e.kind == "LINE":
                    self.add_line(e.data["p1"], e.data["p2"], layer, e.linewidth_mm)
                elif e.kind == "CIRCLE":
                    self.add_circle(e.data["center"], e.data["radius"], layer, e.linewidth_mm)
                elif e.kind == "ARC":
                    self.add_arc(e.data["center"], e.data["radius"],
                                 e.data["start_angle"], e.data["end_angle"], layer, e.linewidth_mm)
                elif e.kind == "LWPOLYLINE":
                    self.add_lwpolyline(e.data["points"], e.bulges, layer,
                                        bool(e.data.get("closed", False)), e.linewidth_mm)
                elif e.kind == "TEXT":
                    self.add_text(e.data["text"], e.data["insert"],
                                  e.data.get("height", 2.5), layer, e.data.get("rotation", 0.0))
                elif e.kind == "MTEXT":
                    self.add_mtext(e.data["text"], e.data["insert"],
                                   e.data.get("height", 2.5), layer)
                else:
                    self.warnings.append(f"未知实体类型 {e.kind}，已跳过")
                    continue
            except expected_errors as exc:
                self.warnings.append(
                    f"实体写出失败 kind={e.kind} layer={layer} "
                    f"data={ {k: v for k, v in e.data.items() if k != 'points'} }: "
                    f"{type(exc).__name__}: {exc}"
                )

    # ----- 保存与自检 --------------------------------------------------------

    def save(self, filepath: str | Path, *,
             manifest_id: str = "", config_hash: str = "", rms_mm: float | None = None,
             debug: bool = False, project_name: str | None = None,
             write_metrics: bool = True) -> dict[str, Any]:
        """
        写出 -> 关闭 -> 回读自检（逐实体比对几何）-> 产出 metrics。
        自检失败：抛 DXFSelfCheckError（exit_code = 3，契约第 5 条）。

        `project_name`：显式指定 `$PROJECTNAME`（S8 的未配准水印用），默认仍写元数据串。
        `write_metrics=False`：由调用方（S8）自己写 `metrics_s8.json`（StageMetrics 格式），
        避免与本方法的报告互相覆盖。
        """
        t0 = time.perf_counter()
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        rms_txt = "未配准(uncalibrated)" if rms_mm is None else f"{rms_mm:.4f}mm"
        notice = ("uncalibrated output: not for fabrication/permitting/quantity takeoff"
                  if rms_mm is None else "")
        meta = f"R2V | manifest={manifest_id} | config={config_hash} | RMS={rms_txt}"
        if notice:
            meta += f" | {notice}"
        if self.warnings:
            meta += " | WARN: " + "; ".join(self.warnings[:5])

        # 元数据落点（实测可落盘并回读）。$COMMENTS 不是合法头变量，别再用。
        self.doc.header["$LASTSAVEDBY"] = f"R2V | {manifest_id} | {config_hash}"[:255]
        self.doc.header["$PROJECTNAME"] = (project_name or meta)[:255]
        try:
            self.doc.header.custom_vars.append("R2V_META", meta[:255])
        except Exception as exc:  # 自定义变量不是必须项，失败不致命但要留痕
            self.warnings.append(f"自定义头变量写入失败: {exc}")

        # 确定性写出（SPEC §3.3 比特级可复现）：固定元数据/GUID/CLASSES 顺序，只写 LF。
        # 详见 src/dxf_repro.py；saveas() 会引入时间戳、随机 GUID 与 Windows CRLF。
        from src.dxf_repro import save_deterministic

        save_deterministic(self.doc, path)
        report = self._self_check(path, expected=self._entities_written)
        report["warnings"] = self.warnings
        report["filepath"] = str(path)
        report["timing_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
        report["count_by_kind"] = {
            kind: sum(1 for e in self._written if e.kind == kind)
            for kind in sorted({e.kind for e in self._written})
        }
        report["error_estimate"] = {
            "coordinate_rounding_mm": 0.5 * 10 ** (-COORD_DECIMALS),
            "self_check_tolerance_mm": SELF_CHECK_TOL,
            "registration_rms_mm": rms_mm,
        }

        # 落盘指标（约定文件名 metrics_<stage>.json，S8 阶段）
        metrics_path = path.with_name("metrics_s8.json")
        if write_metrics:
            metrics_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

        if debug:
            self._write_debug_report(path.with_name("dxf_report.txt"), report)

        if not report["ok"]:
            raise DXFSelfCheckError(
                f"DXF 自检未通过（退出码 {EXIT_SELF_CHECK_FAILED}）："
                f"{report['mismatches'][:5]}；完整报告见 {metrics_path}"
            )
        return report

    def _self_check(self, filepath: Path, expected: int) -> dict[str, Any]:
        """回读比对：实体总数、逐类型计数、以及**每个实体的几何**。

        SELF_CHECK_TOL = 1e-3 mm 直接作用于坐标比对；不一致即为失败。
        """
        mismatches: list[str] = []
        counts: dict[str, int] = {}
        tol = SELF_CHECK_TOL
        sampled = 0
        max_dev = 0.0

        try:
            doc2 = ezdxf.readfile(str(filepath))
            ents = list(doc2.modelspace())
        except Exception as exc:
            raise DXFSelfCheckError(f"回读失败: {type(exc).__name__}: {exc}") from exc

        for t in ("LINE", "LWPOLYLINE", "ARC", "CIRCLE", "TEXT", "MTEXT"):
            counts[t] = sum(1 for e in ents if e.dxftype() == t)
        total = sum(counts.values())
        if total != expected:
            mismatches.append(f"实体总数不符 expect={expected} got={total}")

        by_handle = {e.dxf.handle: e for e in ents}

        def close(a, b) -> float:
            return abs(float(a) - float(b))

        # 1) 期望几何全部逐条比对
        for want in self._written_geom:
            got = by_handle.get(want["handle"])
            if got is None:
                mismatches.append(f"{want['kind']} handle={want['handle']} 在回读结果中不存在")
                continue
            if got.dxf.layer != want["layer"]:
                mismatches.append(
                    f"{want['kind']} handle={want['handle']} 图层 "
                    f"{got.dxf.layer!r} != {want['layer']!r}"
                )
            kind = want["kind"]
            if kind == "LINE":
                dev = max(close(got.dxf.start.x, want["start"][0]),
                          close(got.dxf.start.y, want["start"][1]),
                          close(got.dxf.end.x, want["end"][0]),
                          close(got.dxf.end.y, want["end"][1]))
            elif kind == "CIRCLE":
                dev = max(close(got.dxf.center.x, want["center"][0]),
                          close(got.dxf.center.y, want["center"][1]),
                          close(got.dxf.radius, want["radius"]))
            elif kind == "ARC":
                dev = max(close(got.dxf.center.x, want["center"][0]),
                          close(got.dxf.center.y, want["center"][1]),
                          close(got.dxf.radius, want["radius"]),
                          close(got.dxf.start_angle, want["start_angle"]),
                          close(got.dxf.end_angle, want["end_angle"]))
            elif kind == "LWPOLYLINE":
                pts = list(got.get_points(format="xyseb"))
                if len(pts) != len(want["points"]):
                    mismatches.append(
                        f"LWPOLYLINE handle={want['handle']} 顶点数 "
                        f"{len(pts)} != {len(want['points'])}"
                    )
                    continue
                dev = max(
                    max(close(p[0], q[0]), close(p[1], q[1]), close(p[4], b))
                    for p, q, b in zip(pts, want["points"], want["bulges"])
                )
            elif kind == "TEXT":
                dev = max(close(got.dxf.insert.x, want["insert"][0]),
                          close(got.dxf.insert.y, want["insert"][1]),
                          close(got.dxf.height, want["height"]))
                if got.dxf.text != want["text"]:
                    mismatches.append(
                        f"TEXT handle={want['handle']} 内容 {got.dxf.text!r} != {want['text']!r}"
                    )
            elif kind == "MTEXT":
                ch = got.dxf.get("char_height")
                dev = max(close(got.dxf.insert.x, want["insert"][0]),
                          close(got.dxf.insert.y, want["insert"][1]),
                          close(ch, want["char_height"]) if ch is not None else float("inf"))
                if got.text != want["text"]:
                    mismatches.append(
                        f"MTEXT handle={want['handle']} 内容 {got.text!r} != {want['text']!r}"
                    )
            else:  # pragma: no cover - 类型由 EntityKind 约束
                dev = 0.0
            sampled += 1
            max_dev = max(max_dev, dev)
            if dev > tol:
                mismatches.append(
                    f"{kind} handle={want['handle']} 几何偏差 {dev:.6g} > 容差 {tol}"
                )

        # 2) 文件里多出来的图元（自检必须能发现"凭空多出"）
        extra = set(by_handle) - {w["handle"] for w in self._written_geom}
        if extra:
            mismatches.append(f"回读结果多出 {len(extra)} 个未登记的实体: {sorted(extra)[:5]}")

        return {
            "ok": not mismatches,
            "entity_counts": counts,
            "total": total,
            "expected_total": expected,
            "checked_entities": sampled,
            "max_geometry_deviation_mm": max_dev,
            "mismatches": mismatches,
            "warnings_count": len(self.warnings),
        }

    @staticmethod
    def _write_debug_report(path: Path, report: dict[str, Any]) -> None:
        lines = ["=== R2V DXF 写出报告 ==="]
        for k, v in report.items():
            lines.append(f"{k}: {v}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _main(argv: Sequence[str] | None = None) -> int:
    """最小 CLI：`python -m src.dxf_writer [selftest]` 做一次冒烟自检。

    自检失败按契约第 5 条返回退出码 3。
    """
    import argparse

    parser = argparse.ArgumentParser(description="R2V DXF 写出模块冒烟自检")
    parser.add_argument("command", nargs="?", default="selftest", choices=["selftest"],
                        help="要执行的命令（默认 selftest）")
    parser.add_argument("--out", default=None, help="输出 .dxf 路径（默认临时目录）")
    parser.add_argument("--height-mm", type=float, default=100.0, help="图幅高度（mm）")
    args = parser.parse_args(argv)

    out = Path(args.out) if args.out else Path(tempfile.gettempdir()) / "r2v_selftest.dxf"
    writer = DXFWriter(image_height_mm=args.height_mm,
                       layer_config=LayerConfig(bins=(0.25, 0.42),
                                                names=("LINES_W0.18", "LINES_W0.35",
                                                       "LINES_W0.50")))
    writer.add_line((0, 90), (10, 90), "LINES_W0.35", linewidth_mm=0.35)
    writer.add_circle((20, 80), 5.0, linewidth_mm=0.18)
    writer.add_arc((40, 80), 5.0, 0.0, 90.0, linewidth_mm=0.18)
    writer.add_lwpolyline([(0, 60), (10, 60), (10, 70)], [0.5, 0.0],
                          layer="ARCS", linewidth_mm=0.5)
    writer.add_text("中文 ABC", (0, 50), 2.5)
    try:
        report = writer.save(out, manifest_id="SELFTEST", config_hash="local", debug=True)
    except DXFSelfCheckError as exc:
        print(f"SELFTEST FAILED: {exc}", file=sys.stderr)
        return EXIT_SELF_CHECK_FAILED
    print(json.dumps({k: v for k, v in report.items() if k != "warnings"},
                     ensure_ascii=False, indent=2))
    for w in report["warnings"]:
        print(f"WARN: {w}", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(_main())