# src/coords.py
"""坐标变换：像素 ↔ 毫米 ↔ CAD，全部用 3×3 齐次矩阵 float64（SPEC §3.1）。

铁律
----
1. 内部坐标永远是 **float64、单位 mm**；px↔mm 只在图像读写边界换算。
2. 图像系原点在左上、Y 向下；CAD 系原点在左下、Y 向上，**必须翻转**。
   禁止在别处手写 `h - y`：翻转只允许通过本模块的矩阵完成。
3. 每个变换都必须能求逆（把 CAD 标注叠回原图做 debug 叠加要用）。
4. 变换链：`px_img →(dpi)→ mm_img →(flip Y)→ mm_cad →(配准仿射)→ mm_world`。

用法
----
>>> t = Transform2D.image_to_cad(dpi=300.0, height_px=625)
>>> t.apply([0.0, 0.0])                      # 图像左上角 → CAD 左上角
array([  0.        , 52.91666...])
>>> t.inverse().apply(t.apply([100.0, 200.0]))
array([100., 200.])
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from src.config import mm_per_px

ArrayLike = Sequence[float] | np.ndarray


def _as_matrix(matrix: ArrayLike) -> np.ndarray:
    m = np.asarray(matrix, dtype=np.float64)
    if m.shape != (3, 3):
        raise ValueError(f"变换矩阵必须是 3×3，收到形状 {m.shape}")
    if not np.all(np.isfinite(m)):
        raise ValueError("变换矩阵包含 NaN/Inf")
    if not np.allclose(m[2], (0.0, 0.0, 1.0), atol=0.0, rtol=0.0):
        raise ValueError(f"只支持 2D 仿射（末行必须是 [0,0,1]），收到 {m[2]!r}")
    return m


@dataclass(frozen=True, eq=False)
class Transform2D:
    """2D 仿射变换（3×3 齐次，float64）。

    `eq=False` 是有意的：ndarray 的默认相等会返回数组并让断言语义含糊，
    这里改为逐元素比较的布尔相等。
    """

    matrix: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "matrix", _as_matrix(self.matrix))

    # ----- 构造 -------------------------------------------------------------
    @classmethod
    def identity(cls) -> "Transform2D":
        return cls(np.eye(3, dtype=np.float64))

    @classmethod
    def translation(cls, tx: float, ty: float) -> "Transform2D":
        m = np.eye(3, dtype=np.float64)
        m[0, 2] = tx
        m[1, 2] = ty
        return cls(m)

    @classmethod
    def scale(cls, sx: float, sy: float | None = None) -> "Transform2D":
        if sy is None:
            sy = sx
        if sx == 0 or sy == 0:
            raise ValueError("缩放系数不能为 0（矩阵不可逆）")
        m = np.eye(3, dtype=np.float64)
        m[0, 0] = sx
        m[1, 1] = sy
        return cls(m)

    @classmethod
    def rotation_deg(cls, angle_deg: float, pivot: tuple[float, float] = (0.0, 0.0)) -> "Transform2D":
        a = np.radians(float(angle_deg))
        c, s = float(np.cos(a)), float(np.sin(a))
        rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        px, py = pivot
        return cls.translation(px, py) @ cls(rot) @ cls.translation(-px, -py)

    @classmethod
    def flip_y(cls, height_mm: float) -> "Transform2D":
        """图像系 → CAD 系的 Y 翻转（关于 y = height/2 镜像）。"""
        if height_mm <= 0:
            raise ValueError(f"height_mm 必须 > 0，收到 {height_mm}")
        return cls(np.array([[1.0, 0.0, 0.0], [0.0, -1.0, height_mm], [0.0, 0.0, 1.0]]))

    @classmethod
    def image_to_cad(cls, *, dpi: float, height_px: int | float) -> "Transform2D":
        """图像像素系（左上原点、Y 下）→ CAD 毫米系（左下原点、Y 上）。"""
        if height_px <= 0:
            raise ValueError(f"height_px 必须 > 0，收到 {height_px}")
        mp = mm_per_px(dpi)
        height_mm = float(height_px) * mp
        return cls.flip_y(height_mm) @ cls.scale(mp)

    # ----- 运算 -------------------------------------------------------------
    def compose(self, other: "Transform2D") -> "Transform2D":
        """返回 `self ∘ other`（先应用 other，再应用 self）。"""
        return Transform2D(self.matrix @ other.matrix)

    def __matmul__(self, other: "Transform2D") -> "Transform2D":
        if not isinstance(other, Transform2D):
            raise TypeError(f"只支持 Transform2D 相乘，收到 {type(other)!r}")
        return self.compose(other)

    def inverse(self) -> "Transform2D":
        det = float(np.linalg.det(self.matrix[:2, :2]))
        if abs(det) < 1e-15:
            raise ValueError(f"变换不可逆（det={det:.3e}）")
        return Transform2D(np.linalg.inv(self.matrix))

    def apply(self, points: ArrayLike) -> np.ndarray:
        """应用变换。(2,) → (2,)；(N,2) → (N,2)。"""
        pts = np.asarray(points, dtype=np.float64)
        single = pts.ndim == 1
        if single:
            if pts.shape != (2,):
                raise ValueError(f"单点必须是 (2,)，收到 {pts.shape}")
            pts = pts.reshape(1, 2)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError(f"点集必须是 (N,2)，收到 {pts.shape}")
        if not np.all(np.isfinite(pts)):
            raise ValueError("点集包含 NaN/Inf")
        homo = np.column_stack([pts, np.ones(len(pts), dtype=np.float64)])
        out = homo @ self.matrix.T
        out = out[:, :2]
        return out[0] if single else out

    # ----- 查询 -------------------------------------------------------------
    @property
    def scale_factors(self) -> tuple[float, float]:
        """每轴的缩放系数（列向量范数）；各向异性检测用。"""
        return (
            float(np.linalg.norm(self.matrix[:2, 0])),
            float(np.linalg.norm(self.matrix[:2, 1])),
        )

    @property
    def is_axis_aligned(self) -> bool:
        """是否只含缩放 + 平移（无旋转/剪切）：非对角线元素为 0。"""
        m = self.matrix
        return bool(np.isclose(m[0, 1], 0.0) and np.isclose(m[1, 0], 0.0))

    def to_list(self) -> list[list[float]]:
        """JSON 可序列化形式（3×3，行优先）。"""
        return [[float(v) for v in row] for row in self.matrix]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Transform2D):
            return NotImplemented
        return bool(np.array_equal(self.matrix, other.matrix))

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return f"Transform2D({np.array2string(self.matrix, precision=6, separator=', ')})"


def chain(*transforms: Transform2D) -> Transform2D:
    """按顺序复合：`chain(a, b, c)` = a ∘ b ∘ c（先 c，再 b，最后 a）。"""
    if not transforms:
        return Transform2D.identity()
    out = transforms[0]
    for t in transforms[1:]:
        out = out.compose(t)
    return out


def quantize(values: ArrayLike, decimals: int) -> np.ndarray:
    """按 DXF 写出位宽取整（SPEC §4 S8：4 位小数 = 0.0001 mm）。"""
    if decimals < 0:
        raise ValueError(f"decimals 必须 >= 0，收到 {decimals}")
    arr = np.asarray(values, dtype=np.float64)
    return np.round(arr, decimals)


__all__ = ["Transform2D", "chain", "quantize"]
