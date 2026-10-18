# src/__init__.py
"""r2v-tool：光栅图纸转 CAD（DXF）矢量化工具。"""

from __future__ import annotations

__version__ = "0.2.0-dev"

#: 写进 DXF/清单的能力边界声明（SPEC §0，必须出现在 CLI --help、GUI 首页、DXF 注释头）
CAPABILITY_NOTICE = (
    "输出为输入图像的几何近似，不是对现实物体的测量。"
    "未配准（无控制点/无比例尺）时尺寸不可信，不得用于加工、报建、算量、结算。"
)
