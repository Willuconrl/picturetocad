# src/config.py
"""输入契约与运行参数的 pydantic schema —— 全项目唯一的校验入口。

本模块当前承载 **输入清单契约**（`test_images/manifest.csv`），后续会扩展为
`config.yaml` 的运行时 schema（pydantic-settings）。约束：

1. 非法值必须**在加载时抛错**（`ManifestError`），不允许静默补默认值、截断或跳过行。
2. 参数不硬编码：误差预算基准 DPI 等一律由 `config.yaml` 传入（见 `dpi_budget_scale`）。
3. 校验错误必须带**行号与列名**，否则无法定位坏数据。

manifest.csv 列（用户决策 #1，7 列，顺序即契约）：

    file, preset, true_dpi, page_class, golden_dxf, notes, has_golden

`golden_dxf` 与 `has_golden` 的语义分开：
- `has_golden=true`  → `golden_dxf` 必填，且 `golden_dir/golden_dxf` 必须存在；
- `has_golden=false` → 允许**预留**目标文件名（表示「将来要描但还没描」），文件不要求存在。
  这条是真实需求：QDIZE9403.JPG 这一行就写着 `QDIZE9403.dxf` 但 `has_golden=false`。
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

#: config.yaml 四套 preset 的名字（真相源见 .dsh/skills/r2v-domain/presets.yaml）
PRESET_NAMES: tuple[str, ...] = ("topo-map", "blueprint", "mechanical", "sketch")

#: manifest.csv 的列顺序即契约，顺序漂移视为错误
MANIFEST_COLUMNS: tuple[str, ...] = (
    "file",
    "preset",
    "true_dpi",
    "page_class",
    "golden_dxf",
    "notes",
    "has_golden",
)

PresetName = Literal["topo-map", "blueprint", "mechanical", "sketch"]


class ManifestError(ValueError):
    """manifest.csv 结构或内容非法。消息里必须能定位到行/列。"""


class ManifestRow(BaseModel):
    """manifest.csv 的一行。"""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    file: str = Field(min_length=1, description="test_images/ 下的文件名（不含路径）")
    preset: PresetName | None = Field(default=None, description="套用哪套 preset；空=纯 base")
    true_dpi: float = Field(gt=0, description="图像真实 DPI（不是 EXIF 宣称值；截图填实际）")
    page_class: str = Field(min_length=1, description="图纸类别，如 mechanical_part / synthetic")
    golden_dxf: str | None = Field(default=None, description="golden/ 下的 DXF 文件名")
    notes: str = Field(default="", description="自由备注，必须写明来源与已知限制")
    has_golden: bool = Field(default=False, description="golden DXF 是否已经存在")

    @field_validator("preset", mode="before")
    @classmethod
    def _blank_preset(cls, v: object) -> object:
        if isinstance(v, str) and not v.strip():
            return None
        return v

    @field_validator("golden_dxf", mode="before")
    @classmethod
    def _blank_golden(cls, v: object) -> object:
        if isinstance(v, str) and not v.strip():
            return None
        return v

    @field_validator("file")
    @classmethod
    def _flat_filename(cls, v: str) -> str:
        if "/" in v or "\\" in v:
            raise ValueError("file 只能是 test_images/ 下的文件名，不许带路径")
        if v in (".", ".."):
            raise ValueError("file 非法")
        return v

    @field_validator("golden_dxf")
    @classmethod
    def _relative_dxf(cls, v: str | None) -> str | None:
        if v is None:
            return None
        if "/" in v or "\\" in v:
            raise ValueError("golden_dxf 只能是 golden/ 下的文件名，不许带路径")
        if not v.lower().endswith(".dxf"):
            raise ValueError(f"golden_dxf 必须是 .dxf 文件，收到 {v!r}")
        return v

    @model_validator(mode="after")
    def _golden_consistency(self) -> "ManifestRow":
        if self.has_golden and self.golden_dxf is None:
            raise ValueError("has_golden=true 时必须填 golden_dxf")
        return self


class Manifest(BaseModel):
    """整个 manifest.csv。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rows: tuple[ManifestRow, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_files(self) -> "Manifest":
        seen: dict[str, int] = {}
        for idx, row in enumerate(self.rows):
            key = row.file.lower()  # Windows 下大小写不敏感，按不敏感判重
            if key in seen:
                raise ValueError(f"file 重复：{row.file!r}（第 {seen[key] + 1} 行与第 {idx + 1} 行）")
            seen[key] = idx
        return self

    def row_for(self, filename: str) -> ManifestRow:
        for row in self.rows:
            if row.file.lower() == filename.lower():
                return row
        raise KeyError(f"manifest 里没有 {filename!r}")

    @property
    def with_golden(self) -> tuple[ManifestRow, ...]:
        return tuple(r for r in self.rows if r.has_golden)


def load_manifest(
    path: Path,
    *,
    images_dir: Path | None = None,
    golden_dir: Path | None = None,
    require_files: bool = True,
) -> Manifest:
    """读取并校验 manifest.csv。

    参数：
        path         manifest.csv 路径
        images_dir   存在性校验用；require_files=True 时必填
        golden_dir   has_golden=true 的行用该校验 golden 是否存在
        require_files  True = 逐个确认文件存在（默认；CI 用它防「登记了却不在」）

    抛 ManifestError（带行号/列名），绝不返回部分结果。
    """
    path = Path(path)
    if not path.is_file():
        raise ManifestError(f"manifest 不存在：{path}")
    if require_files and images_dir is None:
        raise ManifestError("require_files=True 时必须提供 images_dir")
    images_dir = Path(images_dir) if images_dir is not None else None
    golden_dir = Path(golden_dir) if golden_dir is not None else None

    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        header = tuple(reader.fieldnames or ())
        if header != MANIFEST_COLUMNS:
            raise ManifestError(
                f"{path.name} 表头不符：期望 {MANIFEST_COLUMNS}，实际 {header}"
                "（列顺序即契约，改动必须同步 src/config.py:MANIFEST_COLUMNS）"
            )
        raw_rows = list(reader)

    rows: list[ManifestRow] = []
    for lineno, raw in enumerate(raw_rows, start=2):  # 第 1 行是表头
        cleaned = {(k or "").strip(): (v.strip() if isinstance(v, str) else v) for k, v in raw.items()}
        cleaned.pop(None, None)  # DictReader 对多余列会塞 None 键
        try:
            rows.append(ManifestRow(**cleaned))
        except ValidationError as exc:
            details = "; ".join(
                f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()
            )
            raise ManifestError(f"{path.name}:{lineno} 行非法 → {details}") from exc

    try:
        manifest = Manifest(rows=tuple(rows))
    except ValidationError as exc:
        details = "; ".join(f"{err['msg']}" for err in exc.errors())
        raise ManifestError(f"{path.name} 整体非法 → {details}") from exc

    if require_files:
        assert images_dir is not None  # 上面已校验
        for row in manifest.rows:
            image = images_dir / row.file
            if not image.is_file():
                raise ManifestError(f"{path.name}: file 指向的图像不存在：{image}")
            if row.has_golden:
                if golden_dir is None:
                    raise ManifestError(
                        f"{path.name}: 行 {row.file!r} 的 has_golden=true，但未提供 golden_dir 无法校验"
                    )
                golden = golden_dir / str(row.golden_dxf)
                if not golden.is_file():
                    raise ManifestError(f"{path.name}: 行 {row.file!r} 的 golden 不存在：{golden}")
    return manifest


def dpi_budget_scale(true_dpi: float, reference_dpi: float) -> float:
    """误差预算放大系数 = reference_dpi / true_dpi（用户决策 #3）。

    SPEC §1.2 的 mm 级预算以 300 DPI 为基准；低 DPI 图上 1 px 对应更大的 mm，
    因此同一「px 级」误差换算到 mm 时按此系数放大。参考 DPI 由
    `config.yaml: qc.budget_reference_dpi` 传入，**此处不写死**。
    """
    if true_dpi <= 0:
        raise ValueError(f"true_dpi 必须 > 0，收到 {true_dpi}")
    if reference_dpi <= 0:
        raise ValueError(f"reference_dpi 必须 > 0，收到 {reference_dpi}")
    return float(reference_dpi) / float(true_dpi)


def mm_per_px(dpi: float) -> float:
    """1 像素 = 多少毫米。300 DPI → 0.084666…；72 DPI → 0.352777…"""
    if dpi <= 0:
        raise ValueError(f"dpi 必须 > 0，收到 {dpi}")
    return 25.4 / float(dpi)


__all__: Sequence[str] = (
    "MANIFEST_COLUMNS",
    "PRESET_NAMES",
    "Manifest",
    "ManifestError",
    "ManifestRow",
    "dpi_budget_scale",
    "load_manifest",
    "mm_per_px",
)
