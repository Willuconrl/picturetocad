# src/config.py
"""输入契约与运行参数的 pydantic schema —— 全项目唯一的校验入口。

本模块承载两件事：
1. **输入清单契约**：`test_images/manifest.csv`（`ManifestRow` / `Manifest` / `load_manifest`）；
2. **运行参数 schema**：`config.yaml`（`R2VConfig` 及其分节模型 / `load_config`），
   由 pydantic-settings 承载，支持 `R2V_<SECTION>__<FIELD>` 环境变量覆盖。

约束：

1. 非法值必须**在加载时抛错**（`ManifestError` / `ConfigError`），不允许静默补默认值、
   截断或跳过行；错误信息必须带**文件、行号或字段路径**，否则无法定位坏数据。
2. 参数不硬编码：误差预算基准 DPI、保留位数等一律由 `config.yaml` 传入
   （见 `dpi_budget_scale`），代码里只允许出现物理常数。
3. `extra="forbid"`：拼错字段名必须失败，不允许被吞成"默认值"。

依赖选择说明（用户决策 #2，别改回去）
------------------------------------
图像处理用 **opencv-python-headless** 而不是 opencv-python：本工具是 Linux 服务器上的
批处理 CLI，没有 GUI；完整版会拽进 libGL/X11 依赖，在无显卡的容器里直接 import 失败。
headless 只是去掉 highgui 窗口，图像算法完全相同，debug 叠加图走 PNG 落盘。
DPI 元数据解析用 **Pillow**（cv2 读 EXIF/pHYs 不可靠，而 SPEC §3.2 要求精确 DPI）。

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
from typing import Any, Literal, Sequence

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

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


# ---------------------------------------------------------------------------
# 运行参数 schema：config.yaml（全部字段 **必填**，缺键即报错，不设代码内默认值）
# ---------------------------------------------------------------------------
Profile = Literal["precision", "balanced", "fast"]
DpiMode = Literal["reject", "warn", "degrade", "ignore"]
DenoiseMethod = Literal["nlmeans", "bilateral", "none"]
BackgroundMethod = Literal["morph_open", "polynomial", "gaussian_ilr"]
UpsampleKernel = Literal["lanczos"]
BinarizeMethod = Literal["sauvola", "otsu", "niblack", "local_gaussian"]
SubpixelMethod = Literal["gradient_normal_quad", "edge_canny_subpix"]
Interpolation = Literal["quadratic", "parabolic_fit"]
ThinningMethod = Literal["zhang_suen", "lee", "parallel"]
ClusterMethod = Literal["dbscan", "kde"]
LayerPriorityItem = Literal["linewidth", "orientation", "color", "position"]
OcrEngine = Literal["paddle", "tesseract", "none"]
RegistrationMode = Literal["none", "scale", "two-point", "affine", "projective"]
DxfVersion = Literal["R2018", "R2013", "R2010", "R2007", "R2004", "R2000"]
LineTypeName = Literal["CONTINUOUS", "DASHED", "HIDDEN"]


class ConfigError(ValueError):
    """config.yaml 读取或校验失败。消息必须能定位到字段路径。"""


class _Section(BaseModel):
    """所有分节模型的共同配置：未知键必须失败。"""

    model_config = ConfigDict(extra="forbid", frozen=True)


class MetaConfig(_Section):
    schema_version: str = Field(..., pattern=r"^\d+\.\d+$")
    profile: Profile
    seed: int = Field(..., ge=0)
    fail_fast: bool


class ResourcesConfig(_Section):
    max_memory_gb: float = Field(..., gt=0)
    tile_size_px: int = Field(..., gt=0)
    tile_overlap_factor: float = Field(..., gt=0)
    seam_blend_px: int = Field(..., ge=0)
    workers: int = Field(..., description="-1 = CPU 核心数")
    dpi_cap: int = Field(..., gt=0)

    @field_validator("workers")
    @classmethod
    def _workers(cls, v: int) -> int:
        if v == 0 or v < -1:
            raise ValueError(f"workers 必须是 -1 或正整数，收到 {v}")
        return v


class QcConfig(_Section):
    dpi_min: float = Field(..., gt=0)
    dpi_mode: DpiMode
    fallback_dpi: float = Field(..., gt=0, description="元数据缺失时的兜底 DPI（SPEC §1.1：默认 300 并 WARN）")
    budget_reference_dpi: float = Field(..., gt=0)
    sharpness_min: float = Field(..., ge=0)
    skew_max_deg: float = Field(..., ge=0, lt=90)
    skew_search_deg: float = Field(..., gt=0, lt=45, description="投影法搜索范围（±）")
    skew_prominence_min: float = Field(..., gt=0, description="投影峰显著性下限，低于则判不可信")
    perspective_max_deviation: float = Field(..., ge=0)
    jpeg_ring_artifact_detect: bool
    jpeg_blockiness_max: float = Field(..., gt=1.0, description="块边界/内部梯度比上限（仅 JPEG 输入）")
    moire_detect: bool
    moire_dot_density_max: float = Field(..., gt=0, description="每 1e4 px² 的小圆点连通域数上限")
    shadow_block_size_ratio: float = Field(..., gt=0)
    shadow_var_threshold: float = Field(..., ge=0)
    stain_area_min_mm2: float = Field(..., gt=0)
    allow_color_input: bool


class BackgroundFlattenConfig(_Section):
    enable: bool | Literal["auto"]
    method: BackgroundMethod
    kernel_scale_linewidths: float = Field(..., gt=0)
    polynomial_degree: int = Field(..., ge=1, le=6)


class DenoiseConfig(_Section):
    method: DenoiseMethod
    nlmeans_h: float = Field(..., ge=0, le=0.5)
    nlmeans_patch_px: int = Field(..., gt=0)
    nlmeans_search_px: int = Field(..., gt=0)
    edge_preserve: bool

    @model_validator(mode="after")
    def _search_covers_patch(self) -> "DenoiseConfig":
        if self.nlmeans_search_px < self.nlmeans_patch_px:
            raise ValueError(
                f"nlmeans_search_px({self.nlmeans_search_px}) 必须 >= nlmeans_patch_px({self.nlmeans_patch_px})"
            )
        return self


class UpsampleConfig(_Section):
    enable: bool
    min_linewidth_px: float = Field(..., gt=0)
    factor: Literal[2, 3]
    kernel: UpsampleKernel


class PreprocessConfig(_Section):
    gray_weights: Literal["bt709"]
    channel_independent: bool
    background_flatten: BackgroundFlattenConfig
    denoise: DenoiseConfig
    invert_auto: bool
    invert_foreground_ratio: float = Field(..., gt=0, lt=1)
    clip_border_px: int = Field(..., ge=0)
    upsample_if_thin: UpsampleConfig


class SauvolaConfig(_Section):
    window_ratio: float = Field(..., gt=0, lt=1)
    k: float = Field(..., gt=0, lt=1)
    r: float = Field(..., gt=0)


class NiblackConfig(_Section):
    window_ratio: float = Field(..., gt=0, lt=1)
    k: float


class LocalGaussianConfig(_Section):
    window_ratio: float = Field(..., gt=0, lt=1)
    offset: float


class BinarizePostConfig(_Section):
    fill_holes: bool
    hole_max_mm2: float = Field(..., gt=0)
    remove_small_objects_mm2: float = Field(..., ge=0)
    thin_line_repair: bool
    break_touching_text: bool


class SubpixelConfig(_Section):
    enable: bool
    method: SubpixelMethod
    gradient_sigma_px: float = Field(..., gt=0)
    interpolation: Interpolation
    min_gradient_mag: float = Field(..., ge=0, le=1)
    corner_preserve: bool


class BinarizeLinewidthConfig(_Section):
    method: Literal["distance_transform_ridge", "steger"]
    robust: Literal["median_mad", "mean_std"]
    per_edge_reestimate: bool


class BinarizeConfig(_Section):
    methods: tuple[BinarizeMethod, ...] = Field(..., min_length=1)
    selector_metric: Literal["foreground_stability"]
    sauvola: SauvolaConfig
    niblack: NiblackConfig
    local_gaussian: LocalGaussianConfig
    post: BinarizePostConfig
    subpixel: SubpixelConfig
    linewidth: BinarizeLinewidthConfig

    @field_validator("methods")
    @classmethod
    def _unique_methods(cls, v: tuple[BinarizeMethod, ...]) -> tuple[BinarizeMethod, ...]:
        if len(set(v)) != len(v):
            raise ValueError(f"binarize.methods 有重复项：{v}")
        return v


class ThinningConfig(_Section):
    method: ThinningMethod
    iterations_limit: int = Field(..., ge=0, description="0 = 自动")


class MedialAxisConfig(_Section):
    use: bool
    prune_by_radius: bool


class SpurRemovalConfig(_Section):
    enable: bool
    length_threshold_linewidths: float = Field(..., gt=0)
    length_threshold_px: None = Field(..., description="永远由 length_threshold_linewidths × 本地线宽派生")
    endpoint_curvature_check: bool

    @field_validator("length_threshold_px")
    @classmethod
    def _derived_only(cls, v: object) -> object:
        if v is not None:
            raise ValueError(
                "spur_removal.length_threshold_px 必须为 null：它永远由 "
                "length_threshold_linewidths × 本地线宽派生，手写绝对值会让阈值跨 DPI 失效"
            )
        return v


class NodeConfig(_Section):
    snap_distance_linewidths: float = Field(..., gt=0)
    classify: bool
    angle_tol_deg: float = Field(..., gt=0, lt=180)
    redundant_collapse: bool
    collinear_tol_px: float = Field(..., ge=0)


class BridgeConfig(_Section):
    enable: bool
    max_gap_linewidths: float = Field(..., gt=0)
    max_angle_deg: float = Field(..., ge=0, lt=90)
    mark_bridged: bool
    break_by_text_aware: bool


class SkeletonAcceptanceConfig(_Section):
    node_type_accuracy_min: float = Field(..., gt=0, le=1)


class SkeletonConfig(_Section):
    dual_path: bool
    thinning: ThinningConfig
    medial_axis: MedialAxisConfig
    spur_removal: SpurRemovalConfig
    node: NodeConfig
    bridge: BridgeConfig
    acceptance: SkeletonAcceptanceConfig


class FittingConfig(_Section):
    tolerance_mm: float = Field(..., gt=0)
    dp_epsilon_px: float | None
    arc_residual_mm: float = Field(..., gt=0, description="分段/圆弧残差上限（绝对毫米，M8-4）")
    arc_residual_factor: float | None = Field(default=None, description="[DEPRECATED] 旧口径，设值会告警")
    arc_residual_px: float | None = Field(default=None, description="[DEPRECATED] 由 arc_residual_mm 换算")
    prefer_arcs: bool
    model_select_alpha: float = Field(..., gt=0, lt=1, description="F 检验显著性水平（模型竞争）")
    full_circle_min_coverage_deg: float = Field(..., gt=0, le=360)
    full_circle_close_tolerance_mm: float = Field(..., gt=0)
    min_segment_length_linewidths: float = Field(..., gt=0)
    collinear_merge_tolerance_mm: float = Field(..., gt=0)
    min_arc_radius_linewidths: float = Field(..., gt=0)
    concentric_tolerance_mm: float = Field(..., gt=0)
    g1_tangent_tolerance_deg: float = Field(..., ge=0, lt=90)
    max_segments_per_curve: int = Field(..., gt=0)


class BorderConfig(_Section):
    detect: bool
    max_rect_area_ratio: float = Field(..., gt=0, le=1)
    parallel_inset_tolerance_mm: float = Field(..., gt=0)
    layer: str = Field(..., min_length=1)


class LinewidthConfig(_Section):
    cluster_method: ClusterMethod
    cluster_levels_min: int = Field(..., ge=1)
    cluster_levels_max: int = Field(..., ge=1)
    cluster_eps_mm: float = Field(..., gt=0)
    snap_to_standard: bool
    standard_widths_mm: tuple[float, ...] = Field(..., min_length=1)
    layer_name_decimals: int = Field(..., ge=0, le=4)
    encode_width_in_layer: bool
    layer_priority: tuple[LayerPriorityItem, ...] = Field(..., min_length=1)
    orthogonalize_tolerance_deg: float = Field(..., ge=0, lt=45)
    fallback_polyline_ratio: float = Field(..., ge=0, le=1)
    border: BorderConfig

    @model_validator(mode="after")
    def _levels_and_widths(self) -> "LinewidthConfig":
        if self.cluster_levels_min > self.cluster_levels_max:
            raise ValueError(
                f"cluster_levels_min({self.cluster_levels_min}) 不能大于 "
                f"cluster_levels_max({self.cluster_levels_max})"
            )
        widths = list(self.standard_widths_mm)
        if widths != sorted(widths):
            raise ValueError(f"standard_widths_mm 必须升序：{widths}")
        if len(set(widths)) != len(widths):
            raise ValueError(f"standard_widths_mm 有重复值：{widths}")
        if any(w <= 0 for w in widths):
            raise ValueError("standard_widths_mm 必须全部 > 0")
        if len(set(self.layer_priority)) != len(self.layer_priority):
            raise ValueError(f"layer_priority 有重复项：{self.layer_priority}")
        return self


class TextSeparationConfig(_Section):
    area_max_mm2: float = Field(..., gt=0)
    aspect_ratio_max: float = Field(..., gt=1)
    min_area_px: int = Field(..., ge=1, description="S6 候选区最小连通域面积（px²）")
    dilate_linewidth_factor: float = Field(..., ge=0, description="基元膨胀半径 = 线宽×本系数")
    dilate_extra_px: float = Field(..., ge=0, description="基元膨胀的额外像素")
    stroke_width_cv_max: float = Field(..., ge=0, le=1)
    distance_to_skeleton_mm: float = Field(..., ge=0)


class LineGroupingConfig(_Section):
    xheight_band_ratio: float = Field(..., gt=1)
    skew_correct: bool
    vertical_cjk: bool
    keep_rotation: bool


class OcrConfig(_Section):
    engine: OcrEngine
    fallback_engine: OcrEngine
    lang: tuple[str, ...] = Field(..., min_length=1)
    text_confidence_min: float = Field(..., ge=0, le=1)
    anchor: Literal["baseline_left"]
    descender_ratio: float = Field(..., ge=0, le=1, description="基线 = 行框下沿 − 比例×行高")
    height_from_xheight: bool
    recognize_dimensions: bool
    recognize_tolerance: bool
    recognize_diameter_symbol: bool


class TextConfig(_Section):
    enable: bool
    min_height_mm: float = Field(..., gt=0)
    max_height_mm: float = Field(..., gt=0)
    separation: TextSeparationConfig
    line_grouping: LineGroupingConfig
    ocr: OcrConfig
    gap_by_text: bool

    @model_validator(mode="after")
    def _height_range(self) -> "TextConfig":
        if self.min_height_mm >= self.max_height_mm:
            raise ValueError(
                f"text.min_height_mm({self.min_height_mm}) 必须小于 max_height_mm({self.max_height_mm})"
            )
        return self


class RegistrationConfig(_Section):
    mode: RegistrationMode
    scale_denominator: float | None
    two_point_min_distance_mm: float = Field(..., gt=0)
    affine_min_control_points: int = Field(..., ge=3)
    projective_min_control_points: int = Field(..., ge=4)
    rms_warn_mm: float = Field(..., gt=0)
    rms_max_mm: float = Field(..., gt=0)
    outlier_sigma: float = Field(..., gt=0)
    projective_requires_perspective_accept: bool
    uncalibrated_watermark: bool

    @model_validator(mode="after")
    def _consistency(self) -> "RegistrationConfig":
        if self.mode == "scale":
            if self.scale_denominator is None:
                raise ValueError("registration.mode=scale 时必须给 scale_denominator（1:N 的 N）")
            if self.scale_denominator <= 0:
                raise ValueError(f"scale_denominator 必须 > 0，收到 {self.scale_denominator}")
        elif self.scale_denominator is not None:
            raise ValueError(
                f"registration.scale_denominator 只在 mode=scale 时有意义"
                f"（当前 mode={self.mode}），请置 null"
            )
        if self.rms_warn_mm >= self.rms_max_mm:
            raise ValueError(
                f"rms_warn_mm({self.rms_warn_mm}) 必须小于 rms_max_mm({self.rms_max_mm})"
            )
        if self.projective_min_control_points <= self.affine_min_control_points:
            raise ValueError(
                "projective_min_control_points 必须大于 affine_min_control_points"
                f"（{self.projective_min_control_points} <= {self.affine_min_control_points}）"
            )
        return self


class LayerDefinition(_Section):
    name: str = Field(..., min_length=1, max_length=255)
    color: int = Field(..., ge=0, le=256)
    linetype: LineTypeName


class DxfConfig(_Section):
    version: DxfVersion
    library: Literal["ezdxf"]
    insunits: Literal[4] = Field(..., description="4 = 毫米；本工具只输出 mm")
    lunits: int = Field(..., ge=0, le=5)
    measurement: Literal[1]
    angle_units: int = Field(..., ge=0, le=5)
    coordinate_decimals: int = Field(..., ge=4, le=8)
    angle_decimals: int = Field(..., ge=0, le=8)
    verify_tolerance_mm: float = Field(..., gt=0)
    verify_on_write: bool
    layer_by_linewidth: bool
    orthogonalize: bool
    detect_titleblock: bool
    detect_centerlines: bool
    detect_symmetry_axis: bool
    debug_layer_color: int = Field(..., ge=0, le=256)
    layers: tuple[LayerDefinition, ...] = Field(..., min_length=1)
    comments_include: tuple[str, ...] = Field(..., min_length=1)

    @model_validator(mode="after")
    def _unique_layers(self) -> "DxfConfig":
        names = [layer.name for layer in self.layers]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ValueError(f"dxf.layers 图层名重复：{dupes}")
        return self


class PresetsConfig(_Section):
    file: str = Field(..., min_length=1)
    default: str | None
    validate_after_apply: bool
    allow_unknown_keys: bool
    record_effective: bool


class R2VConfig(BaseSettings):
    """config.yaml 的根模型。环境变量覆盖：`R2V_QC__DPI_MODE=warn` 之类。"""

    model_config = SettingsConfigDict(
        extra="forbid", frozen=True, env_prefix="R2V_", env_nested_delimiter="__"
    )

    meta: MetaConfig
    resources: ResourcesConfig
    qc: QcConfig
    preprocess: PreprocessConfig
    binarize: BinarizeConfig
    skeleton: SkeletonConfig
    fitting: FittingConfig
    linewidth: LinewidthConfig
    text: TextConfig
    registration: RegistrationConfig
    dxf: DxfConfig
    presets: PresetsConfig

    @model_validator(mode="after")
    def _preset_default_is_known(self) -> "R2VConfig":
        if self.presets.default is not None and self.presets.default not in PRESET_NAMES:
            raise ValueError(
                f"presets.default={self.presets.default!r} 不是已知 preset；"
                f"可选 {list(PRESET_NAMES)}"
            )
        return self


def _format_validation_error(exc: ValidationError, source: str) -> str:
    lines: list[str] = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"]) or "<root>"
        lines.append(f"  {loc}: {err['msg']}")
    return f"{source} 校验失败（{len(lines)} 处）：\n" + "\n".join(lines)


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    """读取 YAML 顶层映射。空文件/非映射/语法错误都抛 ConfigError。"""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"配置文件不存在：{path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path.name} YAML 解析失败：{exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path.name} 顶层必须是映射，收到 {type(data).__name__}")
    return data


def make_config(data: dict[str, Any], *, source: str = "config") -> R2VConfig:
    """把已合并的参数字典校验成 R2VConfig；错误信息带字段路径。"""
    if not isinstance(data, dict):
        raise ConfigError(f"{source} 必须是映射，收到 {type(data).__name__}")
    try:
        return R2VConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(_format_validation_error(exc, source)) from exc


def load_config(path: Path) -> R2VConfig:
    """读取并校验 config.yaml（不含 preset 覆盖）。"""
    path = Path(path)
    return make_config(load_yaml_mapping(path), source=str(path))


def mm_per_px(dpi: float) -> float:
    """1 像素 = 多少毫米。300 DPI → 0.084666…；72 DPI → 0.352777…"""
    if dpi <= 0:
        raise ValueError(f"dpi 必须 > 0，收到 {dpi}")
    return 25.4 / float(dpi)


__all__: Sequence[str] = (
    "MANIFEST_COLUMNS",
    "PRESET_NAMES",
    "ConfigError",
    "Manifest",
    "ManifestError",
    "ManifestRow",
    "R2VConfig",
    "dpi_budget_scale",
    "load_config",
    "load_manifest",
    "load_yaml_mapping",
    "make_config",
    "mm_per_px",
)
