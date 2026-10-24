# scripts/poc_dual_model_segment.py
"""M8-4 离线 PoC：双模型联合切分（LINE/ARC × 切点枚举 × BIC）。

背景
----
M7 的切点判据（"曲率突变点 / 最大残差处递归切分"）在 `synthetic_polyline_bulge`
上始终失效：曲线是「圆弧 + 直线」的拼接，任何单模型的残差阈值都会被高曲率区拉偏。
本 PoC 把它重构成**组合优化**：枚举切点 k 与两侧模型组合 (M1, M2)，
用"SSE + BIC 惩罚"选全局最优。

算法
----
对切点 k（两侧各至少 `MIN_POINTS` 点）与模型对 (M1, M2) ∈ {LINE, ARC}²：
    SSE(k) = SSE(M1, P[:k]) + SSE(M2, P[k:])
    Cost(k) = SSE(k) + lam · ln(n) · p(M1, M2)
其中 p(LINE)=2、p(ARC)=3（theta 参数个数），n 为总点数。
搜索用**粗到细**两级（粗步长 `COARSE_STEP`，再在最优附近 ±`COARSE_STEP` 精搜），
把 O(n) 次圆拟合降到 O(n/step + step)，满足 <10 ms/边的红线。

验收
----
1. bulge 场景能否找到真实切点（弧↔直线交界，真值 ≈ 弧长 273 px 处）；
2. 两侧 RMS 是否都低于 `--tolerance-mm`（默认 0.15 mm）；
3. 单条边耗时是否 < 10 ms。

用法：
    python scripts/poc_dual_model_segment.py
    python scripts/poc_dual_model_segment.py --tolerance-mm 0.10 --json-out outputs/poc_seg.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scipy.spatial import cKDTree  # noqa: E402

from src.config import load_config  # noqa: E402
from src.s1_preprocess import run_s1_preprocess  # noqa: E402
from src.s2_binarize import S2Input, binarize_image  # noqa: E402
from src.s3_skeleton import run_s3_skeleton  # noqa: E402
import src.s4_fit as s4  # noqa: E402

MIN_POINTS = 10
COARSE_STEP = 10
#: BIC 惩罚系数（0 = 纯 SSE，1 = 标准 BIC 量级）
DEFAULT_LAMBDA = 1.0
#: 参与竞争的模型对。`("arc","arc")` 被刻意排除：工程图上"两段同径圆弧相接"极罕见，
#: 而它每个候选点要多做 2 次圆拟合（占扫描耗时 ~40%）。若将来遇到 S 形曲线再单独开。
MODEL_PAIRS = (("line", "line"), ("line", "arc"), ("arc", "line"))


class PrefixLineSSE:
    """用积分图（prefix sums）把**直线 TLS 的残差平方和**降到 O(1)。

    数学依据：过质心的直线 TLS 拟合，其残差平方和**等于散布矩阵的最小特征值**：
        S = [[Sxx, Sxy], [Sxy, Syy]]，λ_min = (Sxx+Syy)/2 − sqrt(((Sxx−Syy)/2)² + Sxy²)
    而 Sxx/Syy/Sxy 都可以由 Σx, Σy, Σxx, Σyy, Σxy 的区间差 O(1) 求出。
    """

    def __init__(self, points: np.ndarray) -> None:
        x = points[:, 0]
        y = points[:, 1]
        self._n = np.concatenate([[0], np.arange(1, len(points) + 1)])
        self._sx = np.concatenate([[0.0], np.cumsum(x)])
        self._sy = np.concatenate([[0.0], np.cumsum(y)])
        self._sxx = np.concatenate([[0.0], np.cumsum(x * x)])
        self._syy = np.concatenate([[0.0], np.cumsum(y * y)])
        self._sxy = np.concatenate([[0.0], np.cumsum(x * y)])

    def sse(self, start: int, end: int) -> float:
        """区间 [start, end) 的直线 TLS SSE。"""
        n = float(self._n[end] - self._n[start])
        if n < 2:
            return float("inf")
        sx = self._sx[end] - self._sx[start]
        sy = self._sy[end] - self._sy[start]
        sxx = self._sxx[end] - self._sxx[start]
        syy = self._syy[end] - self._syy[start]
        sxy = self._sxy[end] - self._sxy[start]
        cxx = sxx - sx * sx / n
        cyy = syy - sy * sy / n
        cxy = sxy - sx * sy / n
        half = (cxx + cyy) / 2.0
        disc = math.sqrt(max(((cxx - cyy) / 2.0) ** 2 + cxy * cxy, 0.0))
        return float(max(half - disc, 0.0))


def _fit(kind: str, points: np.ndarray, *, refine: bool = False):
    """返回 (模型, SSE, 参数个数)。LINE=2 参数，ARC=3 参数。

    `refine=False` 时圆的拟合用 **Kåsa 代数解**（约 30 µs），只有最终胜出的切点
    才做 LM 精化（约 0.5 ms）——这是把 O(n) 次圆拟合压进 10 ms 的关键。
    """
    if kind == "line":
        model = s4.fit_line(points)
        if model is None:
            return None, float("inf"), 2
        normal = np.asarray(model.normal)
        residuals = (points - np.asarray(model.point)) @ normal
        return model, float(np.sum(residuals**2)), 2
    if refine:
        model = s4.fit_circle(points)
    else:
        # 扫描阶段：圆的代数拟合用**隔点采样**（点数减半，系数矩阵条件数几乎不变），
        # SSE 仍对全部点计算，因此评分精度不受影响、只省拟合时间。
        sample = points[::2] if len(points) >= 2 * s4._MIN_CIRCLE_POINTS else points  # noqa: SLF001
        model = s4._circle_kasa(sample)  # noqa: SLF001 - PoC 里复用同一代数解
        if model is None and len(points) >= s4._MIN_CIRCLE_POINTS:  # noqa: SLF001
            model = s4.fit_circle(points)
    if model is None:
        return None, float("inf"), 3
    residuals = np.hypot(points[:, 0] - model.center[0], points[:, 1] - model.center[1]) - model.radius
    return model, float(np.sum(residuals**2)), 3


def evaluate_split(
    points: np.ndarray,
    k: int,
    kind_pair: tuple[str, str],
    lam: float,
    n: int,
    prefix: "PrefixLineSSE",
):
    """评分一个切点。直线侧用积分图 SSE（O(1)），圆弧侧用 Kåsa 代数解。"""
    sses: list[float] = []
    params = 0
    for kind, sl in ((kind_pair[0], slice(0, k)), (kind_pair[1], slice(k, n))):
        if kind == "line":
            sse = prefix.sse(sl.start or 0, sl.stop)
            params += 2
        else:
            _model, sse, p = _fit("arc", points[sl])
            params += p
        if not math.isfinite(sse):
            return None
        sses.append(sse)
    sse = sses[0] + sses[1]
    cost = sse + lam * math.log(max(n, 2)) * params
    return cost, sse, sses[0], sses[1]


def best_split(points: np.ndarray, *, lam: float = DEFAULT_LAMBDA, tol_px: float = 1.0):
    """粗到细枚举，返回 (k, kind_pair, cost, sse, sse_left, sse_right, elapsed_ms)。"""
    n = len(points)
    if n < 2 * MIN_POINTS:
        return None
    pairs = MODEL_PAIRS
    prefix = PrefixLineSSE(points)
    t0 = time.perf_counter()

    def scan(candidates):
        best = None
        for k in candidates:
            for pair in pairs:
                result = evaluate_split(points, k, pair, lam, n, prefix)
                if result is None:
                    continue
                cost = result[0]
                if best is None or cost < best[0]:
                    best = (cost, k, pair, result[1], result[2], result[3])
        return best

    coarse = scan(range(MIN_POINTS, n - MIN_POINTS + 1, COARSE_STEP))
    if coarse is None:
        return None
    center = coarse[1]
    fine = scan(range(max(MIN_POINTS, center - COARSE_STEP), min(n - MIN_POINTS, center + COARSE_STEP) + 1))
    best = fine if (fine is not None and fine[0] <= coarse[0]) else coarse
    elapsed_scan_ms = (time.perf_counter() - t0) * 1000.0
    _cost, k, pair, sse, sse1, sse2 = best
    # 只有最终胜者做 LM 精化（不计入扫描耗时，单独报告）
    t_refine = time.perf_counter()
    _m1, sse1r, _ = _fit(pair[0], points[:k], refine=True)
    _m2, sse2r, _ = _fit(pair[1], points[k:], refine=True)
    refine_ms = (time.perf_counter() - t_refine) * 1000.0
    return {
        "split_index": int(k),
        "models": list(pair),
        "cost": float(_cost),
        "sse_total": float(sse),
        "sse_left": float(sse1r),
        "sse_right": float(sse2r),
        "rms_left_px": float(math.sqrt(sse1r / max(k, 1))),
        "rms_right_px": float(math.sqrt(sse2r / max(n - k, 1))),
        "tolerance_px": float(tol_px),
        "passes_tolerance": bool(
            math.sqrt(sse1r / max(k, 1)) <= tol_px and math.sqrt(sse2r / max(n - k, 1)) <= tol_px
        ),
        "scan_ms": round(elapsed_scan_ms, 3),
        "refine_ms": round(refine_ms, 3),
        "elapsed_ms": round(elapsed_scan_ms + refine_ms, 3),
        "points": int(n),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default=str(ROOT / "test_images" / "synthetic_polyline_bulge.png"))
    parser.add_argument("--dpi", type=float, default=300.0)
    parser.add_argument("--tolerance-mm", type=float, default=0.15)
    parser.add_argument("--lam", type=float, default=DEFAULT_LAMBDA)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args(argv)

    cfg = load_config(ROOT / "config.yaml")
    pre, _ = run_s1_preprocess(args.image, cfg)
    s2, _ = binarize_image(S2Input.from_preprocess(pre), cfg)
    s3, _ = run_s3_skeleton(s2.binary, cfg, dpi=pre.dpi, linewidth_px=s2.linewidth_px)
    contours = np.vstack([np.asarray(c, dtype=np.float64) for c in s2.edges.contours])
    tree = cKDTree(contours)
    edge = max(s3.graph.edges, key=lambda e: e.length_px)
    points = s4.centerline_from_contours(edge.points, tree, contours, linewidth_px=s2.linewidth_px)

    mm_per_px = 25.4 / args.dpi
    tol_px = args.tolerance_mm / mm_per_px
    result = best_split(points, lam=args.lam, tol_px=tol_px)
    if result is None:
        print("ERROR 点数不足以切分", file=sys.stderr)
        return 2
    result["image"] = Path(args.image).name
    result["tolerance_mm"] = args.tolerance_mm
    result["mm_per_px"] = round(mm_per_px, 6)
    result["lambda"] = args.lam
    result["edge_length_px"] = round(float(edge.length_px), 3)
    # 真实切点参考：golden 的折线在 (30,10) mm 处转折（弧段 ≈ 24 mm ≈ 273 px）
    result["expected_split_px_hint"] = 273.0
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
