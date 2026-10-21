# src/ocr_engine.py
"""OCR 引擎适配层：引擎可缺失，缺失时**不抛异常**、走降级路径。

降级契约（用户决策）
--------------------
引擎不可用时，每条行结果仍然带 `text=""` / `confidence=0.0` / `engine="none"` /
`is_placeholder=true` 四个字段——与引擎正常时**字段完全一致**，下游 S8 不需要分支判断，
只看 `confidence >= text_confidence_min` 决定是否写 TEXT。

安装指引写进 WARN（`pip install -r requirements-ocr.txt`），不静默失败。

引擎探测
--------
- `paddle`：`importlib.util.find_spec("paddleocr")` 且 `paddleocr` 能 import；
- `tesseract`：`find_spec("pytesseract")` **且** 系统里有 `tesseract` 可执行文件
  （只装 Python 包不装二进制时调用必失败，必须一并检查）。
"""

from __future__ import annotations

import importlib.util
import shutil
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

INSTALL_HINT = "OCR 引擎不可用：pip install -r requirements-ocr.txt（或见 README 的 OCR 章节）"

#: 引擎名（从高到低优先级）
_ENGINE_ORDER = ("paddle", "tesseract")


@dataclass(frozen=True)
class OcrLineResult:
    text: str
    confidence: float
    engine: str
    is_placeholder: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "confidence": self.confidence,
            "engine": self.engine,
            "is_placeholder": self.is_placeholder,
        }


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):  # 损坏的安装或父包缺失
        return False


def detect_engines() -> list[str]:
    """返回可用引擎列表（按优先级排序）。全不可用时返回空列表。"""
    available: list[str] = []
    if _module_available("paddleocr") and _module_available("paddle"):
        available.append("paddle")
    if _module_available("pytesseract") and shutil.which("tesseract"):
        available.append("tesseract")
    return [name for name in _ENGINE_ORDER if name in available]


def degraded_result(reason: str = "engine_missing") -> OcrLineResult:
    return OcrLineResult(text="", confidence=0.0, engine="none", is_placeholder=True)


def recognize_line(crop: np.ndarray, engine: str) -> OcrLineResult:
    """对单行图像做识别。`crop` 是灰度或 BGR 的 uint8 数组。

    引擎调用失败时**不抛异常**，返回降级结果并把引擎名改成 `none`——
    单行识别失败不应该让整张图崩掉（SPEC §0：不得崩溃、不得编造）。
    """
    if crop.size == 0:
        return degraded_result("empty_crop")
    if engine == "paddle":
        try:
            from paddleocr import PaddleOCR  # 延迟导入：没装也不影响其它模块

            ocr = PaddleOCR(use_angle_cls=False, lang="ch", show_log=False)
            result = ocr.ocr(crop, cls=False)
            texts: list[str] = []
            scores: list[float] = []
            for page in result or []:
                for item in page or []:
                    if len(item) >= 2 and isinstance(item[1], (list, tuple)):
                        texts.append(str(item[1][0]))
                        scores.append(float(item[1][1]))
            if not texts:
                return OcrLineResult(text="", confidence=0.0, engine="paddle", is_placeholder=True)
            return OcrLineResult(
                text="".join(texts),
                confidence=round(float(np.mean(scores)), 6),
                engine="paddle",
                is_placeholder=False,
            )
        except Exception:  # noqa: BLE001 - 引擎内部异常一律降级，不向上抛
            return degraded_result("paddle_failed")
    if engine == "tesseract":
        try:
            import pytesseract

            data = pytesseract.image_to_data(crop, output_type=pytesseract.Output.DICT)
            words = [w for w in data.get("text", []) if w.strip()]
            confidences = [float(c) for c in data.get("conf", []) if str(c).strip() not in ("", "-1")]
            if not words:
                return OcrLineResult(text="", confidence=0.0, engine="tesseract", is_placeholder=True)
            score = float(np.mean(confidences)) / 100.0 if confidences else 0.0
            return OcrLineResult(
                text="".join(words), confidence=round(score, 6),
                engine="tesseract", is_placeholder=False,
            )
        except Exception:  # noqa: BLE001
            return degraded_result("tesseract_failed")
    return degraded_result()


def recognize_lines(crops: Sequence[np.ndarray]) -> tuple[list[OcrLineResult], list[str]]:
    """批量识别。返回 (逐行结果, WARN 列表)。引擎不可用时全部降级并给出一条 WARN。"""
    engines = detect_engines()
    warnings: list[str] = []
    if not engines:
        warnings.append(INSTALL_HINT)
        return [degraded_result() for _ in crops], warnings
    engine = engines[0]
    results = [recognize_line(crop, engine) for crop in crops]
    if all(result.is_placeholder for result in results) and crops:
        warnings.append(f"{engine} 引擎已安装但未识别出文字（可能是字太小或对比度不足）")
    return results, warnings


__all__ = [
    "INSTALL_HINT",
    "OcrLineResult",
    "degraded_result",
    "detect_engines",
    "recognize_line",
    "recognize_lines",
]
