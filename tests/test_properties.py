# tests/test_properties.py
"""性质测试（hypothesis）：把「比对的数学性质」固定下来，防止重构悄悄改语义。"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.config import dpi_budget_scale, mm_per_px
from src.golden_compare import hausdorff_p95

POINTS = st.lists(
    st.tuples(st.floats(-100.0, 100.0), st.floats(-100.0, 100.0)),
    min_size=2,
    max_size=16,
)


def _arr(pairs) -> np.ndarray:  # noqa: ANN001
    return np.asarray(pairs, dtype=np.float64)


@settings(max_examples=30, deadline=None)
@given(POINTS)
def test_hausdorff_identity_is_zero(pairs) -> None:  # noqa: ANN001
    arr = _arr(pairs)
    assert hausdorff_p95(arr, arr) == pytest.approx(0.0, abs=1e-12)


@settings(max_examples=30, deadline=None)
@given(POINTS, POINTS)
def test_hausdorff_is_symmetric(a_pairs, b_pairs) -> None:  # noqa: ANN001
    a, b = _arr(a_pairs), _arr(b_pairs)
    assert hausdorff_p95(a, b) == pytest.approx(hausdorff_p95(b, a), abs=1e-12)


@settings(max_examples=20, deadline=None)
@given(POINTS, POINTS, st.floats(-50.0, 50.0), st.floats(-50.0, 50.0))
def test_hausdorff_is_translation_invariant(a_pairs, b_pairs, dx, dy) -> None:  # noqa: ANN001
    shift = np.array([dx, dy])
    a, b = _arr(a_pairs), _arr(b_pairs)
    moved = hausdorff_p95(a + shift, b + shift)
    base = hausdorff_p95(a, b)
    assert moved == pytest.approx(base, rel=1e-9, abs=1e-9)


@settings(max_examples=30, deadline=None)
@given(st.floats(20.0, 1200.0), st.floats(20.0, 1200.0))
def test_budget_scale_identity(true_dpi, reference_dpi) -> None:  # noqa: ANN001
    scale = dpi_budget_scale(true_dpi, reference_dpi)
    assert scale * true_dpi == pytest.approx(reference_dpi, rel=1e-12)


@settings(max_examples=30, deadline=None)
@given(st.floats(20.0, 1200.0))
def test_budget_scale_tracks_mm_per_px(dpi) -> None:  # noqa: ANN001
    """低 DPI 图 1 px 的 mm 数更大，预算放大倍数与 mm/px 成正比。"""
    assert dpi_budget_scale(dpi, 300.0) == pytest.approx(
        mm_per_px(dpi) / mm_per_px(300.0), rel=1e-12
    )
