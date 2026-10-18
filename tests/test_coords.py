# tests/test_coords.py
"""坐标契约：px ↔ mm ↔ CAD 的矩阵变换与逆变换（SPEC §3.1）。"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.config import mm_per_px
from src.coords import Transform2D, chain, quantize


def test_image_to_cad_maps_corners() -> None:
    height_px = 625
    t = Transform2D.image_to_cad(dpi=300.0, height_px=height_px)
    height_mm = height_px * mm_per_px(300.0)
    np.testing.assert_allclose(t.apply([0.0, 0.0]), [0.0, height_mm], atol=1e-12)
    np.testing.assert_allclose(t.apply([0.0, float(height_px)]), [0.0, 0.0], atol=1e-12)
    width_mm = 736 * mm_per_px(300.0)
    np.testing.assert_allclose(t.apply([736.0, 0.0]), [width_mm, height_mm], atol=1e-12)


def test_image_to_cad_rejects_bad_height() -> None:
    with pytest.raises(ValueError, match="height_px"):
        Transform2D.image_to_cad(dpi=300.0, height_px=0)


def test_inverse_roundtrip_single_and_batch() -> None:
    t = Transform2D.image_to_cad(dpi=300.0, height_px=625)
    inv = t.inverse()
    pt = np.array([123.4, 567.8])
    np.testing.assert_allclose(inv.apply(t.apply(pt)), pt, atol=1e-12)
    batch = np.array([[0.0, 0.0], [100.0, 200.0], [735.0, 624.0]])
    np.testing.assert_allclose(inv.apply(t.apply(batch)), batch, atol=1e-12)


def test_flip_y_twice_is_identity() -> None:
    t = Transform2D.flip_y(100.0)
    assert t @ t == Transform2D.identity()
    np.testing.assert_allclose((t @ t).apply([3.0, 4.0]), [3.0, 4.0])


def test_rotation_90_degrees() -> None:
    t = Transform2D.rotation_deg(90.0)
    np.testing.assert_allclose(t.apply([1.0, 0.0]), [0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(t.apply([0.0, 1.0]), [-1.0, 0.0], atol=1e-12)


def test_rotation_about_pivot_keeps_pivot() -> None:
    t = Transform2D.rotation_deg(37.0, pivot=(10.0, 20.0))
    np.testing.assert_allclose(t.apply([10.0, 20.0]), [10.0, 20.0], atol=1e-12)


def test_scale_factors_and_axis_alignment() -> None:
    t = Transform2D.scale(2.0, 3.0) @ Transform2D.translation(5.0, 1.0)
    assert t.scale_factors == pytest.approx((2.0, 3.0))
    assert t.is_axis_aligned is True
    assert Transform2D.rotation_deg(10.0).is_axis_aligned is False


def test_matrix_validation() -> None:
    with pytest.raises(ValueError, match="3×3"):
        Transform2D(np.eye(2))
    with pytest.raises(ValueError, match="末行"):
        Transform2D(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 1.0]]))
    with pytest.raises(ValueError, match="NaN/Inf"):
        Transform2D(np.array([[np.nan, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))


def test_zero_scale_rejected_and_singular_not_invertible() -> None:
    with pytest.raises(ValueError, match="缩放系数不能为 0"):
        Transform2D.scale(0.0, 1.0)
    singular = Transform2D(np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]))
    with pytest.raises(ValueError, match="不可逆"):
        singular.inverse()


def test_apply_shape_validation() -> None:
    t = Transform2D.identity()
    with pytest.raises(ValueError, match=r"\(N,2\)"):
        t.apply(np.zeros((3, 3)))
    with pytest.raises(ValueError, match="NaN/Inf"):
        t.apply([np.inf, 0.0])


def test_chain_applies_right_to_left() -> None:
    scale = Transform2D.scale(2.0)
    move = Transform2D.translation(1.0, 0.0)
    combined = chain(move, scale)  # 先 scale 再 move
    np.testing.assert_allclose(combined.apply([1.0, 0.0]), [3.0, 0.0])


def test_quantize_decimal_error_bound() -> None:
    values = np.array([1.23456, -0.00004, 2.99999])
    out = quantize(values, 4)
    assert np.all(np.abs(out - values) <= 0.5e-4)
    with pytest.raises(ValueError, match="decimals"):
        quantize(values, -1)


def test_to_list_is_json_friendly() -> None:
    t = Transform2D.translation(1.5, -2.5)
    rows = t.to_list()
    assert rows == [[1.0, 0.0, 1.5], [0.0, 1.0, -2.5], [0.0, 0.0, 1.0]]
    assert all(isinstance(v, float) for row in rows for v in row)


@settings(max_examples=25, deadline=None)
@given(
    st.floats(72.0, 600.0),
    st.integers(64, 4096),
    st.lists(
        st.tuples(st.floats(0.0, 4096.0), st.floats(0.0, 4096.0)), min_size=1, max_size=8
    ),
)
def test_px_to_cad_roundtrip(dpi, height_px, pts) -> None:  # noqa: ANN001
    t = Transform2D.image_to_cad(dpi=dpi, height_px=height_px)
    arr = np.asarray(pts, dtype=np.float64)
    back = t.inverse().apply(t.apply(arr))
    assert np.allclose(back, arr, atol=1e-9, rtol=0)


@settings(max_examples=20, deadline=None)
@given(st.floats(-180.0, 180.0), st.floats(-180.0, 180.0), st.floats(0.0, 100.0))
def test_rotation_composition_adds_angles(a, b, x) -> None:  # noqa: ANN001
    p = [x, 0.0]
    combined = Transform2D.rotation_deg(a) @ Transform2D.rotation_deg(b)
    np.testing.assert_allclose(
        combined.apply(p), Transform2D.rotation_deg(a + b).apply(p), atol=1e-9
    )
