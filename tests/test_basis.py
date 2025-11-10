"""Tests for basis function evaluation.

This module contains unit tests for the basis function evaluation functions,
including Bernstein basis polynomials.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.testing as nptest
import numpy.typing as npt
import pytest
from scipy.interpolate import BPoly

from pantr.basis import eval_Bernstein_basis_1D

NEGATIVE_TOL: float = 1e-10


def _eval_with_scipy_bpoly(
    degree: int, pts: npt.ArrayLike, dtype: npt.DTypeLike
) -> npt.NDArray[np.floating[Any]]:
    """Evaluate all Bernstein basis polynomials via SciPy's BPoly on [0, 1]."""
    t = np.array(pts, dtype=dtype, copy=False).ravel()
    num_pts = t.size
    vals = np.empty((num_pts, degree + 1), dtype=np.result_type(dtype, np.float64))
    breaks = np.array([0.0, 1.0], dtype=dtype)

    for i in range(degree + 1):
        coeffs = np.zeros((degree + 1, 1), dtype=dtype)
        coeffs[i, 0] = 1.0
        bp = BPoly(coeffs, breaks, extrapolate=True)
        vals[:, i] = bp(t)

    return vals.reshape((*np.array(pts).shape, degree + 1))


class TestEvalBernsteinBasis1D:
    """Test suite for eval_Bernstein_basis_1D function."""

    def test_degree_zero(self) -> None:
        """Test Bernstein basis of degree 0."""
        pts = np.array([0.0, 0.5, 1.0])
        result = eval_Bernstein_basis_1D(0, pts)
        expected = np.ones((3, 1))
        nptest.assert_allclose(result, expected)

    def test_degree_one(self) -> None:
        """Test Bernstein basis of degree 1 (linear)."""
        pts = np.array([0.0, 0.5, 1.0])
        result = eval_Bernstein_basis_1D(1, pts)
        # B_0,1(t) = 1-t, B_1,1(t) = t
        expected = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
        nptest.assert_allclose(result, expected)

    def test_degree_two_example(self) -> None:
        """Test Bernstein basis of degree 2 with example from docstring."""
        pts = np.array([0.0, 0.5, 0.75, 1.0])
        result = eval_Bernstein_basis_1D(2, pts)
        expected = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.25, 0.5, 0.25],
                [0.0625, 0.375, 0.5625],
                [0.0, 0.0, 1.0],
            ]
        )
        nptest.assert_allclose(result, expected)

    def test_boundary_points(self) -> None:
        """Test Bernstein basis at boundary points t=0 and t=1."""
        degree = 3
        pts = np.array([0.0, 1.0])
        result = eval_Bernstein_basis_1D(degree, pts)
        # At t=0: only B_0,n(0) = 1, all others are 0
        # At t=1: only B_n,n(1) = 1, all others are 0
        expected = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
        nptest.assert_allclose(result, expected)

    def test_scalar_input(self) -> None:
        """Test with scalar input point."""
        result = eval_Bernstein_basis_1D(2, 0.5)
        expected = np.array([0.25, 0.5, 0.25])
        nptest.assert_allclose(result, expected)

    def test_list_input(self) -> None:
        """Test with list input."""
        result = eval_Bernstein_basis_1D(2, [0.0, 0.5, 1.0])
        expected = np.array([[1.0, 0.0, 0.0], [0.25, 0.5, 0.25], [0.0, 0.0, 1.0]])
        nptest.assert_allclose(result, expected)

    def test_float32_dtype(self) -> None:
        """Test with float32 input dtype."""
        pts = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        result = eval_Bernstein_basis_1D(2, pts)
        assert result.dtype == np.float32
        expected = np.array([[1.0, 0.0, 0.0], [0.25, 0.5, 0.25], [0.0, 0.0, 1.0]], dtype=np.float32)
        nptest.assert_allclose(result, expected)

    def test_float64_dtype(self) -> None:
        """Test with float64 input dtype."""
        pts = np.array([0.0, 0.5, 1.0], dtype=np.float64)
        result = eval_Bernstein_basis_1D(2, pts)
        assert result.dtype == np.float64
        expected = np.array([[1.0, 0.0, 0.0], [0.25, 0.5, 0.25], [0.0, 0.0, 1.0]], dtype=np.float64)
        nptest.assert_allclose(result, expected)

    def test_int_input_converted_to_float64(self) -> None:
        """Test that integer inputs are converted to float64."""
        pts = np.array([0, 1], dtype=np.int32)
        result = eval_Bernstein_basis_1D(2, pts)
        assert result.dtype == np.float64

    def test_negative_degree_raises_error(self) -> None:
        """Test that negative degree raises ValueError."""
        pts = np.array([0.0, 0.5, 1.0])
        with pytest.raises(ValueError, match="degree must be non-negative"):
            eval_Bernstein_basis_1D(-1, pts)

    def test_partition_of_unity(self) -> None:
        """Test that Bernstein basis functions sum to 1 (partition of unity)."""
        degree = 4
        pts = np.linspace(0.0, 1.0, 11)
        result = eval_Bernstein_basis_1D(degree, pts)
        # Sum over the last dimension should be 1 for each point
        sums = np.sum(result, axis=-1)
        nptest.assert_allclose(sums, 1.0, rtol=1e-10)

    @pytest.mark.parametrize("degree", [0, 1, 2, 3, 5, 10])
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    @pytest.mark.parametrize(
        "pts_factory",
        [
            pytest.param(lambda dt: np.array(0.3, dtype=dt), id="scalar"),
            pytest.param(lambda dt: np.linspace(0.0, 1.0, 11, dtype=dt), id="1d-linspace"),
            pytest.param(
                lambda dt: np.array([[0.0, 0.25, 0.5], [0.75, 1.0, 1 / 3]], dtype=dt),
                id="2d-array",
            ),
        ],
    )
    def test_matches_bpoly(
        self,
        degree: int,
        dtype: npt.DTypeLike,
        pts_factory: Callable[[npt.DTypeLike], npt.ArrayLike],
    ) -> None:
        """Compare eval_Bernstein_basis_1D against BPoly for varied degrees/dtypes/shapes."""
        pts = pts_factory(dtype)
        pantr_vals = eval_Bernstein_basis_1D(degree, pts)
        bpoly_vals = _eval_with_scipy_bpoly(degree, pts, dtype)

        rtol = 1e-6 if dtype == np.float32 else 1e-12
        nptest.assert_allclose(pantr_vals, bpoly_vals, rtol=rtol, atol=0.0)

    @pytest.mark.parametrize("degree", [2, 4, 8])
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_outside_unit_interval_matches_bpoly(self, degree: int, dtype: npt.DTypeLike) -> None:
        """Ensure evaluation outside [0, 1] matches BPoly extrapolation."""
        pts = np.array([-0.5, -0.1, 1.1, 1.5], dtype=dtype)
        pantr_vals = eval_Bernstein_basis_1D(degree, pts)
        bpoly_vals = _eval_with_scipy_bpoly(degree, pts, dtype)

        rtol = 1e-6 if dtype == np.float32 else 1e-12
        nptest.assert_allclose(pantr_vals, bpoly_vals, rtol=rtol, atol=0.0)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_random_points_high_degree(self, dtype: npt.DTypeLike) -> None:
        """Stress test with higher degree and random points."""
        rng = np.random.default_rng(1234)
        degree = 12
        pts = rng.random(200).astype(dtype)
        pantr_vals = eval_Bernstein_basis_1D(degree, pts)
        bpoly_vals = _eval_with_scipy_bpoly(degree, pts, dtype)

        rtol = 5e-6 if dtype == np.float32 else 5e-13
        nptest.assert_allclose(pantr_vals, bpoly_vals, rtol=rtol, atol=0.0)

    def test_non_negativity(self) -> None:
        """Test that Bernstein basis functions are non-negative on [0, 1]."""
        degree = 3
        pts = np.linspace(0.0, 1.0, 21)
        result = eval_Bernstein_basis_1D(degree, pts)
        # Allow small numerical errors
        assert np.all(result >= -NEGATIVE_TOL)

    def test_2d_input_shape_preservation(self) -> None:
        """Test that 2D input arrays preserve shape correctly."""
        pts = np.array([[0.0, 0.5], [0.25, 0.75]])
        result = eval_Bernstein_basis_1D(2, pts)
        # Should have shape (2, 2, 3) - original shape + basis dimension
        assert result.shape == (2, 2, 3)

    def test_3d_input_shape_preservation(self) -> None:
        """Test that 3D input arrays preserve shape correctly."""
        pts = np.array([[[0.0], [0.5]], [[0.25], [0.75]]])
        result = eval_Bernstein_basis_1D(1, pts)
        # Should have shape (2, 2, 1, 2) - original shape + basis dimension
        assert result.shape == (2, 2, 1, 2)

    def test_single_point_at_midpoint(self) -> None:
        """Test evaluation at a single midpoint."""
        result = eval_Bernstein_basis_1D(3, 0.5)
        # For degree 3 at t=0.5, all basis functions should be symmetric
        # B_0,3(0.5) = B_3,3(0.5) = (1/2)^3 = 0.125
        # B_1,3(0.5) = B_2,3(0.5) = 3 * (1/2)^3 = 0.375
        expected = np.array([0.125, 0.375, 0.375, 0.125])
        nptest.assert_allclose(result, expected)

    def test_high_degree(self) -> None:
        """Test with higher degree to ensure numerical stability."""
        degree = 10
        pts = np.array([0.0, 0.5, 1.0])
        result = eval_Bernstein_basis_1D(degree, pts)
        # Check shape
        assert result.shape == (3, degree + 1)
        # Check partition of unity
        sums = np.sum(result, axis=-1)
        nptest.assert_allclose(sums, 1.0, rtol=1e-10)
        # Check boundary conditions
        nptest.assert_allclose(result[0, 0], 1.0)
        nptest.assert_allclose(result[0, 1:], 0.0)
        nptest.assert_allclose(result[2, -1], 1.0)
        nptest.assert_allclose(result[2, :-1], 0.0)

    def test_points_outside_unit_interval(self) -> None:
        """Test evaluation at points outside [0, 1]."""
        # Bernstein basis can be evaluated outside [0, 1], though values may be negative
        pts = np.array([-0.5, 1.5])
        result = eval_Bernstein_basis_1D(2, pts)
        # Should not raise an error and should return valid values
        assert result.shape == (2, 3)
        # Check partition of unity still holds
        sums = np.sum(result, axis=-1)
        nptest.assert_allclose(sums, 1.0, rtol=1e-10)
