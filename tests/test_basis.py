"""Tests for basis function evaluation.

This module contains unit tests for the basis function evaluation functions,
including Bernstein basis polynomials.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from typing import Any, Literal

import numpy as np
import numpy.testing as nptest
import numpy.typing as npt
import pytest
from scipy.interpolate import BPoly

from pantr.basis import (
    LagrangeVariant,
    eval_Bernstein_basis,
    eval_Bernstein_basis_1D,
    eval_cardinal_Bspline_basis,
    eval_cardinal_Bspline_basis_1D,
    eval_Lagrange_basis,
    eval_Lagrange_basis_1D,
)
from pantr.quad import PointsLattice
from pantr.tolerance import get_conservative_tolerance, get_default_tolerance

NEGATIVE_TOL: float = get_conservative_tolerance(np.float64)


class BasisType(str, Enum):
    """Enumeration for basis function types."""

    BERNSTEIN = "bernstein"
    CARDINAL_BSPLINE = "cardinal_bspline"
    LAGRANGE = "lagrange"


def _get_basis_function(
    basis_type: BasisType,
) -> Callable[..., npt.NDArray[np.float32 | np.float64]]:
    """Get the multi-dimensional basis function for the given type."""
    if basis_type == BasisType.BERNSTEIN:
        return eval_Bernstein_basis
    if basis_type == BasisType.CARDINAL_BSPLINE:
        return eval_cardinal_Bspline_basis
    if basis_type == BasisType.LAGRANGE:
        return eval_Lagrange_basis
    raise ValueError(f"Unknown basis type: {basis_type}")


def _get_basis_1d_function(
    basis_type: BasisType,
) -> Callable[..., npt.NDArray[np.float32 | np.float64]]:
    """Get the 1D basis function for the given type."""
    if basis_type == BasisType.BERNSTEIN:
        return eval_Bernstein_basis_1D
    if basis_type == BasisType.CARDINAL_BSPLINE:
        return eval_cardinal_Bspline_basis_1D
    if basis_type == BasisType.LAGRANGE:
        return eval_Lagrange_basis_1D
    raise ValueError(f"Unknown basis type: {basis_type}")


def _call_basis_function(
    basis_type: BasisType,
    degrees: list[int],
    pts: npt.ArrayLike | PointsLattice,
    variant: LagrangeVariant | None = None,
    funcs_order: Literal["C", "F"] = "C",
) -> npt.NDArray[np.float32 | np.float64]:
    """Call the appropriate basis function with the given arguments."""
    func = _get_basis_function(basis_type)
    if basis_type == BasisType.LAGRANGE:
        if variant is None:
            variant = LagrangeVariant.EQUISPACES
        return func(degrees, variant, pts, funcs_order)
    return func(degrees, pts, funcs_order)


def _call_basis_1d_function(
    basis_type: BasisType,
    degree: int,
    pts: npt.ArrayLike,
    variant: LagrangeVariant | None = None,
) -> npt.NDArray[np.float32 | np.float64]:
    """Call the appropriate 1D basis function with the given arguments."""
    func = _get_basis_1d_function(basis_type)
    if basis_type == BasisType.LAGRANGE:
        if variant is None:
            variant = LagrangeVariant.EQUISPACES
        return func(degree, variant, pts)
    return func(degree, pts)


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
        nptest.assert_allclose(sums, 1.0, rtol=get_conservative_tolerance(np.float64))


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
    degree: int,
    dtype: npt.DTypeLike,
    pts_factory: Callable[[npt.DTypeLike], npt.ArrayLike],
) -> None:
    """Compare eval_Bernstein_basis_1D against BPoly for varied degrees/dtypes/shapes."""
    pts = pts_factory(dtype)
    pantr_vals = eval_Bernstein_basis_1D(degree, pts)
    bpoly_vals = _eval_with_scipy_bpoly(degree, pts, dtype)

    rtol = get_default_tolerance(dtype)
    nptest.assert_allclose(pantr_vals, bpoly_vals, rtol=rtol, atol=0.0)


@pytest.mark.parametrize("degree", [2, 4, 8])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_outside_unit_interval_matches_bpoly(degree: int, dtype: npt.DTypeLike) -> None:
    """Ensure evaluation outside [0, 1] matches BPoly extrapolation."""
    pts = np.array([-0.5, -0.1, 1.1, 1.5], dtype=dtype)
    pantr_vals = eval_Bernstein_basis_1D(degree, pts)
    bpoly_vals = _eval_with_scipy_bpoly(degree, pts, dtype)

    rtol = get_default_tolerance(dtype)
    nptest.assert_allclose(pantr_vals, bpoly_vals, rtol=rtol, atol=0.0)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_random_points_high_degree(dtype: npt.DTypeLike) -> None:
    """Stress test with higher degree and random points."""
    rng = np.random.default_rng(1234)
    degree = 12
    pts = rng.random(200).astype(dtype)
    pantr_vals = eval_Bernstein_basis_1D(degree, pts)
    bpoly_vals = _eval_with_scipy_bpoly(degree, pts, dtype)

    rtol = 5.0 * get_default_tolerance(dtype)
    nptest.assert_allclose(pantr_vals, bpoly_vals, rtol=rtol, atol=0.0)


def test_non_negativity() -> None:
    """Test that Bernstein basis functions are non-negative on [0, 1]."""
    degree = 3
    pts = np.linspace(0.0, 1.0, 21)
    result = eval_Bernstein_basis_1D(degree, pts)
    # Allow small numerical errors
    assert np.all(result >= -NEGATIVE_TOL)


def test_2d_input_shape_preservation() -> None:
    """Test that 2D input arrays preserve shape correctly."""
    pts = np.array([[0.0, 0.5], [0.25, 0.75]])
    result = eval_Bernstein_basis_1D(2, pts)
    # Should have shape (2, 2, 3) - original shape + basis dimension
    assert result.shape == (2, 2, 3)


def test_3d_input_shape_preservation() -> None:
    """Test that 3D input arrays preserve shape correctly."""
    pts = np.array([[[0.0], [0.5]], [[0.25], [0.75]]])
    result = eval_Bernstein_basis_1D(1, pts)
    # Should have shape (2, 2, 1, 2) - original shape + basis dimension
    assert result.shape == (2, 2, 1, 2)


def test_single_point_at_midpoint() -> None:
    """Test evaluation at a single midpoint."""
    result = eval_Bernstein_basis_1D(3, 0.5)
    # For degree 3 at t=0.5, all basis functions should be symmetric
    # B_0,3(0.5) = B_3,3(0.5) = (1/2)^3 = 0.125
    # B_1,3(0.5) = B_2,3(0.5) = 3 * (1/2)^3 = 0.375
    expected = np.array([0.125, 0.375, 0.375, 0.125])
    nptest.assert_allclose(result, expected)


def test_high_degree() -> None:
    """Test with higher degree to ensure numerical stability."""
    degree = 10
    pts = np.array([0.0, 0.5, 1.0])
    result = eval_Bernstein_basis_1D(degree, pts)
    # Check shape
    assert result.shape == (3, degree + 1)
    # Check partition of unity
    sums = np.sum(result, axis=-1)
    nptest.assert_allclose(sums, 1.0, rtol=get_conservative_tolerance(np.float64))
    # Check boundary conditions
    nptest.assert_allclose(result[0, 0], 1.0)
    nptest.assert_allclose(result[0, 1:], 0.0)
    nptest.assert_allclose(result[2, -1], 1.0)
    nptest.assert_allclose(result[2, :-1], 0.0)


def test_points_outside_unit_interval() -> None:
    """Test evaluation at points outside [0, 1]."""
    # Bernstein basis can be evaluated outside [0, 1], though values may be negative
    pts = np.array([-0.5, 1.5])
    result = eval_Bernstein_basis_1D(2, pts)
    # Should not raise an error and should return valid values
    assert result.shape == (2, 3)
    # Check partition of unity still holds
    sums = np.sum(result, axis=-1)
    nptest.assert_allclose(sums, 1.0, rtol=get_conservative_tolerance(np.float64))


@pytest.mark.parametrize(
    "basis_type", [BasisType.BERNSTEIN, BasisType.CARDINAL_BSPLINE, BasisType.LAGRANGE]
)
class TestEvalBasis:
    """Test suite for multi-dimensional basis functions.

    Tests Bernstein, cardinal B-spline, and Lagrange basis functions.
    """

    def test_2d_array_single_degree(self, basis_type: BasisType) -> None:
        """Test 2D array input with same degree in both dimensions."""
        degrees = [2, 2]
        pts = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]], dtype=np.float64)
        result = _call_basis_function(basis_type, degrees, pts)
        # Should have shape (n_points, n_basis_functions) = (3, 9) for degree 2 in 2D
        assert result.shape == (3, 9)
        # Verify by comparing with manual computation
        basis_x = _call_basis_1d_function(basis_type, 2, pts[:, 0])
        basis_y = _call_basis_1d_function(basis_type, 2, pts[:, 1])
        expected = np.einsum("pi,pj->pij", basis_x, basis_y).reshape(3, 9)
        if basis_type == BasisType.LAGRANGE:
            rtol = get_default_tolerance(np.float64)
            nptest.assert_allclose(result, expected, rtol=rtol, atol=0.0)
        else:
            nptest.assert_allclose(result, expected)

    def test_2d_array_different_degrees(self, basis_type: BasisType) -> None:
        """Test 2D array input with different degrees per dimension."""
        degrees = [1, 2]
        pts = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]], dtype=np.float64)
        result = _call_basis_function(basis_type, degrees, pts)
        assert result.shape == (3, 6)
        basis_x = _call_basis_1d_function(basis_type, 1, pts[:, 0])
        basis_y = _call_basis_1d_function(basis_type, 2, pts[:, 1])
        expected = np.einsum("pi,pj->pij", basis_x, basis_y).reshape(3, 6)
        if basis_type == BasisType.LAGRANGE:
            rtol = get_default_tolerance(np.float64)
            nptest.assert_allclose(result, expected, rtol=rtol, atol=0.0)
        else:
            nptest.assert_allclose(result, expected)

    def test_points_lattice(self, basis_type: BasisType) -> None:
        """Test PointsLattice input."""
        degrees = [2, 2]
        pts_x = np.linspace(0.0, 1.0, 3, dtype=np.float64)
        pts_y = np.linspace(0.0, 1.0, 3, dtype=np.float64)
        lattice = PointsLattice([pts_x, pts_y])
        result = _call_basis_function(basis_type, degrees, lattice)
        assert result.shape == (9, 9)

    def test_points_lattice_mismatched_dimension_raises_error(self, basis_type: BasisType) -> None:
        """Test that mismatched number of degrees vs PointsLattice dimension raises ValueError."""
        degrees = [1, 1, 1]
        pts_x = np.linspace(0.0, 1.0, 3, dtype=np.float64)
        pts_y = np.linspace(0.0, 1.0, 3, dtype=np.float64)
        lattice = PointsLattice([pts_x, pts_y])
        with pytest.raises(
            ValueError, match="The number of evaluators must be equal to the dimension"
        ):
            _call_basis_function(basis_type, degrees, lattice)

    def test_points_lattice_1d(self, basis_type: BasisType) -> None:
        """Test 1D PointsLattice input (to cover return path in _basis_combinator_lattice)."""
        degrees = [2]
        pts_x = np.linspace(0.0, 1.0, 3, dtype=np.float64)
        lattice = PointsLattice([pts_x])
        result = _call_basis_function(basis_type, degrees, lattice)
        assert result.shape == (3, 3)

    def test_funcs_order_c(self, basis_type: BasisType) -> None:
        """Test C-order (default) for basis functions."""
        degrees = [1, 1]
        pts = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]], dtype=np.float64)
        result = _call_basis_function(basis_type, degrees, pts, funcs_order="C")
        assert result.shape == (3, 4)

    def test_funcs_order_f(self, basis_type: BasisType) -> None:
        """Test F-order for basis functions."""
        degrees = [1, 1]
        pts = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]], dtype=np.float64)
        result_f = _call_basis_function(basis_type, degrees, pts, funcs_order="F")
        result_c = _call_basis_function(basis_type, degrees, pts, funcs_order="C")
        assert result_f.shape == result_c.shape == (3, 4)

    def test_3d_array(self, basis_type: BasisType) -> None:
        """Test 3D array input."""
        degrees = [1, 1, 1]
        pts = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [1.0, 1.0, 1.0]], dtype=np.float64)
        result = _call_basis_function(basis_type, degrees, pts)
        assert result.shape == (3, 8)

    def test_dtype_preservation_float32(self, basis_type: BasisType) -> None:
        """Test dtype preservation with float32."""
        degrees = [2, 2]
        pts = np.array([[0.0, 0.0], [0.5, 0.5]], dtype=np.float32)
        result = _call_basis_function(basis_type, degrees, pts)
        assert result.dtype == np.float32

    def test_dtype_preservation_float64(self, basis_type: BasisType) -> None:
        """Test dtype preservation with float64."""
        degrees = [2, 2]
        pts = np.array([[0.0, 0.0], [0.5, 0.5]], dtype=np.float64)
        result = _call_basis_function(basis_type, degrees, pts)
        assert result.dtype == np.float64

    def test_negative_degree_raises_error(self, basis_type: BasisType) -> None:
        """Test that negative degree raises ValueError."""
        degrees = [-1, 2]
        pts = np.array([[0.0, 0.0], [0.5, 0.5]], dtype=np.float64)
        with pytest.raises(ValueError, match="All degrees must be non-negative integers"):
            _call_basis_function(basis_type, degrees, pts)

    def test_partition_of_unity(self, basis_type: BasisType) -> None:
        """Test that multi-dimensional basis functions sum to 1."""
        degrees = [2, 2]
        pts = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]], dtype=np.float64)
        result = _call_basis_function(basis_type, degrees, pts)
        sums = np.sum(result, axis=-1)
        if basis_type in (BasisType.LAGRANGE, BasisType.CARDINAL_BSPLINE):
            rtol = get_default_tolerance(np.float64)
            nptest.assert_allclose(sums, 1.0, rtol=rtol, atol=0.0)
        else:
            nptest.assert_allclose(sums, 1.0, rtol=get_conservative_tolerance(np.float64))

    def test_single_point(self, basis_type: BasisType) -> None:
        """Test with single point."""
        degrees = [1, 1]
        pts = np.array([[0.5, 0.5]], dtype=np.float64)
        result = _call_basis_function(basis_type, degrees, pts)
        assert result.shape == (1, 4)
        if basis_type in (BasisType.LAGRANGE, BasisType.CARDINAL_BSPLINE):
            rtol = get_default_tolerance(np.float64)
            nptest.assert_allclose(np.sum(result), 1.0, rtol=rtol, atol=0.0)
        else:
            nptest.assert_allclose(np.sum(result), 1.0)

    def test_non_ndarray_input_list(self, basis_type: BasisType) -> None:
        """Test with list input (converted to ndarray internally)."""
        degrees = [1, 1]
        pts = [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]
        result = _call_basis_function(basis_type, degrees, pts)
        assert result.shape == (3, 4)

    def test_non_ndarray_input_tuple(self, basis_type: BasisType) -> None:
        """Test with tuple input (converted to ndarray internally)."""
        degrees = [2]
        pts = ((0.0,), (0.5,), (1.0,))
        result = _call_basis_function(basis_type, degrees, pts)
        assert result.shape == (3, 3)

    def test_non_2d_array_raises_error(self, basis_type: BasisType) -> None:
        """Test that non-2D array input raises ValueError."""
        degrees = [1, 1]
        pts_1d = np.array([0.0, 0.5, 1.0], dtype=np.float64)
        with pytest.raises(ValueError, match="Input points must be a 2D array"):
            _call_basis_function(basis_type, degrees, pts_1d)
        pts_3d = np.array([[[0.0, 0.0]]], dtype=np.float64)
        with pytest.raises(ValueError, match="Input points must be a 2D array"):
            _call_basis_function(basis_type, degrees, pts_3d)

    def test_int_dtype_converted_to_float64(self, basis_type: BasisType) -> None:
        """Test that integer dtype is converted to float64."""
        degrees = [1, 1]
        pts = np.array([[0, 0], [1, 1]], dtype=np.int32)
        result = _call_basis_function(basis_type, degrees, pts)
        assert result.dtype == np.float64

    def test_empty_dimension_raises_error(self, basis_type: BasisType) -> None:
        """Test that empty dimension (dim < 1) raises ValueError."""
        degrees = [1]
        pts = np.array([[]], dtype=np.float64).reshape(1, 0)
        with pytest.raises(ValueError, match="The dimension of the points must be at least 1"):
            _call_basis_function(basis_type, degrees, pts)

    def test_mismatched_degrees_dimension_raises_error(self, basis_type: BasisType) -> None:
        """Test that mismatched number of degrees vs dimension raises ValueError."""
        degrees = [1, 1, 1]
        pts = np.array([[0.0, 0.0], [0.5, 0.5]], dtype=np.float64)
        with pytest.raises(
            ValueError, match="The number of evaluators must be equal to the dimension"
        ):
            _call_basis_function(basis_type, degrees, pts)


@pytest.mark.parametrize(
    "variant",
    [
        LagrangeVariant.EQUISPACES,
        LagrangeVariant.GAUSS_LEGENDRE,
        LagrangeVariant.GAUSS_LOBATTO_LEGENDRE,
        LagrangeVariant.CHEBYSHEV_1ST,
        LagrangeVariant.CHEBYSHEV_2ND,
    ],
)
def test_2d_array_single_degree_lagrange_variants(variant: LagrangeVariant) -> None:
    """Test 2D array input with same degree in both dimensions for Lagrange variants."""
    degrees = [2, 2]
    pts = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]], dtype=np.float64)
    result = _call_basis_function(BasisType.LAGRANGE, degrees, pts, variant)
    # Should have shape (n_points, n_basis_functions) = (3, 9) for degree 2 in 2D
    assert result.shape == (3, 9)
    # Verify by comparing with manual computation
    basis_x = _call_basis_1d_function(BasisType.LAGRANGE, 2, pts[:, 0], variant)
    basis_y = _call_basis_1d_function(BasisType.LAGRANGE, 2, pts[:, 1], variant)
    expected = np.einsum("pi,pj->pij", basis_x, basis_y).reshape(3, 9)
    rtol = get_default_tolerance(np.float64)
    nptest.assert_allclose(result, expected, rtol=rtol, atol=0.0)


@pytest.mark.parametrize(
    "variant",
    [
        LagrangeVariant.EQUISPACES,
        LagrangeVariant.GAUSS_LEGENDRE,
        LagrangeVariant.GAUSS_LOBATTO_LEGENDRE,
        LagrangeVariant.CHEBYSHEV_1ST,
        LagrangeVariant.CHEBYSHEV_2ND,
    ],
)
def test_partition_of_unity_lagrange_variants(variant: LagrangeVariant) -> None:
    """Test that multi-dimensional Lagrange basis functions sum to 1 for all variants."""
    degrees = [2, 2]
    pts = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]], dtype=np.float64)
    result = _call_basis_function(BasisType.LAGRANGE, degrees, pts, variant)
    sums = np.sum(result, axis=-1)
    rtol = get_default_tolerance(np.float64)
    nptest.assert_allclose(sums, 1.0, rtol=rtol, atol=0.0)
