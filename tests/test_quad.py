"""Tests for 1D quadrature rules in pantr.quad."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import numpy as np
import numpy.testing as nptest
import numpy.typing as npt
import pytest
from numpy.polynomial import chebyshev

from pantr.quad import (
    get_chebyshev_gauss_1st_kind_quadrature_1D,
    get_chebyshev_gauss_2nd_kind_quadrature_1D,
    get_gauss_legendre_quadrature_1D,
    get_gauss_lobatto_legendre_quadrature_1D,
    get_trapezoidal_quadrature_1D,
)
from pantr.tolerance import get_conservative_tolerance, get_default_tolerance, get_strict_tolerance


def _integrate_polynomial_on_unit_interval(
    power: int,
    nodes: npt.NDArray[np.floating[Any]],
    weights: npt.NDArray[np.floating[Any]],
) -> np.floating[Any]:
    vals = nodes**power
    result = np.sum(weights * vals, dtype=np.result_type(nodes.dtype, weights.dtype))
    return cast(np.floating[Any], result)


class TestTrapezoidal:
    """Tests for get_trapezoidal_quadrature_1D."""

    def test_invalid_n_pts_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            get_trapezoidal_quadrature_1D(0)

    def test_invalid_dtype_raises(self) -> None:
        with pytest.raises(ValueError, match="float32 or float64"):
            get_trapezoidal_quadrature_1D(2, np.int32)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_npts_one_midpoint_and_unit_weight(self, dtype: npt.DTypeLike) -> None:
        nodes, weights = get_trapezoidal_quadrature_1D(1, dtype)
        nptest.assert_allclose(nodes, np.array([0.5], dtype=dtype))
        nptest.assert_allclose(weights, np.array([1.0], dtype=dtype))
        assert nodes.dtype == np.dtype(dtype)
        assert weights.dtype == np.dtype(dtype)

    @pytest.mark.parametrize("n_pts", [2, 5, 11])
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_partition_and_end_weights(self, n_pts: int, dtype: npt.DTypeLike) -> None:
        nodes, weights = get_trapezoidal_quadrature_1D(n_pts, dtype)
        # nodes in [0, 1]
        assert np.all((nodes >= 0.0) & (nodes <= 1.0))
        # weights sum to 1
        nptest.assert_allclose(np.sum(weights), np.array(1.0, dtype=dtype))
        if n_pts > 1:
            h = np.array(1.0 / (n_pts - 1), dtype=dtype)
            nptest.assert_allclose(weights[1:-1], h)
            nptest.assert_allclose(weights[[0, -1]], 0.5 * h)


class TestGaussLegendre:
    """Tests for get_gauss_legendre_quadrature_1D."""

    def test_invalid_n_pts_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            get_gauss_legendre_quadrature_1D(0)

    def test_invalid_dtype_raises(self) -> None:
        with pytest.raises(ValueError, match="float32 or float64"):
            get_gauss_legendre_quadrature_1D(2, np.int32)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_basic_properties(self, dtype: npt.DTypeLike) -> None:
        nodes, weights = get_gauss_legendre_quadrature_1D(4, dtype)
        assert nodes.dtype == np.dtype(dtype)
        assert weights.dtype == np.dtype(dtype)
        assert np.all((nodes >= 0.0) & (nodes <= 1.0))
        assert np.all(weights > 0.0)
        nptest.assert_allclose(
            np.sum(weights, dtype=np.float64), 1.0, rtol=get_strict_tolerance(dtype)
        )

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_polynomial_exactness(self, dtype: npt.DTypeLike) -> None:
        # n points should integrate polynomials up to degree 2n-1 exactly
        n = 4
        nodes, weights = get_gauss_legendre_quadrature_1D(n, dtype)
        rtol = get_default_tolerance(dtype)
        for p in range(2 * n):  # inclusive upper bound 2n-1
            approx = _integrate_polynomial_on_unit_interval(p, nodes, weights)
            exact = 1.0 / (p + 1)
            nptest.assert_allclose(approx, np.array(exact, dtype=dtype), rtol=rtol, atol=0.0)


class TestGaussLobattoLegendre:
    """Tests for get_gauss_lobatto_legendre_quadrature_1D."""

    def test_invalid_n_pts_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            get_gauss_lobatto_legendre_quadrature_1D(1)

    def test_invalid_dtype_raises(self) -> None:
        with pytest.raises(ValueError, match="float32 or float64"):
            get_gauss_lobatto_legendre_quadrature_1D(2, np.int32)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_endpoints_and_sum_weights(self, dtype: npt.DTypeLike) -> None:
        nodes, weights = get_gauss_lobatto_legendre_quadrature_1D(4, dtype)
        # Endpoints included
        nptest.assert_allclose(nodes[0], np.array(0.0, dtype=dtype))
        nptest.assert_allclose(nodes[-1], np.array(1.0, dtype=dtype))
        # weights positive and sum to 1
        assert np.all(weights > 0.0)
        nptest.assert_allclose(
            np.sum(weights, dtype=np.float64), 1.0, rtol=get_strict_tolerance(dtype)
        )

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_polynomial_exactness(self, dtype: npt.DTypeLike) -> None:
        # Degree of exactness: 2n-3
        n = 5
        nodes, weights = get_gauss_lobatto_legendre_quadrature_1D(n, dtype)
        rtol = get_conservative_tolerance(dtype)
        for p in range(2 * n - 2):  # inclusive upper bound 2n-3
            approx = _integrate_polynomial_on_unit_interval(p, nodes, weights)
            exact = 1.0 / (p + 1)
            nptest.assert_allclose(approx, np.array(exact, dtype=dtype), rtol=rtol, atol=0.0)


class TestChebyshevGaussFirstKind:
    """Tests for get_chebyshev_gauss_1st_kind_quadrature_1D."""

    def test_invalid_n_pts_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            get_chebyshev_gauss_1st_kind_quadrature_1D(0)

    def test_invalid_dtype_raises(self) -> None:
        with pytest.raises(ValueError, match="float32 or float64"):
            get_chebyshev_gauss_1st_kind_quadrature_1D(2, np.int32)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_npts_one_midpoint_and_weight_sum(self, dtype: npt.DTypeLike) -> None:
        nodes, weights = get_chebyshev_gauss_1st_kind_quadrature_1D(1, dtype)
        # cheb1 at n=1 returns node 0 on [-1,1] which maps to 0.5
        nptest.assert_allclose(nodes, np.array([0.5], dtype=dtype))
        # Sum of weights equals integral of 1/sqrt(1-x^2) over [0,1] = pi/2
        nptest.assert_allclose(
            np.sum(weights), np.array(np.pi / 2.0, dtype=dtype), rtol=get_strict_tolerance(dtype)
        )
        assert nodes.dtype == np.dtype(dtype)
        assert weights.dtype == np.dtype(dtype)

    @pytest.mark.parametrize("n_pts", [2, 5, 10])
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_nodes_and_total_weight(self, n_pts: int, dtype: npt.DTypeLike) -> None:
        nodes, weights = get_chebyshev_gauss_1st_kind_quadrature_1D(n_pts, dtype)
        # nodes are mapped chebpts1
        cheb1_t = cast(Callable[[int], npt.NDArray[np.float64]], chebyshev.chebpts1)
        mapped = ((cheb1_t(n_pts) + 1.0) * 0.5).astype(dtype)
        nptest.assert_allclose(nodes, mapped, rtol=get_strict_tolerance(dtype))
        # weights sum to pi/2 after scaling to [0,1]
        nptest.assert_allclose(
            np.sum(weights), np.array(np.pi / 2.0, dtype=dtype), rtol=get_strict_tolerance(dtype)
        )
        assert np.all((nodes >= 0.0) & (nodes <= 1.0))


class TestChebyshevGaussSecondKind:
    """Tests for get_chebyshev_gauss_2nd_kind_quadrature_1D."""

    def test_invalid_n_pts_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            get_chebyshev_gauss_2nd_kind_quadrature_1D(1)

    def test_invalid_dtype_raises(self) -> None:
        with pytest.raises(ValueError, match="float32 or float64"):
            get_chebyshev_gauss_2nd_kind_quadrature_1D(2, np.int32)

    @pytest.mark.parametrize("n_pts", [2, 5, 9])
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_nodes_endpoints_and_total_weight(self, n_pts: int, dtype: npt.DTypeLike) -> None:
        nodes, weights = get_chebyshev_gauss_2nd_kind_quadrature_1D(n_pts, dtype)
        # endpoints included for n>=2
        nptest.assert_allclose(nodes[0], np.array(0.0, dtype=dtype))
        nptest.assert_allclose(nodes[-1], np.array(1.0, dtype=dtype))
        # weights sum to (integral of sqrt(1-x^2) over [0,1]) = pi/4 after scaling
        nptest.assert_allclose(
            np.sum(weights), np.array(np.pi / 4.0, dtype=dtype), rtol=get_strict_tolerance(dtype)
        )
        assert np.all((nodes >= 0.0) & (nodes <= 1.0))
