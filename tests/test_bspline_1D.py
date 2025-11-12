"""Tests for bspline_1D module."""

import numpy as np
import pytest

from pantr._bspline_1D_impl import (
    _check_spline_info,
    _compute_num_basis_impl,
    _create_bspline_Bezier_extraction_operators_impl,
    _eval_basis_Cox_de_Boor_impl,
    _get_cardinal_intervals_impl,
    _get_last_knot_smaller_equal_impl,
    _get_multiplicity_of_first_knot_in_domain_impl,
    _get_unique_knots_and_multiplicity_impl,
    _is_in_domain_impl,
)
from pantr.bspline_1D import Bspline1D
from pantr.knots import (
    create_cardinal_Bspline_knot_vector,
    create_uniform_open_knot_vector,
    create_uniform_periodic_knot_vector,
)
from pantr.tolerance import get_strict_tolerance


class TestBspline1DInit:
    """Test Bspline1D initialization."""

    def test_valid_initialization(self) -> None:
        """Test valid Bspline1D initialization."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        degree = 2
        spline = Bspline1D(knots, degree)

        assert spline.degree == 2  # noqa: PLR2004
        assert spline.periodic is False
        np.testing.assert_array_equal(spline.knots, np.array(knots))

    def test_zero_degree_initialization(self) -> None:
        """Test valid Bspline1D initialization."""
        knots = [0.0, 1.0]
        degree = 0
        spline = Bspline1D(knots, degree)

        assert spline.degree == 0
        assert spline.periodic is False
        np.testing.assert_array_equal(spline.knots, np.array(knots))

    def test_periodic_initialization(self) -> None:
        """Test periodic Bspline1D initialization."""
        knots = create_uniform_periodic_knot_vector(num_intervals=3, degree=2, domain=(0.0, 1.0))
        degree = 2
        spline = Bspline1D(knots, degree, periodic=True)

        assert spline.degree == 2  # noqa: PLR2004
        assert spline.periodic is True

    def test_integer_knots_conversion(self) -> None:
        """Test that integer knots are converted to float64."""
        knots = [0, 0, 0, 1, 1, 1]
        degree = 2
        spline = Bspline1D(knots, degree)

        assert spline.dtype == np.float64
        np.testing.assert_array_equal(spline.knots, np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]))

    def test_negative_degree_error(self) -> None:
        """Test that negative degree raises ValueError."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        with pytest.raises(ValueError, match="degree must be non-negative"):
            Bspline1D(knots, -1)

    def test_insufficient_knots_error(self) -> None:
        """Test that insufficient knots raise ValueError."""
        knots = [0.0, 1.0]
        with pytest.raises(ValueError, match="knots must have at least"):
            Bspline1D(knots, 2)

    def test_non_decreasing_knots_error(self) -> None:
        """Test that non-decreasing knots raise ValueError."""
        knots = [0.0, 1.0, 0.5, 1.0, 1.0, 1.0]
        with pytest.raises(ValueError, match="knots must be non-decreasing"):
            Bspline1D(knots, 2)

    def test_invalid_knot_type_error(self) -> None:
        """Test that invalid knot type raises TypeError."""
        knots = "invalid"
        with pytest.raises(
            (TypeError, ValueError),
            match=r"knots must be a 1D numpy array or Python list|knots type must be float",
        ):
            Bspline1D(knots, 2)

    def test_snap_knots_disabled(self) -> None:
        """Test initialization with snap_knots disabled."""
        tol = get_strict_tolerance(np.float64)
        knots = [0.0, 0.0, 0.0 + tol, 1.0, 1.0, 1.0]
        degree = 2
        spline = Bspline1D(knots, degree, snap_knots=False)

        # Knots should remain unchanged
        np.testing.assert_array_equal(spline.knots, np.array(knots))


class TestBspline1DProperties:
    """Test Bspline1D properties."""

    def test_degree_property(self) -> None:
        """Test degree property."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        degree = 2
        spline = Bspline1D(knots, degree)
        assert spline.degree == degree

    def test_knots_property(self) -> None:
        """Test knots property."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        degree = 2
        spline = Bspline1D(knots, degree)
        np.testing.assert_array_equal(spline.knots, np.array(knots))

    def test_periodic_property(self) -> None:
        """Test periodic property."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        degree = 2
        spline = Bspline1D(knots, degree, periodic=True)
        assert spline.periodic is True

    def test_tolerance_property(self) -> None:
        """Test tolerance property."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        degree = 2
        spline = Bspline1D(knots, degree)
        assert spline.tolerance > 0

    def test_dtype_property(self) -> None:
        """Test dtype property."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        degree = 2
        spline = Bspline1D(knots, degree)
        assert spline.dtype == np.float64


class TestBspline1DMethods:
    """Test Bspline1D methods."""

    def test_get_num_basis_non_periodic(self) -> None:
        """Test get_num_basis for non-periodic spline."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        degree = 2
        spline = Bspline1D(knots, degree)
        assert spline.num_basis == 3  # noqa: PLR2004

    def test_get_num_basis_periodic(self) -> None:
        """Test get_num_basis for periodic spline."""
        degree = 2
        knots = create_uniform_periodic_knot_vector(
            num_intervals=3, degree=degree, domain=(0.0, 1.0)
        )
        spline = Bspline1D(knots, degree, periodic=True)
        # For periodic splines, the number of basis functions is reduced
        assert spline.num_basis == 3  # noqa: PLR2004

    def test_get_unique_knots_and_multiplicity_full(self) -> None:
        """Test _get_unique_knots_and_multiplicity for full knot vector."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        degree = 2
        spline = Bspline1D(knots, degree)
        unique_knots, multiplicities = spline.get_unique_knots_and_multiplicity(in_domain=False)

        expected_unique = np.array([0.0, 1.0])
        expected_mults = np.array([3, 3])
        np.testing.assert_array_almost_equal(unique_knots, expected_unique)
        np.testing.assert_array_equal(multiplicities, expected_mults)

    def test_get_unique_knots_and_multiplicity_domain(self) -> None:
        """Test _get_unique_knots_and_multiplicity for domain only."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        degree = 2
        spline = Bspline1D(knots, degree)
        unique_knots, multiplicities = spline.get_unique_knots_and_multiplicity(in_domain=True)

        expected_unique = np.array([0.0, 1.0])
        expected_mults = np.array([3, 3])
        np.testing.assert_array_almost_equal(unique_knots, expected_unique)
        np.testing.assert_array_equal(multiplicities, expected_mults)

    def test_get_num_intervals(self) -> None:
        """Test get_num_intervals method."""
        num_intervals = 2
        degree = 2
        knots = create_uniform_open_knot_vector(num_intervals, degree)
        spline = Bspline1D(knots, degree)
        assert spline.num_intervals == num_intervals

    def test_get_domain(self) -> None:
        """Test get_domain method."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        degree = 2
        spline = Bspline1D(knots, degree)
        domain = spline.domain
        np.testing.assert_allclose(domain, (knots[degree], knots[-degree - 1]))

    def test_has_left_end_open_true(self) -> None:
        """Test has_left_end_open returns True for open left end."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        degree = 2
        spline = Bspline1D(knots, degree)
        assert spline.has_left_end_open() is True

    def test_has_left_end_open_false(self) -> None:
        """Test has_left_end_open returns False for non-open left end."""
        knots = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        degree = 2
        spline = Bspline1D(knots, degree)
        assert spline.has_left_end_open() is False

    def test_has_right_end_open_true(self) -> None:
        """Test has_right_end_open returns True for open right end."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        degree = 2
        spline = Bspline1D(knots, degree)
        assert spline.has_right_end_open() is True

    def test_has_right_end_open_false(self) -> None:
        """Test has_right_end_open returns False for non-open right end."""
        knots = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        degree = 2
        spline = Bspline1D(knots, degree)
        assert spline.has_right_end_open() is False

    def test_has_open_knots_true(self) -> None:
        """Test has_open_knots returns True when both ends are open."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        degree = 2
        spline = Bspline1D(knots, degree)
        assert spline.has_open_knots() is True

    def test_has_open_knots_false(self) -> None:
        """Test has_open_knots returns False when ends are not open."""
        knots = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        degree = 2
        spline = Bspline1D(knots, degree)
        assert spline.has_open_knots() is False

    def test_has_Bezier_like_knots_true(self) -> None:
        """Test has_Bezier_like_knots returns True for Bézier-like configuration."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        degree = 2
        spline = Bspline1D(knots, degree)
        assert bool(spline.has_Bezier_like_knots()) is True

    def test_has_Bezier_like_knots_false(self) -> None:
        """Test has_Bezier_like_knots returns False for non-Bézier-like configuration."""
        knots = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]
        degree = 2
        spline = Bspline1D(knots, degree)
        assert bool(spline.has_Bezier_like_knots()) is False

    def test_has_Bezier_like_knots_periodic_false(self) -> None:
        """Test has_Bezier_like_knots returns False for periodic splines."""
        # Use a valid periodic knot vector
        degree = 2
        knots = create_uniform_periodic_knot_vector(
            num_intervals=3, degree=degree, domain=(0.0, 1.0)
        )
        spline = Bspline1D(knots, degree, periodic=True)
        assert bool(spline.has_Bezier_like_knots()) is False

    def test_get_cardinal_intervals(self) -> None:
        """Test get_cardinal_intervals method."""
        knots = [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0]
        degree = 2
        spline = Bspline1D(knots, degree)
        result = spline.get_cardinal_intervals()

        # Should have 4 intervals, middle ones should be cardinal
        expected = np.array([False, True, True, False])
        np.testing.assert_array_equal(result, expected)


class TestBspline1DWithKnotGenerators:
    """Test Bspline1D with knot vector generators."""

    def test_with_uniform_open_knot_vector(self) -> None:
        """Test Bspline1D with uniform open knot vector."""
        knots = create_uniform_open_knot_vector(num_intervals=2, degree=2, domain=(0.0, 1.0))
        degree = 2
        spline = Bspline1D(knots, degree)

        assert spline.degree == degree
        assert spline.periodic is False
        assert spline.has_open_knots() is True
        assert spline.domain == (knots[degree], knots[-degree - 1])

    def test_with_uniform_periodic_knot_vector(self) -> None:
        """Test Bspline1D with uniform periodic knot vector."""
        degree = 2
        knots = create_uniform_periodic_knot_vector(
            num_intervals=3, degree=degree, domain=(0.0, 1.0)
        )
        spline = Bspline1D(knots, degree, periodic=True)

        assert spline.degree == degree
        assert spline.periodic is True
        assert spline.domain == (knots[degree], knots[-degree - 1])

    def test_with_cardinal_bspline_knot_vector(self) -> None:
        """Test Bspline1D with cardinal B-spline knot vector."""
        degree = 2
        knots = create_cardinal_Bspline_knot_vector(2, degree)
        spline = Bspline1D(knots, degree)

        assert spline.degree == degree
        assert spline.periodic is False
        assert spline.domain == (knots[degree], knots[-degree - 1])


class TestBspline1DEdgeCases:
    """Test Bspline1D edge cases."""

    def test_degree_zero(self) -> None:
        """Test Bspline1D with degree 0."""
        knots = [0.0, 1.0]
        degree = 0
        spline = Bspline1D(knots, degree)

        assert spline.degree == degree
        assert spline.num_basis == 1
        np.testing.assert_allclose(spline.domain, (knots[degree], knots[-degree - 1]))

    def test_single_interval(self) -> None:
        """Test Bspline1D with single interval."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        degree = 2
        spline = Bspline1D(knots, degree)

        assert spline.num_intervals == 1
        assert bool(spline.has_Bezier_like_knots()) is True

    def test_high_degree(self) -> None:
        """Test Bspline1D with high degree."""
        degree = 5
        knots = [0.0] * (degree + 1) + [1.0] * (degree + 1)
        spline = Bspline1D(knots, degree)

        assert spline.degree == degree
        assert spline.num_basis == (degree + 1)
        assert bool(spline.has_Bezier_like_knots()) is True

    def test_float32_precision(self) -> None:
        """Test Bspline1D with float32 precision."""
        knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float32)
        degree = 2
        spline = Bspline1D(knots, degree)

        assert spline.dtype == np.float32
        assert spline.tolerance > 0


class TestBspline1DIntegration:
    """Integration tests for Bspline1D."""

    def test_consistency_across_methods(self) -> None:
        """Test consistency across different Bspline1D methods."""
        knots = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]
        degree = 2
        spline = Bspline1D(knots, degree)

        # Test that domain indices are consistent
        domain = spline.domain
        unique_knots, _ = spline.get_unique_knots_and_multiplicity(in_domain=True)

        assert domain[0] == unique_knots[0]
        np.testing.assert_array_almost_equal(domain[1], unique_knots[-1])

        # Test that number of intervals is consistent
        num_intervals = spline.num_intervals
        assert num_intervals == len(unique_knots) - 1

    def test_periodic_vs_non_periodic_consistency(self) -> None:
        """Test consistency between periodic and non-periodic versions."""
        # Create equivalent knot vectors
        degree = 2
        knots_open = create_uniform_open_knot_vector(
            num_intervals=3, degree=degree, domain=(0.0, 1.0)
        )
        knots_periodic = create_uniform_periodic_knot_vector(
            num_intervals=3, degree=degree, domain=(0.0, 1.0)
        )

        spline_open = Bspline1D(knots_open, degree, periodic=False)
        spline_periodic = Bspline1D(knots_periodic, degree, periodic=True)

        # Both should have the same domain
        assert spline_open.domain == spline_periodic.domain

        # Both should have the same number of intervals
        assert spline_open.num_intervals == spline_periodic.num_intervals

        # But different number of basis functions
        assert spline_open.num_basis != spline_periodic.num_basis

    def test_knot_snapping_consistency(self) -> None:
        """Test that knot snapping doesn't break consistency."""
        # Create knots with small numerical differences
        knots = [0.0, 0.0, 0.0, 0.5000000001, 1.0, 1.0, 1.0]
        degree = 2
        spline = Bspline1D(knots, degree, snap_knots=True)

        # After snapping, should still be valid
        assert spline.num_basis > 0
        assert spline.num_intervals > 0

        # Domain should be well-defined
        domain = spline.domain
        assert domain[0] < domain[1]


class TestAssertSplineInfo:
    """Test the _check_spline_info validation function."""

    def test_valid_inputs(self) -> None:
        """Test that valid inputs don't raise assertions."""
        knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)
        degree = 2
        _check_spline_info(knots, degree)

    def test_invalid_degree(self) -> None:
        """Test that negative degree raises AssertionError."""
        knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)
        degree = -1
        with pytest.raises(AssertionError, match="degree must be non-negative"):
            _check_spline_info(knots, degree)

    def test_insufficient_knots(self) -> None:
        """Test that insufficient knots raise AssertionError."""
        knots = np.array([0.0, 1.0], dtype=np.float64)
        degree = 2
        with pytest.raises(AssertionError, match="knots must have at least"):
            _check_spline_info(knots, degree)

    def test_non_decreasing_knots(self) -> None:
        """Test that non-decreasing knots raise AssertionError."""
        knots = np.array([0.0, 1.0, 0.5, 1.0, 1.0, 1.0], dtype=np.float64)
        degree = 2
        with pytest.raises(AssertionError, match="knots must be non-decreasing"):
            _check_spline_info(knots, degree)


class TestGetMultiplicityOfFirstKnotInDomain:
    """Test the _get_multiplicity_of_first_knot_in_domain_impl function."""

    def test_open_knot_vector(self) -> None:
        """Test multiplicity calculation for open knot vector."""
        knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)
        degree = 2
        tol = 1e-10
        result = _get_multiplicity_of_first_knot_in_domain_impl(knots, degree, tol)
        # First knot in domain (index 2) has multiplicity 3
        assert result == 3  # noqa: PLR2004

    def test_periodic_knot_vector(self) -> None:
        """Test multiplicity calculation for periodic knot vector."""
        knots = np.array([-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0], dtype=np.float64)
        degree = 2
        tol = 1e-10
        result = _get_multiplicity_of_first_knot_in_domain_impl(knots, degree, tol)
        assert result == 1  # First knot in domain (index 2) has multiplicity 1

    def test_negative_tolerance_error(self) -> None:
        """Test that negative tolerance raises AssertionError."""
        knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)
        degree = 2
        with pytest.raises(AssertionError, match="tol must be positive"):
            _get_multiplicity_of_first_knot_in_domain_impl(knots, degree, -1.0)


class TestGetUniqueKnotsAndMultiplicity:
    """Test the _get_unique_knots_and_multiplicity_impl function."""

    def test_open_knot_vector_full(self) -> None:
        """Test unique knots extraction for full knot vector."""
        knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)
        degree = 2
        tol = 1e-10
        unique_knots, multiplicities = _get_unique_knots_and_multiplicity_impl(
            knots, degree, tol, in_domain=False
        )
        expected_unique = np.array([0.0, 1.0])
        expected_mults = np.array([3, 3])
        np.testing.assert_array_almost_equal(unique_knots, expected_unique)
        np.testing.assert_array_equal(multiplicities, expected_mults)

    def test_open_knot_vector_domain_only(self) -> None:
        """Test unique knots extraction for domain only."""
        knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)
        degree = 2
        tol = 1e-10
        unique_knots, multiplicities = _get_unique_knots_and_multiplicity_impl(
            knots, degree, tol, in_domain=True
        )
        expected_unique = np.array([0.0, 1.0])
        expected_mults = np.array([3, 3])
        np.testing.assert_array_almost_equal(unique_knots, expected_unique)
        np.testing.assert_array_equal(multiplicities, expected_mults)

    def test_periodic_knot_vector(self) -> None:
        """Test unique knots extraction for periodic knot vector."""
        knots = np.array([-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0], dtype=np.float64)
        degree = 2
        tol = 1e-10
        unique_knots, multiplicities = _get_unique_knots_and_multiplicity_impl(
            knots, degree, tol, in_domain=True
        )
        expected_unique = np.array([0.0, 0.5, 1.0])
        expected_mults = np.array([1, 1, 1])
        np.testing.assert_array_almost_equal(unique_knots, expected_unique)
        np.testing.assert_array_equal(multiplicities, expected_mults)


class TestIsInDomain:
    """Test the _is_in_domain_impl function."""

    def test_points_in_domain(self) -> None:
        """Test that points within domain return True."""
        knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)
        degree = 2
        pts = np.array([0.0, 0.5, 1.0], dtype=np.float64)
        tol = 1e-10
        result = _is_in_domain_impl(knots, degree, pts, tol)
        np.testing.assert_array_equal(result, [True, True, True])

    def test_points_outside_domain(self) -> None:
        """Test that points outside domain return False."""
        knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)
        degree = 2
        pts = np.array([-0.1, 1.1], dtype=np.float64)
        tol = 1e-10
        result = _is_in_domain_impl(knots, degree, pts, tol)
        np.testing.assert_array_equal(result, [False, False])

    def test_boundary_points(self) -> None:
        """Test that boundary points return True."""
        knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)
        degree = 2
        pts = np.array([0.0, 1.0], dtype=np.float64)
        tol = 1e-10
        result = _is_in_domain_impl(knots, degree, pts, tol)
        np.testing.assert_array_equal(result, [True, True])

    def test_empty_points_array(self) -> None:
        """Test that empty points array raises AssertionError."""
        knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)
        degree = 2
        pts = np.array([], dtype=np.float64)
        tol = 1e-10
        with pytest.raises(AssertionError, match="pts must have at least one element"):
            _is_in_domain_impl(knots, degree, pts, tol)


class TestComputeNumBasis:
    """Test the compute_num_basis_impl function."""

    def test_non_periodic_open(self) -> None:
        """Test basis count for non-periodic open knot vector."""
        knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)
        degree = 2
        periodic = False
        tol = 1e-10
        result = _compute_num_basis_impl(knots, degree, periodic, tol)
        # knots.size - degree - 1 = 6 - 2 - 1 = 3
        assert result == 3  # noqa: PLR2004

    def test_periodic_knot_vector(self) -> None:
        """Test basis count for periodic knot vector."""
        knots = np.array([-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0], dtype=np.float64)
        degree = 2
        periodic = True
        tol = 1e-10
        result = _compute_num_basis_impl(knots, degree, periodic, tol)
        # For periodic: num_basis = knots.size - degree - 1 - regularity - 1
        # regularity = degree - multiplicity_of_first_knot_in_domain
        # multiplicity_of_first_knot_in_domain = 1
        # regularity = 2 - 1 = 1
        # num_basis = 7 - 2 - 1 - 1 - 1 = 2
        assert result == 2  # noqa: PLR2004

    def test_negative_tolerance_error(self) -> None:
        """Test that negative tolerance raises AssertionError."""
        knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)
        degree = 2
        with pytest.raises(AssertionError, match="tol must be positive"):
            _compute_num_basis_impl(knots, degree, False, -1.0)


class TestGetLastKnotSmallerEqual:
    """Test the _get_last_knot_smaller_equal_impl function."""

    def test_basic_functionality(self) -> None:
        """Test basic knot index finding."""
        knots = np.array([0.0, 0.5, 1.0, 1.5, 2.0], dtype=np.float64)
        pts = np.array([0.3, 0.7, 1.2, 1.8], dtype=np.float64)
        result = _get_last_knot_smaller_equal_impl(knots, pts)
        expected = np.array([0, 1, 2, 3])  # Indices of knots <= pts
        np.testing.assert_array_equal(result, expected)

    def test_knots_with_repetitions(self) -> None:
        """Test knots with repetitions index finding."""
        knots = np.array([0.0, 0.5, 1.0, 1.0, 1.5, 2.0], dtype=np.float64)
        pts = np.array([0.3, 0.7, 1.2, 1.8], dtype=np.float64)
        result = _get_last_knot_smaller_equal_impl(knots, pts)
        expected = np.array([0, 1, 3, 4])  # Indices of knots <= pts
        np.testing.assert_array_equal(result, expected)

    def test_exact_knot_matches(self) -> None:
        """Test when points exactly match knots."""
        knots = np.array([0.0, 0.5, 1.0, 1.5, 2.0], dtype=np.float64)
        pts = np.array([0.0, 0.5, 1.0], dtype=np.float64)
        result = _get_last_knot_smaller_equal_impl(knots, pts)
        expected = np.array([0, 1, 2])
        np.testing.assert_array_equal(result, expected)

    def test_non_decreasing_knots_error(self) -> None:
        """Test that non-decreasing knots raise AssertionError."""
        knots = np.array([0.0, 1.0, 0.5, 2.0], dtype=np.float64)
        pts = np.array([0.5], dtype=np.float64)
        with pytest.raises(AssertionError, match="knots must be non-decreasing"):
            _get_last_knot_smaller_equal_impl(knots, pts)


class TestEvaluateBasisCoxDeBoor:
    """Test the _eval_basis_Cox_de_Boor_impl function."""

    def test_bezier_like_evaluation(self) -> None:
        """Test evaluation for Bézier-like knot vector."""
        knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)
        degree = 2
        periodic = False
        tol = 1e-10
        pts = np.array([0.0, 0.5, 1.0], dtype=np.float64)

        basis, first_basis = _eval_basis_Cox_de_Boor_impl(knots, degree, periodic, tol, pts)

        # Check shape
        assert basis.shape == (3, 3)
        assert first_basis.shape == (3,)

        # Check partition of unity
        sums = np.sum(basis, axis=1)
        np.testing.assert_array_almost_equal(sums, np.ones_like(sums))

    def test_general_knot_vector(self) -> None:
        """Test evaluation for general knot vector."""
        knots = np.array([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0], dtype=np.float64)
        degree = 2
        periodic = False
        tol = 1e-10
        pts = np.array([0.25, 0.75], dtype=np.float64)

        basis, first_basis = _eval_basis_Cox_de_Boor_impl(knots, degree, periodic, tol, pts)

        # Check shape
        assert basis.shape == (2, 3)
        assert first_basis.shape == (2,)

        # Check partition of unity
        sums = np.sum(basis, axis=1)
        np.testing.assert_array_almost_equal(sums, np.ones_like(sums))

    def test_periodic_evaluation(self) -> None:
        """Test evaluation for periodic knot vector."""
        knots = np.array([-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0], dtype=np.float64)
        degree = 2
        periodic = True
        tol = 1e-10
        pts = np.array([0.0, 0.5, 1.0], dtype=np.float64)

        basis, first_basis = _eval_basis_Cox_de_Boor_impl(knots, degree, periodic, tol, pts)

        # Check shape
        assert basis.shape == (3, 3)
        assert first_basis.shape == (3,)

        # Check partition of unity
        sums = np.sum(basis, axis=1)
        np.testing.assert_array_almost_equal(sums, np.ones_like(sums))

    def test_outside_domain_error(self) -> None:
        """Test that points outside domain raise AssertionError."""
        knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)
        degree = 2
        periodic = False
        tol = 1e-10
        pts = np.array([-0.1], dtype=np.float64)

        with pytest.raises(AssertionError):
            _eval_basis_Cox_de_Boor_impl(knots, degree, periodic, tol, pts)


class TestGetCardinalIntervals:
    """Test the _get_cardinal_intervals_impl function."""

    def test_uniform_knot_vector(self) -> None:
        """Test cardinal intervals for uniform knot vector."""
        knots = np.array([0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0], dtype=np.float64)
        degree = 2
        tol = 1e-10
        result = _get_cardinal_intervals_impl(knots, degree, tol)

        # Should have 4 intervals, middle ones should be cardinal
        expected = np.array([False, True, True, False])
        np.testing.assert_array_equal(result, expected)

    def test_non_uniform_knot_vector(self) -> None:
        """Test cardinal intervals for non-uniform knot vector."""
        knots = np.array([0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 5.0, 5.0, 5.0], dtype=np.float64)
        degree = 2
        tol = 1e-10
        result = _get_cardinal_intervals_impl(knots, degree, tol)

        # Should have 4 intervals, some might be cardinal due to uniform spacing in some regions
        expected = np.array([False, True, False, False])
        np.testing.assert_array_equal(result, expected)

    def test_all_multiplicity_greater_than_one(self) -> None:
        """Test when all knots have multiplicity > 1."""
        knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)
        degree = 2
        tol = 1e-10
        result = _get_cardinal_intervals_impl(knots, degree, tol)

        # Should return all False
        expected = np.array([False])
        np.testing.assert_array_equal(result, expected)


class TestCreateBsplineBezierExtractionOperators:
    """Test the _create_bspline_Bezier_extraction_operators_impl function."""

    def test_bezier_like_knot_vector(self) -> None:
        """Test extraction operators for Bézier-like knot vector."""
        knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)
        degree = 2
        tol = 1e-10
        result = _create_bspline_Bezier_extraction_operators_impl(knots, degree, tol)

        # Should have 1 interval, 3x3 extraction matrix
        assert result.shape == (1, 3, 3)

        # For Bézier-like knots, extraction matrix should be identity
        np.testing.assert_array_almost_equal(result[0], np.eye(3))

    def test_general_knot_vector(self) -> None:
        """Test extraction operators for general knot vector."""
        knots = np.array([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0], dtype=np.float64)
        degree = 2
        tol = 1e-10
        result = _create_bspline_Bezier_extraction_operators_impl(knots, degree, tol)

        # Should have 2 intervals, 3x3 extraction matrices
        assert result.shape == (2, 3, 3)

        # Check that matrices are not identity (since not Bézier-like)
        assert not np.allclose(result[0], np.eye(3))
        assert not np.allclose(result[1], np.eye(3))

    def test_negative_tolerance_error(self) -> None:
        """Test that negative tolerance raises AssertionError."""
        knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)
        degree = 2
        with pytest.raises(AssertionError, match="tol must be positive"):
            _create_bspline_Bezier_extraction_operators_impl(knots, degree, -1.0)
