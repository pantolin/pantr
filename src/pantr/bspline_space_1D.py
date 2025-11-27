"""BsplineSpace1D utilities for open, floating, and periodic knot vectors."""

import functools
from typing import cast

import numpy as np
from numpy import typing as npt

from ._bspline_space_1D_impl import (
    _get_Bspline_cardinal_intervals_1D_impl,
    _get_Bspline_num_basis_1D_impl,
    _get_knots_ends_and_dtype,
    _get_unique_knots_and_multiplicity_impl,
    _tabulate_Bspline_basis_1D_impl,
    _tabulate_Bspline_Bezier_1D_extraction_impl,
    _tabulate_Bspline_cardinal_1D_extraction_impl,
    _tabulate_Bspline_Lagrange_1D_extraction_impl,
    _validate_knot_input,
)
from .tolerance import get_strict_tolerance


def create_uniform_open_knot_vector(
    num_intervals: int,
    degree: int,
    continuity: int | None = None,
    domain: tuple[np.float32 | np.float64 | float, np.float32 | np.float64 | float] | None = None,
    dtype: npt.DTypeLike | None = None,
) -> npt.NDArray[np.float32 | np.float64]:
    """Create a uniform open knot vector.

    An open knot vector has the first and last knots repeated (degree+1) times,
    ensuring the B-spline interpolates the first and last control points.

    Args:
        num_intervals (int): Number of intervals in the domain. Must be non-negative.
        degree (int): B-spline degree. Must be non-negative.
        continuity (Optional[int]): Continuity level at interior knots.
            Must be between -1 and degree-1. Defaults to degree-1 (maximum continuity).
        domain (Optional[tuple[np.float32 | np.float64 | float, np.float32 | np.float64 | float]]):
            Domain boundaries as (start, end). Defaults to (0.0, 1.0) if not provided.
        dtype (Optional[np.dtype]): Data type for the knot vector.
            If None, inferred from start/end or defaults to float64.

    Returns:
        npt.NDArray[np.floating]: Open knot vector with uniform spacing.

    Raises:
        ValueError: If any parameter is invalid.

    Example:
        >>> create_uniform_open_knot_vector(2, 2, domain=(0.0, 1.0))
        array([0., 0., 0., 0.5, 1., 1., 1.])
    """
    start_value: np.float32 | np.float64 | None
    end_value: np.float32 | np.float64 | None
    if domain is None:
        start_value = None
        end_value = None
    else:
        start_raw, end_raw = domain
        start_value = start_raw if isinstance(start_raw, np.floating) else np.float64(start_raw)
        end_value = end_raw if isinstance(end_raw, np.floating) else np.float64(end_raw)

    start, end, dtype = _get_knots_ends_and_dtype(start_value, end_value, dtype)

    continuity = degree - 1 if continuity is None else continuity

    _validate_knot_input(
        num_intervals,
        degree,
        continuity,
        (start, end),
        dtype,
    )

    unique_knots = np.linspace(start, end, num_intervals + 1, dtype=dtype)
    knots = np.array([start] * (degree + 1), dtype)

    interior_multiplicity = degree - continuity
    for knot in unique_knots[1:-1]:
        knots = np.append(knots, [knot] * interior_multiplicity)

    knots = np.append(knots, [end] * (degree + 1))

    return knots


def create_uniform_periodic_knot_vector(
    num_intervals: int,
    degree: int,
    continuity: int | None = None,
    domain: tuple[np.float32 | np.float64 | float, np.float32 | np.float64 | float] | None = None,
    dtype: npt.DTypeLike | None = np.float64,
) -> npt.NDArray[np.float32 | np.float64]:
    """Create a uniform periodic knot vector.

    A periodic knot vector extends beyond the domain boundaries to ensure
    periodicity of the B-spline basis functions.

    Args:
        num_intervals (int): Number of intervals in the domain. Must be non-negative.
        degree (int): B-spline degree. Must be non-negative.
        continuity (Optional[int]): Continuity level at interior knots.
            Must be between -1 and degree-1. Defaults to degree-1 (maximum continuity).
        domain (Optional[tuple[np.float32 | np.float64 | float, np.float32 | np.float64 | float]]):
            Domain boundaries as (start, end). Defaults to (0.0, 1.0) if not provided.
        dtype (Optional[np.dtype]): Data type for the knot vector.
            If None, inferred from start/end or defaults to float64.

    Returns:
        npt.NDArray[np.floating]: Periodic knot vector with uniform spacing.

    Raises:
        ValueError: If any parameter is invalid.

    Example:
        >>> create_uniform_periodic_knot_vector(2, 2, domain=(0.0, 1.0))
        array([-1. , -0.5,  0. ,  0.5,  1. ,  1.5,  2. ])
    """
    start_value: np.float32 | np.float64 | None
    end_value: np.float32 | np.float64 | None
    if domain is None:
        start_value = None
        end_value = None
    else:
        start_raw, end_raw = domain
        start_value = start_raw if isinstance(start_raw, np.floating) else np.float64(start_raw)
        end_value = end_raw if isinstance(end_raw, np.floating) else np.float64(end_raw)

    start, end, dtype = _get_knots_ends_and_dtype(start_value, end_value, dtype)
    continuity = degree - 1 if continuity is None else continuity

    _validate_knot_input(
        num_intervals,
        degree,
        continuity,
        (start, end),
        dtype,
    )

    # Create uniform spacing for unique interior knots
    unique_knots = np.linspace(start, end, num_intervals + 1, dtype=dtype)

    # Build knot vector with repetitions
    knots = np.array([], dtype=dtype)

    multiplicity = degree - continuity

    # Starting periodic knots.
    length = (end - start) / num_intervals
    knots = np.linspace(
        start - length * (degree - multiplicity + 1),
        start,
        degree + 2 - multiplicity,
        dtype=dtype,
    )[:-1]

    # Interior knots with specified multiplicity
    for knot in unique_knots:
        knots = np.append(knots, [knot] * multiplicity)

    # End periodic knots.
    knots = np.append(
        knots,
        np.linspace(
            end,
            end + length * (degree - multiplicity + 1),
            degree + 2 - multiplicity,
            dtype=dtype,
        )[1:],
    )

    return knots


def create_cardinal_Bspline_knot_vector(
    num_intervals: int,
    degree: int,
    dtype: npt.DTypeLike = np.float64,
) -> npt.NDArray[np.float32 | np.float64]:
    """Create a knot vector for cardinal B-spline basis functions.

    Cardinal B-splines are B-splines defined on uniform knot vectors with
    maximum continuity, where the basis functions in the central region
    have the same shape and are translated versions of each other.

    Args:
        num_intervals (int): Number of intervals in the domain. Must be at least 1.
        degree (int): B-spline degree. Must be non-negative.
        dtype (npt.DTypeLike): Data type for the knot vector.
            It must be either float32 or float64. Defaults to np.float64.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Cardinal B-spline knot vector
            with uniform spacing.

    Raises:
        ValueError: If num_intervals < 1, degree < 0, or dtype is not float32/float64.

    Example:
        >>> create_cardinal_Bspline_knot_vector(2, 2)
        array([-2., -1.,  0.,  1.,  2.,  3., 4.])
    """
    if num_intervals < 1:
        raise ValueError("num_intervals must be at least 1")

    if degree < 0:
        raise ValueError("degree must be non-negative")

    dtype_obj = np.dtype(dtype)
    if dtype_obj not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError("dtype must be float32 or float64")

    start_value: np.float32 | np.float64
    end_value: np.float32 | np.float64
    if dtype_obj == np.dtype(np.float64):
        start_value = np.float64(0)
        end_value = np.float64(num_intervals)
    else:
        start_value = np.float32(0)
        end_value = np.float32(num_intervals)

    return create_uniform_periodic_knot_vector(
        num_intervals,
        degree,
        continuity=degree - 1,
        domain=(start_value, end_value),
        dtype=dtype_obj,
    )


@functools.cache
def _cached_unique_knots_and_multiplicity(
    knots_repr: tuple[bytes, str, int],
    degree: int,
    tol: float,
    in_domain: bool = False,
) -> tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.int_]]:
    """Compute unique knots and multiplicities with memoization.

    Args:
        knots_repr (tuple[bytes, str, int]): Serialized knot vector bytes, dtype string, and size.
        degree (int): B-spline degree.
        tol (float): Tolerance value.
        in_domain (bool): Whether to restrict to the spline domain.

    Returns:
        tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.int_]]:
            Unique knots and corresponding multiplicities.
    """
    knots_bytes, dtype_str, size = knots_repr
    dtype = np.dtype(dtype_str)
    knots = np.frombuffer(knots_bytes, dtype=dtype, count=size).copy()
    tol_value = dtype.type(tol)
    return cast(
        tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.int_]],
        _get_unique_knots_and_multiplicity_impl(knots, degree, tol_value, in_domain),
    )


class BsplineSpace1D:
    """A class representing a 1D B-spline with configurable degree and knot vector.

    This class provides methods to analyze B-spline properties, validate input
    parameters, compute various geometric characteristics of the spline,
    and access various properties of the B-spline.

    Attributes:
        _tol (np.float32 | np.float64): Tolerance value for numerical comparisons.
        _knots (npt.NDArray[np.float32 | np.float64]): Knot vector defining the B-spline.
        _degree (int): Polynomial degree of the B-spline.
        _periodic (bool): Whether the B-spline is periodic.
    """

    _tol: float
    _knots: npt.NDArray[np.float32 | np.float64]
    _degree: int
    _periodic: bool

    def __init__(
        self,
        knots: npt.ArrayLike,
        degree: int,
        periodic: bool = False,
        snap_knots: bool | None = True,
    ) -> None:
        """Initialize a B-spline 1D object.

        Args:
            knots (npt.ArrayLike): Knot vector defining the B-spline. Must be non-decreasing
                and have at least 2*degree+2 elements.
            degree (int): Polynomial degree of the B-spline. Must be non-negative.
            periodic (bool): Whether the B-spline is periodic. Defaults to False.
            snap_knots (bool | None): Whether to snap nearby knots to avoid numerical issues.
                Defaults to True.

        Raises:
            ValueError: If degree is negative, knots are insufficient, or
                knots are not non-decreasing.
            TypeError: If knots cannot be converted to a numpy array.
        """
        BsplineSpace1D._validate_input(knots, degree, periodic)

        self._knots = np.asarray(knots)
        if np.issubdtype(self._knots.dtype, np.integer):
            self._knots = self._knots.astype(np.float64)

        self._tol = BsplineSpace1D._create_tolerance(self.dtype)

        self._degree = int(degree)
        self._periodic = bool(periodic)

        if snap_knots:
            self._snap_knots()

    @staticmethod
    def _validate_input(
        knots: npt.ArrayLike,
        degree: int,
        periodic: bool = False,
    ) -> None:
        """Validate the B-spline input parameters.

        Args:
            knots (npt.ArrayLike): Knot vector to validate.
            degree (int): Degree to validate.
            periodic (bool): Whether the B-spline is periodic.

        Raises:
            ValueError: If degree is negative, knots are insufficient, or
                knots are not non-decreasing.
            TypeError: If knots cannot be converted to a numpy array.
        """
        if degree < 0:
            raise ValueError("degree must be non-negative")

        if isinstance(knots, list):
            knots = np.array(knots)
        elif not isinstance(knots, np.ndarray):
            raise TypeError("knots must be a 1D numpy array or Python list")

        if np.issubdtype(knots.dtype, np.integer):
            knots = knots.astype(np.float64)

        dtype = knots.dtype
        tol = BsplineSpace1D._create_tolerance(dtype)

        if not isinstance(knots, np.ndarray) or knots.ndim != 1:
            raise TypeError("knots must be a 1D numpy array or Python list")

        if knots.dtype not in (np.float32, np.float64):
            raise ValueError("knots type must be float (32 or 64 bits)")

        if knots.size < (2 * degree + 2):
            raise ValueError("knots must have at least 2*degree+2 elements")

        if not np.all(np.diff(knots) >= 0):
            raise ValueError("knots must be non-decreasing")

        num_basis = _get_Bspline_num_basis_1D_impl(knots, degree, periodic, tol)
        if num_basis < (degree + 1):
            raise ValueError("Not enough knots for the specified degree")

    @staticmethod
    def _create_tolerance(dtype: npt.DTypeLike) -> float:
        """Create tolerance value based on data type.

        Right now, strict tolerance is used.

        Args:
            dtype (np.dtype): NumPy data type.

        Returns:
            float: Tolerance value appropriate for the given data type.
        """
        return float(get_strict_tolerance(dtype))

    def _snap_knots(self) -> None:
        """Snap knots within tolerance to avoid numerical precision issues.

        This method rounds knots to a precision determined by the stored tolerance
        and then averages any knots that are close together.

        It modifies the knot vector in place.
        """
        scale = 1.0 / self._tol
        rounded = np.round(self._knots * scale) / scale
        unique_vals = np.unique(rounded)

        snapped_knots = self._knots.copy()
        for val in unique_vals:
            mask = np.isclose(rounded, val, atol=0)
            snapped_knots[mask] = np.mean(self._knots[mask], dtype=self.dtype)
        self._knots = snapped_knots

    @property
    def degree(self) -> int:
        """Get the polynomial degree of the B-spline.

        Returns:
            int: The degree.
        """
        return self._degree

    @property
    def knots(self) -> npt.NDArray[np.float32 | np.float64]:
        """Get the knot vector.

        Returns:
            npt.NDArray[np.float32 | np.float64]: The knot vector.
        """
        return self._knots

    @property
    def periodic(self) -> bool:
        """Check if the B-spline is periodic.

        Returns:
            bool: True if periodic, False otherwise.
        """
        return self._periodic

    @property
    def tolerance(self) -> float:
        """Get the tolerance value used for numerical comparisons.

        Returns:
            float: The tolerance value.
        """
        return self._tol

    @property
    def dtype(self) -> npt.DTypeLike:
        """Get the data type of the knot vector (and used in computations).

        Returns:
            npt.DTypeLike: The numpy data type of the knots.
        """
        return self._knots.dtype

    @functools.cached_property
    def num_basis(self) -> int:
        """Get the number of basis functions.

        This depends on the knot vector length and the degree, but
        also on whether the B-spline is periodic.

        Returns:
            int: Number of basis functions.
        """
        return cast(
            int,
            _get_Bspline_num_basis_1D_impl(self._knots, self._degree, self._periodic, self._tol),
        )

    def get_unique_knots_and_multiplicity(
        self,
        in_domain: bool = False,
    ) -> tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.int_]]:
        """Get unique knots and their multiplicities.

        Args:
            in_domain (bool): If True, only consider knots in the domain.
                Defaults to False.

        Returns:
            tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.int_]]: Tuple of
            (unique_knots, multiplicities) where unique_knots contains the distinct knot values
            and multiplicities contains the corresponding multiplicity counts.
        """
        knots_repr = (self._knots.tobytes(), self._knots.dtype.str, int(self._knots.size))
        degree = int(self._degree)
        tol = float(self._tol)
        return _cached_unique_knots_and_multiplicity(knots_repr, degree, tol, in_domain)

    @functools.cached_property
    def num_intervals(self) -> np.int_:
        """Get the number of intervals in the domain.

        Returns:
            np.int_: Number of intervals.

        Example:
            >>> bspline = BsplineSpace1D([0, 0, 0, 1, 2, 2, 2], 2)
            >>> bspline.get_num_intervals
            2
        """
        unique_knots, _ = self.get_unique_knots_and_multiplicity(in_domain=True)
        return np.int_(len(unique_knots) - 1)

    def _get_domain_indices(self) -> tuple[np.int_, np.int_]:
        """Get the domain boundary indices of the knot vector.

        I.e., the indices of the knot vector that define the domain.

        Returns:
            tuple[np.int_, np.int_]: Tuple of (start_index, end_index) defining the domain.
        """
        return (np.int_(self._degree), np.int_(self._knots.size - self._degree - 1))

    @functools.cached_property
    def domain(self) -> tuple[np.float32 | np.float64, np.float32 | np.float64]:
        """Get the knot vector domain.

        Returns:
            tuple[np.float32 | np.float64, np.float32 | np.float64]: Tuple of
            (start_value, end_value) defining the domain.

        Example:
            >>> bspline = BsplineSpace1D([0, 0, 0, 1, 2, 2, 2], 2)
            >>> bspline.domain
            (0.0, 2.0)
        """
        i0, i1 = self._get_domain_indices()
        return (self._knots[i0], self._knots[i1])

    def has_left_end_open(self) -> bool:
        """Check if the left end of the B-spline is open.

        A left end is open if the first degree+1 knots are equal.

        Returns:
            bool: True if the left end is open, False otherwise.
        """
        if self.periodic:
            return False

        # Check if the first degree+1 knots are equal
        # (we know that they are non-decreasing).
        return bool(np.isclose(self._knots[0], self._knots[self._degree], atol=self._tol))

    def has_right_end_open(self) -> bool:
        """Check if the right end of the B-spline is open.

        A right end is open if the last degree+1 knots are equal.

        Returns:
            bool: True if the right end is open, False otherwise.
        """
        if self.periodic:
            return False

        # Check if the last degree+1 knots are equal
        # (we know that they are non-decreasing).
        return bool(np.isclose(self._knots[-self._degree - 1], self._knots[-1], atol=self._tol))

    def has_open_knots(self) -> bool:
        """Check if the B-spline has open ends.

        Returns:
            bool: True if both ends are open, False otherwise.
        """
        return self.has_left_end_open() and self.has_right_end_open()

    def has_Bezier_like_knots(self) -> bool:
        """Check if the knot vector represents a Bézier-like configuration.

        A Bézier-like configuration has open ends and only one non-zero span.

        Returns:
            bool: True if knots have open ends and only one span.

        Example:
            >>> bspline = BsplineSpace1D([1, 1, 1, 3, 3, 3], 2)
            >>> bspline.has_Bezier_like_knots()
            True
        """
        return (
            (not self._periodic) and self.has_open_knots() and self.num_basis == (self._degree + 1)
        )

    def get_cardinal_intervals(self) -> npt.NDArray[np.bool_]:
        """Get boolean array indicating whether the intervals (non-zero spans) are cardinal or not.

        An interval is cardinal if has the same length as the degree-1
        previous and the degree-1 next intervals.

        In the case of open knot vectors, this definition automatically
        discards the first degree-1 and the last degree-1 intervals.

        Returns:
            npt.NDArray[np.bool_]: Boolean array where True indicates cardinal intervals.
                It has length equal to the number of intervals.

        Example:
            >>> bspline = BsplineSpace1D([0, 0, 0, 1, 2, 3, 4, 5, 6, 6, 6], 2)
            >>> bspline.get_cardinal_intervals()
            array([False, False, True, True, False, False])

            >>> bspline = BsplineSpace1D([0, 0, 0, 1, 2, 3, 4, 5, 5, 6, 6, 6], 2)
            >>> bspline.get_cardinal_intervals()
            array([False, False, True, False, False, False])

            >>> bspline = BsplineSpace1D([0, 1, 2, 3, 4, 5, 6, 7, 10], 3)
            >>> bspline.get_cardinal_intervals()
            array([True, False])
        """
        return cast(
            npt.NDArray[np.bool_],
            _get_Bspline_cardinal_intervals_1D_impl(self._knots, self._degree, self._tol),
        )

    def tabulate_basis(
        self, pts: npt.ArrayLike
    ) -> tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.int_]]:
        """Evaluate the B-spline basis functions at the given points.

        Args:
            pts (npt.ArrayLike): Evaluation points.

        Returns:
            tuple[
                npt.NDArray[np.float32] | npt.NDArray[np.float64],
                npt.NDArray[np.int_]
            ]: Tuple containing:
                - basis_values: (npt.NDArray[np.float32] | npt.NDArray[np.float64])
                  Array of shape matching `pts` with the last dimension length (degree+1),
                  containing the basis function values evaluated at each point.
                - first_basis_indices: (npt.NDArray[np.int_])
                  1D integer array indicating the index of the first nonzero basis function
                  for each evaluation point. The length is the same as the number
                  of evaluation points.

        Raises:
                ValueError: If any evaluation points are outside the B-spline domain.

        Example:
            >>> bspline = BsplineSpace1D([0, 0, 0, 0.25, 0.7, 0.7, 1, 1, 1], 2)
            >>> bspline.tabulate_basis([0.0, 0.5, 0.75, 1.0])
            (array([[1.        , 0.        , 0.        ],
                    [0.12698413, 0.5643739 , 0.30864198],
                    [0.69444444, 0.27777778, 0.02777778],
                    [0.        , 0.        , 1.        ]]),
             array([0, 1, 3, 3]))
        """
        return _tabulate_Bspline_basis_1D_impl(self, pts)

    def tabulate_Bezier_extraction_operators(self) -> npt.NDArray[np.float32 | np.float64]:
        """Create Bézier extraction operators of the B-spline.

        Returns:
            npt.NDArray[np.float32 | np.float64]: Array of extraction matrices with shape
                (n_intervals, degree+1, degree+1) where each matrix transforms
                Bernstein basis functions to B-spline basis functions for that interval.

                Each matrix C[i, :, :] transforms Bernstein basis functions
                to B-spline basis functions for the i-th interval as
                    C[i, :, :] @ [Bernstein values] = [B-spline values in interval].
        """
        return cast(
            npt.NDArray[np.float32 | np.float64],
            _tabulate_Bspline_Bezier_1D_extraction_impl(self.knots, self.degree, self.tolerance),
        )

    def tabulate_Lagrange_extraction_operators(self) -> npt.NDArray[np.float32 | np.float64]:
        """Create Lagrange extraction operators of the B-spline.

        Returns:
            npt.NDArray[np.float32 | np.float64]: Array of extraction matrices with shape
                (n_intervals, degree+1, degree+1) where each matrix transforms
                Lagrange basis functions to B-spline basis functions for that interval.

                Each matrix C[i, :, :] transforms Lagrange basis functions
                to B-spline basis functions for the i-th interval as
                    C[i, :, :] @ [Lagrange values] = [B-spline values in interval].
        """
        return _tabulate_Bspline_Lagrange_1D_extraction_impl(
            self.knots, self.degree, self.tolerance
        )

    def tabulate_cardinal_extraction_operators(self) -> npt.NDArray[np.float32 | np.float64]:
        """Create cardinal B-spline extraction operators of the B-spline.

        Returns:
            npt.NDArray[np.float32 | np.float64]: Array of extraction matrices with shape
                (n_intervals, degree+1, degree+1) where each matrix transforms
                cardinal spline basis functions to B-spline basis functions for that interval.

                Each matrix C[i, :, :] transforms cardinal spline basis functions
                to B-spline basis functions for the i-th interval as
                    C[i, :, :] @ [cardinal values] = [B-spline values in interval].
        """
        return _tabulate_Bspline_cardinal_1D_extraction_impl(
            self.knots, self.degree, self.tolerance
        )
