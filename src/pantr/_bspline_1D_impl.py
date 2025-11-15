"""Implementation functions for 1D B-spline space operations.

This module provides low-level, Numba-accelerated implementations of B-spline
operations including basis function evaluation, knot analysis, and geometric
computations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
import numpy.typing as npt
from numba import njit
from numba.core import types as nb_types

from ._basis_impl import _eval_Bernstein_basis_1D_impl
from ._basis_utils import _normalize_basis_output_1D, _normalize_points_1D
from .basis import LagrangeVariant
from .change_basis_1D import (
    create_cardinal_to_Bernstein_change_basis,
    create_Lagrange_to_Bernstein_change_basis,
)

if TYPE_CHECKING:
    from .bspline_1D import BsplineSpace1D

nb_Tuple = nb_types.Tuple
nb_bool = nb_types.boolean
float32 = nb_types.float32
float64 = nb_types.float64
intp = nb_types.intp
void = nb_types.void


@njit(
    [
        void(float32[::1], intp),
        void(float64[::1], intp),
    ],
    cache=True,
)  # type: ignore[misc]
def _check_spline_info(knots: npt.NDArray[np.float32 | np.float64], degree: int) -> None:
    """Validate basic constraints on B-spline knot vector and degree.

    Performs core checks on a B-spline knot vector and polynomial degree,
    raising an ValueError if any validation fails.

    Args:
        knots (npt.NDArray[np.float32 | np.float64]): 1D array representing the B-spline
            knot vector to check.
        degree (int): Non-negative polynomial degree for the B-spline.

    Raises:
        TypeError: If `knots` is not 1-dimensional.
        ValueError: If `degree` is negative,
            if there are fewer than `2*degree+2` knots, or if the knot vector
            is not non-decreasing.
    """
    if knots.ndim != 1:
        raise TypeError("knots must be a 1D array")
    if degree < 0:
        raise ValueError("degree must be non-negative")
    if knots.size < (2 * degree + 2):
        raise ValueError("knots must have at least 2*degree+2 elements")
    if not np.all(np.diff(knots) >= knots.dtype.type(0.0)):
        raise ValueError("knots must be non-decreasing")


@njit(
    [
        intp(float32[::1], intp, float32),
        intp(float64[::1], intp, float64),
    ],
    cache=True,
)  # type: ignore[misc]
def _get_multiplicity_of_first_knot_in_domain_impl(
    knots: npt.NDArray[np.float32 | np.float64],
    degree: int,
    tol: float,
) -> int:
    """Get the multiplicity of the first knot in the domain (i.e., the `degree`-th knot).

    Args:
        knots (npt.NDArray[np.float32 | np.float64]): B-spline knot vector.
        degree (int): B-spline degree.
        tol (float): Tolerance for numerical comparisons.

    Returns:
        int: Multiplicity of the first knot in the domain.

    Raises:
        ValueError: If tolerance is not positive or basic validation
            of knots and degree fails.
    """
    _check_spline_info(knots, degree)
    if tol <= 0:
        raise ValueError("tol must be positive")

    first_knot = knots[degree]
    return int(np.sum(np.isclose(knots[: degree + 1], first_knot, atol=tol)))


@njit(
    [
        nb_Tuple((float32[:], intp[:]))(float32[::1], intp, float32, nb_bool),
        nb_Tuple((float64[:], intp[:]))(float64[::1], intp, float64, nb_bool),
    ],
    cache=True,
)  # type: ignore[misc]
def _get_unique_knots_and_multiplicity_impl(
    knots: npt.NDArray[np.float32 | np.float64],
    degree: int,
    tol: float,
    in_domain: bool = False,
) -> tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.int_]]:
    """Get unique knots and their multiplicities.

    Args:
        knots (npt.NDArray[np.float32 | np.float64]): B-spline knot vector.
        degree (int): B-spline degree.
        tol (float): Tolerance for numerical comparisons.
        in_domain (bool): If True, only consider knots in the domain.
            Defaults to False.

    Returns:
        tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.int_]]: Tuple of
            (unique_knots, multiplicities) where unique_knots contains the distinct knot values
            and multiplicities contains the corresponding multiplicity counts. Both arrays have
            the same length.

    Raises:
        ValueError: If tolerance is not positive or basic validation
            of knots and degree fails.
    """
    _check_spline_info(knots, degree)

    if tol <= 0:
        raise ValueError("tol must be positive")

    # Round to tolerance precision for grouping
    dtype = knots.dtype
    scale = dtype.type(1.0 / tol)
    rounded_knots = np.round(knots * scale) / scale

    n = knots.size
    # unique_rounded_knots = np.empty(n, dtype=rounded_knots.dtype)
    unique_rounded_knots_ids = np.empty(n, dtype=np.int_)
    mult = np.zeros(n, dtype=np.int_)

    if in_domain:
        rknot_0, rknot_1 = rounded_knots[degree], rounded_knots[-degree - 1]
    else:
        rknot_0, rknot_1 = rounded_knots[0], rounded_knots[-1]

    j = -1
    last_rknot = np.nan

    for i, rknot in enumerate(rounded_knots):
        if rknot < rknot_0:
            continue
        elif rknot > rknot_1:
            break

        if rknot == last_rknot:
            mult[j] += 1
        else:
            j += 1
            last_rknot = rknot
            unique_rounded_knots_ids[j] = i
            mult[j] = 1

    unique_knots = rounded_knots[unique_rounded_knots_ids[: j + 1]]
    mults = mult[: j + 1]
    return unique_knots, mults


@njit(  # type: ignore[misc]
    [
        nb_bool[:](float32[::1], intp, float32[::1], float32),
        nb_bool[:](float64[::1], intp, float64[::1], float64),
    ],
    cache=True,
)
def _is_in_domain_impl(
    knots: npt.NDArray[np.float32 | np.float64],
    degree: int,
    pts: npt.NDArray[np.float32 | np.float64],
    tol: float,
) -> npt.NDArray[np.bool_]:
    """Check if points are within the B-spline domain (up to tolerance).

    Args:
        knots (npt.NDArray[np.float32 | np.float64]): B-spline knot vector.
        degree (int): B-spline degree.
        pts (npt.NDArray[np.float32 | np.float64]): Points to check.
        tol (float): Tolerance for numerical comparisons.

    Returns:
        npt.NDArray[np.bool_]: Boolean array where True indicates points
            are within the domain. It has the same length as the number of points.

    Raises:
        TypeError: If `pts` is not 1-dimensional.
        ValueError: If tolerance is not positive or basic validation
            of knots and degree fails.
    """
    _check_spline_info(knots, degree)

    if tol <= 0:
        raise ValueError("tol must be positive")
    if pts.ndim != 1:
        raise TypeError("pts must be a 1D array")
    if pts.size == 0:
        raise ValueError("pts must have at least one element")

    knot_begin, knot_end = knots[degree], knots[-degree - 1]
    return np.logical_and(  # type: ignore[no-any-return]
        (knot_begin < pts) | np.isclose(knot_begin, pts, atol=tol),
        (pts < knot_end) | np.isclose(pts, knot_end, atol=tol),
    )


@njit(
    [
        intp(float32[::1], intp, nb_bool, float32),
        intp(float64[::1], intp, nb_bool, float64),
    ],
    cache=True,
)  # type: ignore[misc]
def _compute_num_basis_impl(
    knots: npt.NDArray[np.float32 | np.float64],
    degree: int,
    periodic: bool,
    tol: float,
) -> int:
    """Compute the number of basis functions.

    In the non-periodic case, the number of basis functions is given by
    the number of knots minus the degree minus 1.

    In the periodic case, the number of basis functions is computed as
    before, minus the regularity at the domain's beginning/end minus 1.

    Args:
        knots (npt.NDArray[np.float32 | np.float64]): B-spline knot vector.
        degree (int): B-spline degree.
        periodic (bool): Whether the B-spline is periodic.
        tol (np.float32 | np.float64): Tolerance for numerical comparisons.

    Returns:
        int: Number of basis functions.

    Raises:
        ValueError: If tolerance is not positive or basic validation
            of knots and degree fails.
    """
    _check_spline_info(knots, degree)

    if tol <= 0:
        raise ValueError("tol must be positive")

    num_basis = int(len(knots) - degree - 1)

    if periodic:
        # Determining the number of extra basis required in the periodic case.
        # This depends on the regularity of the knot vector at domain's
        # begining.
        regularity = degree - _get_multiplicity_of_first_knot_in_domain_impl(knots, degree, tol)
        num_basis -= regularity + 1

    return num_basis


@njit(
    [
        intp[:](float32[::1], float32[:]),
        intp[:](float64[::1], float64[:]),
    ],
    cache=True,
)  # type: ignore[misc]
def _get_last_knot_smaller_equal_impl(
    knots: npt.NDArray[np.float32 | np.float64],
    pts: npt.NDArray[np.float32 | np.float64],
) -> npt.NDArray[np.int_]:
    """Get the index of the last knot which is less than or equal to each point in pts.

    Args:
        knots (npt.NDArray[np.float32 | np.float64]): B-spline knot vector (must be non-decreasing).
        pts (npt.NDArray[np.float32 | np.float64]): Points (1D array) to find knot indices for.

    Returns:
        npt.NDArray[np.int_]: Array of computed indices, one for each point in pts.

    Raises:
        TypeError: If `knots` or `pts` is not 1-dimensional.
        ValueError: If knots are not non-decreasing,
            or points array has no elements.
    """
    if knots.ndim != 1:
        raise TypeError("knots must be a 1D array")
    if not np.all(np.diff(knots) >= knots.dtype.type(0.0)):
        raise ValueError("knots must be non-decreasing")
    if pts.ndim != 1:
        raise TypeError("pts must be a 1D array")
    if pts.size == 0:
        raise ValueError("pts must have at least one element")

    return np.searchsorted(knots, pts, side="right") - 1


@njit(
    [
        nb_Tuple((float32[:, ::1], intp[:]))(float32[::1], intp, nb_bool, float32, float32[::1]),
        nb_Tuple((float64[:, ::1], intp[:]))(float64[::1], intp, nb_bool, float64, float64[::1]),
    ],
    cache=True,
)  # type: ignore[misc]
def _eval_basis_Cox_de_Boor_impl(
    knots: npt.NDArray[np.float32 | np.float64],
    degree: int,
    periodic: bool,
    tol: float,
    pts: npt.NDArray[np.float32 | np.float64],
) -> tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.int_]]:
    """Evaluate B-spline basis functions using Cox-de Boor recursion.

    This function implements Algorithm 2.23 from "Spline Methods Draft" by Tom Lyche.

    Args:
        knots (npt.NDArray[np.float32 | np.float64]): B-spline knot vector.
        degree (int): B-spline degree.
        periodic (bool): Whether the B-spline is periodic.
        tol (float): Tolerance for numerical comparisons.
        pts (npt.NDArray[np.float32 | np.float64]): Points (1D array) to evaluate basis
            functions at.

    Returns:
        tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.int_]]: Tuple of
            (basis_values, first_basis_indices) where basis_values contains the basis
            function values at each point and first_basis_indices contains the index of
            the first non-zero basis function for each point.

    Raises:
        TypeError: If `pts` is not 1-dimensional.
        ValueError: If the knot vector or degree fails basic validation, if tol is negative,
            if pts has no elements, or if points are outside domain.
    """
    # See Spline Methods Draft, by Tom Lychee. Algorithm 2.23

    _check_spline_info(knots, degree)

    if tol < 0:
        raise ValueError("tol must be positive")
    if pts.ndim != 1:
        raise TypeError("pts must be a 1D array")
    if pts.size == 0:
        raise ValueError("pts must have at least one element")
    if not np.all(_is_in_domain_impl(knots, degree, pts, tol)):
        raise ValueError("pts are outside domain.")

    knot_ids = _get_last_knot_smaller_equal_impl(knots, pts)

    dtype = knots.dtype
    zero = dtype.type(0.0)
    one = dtype.type(1.0)

    order = degree + 1
    n_pts = pts.size

    basis = np.zeros((n_pts, order), dtype=dtype)
    basis[:, -1] = one

    # Here we account for the case where the evaluation point
    # coincides with the last knot.
    num_basis = _compute_num_basis_impl(knots, degree, periodic, tol)
    first_basis = np.minimum(knot_ids - degree, num_basis - order)

    for pt_id in range(n_pts):
        knot_id = knot_ids[pt_id]

        if knot_id == (knots.size - 1):
            continue

        pt = pts[pt_id]
        basis_i = basis[pt_id, :]
        local_knots = knots[knot_id - degree + 1 : knot_id + order]

        for sub_degree in range(1, order):
            k0, k1 = local_knots[0], local_knots[sub_degree]
            diff = k1 - k0
            inv_diff = zero if diff < tol else one / diff

            for bs_id in range(degree - sub_degree, degree):
                basis_i[bs_id] *= (pt - k0) * inv_diff

                k0, k1 = local_knots[bs_id], local_knots[bs_id + sub_degree]
                diff = k1 - k0
                inv_diff = zero if diff < tol else one / diff

                basis_i[bs_id] += (k1 - pt) * inv_diff * basis_i[bs_id + 1]

            basis_i[-1] *= (pt - k0) * inv_diff

    return basis, first_basis


@njit(
    [
        nb_bool[:](float32[::1], intp, float32),
        nb_bool[:](float64[::1], intp, float64),
    ],
    cache=True,
)  # type: ignore[misc]
def _get_cardinal_intervals_impl(
    knots: npt.NDArray[np.float32 | np.float64], degree: int, tol: float
) -> npt.NDArray[np.bool_]:
    """Get boolean array indicating whether intervals are cardinal.

    An interval is cardinal if it has the same length as the degree-1
    previous and the degree-1 next intervals.

    In the case of open knot vectors, this definition automatically
    discards the first degree-1 and the last degree-1 intervals.

    Args:
        knots (npt.NDArray[np.float32 | np.float64]): B-spline knot vector.
        degree (int): B-spline degree.
        tol (float): Tolerance for numerical comparisons.

    Returns:
        npt.NDArray[np.bool_]: Boolean array where True indicates cardinal intervals.
            It has length equal to the number of intervals.

    Raises:
        ValueError: If the knot vector or degree fails basic validation or if tol is negative.
    """
    _, mult = _get_unique_knots_and_multiplicity_impl(knots, degree, tol, in_domain=True)
    num_intervals = len(mult) - 1

    cardinal = np.full(num_intervals, np.False_, dtype=np.bool_)

    if np.all(mult > 1):
        return cardinal

    knot_id = degree

    # Note: this loop could be shortened by only looking at those
    # intervals for which the multiplicity of the first knot is 1.
    # This would require to compute knot_id differently.
    for elem_id in range(num_intervals):
        if mult[elem_id] == 1 and mult[elem_id + 1] == 1:
            local_knots = knots[knot_id - degree + 1 : knot_id + degree + 1]
            lengths = np.diff(local_knots)
            if np.all(np.isclose(lengths, lengths[degree - 1], atol=tol)):
                cardinal[elem_id] = np.True_

        knot_id += mult[elem_id + 1]

    return cardinal


@njit(
    [
        float32[:, :, ::1](float32[::1], intp, float32),
        float64[:, :, ::1](float64[::1], intp, float64),
    ],
    cache=True,
)  # type: ignore[misc]
def _create_bspline_Bezier_extraction_impl(
    knots: npt.NDArray[np.float32 | np.float64], degree: int, tol: float
) -> npt.NDArray[np.float32 | np.float64]:
    r"""Create Bézier extraction operators for each interval.

    This function computes the extraction operators that transform Bernstein
    into B-spline basis functions for each interval.
    For each interval \( i \), the Bézier extraction operator \( C_i \) satisfies:

        \[
        N_i(x) = C_i @ B(ξ)
        \]

    where:
      - N_i(x) is the vector of B-spline basis functions nonzero on the interval \( i \),
        evaluated at \( x \),
      - \( B(ξ) \) is the vector of Bernstein basis functions on the reference interval \([0, 1]\),
        evaluated at \( ξ \),
      - \( C_i \) is the extraction matrix for interval \( i \),
      - \( x \) is the physical coordinate, \( ξ \) is the local (reference) referred to \([0, 1]\).

    Args:
        knots (npt.NDArray[np.float32 | np.float64]): B-spline knot vector.
        degree (int): B-spline degree.
        tol (float): Tolerance for numerical comparisons.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Array of extraction matrices with shape
            (intervals, degree+1, degree+1) where each matrix transforms
            Bernstein basis functions to B-spline basis functions for that interval.

    Raises:
        ValueError: If the knot vector or degree fails basic validation or if tol is negative.
    """
    if tol < 0:
        raise ValueError("tol must be positive")

    unique_knots, mults = _get_unique_knots_and_multiplicity_impl(
        knots, degree, tol, in_domain=True
    )

    _check_spline_info(knots, degree)

    n_elems = len(unique_knots) - 1

    dtype = knots.dtype
    one = dtype.type(1.0)

    # Initialize identity matrix for every element.
    Cs = np.zeros((n_elems, degree + 1, degree + 1), dtype=dtype)
    Cs[:, : degree + 1, : degree + 1] = np.eye(degree + 1, dtype=dtype)

    mult = _get_multiplicity_of_first_knot_in_domain_impl(knots, degree, tol)

    # If not open first knot, additional knot insertion is needed.
    if mult < (degree + 1):
        C = Cs[0]
        reg = degree - mult

        t = knots[degree]
        for r in range(reg):
            lcl_knots = knots[r:]
            for k in range(1, degree - r):
                alpha = (t - lcl_knots[k]) / (lcl_knots[k + degree - r] - lcl_knots[k])
                C[:, k - 1] = alpha * C[:, k] + (one - alpha) * C[:, k - 1]

    alphas = np.zeros(degree - 1, dtype=dtype)

    knt_id = degree
    mult = 0

    for elem_id in range(n_elems):
        knt_id += mult
        mult = mults[elem_id + 1]

        if mult >= degree:
            continue

        lcl_knots = knots[knt_id : knt_id + degree + 1]
        alphas[: degree - mult] = (lcl_knots[1] - lcl_knots[0]) / (
            lcl_knots[mult + 1 :] - lcl_knots[0]
        )

        C = Cs[elem_id]

        reg = degree - mult
        for r in range(1, reg + 1):
            s = mult + r
            for k in range(degree, s - 1, -1):
                alpha = alphas[k - s]
                C[:, k] = alpha * C[:, k] + (one - alpha) * C[:, k - 1]

            if elem_id < (n_elems - 1):
                Cs[elem_id + 1, reg - r : reg + 1, reg - r] = C[degree - r : degree + 1, degree]

    return Cs


def _create_bspline_Lagrange_extraction_impl(
    knots: npt.NDArray[np.float32 | np.float64],
    degree: int,
    tol: float,
    lagrange_variant: LagrangeVariant = LagrangeVariant.EQUISPACES,
) -> npt.NDArray[np.float32 | np.float64]:
    """Create Lagrange extraction operators for a B-spline.

    Args:
        knots (npt.NDArray[np.float32 | np.float64]): B-spline knot vector.
        degree (int): B-spline degree.
        tol (float): Tolerance for numerical comparisons.
        lagrange_variant (LagrangeVariant): Lagrange point distribution
            (e.g., equispaced, gauss lobatto legendre, etc). Defaults to LagrangeVariant.EQUISPACES.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Array of extraction matrices with shape
            (n_intervals, degree+1, degree+1) where each matrix transforms
            Lagrange basis functions to B-spline basis functions for that interval.

            Each matrix C[i, :, :] transforms Bernstein basis functions
            to B-spline basis functions for the i-th interval as
                C[i, :, :] @ [Lagrange values] = [B-spline values in interval].
    """
    if tol < 0:
        raise ValueError("tol must be positive")

    _check_spline_info(knots, degree)

    C = cast(
        npt.NDArray[np.float32 | np.float64],
        _create_bspline_Bezier_extraction_impl(knots, degree, tol),
    )

    dtype = knots.dtype
    lagr_to_bzr = create_Lagrange_to_Bernstein_change_basis(degree, lagrange_variant, dtype)
    C[:] = C @ lagr_to_bzr

    return C


def _create_bspline_cardinal_extraction_impl(
    knots: npt.NDArray[np.float32 | np.float64],
    degree: int,
    tol: float,
) -> npt.NDArray[np.float32 | np.float64]:
    """Create cardinal B-spline extraction operators.

    For cardinal intervals, the extraction matrix is set to the identity matrix

    Args:
        knots (npt.NDArray[np.float32 | np.float64]): B-spline knot vector.
        degree (int): B-spline degree.
        tol (float): Tolerance for numerical comparisons.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Array of extraction matrices with shape
            (n_intervals, degree+1, degree+1) where each matrix transforms
            cardinal B-spline basis functions to B-spline basis functions for that interval.

            Each matrix C[i, :, :] transforms cardinal B-spline basis functions
            to B-spline basis functions for the i-th interval as
                C[i, :, :] @ [cardinal values] = [B-spline values in interval].
    """
    if tol < 0:
        raise ValueError("tol must be positive")

    _check_spline_info(knots, degree)

    C = cast(
        npt.NDArray[np.float32 | np.float64],
        _create_bspline_Bezier_extraction_impl(knots, degree, tol),
    )

    dtype = knots.dtype
    card_to_bzr = create_cardinal_to_Bernstein_change_basis(degree, dtype)
    C[:] = C @ card_to_bzr

    for i in np.where(_get_cardinal_intervals_impl(knots, degree, tol))[0]:
        C[i, :, :] = np.eye(degree + 1, dtype=dtype)
    return C


def _eval_Bspline_basis_Bernstein_like_1D(
    spline: BsplineSpace1D,
    pts: npt.NDArray[np.float32 | np.float64],
) -> tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.int_]]:
    """Evaluate B-spline basis functions when they reduce to Bernstein polynomials.

    This function is used when the B-spline has Bézier-like knots, allowing
    direct evaluation using Bernstein basis functions.

    Args:
        spline (BsplineSpace1D): B-spline object with Bézier-like knots.
        pts (npt.NDArray[np.float32 | np.float64]): Evaluation points.

    Returns:
        tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.int_]]: Tuple of
            (basis_values, first_basis_indices) where basis_values is an array of shape
            (number pts, degree+1) that contains the Bernstein basis function values and
            first_basis_indices contains the indices of the first non-zero basis function
            for each point.

    Raises:
        ValueError: If the B-spline does not have Bézier-like knots.
    """
    if not spline.has_Bezier_like_knots():
        raise ValueError("B-spline does not have Bézier-like knots.")

    # map the points to the reference interval [0, 1]
    k0, k1 = spline.domain
    pts = (pts - k0) / (k1 - k0)

    # the first basis function is always the 0
    first_basis_ids = np.zeros(pts.size, dtype=np.int_)

    return _eval_Bernstein_basis_1D_impl(spline.degree, pts), first_basis_ids


def _eval_Bspline_basis_1D_impl(
    spline: BsplineSpace1D, pts: npt.ArrayLike
) -> tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.int_]]:
    """Evaluate B-spline basis functions at given points.

    This function automatically selects the most efficient evaluation method:
    - For Bézier-like knots: direct Bernstein evaluation
    - For general knots: Cox-de Boor recursion

    In both cases it calls vectorized or numba implementations.

    Args:
        spline (BsplineSpace1D): B-spline object defining the basis.
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
              for each evaluation point. The length is the same as the number of evaluation points.

    Raises:
        ValueError: If any evaluation points are outside the B-spline domain.

    Example:
        >>> bspline = BsplineSpace1D([0, 0, 0, 0.25, 0.7, 0.7, 1, 1, 1], 2)
        >>> _eval_Bspline_basis_1D_impl(bspline, [0.0, 0.5, 0.75, 1.0])
        (array([[1.        , 0.        , 0.        ],
                [0.12698413, 0.5643739 , 0.30864198],
                [0.69444444, 0.27777778, 0.02777778],
                [0.        , 0.        , 1.        ]]),
         array([0, 1, 3, 3]))
    """
    # Get input shape before normalization (handle scalars and lists)
    if isinstance(pts, np.ndarray):
        input_shape = pts.shape
    elif isinstance(pts, list | tuple):
        input_shape = np.array(pts).shape
    else:  # scalar
        input_shape = ()

    pts = _normalize_points_1D(pts)

    if not np.all(_is_in_domain_impl(spline.knots, spline.degree, pts, spline.tolerance)):
        raise ValueError(
            f"One or more values in pts are outside the knot vector domain {spline.domain}"
        )

    if spline.has_Bezier_like_knots():
        B, first_indices = _eval_Bspline_basis_Bernstein_like_1D(spline, pts)
    else:
        B, first_indices = _eval_basis_Cox_de_Boor_impl(
            spline.knots, spline.degree, spline.periodic, spline.tolerance, pts
        )

    first_indices = (
        first_indices.squeeze() if len(input_shape) == 0 else first_indices.reshape(input_shape)
    )

    return _normalize_basis_output_1D(B, input_shape), first_indices


def _validate_knot_input(
    num_intervals: int,
    degree: int,
    continuity: int,
    domain: tuple[np.float32 | np.float64, np.float32 | np.float64],
    dtype: npt.DTypeLike,
) -> None:
    """Validate input parameters for knot vector generation.

    Args:
        num_intervals (int_): Number of intervals in the domain.
        degree (int): B-spline degree.
        continuity (int): Continuity level at interior knots.
        domain (tuple[np.float32 | np.float64, np.float32 | np.float64]):
            Domain boundaries as (start, end).
        dtype (np.dtype): Data type for the knot vector.

    Raises:
        ValueError: If any parameter is invalid.
    """
    if domain[0] >= domain[1]:
        raise ValueError("domain[0] must be less than domain[1]")

    if num_intervals < 0:
        raise ValueError("num_intervals must be non-negative")

    if degree < 0:
        raise ValueError("degree must be non-negative")

    if continuity < -1 or continuity >= degree:
        raise ValueError(f"Continuity must be between -1 and {degree - 1} for degree {degree}.")

    if dtype not in (
        np.dtype(np.float64),
        np.dtype(np.float32),
        np.float32,
        np.float64,
    ):
        raise ValueError("dtype must be float64 or float32")


def _ensure_scalar_arrays(
    values: dict[str, float | int | np.floating[Any] | None],
) -> dict[str, npt.NDArray[np.generic]]:
    """Convert scalar inputs to zero-dimensional arrays.

    Args:
        values (dict[str, Optional[float | int | np.floating]]): Mapping of parameter
            names to scalar values.

    Returns:
        dict[str, npt.NDArray[np.generic]]: Mapping of provided names to 0-D arrays.

    Raises:
        ValueError: If any value is not scalar.
    """
    arrays: dict[str, npt.NDArray[np.generic]] = {}
    for name, value in values.items():
        if value is None:
            continue
        array_value = np.array(value)
        if array_value.ndim != 0:
            raise ValueError(f"{name} must be a scalar value")
        arrays[name] = array_value
    return arrays


def _resolve_dtype_from_arrays(
    arrays: dict[str, npt.NDArray[np.generic]],
    requested_dtype: npt.DTypeLike | None,
) -> np.dtype[np.floating[Any]]:
    """Resolve the floating dtype to use for knot endpoints.

    Args:
        arrays (dict[str, npt.NDArray[np.generic]]): Scalar arrays for each value.
        requested_dtype (Optional[npt.DTypeLike]): Explicit dtype request.

    Returns:
        np.dtype[np.floating[Any]]: Resolved floating-point dtype.

    Raises:
        ValueError: If the dtype is invalid or inconsistent across values.
    """
    if requested_dtype is not None:
        dtype_obj = np.dtype(requested_dtype)
        if dtype_obj.kind != "f":
            raise ValueError("dtype must be a floating-point type")
        for name, array_value in arrays.items():
            if array_value.dtype != dtype_obj:
                raise ValueError(f"{name} must be of type dtype {dtype_obj}")
        return cast(np.dtype[np.floating[Any]], dtype_obj)

    inferred_dtype: np.dtype[np.floating[Any]] | None = None
    for array_value in arrays.values():
        candidate = (
            cast(np.dtype[np.floating[Any]], np.dtype(array_value.dtype))
            if array_value.dtype.kind == "f"
            else cast(np.dtype[np.floating[Any]], np.dtype(np.float64))
        )
        if inferred_dtype is None:
            inferred_dtype = candidate
        elif candidate != inferred_dtype:
            raise ValueError("start and end must have the same dtype")

    return (
        inferred_dtype
        if inferred_dtype is not None
        else cast(np.dtype[np.floating[Any]], np.dtype(np.float64))
    )


def _coerce_scalar(
    array_value: npt.NDArray[np.generic] | None,
    dtype_obj: np.dtype[np.floating[Any]],
    default: float,
) -> np.floating[Any]:
    """Convert a scalar array to the target dtype or use the default value.

    Args:
        array_value (Optional[npt.NDArray[np.generic]]): Scalar array to convert.
        dtype_obj (np.dtype[np.floating[Any]]): Target floating dtype.
        default (float): Default value when the array is None.

    Returns:
        np.floating[Any]: Value converted to the requested dtype.
    """
    if array_value is None:
        return dtype_obj.type(default)
    return dtype_obj.type(array_value.astype(dtype_obj, copy=False).item())


def _get_ends_and_type(
    start: float | int | np.floating[Any] | None = None,
    end: float | int | np.floating[Any] | None = None,
    dtype: npt.DTypeLike | None = None,
) -> tuple[np.floating[Any], np.floating[Any], np.dtype[np.floating[Any]]]:
    """Get the start, end, and dtype for a knot vector.

    Args:
        start (Optional[float | int | np.floating]): Start value of the domain.
            Defaults to 0.0 if not provided.
        end (Optional[float | int | np.floating]): End value of the domain.
            Defaults to 1.0 if not provided.
        dtype (Optional[npt.DTypeLike]): Data type for the knot vector.
            If None, inferred from start/end or defaults to float64.

    Returns:
        tuple[np.floating, np.floating, np.dtype]: Tuple of (start, end, dtype).

    Raises:
        ValueError: If inputs are non-scalar, have incompatible dtypes, or if end <= start.
    """
    arrays = _ensure_scalar_arrays({"start": start, "end": end})
    dtype_obj = _resolve_dtype_from_arrays(arrays, dtype)
    start_value = _coerce_scalar(arrays.get("start"), dtype_obj, 0.0)
    end_value = _coerce_scalar(arrays.get("end"), dtype_obj, 1.0)

    if end_value <= start_value:
        raise ValueError("end must be greater than start")

    return start_value, end_value, dtype_obj
