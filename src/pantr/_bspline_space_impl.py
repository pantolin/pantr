"""Implementation functions for B-spline space operations.

This module provides low-level, Numba-accelerated implementations of B-spline
operations including basis function evaluation, knot analysis, and geometric
computations for both 1D and multi-dimensional B-splines.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar, cast

import numba as nb
import numpy as np
import numpy.typing as npt
from numba.core import types as nb_types

from ._basis_impl import _tabulate_Bernstein_basis_1D_impl
from ._basis_utils import (
    _compute_final_output_shape_1D,
    _normalize_points_1D,
    _validate_out_array_1D,
    _validate_out_array_3d_float,
    _validate_out_array_bool,
    _validate_out_array_first_basis,
)
from .basis import LagrangeVariant
from .change_basis import (
    compute_cardinal_to_Bernstein_change_basis_1D,
    compute_Lagrange_to_Bernstein_change_basis_1D,
)
from .quad import PointsLattice

F = TypeVar("F", bound=Callable[..., Any])

if TYPE_CHECKING:
    from .bspline_space import BsplineSpace, BsplineSpace1D

    # During type-checking, make the decorator a no-op that preserves types.
    def nb_jit(*args: object, **kwargs: object) -> Callable[[F], F]:
        def decorator(func: F) -> F:
            return func

        return decorator
else:
    # At runtime, use the real Numba decorator.
    nb_jit = nb.jit  # type: ignore[attr-defined]

nb_Tuple = nb_types.Tuple
nb_bool = nb_types.boolean
float32 = nb_types.float32
float64 = nb_types.float64
intp = nb_types.intp
void = nb_types.void


@nb_jit(
    nopython=True,
    cache=True,
    parallel=False,
)
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


@nb_jit(
    nopython=True,
    cache=True,
    parallel=False,
)
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

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    first_knot = knots[degree]
    return int(np.sum(np.isclose(knots[: degree + 1], first_knot, atol=tol)))


@nb_jit(
    nopython=True,
    cache=True,
    parallel=False,
)
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

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
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


@nb_jit(
    nopython=True,
    cache=True,
    parallel=False,
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

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    knot_begin, knot_end = knots[degree], knots[-degree - 1]
    return np.logical_and(  # type: ignore[no-any-return]
        (knot_begin < pts) | np.isclose(knot_begin, pts, atol=tol),
        (pts < knot_end) | np.isclose(pts, knot_end, atol=tol),
    )


@nb_jit(
    nopython=True,
    cache=True,
    parallel=False,
)
def _get_Bspline_num_basis_1D_impl(
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

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    num_basis = int(len(knots) - degree - 1)

    if periodic:
        # Determining the number of extra basis required in the periodic case.
        # This depends on the regularity of the knot vector at domain's
        # begining.
        regularity = degree - _get_multiplicity_of_first_knot_in_domain_impl(knots, degree, tol)
        num_basis -= regularity + 1

    return num_basis


@nb_jit(
    nopython=True,
    cache=True,
    parallel=False,
)
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

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    return np.searchsorted(knots, pts, side="right") - 1


@nb_jit(
    nopython=True,
    cache=True,
    parallel=False,
)
def _compute_basis_Cox_de_Boor_impl(  # noqa: PLR0913
    knots: npt.NDArray[np.float32 | np.float64],
    degree: int,
    periodic: bool,
    tol: float,
    pts: npt.NDArray[np.float32 | np.float64],
    out_basis: npt.NDArray[np.float32 | np.float64],
    out_first_basis: npt.NDArray[np.int_],
) -> None:
    """Evaluate B-spline basis functions using Cox-de Boor recursion.

    This function implements Algorithm 2.23 from "Spline Methods Draft" by Tom Lyche.
    Results are written directly to the output arrays (C-style).

    Args:
        knots (npt.NDArray[np.float32 | np.float64]): B-spline knot vector.
        degree (int): B-spline degree.
        periodic (bool): Whether the B-spline is periodic.
        tol (float): Tolerance for numerical comparisons.
        pts (npt.NDArray[np.float32 | np.float64]): Points (1D array) to evaluate basis
            functions at.
        out_basis (npt.NDArray[np.float32 | np.float64]): Output array for basis values.
            Must have shape (n_pts, degree+1) and dtype matching the `knots` dtype.
        out_first_basis (npt.NDArray[np.int_]): Output array for first basis indices.
            Must have shape (n_pts,) and dtype int.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    # See Spline Methods Draft, by Tom Lychee. Algorithm 2.23

    order = degree + 1
    n_pts = pts.size

    knot_ids = _get_last_knot_smaller_equal_impl(knots, pts)

    dtype = knots.dtype
    zero = dtype.type(0.0)
    one = dtype.type(1.0)

    # Initialize basis array
    out_basis.fill(zero)
    out_basis[:, -1] = one

    # Here we account for the case where the evaluation point
    # coincides with the last knot.
    num_basis = _get_Bspline_num_basis_1D_impl(knots, degree, periodic, tol)
    out_first_basis[:] = np.minimum(knot_ids - degree, num_basis - order)

    for pt_id in range(n_pts):
        knot_id = knot_ids[pt_id]

        if knot_id == (knots.size - 1):
            continue

        pt = pts[pt_id]
        basis_i = out_basis[pt_id, :]
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


@nb_jit(
    nopython=True,
    cache=True,
    parallel=False,
)
def _get_Bspline_cardinal_intervals_1D_core(
    knots: npt.NDArray[np.float32 | np.float64],
    degree: int,
    tol: float,
    out: npt.NDArray[np.bool_],
) -> None:
    """Core implementation to compute cardinal intervals, writing to output array.

    An interval is cardinal if it has the same length as the degree-1
    previous and the degree-1 next intervals.

    In the case of open knot vectors, this definition automatically
    discards the first degree-1 and the last degree-1 intervals.

    Args:
        knots (npt.NDArray[np.float32 | np.float64]): B-spline knot vector.
        degree (int): B-spline degree.
        tol (float): Tolerance for numerical comparisons.
        out (npt.NDArray[np.bool_]): Output array where results will be written.
            Must have the correct shape (no validation performed inside this
            numba-compiled function).

    Note:
        This is a Numba-compiled function optimized for performance. It
        expects pre-validated inputs and assumes the output array has the
        correct shape. Inputs are assumed to be correct (no validation performed).
        For general use, call _get_Bspline_cardinal_intervals_1D_impl instead.
    """
    _, mult = _get_unique_knots_and_multiplicity_impl(knots, degree, tol, in_domain=True)
    num_intervals = len(mult) - 1

    out.fill(np.False_)

    if np.all(mult > 1):
        return

    knot_id = degree

    # Note: this loop could be shortened by only looking at those
    # intervals for which the multiplicity of the first knot is 1.
    # This would require to compute knot_id differently.
    for elem_id in range(num_intervals):
        if mult[elem_id] == 1 and mult[elem_id + 1] == 1:
            local_knots = knots[knot_id - degree + 1 : knot_id + degree + 1]
            lengths = np.diff(local_knots)
            if np.all(np.isclose(lengths, lengths[degree - 1], atol=tol)):
                out[elem_id] = np.True_

        knot_id += mult[elem_id + 1]


def _get_Bspline_cardinal_intervals_1D_impl(
    knots: npt.NDArray[np.float32 | np.float64],
    degree: int,
    tol: float,
    out: npt.NDArray[np.bool_] | None = None,
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
        out (npt.NDArray[np.bool_] | None): Optional output array where the result will be
            stored. If None, a new array is allocated. Must have the correct shape and dtype
            if provided. This follows NumPy's style for output arrays. Defaults to None.

    Returns:
        npt.NDArray[np.bool_]: Boolean array where True indicates cardinal intervals.
            It has length equal to the number of intervals. If `out` was provided,
            returns the same array.

    Raises:
        ValueError: If the knot vector or degree fails basic validation or if tol is negative.
        ValueError: If `out` is provided and has incorrect shape or dtype.
    """
    _, mult = _get_unique_knots_and_multiplicity_impl(knots, degree, tol, in_domain=True)
    num_intervals = len(mult) - 1

    if out is None:
        out = np.empty(num_intervals, dtype=np.bool_)
    else:
        _validate_out_array_bool(out, (num_intervals,))

    _get_Bspline_cardinal_intervals_1D_core(knots, degree, tol, out)

    return out


@nb_jit(
    nopython=True,
    cache=True,
    parallel=False,
)
def _tabulate_Bspline_Bezier_1D_extraction_core(
    knots: npt.NDArray[np.float32 | np.float64],
    degree: int,
    tol: float,
    out: npt.NDArray[np.float32 | np.float64],
) -> None:
    r"""Core implementation to compute Bézier extraction operators, writing to output array.

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
        out (npt.NDArray[np.float32 | np.float64]): Output array where results will be written.
            Must have the correct shape (n_elems, degree+1, degree+1) and dtype matching knots
            (no validation performed inside this numba-compiled function).

    Note:
        This is a Numba-compiled function optimized for performance. It
        expects pre-validated inputs and assumes the output array has the
        correct shape and dtype. Inputs are assumed to be correct (no validation performed).
        For general use, call _tabulate_Bspline_Bezier_1D_extraction_impl instead.
    """
    unique_knots, mults = _get_unique_knots_and_multiplicity_impl(
        knots, degree, tol, in_domain=True
    )

    n_elems = len(unique_knots) - 1

    dtype = knots.dtype
    one = dtype.type(1.0)

    # Initialize identity matrix for every element.
    out.fill(0.0)
    out[:, : degree + 1, : degree + 1] = np.eye(degree + 1, dtype=dtype)

    mult = _get_multiplicity_of_first_knot_in_domain_impl(knots, degree, tol)

    # If not open first knot, additional knot insertion is needed.
    if mult < (degree + 1):
        C = out[0]
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

        C = out[elem_id]

        reg = degree - mult
        for r in range(1, reg + 1):
            s = mult + r
            for k in range(degree, s - 1, -1):
                alpha = alphas[k - s]
                C[:, k] = alpha * C[:, k] + (one - alpha) * C[:, k - 1]

            if elem_id < (n_elems - 1):
                out[elem_id + 1, reg - r : reg + 1, reg - r] = C[degree - r : degree + 1, degree]


def _tabulate_Bspline_Bezier_1D_extraction_impl(
    knots: npt.NDArray[np.float32 | np.float64],
    degree: int,
    tol: float,
    out: npt.NDArray[np.float32 | np.float64] | None = None,
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
        out (npt.NDArray[np.float32 | np.float64] | None): Optional output array where the result
            will be stored. If None, a new array is allocated. Must have the correct shape and dtype
            if provided. This follows NumPy's style for output arrays. Defaults to None.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Array of extraction matrices with shape
            (intervals, degree+1, degree+1) where each matrix transforms
            Bernstein basis functions to B-spline basis functions for that interval.
            If `out` was provided, returns the same array.

    Raises:
        ValueError: If the knot vector or degree fails basic validation or if tol is negative.
        ValueError: If `out` is provided and has incorrect shape or dtype.
    """
    if tol < 0:
        raise ValueError("tol must be positive")

    _check_spline_info(knots, degree)

    unique_knots, _ = _get_unique_knots_and_multiplicity_impl(knots, degree, tol, in_domain=True)

    n_elems = len(unique_knots) - 1
    dtype = knots.dtype
    expected_shape = (n_elems, degree + 1, degree + 1)

    if out is None:
        out = np.empty(expected_shape, dtype=dtype)
    else:
        _validate_out_array_3d_float(out, expected_shape, dtype)

    _tabulate_Bspline_Bezier_1D_extraction_core(knots, degree, tol, out)

    return out


def _tabulate_Bspline_Lagrange_1D_extraction_impl(
    knots: npt.NDArray[np.float32 | np.float64],
    degree: int,
    tol: float,
    lagrange_variant: LagrangeVariant = LagrangeVariant.EQUISPACES,
    out: npt.NDArray[np.float32 | np.float64] | None = None,
) -> npt.NDArray[np.float32 | np.float64]:
    """Create Lagrange extraction operators for a B-spline.

    Args:
        knots (npt.NDArray[np.float32 | np.float64]): B-spline knot vector.
        degree (int): B-spline degree.
        tol (float): Tolerance for numerical comparisons.
        lagrange_variant (LagrangeVariant): Lagrange point distribution
            (e.g., equispaced, gauss lobatto legendre, etc). Defaults to LagrangeVariant.EQUISPACES.
        out (npt.NDArray[np.float32 | np.float64] | None): Optional output array where the result
            will be stored. If None, a new array is allocated. Must have the correct shape and dtype
            if provided. This follows NumPy's style for output arrays. Defaults to None.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Array of extraction matrices with shape
            (n_intervals, degree+1, degree+1) where each matrix transforms
            Lagrange basis functions to B-spline basis functions for that interval.

            Each matrix C[i, :, :] transforms Bernstein basis functions
            to B-spline basis functions for the i-th interval as
                C[i, :, :] @ [Lagrange values] = [B-spline values in interval].
            If `out` was provided, returns the same array.

    Raises:
        ValueError: If the knot vector or degree fails basic validation or if tol is negative.
        ValueError: If `out` is provided and has incorrect shape or dtype.
    """
    if tol < 0:
        raise ValueError("tol must be positive")

    _check_spline_info(knots, degree)

    unique_knots, _ = _get_unique_knots_and_multiplicity_impl(knots, degree, tol, in_domain=True)
    n_elems = len(unique_knots) - 1
    dtype = knots.dtype
    expected_shape = (n_elems, degree + 1, degree + 1)

    if out is None:
        out = np.empty(expected_shape, dtype=dtype)
    else:
        _validate_out_array_3d_float(out, expected_shape, dtype)

    # Compute Bezier extraction into out
    _tabulate_Bspline_Bezier_1D_extraction_impl(knots, degree, tol, out=out)

    # Transform to Lagrange extraction in-place to avoid extra copy
    # For every interval, right-multiply the Bezier-to-B-spline extraction by lagr_to_bzr in-place.
    lagr_to_bzr = compute_Lagrange_to_Bernstein_change_basis_1D(degree, lagrange_variant, dtype)
    for i in range(out.shape[0]):
        # Use out[i, :, :] = out[i, :, :] @ lagr_to_bzr, but perform in-place via np.matmul
        # to avoid extra allocation.
        np.matmul(out[i, :, :], lagr_to_bzr, out=out[i, :, :])

    return out


def _tabulate_Bspline_cardinal_1D_extraction_impl(
    knots: npt.NDArray[np.float32 | np.float64],
    degree: int,
    tol: float,
    out: npt.NDArray[np.float32 | np.float64] | None = None,
) -> npt.NDArray[np.float32 | np.float64]:
    """Create cardinal B-spline extraction operators.

    For cardinal intervals, the extraction matrix is set to the identity matrix

    Args:
        knots (npt.NDArray[np.float32 | np.float64]): B-spline knot vector.
        degree (int): B-spline degree.
        tol (float): Tolerance for numerical comparisons.
        out (npt.NDArray[np.float32 | np.float64] | None): Optional output array where the result
            will be stored. If None, a new array is allocated. Must have the correct shape and dtype
            if provided. This follows NumPy's style for output arrays. Defaults to None.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Array of extraction matrices with shape
            (n_intervals, degree+1, degree+1) where each matrix transforms
            cardinal B-spline basis functions to B-spline basis functions for that interval.

            Each matrix C[i, :, :] transforms cardinal B-spline basis functions
            to B-spline basis functions for the i-th interval as
                C[i, :, :] @ [cardinal values] = [B-spline values in interval].
            If `out` was provided, returns the same array.

    Raises:
        ValueError: If the knot vector or degree fails basic validation or if tol is negative.
        ValueError: If `out` is provided and has incorrect shape or dtype.
    """
    if tol < 0:
        raise ValueError("tol must be positive")

    _check_spline_info(knots, degree)

    unique_knots, _ = _get_unique_knots_and_multiplicity_impl(knots, degree, tol, in_domain=True)
    n_elems = len(unique_knots) - 1
    dtype = knots.dtype
    expected_shape = (n_elems, degree + 1, degree + 1)

    if out is None:
        out = np.empty(expected_shape, dtype=dtype)
    else:
        _validate_out_array_3d_float(out, expected_shape, dtype)

    # Compute Bezier extraction into out
    _tabulate_Bspline_Bezier_1D_extraction_impl(knots, degree, tol, out=out)

    # Transform to cardinal extraction
    card_to_bzr = compute_cardinal_to_Bernstein_change_basis_1D(degree, dtype)
    # Transform to cardinal extraction in-place to avoid unnecessary copy
    # out[...] = out @ card_to_bzr is not strictly in-place (it creates a new array then assigns)
    # To perform an in-place transformation, use np.matmul (or @) with out as the output
    np.matmul(out, card_to_bzr, out=out)

    # Set identity for cardinal intervals
    cardinal_intervals = _get_Bspline_cardinal_intervals_1D_impl(knots, degree, tol)
    for i in np.where(cardinal_intervals)[0]:
        out[i, :, :] = np.eye(degree + 1, dtype=dtype)

    return out


def _tabulate_Bspline_basis_Bernstein_like_1D(
    spline: BsplineSpace1D,
    pts: npt.NDArray[np.float32 | np.float64],
    out_basis: npt.NDArray[np.float32 | np.float64] | None = None,
    out_first_basis: npt.NDArray[np.int_] | None = None,
) -> tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.int_]]:
    """Evaluate B-spline basis functions when they reduce to Bernstein polynomials.

    This function is used when the B-spline has Bézier-like knots, allowing
    direct evaluation using Bernstein basis functions.

    Args:
        spline (BsplineSpace1D): B-spline object with Bézier-like knots.
        pts (npt.NDArray[np.float32 | np.float64]): Evaluation points (already normalized to 1D).
        out_basis (npt.NDArray[np.float32 | np.float64] | None): Optional output array where the
            basis values will be stored. If None, a new array is allocated. Must have the
            correct shape (num_pts, degree+1) and dtype if provided. This follows NumPy's
            style for output arrays. Defaults to None.
        out_first_basis (npt.NDArray[np.int_] | None): Optional output array where the
            first basis indices will be stored. If None, a new array is allocated. Must have
            the correct shape (num_pts,) and dtype np.int_ if provided. This follows NumPy's
            style for output arrays. Defaults to None.

    Returns:
        tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.int_]]: Tuple of
            (basis_values, first_basis_indices) where basis_values is an array of shape
            (number pts, degree+1) that contains the Bernstein basis function values and
            first_basis_indices contains the indices of the first non-zero basis function
            for each point. If `out_basis` or `out_first_basis` was provided,
            returns the same array(s).

    Raises:
        ValueError: If the B-spline does not have Bézier-like knots.
        ValueError: If `out_basis` or `out_first_basis` is provided and has incorrect shape
            or dtype.
    """
    if not spline.has_Bezier_like_knots():
        raise ValueError("B-spline does not have Bézier-like knots.")

    # map the points to the reference interval [0, 1]
    k0, k1 = spline.domain
    pts_normalized = (pts - k0) / (k1 - k0)

    num_pts = pts.size
    expected_first_basis_shape = (num_pts,)

    if out_first_basis is None:
        out_first_basis = np.empty(expected_first_basis_shape, dtype=np.int_)
    else:
        _validate_out_array_first_basis(out_first_basis, expected_first_basis_shape)

    # the first basis function is always the 0
    out_first_basis.fill(0)

    # Compute Bernstein basis - pass out_basis directly since pts_normalized is already 1D
    # and _tabulate_Bernstein_basis_1D_impl will handle shape validation
    B = _tabulate_Bernstein_basis_1D_impl(spline.degree, pts_normalized, out=out_basis)

    return B, out_first_basis


def _tabulate_Bspline_basis_1D_impl(
    spline: BsplineSpace1D,
    pts: npt.ArrayLike,
    out_basis: npt.NDArray[np.float32 | np.float64] | None = None,
    out_first_basis: npt.NDArray[np.int_] | None = None,
) -> tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.int_]]:
    """Evaluate B-spline basis functions at given points.

    This function automatically selects the most efficient evaluation method:
    - For Bézier-like knots: direct Bernstein evaluation
    - For general knots: Cox-de Boor recursion

    In both cases it calls vectorized or numba implementations.

    Args:
        spline (BsplineSpace1D): B-spline object defining the basis.
        pts (npt.ArrayLike): Evaluation points.
        out_basis (npt.NDArray[np.float32 | np.float64] | None): Optional output array where the
            basis values will be stored. If None, a new array is allocated. Must have the
            correct shape and dtype if provided. This follows NumPy's style for output arrays.
            Defaults to None.
        out_first_basis (npt.NDArray[np.int_] | None): Optional output array where the
            first basis indices will be stored. If None, a new array is allocated. Must have
            the correct shape and dtype np.int_ if provided. This follows NumPy's style for
            output arrays. Defaults to None.

    Returns:
        tuple[
            npt.NDArray[np.float32] | npt.NDArray[np.float64],
            npt.NDArray[np.int_]
        ]: Tuple containing:
            - basis_values: (npt.NDArray[np.float32] | npt.NDArray[np.float64])
              Array of shape matching `pts` with the last dimension length (degree+1),
              containing the basis function values evaluated at each point.
              If `out_basis` was provided, returns the same array.
            - first_basis_indices: (npt.NDArray[np.int_])
              1D integer array indicating the index of the first nonzero basis function
              for each evaluation point. The length is the same as the number of evaluation points.
              If `out_first_basis` was provided, returns the same array.

    Raises:
        ValueError: If any evaluation points are outside the B-spline domain, or if `out_basis`
            or `out_first_basis` is provided and has incorrect shape or dtype.

    Example:
        >>> bspline = BsplineSpace1D([0, 0, 0, 0.25, 0.7, 0.7, 1, 1, 1], 2)
        >>> _tabulate_Bspline_basis_1D_impl(bspline, [0.0, 0.5, 0.75, 1.0])
        (array([[1.        , 0.        , 0.        ],
                [0.12698413, 0.5643739 , 0.30864198],
                [0.69444444, 0.27777778, 0.02777778],
                [0.        , 0.        , 1.        ]]),
         array([0, 1, 3, 3]))
    """
    input_shape = np.shape(pts)
    pts = _normalize_points_1D(pts)

    if not np.all(_is_in_domain_impl(spline.knots, spline.degree, pts, spline.tolerance)):
        raise ValueError(
            f"One or more values in pts are outside the knot vector domain {spline.domain}"
        )

    num_pts = pts.shape[0]
    n_basis = spline.degree + 1
    expected_final_shape = _compute_final_output_shape_1D(input_shape, n_basis)
    expected_dtype = pts.dtype
    expected_first_basis_shape = input_shape

    if out_basis is None:
        out_basis = np.empty(expected_final_shape, dtype=expected_dtype)
    _validate_out_array_1D(out_basis, expected_final_shape, expected_dtype)
    basis_normalized = out_basis.reshape(num_pts, n_basis)

    if out_first_basis is None:
        out_first_basis = np.empty(expected_first_basis_shape, dtype=np.int_)
    _validate_out_array_first_basis(out_first_basis, expected_first_basis_shape)
    first_indices_normalized = out_first_basis.reshape(num_pts)

    if spline.has_Bezier_like_knots():
        _tabulate_Bspline_basis_Bernstein_like_1D(
            spline, pts, basis_normalized, first_indices_normalized
        )
    else:
        _compute_basis_Cox_de_Boor_impl(
            spline.knots,
            spline.degree,
            spline.periodic,
            spline.tolerance,
            pts,
            basis_normalized,
            first_indices_normalized,
        )

    return out_basis, out_first_basis


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


def _get_knots_ends_and_dtype(
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


def _tabulate_Bspline_basis_for_points_array_impl(
    spline: BsplineSpace,
    pts: npt.NDArray[np.float32 | np.float64],
    out_basis: npt.NDArray[np.float32 | np.float64] | None = None,
    out_first_basis: npt.NDArray[np.int_] | None = None,
) -> tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.int_]]:
    """Evaluate multi-dimensional B-spline basis functions at given points.

    Args:
        spline (BsplineSpace): B-spline object defining the basis.
        pts (npt.NDArray[np.float32 | np.float64]): Evaluation points.
            Must be a 2D array with shape (num_pts, dim).
        out_basis (npt.NDArray[np.float32 | np.float64] | None): Optional output array where the
            basis values will be stored. If None, a new array is allocated. Must have the
            correct shape and dtype if provided. This follows NumPy's style for output arrays.
            Defaults to None.
        out_first_basis (npt.NDArray[np.int_] | None): Optional output array where the
            first basis indices will be stored. If None, a new array is allocated. Must have
            the correct shape (num_pts, dim) and dtype np.int_ if provided. This follows NumPy's
            style for output arrays. Defaults to None.

    Returns:
        tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.int_]]: Tuple containing:
            - basis_values: (npt.NDArray[np.float32 | np.float64])
              Array of shape (num_pts, order[0], order[1], ..., order[d-1])
              containing the basis function values evaluated at each point.
              If `out_basis` was provided, returns the same array.
            - first_basis_indices: (npt.NDArray[np.int_])
              2D integer array indicating the index of the first nonzero basis function
              for each evaluation point in each direction. The shape is (num_pts, dim).
              If `out_first_basis` was provided, returns the same array.

    Raises:
        ValueError: If pts is not a 2D array or does not have the correct number of columns.
        ValueError: If one or more values in pts are outside the knot vector domain, or
            if `out_basis` or `out_first_basis` is provided and has incorrect shape or dtype.
    """
    if pts.ndim != 2:  # noqa: PLR2004
        raise ValueError("pts must be a 2D array")
    if pts.shape[1] != spline.dim:
        raise ValueError(f"pts must have {spline.dim} columns")

    splines_1D = spline.spaces
    for dir in range(spline.dim):
        spline_1D = splines_1D[dir]
        pts_dir = np.ascontiguousarray(pts[:, dir])
        if not np.all(
            _is_in_domain_impl(spline_1D.knots, spline_1D.degree, pts_dir, spline_1D.tolerance)
        ):
            raise ValueError(
                f"One or more values in pts[:, {dir}] are outside the knot vector"
                f" domain {spline.domain}"
            )

    order = tuple(int(degree + 1) for degree in spline.degrees)
    num_pts = pts.shape[0]
    expected_basis_shape = (num_pts, *order)
    expected_dtype = np.dtype(spline.dtype)
    expected_first_basis_shape = (num_pts, spline.dim)

    if out_basis is None:
        out_basis = cast(
            npt.NDArray[np.float32 | np.float64],
            np.empty(expected_basis_shape, dtype=expected_dtype),
        )
    else:
        _validate_out_array_1D(out_basis, expected_basis_shape, expected_dtype)

    if out_first_basis is None:
        out_first_basis = np.empty(expected_first_basis_shape, dtype=np.int_)
    else:
        _validate_out_array_first_basis(out_first_basis, expected_first_basis_shape)

    # Combine 1D basis along each direction using outer product to form the
    # tensor-product multidimensional basis.
    # The multidimensional basis Bs will have shape (num_pts, order[0], order[1], ..., order[d-1])

    # Start with the basis functions of the first direction
    pts_0 = np.ascontiguousarray(pts[:, 0])
    first_out = out_basis if (spline.dim == 1 and out_basis is not None) else None
    B_multi, first_idx_0 = splines_1D[0].tabulate_basis(pts_0, out_basis=first_out)

    out_first_basis[:, 0] = first_idx_0

    for dir in range(1, spline.dim):
        pts_dir = np.ascontiguousarray(pts[:, dir])
        Bdir, first_idx_dir = splines_1D[dir].tabulate_basis(pts_dir)
        out_first_basis[:, dir] = first_idx_dir
        # At each step, expand B_multi to add a new axis at the end,
        # and outer product with the next B_1D
        expanded_dir_shape = (num_pts,) + ((1,) * dir) + (order[dir],)
        Bdir_view = Bdir.reshape(expanded_dir_shape)

        is_last_dir = dir == (spline.dim - 1)
        if is_last_dir and out_basis is not None:
            B_multi = np.multiply(B_multi[..., np.newaxis], Bdir_view, out=out_basis)
        else:
            B_multi = np.multiply(B_multi[..., np.newaxis], Bdir_view)

    return out_basis, out_first_basis


def _tabulate_Bspline_basis_for_points_lattice_impl(
    spline: BsplineSpace,
    pts: PointsLattice,
    out_basis: npt.NDArray[np.float32 | np.float64] | None = None,
    out_first_basis: npt.NDArray[np.int_] | None = None,
) -> tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.int_]]:
    """Evaluate B-spline basis functions at points in a lattice structure.

    This function computes the tensor product of the 1D B-spline basis
    functions evaluated over the lattice points defined by 'pts'. The output
    arrays capture the non-zero local basis values and their corresponding
    global starting indices efficiently for the full grid.

    Args:
        spline (BsplineSpace): B-spline object defining the basis (knots, degree, etc.).
        pts (PointsLattice): Evaluation points defined as a tensor product lattice.
        out_basis (npt.NDArray[np.float32 | np.float64] | None): Optional output array where the
            basis values will be stored. If None, a new array is allocated. Must have the
            correct shape and dtype if provided. This follows NumPy's style for output arrays.
            Defaults to None.
        out_first_basis (npt.NDArray[np.int_] | None): Optional output array where the
            first basis indices will be stored. If None, a new array is allocated. Must have
            the correct shape (n_pts_0, n_pts_1, ..., n_pts_d, dim) and dtype np.int_ if
            provided. This follows NumPy's style for output arrays. Defaults to None.

    Returns:
        tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.int_]]: Tuple containing:
            - basis_values: (npt.NDArray[np.float32 | np.float64])
              Array of shape (n_pts_0, n_pts_1, ..., n_pts_d, k_0, k_1, ..., k_d)
              where 'n_pts_i' is the number of points in dimension $i$, and $k_i$
              is the number of local non-zero basis functions (typically degree + 1).
              This array contains the tensor product of basis function values evaluated
              at each grid point. If `out_basis` was provided, returns the same array.
            - first_basis_indices: (npt.NDArray[np.int_])
              Array of shape (n_pts_0, n_pts_1, ..., n_pts_d, dim) indicating the global
              index of the first nonzero basis function for each evaluation point in each direction.
              If `out_first_basis` was provided, returns the same array.

    Raises:
        ValueError: If pts dimension does not match spline dimension.
        ValueError: If one or more values in pts are outside the knot vector domain.
        ValueError: If one or more values in pts are outside the corresponding knot vector domain,
            or if `out_basis` or `out_first_basis` is provided and has incorrect shape or dtype.
    """
    if pts.dim != spline.dim:
        raise ValueError(f"pts must have {spline.dim} columns")

    for dir in range(spline.dim):
        if not np.all(
            _is_in_domain_impl(
                spline.spaces[dir].knots,
                spline.spaces[dir].degree,
                pts.pts_per_dir[dir],
                spline.spaces[dir].tolerance,
            )
        ):
            raise ValueError(
                f"One or more values in pts.pts_per_dir[{dir}] are outside the knot vector"
                f" domain {spline.spaces[dir].domain}"
            )

    # The domain check logic would typically be inserted here before
    # the 1D tabulation, but is omitted for brevity in this simplified template.

    # 1. Compute 1D components
    results_1d = [s.tabulate_basis(p) for s, p in zip(spline.spaces, pts.pts_per_dir, strict=True)]
    Bs_tuple, first_idxs = zip(*results_1d, strict=True)
    Bs: list[npt.NDArray[np.float32 | np.float64]] = list(Bs_tuple)

    # 2. Combine basis functions using tensor product (Broadcasting approach)
    ndim = spline.dim

    pts_shape = tuple(B.shape[0] for B in Bs)
    order_shape = tuple(B.shape[1] for B in Bs)
    expected_basis_shape = pts_shape + order_shape
    expected_dtype = Bs[0].dtype
    expected_first_basis_shape = (*pts_shape, ndim)

    if out_basis is None:
        out_basis = cast(
            npt.NDArray[np.float32 | np.float64],
            np.empty(expected_basis_shape, dtype=expected_dtype),
        )
    else:
        _validate_out_array_1D(out_basis, expected_basis_shape, expected_dtype)

    if out_first_basis is None:
        out_first_basis = np.empty(expected_first_basis_shape, dtype=np.int_)
    else:
        _validate_out_array_first_basis(out_first_basis, expected_first_basis_shape)

    # 3. Combine first_indices into a multi-dimensional array (Meshgrid approach)
    # Result shape: (n_pts_0, n_pts_1, ..., n_pts_d, dim)
    first_mesh = np.meshgrid(*first_idxs, indexing="ij")
    for axis, grid in enumerate(first_mesh):
        out_first_basis[..., axis] = grid

    # Handle 1D case separately
    first_B = Bs[0]
    if ndim == 1:
        np.copyto(out_basis, first_B.reshape(expected_basis_shape))
        return out_basis, out_first_basis

    # Initialize with first array for multi-dimensional case
    new_shape = [1] * (2 * ndim)
    new_shape[0] = first_B.shape[0]
    new_shape[ndim] = first_B.shape[1]
    B_multi: npt.NDArray[np.float32 | np.float64] = first_B.reshape(new_shape)

    # Multi-dimensional case: iterate through remaining dimensions
    remaining_Bs = list(Bs[1:])
    for i, B in enumerate(remaining_Bs, start=1):
        new_shape = [1] * (2 * ndim)
        new_shape[i] = B.shape[0]
        new_shape[ndim + i] = B.shape[1]
        B_view = B.reshape(new_shape)

        is_last_iteration = i == len(remaining_Bs)
        if is_last_iteration:
            np.multiply(B_multi, B_view, out=out_basis)
        else:
            B_multi = np.multiply(B_multi, B_view)

    return out_basis, out_first_basis


def _tabulate_Bspline_basis_impl(
    spline: BsplineSpace,
    pts: npt.NDArray[np.float32 | np.float64] | PointsLattice,
    out_basis: npt.NDArray[np.float32 | np.float64] | None = None,
    out_first_basis: npt.NDArray[np.int_] | None = None,
) -> tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.int_]]:
    """Evaluate multi-dimensional B-spline basis functions at given points.

    Args:
        spline (BsplineSpace): B-spline object defining the basis.
        pts (npt.NDArray[np.float32 | np.float64] | PointsLattice): Evaluation points.
            Must be a 2D array with shape (num_pts, dim) or a PointsLattice object.
        out_basis (npt.NDArray[np.float32 | np.float64] | None): Optional output array where the
            basis values will be stored. If None, a new array is allocated. Must have the
            correct shape and dtype if provided. This follows NumPy's style for output arrays.
            Defaults to None.
        out_first_basis (npt.NDArray[np.int_] | None): Optional output array where the
            first basis indices will be stored. If None, a new array is allocated. Must have
            the correct shape and dtype np.int_ if provided. This follows NumPy's style for
            output arrays. Defaults to None.

    Returns:
        tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.int_]]: Tuple containing:
            - basis_values: (npt.NDArray[np.float32 | np.float64])
              Array of shape (num_pts, order[0], order[1], ..., order[d-1])
              containing the basis function values evaluated at each point.
              If `out_basis` was provided, returns the same array.
            - first_basis_indices: (npt.NDArray[np.int_])
              Array of shape (num_pts, dim) or (n_pts_0, n_pts_1, ..., n_pts_d, dim) indicating
              the global index of the first nonzero basis function for each evaluation point.
              If `out_first_basis` was provided, returns the same array.

    Raises:
        ValueError: If pts dimension does not match spline dimension.
        ValueError: If one or more values in pts are outside the knot vector domain.
        ValueError: If one or more values in pts are outside the corresponding knot vector domain,
            or if `out_basis` or `out_first_basis` is provided and has incorrect shape or dtype.
    """
    if isinstance(pts, PointsLattice):
        return _tabulate_Bspline_basis_for_points_lattice_impl(
            spline, pts, out_basis=out_basis, out_first_basis=out_first_basis
        )
    else:
        return _tabulate_Bspline_basis_for_points_array_impl(
            spline, pts, out_basis=out_basis, out_first_basis=out_first_basis
        )


def _warmup_numba_functions() -> None:
    """Precompile numba functions with float64 signatures for faster first call.

    This function triggers compilation of the numba-decorated functions
    with float64 arrays, ensuring they are cached and ready for use.
    """
    # Small dummy arrays for warmup
    knots_dummy = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)
    pts_dummy = np.array([0.5], dtype=np.float64)
    tol_dummy = 1e-10
    degree_dummy = 2

    # Warmup each numba function with float64
    _check_spline_info(knots_dummy, degree_dummy)
    _get_multiplicity_of_first_knot_in_domain_impl(knots_dummy, degree_dummy, tol_dummy)
    _get_unique_knots_and_multiplicity_impl(knots_dummy, degree_dummy, tol_dummy, False)
    _is_in_domain_impl(knots_dummy, degree_dummy, pts_dummy, tol_dummy)
    _get_Bspline_num_basis_1D_impl(knots_dummy, degree_dummy, False, tol_dummy)
    _get_last_knot_smaller_equal_impl(knots_dummy, pts_dummy)
    n_pts_dummy = pts_dummy.size
    basis_dummy = np.empty((n_pts_dummy, degree_dummy + 1), dtype=np.float64)
    first_basis_dummy = np.empty(n_pts_dummy, dtype=np.int_)
    _compute_basis_Cox_de_Boor_impl(
        knots_dummy, degree_dummy, False, tol_dummy, pts_dummy, basis_dummy, first_basis_dummy
    )
    _get_Bspline_cardinal_intervals_1D_impl(knots_dummy, degree_dummy, tol_dummy)
    _tabulate_Bspline_Bezier_1D_extraction_impl(knots_dummy, degree_dummy, tol_dummy)


# Precompile numba functions on module import (skip during type checking)
if not TYPE_CHECKING:
    _warmup_numba_functions()
