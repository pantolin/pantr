"""Numba-backed core implementations for 1D Bernstein and cardinal B-spline bases."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, TypeVar, cast

import numba as nb
import numpy as np
import numpy.typing as npt
from scipy.interpolate import BarycentricInterpolator

from ._basis_utils import (
    _compute_final_output_shape_1D,
    _compute_output_shape_multidimensional,
    _normalize_points_1D,
    _validate_out_array_1D,
    _validate_out_array_multidimensional,
)
from .quad import (
    PointsLattice,
    get_chebyshev_gauss_1st_kind_quadrature_1D,
    get_chebyshev_gauss_2nd_kind_quadrature_1D,
    get_gauss_legendre_quadrature_1D,
    get_gauss_lobatto_legendre_quadrature_1D,
    get_trapezoidal_quadrature_1D,
)

F = TypeVar("F", bound=Callable[..., Any])

if TYPE_CHECKING:
    from .basis import LagrangeVariant

    # During type-checking, make the decorator a no-op that preserves types.
    def nb_jit(*args: object, **kwargs: object) -> Callable[[F], F]:
        def decorator(func: F) -> F:
            return func

        return decorator
else:
    # At runtime, use the real Numba decorator.
    nb_jit = nb.jit  # type: ignore[attr-defined]


@nb_jit(
    nopython=True,
    cache=True,
    parallel=False,
)
def _tabulate_Bernstein_basis_1D_core(
    n: np.int32,
    t: npt.NDArray[np.float32 | np.float64],
    out: npt.NDArray[np.float32 | np.float64],
) -> None:
    """Evaluate Bernstein basis polynomials of degree n at points t.

    Computes all (n+1) Bernstein basis polynomials B_i,n(t) for i=0,...,n
    at each evaluation point in t. Writes the result to the provided output array,
    where out[j, i] contains the value of the i-th basis polynomial evaluated at point t_j.

    The implementation uses a recurrence relation for efficient computation:
    - Base case: B_0,n(t) = (1-t)^n
    - Recurrence: B_i,n(t) = B_{i-1},n(t) * ((n-i+1)/i) * (t/(1-t))

    Special handling is applied for points where t=1.0, where the recurrence
    relation would produce NaN values. At t=1.0, only the last basis function
    B_n,n(1) = 1, while all others are 0.

    Args:
        n (np.int32): Degree of the Bernstein polynomials. Must be non-negative.
        t (npt.NDArray[np.float32 | np.float64]): 1D array of
            evaluation points. Must be a contiguous array of float32 or float64
            values. Points should typically be in [0, 1], though the function
            will compute values for any real t.
        out (npt.NDArray[np.float32 | np.float64]): Output array
            of shape (len(t), n+1) and dtype matching t. The function writes
            the evaluated basis functions to this array. Must have the correct
            shape and dtype (no validation performed inside this numba-compiled function).

    Note:
        This is a Numba-compiled function optimized for performance. It
        expects pre-normalized inputs (1D contiguous arrays) and assumes the
        output array has the correct shape and dtype. For general use,
        call _tabulate_Bernstein_basis_1D_impl instead.
    """
    if n == 0:
        # The basis is just B_0,0(pts) = 1
        for j in range(out.shape[0]):
            out[j, 0] = 1.0
        return

    # Process each point
    for j in range(t.shape[0]):
        u = t[j]
        if u == 1.0:
            # At t=1.0: only B_n,n(1) = 1, all others are 0
            for i in range(out.shape[1]):
                out[j, i] = 0.0
            out[j, n] = 1.0
        else:
            # For t != 1.0, use recurrence relation
            one_minus_u = 1.0 - u
            out[j, 0] = np.power(one_minus_u, n)
            if n > 0:
                t_over_1mt = u / one_minus_u
                for i in range(1, n + 1):
                    const_factor = (n - i + 1.0) / i
                    out[j, i] = out[j, i - 1] * const_factor * t_over_1mt


@nb_jit(
    nopython=True,
    cache=True,
    parallel=False,
)
def _tabulate_cardinal_Bspline_basis_1D_core(
    n: np.int32,
    t: npt.NDArray[np.float32 | np.float64],
    out: npt.NDArray[np.float32 | np.float64],
) -> None:
    """Evaluate the central cardinal B-spline basis of degree n on [0, 1].

    Computes the (n+1) nonzero B-spline basis functions active over the central
    unit span [0, 1] of a uniform knot vector with unit spacing, at each
    evaluation point in t. Values are zero outside [0, 1]. Uses the stable
    Cox-de Boor (BasisFuns) recursion specialized to span index i=0.

    Args:
        n (np.int32): Degree of the B-spline basis (>= 0).
        t (npt.NDArray[np.float32 | np.float64]): 1D array of
            evaluation points (float32 or float64).
        out (npt.NDArray[np.float32 | np.float64]): Output array
            of shape (len(t), n+1) and dtype matching t. The function writes
            the evaluated basis functions to this array. Must have the correct
            shape and dtype (no validation performed inside this numba-compiled function).
    """
    num_pts = t.shape[0]
    # Initialize first column to 1.0
    for j in range(num_pts):
        out[j, 0] = 1.0

    if n == 0:  # Degree-0: basis function is constant.
        return

    for j in range(num_pts):
        u = t[j]
        one_minus_u = 1.0 - u

        for k in range(1, n + 1):
            inv_k = 1.0 / k
            saved = 0.0
            for r in range(k):
                Nr_old = out[j, r]
                term = (r + one_minus_u) * inv_k
                out[j, r] = saved + Nr_old * term
                saved = Nr_old * (1.0 - term)
            out[j, k] = saved


def _get_lagrange_points(
    variant: LagrangeVariant, n_pts: int, dtype: npt.DTypeLike = np.float64
) -> npt.NDArray[np.float32 | np.float64]:
    """Get nodes for a Lagrange basis on [0, 1] for the given variant and size.

    Note that for Gauss-Lobatto-Legendre and Chebyshev second kind the
    number of points must be at least 2.

    Args:
        variant (LagrangeVariant): The variant of the Lagrange basis.
        n_pts (int): The number of points.
        dtype (npt.DTypeLike): The dtype of the nodes. If must be float32 or float64.
            Defaults to float64.

    Returns:
        npt.NDArray[np.float32 | np.float64]: The nodes.

    Raises:
        ValueError: If dtype is not float32 or float64.
    """
    variant_value = getattr(variant, "value", variant)
    if variant_value == "equispaces":
        return get_trapezoidal_quadrature_1D(n_pts, dtype)[0]
    elif variant_value == "gauss_legendre":
        return get_gauss_legendre_quadrature_1D(n_pts, dtype)[0]
    elif variant_value == "gauss_lobatto_legendre":
        return get_gauss_lobatto_legendre_quadrature_1D(n_pts, dtype)[0]
    elif variant_value == "chebyshev_1st":
        return get_chebyshev_gauss_1st_kind_quadrature_1D(n_pts, dtype)[0]
    else:  # "chebyshev_2nd"
        return get_chebyshev_gauss_2nd_kind_quadrature_1D(n_pts, dtype)[0]


def _tabulate_lagrange_basis_1D_core(
    n: int,
    variant: LagrangeVariant,
    t: npt.NDArray[np.float32 | np.float64],
    B_normalized: npt.NDArray[np.float32 | np.float64],
) -> None:
    """Compute Lagrange basis values into the provided normalized array.

    Args:
        n (int): Degree of the Lagrange basis. Must be non-negative.
        variant (LagrangeVariant): Variant of the Lagrange basis.
        t (npt.NDArray[np.float32 | np.float64]): Normalized 1D array
            of evaluation points.
        B_normalized (npt.NDArray[np.float32 | np.float64]): Output array
            of shape (len(t), n+1) where results will be written. Must have the correct
            shape and dtype matching t.
    """
    n_pts = n + 1

    # Degree-zero: basis is constant 1 for all evaluation points and variants.
    if n == 0:
        B_normalized[:, 0] = 1.0
        return

    nodes = _get_lagrange_points(variant, n + 1, t.dtype)

    # Ensure nodes are strictly increasing for numerical robustness and consistency
    perm = np.argsort(nodes)
    nodes_sorted = nodes[perm]
    # Precompute inverse permutation: inv_perm[idx_in_sorted] = original_index
    inv_perm = np.empty_like(perm)
    inv_perm[perm] = np.arange(n_pts, dtype=perm.dtype)

    for j in range(n_pts):
        # Set 1 at the position corresponding to node j in the sorted order
        y_sorted = np.zeros(n_pts, dtype=t.dtype)
        y_sorted[inv_perm[j]] = 1.0

        interpolator = BarycentricInterpolator(nodes_sorted, y_sorted)

        # SciPy may upcast internally; ensure we preserve input dtype.
        B_normalized[:, j] = np.asarray(interpolator(t), dtype=t.dtype)

    # Snap to exact Kronecker-delta at interpolation nodes to avoid tiny roundoff
    # off-diagonals (notably with float32 and Chebyshev nodes).
    if B_normalized.dtype == np.float32:
        eps = np.finfo(np.float32).eps * 16.0
    else:
        eps = np.finfo(np.float64).eps * 16.0
    diffs = np.abs(t[:, None] - nodes_sorted[None, :])
    matches = diffs <= eps
    if np.any(matches):
        row_has_match = np.any(matches, axis=1)
        if np.any(row_has_match):
            matched_k = np.argmax(matches, axis=1)
            rows = np.nonzero(row_has_match)[0]
            ks = matched_k[rows]
            cols = perm[ks]
            # Zero full matched rows then set the diagonal 1
            B_normalized[rows, :] = 0.0
            B_normalized[rows, cols] = 1.0


@nb_jit(
    nopython=True,
    cache=True,
    parallel=False,
)
def _tabulate_Legendre_basis_1D_core(
    n: np.int32,
    t: npt.NDArray[np.float32 | np.float64],
    out: npt.NDArray[np.float32 | np.float64],
) -> None:
    """Evaluate normalized Shifted Legendre basis polynomials of degree n at points t.

    Computes all (n+1) basis polynomials p_i(t) for i=0,...,n
    at each evaluation point in t. Writes the result to the provided output array,
    where out[j, i] contains the value of the i-th basis polynomial evaluated at point t_j.

    The implementation uses the recurrence relation for normalized shifted Legendre polynomials:
    p_0(x) = 1
    p_1(x) = sqrt(3)(2x-1)
    p_i(x) = (sqrt(2i-1)sqrt(2i+1)/i) * (2x-1) * p_{i-1}(x)
             - ((i-1)/i) * sqrt((2i+1)/(2i-3)) * p_{i-2}(x)

    Args:
        n (np.int32): Degree of the Legendre polynomials. Must be non-negative.
        t (npt.NDArray[np.float32 | np.float64]): 1D array of
            evaluation points.
        out (npt.NDArray[np.float32 | np.float64]): Output array
            of shape (len(t), n+1) and dtype matching t. The function writes
            the evaluated basis functions to this array. Must have the correct
            shape and dtype (no validation performed inside this numba-compiled function).
    """
    num_pts = t.shape[0]

    # p_0(x) = 1
    for j in range(num_pts):
        out[j, 0] = 1.0

    if n == 0:
        return

    # p_1(x) = sqrt(3)(2x-1)
    sqrt3 = np.sqrt(3.0)

    # Compute 2x - 1 for all points and p_1
    for j in range(num_pts):
        two_x_minus_1 = 2.0 * t[j] - 1.0
        out[j, 1] = sqrt3 * two_x_minus_1

    for i in range(2, n + 1):
        # Coefficients
        # a_i = (sqrt(2i-1)sqrt(2i+1)/i)
        # b_i = ((i-1)/i) * sqrt((2i+1)/(2i-3))

        i_float = float(i)
        sqrt_2i_minus_1 = np.sqrt(2.0 * i_float - 1.0)
        sqrt_2i_plus_1 = np.sqrt(2.0 * i_float + 1.0)
        sqrt_2i_minus_3 = np.sqrt(2.0 * i_float - 3.0)

        a_i = (sqrt_2i_minus_1 * sqrt_2i_plus_1) / i_float
        b_i = ((i_float - 1.0) / i_float) * (sqrt_2i_plus_1 / sqrt_2i_minus_3)

        for j in range(num_pts):
            two_x_minus_1 = 2.0 * t[j] - 1.0
            out[j, i] = a_i * two_x_minus_1 * out[j, i - 1] - b_i * out[j, i - 2]


def _tabulate_basis_1D_impl_helper(
    n: int,
    t: npt.ArrayLike,
    core_func: Callable[
        [np.int32, npt.NDArray[np.float32 | np.float64], npt.NDArray[np.float32 | np.float64]],
        None,
    ],
    out: npt.NDArray[np.float32 | np.float64] | None = None,
) -> npt.NDArray[np.float32 | np.float64]:
    """Common implementation for tabulating 1D basis functions.

    Handles input normalization, output allocation/validation, and dtype dispatching
    for calling the appropriate core function.

    Args:
        n (int): Degree of the basis polynomials. Must be non-negative.
        t (npt.ArrayLike): Evaluation points. Can be a scalar, list, or numpy array.
            Types different from float32 or float64 are automatically converted to float64.
        core_func: Core function to call for computation. Must accept
            (np.int32, npt.NDArray[float32/float64], npt.NDArray[float32/float64]) -> None.
        out (npt.NDArray[np.float32 | np.float64] | None): Optional output array
            where the result will be stored. If None, a new array is allocated.
            Must have the correct shape and dtype if provided. Defaults to None.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Evaluated basis functions, with the same shape
        as the input points and the last dimension equal to (degree + 1).
        If `out` was provided, returns the same array.

    Raises:
        ValueError: If degree is negative, or if `out` is provided and has incorrect
            shape or dtype.
    """
    if n < 0:
        raise ValueError("degree must be non-negative")

    # Get input shape before normalization (handle scalars and lists)
    if isinstance(t, np.ndarray):
        input_shape = t.shape
    elif isinstance(t, list | tuple):
        input_shape = np.array(t).shape
    else:  # scalar
        input_shape = ()

    t = _normalize_points_1D(t)
    num_pts = t.shape[0]
    n_basis = n + 1

    # Determine expected shapes
    expected_normalized_shape = (num_pts, n_basis)
    expected_final_shape = _compute_final_output_shape_1D(input_shape, n_basis)

    if out is None:
        out = np.empty(expected_final_shape, dtype=t.dtype)
    else:
        _validate_out_array_1D(out, expected_final_shape, cast(npt.DTypeLike, t.dtype))

    B_normalized = out.reshape(expected_normalized_shape)

    core_func(np.int32(n), t, B_normalized)

    return out


def _tabulate_Bernstein_basis_1D_impl(
    n: int,
    t: npt.ArrayLike,
    out: npt.NDArray[np.float32 | np.float64] | None = None,
) -> npt.NDArray[np.float32 | np.float64]:
    """Evaluate the Bernstein basis polynomials of the given degree at the given points.

    Args:
        n (int): Degree of the Bernstein polynomials. Must be non-negative.
        t (npt.ArrayLike): Evaluation points. Can be a scalar, list, or numpy array.
            Types different from float32 or float64 are automatically converted to float64.
        out (npt.NDArray[np.float32 | np.float64] | None): Optional output array
            where the result will be stored. If None, a new array is allocated.
            Must have the correct shape and dtype if provided. Defaults to None.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Evaluated basis functions, with the same shape
        as the input points and the last dimension equal to (degree + 1).
        If `out` was provided, returns the same array.

    Raises:
        ValueError: If degree is negative, or if `out` is provided and has incorrect
            shape or dtype.

    Example:
        >>> _tabulate_Bernstein_basis_1D_impl(2, [0.0, 0.5, 0.75, 1.0])
        array([[1.    , 0.    , 0.    ],
               [0.25  , 0.5   , 0.25  ],
               [0.0625, 0.375 , 0.5625],
               [0.    , 0.    , 1.    ]])
    """
    return _tabulate_basis_1D_impl_helper(n, t, _tabulate_Bernstein_basis_1D_core, out)


def _tabulate_cardinal_Bspline_basis_1D_impl(
    n: int,
    t: npt.ArrayLike,
    out: npt.NDArray[np.float32 | np.float64] | None = None,
) -> npt.NDArray[np.float32 | np.float64]:
    r"""Evaluate the cardinal B-spline basis polynomials of given degree at given points.

    The cardinal B-spline basis is the set of B-spline basis functions defined
    on an interval of maximum continuity that has degree-1 contiguous
    knot spans on each side with the same length as the interval itself.
    These basis functions appear in the central knot spans
    in the case of maximum regularity uniform knot vectors.

    Explicit expression:
    \[
    B_{n,i}(t) = (1/n!) * sum_{j=0}^{n-i} binom(n+1, j) * (-1)^j * (t + n - i - j)^n
    \]
    where \( B_{n,i}(t) \) is the B-spline basis function of degree \( n \) and index \( i \)
     at point \( t \), and \( binom(a, b) \) is the binomial coefficient.

    Its actual implementation is based on the Cox-de Boor recursion formula for the
    central cardinal B-spline basis.

    Args:
        n (int): Degree of the B-spline basis. Must be non-negative.
        t (npt.ArrayLike): Evaluation points. Can be a scalar, list, or numpy array.
            Types different from float32 or float64 are automatically converted to float64.
        out (npt.NDArray[np.float32 | np.float64] | None): Optional output array
            where the result will be stored. If None, a new array is allocated.
            Must have the correct shape and dtype if provided. Defaults to None.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Evaluated basis functions, with the same shape
        as the input points and the last dimension equal to (degree + 1).
        If `out` was provided, returns the same array.

    Raises:
        ValueError: If provided degree is negative, or if `out` is provided and has incorrect
            shape or dtype.

    Example:
        >>> tabulate_cardinal_Bspline_basis_1D(2, [0.0, 0.5, 1.0])
        array([[0.5    , 0.5    , 0.     ],
               [0.125  , 0.75   , 0.125  ],
               [0.03125, 0.6875 , 0.28125],
               [0.     , 0.5    , 0.5    ]])

    """
    return _tabulate_basis_1D_impl_helper(n, t, _tabulate_cardinal_Bspline_basis_1D_core, out)


def _tabulate_Lagrange_basis_1D_impl(
    n: int,
    variant: LagrangeVariant,
    t: npt.ArrayLike,
    out: npt.NDArray[np.float32 | np.float64] | None = None,
) -> npt.NDArray[np.float32 | np.float64]:
    r"""Evaluate Lagrange basis polynomials at points using the specified variant.

    The polynomials are defined in the interval [0, 1] and are given by the formula:
    \[
    L_{n,i}(t) = \prod_{j=0}^{n} \frac{t - x_j}{x_i - x_j}
    \]
    where \( x_i \) are the points at which the basis is evaluated.

    The variant determines the points at which the basis is evaluated.
    central cardinal B-spline basis.

    Args:
        n (int): Degree of the B-spline basis. Must be non-negative.
        variant (LagrangeVariant): Variant of the Lagrange basis.
        t (npt.ArrayLike): Evaluation points. Can be a scalar, list, or numpy array.
            Types different from float32 or float64 are automatically converted to float64.
        out (npt.NDArray[np.float32 | np.float64] | None): Optional output array
            where the result will be stored. If None, a new array is allocated.
            Must have the correct shape and dtype if provided. Defaults to None.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Evaluated basis functions, with the same shape
        as the input points and the last dimension equal to (degree + 1).
        If `out` was provided, returns the same array.

    Raises:
        ValueError: If provided degree is negative, or if `out` is provided and has incorrect
            shape or dtype.
    """

    def core_func(
        n_int32: np.int32,
        t_array: npt.NDArray[np.float32 | np.float64],
        out_array: npt.NDArray[np.float32 | np.float64],
    ) -> None:
        return _tabulate_lagrange_basis_1D_core(int(n_int32), variant, t_array, out_array)

    return _tabulate_basis_1D_impl_helper(n, t, core_func, out)


def _tabulate_Legendre_basis_1D_impl(
    n: int,
    t: npt.ArrayLike,
    out: npt.NDArray[np.float32 | np.float64] | None = None,
) -> npt.NDArray[np.float32 | np.float64]:
    """Evaluate the normalized Shifted Legendre basis polynomials of the given degree.

    Args:
        n (int): Degree of the Legendre polynomials. Must be non-negative.
        t (npt.ArrayLike): Evaluation points. Can be a scalar, list, or numpy array.
        out (npt.NDArray[np.float32 | np.float64] | None): Optional output array
            where the result will be stored. If None, a new array is allocated.
            Must have the correct shape and dtype if provided. Defaults to None.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Evaluated basis functions.
        If `out` was provided, returns the same array.

    Raises:
        ValueError: If degree is negative, or if `out` is provided and has incorrect
            shape or dtype.
    """
    return _tabulate_basis_1D_impl_helper(n, t, _tabulate_Legendre_basis_1D_core, out)


def _compute_basis_combinator_matrix_for_points_lattice(
    evaluators_1D: tuple[Callable[[npt.ArrayLike], npt.NDArray[np.float32 | np.float64]]],
    pts: PointsLattice,
    order: Literal["C", "F"] = "C",
    out: npt.NDArray[np.float32 | np.float64] | None = None,
) -> npt.NDArray[np.float32 | np.float64]:
    """Combine 1D basis functions evaluated at a points lattice.

    This function efficiently evaluates a set of 1D basis functions at a tensor-product grid
    of points, exploiting the structured nature of the grid to minimize redundant computation.
    It handles both C-order (last index varies fastest) and F-order (first index varies fastest)
    meshgrid layouts, and correctly combines the results according to the specified function
    and point orderings.

    Args:
        evaluators_1D (tuple[Callable[[npt.ArrayLike], npt.NDArray[np.float32 | np.float64]]]):
            Tuple of 1D basis function evaluators.
            Each evaluator takes a 1D array of points and returns an array of basis function values.
        pts (PointsLattice): Points lattice.
        order (Literal["C", "F"]): Ordering of the basis functions and points:
            'F' for Fortran-order (first index varies fastest),
            'C' for C-order (last index varies fastest). Default is 'C'.
        out (npt.NDArray[np.float32 | np.float64] | None): Optional output array
            where the result will be stored. If None, a new array is allocated.
            Must have the correct shape and dtype if provided. This follows NumPy's
            style for output arrays. Defaults to None.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Array of shape (n_points, n_basis_functions)
        containing the combined basis function values. Both dimensions (points and basis functions)
        are ordered according to the specified order. If `out` was provided, returns the same array.

    Raises:
        ValueError: If the dimension of the points is less than 1,
        or the number of evaluators is not equal to the dimension of the points,
        or if `out` is provided and has incorrect shape or dtype.
    """
    dim = pts.dim
    if dim != len(evaluators_1D):
        raise ValueError("The number of evaluators must be equal to the dimension of the points.")

    pts_per_dir = pts.pts_per_dir

    # Compute the expected output shape and evaluate all dimensions
    # We'll reuse these evaluations in the computation below
    vals_per_dim: list[npt.NDArray[np.float32 | np.float64]] = []
    n_basis_per_dim: list[int] = []
    n_pts_per_dim: list[int] = []

    for dir in range(dim):
        vals_1D = evaluators_1D[dir](pts_per_dir[dir])
        vals_per_dim.append(vals_1D)
        n_basis_per_dim.append(vals_1D.shape[1])
        n_pts_per_dim.append(vals_1D.shape[0])

    n_points = int(np.prod(n_pts_per_dim))
    n_basis_functions = int(np.prod(n_basis_per_dim))
    expected_shape = _compute_output_shape_multidimensional(n_points, n_basis_functions)

    # Determine dtype from the points lattice
    expected_dtype = pts.dtype

    # Validate or allocate output array
    if out is None:
        out = np.empty(expected_shape, dtype=expected_dtype)
    else:
        _validate_out_array_multidimensional(out, expected_shape, expected_dtype)

    # Same ordering (C or F) is used for both points and functions.
    op_str = "pi,qj->qpji" if order == "F" else "pi,qj->pqij"

    # Handle the 1D case separately
    if dim == 1:
        out[:] = vals_per_dim[0]
    else:
        # Use the pre-computed evaluations
        current = vals_per_dim[0].copy()  # We need to modify this, so make a copy

        for dir in range(1, dim - 1):
            vals_1D = vals_per_dim[dir]
            n_rows = current.shape[0] * vals_1D.shape[0]
            n_cols = current.shape[1] * vals_1D.shape[1]
            # einsum produces (n_pts_0, n_pts_1, n_basis_0, n_basis_1)
            # or (n_pts_1, n_pts_0, n_basis_1, n_basis_0).
            # This is then reshaped to (n_pts_total, n_basis_total) according to the order.
            current = np.einsum(op_str, current, vals_1D).reshape(n_rows, n_cols)

        # Last iteration: write directly to out to avoid copy
        # Reshape out to the 4D shape that einsum expects
        vals_1D = vals_per_dim[dim - 1]
        final_4d_shape = (
            (current.shape[0], vals_1D.shape[0], current.shape[1], vals_1D.shape[1])
            if order == "C"
            else (vals_1D.shape[0], current.shape[0], vals_1D.shape[1], current.shape[1])
        )
        np.einsum(op_str, current, vals_1D, out=out.reshape(final_4d_shape))

    return out


def _compute_basis_combinator_matrix_for_points_array(
    evaluators_1D: tuple[Callable[[npt.ArrayLike], npt.NDArray[np.float32 | np.float64]]],
    pts: npt.ArrayLike,
    funcs_order: Literal["C", "F"] = "C",
    out: npt.NDArray[np.float32 | np.float64] | None = None,
) -> npt.NDArray[np.float32 | np.float64]:
    """Evaluate and combine 1D basis functions at a collection of points.

    Args:
        evaluators_1D (tuple[Callable[[npt.ArrayLike], npt.NDArray[np.float32 | np.float64]]]):
            Tuple of 1D basis function evaluators.
            Each evaluator takes a 1D array of points and returns an array of basis function values.
        pts (npt.ArrayLike): Points to evaluate the basis functions at.
        funcs_order (Literal["C", "F"]): Ordering of the basis functions: 'F' for Fortran-order
            (first index varies fastest) or 'C' for C-order (last index varies fastest).
        out (npt.NDArray[np.float32 | np.float64] | None): Optional output array
            where the result will be stored. If None, a new array is allocated.
            Must have the correct shape and dtype if provided. This follows NumPy's
            style for output arrays. Defaults to None.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Array of shape (n_points, n_basis_functions)
        containing the combined basis function values. If `out` was provided,
        returns the same array.

    Raises:
        ValueError: If the dimension of the points is not 2,
        the dtype of the points is not float32 or float64,
        or the number of evaluators is less than 1,
        or is not equal to the dimension of the points,
        or if `out` is provided and has incorrect shape or dtype.
    """
    if not isinstance(pts, np.ndarray):
        pts = np.array(pts)

    if pts.ndim != 2:  # noqa: PLR2004
        raise ValueError("Input points must be a 2D array.")

    if pts.dtype not in (np.float32, np.float64):
        pts = pts.astype(np.float64)

    dim = pts.shape[1]
    if dim < 1:
        raise ValueError("The dimension of the points must be at least 1.")
    if len(evaluators_1D) != dim:
        raise ValueError("The number of evaluators must be equal to the dimension of the points.")

    # Compute the expected output shape and evaluate all dimensions
    # We'll reuse these evaluations in the computation below
    n_pts = pts.shape[0]
    vals_per_dim: list[npt.NDArray[np.float32 | np.float64]] = []
    n_basis_per_dim: list[int] = []

    for dir in range(dim):
        vals_1D = evaluators_1D[dir](pts[:, dir])
        vals_per_dim.append(vals_1D)
        n_basis_per_dim.append(vals_1D.shape[1])

    n_basis_functions = int(np.prod(n_basis_per_dim))
    expected_shape = _compute_output_shape_multidimensional(n_pts, n_basis_functions)
    expected_dtype = pts.dtype

    # Validate or allocate output array
    if out is None:
        out = np.empty(expected_shape, dtype=expected_dtype)
    else:
        _validate_out_array_multidimensional(out, expected_shape, expected_dtype)

    # Handle the 1D case separately
    if dim == 1:
        out[:] = vals_per_dim[0]
    else:
        funcs_order_str = "ji" if funcs_order == "F" else "ij"
        op_str = f"pi,pj->p{funcs_order_str}"

        # Use the pre-computed evaluations
        current = vals_per_dim[0].copy()  # We need to modify this, so make a copy
        for dir in range(1, dim - 1):
            vals_1D = vals_per_dim[dir]
            current = np.einsum(op_str, current, vals_1D).reshape(n_pts, -1)

        # Last iteration: write directly to out to avoid copy
        # Reshape out to the 3D shape that einsum expects
        current_n_basis = current.shape[1]
        n_basis_1D = vals_per_dim[dim - 1].shape[1]
        final_shape = (
            (n_pts, current_n_basis, n_basis_1D)
            if funcs_order_str == "ij"
            else (n_pts, n_basis_1D, current_n_basis)
        )
        np.einsum(op_str, current, vals_per_dim[dim - 1], out=out.reshape(final_shape))

    return out


def _compute_basis_1D_combinator_matrix(
    evaluators_1D: tuple[Callable[[npt.ArrayLike], npt.NDArray[np.float32 | np.float64]]],
    pts: npt.ArrayLike | PointsLattice,
    order: Literal["C", "F"] = "C",
    out: npt.NDArray[np.float32 | np.float64] | None = None,
) -> npt.NDArray[np.float32 | np.float64]:
    """Combine 1D basis function evaluations into a multidimensional tensor-product basis.

    Evaluates and combines a tuple of 1D basis function evaluators at the given points,
    supporting both general arrays
    of points and tensor-product grids (points lattices).
    Enforces the specified ordering (C or F) for the basis functions
    (and points in the case of a points lattice).

    Args:
        evaluators_1D (tuple[Callable[[npt.ArrayLike], npt.NDArray[np.float32 | np.float64]]]):
            Tuple of 1D basis function evaluators.
            Each evaluator takes a 1D array of points and returns an array of basis function values.
        pts (npt.ArrayLike | PointsLattice): Points to evaluate the basis functions at.
            It may be a general 2D numpy array of points (where each row is a point)
            or a points lattice.
        order (str): Ordering of the basis functions (and points in the case of a points lattice):
            'F' for Fortran-order (first index varies fastest),
            'C' for C-order (last index varies fastest). Defaults to 'C'.
        out (npt.NDArray[np.float32 | np.float64] | None): Optional output array
            where the result will be stored. If None, a new array is allocated.
            Must have the correct shape and dtype if provided. This follows NumPy's
            style for output arrays. Defaults to None.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Array of shape (n_points, n_basis_functions)
        containing the combined basis function values.
        Functions are ordered according to the specified order.
        In the case of a points lattice, points are also ordered according to the specified order.
        If `out` was provided, returns the same array.

    Raises:
        ValueError: If the dimension of the points is not 2,
        the dtype of the points is not float32 or float64,
        or the number of evaluators is less than 1,
        or is not equal to the dimension of the points,
        or if `out` is provided and has incorrect shape or dtype.
    """
    if isinstance(pts, PointsLattice):
        return _compute_basis_combinator_matrix_for_points_lattice(evaluators_1D, pts, order, out)
    else:
        return _compute_basis_combinator_matrix_for_points_array(evaluators_1D, pts, order, out)


def _warmup_numba_functions() -> None:
    """Precompile numba functions with float64 signatures for faster first call.

    This function triggers compilation of the numba-decorated core functions
    with float64 arrays, ensuring they are cached and ready for use.
    """
    # Small dummy arrays for warmup
    t_dummy = np.array([0.0, 0.5, 1.0], dtype=np.float64)
    out_dummy = np.empty((3, 2), dtype=np.float64)

    # Warmup each core function with float64
    _tabulate_Bernstein_basis_1D_core(np.int32(1), t_dummy, out_dummy)
    _tabulate_cardinal_Bspline_basis_1D_core(np.int32(1), t_dummy, out_dummy)
    _tabulate_Legendre_basis_1D_core(np.int32(1), t_dummy, out_dummy)


# Precompile numba functions on module import (skip during type checking)
if not TYPE_CHECKING:
    _warmup_numba_functions()
