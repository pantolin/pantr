"""Numba-backed core implementations for 1D Bernstein and cardinal B-spline bases."""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar, cast

import numba as nb  # type: ignore[import-untyped]
import numpy as np
import numpy.typing as npt

from ._basis_utils import _normalize_basis_output_1D, _normalize_points_1D

F = TypeVar("F", bound=Callable[..., Any])

if TYPE_CHECKING:
    # During type-checking, make the decorator a no-op that preserves types.
    def nb_jit(*args: object, **kwargs: object) -> Callable[[F], F]:
        def decorator(func: F) -> F:
            return func

        return decorator
else:
    # At runtime, use the real Numba decorator.
    nb_jit = nb.jit  # type: ignore[attr-defined]


@nb_jit(
    [
        nb.float32[:, :](nb.int32, nb.float32[:]),
        nb.float64[:, :](nb.int32, nb.float64[:]),
    ],
    nopython=True,
    cache=True,
    parallel=False,
)
def _eval_Bernstein_basis_1D_core(
    n: np.int32, t: npt.NDArray[np.float32] | npt.NDArray[np.float64]
) -> npt.NDArray[np.float32 | np.float64]:
    """Evaluate Bernstein basis polynomials of degree n at points t.

    Computes all (n+1) Bernstein basis polynomials B_i,n(t) for i=0,...,n
    at each evaluation point in t. The result is a 2D array where B[j, i]
    contains the value of the i-th basis polynomial evaluated at point t_j.

    The implementation uses a recurrence relation for efficient computation:
    - Base case: B_0,n(t) = (1-t)^n
    - Recurrence: B_i,n(t) = B_{i-1},n(t) * ((n-i+1)/i) * (t/(1-t))

    Special handling is applied for points where t=1.0, where the recurrence
    relation would produce NaN values. At t=1.0, only the last basis function
    B_n,n(1) = 1, while all others are 0.

    Args:
        n (np.int32): Degree of the Bernstein polynomials. Must be non-negative.
        t (npt.NDArray[np.float32] | npt.NDArray[np.float64]): 1D array of
            evaluation points. Must be a contiguous array of float32 or float64
            values. Points should typically be in [0, 1], though the function
            will compute values for any real t.

    Returns:
        npt.NDArray[np.float32] | npt.NDArray[np.float64]: 2D array of shape
        (len(t), n+1) containing the evaluated basis functions. The dtype
        matches the dtype of t (float32 or float64). Element B[j, i] contains
        B_i,n(t_j).

    Note:
        This is a Numba-compiled function optimized for performance. It
        expects pre-normalized inputs (1D contiguous arrays). For general
        use, call _eval_Bernstein_basis_1D_impl instead.
    """
    if n == 0:
        # The basis is just B_0,0(pts) = 1
        ones: npt.NDArray[np.float32 | np.float64] = np.ones((t.shape[0], 1), dtype=t.dtype)
        return ones

    # Initialize the output array
    # B[j, i] will hold B_i,n(t_j)
    B: npt.NDArray[np.float32 | np.float64] = np.zeros((t.shape[0], n + 1), dtype=t.dtype)

    # 1. Handle points where t is not 1.0
    idx_t_ne_1 = np.where(t != 1.0)[0]
    if idx_t_ne_1.size > 0:
        t_ne_1: npt.NDArray[np.float32 | np.float64] = t[idx_t_ne_1]
        B_ne_1: npt.NDArray[np.float32 | np.float64] = np.zeros(
            (t_ne_1.shape[0], n + 1), dtype=t.dtype
        )
        B_ne_1[:, 0] = np.power(1.0 - t_ne_1, n)
        t_over_1mt: npt.NDArray[np.float32 | np.float64] = t_ne_1 / (1.0 - t_ne_1)
        for i in range(1, n + 1):
            const_factor: npt.NDArray[np.float32 | np.float64] = np.full(
                t_ne_1.shape, (n - i + 1) / i, dtype=t.dtype
            )
            B_ne_1[:, i] = B_ne_1[:, i - 1] * const_factor * t_over_1mt
        B[idx_t_ne_1] = B_ne_1

    # 2. Handle points where t is 1.0
    idx_t_eq_1 = np.where(t == 1.0)[0]
    if idx_t_eq_1.size > 0:
        B[idx_t_eq_1, :] = 0.0
        B[idx_t_eq_1, -1] = 1.0

    return B


def _eval_Bernstein_basis_1D_impl(n: int, t: npt.ArrayLike) -> npt.NDArray[np.float32 | np.float64]:
    """Evaluate the Bernstein basis polynomials of the given degree at the given points.

    Args:
        n (int): Degree of the Bernstein polynomials. Must be non-negative.
        t (npt.ArrayLike): Evaluation points. Can be a scalar, list, or numpy array.
            Types different from float32 or float64 are automatically converted to float64.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Evaluated basis functions, with the same shape
        as the input points and the last dimension equal to (degree + 1).

    Raises:
        ValueError: If degree is negative.

    Example:
        >>> _eval_Bernstein_basis_1D_impl(2, [0.0, 0.5, 0.75, 1.0])
        array([[1.    , 0.    , 0.    ],
               [0.25  , 0.5   , 0.25  ],
               [0.0625, 0.375 , 0.5625],
               [0.    , 0.    , 1.    ]])
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

    # Narrow union dtype for mypy by branching on dtype and casting accordingly.
    if t.dtype == np.float32:
        B = _eval_Bernstein_basis_1D_core(np.int32(n), cast(npt.NDArray[np.float32], t))
    else:
        B = _eval_Bernstein_basis_1D_core(np.int32(n), cast(npt.NDArray[np.float64], t))

    return _normalize_basis_output_1D(B, input_shape)


@nb_jit(
    [
        nb.float32[:, :](nb.int32, nb.float32[:]),
        nb.float64[:, :](nb.int32, nb.float64[:]),
    ],
    nopython=True,
    cache=True,
    parallel=False,
)
def _eval_cardinal_Bspline_basis_1D_core(
    n: np.int32, t: npt.NDArray[np.float32] | npt.NDArray[np.float64]
) -> npt.NDArray[np.float32 | np.float64]:
    """Evaluate the central cardinal B-spline basis of degree n on [0, 1].

    Computes the (n+1) nonzero B-spline basis functions active over the central
    unit span [0, 1] of a uniform knot vector with unit spacing, at each
    evaluation point in t. Values are zero outside [0, 1]. Uses the stable
    Cox-de Boor (BasisFuns) recursion specialized to span index i=0.

    Args:
        n: Degree of the B-spline basis (>= 0).
        t: 1D array of evaluation points (float32 or float64).

    Returns:
        2D array of shape (len(t), n+1) with evaluated basis values.
    """
    num_pts = t.shape[0]
    out: npt.NDArray[np.float32 | np.float64] = np.zeros((num_pts, n + 1), dtype=t.dtype)

    if n == 0:
        # Degree-0: characteristic function of [0, 1] (include both endpoints).
        for j in range(num_pts):
            u = t[j]
            if 0.0 <= u <= 1.0:
                out[j, 0] = 1.0
        return out

    # Temporary arrays reused per point
    left = np.empty(n + 1, dtype=t.dtype)
    right = np.empty(n + 1, dtype=t.dtype)
    N = np.empty(n + 1, dtype=t.dtype)

    for j in range(num_pts):
        u = t[j]
        if not (0.0 <= u <= 1.0):
            # Outside the central span: remains zeros
            continue

        # BasisFuns for uniform knots U_k = k, degree n, span index i = 0.
        N[0] = 1.0
        for k in range(1, n + 1):
            # For i = 0 and uniform unit knots:
            # left[k]  = u - U[i+1-k] = u - (1 - k) = u + k - 1
            # right[k] = U[i+k] - u   = k - u
            left[k] = u + (k - 1.0)
            right[k] = (k - 0.0) - u

            saved = 0.0
            for r in range(k):
                denom = right[r + 1] + left[k - r]
                temp = 0.0
                if denom != 0.0:
                    temp = N[r] / denom
                N[r] = saved + right[r + 1] * temp
                saved = left[k - r] * temp
            N[k] = saved

        # N[0..n] directly correspond to i = 0..n on the central span
        for i in range(n + 1):
            out[j, i] = N[i]

    return out


def _eval_cardinal_Bspline_basis_1D_impl(
    n: int, t: npt.ArrayLike
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

    Args:
        n (int): Degree of the B-spline basis. Must be non-negative.
        t (npt.ArrayLike): Evaluation points. Can be a scalar, list, or numpy array.
            Types different from float32 or float64 are automatically converted to float64.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Evaluated basis functions, with the same shape
        as the input points and the last dimension equal to (degree + 1).

    Raises:
        ValueError: If provided degree is negative.

    Example:
        >>> evaluate_cardinal_Bspline_basis(2, [0.0, 0.5, 1.0])
        array([[0.5    , 0.5    , 0.     ],
               [0.125  , 0.75   , 0.125  ],
               [0.03125, 0.6875 , 0.28125],
               [0.     , 0.5    , 0.5    ]])

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

    # Narrow union dtype for mypy by branching on dtype and casting accordingly.
    if t.dtype == np.float32:
        B = _eval_cardinal_Bspline_basis_1D_core(np.int32(n), cast(npt.NDArray[np.float32], t))
    else:
        B = _eval_cardinal_Bspline_basis_1D_core(np.int32(n), cast(npt.NDArray[np.float64], t))

    return _normalize_basis_output_1D(B, input_shape)
