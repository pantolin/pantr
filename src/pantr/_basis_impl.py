from typing import Any, cast

import numba as nb  # type: ignore[import-untyped]
import numpy as np
import numpy.typing as npt

from ._basis_utils import _normalize_basis_output_1D, _normalize_points_1D

nb_jit = cast(Any, nb.jit)


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
        return np.ones((t.shape[0], 1), dtype=t.dtype)

    # Initialize the output array
    # B[j, i] will hold B_i,n(t_j)
    B = np.zeros((t.shape[0], n + 1), dtype=t.dtype)

    # 1. Handle points where t is not 1.0
    idx_t_ne_1 = np.where(t != 1.0)[0]
    if idx_t_ne_1.size > 0:
        t_ne_1 = t[idx_t_ne_1]
        B_ne_1 = np.zeros((t_ne_1.shape[0], n + 1), dtype=t.dtype)
        B_ne_1[:, 0] = np.power(1.0 - t_ne_1, n)
        t_over_1mt = t_ne_1 / (1.0 - t_ne_1)
        for i in range(1, n + 1):
            B_ne_1[:, i] = B_ne_1[:, i - 1] * ((n - i + 1) / i) * t_over_1mt
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

    B = _eval_Bernstein_basis_1D_core(np.int32(n), t)

    # 5. Return
    return _normalize_basis_output_1D(B, input_shape)
