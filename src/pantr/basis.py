"""Basis function evaluation for various polynomial bases.

This module provides functions to evaluate different types of basis functions
including Bernstein, cardinal B-spline, and Lagrange.
"""

from numbers import Integral

import numpy as np
import numpy.typing as npt

from ._basis_impl import _eval_Bernstein_basis_1D_impl


def eval_Bernstein_basis_1D(
    degree: Integral, pts: npt.ArrayLike
) -> npt.NDArray[np.float32 | np.float64]:
    """Evaluate the Bernstein basis polynomials of the given degree at the given points.

    Args:
        degree (Integral): Degree of the Bernstein polynomials. Must be non-negative.
        pts (npt.ArrayLike): Evaluation points. Evaluation points. Can be a scalar, list, or numpy array.
            Types different from float32 or float64 are automatically converted to float64.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Evaluated basis functions, with the same shape as
        the input points and the last dimension equal to (degree + 1).

    Raises:
        ValueError: If degree is negative.

    Example:
        >>> eval_Bernstein_basis_1D(2, [0.0, 0.5, 0.75, 1.0])
        array([[1.    , 0.    , 0.    ],
               [0.25  , 0.5   , 0.25  ],
               [0.0625, 0.375 , 0.5625],
               [0.    , 0.    , 1.    ]])
    """
    return _eval_Bernstein_basis_1D_impl(degree, pts)
