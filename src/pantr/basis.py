"""Basis function evaluation for various polynomial bases."""

from enum import Enum

import numpy as np
import numpy.typing as npt

from ._basis_impl import (
    _eval_Bernstein_basis_1D_impl,
    _eval_cardinal_Bspline_basis_1D_impl,
    _eval_Lagrange_basis_1D_impl,
)


class LagrangeVariant(Enum):
    """Enumeration for Lagrange polynomial variants.

    Attributes:
        EQUISPACES (LagrangeVariant): Equispaced points.
        GAUSS_LEGENDRE (LagrangeVariant): Gauss-Legendre points (roots of Legendre polynomial).
        GAUSS_LOBATTO_LEGENDRE (LagrangeVariant): Gauss-Lobatto-Legendre points.
        CHEBYSHEV_1ST (LagrangeVariant): Chebyshev 1st kind points
            (x = [pi*(k + 0.5)/npts for k in range(npts)]).
        CHEBYSHEV_2ND (LagrangeVariant): Chebyshev 2nd kind points
            (x = [pi*k/(npts - 1) for k in range(npts)]).
    """

    EQUISPACES = "equispaces"
    GAUSS_LEGENDRE = "gauss_legendre"
    GAUSS_LOBATTO_LEGENDRE = "gauss_lobatto_legendre"
    CHEBYSHEV_1ST = "chebyshev_1st"
    CHEBYSHEV_2ND = "chebyshev_2nd"


def eval_Bernstein_basis_1D(
    degree: int, pts: npt.ArrayLike
) -> npt.NDArray[np.float32 | np.float64]:
    """Evaluate the Bernstein basis polynomials of the given degree at the given points.

    Args:
        degree (int): Degree of the Bernstein polynomials. Must be non-negative.
        pts (npt.ArrayLike): Evaluation points. Can be a scalar, list, or
            numpy array. Types different from float32 or float64 are
            automatically converted to float64.

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


def eval_cardinal_Bspline_basis_1D(
    degree: int, pts: npt.ArrayLike
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
        degree (int): Degree of the B-spline basis. Must be non-negative.
        pts (npt.ArrayLike): Evaluation points. Can be a scalar, list, or numpy array.
            Types different from float32 or float64 are automatically converted to float64.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Evaluated basis functions, with the same shape
        as the input points and the last dimension equal to (degree + 1).

    Raises:
        ValueError: If provided degree is negative.

    Example:
        >>> eval_cardinal_Bspline_basis_1D(2, [0.0, 0.5, 1.0])
        array([[0.5    , 0.5    , 0.     ],
               [0.125  , 0.75   , 0.125  ],
               [0.03125, 0.6875 , 0.28125],
               [0.     , 0.5    , 0.5    ]])

    """
    return _eval_cardinal_Bspline_basis_1D_impl(degree, pts)


def eval_Lagrange_basis_1D(
    degree: int, variant: LagrangeVariant, pts: npt.ArrayLike
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
        degree (int): Degree of the B-spline basis. Must be non-negative.
        variant (LagrangeVariant): Variant of the Lagrange basis.
        pts (npt.ArrayLike): Evaluation points. Can be a scalar, list, or numpy array.
            Types different from float32 or float64 are automatically converted to float64.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Evaluated basis functions, with the same shape
        as the input points and the last dimension equal to (degree + 1).

    Raises:
        ValueError: If provided degree is negative.
    """
    return _eval_Lagrange_basis_1D_impl(degree, variant, pts)
