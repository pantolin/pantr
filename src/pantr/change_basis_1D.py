"""Change of basis operators for various polynomial bases in 1D.

# This module provides functions to create transformation matrices between different
# polynomial bases including Lagrange, Bernstein, cardinal B-spline, and monomial
# bases, as well as B-splines.
#
"""

import numpy as np
import numpy.typing as npt

from ._basis_impl import _get_lagrange_points
from .basis import LagrangeVariant, eval_Bernstein_basis_1D

#     from .bspline_1D import Bspline1D


# def _compute_gauss_quadrature(
#     n_points: np.int_,
# ) -> tuple[FloatArray_64, FloatArray_64]:
#     """Compute Gauss-Legendre quadrature points and weights for the [0, 1] interval.

#     Args:
#         n_points (np.int_): The number of quadrature points to generate.
#             Must be positive.

#     Returns:
#         tuple[FloatArray_64, FloatArray_64]: Tuple containing:
#             - points (FloatArray_64): The quadrature points in the [0, 1] interval.
#             - weights (FloatArray_64): The corresponding quadrature weights.

#     Raises:
#         AssertionError: If number of points is not positive.
#     """
#     assert n_points >= 1, "Number of points must be positive"

#     # Get standard Gauss-Legendre points and weights on [-1, 1]
#     points_ref, weights_ref = np.polynomial.legendre.leggauss(n_points)

#     # Affine transformation to map points from [-1, 1] to [0, 1]
#     # t = (x + 1) / 2
#     points = np.float64(0.5) * (points_ref + np.float64(1.0))

#     # Adjust weights for the change of interval: dt = dx / 2
#     weights = np.float64(0.5) * weights_ref

#     return points, weights


def create_Lagrange_to_Bernstein_basis_operator(
    degree: int,
    lagrange_variant: LagrangeVariant = LagrangeVariant.EQUISPACES,
    dtype: npt.DTypeLike = np.float64,
) -> npt.NDArray[np.float32 | np.float64]:
    """Construct the matrix mapping Lagrange basis evaluations to Bernstein basis evaluations.

    Note:
        Both Bernstein and Lagrange bases follow the standard ordering (see https://en.wikipedia.org/wiki/Bernstein_polynomial).

    Args:
        degree (int): Polynomial degree. Must be at least 1.
        lagrange_variant (LagrangeVariant): Lagrange point distribution
            (e.g., equispaced, gauss lobatto legendre, etc). Defaults to LagrangeVariant.EQUISPACES.
        dtype (npt.DTypeLike): Floating point type for the output matrix.
            Defaults to np.float64.

    Returns:
        npt.NDArray[np.float32 | np.float64]: (degree+1, degree+1) transformation matrix C such that
            C @ [Lagrange values] = [Bernstein values].

    Raises:
        ValueError: If degree is lower than 1.
    """
    if degree < 1:
        raise ValueError("Degree must at least 1")

    points = _get_lagrange_points(lagrange_variant, degree + 1, dtype)

    return eval_Bernstein_basis_1D(degree, points).T


def create_Bernstein_to_Lagrange_basis_operator(
    degree: int,
    lagrange_variant: LagrangeVariant = LagrangeVariant.EQUISPACES,
    dtype: npt.DTypeLike = np.float64,
) -> npt.NDArray[np.float32 | np.float64]:
    """Construct the matrix mapping Bernstein basis evaluations to Lagrange basis evaluations.

    Note:
        Both Bernstein and Lagrange bases follow the standard ordering
        (see https://en.wikipedia.org/wiki/Bernstein_polynomial).


    Args:
        degree (int): Polynomial degree. Must be at least 1.
        lagrange_variant (LagrangeVariant): Lagrange point distribution
            (e.g., equispaced, gauss lobatto legendre, etc). Defaults to LagrangeVariant.EQUISPACES.
        dtype (npt.DTypeLike): Floating point type for the output matrix.
            Defaults to np.float64.

    Returns:
        npt.NDArray[np.float32 | np.float64]: (degree+1, degree+1) transformation matrix C such that
            C @ [Bernstein values] = [Lagrange values].

    Raises:
        ValueError: If degree is lower than 1.
    """
    if degree < 1:
        raise ValueError("Degree must at least 1")

    C = create_Lagrange_to_Bernstein_basis_operator(degree, lagrange_variant, dtype)
    return np.linalg.inv(C)


# def _create_change_basis_operator(
#     new_basis_eval: callable,
#     old_basis_eval: callable,
#     n_quad_pts: int,
#     dtype: np.dtype = np.float64,
# ) -> npt.NDArray[np.floating]:
#     """Create a change of basis operator using numerical quadrature.

#     This function computes the transformation matrix M that satisfies:
#         new_basis(x) = M @ old_basis(x)

#     The matrix is computed by solving the system C = G M^T where:
#     - G is the Gram matrix of the new basis
#     - C is the mixed inner product matrix between new and old bases

#     Args:
#         new_basis_eval (callable): Function that evaluates the new basis at points.
#         old_basis_eval (callable): Function that evaluates the old basis at points.
#         n_quad_pts (int): Number of quadrature points for numerical integration.
#             Must be positive.
#         dtype (np.dtype): Floating point type for the output matrix.
#             Defaults to np.float64.

#     Returns:
#         npt.NDArray[np.floating]: Change of basis transformation matrix.

#     Raises:
#         ValueError: If number of quadrature points is not positive.
#     """
#     if n_quad_pts <= 0:
#         raise ValueError("Number of quadrature points must be positive.")

#     # 1. Get Gauss quadrature points and weights for the inner product on [0, 1]
#     points, weights = _compute_gauss_quadrature(n_quad_pts)
#     points, weights = points.astype(dtype), weights.astype(dtype)

#     # 2. Pre-evaluate all basis functions at all quadrature points for efficiency
#     new_basis = new_basis_eval(points)
#     old_basis = old_basis_eval(points)

#     # 3. Compute the Gram matrix G for the new basis B: G_kj = <b_k, b_j>
#     # The inner product <f, g> is approximated by sum(w_m * f(x_m) * g(x_m))
#     weights_diag = np.diag(weights)
#     G = new_basis.T @ weights_diag @ new_basis

#     # 4. Compute the mixed inner product matrix C: C_ki = <b_k, a_i>
#     C = new_basis.T @ weights_diag @ old_basis

#     # 5. Solve the system C = G M^T for M^T, which means M = (G^-1 C)^T
#     return np.linalg.solve(G, C).T


# def create_Bernstein_to_cardinal_basis_operator(
#     degree: int,
#     dtype: np.dtype = np.float64,
# ) -> npt.NDArray[np.floating]:
#     """Create transformation matrix from Bernstein to cardinal B-spline basis.

#     Args:
#         degree (int): Polynomial degree. Must be non-negative.
#         dtype (np.dtype): Floating point type for the output matrix.
#             Defaults to np.float64.

#     Returns:
#         npt.NDArray[np.floating]: (degree+1, degree+1) transformation matrix C such that
#             C @ [Bernstein values] = [cardinal values].

#     Raises:
#         ValueError: If degree is negative.
#     """
#     if degree < 0:
#         raise ValueError("Degree must be non-negative")

#     def bernstein(pts):
#         return evaluate_Bernstein_basis_1D(degree, pts)

#     def cardinal(pts):
#         return evaluate_cardinal_Bspline_basis_1D(degree, pts)

#     return _create_change_basis_operator(
#         new_basis_eval=bernstein,
#         old_basis_eval=cardinal,
#         n_quad_pts=degree + 1,
#         dtype=dtype,
#     )


# def create_cardinal_to_Bernstein_basis_operator(
#     degree: int,
#     dtype: np.dtype = np.float64,
# ) -> npt.NDArray[np.floating]:
#     """Create transformation matrix from cardinal B-spline to Bernstein basis.

#     Args:
#         degree (int): Polynomial degree. Must be non-negative.
#         dtype (np.dtype): Floating point type for the output matrix.
#             Defaults to np.float64.

#     Returns:
#         npt.NDArray[np.floating]: (degree+1, degree+1) transformation matrix C such that
#             C @ [cardinal values] = [Bernstein values].

#     Raises:
#         ValueError: If degree is negative.
#     """
#     if degree < 0:
#         raise ValueError("Degree must be non-negative")

#     def cardinal(pts):
#         return evaluate_cardinal_Bspline_basis_1D(degree, pts)

#     def bernstein(pts):
#         return evaluate_Bernstein_basis_1D(degree, pts)

#     return _create_change_basis_operator(
#         new_basis_eval=cardinal,
#         old_basis_eval=bernstein,
#         n_quad_pts=degree + 1,
#         dtype=dtype,
#     )


# def create_Bezier_extraction_operators(spline: "Bspline1D") -> FloatArray_32_64:
#     """Create Bézier extraction operators for a B-spline.

#     Args:
#         spline (Bspline1D): B-spline object.

#     Returns:
#         FloatArray_32_64: Array of extraction matrices with shape
#             (n_intervals, degree+1, degree+1) where each matrix transforms
#             Bernstein basis functions to B-spline basis functions for that interval.

#             Each matrix C[i, :, :] transforms Bernstein basis functions
#             to B-spline basis functions for the i-th interval as
#                 C[i, :, :] @ [Bernstein values] = [B-spline values in interval].
#     """
#     return create_bspline_Bezier_extraction_operators_impl(
#         spline.knots, spline.degree, spline.tolerance
#     )


# def create_Lagrange_extraction_operators(
#     spline: "Bspline1D",
#     lagrange_variant: LagrangeVariant = LagrangeVariant.equispaced,
# ) -> FloatArray_32_64:
#     """Create Lagrange extraction operators for a B-spline.

#     Args:
#         spline (Bspline1D): B-spline object.
#         lagrange_variant (LagrangeVariant): Lagrange point distribution
#             (e.g., equispaced, GLL, etc). Defaults to LagrangeVariant.equispaced.

#     Returns:
#         FloatArray_32_64: Array of extraction matrices with shape
#             (n_intervals, degree+1, degree+1) where each matrix transforms
#             Lagrange basis functions to B-spline basis functions for that interval.

#             Each matrix C[i, :, :] transforms Bernstein basis functions
#             to B-spline basis functions for the i-th interval as
#                 C[i, :, :] @ [Lagrange values] = [B-spline values in interval].
#     """
#     if spline.degree < 1:
#         raise ValueError("Degree must at least 1")

#     C = create_Bezier_extraction_operators(spline)
#     lagr_to_bzr = create_Lagrange_to_Bernstein_basis_operator(
#         spline.degree, lagrange_variant, spline.dtype
#     )
#     C[:] = C @ lagr_to_bzr
#     return C


# def create_cardinal_extraction_operators(spline: "Bspline1D") -> FloatArray_32_64:
#     """Create cardinal B-spline extraction operators.

#     For cardinal intervals, the extraction matrix is set to the identity matrix

#     Args:
#         spline (Bspline1D): B-spline object.

#     Returns:
#         FloatArray_32_64: Array of extraction matrices with shape
#             (n_intervals, degree+1, degree+1) where each matrix transforms
#             cardinal B-spline basis functions to B-spline basis functions for that interval.

#             Each matrix C[i, :, :] transforms cardinal B-spline basis functions
#             to B-spline basis functions for the i-th interval as
#                 C[i, :, :] @ [cardinal values] = [B-spline values in interval].
#     """
#     C = create_Bezier_extraction_operators(spline)
#     card_to_bzr = create_cardinal_to_Bernstein_basis_operator(spline.degree, spline.dtype)
#     C[:] = C @ card_to_bzr

#     for i in np.where(spline.get_cardinal_intervals())[0]:
#         C[i, :, :] = np.eye(spline.degree + 1, dtype=spline.dtype)
#     return C
