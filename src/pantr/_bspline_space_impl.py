"""Implementation functions for multi-dimensionalB-spline space operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from ._bspline_space_1D_impl import _is_in_domain_impl
from .quad import PointsLattice

if TYPE_CHECKING:
    from .bspline_space import BsplineSpace


def _tabulate_Bspline_basis_for_points_array_impl(
    spline: BsplineSpace,
    pts: npt.NDArray[np.float32 | np.float64],
) -> tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.int_]]:
    """Evaluate B-spline basis functions at given points.

    Args:
        spline (BsplineSpace): B-spline object defining the basis.
        pts (npt.NDArray[np.float32 | np.float64]): Evaluation points.
            Must be a 2D array with shape (num_pts, dim).

    Returns:
        tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.int_]]: Tuple containing:
            - basis_values: (npt.NDArray[np.float32 | np.float64])
              Array of shape (num_pts, order[0], order[1], ..., order[d-1])
              containing the basis function values evaluated at each point.
            - first_basis_indices: (npt.NDArray[np.int_])
              2D integer array indicating the index of the first nonzero basis function
              for each evaluation point in each direction. The shape is (num_pts, dim).

    Raises:
        ValueError: If pts is not a 2D array or does not have the correct number of columns.
        ValueError: If one or more values in pts are outside the knot vector domain.
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

    order = np.array(spline.degrees) + 1
    num_pts = pts.shape[0]

    # Combine 1D basis along each direction using outer product to form the
    # tensor-product multidimensional basis.
    # The multidimensional basis Bs will have shape (num_pts, order[0], order[1], ..., order[d-1])

    # Start with the basis functions of the first direction
    pts_0 = np.ascontiguousarray(pts[:, 0])
    B_multi, first_idx_0 = splines_1D[0].tabulate_basis(pts_0)
    first_indices_list = [first_idx_0]
    for dir in range(1, spline.dim):
        pts_dir = np.ascontiguousarray(pts[:, dir])
        Bdir, first_idx_dir = splines_1D[dir].tabulate_basis(pts_dir)
        first_indices_list.append(first_idx_dir)
        # At each step, expand B_multi to add a new axis at the end,
        # and outer product with the next B_1D
        B_multi = np.multiply(
            B_multi[..., np.newaxis],  # shape (..., n1, 1)
            Bdir[:, np.newaxis, ...],  # shape (num_pts, 1, n2) (for d=1)
        )
        # merge leading axes for all points
        shape = (num_pts, *tuple(o for o in order[: dir + 1]))
        B_multi = B_multi.reshape(shape)

    # Stack first indices from all directions into shape (num_pts, dim)
    first_indices_1D = np.stack(first_indices_list, axis=1)

    return B_multi, first_indices_1D


def _tabulate_Bspline_basis_for_points_lattice_impl(
    spline: BsplineSpace,
    pts: PointsLattice,
) -> tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.int_]]:
    """Evaluate B-spline basis functions at points in a lattice structure.

    This function computes the tensor product of the 1D B-spline basis
    functions evaluated over the lattice points defined by 'pts'. The output
    arrays capture the non-zero local basis values and their corresponding
    global starting indices efficiently for the full grid.

    Args:
        spline (BsplineSpace): B-spline object defining the basis (knots, degree, etc.).
        pts (PointsLattice): Evaluation points defined as a tensor product lattice.

    Returns:
        tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.int_]]: Tuple containing:
            - basis_values: (npt.NDArray[np.float32 | np.float64])
              Array of shape (n_pts_0, n_pts_1, ..., n_pts_d, k_0, k_1, ..., k_d)
              where 'n_pts_i' is the number of points in dimension $i$, and $k_i$
              is the number of local non-zero basis functions (typically degree + 1).
              This array contains the tensor product of basis function values evaluated
              at each grid point.
            - first_basis_indices: (npt.NDArray[np.int_])
              Array of shape (n_pts_0, n_pts_1, ..., n_pts_d, dim) indicating the global
              index of the first nonzero basis function for each evaluation point in each direction.

    Raises:
        ValueError: If pts dimension does not match spline dimension.
        ValueError: If one or more values in pts are outside the knot vector domain.
        ValueError: If one or more values in pts are outside the corresponding knot vector domain.
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

    # Initialize with first array
    first_B = Bs[0]
    new_shape = [1] * (2 * ndim)
    new_shape[0] = first_B.shape[0]
    new_shape[ndim] = first_B.shape[1]
    B_multi: npt.NDArray[np.float32 | np.float64] = first_B.reshape(new_shape)

    for i, B in enumerate(Bs[1:], start=1):
        # Shape B_multi: (n_pts_0, ..., n_pts_d, k_0, ..., k_d)
        new_shape = [1] * (2 * ndim)
        new_shape[i] = B.shape[0]
        new_shape[ndim + i] = B.shape[1]

        B_multi = B_multi * B.reshape(new_shape)

    # 3. Combine first_indices into a multi-dimensional array (Meshgrid approach)
    # Result shape: (n_pts_0, n_pts_1, ..., n_pts_d, dim)
    first_indices: npt.NDArray[np.int_] = np.stack(np.meshgrid(*first_idxs, indexing="ij"), axis=-1)

    return B_multi, first_indices


def _tabulate_Bspline_basis_impl(
    spline: BsplineSpace,
    pts: npt.NDArray[np.float32 | np.float64] | PointsLattice,
) -> tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.int_]]:
    if isinstance(pts, PointsLattice):
        return _tabulate_Bspline_basis_for_points_lattice_impl(spline, pts)
    else:
        return _tabulate_Bspline_basis_for_points_array_impl(spline, pts)
