"""Utility functions for basis function evaluation."""

import numpy as np
from numpy import typing as npt


def _normalize_points_1D(pts: npt.ArrayLike) -> npt.NDArray[np.float32 | np.float64]:
    """Normalize points to a 1D float array for basis function evaluation.

    Converts input points (scalar, list, or numpy array) to a 1D numpy array
    with floating point dtype. Types different from float32 or float64 are automatically converted to float64.
    Zero-dimensional arrays (scalars) are converted to 1D arrays with a single
    element. Multi-dimensional arrays will be flattened to 1D,

    Returns:
        A 1D numpy array with floating point dtype (np.float32 or np.float64).
        The dtype is preserved from the input if it's already a floating point
        type, otherwise converted to np.float64. The array is guaranteed to have
        exactly one dimension (ndim == 1).
    """
    if not isinstance(pts, np.ndarray):
        pts = np.array(pts)

    if pts.dtype not in (np.float32, np.float64):
        pts = pts.astype(np.float64)

    if pts.ndim == 0:
        pts = np.array([pts], dtype=pts.dtype)
    elif pts.ndim > 1:
        pts = pts.ravel()

    return pts


def _normalize_basis_output_1D(
    arr: npt.NDArray[np.float32 | np.float64], input_shape: tuple[int, ...]
) -> npt.NDArray[np.float32 | np.float64]:
    """
    Normalize the output of a 1D basis function evaluation to the input shape.

    The output array will be reshaped to have the same shape as the input points,
    with the last dimension being the number of basis functions.

    Args:
        arr (npt.NDArray[np.float32 | np.float64]): The output array.
        input_shape (tuple[int, ...]): The shape of the input points (before normalization).

    Returns:
        npt.NDArray[np.float32 | np.float64]: The normalized output array.
    """
    if len(input_shape) == 0:
        return arr.squeeze()
    else:
        return arr.reshape(*input_shape, -1)
