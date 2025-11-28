"""Utility functions for basis function evaluation."""

import numpy as np
from numpy import typing as npt


def _normalize_points_1D(pts: npt.ArrayLike) -> npt.NDArray[np.float32 | np.float64]:
    """Normalize points to a 1D float array for basis function evaluation.

    Converts input points (scalar, list, or numpy array) to a 1D numpy array
    with floating point dtype. Types different from float32 or float64 are
    automatically converted to float64.
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


def _compute_final_output_shape_1D(input_shape: tuple[int, ...], n_basis: int) -> tuple[int, ...]:
    """Compute the final output shape for 1D basis functions.

    Args:
        input_shape (tuple[int, ...]): The shape of the input points (before normalization).
        n_basis (int): The number of basis functions (degree + 1).

    Returns:
        tuple[int, ...]: The final output shape.
    """
    if len(input_shape) == 0:
        # Scalar input: output shape is (n_basis,)
        return (n_basis,)
    else:
        # Non-scalar input: output shape is (*input_shape, n_basis)
        return (*input_shape, n_basis)


def _normalize_basis_output_1D(
    arr: npt.NDArray[np.float32 | np.float64], input_shape: tuple[int, ...]
) -> npt.NDArray[np.float32 | np.float64]:
    """Normalize the output of a 1D basis function evaluation to the input shape.

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


def _validate_out_array_1D(
    out: npt.NDArray[np.float32 | np.float64],
    expected_shape: tuple[int, ...],
    expected_dtype: npt.DTypeLike,
) -> None:
    """Validate that the output array has the correct shape and dtype.

    This function follows NumPy's style for output array validation.
    Checks that the array has the expected shape and dtype.

    Args:
        out (npt.NDArray[np.float32 | np.float64]): The output array to validate.
        expected_shape (tuple[int, ...]): The expected shape of the output array.
        expected_dtype (npt.DTypeLike): The expected dtype (should be np.float32 or np.float64).

    Raises:
        ValueError: If the array shape or dtype does not match expectations.
    """
    if out.shape != expected_shape:
        raise ValueError(f"Output array has shape {out.shape}, but expected shape {expected_shape}")
    if out.dtype != expected_dtype:
        raise ValueError(f"Output array has dtype {out.dtype}, but expected dtype {expected_dtype}")
    if not out.flags.writeable:
        raise ValueError("Output array is not writeable")
