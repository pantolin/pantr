"""Knot vector generation utilities for B-splines.

This module provides functions to create various types of knot vectors including
uniform open, uniform periodic, and cardinal B-spline knot vectors with
configurable continuity and domain parameters.
"""

from typing import Any, cast

import numpy as np
import numpy.typing as npt


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


def _get_ends_and_type(
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


def create_uniform_open_knot_vector(
    num_intervals: int,
    degree: int,
    continuity: int | None = None,
    domain: tuple[np.float32 | np.float64 | float, np.float32 | np.float64 | float] | None = None,
    dtype: npt.DTypeLike | None = None,
) -> npt.NDArray[np.float32 | np.float64]:
    """Create a uniform open knot vector.

    An open knot vector has the first and last knots repeated (degree+1) times,
    ensuring the B-spline interpolates the first and last control points.

    Args:
        num_intervals (int): Number of intervals in the domain. Must be non-negative.
        degree (int): B-spline degree. Must be non-negative.
        continuity (Optional[int]): Continuity level at interior knots.
            Must be between -1 and degree-1. Defaults to degree-1 (maximum continuity).
        domain (Optional[tuple[np.float32 | np.float64 | float, np.float32 | np.float64 | float]]):
            Domain boundaries as (start, end). Defaults to (0.0, 1.0) if not provided.
        dtype (Optional[np.dtype]): Data type for the knot vector.
            If None, inferred from start/end or defaults to float64.

    Returns:
        npt.NDArray[np.floating]: Open knot vector with uniform spacing.

    Raises:
        ValueError: If any parameter is invalid.

    Example:
        >>> create_uniform_open_knot_vector(2, 2, domain=(0.0, 1.0))
        array([0., 0., 0., 0.5, 1., 1., 1.])
    """
    start_value: np.float32 | np.float64 | None
    end_value: np.float32 | np.float64 | None
    if domain is None:
        start_value = None
        end_value = None
    else:
        start_raw, end_raw = domain
        start_value = start_raw if isinstance(start_raw, np.floating) else np.float64(start_raw)
        end_value = end_raw if isinstance(end_raw, np.floating) else np.float64(end_raw)

    start, end, dtype = _get_ends_and_type(start_value, end_value, dtype)

    continuity = degree - 1 if continuity is None else continuity

    _validate_knot_input(
        num_intervals,
        degree,
        continuity,
        (start, end),
        dtype,
    )

    unique_knots = np.linspace(start, end, num_intervals + 1, dtype=dtype)
    knots = np.array([start] * (degree + 1), dtype)

    interior_multiplicity = degree - continuity
    for knot in unique_knots[1:-1]:
        knots = np.append(knots, [knot] * interior_multiplicity)

    knots = np.append(knots, [end] * (degree + 1))

    return knots


def create_uniform_periodic_knot_vector(
    num_intervals: int,
    degree: int,
    continuity: int | None = None,
    domain: tuple[np.float32 | np.float64 | float, np.float32 | np.float64 | float] | None = None,
    dtype: npt.DTypeLike | None = np.float64,
) -> npt.NDArray[np.float32 | np.float64]:
    """Create a uniform periodic knot vector.

    A periodic knot vector extends beyond the domain boundaries to ensure
    periodicity of the B-spline basis functions.

    Args:
        num_intervals (int): Number of intervals in the domain. Must be non-negative.
        degree (int): B-spline degree. Must be non-negative.
        continuity (Optional[int]): Continuity level at interior knots.
            Must be between -1 and degree-1. Defaults to degree-1 (maximum continuity).
        domain (Optional[tuple[np.float32 | np.float64 | float, np.float32 | np.float64 | float]]):
            Domain boundaries as (start, end). Defaults to (0.0, 1.0) if not provided.
        dtype (Optional[np.dtype]): Data type for the knot vector.
            If None, inferred from start/end or defaults to float64.

    Returns:
        npt.NDArray[np.floating]: Periodic knot vector with uniform spacing.

    Raises:
        ValueError: If any parameter is invalid.

    Example:
        >>> create_uniform_periodic_knot_vector(2, 2, domain=(0.0, 1.0))
        array([-1. , -0.5,  0. ,  0.5,  1. ,  1.5,  2. ])
    """
    start_value: np.float32 | np.float64 | None
    end_value: np.float32 | np.float64 | None
    if domain is None:
        start_value = None
        end_value = None
    else:
        start_raw, end_raw = domain
        start_value = start_raw if isinstance(start_raw, np.floating) else np.float64(start_raw)
        end_value = end_raw if isinstance(end_raw, np.floating) else np.float64(end_raw)

    start, end, dtype = _get_ends_and_type(start_value, end_value, dtype)
    continuity = degree - 1 if continuity is None else continuity

    _validate_knot_input(
        num_intervals,
        degree,
        continuity,
        (start, end),
        dtype,
    )

    # Create uniform spacing for unique interior knots
    unique_knots = np.linspace(start, end, num_intervals + 1, dtype=dtype)

    # Build knot vector with repetitions
    knots = np.array([], dtype=dtype)

    multiplicity = degree - continuity

    # Starting periodic knots.
    length = (end - start) / num_intervals
    knots = np.linspace(
        start - length * (degree - multiplicity + 1),
        start,
        degree + 2 - multiplicity,
        dtype=dtype,
    )[:-1]

    # Interior knots with specified multiplicity
    for knot in unique_knots:
        knots = np.append(knots, [knot] * multiplicity)

    # End periodic knots.
    knots = np.append(
        knots,
        np.linspace(
            end,
            end + length * (degree - multiplicity + 1),
            degree + 2 - multiplicity,
            dtype=dtype,
        )[1:],
    )

    return knots


def create_cardinal_Bspline_knot_vector(
    num_intervals: int,
    degree: int,
    dtype: npt.DTypeLike = np.float64,
) -> npt.NDArray[np.float32 | np.float64]:
    """Create a knot vector for cardinal B-spline basis functions.

    Cardinal B-splines are B-splines defined on uniform knot vectors with
    maximum continuity, where the basis functions in the central region
    have the same shape and are translated versions of each other.

    Args:
        num_intervals (int): Number of intervals in the domain. Must be at least 1.
        degree (int): B-spline degree. Must be non-negative.
        dtype (npt.DTypeLike): Data type for the knot vector.
            It must be either float32 or float64. Defaults to np.float64.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Cardinal B-spline knot vector
            with uniform spacing.

    Raises:
        ValueError: If num_intervals < 1, degree < 0, or dtype is not float32/float64.

    Example:
        >>> create_cardinal_Bspline_knot_vector(2, 2)
        array([-2., -1.,  0.,  1.,  2.,  3., 4.])
    """
    if num_intervals < 1:
        raise ValueError("num_intervals must be at least 1")

    if degree < 0:
        raise ValueError("degree must be non-negative")

    dtype_obj = np.dtype(dtype)
    if dtype_obj not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError("dtype must be float32 or float64")

    start_value: np.float32 | np.float64
    end_value: np.float32 | np.float64
    if dtype_obj == np.dtype(np.float64):
        start_value = np.float64(0)
        end_value = np.float64(num_intervals)
    else:
        start_value = np.float32(0)
        end_value = np.float32(num_intervals)

    return create_uniform_periodic_knot_vector(
        num_intervals,
        degree,
        continuity=degree - 1,
        domain=(start_value, end_value),
        dtype=dtype_obj,
    )
