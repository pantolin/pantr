"""Tests for tolerance utilities."""

from __future__ import annotations

import numpy as np
import pytest

from pantr.tolerance import (
    get_conservative_tolerance,
    get_default_tolerance,
    get_machine_epsilon,
    get_strict_tolerance,
    get_tolerance_info,
)

DEFAULT_TOL_F64: float = 1e-12
STRICT_TOL_F64: float = 1e-15
CONSERVATIVE_TOL_F64: float = 1e-10
DEFAULT_TOL_F32: float = 1e-6


class TestTolerance:
    """Test suite for tolerance utilities."""

    @pytest.mark.parametrize(
        ("dtype", "expected"),
        [
            (np.float16, 1e-3),
            (np.float32, 1e-6),
            ("float64", 1e-12),
            (np.longdouble, 1e-15),
        ],
    )
    def test_get_default_tolerance(
        self, dtype: np.dtype | type[np.floating], expected: float
    ) -> None:
        """Test get_default_tolerance with various dtypes."""
        assert get_default_tolerance(dtype) == expected

    @pytest.mark.parametrize(
        ("dtype", "expected"),
        [
            (np.float16, 1e-4),
            (np.float32, 1e-7),
            ("float64", 1e-15),
            (np.longdouble, 1e-18),
        ],
    )
    def test_get_strict_tolerance(
        self, dtype: np.dtype | type[np.floating], expected: float
    ) -> None:
        """Test get_strict_tolerance with various dtypes."""
        assert get_strict_tolerance(dtype) == expected

    @pytest.mark.parametrize(
        ("dtype", "expected"),
        [
            (np.float16, 1e-2),
            (np.float32, 1e-5),
            ("float64", 1e-10),
            (np.longdouble, 1e-12),
        ],
    )
    def test_get_conservative_tolerance(
        self, dtype: np.dtype | type[np.floating], expected: float
    ) -> None:
        """Test get_conservative_tolerance with various dtypes."""
        assert get_conservative_tolerance(dtype) == expected

    @pytest.mark.parametrize(
        "dtype",
        [np.float16, np.float32, "float64", np.longdouble],
    )
    def test_get_machine_epsilon(self, dtype: np.dtype | type[np.floating]) -> None:
        """Test get_machine_epsilon against np.finfo."""
        assert get_machine_epsilon(dtype) == np.finfo(dtype).eps

    def test_invalid_dtype_raises_error(self) -> None:
        """Test that an unsupported dtype raises a ValueError."""
        with pytest.raises(ValueError, match="Unsupported dtype"):
            get_default_tolerance(np.int32)
        with pytest.raises(ValueError, match="Unsupported dtype"):
            get_strict_tolerance("int64")
        with pytest.raises(ValueError, match="Unsupported dtype"):
            get_conservative_tolerance(np.complex64)
        with pytest.raises(ValueError, match="Unsupported dtype"):
            get_machine_epsilon(np.uint8)

    def test_get_tolerance_info(self) -> None:
        """Test the get_tolerance_info dictionary."""
        dtype = np.float64
        info = get_tolerance_info(dtype)

        assert info["dtype"] == dtype
        assert info["machine_epsilon"] == np.finfo(dtype).eps
        assert info["default_tolerance"] == DEFAULT_TOL_F64
        assert info["strict_tolerance"] == STRICT_TOL_F64
        assert info["conservative_tolerance"] == CONSERVATIVE_TOL_F64
        assert info["precision_bits"] == np.finfo(dtype).precision
        assert info["precision_decimals"] == np.finfo(dtype).precision
        assert info["resolution"] == np.finfo(dtype).resolution
        assert info["max_value"] == np.finfo(dtype).max
        assert info["min_value"] == np.finfo(dtype).tiny

    def test_get_tolerance_info_string_dtype(self) -> None:
        """Test get_tolerance_info with a string dtype."""
        dtype_str = "float32"
        info = get_tolerance_info(dtype_str)
        finfo = np.finfo(dtype_str)

        assert info["dtype"] == dtype_str
        assert info["machine_epsilon"] == finfo.eps
        assert info["default_tolerance"] == DEFAULT_TOL_F32

    def test_tolerance_info_keys(self) -> None:
        """Test that get_tolerance_info returns all expected keys."""
        info = get_tolerance_info(np.float32)
        expected_keys = {
            "dtype",
            "machine_epsilon",
            "default_tolerance",
            "strict_tolerance",
            "conservative_tolerance",
            "precision_bits",
            "precision_decimals",
            "resolution",
            "max_value",
            "min_value",
        }
        assert set(info.keys()) == expected_keys
