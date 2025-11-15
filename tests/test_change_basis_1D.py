"""Tests for change_basis_1D module."""

import numpy as np
import numpy.typing as npt
import pytest

from pantr.basis import (
    LagrangeVariant,
    eval_Bernstein_basis_1D,
    eval_cardinal_Bspline_basis_1D,
    eval_Lagrange_basis_1D,
)
from pantr.change_basis_1D import (
    _create_change_basis,
    create_Bernstein_to_cardinal_change_basis,
    create_Bernstein_to_Lagrange_change_basis,
    create_cardinal_to_Bernstein_change_basis,
    create_Lagrange_to_Bernstein_change_basis,
)


class TestLagrangeToBernsteinBasisOperator:
    """Test the create_Lagrange_to_Bernstein_change_basis function."""

    def test_degree_zero_error(self) -> None:
        """Test that degree lower than 1 raises ValueError."""
        with pytest.raises(ValueError, match="Degree must at least 1"):
            create_Lagrange_to_Bernstein_change_basis(0)

    def test_negative_degree_error(self) -> None:
        """Test that negative degree raises ValueError."""
        with pytest.raises(ValueError, match="Degree must at least 1"):
            create_Lagrange_to_Bernstein_change_basis(-1)

    def test_invalid_dtype_error(self) -> None:
        """Test that invalid dtype raises ValueError."""
        with pytest.raises(ValueError, match="dtype must be float32 or float64"):
            create_Lagrange_to_Bernstein_change_basis(2, dtype=np.int32)
        with pytest.raises(ValueError, match="dtype must be float32 or float64"):
            create_Lagrange_to_Bernstein_change_basis(2, dtype=np.float16)


class TestBernsteinToLagrangeBasisOperator:
    """Test the create_Bernstein_to_Lagrange_basis function."""

    def test_degree_zero_error(self) -> None:
        """Test that degree lower than 1 raises ValueError."""
        with pytest.raises(ValueError, match="Degree must at least 1"):
            create_Bernstein_to_Lagrange_change_basis(0)

    def test_negative_degree_error(self) -> None:
        """Test that negative degree raises ValueError."""
        with pytest.raises(ValueError, match="Degree must at least 1"):
            create_Bernstein_to_Lagrange_change_basis(-1)

    def test_invalid_dtype_error(self) -> None:
        """Test that invalid dtype raises ValueError."""
        with pytest.raises(ValueError, match="dtype must be float32 or float64"):
            create_Bernstein_to_Lagrange_change_basis(2, dtype=np.int32)
        with pytest.raises(ValueError, match="dtype must be float32 or float64"):
            create_Bernstein_to_Lagrange_change_basis(2, dtype=np.float16)

    def test_inverse_relationship(self) -> None:
        """Test that Bernstein to Lagrange is inverse of Lagrange to Bernstein."""
        degree = 2
        variant = LagrangeVariant.EQUISPACES
        lagrange_to_bernstein = create_Lagrange_to_Bernstein_change_basis(degree, variant)
        bernstein_to_lagrange = create_Bernstein_to_Lagrange_change_basis(degree, variant)

        # Should be inverse matrices
        identity = lagrange_to_bernstein @ bernstein_to_lagrange
        np.testing.assert_array_almost_equal(identity, np.eye(degree + 1))

    @pytest.mark.parametrize(
        "variant",
        [
            LagrangeVariant.EQUISPACES,
            LagrangeVariant.GAUSS_LEGENDRE,
            LagrangeVariant.GAUSS_LOBATTO_LEGENDRE,
            LagrangeVariant.CHEBYSHEV_1ST,
            LagrangeVariant.CHEBYSHEV_2ND,
        ],
    )
    @pytest.mark.parametrize("degree", [1, 2, 3, 4])
    def test_values(self, degree: int, variant: LagrangeVariant) -> None:
        """Test Bernstein evaluations transformed with the operator return Lagrange evaluations."""
        n_pts = 10
        tt = np.linspace(0.0, 1.0, n_pts)

        bernsteins = eval_Bernstein_basis_1D(degree, tt)
        lagranges = eval_Lagrange_basis_1D(degree, variant, tt)

        C = create_Bernstein_to_Lagrange_change_basis(degree, variant)
        np.testing.assert_array_almost_equal(bernsteins @ C.T, lagranges)

        C_inv = create_Lagrange_to_Bernstein_change_basis(degree, variant)
        np.testing.assert_array_almost_equal(lagranges @ C_inv.T, bernsteins)


class TestCardinalToBernsteinBasisOperator:
    """Test the create_cardinal_to_Bernstein_basis function."""

    def test_inverse_relationship(self) -> None:
        """Test that cardinal to Bernstein is inverse of Bernstein to cardinal."""
        degree = 2
        bernstein_to_cardinal = create_Bernstein_to_cardinal_change_basis(degree)
        cardinal_to_bernstein = create_cardinal_to_Bernstein_change_basis(degree)

        # Should be inverse matrices
        identity = bernstein_to_cardinal @ cardinal_to_bernstein
        np.testing.assert_array_almost_equal(identity, np.eye(degree + 1))

    def test_negative_degree_error(self) -> None:
        """Test that negative degree raises ValueError."""
        with pytest.raises(ValueError, match="Degree must be non-negative"):
            create_cardinal_to_Bernstein_change_basis(-1)
        with pytest.raises(ValueError, match="Degree must be non-negative"):
            create_Bernstein_to_cardinal_change_basis(-1)

    def test_invalid_dtype_error(self) -> None:
        """Test that invalid dtype raises ValueError."""
        with pytest.raises(ValueError, match="dtype must be float32 or float64"):
            create_Bernstein_to_cardinal_change_basis(2, dtype=np.int32)
        with pytest.raises(ValueError, match="dtype must be float32 or float64"):
            create_Bernstein_to_cardinal_change_basis(2, dtype=np.float16)
        with pytest.raises(ValueError, match="dtype must be float32 or float64"):
            create_cardinal_to_Bernstein_change_basis(2, dtype=np.int32)
        with pytest.raises(ValueError, match="dtype must be float32 or float64"):
            create_cardinal_to_Bernstein_change_basis(2, dtype=np.float16)

    def test_values(self) -> None:
        """Test that cardinal evaluations transformed with operator return Bernstein evaluations."""
        for degree in [1, 2, 3, 4]:
            n_pts = 10
            tt = np.linspace(0.0, 1.0, n_pts)
            bernsteins = eval_Bernstein_basis_1D(degree, tt)
            cardinals = eval_cardinal_Bspline_basis_1D(degree, tt)

            C = create_cardinal_to_Bernstein_change_basis(degree)
            np.testing.assert_array_almost_equal(bernsteins, cardinals @ C.T)

            C_inv = create_Bernstein_to_cardinal_change_basis(degree)
            np.testing.assert_array_almost_equal(cardinals, bernsteins @ C_inv.T)


class TestCreateChangeBasis:
    """Test the _create_change_basis private function."""

    def test_invalid_n_quad_pts_error(self) -> None:
        """Test that non-positive n_quad_pts raises ValueError."""
        degree = 2

        def bernstein(
            pts: npt.NDArray[np.float32 | np.float64],
        ) -> npt.NDArray[np.float32 | np.float64]:
            return eval_Bernstein_basis_1D(degree, pts)

        def cardinal(
            pts: npt.NDArray[np.float32 | np.float64],
        ) -> npt.NDArray[np.float32 | np.float64]:
            return eval_cardinal_Bspline_basis_1D(degree, pts)

        with pytest.raises(ValueError, match="Number of quadrature points must be positive"):
            _create_change_basis(bernstein, cardinal, n_quad_pts=0)

        with pytest.raises(ValueError, match="Number of quadrature points must be positive"):
            _create_change_basis(bernstein, cardinal, n_quad_pts=-1)

    def test_invalid_dtype_error(self) -> None:
        """Test that invalid dtype raises ValueError."""
        degree = 2

        def bernstein(
            pts: npt.NDArray[np.float32 | np.float64],
        ) -> npt.NDArray[np.float32 | np.float64]:
            return eval_Bernstein_basis_1D(degree, pts)

        def cardinal(
            pts: npt.NDArray[np.float32 | np.float64],
        ) -> npt.NDArray[np.float32 | np.float64]:
            return eval_cardinal_Bspline_basis_1D(degree, pts)

        with pytest.raises(ValueError, match="dtype must be float32 or float64"):
            _create_change_basis(bernstein, cardinal, n_quad_pts=3, dtype=np.int32)

        with pytest.raises(ValueError, match="dtype must be float32 or float64"):
            _create_change_basis(bernstein, cardinal, n_quad_pts=3, dtype=np.float16)
