"""Tests for change_basis_1D module."""

import numpy as np
import pytest

from pantr.basis import LagrangeVariant, eval_Bernstein_basis_1D, eval_Lagrange_basis_1D
from pantr.change_basis_1D import (
    create_Bernstein_to_Lagrange_basis_operator,
    create_Lagrange_to_Bernstein_basis_operator,
)


class TestBernsteinToLagrangeBasisOperator:
    """Test the create_Bernstein_to_Lagrange_basis_operator function."""

    def test_degree_zero_error(self) -> None:
        """Test that degree lower than 1 raises ValueError."""
        with pytest.raises(ValueError, match="Degree must at least 1"):
            create_Bernstein_to_Lagrange_basis_operator(0)

    def test_negative_degree_error(self) -> None:
        """Test that negative degree raises ValueError."""
        with pytest.raises(ValueError, match="Degree must at least 1"):
            create_Bernstein_to_Lagrange_basis_operator(-1)

    def test_inverse_relationship(self) -> None:
        """Test that Bernstein to Lagrange is inverse of Lagrange to Bernstein."""
        degree = 2
        variant = LagrangeVariant.EQUISPACES
        lagrange_to_bernstein = create_Lagrange_to_Bernstein_basis_operator(degree, variant)
        bernstein_to_lagrange = create_Bernstein_to_Lagrange_basis_operator(degree, variant)

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

        C = create_Bernstein_to_Lagrange_basis_operator(degree, variant)
        np.testing.assert_array_almost_equal(bernsteins @ C.T, lagranges)

        C_inv = create_Lagrange_to_Bernstein_basis_operator(degree, variant)
        np.testing.assert_array_almost_equal(lagranges @ C_inv.T, bernsteins)
