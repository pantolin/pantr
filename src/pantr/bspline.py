"""Bspline class and  utilities."""

import numpy as np
from numpy import typing as npt

from .bspline_space import BsplineSpace


class Bspline:
    """A class representing a B-spline."""

    def __init__(
        self, space: BsplineSpace, control_points: npt.ArrayLike, is_rational: bool = False
    ) -> None:
        """Initialize a B-spline.

        Args:
            space (BsplineSpace): The B-spline space.
            control_points (npt.ArrayLike): The control points.
            is_rational (bool): Whether the B-spline is rational.

        Raises:
            ValueError: If the number of control points is not a multiple
            of the number of basis functions.
            ValueError: If the B-spline has rank smaller than 1.
        """
        self._space = space

        control_points = np.asarray(control_points)
        num_basis = space.num_total_basis
        if control_points.size % num_basis != 0:
            raise ValueError(
                f"The number of control points must be a multiple of the number of basis functions."
                f"Got {control_points.size} control points and {num_basis} basis functions."
            )

        self._control_points = control_points.reshape([*space.num_basis, -1])

        if self._control_points.dtype != self._space.dtype:
            raise ValueError(
                f"The control points must have the same dtype as the B-spline space."
                f"Got {self._control_points.dtype} control points and {self._space.dtype} "
                "B-spline space."
            )

        self._is_rational = is_rational

        if self.rank <= 0:
            raise ValueError(f"The B-spline must have at least rank one. Got rank {self.rank}")

    @property
    def dim(self) -> int:
        """The dimension of the B-spline."""
        return self._space.dim

    @property
    def degree(self) -> tuple[int, ...]:
        """The degrees of the B-spline."""
        return self._space.degrees

    @property
    def space(self) -> BsplineSpace:
        """The B-spline space."""
        return self._space

    @property
    def control_points(self) -> npt.NDArray[np.float32 | np.float64]:
        """The control points of the B-spline."""
        return self._control_points

    @property
    def is_rational(self) -> bool:
        """Whether the B-spline is rational."""
        return self._is_rational

    @property
    def rank(self) -> int:
        """The rank of the B-spline."""
        rk = self._control_points.ndim - self.dim
        return rk - 1 if self.is_rational else rk
