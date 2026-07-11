"""! @brief Quick Doxygen documentation for test module."""
"""
Tests unitaires pour generation/periodicity.py — PeriodicManager.
"""
import pytest
import numpy as np
from unittest.mock import MagicMock
from generation.periodicity import PeriodicManager


def _make_fiber(control_points, radius=0.05):
    f = MagicMock()
    f.control_points = np.array(control_points, dtype=float)
    f.centerline = f.control_points.copy()
    f.radius = radius
    p_min = np.min(f.centerline, axis=0) - radius
    p_max = np.max(f.centerline, axis=0) + radius
    f.bbox = (p_min, p_max)
    f.refresh_geometry = MagicMock()
    return f


    """! @brief Test class for functionality. """
class TestWrapFiber:

    def test_fiber_inside_not_moved(self):
        f = _make_fiber([[0.5, 0.5, 0.5], [0.6, 0.5, 0.5]])
        moved = PeriodicManager.wrap_fiber(f, (1.0, 1.0, 1.0))
        assert moved is False

    def test_fiber_outside_positive_wrapped(self):
        f = _make_fiber([[1.2, 0.5, 0.5], [1.3, 0.5, 0.5]])
        moved = PeriodicManager.wrap_fiber(f, (1.0, 1.0, 1.0))
        assert moved is True
        assert np.mean(f.control_points[:, 0]) < 1.0

    def test_fiber_outside_negative_wrapped(self):
        f = _make_fiber([[-0.3, 0.5, 0.5], [-0.1, 0.5, 0.5]])
        moved = PeriodicManager.wrap_fiber(f, (1.0, 1.0, 1.0))
        assert moved is True
        assert np.mean(f.control_points[:, 0]) > 0.0


    """! @brief Test class for functionality. """
class TestGenerateGhosts:

    def test_interior_fiber_no_ghosts(self):
        """Une fibre bien à l'intérieur ne doit pas générer de ghosts."""
        f = _make_fiber([[0.3, 0.3, 0.3], [0.7, 0.7, 0.7]], radius=0.001)
        shifts = PeriodicManager.generate_ghosts(f, (1.0, 1.0, 1.0))
        assert len(shifts) == 0

    def test_fiber_near_face_generates_ghosts(self):
        """Une fibre proche d'une face doit générer au moins 1 ghost."""
        f = _make_fiber([[0.02, 0.5, 0.5], [0.1, 0.5, 0.5]], radius=0.05)
        shifts = PeriodicManager.generate_ghosts(f, (1.0, 1.0, 1.0))
        assert len(shifts) >= 1
        # Le ghost doit être décalé de +Lx
        has_positive_x_shift = any(s[0] > 0 for s in shifts)
        assert has_positive_x_shift

    def test_fiber_near_corner_generates_multiple_ghosts(self):
        """Une fibre proche d'un coin doit générer plusieurs ghosts."""
        f = _make_fiber([[0.02, 0.02, 0.02], [0.05, 0.05, 0.05]], radius=0.05)
        shifts = PeriodicManager.generate_ghosts(f, (1.0, 1.0, 1.0))
        assert len(shifts) >= 3  # Au moins face + arête

    def test_ghost_shifts_are_multiples_of_box(self):
        """Chaque composante d'un shift doit être 0, +L ou -L."""
        f = _make_fiber([[0.02, 0.5, 0.98], [0.05, 0.5, 0.99]], radius=0.05)
        shifts = PeriodicManager.generate_ghosts(f, (1.0, 1.0, 1.0))
        for s in shifts:
            for i in range(3):
                assert s[i] in [-1.0, 0.0, 1.0]
