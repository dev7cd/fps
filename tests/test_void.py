"""! @brief Quick Doxygen documentation for test module."""
"""
Tests unitaires pour core/void.py — Void.
Note : Void utilise @dataclass, on teste ses propriétés et la méthode contains_point.
       La méthode intersect_fiber n'est pas encore implémentée (stub).
"""
import pytest
import numpy as np
from core.void import Void


    """! @brief Test class for functionality. """
class TestVoidCreation:
    """Vérifie la création d'un objet Void."""

    def test_basic_creation(self):
        v = Void(id=1, center=np.array([0.5, 0.5, 0.5]), radius=0.1)
        assert v.id == 1
        assert v.radius == 0.1
        np.testing.assert_array_equal(v.center, [0.5, 0.5, 0.5])

    def test_zero_radius(self):
        v = Void(id=0, center=np.array([0.0, 0.0, 0.0]), radius=0.0)
        assert v.radius == 0.0


    """! @brief Test class for functionality. """
class TestContainsPoint:
    """Vérifie la détection de points à l'intérieur de la bulle."""

    def test_center_is_inside(self):
        v = Void(id=1, center=np.array([1.0, 1.0, 1.0]), radius=0.5)
        assert v.contains_point(np.array([1.0, 1.0, 1.0]))

    def test_point_inside(self):
        v = Void(id=1, center=np.array([0.0, 0.0, 0.0]), radius=1.0)
        assert v.contains_point(np.array([0.5, 0.0, 0.0]))

    def test_point_on_surface(self):
        v = Void(id=1, center=np.array([0.0, 0.0, 0.0]), radius=1.0)
        assert v.contains_point(np.array([1.0, 0.0, 0.0]))

    def test_point_outside(self):
        v = Void(id=1, center=np.array([0.0, 0.0, 0.0]), radius=1.0)
        assert not v.contains_point(np.array([2.0, 0.0, 0.0]))

    def test_point_just_outside(self):
        v = Void(id=1, center=np.array([0.0, 0.0, 0.0]), radius=1.0)
        assert not v.contains_point(np.array([1.001, 0.0, 0.0]))

    def test_point_diagonal(self):
        v = Void(id=1, center=np.array([0.0, 0.0, 0.0]), radius=1.0)
        # Distance = sqrt(3) ≈ 1.732 > 1.0
        assert not v.contains_point(np.array([1.0, 1.0, 1.0]))

    def test_offset_center(self):
        v = Void(id=1, center=np.array([5.0, 5.0, 5.0]), radius=0.5)
        assert v.contains_point(np.array([5.2, 5.1, 5.0]))
        assert not v.contains_point(np.array([0.0, 0.0, 0.0]))

    def test_zero_radius_only_center(self):
        v = Void(id=1, center=np.array([1.0, 2.0, 3.0]), radius=0.0)
        assert v.contains_point(np.array([1.0, 2.0, 3.0]))
        assert not v.contains_point(np.array([1.0, 2.0, 3.001]))
