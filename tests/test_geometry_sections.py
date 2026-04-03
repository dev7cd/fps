"""
Tests unitaires pour geometry/sections.py — Sections et factory.
"""
import pytest
import numpy as np
from geometry.sections import (
    CircularSection,
    SuperEllipticalSection,
    create_section,
)


class TestCircularSection:
    """Vérifie la section circulaire."""

    def test_area(self):
        s = CircularSection(radius=1.0)
        assert s.get_area() == pytest.approx(np.pi, rel=1e-6)

    def test_area_small_radius(self):
        s = CircularSection(radius=0.02)
        assert s.get_area() == pytest.approx(np.pi * 0.02**2, rel=1e-6)

    def test_contains_center(self):
        s = CircularSection(radius=1.0)
        assert s.contains_point(0.0, 0.0)

    def test_contains_inside(self):
        s = CircularSection(radius=1.0)
        assert s.contains_point(0.5, 0.5)

    def test_not_contains_outside(self):
        s = CircularSection(radius=1.0)
        assert not s.contains_point(1.0, 1.0)  # sqrt(2) > 1

    def test_generate_points_shape(self):
        s = CircularSection(radius=1.0)
        pts = s.generate_points(32)
        assert pts.shape == (32, 3)

    def test_generate_points_on_circle(self):
        """Tous les points générés doivent être sur le cercle de rayon 1."""
        s = CircularSection(radius=1.0)
        pts = s.generate_points(100)
        radii = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
        np.testing.assert_allclose(radii, 1.0, atol=1e-6)

    def test_z_coordinate_zero(self):
        s = CircularSection(radius=1.0)
        pts = s.generate_points(20)
        np.testing.assert_allclose(pts[:, 2], 0.0)


class TestSuperEllipticalSection:
    """Vérifie la section superelliptique."""

    def test_ellipse_area(self):
        """Avec n=2, c'est une ellipse : aire = pi*a*b."""
        s = SuperEllipticalSection(a=2.0, b=1.0, n=2.0)
        assert s.get_area() == pytest.approx(np.pi * 2.0, rel=1e-4)

    def test_squircle_area_larger_than_ellipse(self):
        """Avec n>2, l'aire doit être plus grande qu'une ellipse de mêmes axes."""
        ellipse = SuperEllipticalSection(a=1.0, b=1.0, n=2.0)
        squircle = SuperEllipticalSection(a=1.0, b=1.0, n=4.0)
        assert squircle.get_area() > ellipse.get_area()

    def test_contains_center(self):
        s = SuperEllipticalSection(a=1.0, b=0.5, n=2.5)
        assert s.contains_point(0.0, 0.0)

    def test_contains_on_axis(self):
        s = SuperEllipticalSection(a=2.0, b=1.0, n=2.5)
        assert s.contains_point(1.5, 0.0)
        assert s.contains_point(0.0, 0.8)

    def test_not_contains_outside(self):
        s = SuperEllipticalSection(a=1.0, b=1.0, n=2.5)
        assert not s.contains_point(1.5, 0.0)

    def test_generate_points_shape(self):
        s = SuperEllipticalSection(a=1.0, b=0.5, n=3.0)
        pts = s.generate_points(40)
        assert pts.shape == (40, 3)


class TestCreateSectionFactory:
    """Vérifie la factory create_section."""

    def test_circular(self):
        s = create_section('circular', radius=0.5)
        assert isinstance(s, CircularSection)
        assert s.get_area() == pytest.approx(np.pi * 0.25, rel=1e-6)

    def test_superelliptical(self):
        s = create_section('superelliptical', major_radius=0.02, minor_radius=0.015, exponent=2.5)
        assert isinstance(s, SuperEllipticalSection)
        assert s.a == 0.02
        assert s.b == 0.015
        assert s.n == 2.5

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError):
            create_section('hexagonal')

    def test_default_radius_fallback(self):
        s = create_section('circular')
        assert s.get_area() == pytest.approx(np.pi, rel=1e-6)  # radius=1.0 par défaut
