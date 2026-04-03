"""
Tests unitaires pour geometry/frames.py — Bishop frame et tangentes.
"""
import pytest
import numpy as np
from geometry.frames import compute_tangents, compute_bishop_frame


class TestComputeTangents:
    """Vérifie le calcul des vecteurs tangents."""

    def test_straight_line_x(self):
        pts = np.array([[0,0,0],[1,0,0],[2,0,0],[3,0,0]], dtype=float)
        T = compute_tangents(pts)
        for t in T:
            assert np.allclose(np.abs(t), [1, 0, 0], atol=1e-6)

    def test_unit_norm(self):
        """Tous les vecteurs tangents doivent être unitaires."""
        pts = np.array([[0,0,0],[1,1,0],[2,0,1],[3,1,1]], dtype=float)
        T = compute_tangents(pts)
        norms = np.linalg.norm(T, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)

    def test_minimum_points_error(self):
        with pytest.raises(ValueError):
            compute_tangents(np.array([[0,0,0]]))

    def test_output_shape(self):
        pts = np.random.rand(20, 3)
        T = compute_tangents(pts)
        assert T.shape == (20, 3)


class TestBishopFrame:
    """Vérifie les propriétés du repère de Bishop (transport parallèle)."""

    def test_output_shapes(self):
        pts = np.array([[i, 0, 0] for i in range(10)], dtype=float)
        T, N, B = compute_bishop_frame(pts)
        assert T.shape == (10, 3)
        assert N.shape == (10, 3)
        assert B.shape == (10, 3)

    def test_orthonormality(self):
        """T, N, B doivent former un repère orthonormé à chaque point."""
        theta = np.linspace(0, 2 * np.pi, 50)
        pts = np.column_stack([np.cos(theta), np.sin(theta), theta * 0.1])
        T, N, B = compute_bishop_frame(pts)

        for i in range(len(pts)):
            assert np.linalg.norm(T[i]) == pytest.approx(1.0, abs=1e-6)
            assert np.linalg.norm(N[i]) == pytest.approx(1.0, abs=1e-6)
            assert np.linalg.norm(B[i]) == pytest.approx(1.0, abs=1e-6)
            assert np.dot(T[i], N[i]) == pytest.approx(0.0, abs=1e-4)
            assert np.dot(T[i], B[i]) == pytest.approx(0.0, abs=1e-4)
            assert np.dot(N[i], B[i]) == pytest.approx(0.0, abs=1e-4)

    def test_straight_line_no_twist(self):
        """Sur une droite, N et B doivent rester constants."""
        pts = np.array([[i, 0, 0] for i in range(20)], dtype=float)
        T, N, B = compute_bishop_frame(pts)
        for i in range(1, len(pts)):
            assert np.allclose(N[i], N[0], atol=1e-6)
            assert np.allclose(B[i], B[0], atol=1e-6)

    def test_tangent_aligned_with_curve(self):
        """La tangente doit pointer dans la direction de progression."""
        pts = np.array([[0,0,0],[1,0,0],[2,0,0],[3,0,0]], dtype=float)
        T, N, B = compute_bishop_frame(pts)
        for t in T:
            assert t[0] > 0.9  # Principalement en X
