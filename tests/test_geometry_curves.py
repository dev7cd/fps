# SPDX-FileCopyrightText: 2026 Devine Ngouloubi <exauce-devine.ngouloubi@unicaen.fr>
# SPDX-License-Identifier: AGPL-3.0-or-later

"""! @brief Quick Doxygen documentation for test module."""
"""
Tests unitaires pour geometry/curves.py — CatmullRomSpline.
"""
import pytest
import numpy as np
from geometry.curves import CatmullRomSpline, generate_random_control_points


class TestCatmullRomInterpolation:
    """Checks the fundamental properties of the Catmull-Rom spline."""

    def test_output_shape(self):
        """La sortie doit avoir exactement num_points lignes et 3 colonnes."""
        pts = np.array([[0,0,0],[1,0,0],[2,1,0],[3,1,0],[4,0,0]], dtype=float)
        result = CatmullRomSpline.interpolate(pts, num_points=50)
        assert result.shape == (50, 3)

    def test_passes_near_control_points(self):
        """The curve must pass close to the interior control points."""
        pts = np.array([[0,0,0],[1,0,0],[2,1,0],[3,1,0],[4,0,0]], dtype=float)
        curve = CatmullRomSpline.interpolate(pts, num_points=500)
        for cp in pts[1:-1]:
            dists = np.linalg.norm(curve - cp, axis=1)
            assert np.min(dists) < 0.15, f"Courbe trop loin du point {cp}"

    def test_endpoints(self):
        """The curve endpoints must be close to the first/last points."""
        pts = np.array([[0,0,0],[1,0,0],[2,1,0],[3,0,0]], dtype=float)
        curve = CatmullRomSpline.interpolate(pts, num_points=200)
        assert np.linalg.norm(curve[0] - pts[0]) < 0.2
        assert np.linalg.norm(curve[-1] - pts[-1]) < 0.2

    def test_straight_line_stays_straight(self):
        """Collinear points must produce a nearly straight curve."""
        pts = np.array([[0,0,0],[1,0,0],[2,0,0],[3,0,0],[4,0,0]], dtype=float)
        curve = CatmullRomSpline.interpolate(pts, num_points=100)
        assert np.max(np.abs(curve[:, 1])) < 1e-6
        assert np.max(np.abs(curve[:, 2])) < 1e-6

    def test_few_points_fallback(self):
        """With fewer than 4 points, the method must still work (linear fallback)."""
        pts = np.array([[0,0,0],[1,1,1]], dtype=float)
        curve = CatmullRomSpline.interpolate(pts, num_points=10)
        assert curve.shape == (10, 3)

    def test_monotonic_arc_length(self):
        """The curvilinear abscissa must be increasing (no backtracking)."""
        pts = np.array([[0,0,0],[1,0,0],[2,0.5,0],[3,0,0],[4,0,0]], dtype=float)
        curve = CatmullRomSpline.interpolate(pts, num_points=200)
        segments = np.linalg.norm(np.diff(curve, axis=0), axis=1)
        assert np.all(segments >= 0)


class TestLinearResample:
    """Checks the equidistant resampling."""

    def test_equidistant_output(self):
        pts = np.array([[0,0,0],[1,0,0],[3,0,0]], dtype=float)
        resampled = CatmullRomSpline._linear_resample(pts, 5)
        segments = np.linalg.norm(np.diff(resampled, axis=0), axis=1)
        assert np.allclose(segments, segments[0], atol=1e-10)

    def test_preserves_total_length(self):
        pts = np.array([[0,0,0],[1,0,0],[1,1,0]], dtype=float)
        resampled = CatmullRomSpline._linear_resample(pts, 50)
        original_len = np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))
        resampled_len = np.sum(np.linalg.norm(np.diff(resampled, axis=0), axis=1))
        assert resampled_len == pytest.approx(original_len, rel=1e-2)


class TestGenerateRandomControlPoints:
    """Checks the random control-point generation."""

    def test_output_shape(self):
        rng = np.random.default_rng(42)
        pts = generate_random_control_points(10, 0.1, 0.01, (1.0, 1.0, 1.0), rng)
        assert pts.shape == (10, 3)

    def test_first_point_in_box(self):
        rng = np.random.default_rng(42)
        pts = generate_random_control_points(5, 0.1, 0.01, (2.0, 3.0, 4.0), rng)
        assert 0 <= pts[0, 0] <= 2.0
        assert 0 <= pts[0, 1] <= 3.0
        assert 0 <= pts[0, 2] <= 4.0

    def test_bias_x_alignment(self):
        """With a strong bias in X, the displacements must be mostly in X."""
        rng = np.random.default_rng(42)
        pts = generate_random_control_points(
            50, 0.1, 0.001, (10.0, 10.0, 10.0), rng,
            orientation_bias='x', bias_strength=0.95
        )
        displacements = np.diff(pts, axis=0)
        mean_abs = np.mean(np.abs(displacements), axis=0)
        assert mean_abs[0] > mean_abs[1]
        assert mean_abs[0] > mean_abs[2]

    def test_reproducibility(self):
        rng1 = np.random.default_rng(123)
        rng2 = np.random.default_rng(123)
        pts1 = generate_random_control_points(10, 0.1, 0.01, (1,1,1), rng1)
        pts2 = generate_random_control_points(10, 0.1, 0.01, (1,1,1), rng2)
        np.testing.assert_array_equal(pts1, pts2)
