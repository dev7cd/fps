"""
Tests unitaires pour collision/detector_math.py.
Vérifie les fonctions numba de distance segment-segment et de détection de collision.
"""
import pytest
import numpy as np
from collision.detector_math import (
    _segment_segment_dist_sq,
    check_collision_numba,
    min_dist_segments_numba,
    min_dist_periodic_segments_numba,
)


class TestSegmentSegmentDistSq:
    """Tests de la distance² minimale entre deux segments 3D."""

    def test_identical_segments(self):
        """Deux segments identiques : distance = 0."""
        d = _segment_segment_dist_sq(
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
        )
        assert d == pytest.approx(0.0, abs=1e-10)

    def test_parallel_segments_unit_apart(self):
        """Deux segments parallèles séparés de 1.0 en Y."""
        d = _segment_segment_dist_sq(
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 1.0, 1.0, 0.0,
        )
        assert d == pytest.approx(1.0, abs=1e-10)

    def test_perpendicular_crossing(self):
        """Deux segments perpendiculaires qui se croisent au même point."""
        d = _segment_segment_dist_sq(
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            0.5, -0.5, 0.0, 0.5, 0.5, 0.0,
        )
        assert d == pytest.approx(0.0, abs=1e-10)

    def test_perpendicular_offset_z(self):
        """Deux segments perpendiculaires décalés en Z de 2.0."""
        d = _segment_segment_dist_sq(
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            0.5, -0.5, 2.0, 0.5, 0.5, 2.0,
        )
        assert d == pytest.approx(4.0, abs=1e-10)

    def test_point_to_point(self):
        """Deux segments dégénérés en points."""
        d = _segment_segment_dist_sq(
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            3.0, 4.0, 0.0, 3.0, 4.0, 0.0,
        )
        assert d == pytest.approx(25.0, abs=1e-10)

    def test_point_to_segment(self):
        """Un point et un segment : distance = projection perpendiculaire."""
        # Point (0,1,0), segment de (0,0,0) à (1,0,0)
        # Distance minimale = 1.0 (projection sur le segment)
        d = _segment_segment_dist_sq(
            0.0, 1.0, 0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
        )
        assert d == pytest.approx(1.0, abs=1e-10)

    def test_non_overlapping_collinear(self):
        """Deux segments collinéaires non-chevauchants."""
        d = _segment_segment_dist_sq(
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            3.0, 0.0, 0.0, 4.0, 0.0, 0.0,
        )
        assert d == pytest.approx(4.0, abs=1e-10)

    def test_t_shape(self):
        """Segments en T : un horizontal, un vertical partant du milieu."""
        d = _segment_segment_dist_sq(
            0.0, 0.0, 0.0, 2.0, 0.0, 0.0,
            1.0, 0.0, 0.0, 1.0, 1.0, 0.0,
        )
        assert d == pytest.approx(0.0, abs=1e-10)


class TestCheckCollisionNumba:
    """Tests de la détection de collision entre polylignes."""

    def test_collision_crossing_fibers(self):
        """Deux fibres qui se croisent doivent être en collision."""
        pts1 = np.array([[0.0, 0.5, 0.5], [1.0, 0.5, 0.5]], dtype=np.float64)
        pts2 = np.array([[0.5, 0.0, 0.5], [0.5, 1.0, 0.5]], dtype=np.float64)
        assert check_collision_numba(pts1, pts2, 0.1, 0.1) is True

    def test_no_collision_distant_fibers(self):
        """Deux fibres éloignées ne doivent pas être en collision."""
        pts1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
        pts2 = np.array([[0.0, 5.0, 0.0], [1.0, 5.0, 0.0]], dtype=np.float64)
        assert check_collision_numba(pts1, pts2, 0.1, 0.1) is False

    def test_collision_depends_on_radius(self):
        """Deux fibres parallèles séparées de 0.5 : collision si r1+r2 > 0.5."""
        pts1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
        pts2 = np.array([[0.0, 0.5, 0.0], [1.0, 0.5, 0.0]], dtype=np.float64)
        # r1 + r2 = 0.2 < 0.5 → pas de collision
        assert check_collision_numba(pts1, pts2, 0.1, 0.1) is False
        # r1 + r2 = 0.6 > 0.5 → collision
        assert check_collision_numba(pts1, pts2, 0.3, 0.3) is True

    def test_multipoint_polyline(self):
        """Polylignes avec plusieurs segments."""
        pts1 = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ], dtype=np.float64)
        pts2 = np.array([
            [1.0, 0.3, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 2.0, 0.0],
        ], dtype=np.float64)
        # Distance minimale = 0.3, r1+r2 = 0.4 > 0.3 → collision
        assert check_collision_numba(pts1, pts2, 0.2, 0.2) is True

    def test_single_point_polyline(self):
        """Polyligne dégénérée en un seul point : pas de segment → pas de collision."""
        pts1 = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        pts2 = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        # n1 = 0 segments, la boucle ne s'exécute pas
        assert check_collision_numba(pts1, pts2, 1.0, 1.0) is False


class TestMinDistSegmentsNumba:
    """Tests de la distance minimale entre polylignes."""

    def test_crossing_polylines(self):
        pts1 = np.array([[0.0, 0.5, 0.0], [1.0, 0.5, 0.0]], dtype=np.float64)
        pts2 = np.array([[0.5, 0.0, 0.0], [0.5, 1.0, 0.0]], dtype=np.float64)
        d = min_dist_segments_numba(pts1, pts2)
        assert d == pytest.approx(0.0, abs=1e-6)

    def test_parallel_polylines(self):
        pts1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
        pts2 = np.array([[0.0, 3.0, 0.0], [1.0, 3.0, 0.0]], dtype=np.float64)
        d = min_dist_segments_numba(pts1, pts2)
        assert d == pytest.approx(3.0, abs=1e-6)

    def test_3d_distance(self):
        pts1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
        pts2 = np.array([[0.0, 0.0, 5.0], [1.0, 0.0, 5.0]], dtype=np.float64)
        d = min_dist_segments_numba(pts1, pts2)
        assert d == pytest.approx(5.0, abs=1e-6)


class TestMinDistPeriodicSegmentsNumba:
    """Tests de la distance minimale avec conditions périodiques."""

    def test_no_wrap_needed(self):
        """Fibres proches sans besoin de correction périodique."""
        pts1 = np.array([[0.5, 0.5, 0.5], [0.6, 0.5, 0.5]], dtype=np.float64)
        pts2 = np.array([[0.5, 0.7, 0.5], [0.6, 0.7, 0.5]], dtype=np.float64)
        dims = np.array([1.0, 1.0, 1.0])
        d = min_dist_periodic_segments_numba(pts1, pts2, dims)
        assert d == pytest.approx(0.2, abs=1e-6)

    def test_wrap_around(self):
        """Fibres aux bords opposés : la distance périodique doit être courte."""
        pts1 = np.array([[0.05, 0.5, 0.5], [0.1, 0.5, 0.5]], dtype=np.float64)
        pts2 = np.array([[0.95, 0.5, 0.5], [0.99, 0.5, 0.5]], dtype=np.float64)
        dims = np.array([1.0, 1.0, 1.0])
        d = min_dist_periodic_segments_numba(pts1, pts2, dims)
        # Sans périodicité : ~0.85. Avec : ~0.06
        assert d < 0.15

    def test_symmetric(self):
        """La distance périodique doit être symétrique."""
        pts1 = np.array([[0.1, 0.5, 0.5], [0.2, 0.5, 0.5]], dtype=np.float64)
        pts2 = np.array([[0.8, 0.5, 0.5], [0.9, 0.5, 0.5]], dtype=np.float64)
        dims = np.array([1.0, 1.0, 1.0])
        d1 = min_dist_periodic_segments_numba(pts1, pts2, dims)
        d2 = min_dist_periodic_segments_numba(pts2, pts1, dims)
        assert d1 == pytest.approx(d2, abs=1e-6)
