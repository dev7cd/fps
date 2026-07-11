"""! @brief Quick Doxygen documentation for test module."""
"""
Tests unitaires pour validation/statistics.py — Descripteurs spatiaux.
We use known distributions (regular grid, CSR) to validate.
"""
import pytest
import numpy as np
from unittest.mock import MagicMock
from validation.statistics import (
    _extract_2d_centroids,
    nearest_neighbor_distance,
    ripley_k_function,
    pair_correlation_function,
    voronoi_statistics,
)


def _make_fake_fibers_grid(nx=5, ny=5, spacing=0.2, box_size=1.0):
    """Creates dummy fibers on a regular 2D grid (XY plane)."""
    fibers = []
    for i in range(nx):
        for j in range(ny):
            x = (i + 0.5) * spacing
            y = (j + 0.5) * spacing
            f = MagicMock()
            f.centerline = np.array([[x, y, 0.0], [x, y, 1.0]])
            f.radius = 0.01
            fibers.append(f)
    return fibers


def _make_fake_fibers_random(n=100, box_size=1.0, seed=42):
    """Creates dummy fibers with random positions (CSR)."""
    rng = np.random.default_rng(seed)
    fibers = []
    for _ in range(n):
        x, y = rng.uniform(0, box_size, 2)
        f = MagicMock()
        f.centerline = np.array([[x, y, 0.0], [x, y, 1.0]])
        f.radius = 0.01
        fibers.append(f)
    return fibers


class TestExtract2DCentroids:

    def test_xy_projection(self):
        fibers = _make_fake_fibers_grid(3, 3)
        centroids = _extract_2d_centroids(fibers, 'xy')
        assert centroids.shape == (9, 2)

    def test_xz_projection(self):
        f = MagicMock()
        f.centerline = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 5.0]])
        centroids = _extract_2d_centroids([f], 'xz')
        assert centroids[0, 0] == pytest.approx(1.0)
        assert centroids[0, 1] == pytest.approx(4.0)  # mean(3, 5)


class TestNearestNeighborDistance:

    def test_regular_grid_nnd(self):
        """On a regular grid, the NND must be constant and equal to the spacing."""
        fibers = _make_fake_fibers_grid(5, 5, spacing=0.2)
        result = nearest_neighbor_distance(fibers, (1.0, 1.0, 1.0))
        assert result['nnd_mean'] == pytest.approx(0.2, abs=0.02)
        assert result['nnd_std'] < 0.02  # Very low dispersion

    def test_single_fiber(self):
        fibers = _make_fake_fibers_grid(1, 1)
        result = nearest_neighbor_distance(fibers, (1.0, 1.0, 1.0))
        assert result['nnd_mean'] == 0.0

    def test_nnd_positive(self):
        fibers = _make_fake_fibers_random(50)
        result = nearest_neighbor_distance(fibers, (1.0, 1.0, 1.0))
        assert result['nnd_mean'] > 0


class TestRipleyKFunction:

    def test_csr_follows_poisson(self):
        """For a CSR process, K(h) must be close to pi*h^2."""
        fibers = _make_fake_fibers_random(200, seed=0)
        result = ripley_k_function(fibers, (1.0, 1.0, 1.0))
        K_h = np.array(result['K_h'])
        K_poisson = np.array(result['K_poisson'])
        h = np.array(result['h_values'])
        # On compare sur la partie centrale (pas les bords)
        mask = (h > 0.05) & (h < 0.3)
        if np.any(mask):
            ratio = K_h[mask] / np.maximum(K_poisson[mask], 1e-10)
            assert np.mean(ratio) == pytest.approx(1.0, abs=0.3)

    def test_empty_fibers(self):
        result = ripley_k_function([], (1.0, 1.0, 1.0))
        assert result['K_h'] == []


class TestPairCorrelationFunction:

    def test_csr_g_near_one(self):
        """Pour un CSR, g(r) doit osciller autour de 1."""
        fibers = _make_fake_fibers_random(300, seed=7)
        result = pair_correlation_function(fibers, (1.0, 1.0, 1.0))
        g_r = np.array(result['g_r'])
        # Moyenne de g(r) sur la zone centrale
        mid = len(g_r) // 4
        assert np.mean(g_r[mid:3*mid]) == pytest.approx(1.0, abs=0.4)

    def test_empty(self):
        result = pair_correlation_function([], (1.0, 1.0, 1.0))
        assert result['g_r'] == []


class TestVoronoiStatistics:

    def test_regular_grid_low_cv(self):
        """A regular grid must have a Voronoi CV close to 0."""
        fibers = _make_fake_fibers_grid(5, 5, spacing=0.2)
        result = voronoi_statistics(fibers, (1.0, 1.0, 1.0))
        assert result['areas_cv'] < 0.15

    def test_random_cv_near_poisson(self):
        """Un processus CSR doit avoir un CV de Voronoi proche de 0.53."""
        fibers = _make_fake_fibers_random(500, seed=99)
        result = voronoi_statistics(fibers, (1.0, 1.0, 1.0))
        assert result['areas_cv'] == pytest.approx(0.53, abs=0.15)

    def test_too_few_fibers(self):
        fibers = _make_fake_fibers_grid(1, 1)
        result = voronoi_statistics(fibers, (1.0, 1.0, 1.0))
        assert result['areas_mean'] == 0.0
