"""
Tests unitaires pour validation/topology.py — TopologyValidator.
"""
import pytest
import numpy as np
from unittest.mock import MagicMock
from validation.topology import TopologyValidator


def _make_fiber(fid, pts, radius):
    f = MagicMock()
    f.id = fid
    f.centerline = np.ascontiguousarray(pts, dtype=np.float64)
    f.radius = radius
    return f


class TestCheckClearance:

    def test_no_intersection_parallel(self):
        """Deux fibres parallèles bien séparées : gap > 0."""
        f1 = _make_fiber(1, [[0, 0, 0], [1, 0, 0]], 0.05)
        f2 = _make_fiber(2, [[0, 0.5, 0], [1, 0.5, 0]], 0.05)
        validator = TopologyValidator((2.0, 2.0, 2.0))
        gap = validator.check_clearance([f1, f2])
        assert gap == pytest.approx(0.4, abs=0.01)

    def test_intersection_detected(self):
        """Deux fibres qui se croisent : gap < 0."""
        f1 = _make_fiber(1, [[0, 0.5, 0.5], [1, 0.5, 0.5]], 0.1)
        f2 = _make_fiber(2, [[0.5, 0, 0.5], [0.5, 1, 0.5]], 0.1)
        validator = TopologyValidator((2.0, 2.0, 2.0))
        gap = validator.check_clearance([f1, f2])
        assert gap < 0

    def test_single_fiber(self):
        """Une seule fibre : gap infini."""
        f1 = _make_fiber(1, [[0, 0, 0], [1, 0, 0]], 0.05)
        validator = TopologyValidator((2.0, 2.0, 2.0))
        gap = validator.check_clearance([f1])
        assert gap == float('inf')


class TestGapDistribution:

    def test_returns_dict_keys(self):
        f1 = _make_fiber(1, [[0, 0, 0], [1, 0, 0]], 0.05)
        f2 = _make_fiber(2, [[0, 0.5, 0], [1, 0.5, 0]], 0.05)
        validator = TopologyValidator((2.0, 2.0, 2.0))
        result = validator.compute_gap_distribution([f1, f2])
        assert 'min_gap' in result
        assert 'mean_gap' in result

    def test_empty_fibers(self):
        validator = TopologyValidator((1.0, 1.0, 1.0))
        result = validator.compute_gap_distribution([])
        assert result == {}
