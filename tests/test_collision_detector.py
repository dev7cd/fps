# SPDX-FileCopyrightText: 2026 Devine Ngouloubi <exauce-devine.ngouloubi@unicaen.fr>
# SPDX-License-Identifier: AGPL-3.0-or-later

"""! @brief Quick Doxygen documentation for test module."""
"""
Tests unitaires pour collision/detector.py — prise en compte de --clearance.
"""
import pytest
import numpy as np

from core.config import FiberPackingConfig
from core.fiber import Fiber
from collision.detector import CollisionDetector

RADIUS = 0.02
CLEARANCE = 0.005
CONTROL_POINTS = np.array([[0.1, 0.5, 0.5], [1.5, 0.5, 0.5], [2.9, 0.5, 0.5]])


def _make_detector(clearance):
    cfg = FiberPackingConfig(box_dims=(3.0, 1.0, 1.0), fiber_radius=RADIUS,
                             min_clearance=clearance)
    detector = CollisionDetector(cfg)
    reference = Fiber(1, CONTROL_POINTS, RADIUS, vars(cfg))
    detector.add_fibers_group([reference])
    return cfg, detector, reference


def _parallel_fiber(cfg, surface_gap, fiber_id=2):
    """Straight fiber parallel to the reference one, separated by `surface_gap`."""
    pts = CONTROL_POINTS.copy()
    pts[:, 1] += 2 * RADIUS + surface_gap
    return Fiber(fiber_id, pts, RADIUS, vars(cfg))


class TestDetectorClearance:
    """The configured min_clearance must be enforced, not only non-intersection."""

    def test_clearance_stored(self):
        _, detector, _ref = _make_detector(CLEARANCE)
        assert detector.min_clearance == pytest.approx(CLEARANCE)

    def test_gap_below_clearance_is_rejected(self):
        cfg, detector, reference = _make_detector(CLEARANCE)
        candidate = _parallel_fiber(cfg, CLEARANCE * 0.6)
        assert detector.check_collision_fine(candidate, reference) is True
        assert detector.is_group_valid([candidate]) is False

    def test_gap_above_clearance_is_accepted(self):
        cfg, detector, _ref = _make_detector(CLEARANCE)
        candidate = _parallel_fiber(cfg, CLEARANCE * 1.2)
        assert detector.is_group_valid([candidate]) is True

    def test_same_gap_accepted_without_clearance(self):
        """Without clearance the tight candidate stays valid (non-intersecting)."""
        cfg, detector, _ref = _make_detector(0.0)
        candidate = _parallel_fiber(cfg, CLEARANCE * 0.6)
        assert detector.is_group_valid([candidate]) is True

    def test_point_and_segment_honour_clearance(self):
        cfg, detector, _ref = _make_detector(CLEARANCE)
        # Point centred at a distance leaving a surface gap smaller than the clearance
        p = CONTROL_POINTS[1].copy()
        p[1] += 2 * RADIUS + CLEARANCE * 0.5
        assert detector.is_point_free(p, RADIUS) is False

        q = CONTROL_POINTS[1].copy()
        q[1] += 2 * RADIUS + CLEARANCE * 2.0
        assert detector.is_point_free(q, RADIUS) is True

        a, b = p.copy(), p.copy()
        a[0] -= 0.2
        b[0] += 0.2
        assert detector.is_segment_free(a, b, RADIUS) is False
