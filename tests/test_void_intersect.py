# SPDX-FileCopyrightText: 2026 Devine Ngouloubi <exauce-devine.ngouloubi@unicaen.fr>
# SPDX-License-Identifier: AGPL-3.0-or-later

"""! @brief Quick Doxygen documentation for test module."""
"""
Unit tests for Void.intersect_fiber (corrected implementation).
"""
import pytest
import numpy as np
from core.void import Void
from tests.conftest import FakeFiber


class TestIntersectFiber:

    def test_void_on_fiber_path(self):
        """A void centred on the centerline must intersect."""
        fiber = FakeFiber(1, [[0,0,0],[1,0,0],[2,0,0]], radius=0.05)
        void = Void(id=1, center=np.array([1.0, 0.0, 0.0]), radius=0.01)
        assert void.intersect_fiber(fiber)

    def test_void_far_away(self):
        """A distant void must not intersect."""
        fiber = FakeFiber(1, [[0,0,0],[1,0,0]], radius=0.05)
        void = Void(id=1, center=np.array([5.0, 5.0, 5.0]), radius=0.1)
        assert not void.intersect_fiber(fiber)

    def test_void_touching_fiber_surface(self):
        """Un void dont le bord touche la surface de la fibre."""
        fiber = FakeFiber(1, [[0,0,0],[2,0,0]], radius=0.1)
        # Distance from the void centre to the fiber = 0.2, r_void + r_fiber = 0.1 + 0.1 = 0.2
        void = Void(id=1, center=np.array([1.0, 0.2, 0.0]), radius=0.1)
        assert void.intersect_fiber(fiber)

    def test_void_just_outside(self):
        """A void just beyond the reach."""
        fiber = FakeFiber(1, [[0,0,0],[2,0,0]], radius=0.1)
        void = Void(id=1, center=np.array([1.0, 0.25, 0.0]), radius=0.1)
        assert not void.intersect_fiber(fiber)

    def test_void_near_endpoint(self):
        """A void near one end of the fiber."""
        fiber = FakeFiber(1, [[0,0,0],[1,0,0]], radius=0.05)
        void = Void(id=1, center=np.array([0.0, 0.04, 0.0]), radius=0.02)
        assert void.intersect_fiber(fiber)

    def test_void_perpendicular_to_segment(self):
        """Un void perpendiculaire au milieu d'un segment."""
        fiber = FakeFiber(1, [[0,0,0],[2,0,0]], radius=0.1)
        void = Void(id=1, center=np.array([1.0, 0.15, 0.0]), radius=0.1)
        assert void.intersect_fiber(fiber)

    def test_zero_radius_void(self):
        """Un void de rayon 0 sur la fibre."""
        fiber = FakeFiber(1, [[0,0,0],[1,0,0]], radius=0.1)
        void = Void(id=1, center=np.array([0.5, 0.05, 0.0]), radius=0.0)
        assert void.intersect_fiber(fiber)
