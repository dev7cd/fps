# SPDX-FileCopyrightText: 2026 Devine Ngouloubi <exauce-devine.ngouloubi@unicaen.fr>
# SPDX-License-Identifier: AGPL-3.0-or-later

"""! @brief Quick Doxygen documentation for test module."""
"""
Unit tests for generation/optimizer_dynamic.py (collective rearrangement).
"""
import pytest
import numpy as np

from core.config import FiberPackingConfig
from core.fiber import Fiber
from collision.detector import CollisionDetector
from collision.detector_math import min_dist_periodic_images_numba
from generation.optimizer_dynamic import DynamicOptimizer

RADIUS = 0.02
CLEARANCE = 0.005


def _config(**overrides):
    params = dict(
        box_dims=(1.0, 1.0, 1.0),
        target_volume_fraction=0.02,
        fiber_radius=RADIUS,
        min_clearance=CLEARANCE,
        seed=11,
        fiber_section_type='circular',
        section_parameters={'major_radius': RADIUS, 'minor_radius': RADIUS, 'exponent': 2.0},
        step_length_mean=0.05,
        min_control_points=8,
        max_control_points=8,
    )
    params.update(overrides)
    return FiberPackingConfig(**params)


def _straight_fiber(cfg, fiber_id, origin, direction=(1.0, 0.0, 0.0)):
    """Fiber whose control points are evenly spaced along a direction."""
    step = cfg.step_length_mean
    pts = np.array([np.asarray(origin, dtype=float) + k * step * np.asarray(direction, dtype=float)
                    for k in range(cfg.max_control_points)])
    return Fiber(fiber_id, pts, RADIUS, vars(cfg))


def _optimizer(cfg):
    return DynamicOptimizer(cfg, CollisionDetector(cfg))


class TestRelaxation:
    """The relaxation must turn an overlapping pair into an admissible one."""

    def test_overlapping_pair_is_separated(self):
        cfg = _config()
        opt = _optimizer(cfg)

        # Two parallel fibers far closer than radii + clearance
        f1 = _straight_fiber(cfg, 1, (0.2, 0.5, 0.5))
        f2 = _straight_fiber(cfg, 2, (0.2, 0.5 + 0.5 * RADIUS, 0.5))
        fibers = [f1, f2]

        required = 2 * RADIUS + CLEARANCE
        before = min_dist_periodic_images_numba(f1.centerline, f2.centerline, opt.dims)
        assert before < required, "the fixture must start overlapping"

        resolved, residual = opt._relax(fibers)

        after = min_dist_periodic_images_numba(f1.centerline, f2.centerline, opt.dims)
        assert resolved is True
        assert residual == {}
        assert after >= required

    def test_admissible_configuration_is_left_alone(self):
        cfg = _config()
        opt = _optimizer(cfg)
        f1 = _straight_fiber(cfg, 1, (0.2, 0.2, 0.5))
        f2 = _straight_fiber(cfg, 2, (0.2, 0.8, 0.5))
        before = f1.control_points.copy()

        resolved, _ = opt._relax([f1, f2])

        assert resolved is True
        np.testing.assert_allclose(f1.control_points, before)

    def test_self_image_overlap_is_resolved(self):
        """A fiber crossing the box must not end up touching its own image."""
        cfg = _config(box_dims=(0.35, 1.0, 1.0))
        opt = _optimizer(cfg)
        # 8 points spaced by 0.05 span 0.35: the two ends meet through the x face
        fiber = _straight_fiber(cfg, 1, (0.0, 0.5, 0.5))

        resolved, _ = opt._relax([fiber])

        gap = min_dist_periodic_images_numba(fiber.centerline, fiber.centerline,
                                             opt.dims, True)
        assert resolved is True
        assert gap >= 2 * RADIUS + CLEARANCE


class TestContactScan:
    """The cached scan must agree with the exhaustive one."""

    def test_cache_matches_full_scan(self):
        cfg = _config()
        opt = _optimizer(cfg)
        fibers = [_straight_fiber(cfg, 1, (0.2, 0.5, 0.5)),
                  _straight_fiber(cfg, 2, (0.2, 0.5 + 0.5 * RADIUS, 0.5)),
                  _straight_fiber(cfg, 3, (0.2, 0.1, 0.1))]

        plain = opt._collect_contacts(fibers)

        cache, dirty = {}, set(range(len(fibers)))
        opt._collect_contacts(fibers, 0.0, cache, dirty)
        # Nothing moved: the second pass must be served from the cache alone
        cached = opt._collect_contacts(fibers, 0.0, cache, set())

        assert [c[:2] for c in plain] == [c[:2] for c in cached]
        assert [pytest.approx(c[4]) for c in plain] == [c[4] for c in cached]

    def test_distant_fibers_produce_no_contact(self):
        cfg = _config()
        opt = _optimizer(cfg)
        fibers = [_straight_fiber(cfg, 1, (0.1, 0.1, 0.1)),
                  _straight_fiber(cfg, 2, (0.1, 0.5, 0.5))]
        assert opt._collect_contacts(fibers) == []

    def test_margin_widens_the_scan(self):
        cfg = _config()
        opt = _optimizer(cfg)
        gap = 0.5 * CLEARANCE
        fibers = [_straight_fiber(cfg, 1, (0.2, 0.5, 0.5)),
                  _straight_fiber(cfg, 2, (0.2, 0.5 + 2 * RADIUS + CLEARANCE + gap, 0.5))]
        assert opt._collect_contacts(fibers, 0.0) == []
        assert len(opt._collect_contacts(fibers, gap * 2)) == 1


class TestChainGeometry:
    """Pushing fibers around must not rewrite their length statistics."""

    def test_step_length_is_restored(self):
        cfg = _config()
        opt = _optimizer(cfg)
        fiber = _straight_fiber(cfg, 1, (0.2, 0.5, 0.5))

        # Stretch the chain well beyond the generation step
        fiber.control_points = fiber.control_points * 1.5
        opt._restore_chain(fiber)

        steps = np.linalg.norm(np.diff(fiber.control_points, axis=0), axis=1)
        assert steps == pytest.approx(cfg.step_length_mean, rel=0.05)

    def test_relaxation_preserves_fiber_length(self):
        cfg = _config()
        opt = _optimizer(cfg)
        f1 = _straight_fiber(cfg, 1, (0.2, 0.5, 0.5))
        f2 = _straight_fiber(cfg, 2, (0.2, 0.5 + 0.5 * RADIUS, 0.5))

        def curvilinear(f):
            return float(np.sum(np.linalg.norm(np.diff(f.centerline, axis=0), axis=1)))

        before = curvilinear(f1)
        opt._relax([f1, f2])
        assert curvilinear(f1) == pytest.approx(before, rel=0.05)


class TestWeights:
    """The push is local: it peaks on the contact and fades along the fiber."""

    def test_peak_sits_on_the_contact(self):
        cfg = _config()
        opt = _optimizer(cfg)
        fiber = _straight_fiber(cfg, 1, (0.2, 0.5, 0.5))
        n_seg = len(fiber.centerline) - 1

        weights = opt._weights(fiber, 0.0)
        assert np.argmax(weights) == 0

        weights = opt._weights(fiber, float(n_seg))
        assert np.argmax(weights) == len(fiber.control_points) - 1

        weights = opt._weights(fiber, n_seg / 2.0)
        assert weights.max() == pytest.approx(1.0, abs=0.05)
        assert weights[0] < weights[len(weights) // 2]
