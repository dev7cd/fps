# SPDX-FileCopyrightText: 2026 Devine Ngouloubi <exauce-devine.ngouloubi@unicaen.fr>
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Regression test: the CSAW backtracking must terminate for straight fibers.
"""
import numpy as np

from core.config import FiberPackingConfig
from core.fiber import Fiber
from collision.detector import CollisionDetector
from generation.generator import FiberGenerator


def test_straight_fiber_blocked_midway_terminates():
    """With max_curvature_angle = 0 every proposal is identical, so a walk
    blocked after a few steps used to pop and re-add the same point forever."""
    radius = 0.01
    config = FiberPackingConfig(
        box_dims=(1.0, 1.0, 1.0), seed=0, fiber_radius=radius,
        fiber_section_type="circular", section_parameters={"radius": radius},
        min_control_points=11, max_control_points=11, step_length_mean=0.05,
        generation_parameters={"orientation_bias": "free", "bias_vectors": None,
                               "bias_strength": 0.0, "max_curvature_angle": 0.0,
                               "max_packing_attempts": 10},
        enable_optimizer=False, generate_porosity=False)
    detector = CollisionDetector(config)
    generator = FiberGenerator(config, detector)

    # Start at the centre heading +x; a wall of fibers along y blocks x = 0.65.
    start = np.array([0.5, 0.5, 0.5])
    generator._find_start_point = lambda r: start.copy()
    generator._initial_direction = lambda: np.array([1.0, 0.0, 0.0])
    wall = Fiber(1, np.array([[0.65, 0.0, 0.5], [0.65, 1.0, 0.5]]), 0.05, vars(config))
    detector.add_fibers_group([wall])

    assert generator.generate_fiber(2) is None
