"""
generation/optimizer_dynamic.py
Dynamic optimiser for RVE densification through agitation and compression.
Implements Monte-Carlo-type methods to increase the volume fraction.
"""

import numpy as np
import logging
import copy
from typing import List, Tuple, Set, Dict
from collision.detector import CollisionDetector
from generation.periodicity import PeriodicManager
from dataclasses import dataclass, asdict


## @var logger
#  @brief Logger for the dynamic optimisation module.
logger = logging.getLogger(__name__)


class DynamicOptimizer:
    """!
    @class DynamicOptimizer
    @brief Handles the post-generation densification of the fibers.

    This class increases the volume fraction (Vf) by iteratively moving the
    existing fibers to create space for new injections.
    """
    def __init__(self, config, detector):
        """!
        @brief Initialises the optimiser with the configuration and the collision detector.
        @param config FiberPackingConfig Problem configuration.
        @param detector CollisionDetector Collision detector.
        """
        ## @var config
        #  @brief Global configuration.
        self.config = config
        ## @var detector
        #  @brief Collision detector.
        self.detector = detector
        ## @var rng
        #  @brief Random number generator.
        self.rng = np.random.default_rng(config.seed)

    def optimize_and_fill(self, fibers: List):
        """!
        @brief Main densification cycle (compression and shaking).
        @param fibers List List of initial fibers to densify.
        @return List Updated and completed list of fibers.
        """
        target_vf = self.config.target_volume_fraction
        # Use the real-volume method for better accuracy
        current_vf = self._calculate_vf(fibers)

        logger.info(f"Densification start. Current Vf: {current_vf:.3f}")

        iteration = 0
        max_iterations = 100
        while current_vf < target_vf and iteration < max_iterations:
            iteration += 1

            # 1. Agitation (jitter)
            self._apply_global_jitter(fibers)

            # 2. Compression towards the centre
            self._apply_centripetal_compression(fibers, intensity=0.01)

            # 3. Resolution of residual collisions
            self._resolve_all_collisions(fibers)

            # 4. Injection of new fibers
            added_fibers = self._inject_additional_fibers(fibers)
            if len(added_fibers) > 0:
                fibers.extend(added_fibers)
                logger.info(f"Injected {len(added_fibers)} new fibers after compaction.")

            current_vf = self._calculate_vf(fibers)
            logger.info(f"Iteration {iteration}: Vf = {current_vf:.4f}")

        return fibers

    def _resolve_all_collisions(self, fibers):
        """!
        @brief Identifies and attempts to resolve collisions via micro-displacements.
        @param fibers List of fibers to process.
        """
        for _ in range(3): # Resolution attempts
            collisions = self._detect_collision_pairs(fibers)
            if not collisions:
                break

            for f1, f2 in collisions:
                # nudge moves f1 away from f2
                self._nudge_fibers(f1, f2, fibers)

    def _detect_collision_pairs(self, fibers) -> List[Tuple]:
        """!
        @brief Detects the pairs of fibers (or ghosts) that intersect.
        @return List of tuples (fiber_1, fiber_2_or_ghost).
        """
        colliding_pairs = []
        seen_collisions = set() # Avoid processing both (A,B) and (B,A)

        for f1 in fibers:
            neighbors = self.detector.grid.query_neighbors(f1.bbox, exclude_id=f1.id)
            for nb in neighbors:
                # nb may be a ghost; compare the parent_id values
                pair_id = tuple(sorted((f1.id, nb.parent_id)))
                if pair_id in seen_collisions:
                    continue

                if self.detector.check_collision_fine(f1, nb):
                    # nb is the physical object in collision; add it
                    colliding_pairs.append((f1, nb))
                    seen_collisions.add(pair_id)
        return colliding_pairs

    def _nudge_fibers(self, f1, target, all_fibers):
        """!
        @brief Applies a repulsion force to f1 relative to a target.
        @param f1 The fiber to move.
        @param target The fiber or ghost causing the collision.
        @param all_fibers Global fiber context.
        """
        c1 = np.mean(f1.centerline, axis=0)
        c2 = np.mean(target.centerline, axis=0) # Works for both root and ghost

        vec = c1 - c2
        dist = np.linalg.norm(vec)
        if dist == 0:
            vec = self.rng.standard_normal(3)

        push = (vec / np.linalg.norm(vec)) * (f1.radius * 0.05)

        old_pts = f1.control_points.copy()
        f1.control_points += push
        f1.refresh_geometry()

        # Validation: f1 must not be in collision at its new position
        if self._has_any_collision(f1):
            # Move failed
            f1.control_points = old_pts
            f1.refresh_geometry()

    def _has_any_collision(self, fiber):
        """!
        @brief Checks whether a given fiber collides with the rest of the domain.
        @return True if a collision is detected.
        """
        neighbors = self.detector.grid.query_neighbors(fiber.bbox, exclude_id=fiber.id)
        for nb in neighbors:
            if self.detector.check_collision_fine(fiber, nb):
                return True
        return False

    def _calculate_vf(self, fibers):
        """!
        @brief Computes the current real volume fraction.
        """
        if not fibers: return 0.0
        total_vol = sum(f.get_real_volume() for f in fibers)
        return total_vol / np.prod(self.config.box_dims)

    def _apply_global_jitter(self, fibers):
        """!
        @brief Applies a slight random displacement to all fibers (thermal agitation).
        @param fibers List of fibers to shake.
        """
        for f in fibers:
            old_pts = f.control_points.copy()
            shift = self.rng.uniform(-0.01, 0.01, 3) * f.radius
            f.control_points += shift
            f.refresh_geometry()
            if self._has_any_collision(f):
                f.control_points = old_pts
                f.refresh_geometry()

    def _apply_centripetal_compression(self, fibers, intensity):
        """!
        @brief Moves the fibers towards the domain centre to free space near the walls.
        @param fibers List of fibers.
        @param intensity Compression strength (0.0 to 1.0).
        """
        center = np.array(self.config.box_dims) / 2
        for f in fibers:
            centroid = np.mean(f.control_points, axis=0)
            push = (center - centroid) * intensity
            old_pts = f.control_points.copy()
            f.control_points += push
            f.refresh_geometry()
            if self._has_any_collision(f):
                f.control_points = old_pts
                f.refresh_geometry()

    def _inject_additional_fibers(self, fibers):
        """!
        @brief Attempts to insert new fibers into the freed spaces.
        @param fibers Current list of fibers.
        @return List of the newly accepted fibers.
        """
        from generation.generator import FiberGenerator
        # Note: the detector already has the previous fibers indexed
        gen = FiberGenerator(self.config, self.detector)
        added = []
        for i in range(10): # Attempt 10 injections
            new_f = gen.generate_fiber(len(fibers) + i + 1)
            if new_f:
                added.append(new_f)
                # Immediate addition to the grid for the following ones
                self.detector.add_fibers_group([new_f])
        return added
