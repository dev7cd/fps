"""
generation/porosity_gen.py
Optimised porosity generator.
"""

import numpy as np
import logging
from typing import List, Tuple, Optional
from core.config import FiberPackingConfig
from core.void import Void
from core.fiber import Fiber

## @var logger
#  @brief Logger for the porosity generation module.
logger = logging.getLogger(__name__)

class PorosityGenerator:
    """!
    @class PorosityGenerator
    @brief Generates spherical void inclusions within the RVE.
    @details Implements an optimized RSA (Random Sequential Adsorption) algorithm
    with periodic boundary conditions and collision detection against fibers.
    """

    def __init__(self, config: FiberPackingConfig):
        """!
        @brief Initializes the porosity generator.
        @param config Configuration object containing domain and porosity parameters.
        """
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.box_dims = np.array(config.box_dims)

    def generate_voids(self, existing_fibers: List[Fiber]) -> List[Void]:
        """!
        @brief Main generation loop using an optimized RSA algorithm.
        @param existing_fibers List of fibers already present in the domain.
        @return A list of generated Void objects.
        """
        if not self.config.generate_porosity:
            return []

        target_vf = self.config.target_void_fraction
        box_vol = self.config.box_volume
        target_vol = box_vol * target_vf

        current_vol = 0.0
        voids: List[Void] = []
        void_id_counter = 0

        logger.info(f"Porosity generation: target {target_vf*100:.1f}% ({target_vol:.4e} m3)")

        # Configuration
        mean_r = self.config.void_radius_mean
        std_r = self.config.void_radius_std

        # Pre-compute the fiber data for optimisation
        fiber_data = []
        for fib in existing_fibers:
            # Use the fiber centerline directly
            centerline = fib.centerline
            radius = fib.radius

            # Compute BBox for optimisation
            min_pt = np.min(centerline, axis=0) - radius
            max_pt = np.max(centerline, axis=0) + radius

            fiber_data.append({
                'centerline': centerline,
                'radius': radius,
                'bbox_min': min_pt,
                'bbox_max': max_pt
            })

        # Algorithm parameters
        max_attempts = 30000
        attempts = 0
        consecutive_failures = 0

        while current_vol < target_vol and attempts < max_attempts:
            attempts += 1

            # 1. Adaptive radius generation
            if consecutive_failures > 50:
                # Progressive radius reduction after failures
                radius_factor = max(0.3, 1.0 - consecutive_failures / 200.0)
                radius = self.rng.normal(mean_r * radius_factor, std_r * radius_factor)
            else:
                radius = self.rng.normal(mean_r, std_r)

            radius = max(radius, mean_r * 0.2)  # Lower bound

            # 2. Position generation
            center = self.rng.uniform(radius, self.box_dims - radius)

            # 3. Optimised periodicity handling
            candidates = self._create_periodic_images_fast(center, radius, void_id_counter)

            if not candidates:
                consecutive_failures += 1
                continue

            # 4. Optimised collision check
            is_valid = True

            # A. Void-void test (vectorised)
            if voids and not self._check_void_void_collision_fast(candidates, voids):
                is_valid = False

            # B. Void-fiber test (only if void-void is OK)
            if is_valid and fiber_data:
                if not self._check_void_fiber_collision_fast(candidates, fiber_data):
                    is_valid = False

            # 5. Registration
            if is_valid:
                voids.extend(candidates)
                vol_added = (4/3) * np.pi * (radius**3)
                current_vol += vol_added
                void_id_counter += 1
                consecutive_failures = 0

                if len(voids) % 100 == 0:
                    logger.debug(f"Porosity: {current_vol/box_vol:.2%} ({len(voids)} voids)...")
            else:
                consecutive_failures += 1

        if attempts >= max_attempts:
            logger.warning(f"Porosity: attempt limit reached. Final Vf: {current_vol/box_vol:.2%}")

        logger.info(f"Porosity generated: {len(voids)} voids, Vf = {current_vol/box_vol:.2%}")
        return voids

    def _create_periodic_images_fast(self, center: np.ndarray, radius: float, uid: int) -> List[Void]:
        """!
        @brief Creates the primary void and its periodic replicas (ghosts).
        @param center Center coordinates of the primary void.
        @param radius Radius of the void.
        @param uid Unique identifier for the void group.
        @return List of Void objects (primary + periodic images).
        """
        Lx, Ly, Lz = self.box_dims
        images = []

        # Determine which shifts are required
        shifts_x = [0]
        if center[0] - radius < 0: shifts_x.append(1)
        if center[0] + radius > Lx: shifts_x.append(-1)

        shifts_y = [0]
        if center[1] - radius < 0: shifts_y.append(1)
        if center[1] + radius > Ly: shifts_y.append(-1)

        shifts_z = [0]
        if center[2] - radius < 0: shifts_z.append(1)
        if center[2] + radius > Lz: shifts_z.append(-1)

        # Combinatorial generation
        shift_vectors = []
        for dx in shifts_x:
            for dy in shifts_y:
                for dz in shifts_z:
                    shift_vec = np.array([dx*Lx, dy*Ly, dz*Lz])
                    shift_vectors.append(shift_vec)

        # Vectorised overlap check
        for shift_vec in shift_vectors:
            img_center = center + shift_vec

            # Fast overlap check
            min_s = img_center - radius
            max_s = img_center + radius

            # Overlap with the box [0, L]
            overlap = (
                min_s[0] < Lx and max_s[0] > 0 and
                min_s[1] < Ly and max_s[1] > 0 and
                min_s[2] < Lz and max_s[2] > 0
            )

            if overlap:
                v = Void(id=uid, center=img_center, radius=radius)
                images.append(v)

        return images

    def _check_void_void_collision_fast(self, candidates: List[Void], existing_voids: List[Void]) -> bool:
        """!
        @brief Vectorized check for collisions between new candidates and existing voids.
        @param candidates List of new void images to check.
        @param existing_voids List of voids already accepted in the domain.
        @return True if no collision is detected, False otherwise.
        """
        if not existing_voids:
            return True

        # Extract arrays for vectorisation
        cand_centers = np.array([v.center for v in candidates])  # (M, 3)
        cand_radii = np.array([v.radius for v in candidates])    # (M,)

        exist_centers = np.array([v.center for v in existing_voids])  # (N, 3)
        exist_radii = np.array([v.radius for v in existing_voids])    # (N,)

        # Vectorised computation of distances and thresholds
        # (M, 1, 3) - (1, N, 3) -> (M, N, 3)
        diff = cand_centers[:, np.newaxis, :] - exist_centers[np.newaxis, :, :]
        dists_sq = np.sum(diff * diff, axis=2)  # (M, N)

        # Collision thresholds (r1 + r2)^2
        radii_sum = cand_radii[:, np.newaxis] + exist_radii[np.newaxis, :]  # (M, N)
        thresholds = radii_sum * radii_sum  # (M, N)

        # Check: no collision
        return not np.any(dists_sq < thresholds)

    def _check_void_fiber_collision_fast(self, candidates: List[Void], fiber_data: List[dict]) -> bool:
        """!
        @brief Optimized check for collisions between voids and fibers.
        @param candidates List of void images to check.
        @param fiber_data Pre-processed fiber geometric data (centerlines and BBoxes).
        @return True if no collision is detected, False otherwise.
        """
        for void in candidates:
            void_r = void.radius
            void_c = void.center
            void_r_sq = void_r * void_r

            # Void BBox for optimisation
            void_min = void_c - void_r
            void_max = void_c + void_r

            for fib_data in fiber_data:
                # Fast BBox test
                fib_min = fib_data['bbox_min']
                fib_max = fib_data['bbox_max']

                if (void_max[0] < fib_min[0] or void_min[0] > fib_max[0] or
                    void_max[1] < fib_min[1] or void_min[1] > fib_max[1] or
                    void_max[2] < fib_min[2] or void_min[2] > fib_max[2]):
                    continue  # No BBox intersection

                # Precise check
                fib_pts = fib_data['centerline']
                fib_r = fib_data['radius']
                min_dist_sq = (void_r + fib_r) ** 2

                # Vectorised squared-distance computation
                diff = fib_pts - void_c
                dists_sq = np.sum(diff * diff, axis=1)

                if np.any(dists_sq < min_dist_sq):
                    return False  # Collision detected

        return True

    # --- Compatibility methods (kept for the interface) ---

    def _check_void_void_collision(self, candidates: List[Void], existing_voids: List[Void]) -> bool:
        """!
        @brief Compatibility wrapper for void-void collision check.
        @param candidates New void candidates.
        @param existing_voids Existing voids.
        @return Boolean result of the check.
        """
        return self._check_void_void_collision_fast(candidates, existing_voids)

    def _check_void_fiber_collision(self, candidates: List[Void], fibers: List[Fiber]) -> bool:
        """!
        @brief Compatibility wrapper for void-fiber collision check.
        @details Converts Fiber objects to optimized dict format before calling the fast checker.
        @param candidates New void candidates.
        @param fibers List of Fiber objects.
        @return Boolean result of the check.
        """
        # Convert the fibers to the optimised format
        fiber_data = []
        for fib in fibers:
            centerline = fib.centerline
            radius = fib.radius
            min_pt = np.min(centerline, axis=0) - radius
            max_pt = np.max(centerline, axis=0) + radius
            fiber_data.append({
                'centerline': centerline,
                'radius': radius,
                'bbox_min': min_pt,
                'bbox_max': max_pt
            })
        return self._check_void_fiber_collision_fast(candidates, fiber_data)

    # --- Basic geometric methods ---

    def _sq_dist_point_polyline(self, point: np.ndarray, polyline: np.ndarray) -> float:
        """!
        @brief Calculates the minimum squared distance between a point and a polyline.
        @param point 3D coordinates of the point.
        @param polyline Array of points forming the polyline.
        @return Minimum squared distance.
        """
        # Vectorised computation
        diff = polyline - point
        dists_sq = np.sum(diff * diff, axis=1)
        return np.min(dists_sq)

    def _sq_dist_point_segment(self, p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        """!
        @brief Calculates the squared distance between a point and a line segment.
        @param p The point.
        @param a Start point of the segment.
        @param b End point of the segment.
        @return Squared distance.
        """
        ab = b - a
        ap = p - a

        denom = np.dot(ab, ab)
        if denom < 1e-12:
            return np.dot(ap, ap)

        t = np.dot(ap, ab) / denom
        t_clamped = np.clip(t, 0.0, 1.0)

        closest = a + t_clamped * ab
        dist_vec = p - closest

        return np.dot(dist_vec, dist_vec)
