# SPDX-FileCopyrightText: 2026 Devine Ngouloubi <exauce-devine.ngouloubi@unicaen.fr>
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
collision/detector.py
Optimised detector with strict cache handling and N^2 safety methods.
"""

import numpy as np
from typing import List, Tuple
from core.grid_structure import SpatialGrid
from generation.periodicity import PeriodicManager
from .detector_math import check_collision_numba

class CollisionDetector:
    """!
    @class CollisionDetector
    @brief Manager for collisions between fibers and RVE objects.

    Uses a spatial grid (SpatialGrid) to accelerate neighbour queries and
    Numba compute kernels for the precise geometric tests.
    """
    def __init__(self, config):
        """!
        @brief Initialises the detector with the given configuration.
        @param config FiberPackingConfig Configuration object holding the RVE parameters.
        """
        self.config = config ##< Simulation configuration
        ## Minimum empty gap enforced between two fiber surfaces (0.0 = simple contact)
        self.min_clearance = max(0.0, float(getattr(config, 'min_clearance', 0.0) or 0.0))
        cell_size = 2 * config.fiber_radius + self.min_clearance
        ## Spatial grid for test acceleration (queries padded by the clearance)
        self.grid = SpatialGrid(np.array(config.box_dims), cell_size,
                                query_margin=self.min_clearance)

    def _bboxes_close(self, bbox1, bbox2) -> bool:
        """!
        @brief Cheap rejection test between two bounding boxes.
        @details Boxes already include the fiber radii; separating them by more
        than the clearance rules out any contact, so the O(N*M) segment kernel
        can be skipped. Purely a shortcut: it never changes a verdict.
        @return bool True if the boxes are close enough to require the fine test.
        """
        min1, max1 = bbox1
        min2, max2 = bbox2
        c = self.min_clearance
        return bool(np.all(min1 - c <= max2) and np.all(min2 - c <= max1))

    def check_collision_fine(self, f1, f2) -> bool:
        """!
        @brief Unified method to test the collision between two fibers.
        @param f1 Fiber First fiber.
        @param f2 Fiber Second fiber.
        @return bool True if a collision (or a clearance violation) is detected, False otherwise.
        """
        if not self._bboxes_close(f1.bbox, f2.bbox):
            return False
        return check_collision_numba(f1.centerline, f2.centerline,
                                     f1.radius, f2.radius, self.min_clearance)

    def is_group_valid(self, fiber_group: List) -> bool:
        """!
        @brief Checks whether a group of fibers is free of internal and external collisions.
        @param fiber_group List[Fiber] List of fibers to validate (root + periodic ghosts).
        @return bool True if the group is valid, False otherwise.
        """
        # Internal check: a fiber must not overlap its own periodic images.
        # The grid queries below exclude parent_id, so this pair is invisible to them.
        n = len(fiber_group)
        for a in range(n):
            fa = fiber_group[a]
            for b in range(a + 1, n):
                fb = fiber_group[b]
                if self.check_collision_fine(fa, fb):
                    return False

        for fiber in fiber_group:
            neighbors = self.grid.query_neighbors(fiber.bbox, fiber.parent_id)
            for nb in neighbors:
                if self.check_collision_fine(fiber, nb):
                    return False
        return True

    def is_segment_free(self, p1: np.ndarray, p2: np.ndarray, radius: float) -> bool:
        """!
        @brief Checks whether a cylindrical segment is collision-free.
        @param p1 np.ndarray Start point of the segment.
        @param p2 np.ndarray End point of the segment.
        @param radius float Radius of the cylinder.
        @return bool True if the segment is free, False otherwise.
        """
        s_min = np.minimum(p1, p2) - radius
        s_max = np.maximum(p1, p2) + radius
        probe_bbox = (s_min, s_max)
        neighbors = self.grid.query_neighbors(probe_bbox, exclude_id=-1)
        seg_points = np.ascontiguousarray(np.vstack((p1, p2)))
        for nb in neighbors:
            if not self._bboxes_close(probe_bbox, nb.bbox):
                continue
            if check_collision_numba(seg_points, nb.centerline, radius, nb.radius,
                                     self.min_clearance):
                return False
        return True

    def is_point_free(self, point: np.ndarray, radius: float) -> bool:
        """!
        @brief Checks whether a sphere of radius 'radius' can be placed without collision.
        @param point np.ndarray Centre of the sphere.
        @param radius float Radius of the sphere.
        @return bool True if the space is free, False otherwise.
        """
        s_min = point - radius
        s_max = point + radius
        probe_bbox = (s_min, s_max)
        neighbors = self.grid.query_neighbors(probe_bbox, exclude_id=-1)
        # Degenerate segment (point duplicated): a (1, 3) array would expose
        # zero segments to the kernel and the test would always pass.
        pt_arr = np.ascontiguousarray(np.vstack((point, point)))
        for nb in neighbors:
            if not self._bboxes_close(probe_bbox, nb.bbox):
                continue
            if check_collision_numba(pt_arr, nb.centerline, radius, nb.radius,
                                     self.min_clearance):
                return False
        return True

    def add_fibers_group(self, fiber_group: List):
        """!
        @brief Adds a group of fibers to the spatial grid.
        @param fiber_group List[Fiber] Group of fibers to add.
        """
        for f in fiber_group:
            self.grid.add_fiber(f)

    def rebuild_index(self, fibers: List):
        """!
        @brief Rebuilds the spatial index from scratch for the given root fibers.
        @details Regenerates the periodic ghosts as well, so the index always
        matches the current geometry. Any code that moves fibers must call this
        (or remove/re-add them) before querying the grid again.
        @param fibers List[Fiber] Root fibers (ghosts are recreated internally).
        """
        from core.fiber import Fiber
        from generation.periodicity import PeriodicManager

        self.grid.clear()
        for f in fibers:
            self.grid.add_fiber(f)
            for shift in PeriodicManager.generate_ghosts(f, self.config.box_dims):
                ghost = Fiber(f.id, f.control_points + shift, f.radius,
                              vars(self.config), is_ghost=True, parent_id=f.id)
                self.grid.add_fiber(ghost)

    def is_fiber_valid_periodic(self, fiber) -> bool:
        """!
        @brief Checks the validity of a fiber accounting for periodic conditions.
        @param fiber Fiber The fiber to test.
        @return bool True if the fiber is valid, False otherwise.
        """
        box_dims = self.config.box_dims

        # 1. Test against direct neighbours
        neighbors = self.grid.query_neighbors(fiber.bbox, exclude_id=fiber.parent_id)
        for nb in neighbors:
            if self.check_collision_fine(fiber, nb):
                return False

        # 2. Test the clones (periodic images)
        # Note: import PeriodicManager here if needed, or at the top of the file
        from generation.periodicity import PeriodicManager
        ghost_shifts = PeriodicManager.generate_ghosts(fiber, box_dims)

        for shift in ghost_shifts:
            ghost_points = fiber.centerline + shift
            ghost_bbox = (fiber.bbox[0] + shift, fiber.bbox[1] + shift)

            # Search for neighbours close to the ghost image
            neighbors = self.grid.query_neighbors(ghost_bbox, exclude_id=fiber.parent_id)
            for nb in neighbors:
                if not self._bboxes_close(ghost_bbox, nb.bbox):
                    continue
                if check_collision_numba(ghost_points, nb.centerline, fiber.radius, nb.radius,
                                         self.min_clearance):
                    return False
        return True
