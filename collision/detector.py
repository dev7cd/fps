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
        cell_size = 2 * config.fiber_radius + getattr(config, 'min_clearance', 0.0)
        self.grid = SpatialGrid(np.array(config.box_dims), cell_size) ##< Spatial grid for test acceleration

    def check_collision_fine(self, f1, f2) -> bool:
        """!
        @brief Unified method to test the collision between two fibers.
        @param f1 Fiber First fiber.
        @param f2 Fiber Second fiber.
        @return bool True if a collision is detected, False otherwise.
        """
        return check_collision_numba(f1.centerline, f2.centerline, f1.radius, f2.radius)

    def is_group_valid(self, fiber_group: List) -> bool:
        """!
        @brief Checks whether a group of fibers is free of internal and external collisions.
        @param fiber_group List[Fiber] List of fibers to validate.
        @return bool True if the group is valid, False otherwise.
        """
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
        neighbors = self.grid.query_neighbors((s_min, s_max), exclude_id=-1)
        seg_points = np.ascontiguousarray(np.vstack((p1, p2)))
        for nb in neighbors:
            if check_collision_numba(seg_points, nb.centerline, radius, nb.radius):
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
        neighbors = self.grid.query_neighbors((s_min, s_max), exclude_id=-1)
        pt_arr = point.reshape(1,3)
        for nb in neighbors:
            if check_collision_numba(pt_arr, nb.centerline, radius, nb.radius):
                return False
        return True

    def add_fibers_group(self, fiber_group: List):
        """!
        @brief Adds a group of fibers to the spatial grid.
        @param fiber_group List[Fiber] Group of fibers to add.
        """
        for f in fiber_group:
            self.grid.add_fiber(f)

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
                if check_collision_numba(ghost_points, nb.centerline, fiber.radius, nb.radius):
                    return False
        return True
