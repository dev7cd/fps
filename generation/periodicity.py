"""
generation/periodicity.py
Periodicity manager (ghost fibers).
Updated to handle strict alignment.
"""

import numpy as np
from typing import List, Tuple, Optional
from core.fiber import Fiber

class PeriodicManager:
    """!
    @class PeriodicManager
    @brief Handles the spatial periodicity logic for fibers.
    """
    @staticmethod
    def wrap_fiber(fiber, box_dims):
        """!
        @brief Moves the fiber centroid back inside the domain.

        @param fiber Fiber The Fiber object to check/move.
        @param box_dims np.ndarray Domain dimensions [Lx, Ly, Lz].
        @return bool True if the fiber was translated, False otherwise.
        @details Adjusts the control points by a periodic translation if needed.
        """
        centroid = np.mean(fiber.control_points, axis=0)
        shift = np.zeros(3)

        for i in range(3):
            if centroid[i] < 0: shift[i] = box_dims[i]
            if centroid[i] > box_dims[i]: shift[i] = -box_dims[i]

        if np.any(shift != 0):
            fiber.control_points += shift
            fiber.refresh_geometry()
            return True # The fiber was moved
        return False

    @staticmethod
    def generate_ghosts(fiber, box_dims):
        """!
        @brief Computes the translation vectors for the ghost (periodic) images.

        @param fiber Fiber The source fiber.
        @param box_dims np.ndarray Domain dimensions.
        @return list List of numpy vectors (translations) to create the periodic clones.
        """
        Lx, Ly, Lz = box_dims
        rad = fiber.radius
        min_p, max_p = fiber.bbox

        shifts = []
        # Test the 26 neighbouring directions (3^3 - 1)
        for dx in [-Lx, 0, Lx]:
            for dy in [-Ly, 0, Ly]:
                for dz in [-Lz, 0, Lz]:
                    if dx == 0 and dy == 0 and dz == 0: continue

                    # Does the shifted BBox intersect the box [0, dims]?
                    if (max_p[0] + dx > 0 and min_p[0] + dx < Lx and
                        max_p[1] + dy > 0 and min_p[1] + dy < Ly and
                        max_p[2] + dz > 0 and min_p[2] + dz < Lz):
                        shifts.append(np.array([dx, dy, dz]))

        return shifts # List of translation vectors for the ghosts
