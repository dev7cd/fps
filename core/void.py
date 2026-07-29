# SPDX-FileCopyrightText: 2026 Devine Ngouloubi <exauce-devine.ngouloubi@unicaen.fr>
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
core/void.py
Representation of porosity defects (spherical or ellipsoidal bubbles).
"""
import numpy as np
from dataclasses import dataclass
from .fiber import Fiber

@dataclass
class Void:
    """!
    @class Void
    @brief Represents a porosity defect (air bubble) in the RVE.

    Currently implemented as a sphere, this class handles collisions with
    fibers and the computation of the void fraction.
    """
    id: int            ##< Unique identifier of the defect
    center: np.ndarray ##< Coordinates (x, y, z) of the centre
    radius: float      ##< Radius of the bubble (spherical)

    def contains_point(self, point: np.ndarray) -> bool:
        """!
        @brief Checks whether a point (x, y, z) lies inside the bubble.
        @param point np.ndarray Coordinates of the point to test.
        @return bool True if the point is inside, False otherwise.
        """
        return np.linalg.norm(point - self.center) <= self.radius

    def intersect_fiber(self, fiber: 'Fiber') -> bool:
        """!
        @brief Collision check between the bubble and a fiber.

        Computes the minimum distance between the bubble centre and each
        segment of the fiber centerline. If this distance is smaller than the
        sum of the radii (bubble + fiber), there is an intersection.

        @param fiber Fiber Instance of the Fiber class to test.
        @return bool True if an intersection is detected, False otherwise.
        """
        pts = fiber.centerline
        if pts is None or len(pts) < 2:
            return False

        threshold_sq = (self.radius + fiber.radius) ** 2

        for i in range(len(pts) - 1):
            a = pts[i]
            b = pts[i + 1]
            ab = b - a
            ap = self.center - a

            denom = np.dot(ab, ab)
            if denom < 1e-12:
                dist_sq = np.dot(ap, ap)
            else:
                t = np.clip(np.dot(ap, ab) / denom, 0.0, 1.0)
                closest = a + t * ab
                diff = self.center - closest
                dist_sq = np.dot(diff, diff)

            if dist_sq <= threshold_sq:
                return True

        return False
