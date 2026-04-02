"""
collision/detector.py
Détecteur optimisé avec gestion stricte du cache et méthodes N^2 de sécurité.
"""

import numpy as np
from typing import List, Tuple
from core.grid_structure import SpatialGrid
from generation.periodicity import PeriodicManager
from .detector_math import check_collision_numba

class CollisionDetector:
    def __init__(self, config):
        self.config = config
        cell_size = 2 * config.fiber_radius + getattr(config, 'min_clearance', 0.0)
        self.grid = SpatialGrid(np.array(config.box_dims), cell_size)

    def check_collision_fine(self, f1, f2) -> bool:
        """Méthode unifiée pour tester deux fibres."""
        return check_collision_numba(f1.centerline, f2.centerline, f1.radius, f2.radius)

    def is_group_valid(self, fiber_group: List) -> bool:
        for fiber in fiber_group:
            neighbors = self.grid.query_neighbors(fiber.bbox, fiber.parent_id)
            for nb in neighbors:
                if self.check_collision_fine(fiber, nb):
                    return False
        return True

    def is_segment_free(self, p1: np.ndarray, p2: np.ndarray, radius: float) -> bool:
        s_min = np.minimum(p1, p2) - radius
        s_max = np.maximum(p1, p2) + radius
        neighbors = self.grid.query_neighbors((s_min, s_max), exclude_id=-1)
        seg_points = np.ascontiguousarray(np.vstack((p1, p2)))
        for nb in neighbors:
            if check_collision_numba(seg_points, nb.centerline, radius, nb.radius):
                return False
        return True

    def is_point_free(self, point: np.ndarray, radius: float) -> bool:
        """Vérifie si une sphère de rayon 'radius' peut être placée."""
        s_min = point - radius
        s_max = point + radius
        neighbors = self.grid.query_neighbors((s_min, s_max), exclude_id=-1)
        pt_arr = point.reshape(1,3)
        for nb in neighbors:
            if check_collision_numba(pt_arr, nb.centerline, radius, nb.radius):
                return False
        return True
    
    def add_fibers_group(self, fiber_group: List):
        for f in fiber_group:
            self.grid.add_fiber(f)

    def is_fiber_valid_periodic(self, fiber) -> bool:
        """Utilisée lors du Shaking et de l'Optimisation."""
        box_dims = self.config.box_dims
        
        # 1. Test contre voisins directs
        neighbors = self.grid.query_neighbors(fiber.bbox, exclude_id=fiber.parent_id)
        for nb in neighbors:
            if self.check_collision_fine(fiber, nb):
                return False
                
        # 2. Test des clones (images périodiques)
        # Note : On importe PeriodicManager ici si nécessaire ou au début du fichier
        from generation.periodicity import PeriodicManager
        ghost_shifts = PeriodicManager.generate_ghosts(fiber, box_dims)
        
        for shift in ghost_shifts:
            ghost_points = fiber.centerline + shift
            ghost_bbox = (fiber.bbox[0] + shift, fiber.bbox[1] + shift)
            
            # Recherche de voisins proches de l'image ghost
            neighbors = self.grid.query_neighbors(ghost_bbox, exclude_id=fiber.parent_id)
            for nb in neighbors:
                if check_collision_numba(ghost_points, nb.centerline, fiber.radius, nb.radius):
                    return False
        return True