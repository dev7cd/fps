"""
generation/periodicity.py
Gestionnaire de périodicité (Ghost Fibers).
Mise à jour pour gérer l'alignement strict.
"""

import numpy as np
from typing import List, Tuple, Optional
from core.fiber import Fiber

class PeriodicManager:
    @staticmethod
    def wrap_fiber(fiber, box_dims):
        """
        Assure que le centre de gravité de la fibre est dans le domaine [0, dims].
        Ajuste les points de contrôle par translation périodique.
        """
        centroid = np.mean(fiber.control_points, axis=0)
        shift = np.zeros(3)
        
        for i in range(3):
            if centroid[i] < 0: shift[i] = box_dims[i]
            if centroid[i] > box_dims[i]: shift[i] = -box_dims[i]
        
        if np.any(shift != 0):
            fiber.control_points += shift
            fiber.refresh_geometry()
            return True # La fibre a été déplacée
        return False

    @staticmethod
    def generate_ghosts(fiber, box_dims):
        """Génère les clones nécessaires si la fibre touche une paroi."""
        Lx, Ly, Lz = box_dims
        rad = fiber.radius
        min_p, max_p = fiber.bbox
        
        shifts = []
        # On teste les 26 directions voisines (3^3 - 1)
        for dx in [-Lx, 0, Lx]:
            for dy in [-Ly, 0, Ly]:
                for dz in [-Lz, 0, Lz]:
                    if dx == 0 and dy == 0 and dz == 0: continue
                    
                    # Est-ce que la BBox décalée intersecte la boîte [0, dims]?
                    if (max_p[0] + dx > 0 and min_p[0] + dx < Lx and
                        max_p[1] + dy > 0 and min_p[1] < Ly and
                        max_p[2] + dz > 0 and min_p[2] < Lz):
                        shifts.append(np.array([dx, dy, dz]))
        
        return shifts # Liste des vecteurs de translation pour les ghosts