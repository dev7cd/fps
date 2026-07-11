"""
generation/periodicity.py
Gestionnaire de périodicité (Ghost Fibers).
Mise à jour pour gérer l'alignement strict.
"""

import numpy as np
from typing import List, Tuple, Optional
from core.fiber import Fiber

class PeriodicManager:
    """
    @class PeriodicManager
    @brief Gère la logique de périodicité spatiale pour les fibres.
    """
    @staticmethod
    def wrap_fiber(fiber, box_dims):
        """! @brief Placeholder.
        @param fiber, box_dims 
        @return None
        """
        """
        @brief Replace le centre de gravité de la fibre à l'intérieur du domaine.
        
        @param fiber Fiber L'objet Fiber à vérifier/déplacer.
        @param box_dims np.ndarray Dimensions [Lx, Ly, Lz] du domaine.
        @return bool True si la fibre a été translatée, False sinon.
        @details Ajuste les points de contrôle par translation périodique si nécessaire.
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
        """! @brief Placeholder.
        @param fiber, box_dims 
        @return None
        """
        """
        @brief Calcule les vecteurs de translation pour les images fantômes (ghosts).
        
        @param fiber Fiber La fibre source.
        @param box_dims np.ndarray Dimensions du domaine.
        @return list Liste de vecteurs numpy (translations) pour créer les clones périodiques.
        """
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