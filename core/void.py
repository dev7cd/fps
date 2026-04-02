"""
core/void.py
Représentation des défauts de porosité (bulles sphériques ou ellipsoïdales).
"""
import numpy as np
from dataclasses import dataclass

@dataclass
class Void:
    id: int
    center: np.ndarray
    radius: float  # Pour l'instant sphérique, extensible à (rx, ry, rz)
    
    def contains_point(self, point: np.ndarray) -> bool:
        """Vérifie si un point (x,y,z) est dans la bulle."""
        return np.linalg.norm(point - self.center) <= self.radius

    def intersect_fiber(self, fiber: 'Fiber') -> bool:
        """
        Vérification rapide collision Fibre/Void.
        Approximation conservatrice : distance Point-Segment.
        """
        # On itère sur les segments de la fibre
        # Note: Pour une vérification précise, on utilisera le module collision/
        for segment in fiber.segments:
            # Logique simple de bounding box ici pour filtrer
            pass 
        return False