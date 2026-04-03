"""
core/void.py
Représentation des défauts de porosité (bulles sphériques ou ellipsoïdales).
"""
import numpy as np
from dataclasses import dataclass
from .fiber import Fiber

@dataclass
class Void:
    """
    @class Void
    @brief Représente un défaut de porosité (bulle d'air) dans le RVE.
    
    Actuellement implémenté comme une sphère, cette classe permet de gérer
    les collisions avec les fibres et de calculer la fraction de vide.
    """
    id: int            #< Identifiant unique du défaut
    center: np.ndarray #< Coordonnées (x, y, z) du centre
    radius: float      #< Rayon de la bulle (sphérique)
    
    def contains_point(self, point: np.ndarray) -> bool:
        """
        @brief Vérifie si un point (x,y,z) est situé à l'intérieur de la bulle.
        @param point Coordonnées du point à tester.
        @return True si le point est à l'intérieur, False sinon.
        """
        return np.linalg.norm(point - self.center) <= self.radius

    def intersect_fiber(self, fiber: 'Fiber') -> bool:
        """
        @brief Vérification de collision entre la bulle et une fibre.
        
        Calcule la distance minimale entre le centre de la bulle et chaque
        segment de la centerline de la fibre. Si cette distance est inférieure
        à la somme des rayons (bulle + fibre), il y a intersection.
        
        @param fiber Instance de la classe Fiber à tester.
        @return True si une intersection est détectée.
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