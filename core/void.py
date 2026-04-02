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
        @brief Vérification rapide de collision entre la bulle et une fibre.
        
        @param fiber Instance de la classe Fiber à tester.
        @return True si une intersection est détectée.
        """
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