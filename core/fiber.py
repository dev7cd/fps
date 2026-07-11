## @file fiber.py
#  @brief Définition de la classe Fiber représentant une fibre individuelle dans le RVE.

import numpy as np
from typing import Optional, Dict, Tuple, List
from geometry.curves import CatmullRomSpline
from geometry.frames import compute_bishop_frame
from geometry.sections import create_section

class Fiber:
    """!
    @brief Représente une fibre avec sa trajectoire, sa section et ses propriétés géométriques.
    @details Cette classe gère l'interpolation de la trajectoire via des splines, le calcul
    des repères locaux (Bishop) et le calcul de l'encombrement spatial (BBox).
    """

    def __init__(self, fiber_id: int, control_points: np.ndarray, radius: float, config_params: Dict, **kwargs):
        """!
        @brief Constructeur de la classe Fiber.
        @param fiber_id int Identifiant unique de la fibre.
        @param control_points np.ndarray Points de contrôle définissant la trajectoire.
        @param radius float Rayon de l'enveloppe circulaire (utilisé pour les collisions).
        @param config_params Dict Dictionnaire de configuration (type de section, paramètres).
        @param kwargs Arguments optionnels : is_ghost (bool), parent_id (int).
        """
        self.id = fiber_id  ##< Identifiant unique de la fibre  ##< Identifiant unique de la fibre
        self.radius = radius  ##< Rayon de l'enveloppe circulaire
        self.control_points = control_points  ##< Points de contrôle de la trajectoire
        self.is_ghost = kwargs.get('is_ghost', False)  ##< Indique si la fibre est une image périodique
        self.parent_id = kwargs.get('parent_id', fiber_id)  ##< ID de la fibre parente (pour les ghosts)

        # Initialisation de la section réelle (Superellipse, etc.)
        self.section_type = config_params.get('fiber_section_type', 'circular')  ##< Type de section (ex: superelliptical)
        self.section_params = config_params.get('section_parameters', {})  ##< Paramètres géométriques de la section
        # Injection du rayon actuel pour la géométrie réelle
        if 'radius' not in self.section_params: self.section_params['radius'] = radius
        self.section_profile = create_section(self.section_type, **self.section_params)  ##< Objet représentant le profil de section

        # Cache géométrique
        self.centerline: Optional[np.ndarray] = None  ##< Ligne moyenne interpolée (N, 3)
        self.bbox: Optional[Tuple[np.ndarray, np.ndarray]] = None  ##< Boîte englobante (min, max)
        self.T: Optional[np.ndarray] = None  ##< Vecteurs tangents
        self.N: Optional[np.ndarray] = None  ##< Vecteurs normaux
        self.B: Optional[np.ndarray] = None  ##< Vecteurs binormaux

        self.refresh_geometry()

    def refresh_geometry(self):
        """!
        @brief Met à jour les données géométriques dérivées.
        @details Calcule la ligne moyenne par interpolation, génère les repères de Bishop (T, N, B)
        et met à jour la boîte englobante (Bounding Box).
        @return None
        """
        self.centerline = CatmullRomSpline.interpolate(self.control_points, num_points=100)
        # Calcul des repères locaux pour GMSH et Voxelizer
        self.T, self.N, self.B = compute_bishop_frame(self.centerline)

        p_min = np.min(self.centerline, axis=0) - self.radius
        p_max = np.max(self.centerline, axis=0) + self.radius
        self.bbox = (p_min, p_max)

    def get_real_volume(self) -> float:
        """!
        @brief Calcule le volume réel de la fibre.
        @return float Volume basé sur l'aire de la section réelle et la longueur curviligne.
        """
        area = self.section_profile.get_area()
        pts = self.centerline
        length = np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))
        return area * length
