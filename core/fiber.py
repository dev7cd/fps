"""
core/fiber.py
"""
import numpy as np
from typing import Optional, Dict, Tuple, List
from geometry.curves import CatmullRomSpline
from geometry.frames import compute_bishop_frame
from geometry.sections import create_section

class Fiber:
    def __init__(self, fiber_id: int, control_points: np.ndarray, radius: float, config_params: Dict, **kwargs):
        self.id = fiber_id
        self.radius = radius
        self.control_points = control_points
        self.is_ghost = kwargs.get('is_ghost', False)
        self.parent_id = kwargs.get('parent_id', fiber_id)
        
        # Initialisation de la section réelle (Superellipse, etc.)
        self.section_type = config_params.get('fiber_section_type', 'circular')
        self.section_params = config_params.get('section_parameters', {})
        # Injection du rayon actuel pour la géométrie réelle
        if 'radius' not in self.section_params: self.section_params['radius'] = radius
        self.section_profile = create_section(self.section_type, **self.section_params)
        
        # Cache géométrique
        self.centerline: Optional[np.ndarray] = None
        self.bbox: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.T: Optional[np.ndarray] = None # Tangentes
        self.N: Optional[np.ndarray] = None # Normales
        self.B: Optional[np.ndarray] = None # Binormales
        
        self.refresh_geometry()

    def refresh_geometry(self):
        """Calcule la ligne moyenne, les repères de Bishop et la BBox."""
        self.centerline = CatmullRomSpline.interpolate(self.control_points, num_points=100)
        # Calcul des repères locaux pour GMSH et Voxelizer
        self.T, self.N, self.B = compute_bishop_frame(self.centerline)
        
        p_min = np.min(self.centerline, axis=0) - self.radius
        p_max = np.max(self.centerline, axis=0) + self.radius
        self.bbox = (p_min, p_max)

    def get_real_volume(self):
        area = self.section_profile.get_area()
        pts = self.centerline
        length = np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))
        return area * length