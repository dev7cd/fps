## @file fiber.py
#  @brief Definition of the Fiber class representing an individual fiber in the RVE.

import numpy as np
from typing import Optional, Dict, Tuple, List
from geometry.curves import CatmullRomSpline
from geometry.frames import compute_bishop_frame
from geometry.sections import create_section

class Fiber:
    """!
    @brief Represents a fiber with its trajectory, cross-section and geometric properties.
    @details This class handles trajectory interpolation via splines, the computation
    of local frames (Bishop) and the computation of the spatial extent (BBox).
    """

    def __init__(self, fiber_id: int, control_points: np.ndarray, radius: float, config_params: Dict, **kwargs):
        """!
        @brief Constructor of the Fiber class.
        @param fiber_id int Unique identifier of the fiber.
        @param control_points np.ndarray Control points defining the trajectory.
        @param radius float Radius of the circular envelope (used for collisions).
        @param config_params Dict Configuration dictionary (section type, parameters).
        @param kwargs Optional arguments: is_ghost (bool), parent_id (int).
        """
        self.id = fiber_id  ##< Unique identifier of the fiber
        self.radius = radius  ##< Radius of the circular envelope
        self.control_points = control_points  ##< Control points of the trajectory
        self.is_ghost = kwargs.get('is_ghost', False)  ##< Whether the fiber is a periodic image
        self.parent_id = kwargs.get('parent_id', fiber_id)  ##< ID of the parent fiber (for ghosts)

        # Initialisation of the real section (super-ellipse, etc.)
        self.section_type = config_params.get('fiber_section_type', 'circular')  ##< Section type (e.g. superelliptical)
        self.section_params = config_params.get('section_parameters', {})  ##< Geometric parameters of the section
        # Inject the current radius into the real geometry
        if 'radius' not in self.section_params: self.section_params['radius'] = radius
        self.section_profile = create_section(self.section_type, **self.section_params)  ##< Object representing the section profile

        # Geometric cache
        self.centerline: Optional[np.ndarray] = None  ##< Interpolated centerline (N, 3)
        self.bbox: Optional[Tuple[np.ndarray, np.ndarray]] = None  ##< Bounding box (min, max)
        self.T: Optional[np.ndarray] = None  ##< Tangent vectors
        self.N: Optional[np.ndarray] = None  ##< Normal vectors
        self.B: Optional[np.ndarray] = None  ##< Binormal vectors

        self.refresh_geometry()

    def refresh_geometry(self):
        """!
        @brief Updates the derived geometric data.
        @details Computes the centerline by interpolation, generates the Bishop frames (T, N, B)
        and updates the bounding box.
        @return None
        """
        self.centerline = CatmullRomSpline.interpolate(self.control_points, num_points=100)
        # Compute local frames for GMSH and Voxelizer
        self.T, self.N, self.B = compute_bishop_frame(self.centerline)

        p_min = np.min(self.centerline, axis=0) - self.radius
        p_max = np.max(self.centerline, axis=0) + self.radius
        self.bbox = (p_min, p_max)

    def get_real_volume(self) -> float:
        """!
        @brief Computes the real volume of the fiber.
        @return float Volume based on the real section area and the curvilinear length.
        """
        area = self.section_profile.get_area()
        pts = self.centerline
        length = np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))
        return area * length
