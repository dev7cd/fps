"""
@file __init__.py
@brief Geometry module containing mathematical and geometric primitives.
@details Provides tools for cross-section definitions, spline interpolation, 
and local frame computation (Bishop frames) for 3D curves.
"""

from .sections import ( # type: ignore
    BaseSection,
    CircularSection,
    SuperEllipticalSection,
    create_section
)
from .curves import ( # type: ignore
    CatmullRomSpline,
    generate_random_control_points
)
from .frames import ( # type: ignore
    compute_bishop_frame,
    compute_tangents
)

__all__ = [
    # Note: These functions are expected to be implemented in this module 
    # or imported if they exist in submodules. 
    # Currently, they are listed in __all__ but not imported above.
    
    'project_point_on_plane_along_vector',
    'get_plane_parameters',
    'compute_sheared_section',
    'BaseSection',
    'CircularSection',
    'SuperEllipticalSection',
    'create_section',
    'CatmullRomSpline',
    'compute_bishop_frame',
    'compute_tangents'
]
