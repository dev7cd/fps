"""
geometry/
Module contenant les primitives mathématiques et géométriques.
"""

from .sections import (
    BaseSection,
    CircularSection,
    SuperEllipticalSection,
    create_section
)
from .curves import (
    CatmullRomSpline,
    generate_random_control_points
)
from .frames import (
    compute_bishop_frame,
    compute_tangents
)

__all__ = [
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
