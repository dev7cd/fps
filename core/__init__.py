"""
core/
Module central définissant les structures de données fondamentales du système.
Contient :
- La configuration globale (FiberPackingConfig)
- Les objets physiques (Fiber, Void)
- Les structures de données géométriques (CutData)
- Les structures d'accélération spatiale (SpatialGrid)
"""

from .config import FiberPackingConfig
from .fiber import Fiber
from .void import Void
from .grid_structure import SpatialGrid

__all__ = [
    'FiberPackingConfig',
    'Fiber',
    'Void',
    'SpatialGrid'
]