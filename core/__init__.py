"""
@dir core/
@brief Module central définissant les structures de données fondamentales du système.

Ce module regroupe les composants essentiels pour la représentation du RVE :
 - @ref FiberPackingConfig : Gestion de la configuration et des paramètres.
 - @ref Fiber : Représentation géométrique et physique des fibres.
 - @ref Void : Gestion des défauts de porosité.
 - @ref SpatialGrid : Structure d'accélération pour la détection de collisions.
 - Structures de données géométriques (CutData).
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