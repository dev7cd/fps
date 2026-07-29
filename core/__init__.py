# SPDX-FileCopyrightText: 2026 Devine Ngouloubi <exauce-devine.ngouloubi@unicaen.fr>
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
@dir core/
@brief Central module defining the fundamental data structures of the system.

This module gathers the essential components for the RVE representation:
 - @ref FiberPackingConfig : Configuration and parameter management.
 - @ref Fiber : Geometric and physical representation of the fibers.
 - @ref Void : Porosity defect management.
 - @ref SpatialGrid : Acceleration structure for collision detection.
 - Geometric data structures (CutData).
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
