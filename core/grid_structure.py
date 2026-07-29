# SPDX-FileCopyrightText: 2026 Devine Ngouloubi <exauce-devine.ngouloubi@unicaen.fr>
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
core/grid_structure.py
Spatial structure for collisions.
Stores a direct reference to the Fiber object to handle ghosts.
Supports fiber removal via reverse-mapping.
"""

import numpy as np
from typing import List, Tuple, Dict, Set, Any


class SpatialGrid:
    """!
    @class SpatialGrid
    @brief Spatial acceleration structure (uniform grid) for collision detection.

    Allows fast storage and querying of the fibers present in a given region
    of the RVE. Maintains a reverse-mapping to ease fiber removal.
    """

    def __init__(self, box_dims: np.ndarray, cell_size: float):
        """!
        @brief Constructor of the spatial grid.
        @param box_dims np.ndarray Domain dimensions (Lx, Ly, Lz).
        @param cell_size float Size of a cubic grid cell.
        """
        self.box_dims = box_dims ##< RVE domain dimensions
        self.cell_size = cell_size ##< Side length of a cell
        ## Occupied cells: coordinates (ix, iy, iz) -> list of Fiber objects
        self.cells: Dict[Tuple[int, int, int], List[Any]] = {}
        ## Reverse-mapping: parent_id -> set of occupied cell keys
        self.fiber_cells: Dict[int, Set[Tuple[int, int, int]]] = {}

    def _get_cell_coords(self, point: np.ndarray) -> Tuple[int, int, int]:
        """!
        @brief Computes the cell indices for a 3D point.
        @param point np.ndarray Coordinates (x, y, z).
        @return Tuple[int, int, int] Integer indices (ix, iy, iz).
        """
        return tuple(np.floor(point / self.cell_size).astype(int))

    def add_fiber(self, fiber: Any):
        """!
        @brief Indexes each segment of the fiber into the corresponding cells.

        Computes the envelope of each centerline segment and adds the fiber
        to every cell intersected by that envelope.
        @param fiber Fiber Instance of the Fiber class to index.
        """
        pts = fiber.centerline
        rad = fiber.radius
        pid = fiber.parent_id

        if pid not in self.fiber_cells:
            self.fiber_cells[pid] = set()

        for i in range(len(pts) - 1):
            p1, p2 = pts[i], pts[i+1]
            seg_min = np.minimum(p1, p2) - rad
            seg_max = np.maximum(p1, p2) + rad

            start_idx = np.floor(seg_min / self.cell_size).astype(int)
            end_idx = np.floor(seg_max / self.cell_size).astype(int)

            for ix in range(start_idx[0], end_idx[0] + 1):
                for iy in range(start_idx[1], end_idx[1] + 1):
                    for iz in range(start_idx[2], end_idx[2] + 1):
                        idx = (ix, iy, iz)
                        if idx not in self.cells:
                            self.cells[idx] = []
                        self.cells[idx].append(fiber)
                        self.fiber_cells[pid].add(idx)

    def remove_fiber(self, parent_id: int):
        """!
        @brief Removes all references of a fiber (and its ghosts) from the grid.

        @param parent_id int Unique identifier of the parent fiber to remove.
        """
        cell_keys = self.fiber_cells.pop(parent_id, set())
        for key in cell_keys:
            if key in self.cells:
                self.cells[key] = [f for f in self.cells[key] if f.parent_id != parent_id]
                if not self.cells[key]:
                    del self.cells[key]

    def query_neighbors(self, bbox: Tuple[np.ndarray, np.ndarray], exclude_id: int) -> List[Any]:
        """!
        @brief Returns the unique fibers located in the neighbourhood of a bounding box.

        @param bbox Tuple[np.ndarray, np.ndarray] Tuple (p_min, p_max) defining the search region.
        @param exclude_id int Identifier of the fiber to exclude from the results (usually itself).
        @return List[Fiber] List of Fiber objects potentially in collision.
        """
        p_min, p_max = bbox
        start_idx = np.floor(p_min / self.cell_size).astype(int)
        end_idx = np.floor(p_max / self.cell_size).astype(int)

        neighbors = []
        seen_ids = {exclude_id}

        for ix in range(start_idx[0], end_idx[0] + 1):
            for iy in range(start_idx[1], end_idx[1] + 1):
                for iz in range(start_idx[2], end_idx[2] + 1):
                    cell = self.cells.get((ix, iy, iz))
                    if cell:
                        for fib in cell:
                            if fib.parent_id not in seen_ids:
                                neighbors.append(fib)
                                seen_ids.add(fib.parent_id)
        return neighbors
