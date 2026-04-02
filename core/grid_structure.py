"""
core/grid_structure.py
Structure spatiale pour les collisions.
Stocke une référence directe vers l'objet Fibre pour gérer les ghosts.
Supporte la suppression de fibres via reverse-mapping.
"""

import numpy as np
from typing import List, Tuple, Dict, Set, Any


class SpatialGrid:
    def __init__(self, box_dims: np.ndarray, cell_size: float):
        self.box_dims = box_dims
        self.cell_size = cell_size
        # Cellules occupées : coordonnées -> liste de fibres
        self.cells: Dict[Tuple[int, int, int], List[Any]] = {}
        # Reverse-mapping : parent_id -> ensemble de cellules occupées
        self.fiber_cells: Dict[int, Set[Tuple[int, int, int]]] = {}

    def _get_cell_coords(self, point: np.ndarray) -> Tuple[int, int, int]:
        return tuple(np.floor(point / self.cell_size).astype(int))

    def add_fiber(self, fiber: Any):
        """Indexe chaque segment de la fibre dans les cellules correspondantes."""
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
        """Supprime toutes les références d'une fibre (et ses ghosts) de la grille."""
        cell_keys = self.fiber_cells.pop(parent_id, set())
        for key in cell_keys:
            if key in self.cells:
                self.cells[key] = [f for f in self.cells[key] if f.parent_id != parent_id]
                if not self.cells[key]:
                    del self.cells[key]

    def query_neighbors(self, bbox: Tuple[np.ndarray, np.ndarray], exclude_id: int) -> List[Any]:
        """Retourne les fibres uniques proches de la BBox fournie."""
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
