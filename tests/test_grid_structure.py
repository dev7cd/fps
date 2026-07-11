"""! @brief Quick Doxygen documentation for test module."""
"""
Tests unitaires pour core/grid_structure.py — SpatialGrid.
"""
import pytest
import numpy as np
from core.grid_structure import SpatialGrid


class TestSpatialGridInit:
    """Checks the initialisation of the spatial grid."""

    def test_empty_grid(self):
        grid = SpatialGrid(np.array([1.0, 1.0, 1.0]), cell_size=0.2)
        assert len(grid.cells) == 0
        assert len(grid.fiber_cells) == 0

    def test_cell_size_stored(self):
        grid = SpatialGrid(np.array([2.0, 2.0, 2.0]), cell_size=0.5)
        assert grid.cell_size == 0.5

    def test_box_dims_stored(self):
        dims = np.array([3.0, 4.0, 5.0])
        grid = SpatialGrid(dims, cell_size=1.0)
        np.testing.assert_array_equal(grid.box_dims, dims)


class TestGetCellCoords:
    """Checks the computation of the cell coordinates."""

    def test_origin(self):
        grid = SpatialGrid(np.array([1.0, 1.0, 1.0]), cell_size=0.5)
        assert grid._get_cell_coords(np.array([0.0, 0.0, 0.0])) == (0, 0, 0)

    def test_positive_point(self):
        grid = SpatialGrid(np.array([1.0, 1.0, 1.0]), cell_size=0.5)
        assert grid._get_cell_coords(np.array([0.7, 0.3, 0.9])) == (1, 0, 1)

    def test_exact_boundary(self):
        grid = SpatialGrid(np.array([1.0, 1.0, 1.0]), cell_size=0.5)
        assert grid._get_cell_coords(np.array([0.5, 0.5, 0.5])) == (1, 1, 1)

    def test_negative_coords(self):
        grid = SpatialGrid(np.array([1.0, 1.0, 1.0]), cell_size=0.5)
        coords = grid._get_cell_coords(np.array([-0.1, -0.1, -0.1]))
        assert coords == (-1, -1, -1)


class TestAddFiber:
    """Checks the addition of fibers to the grid."""

    def test_add_single_fiber(self, simple_fiber_a):
        grid = SpatialGrid(np.array([2.0, 2.0, 2.0]), cell_size=0.2)
        grid.add_fiber(simple_fiber_a)
        assert simple_fiber_a.parent_id in grid.fiber_cells
        assert len(grid.cells) > 0

    def test_add_two_fibers(self, simple_fiber_a, simple_fiber_b):
        grid = SpatialGrid(np.array([2.0, 2.0, 2.0]), cell_size=0.2)
        grid.add_fiber(simple_fiber_a)
        grid.add_fiber(simple_fiber_b)
        assert simple_fiber_a.parent_id in grid.fiber_cells
        assert simple_fiber_b.parent_id in grid.fiber_cells

    def test_fiber_occupies_multiple_cells(self, simple_fiber_a):
        grid = SpatialGrid(np.array([2.0, 2.0, 2.0]), cell_size=0.1)
        grid.add_fiber(simple_fiber_a)
        # Une fibre de longueur 1.0 avec cell_size=0.1 doit occuper plusieurs cellules
        assert len(grid.fiber_cells[simple_fiber_a.parent_id]) > 1


class TestRemoveFiber:
    """Checks the removal of fibers from the grid."""

    def test_remove_existing_fiber(self, simple_fiber_a):
        grid = SpatialGrid(np.array([2.0, 2.0, 2.0]), cell_size=0.2)
        grid.add_fiber(simple_fiber_a)
        grid.remove_fiber(simple_fiber_a.parent_id)
        assert simple_fiber_a.parent_id not in grid.fiber_cells
        # Check that no cell still contains the fiber
        for cell_fibers in grid.cells.values():
            for f in cell_fibers:
                assert f.parent_id != simple_fiber_a.parent_id

    def test_remove_nonexistent_fiber(self):
        grid = SpatialGrid(np.array([1.0, 1.0, 1.0]), cell_size=0.2)
        # Ne doit pas lever d'exception
        grid.remove_fiber(999)

    def test_remove_cleans_empty_cells(self, simple_fiber_a):
        grid = SpatialGrid(np.array([2.0, 2.0, 2.0]), cell_size=0.2)
        grid.add_fiber(simple_fiber_a)
        grid.remove_fiber(simple_fiber_a.parent_id)
        # All empty cells must have been removed
        for key, fibers in grid.cells.items():
            assert len(fibers) > 0, f"Empty cell not cleaned up: {key}"

    def test_remove_one_keeps_other(self, simple_fiber_a, simple_fiber_b):
        grid = SpatialGrid(np.array([2.0, 2.0, 2.0]), cell_size=0.2)
        grid.add_fiber(simple_fiber_a)
        grid.add_fiber(simple_fiber_b)
        grid.remove_fiber(simple_fiber_a.parent_id)
        assert simple_fiber_b.parent_id in grid.fiber_cells
        assert simple_fiber_a.parent_id not in grid.fiber_cells


class TestQueryNeighbors:
    """Checks the neighbourhood queries."""

    def test_find_nearby_fiber(self, simple_fiber_a, simple_fiber_b):
        grid = SpatialGrid(np.array([2.0, 2.0, 2.0]), cell_size=0.2)
        grid.add_fiber(simple_fiber_a)
        grid.add_fiber(simple_fiber_b)
        # Chercher autour de la fibre A — la fibre B croise au centre
        neighbors = grid.query_neighbors(simple_fiber_a.bbox, exclude_id=simple_fiber_a.parent_id)
        neighbor_ids = [f.parent_id for f in neighbors]
        assert simple_fiber_b.parent_id in neighbor_ids

    def test_exclude_self(self, simple_fiber_a):
        grid = SpatialGrid(np.array([2.0, 2.0, 2.0]), cell_size=0.2)
        grid.add_fiber(simple_fiber_a)
        neighbors = grid.query_neighbors(simple_fiber_a.bbox, exclude_id=simple_fiber_a.parent_id)
        neighbor_ids = [f.parent_id for f in neighbors]
        assert simple_fiber_a.parent_id not in neighbor_ids

    def test_no_neighbors_for_distant_fiber(self, simple_fiber_a, distant_fiber):
        grid = SpatialGrid(np.array([10.0, 10.0, 10.0]), cell_size=0.2)
        grid.add_fiber(simple_fiber_a)
        neighbors = grid.query_neighbors(distant_fiber.bbox, exclude_id=distant_fiber.parent_id)
        assert len(neighbors) == 0

    def test_unique_results(self, simple_fiber_a, simple_fiber_b):
        """Each neighbouring fiber must appear only once."""
        grid = SpatialGrid(np.array([2.0, 2.0, 2.0]), cell_size=0.1)
        grid.add_fiber(simple_fiber_a)
        grid.add_fiber(simple_fiber_b)
        neighbors = grid.query_neighbors(simple_fiber_a.bbox, exclude_id=simple_fiber_a.parent_id)
        ids = [f.parent_id for f in neighbors]
        assert len(ids) == len(set(ids)), "Duplicates detected in the results"

    def test_empty_grid_returns_empty(self):
        grid = SpatialGrid(np.array([1.0, 1.0, 1.0]), cell_size=0.2)
        bbox = (np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0]))
        neighbors = grid.query_neighbors(bbox, exclude_id=0)
        assert neighbors == []
