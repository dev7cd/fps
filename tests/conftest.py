"""
Fixtures partagées pour les tests du projet fps.
"""
import pytest
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


class FakeFiber:
    """Objet léger imitant une Fiber pour les tests de SpatialGrid et CollisionDetector."""

    def __init__(self, fiber_id, centerline, radius, parent_id=None):
        self.id = fiber_id
        self.parent_id = parent_id if parent_id is not None else fiber_id
        self.centerline = np.ascontiguousarray(centerline, dtype=np.float64)
        self.radius = radius
        p_min = np.min(self.centerline, axis=0) - self.radius
        p_max = np.max(self.centerline, axis=0) + self.radius
        self.bbox = (p_min, p_max)


@pytest.fixture
def simple_fiber_a():
    """Fibre rectiligne le long de l'axe X, de (0,0.5,0.5) à (1,0.5,0.5)."""
    pts = np.array([
        [0.0, 0.5, 0.5],
        [0.25, 0.5, 0.5],
        [0.5, 0.5, 0.5],
        [0.75, 0.5, 0.5],
        [1.0, 0.5, 0.5],
    ])
    return FakeFiber(fiber_id=1, centerline=pts, radius=0.05)


@pytest.fixture
def simple_fiber_b():
    """Fibre rectiligne le long de l'axe Y, de (0.5,0,0.5) à (0.5,1,0.5)."""
    pts = np.array([
        [0.5, 0.0, 0.5],
        [0.5, 0.25, 0.5],
        [0.5, 0.5, 0.5],
        [0.5, 0.75, 0.5],
        [0.5, 1.0, 0.5],
    ])
    return FakeFiber(fiber_id=2, centerline=pts, radius=0.05)


@pytest.fixture
def distant_fiber():
    """Fibre éloignée, sans collision possible avec les fibres A et B."""
    pts = np.array([
        [5.0, 5.0, 5.0],
        [5.5, 5.0, 5.0],
        [6.0, 5.0, 5.0],
    ])
    return FakeFiber(fiber_id=3, centerline=pts, radius=0.05)
