# SPDX-FileCopyrightText: 2026 Devine Ngouloubi <exauce-devine.ngouloubi@unicaen.fr>
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Tests of the cross-section domain of the voxel export (D01).

The voxeliser only represents circular sections; any other section must be
refused before rasterisation instead of being exported as a circular tube.
"""
import os
import subprocess
import sys

import numpy as np
import pytest

import export.voxelizer as voxelizer
from core.config import FiberPackingConfig
from core.fiber import Fiber
from export.voxelizer import (
    TAG_FIBER,
    UnsupportedSectionError,
    Voxelizer,
    check_voxel_section_type,
)

FPS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Discriminating case of D01: straight fiber along Z in the unit box.
POINTS = np.array([
    [0.5, 0.5, 0.2],
    [0.5, 0.5, 0.4],
    [0.5, 0.5, 0.6],
    [0.5, 0.5, 0.8],
])
ENVELOPE_RADIUS = 0.08
CIRCULAR = {
    'fiber_section_type': 'circular',
    'section_parameters': {'radius': 0.08},
}
SUPERELLIPTICAL = {
    'fiber_section_type': 'superelliptical',
    'section_parameters': {'major_radius': 0.06, 'minor_radius': 0.03,
                           'exponent': 2.5, 'radius': 0.08},
}


def make_fiber(fiber_id, params, points=POINTS):
    return Fiber(fiber_id, points, ENVELOPE_RADIUS, params)


@pytest.fixture
def voxelizer_unit_box():
    return Voxelizer(FiberPackingConfig(box_dims=(1.0, 1.0, 1.0), fiber_radius=ENVELOPE_RADIUS))


@pytest.fixture
def forbid_rasterisation(monkeypatch):
    """Makes any call to the fiber rasteriser fail the test."""
    def _fail(*args, **kwargs):
        raise AssertionError("rasterisation started for a refused collection")
    monkeypatch.setattr(voxelizer, "_raster_fiber_numba", _fail)


def test_circular_section_keeps_reference_grid(voxelizer_unit_box):
    grid = voxelizer_unit_box.create_grid([make_fiber(1, CIRCULAR)], [], 64)

    assert grid.shape == (64, 64, 64)
    assert grid.dtype == np.uint8
    assert set(np.unique(grid)) == {0, TAG_FIBER}
    # Non-regression count recorded in T02 before the fix.
    assert int(np.sum(grid == TAG_FIBER)) == 3648


def test_superelliptical_section_is_refused(voxelizer_unit_box, forbid_rasterisation):
    with pytest.raises(UnsupportedSectionError, match="superelliptical"):
        voxelizer_unit_box.create_grid([make_fiber(1, SUPERELLIPTICAL)], [], 64)


def test_mixed_collection_is_refused(voxelizer_unit_box, forbid_rasterisation):
    shifted = POINTS + np.array([0.25, 0.0, 0.0])
    fibers = [make_fiber(1, CIRCULAR), make_fiber(2, SUPERELLIPTICAL, shifted)]

    with pytest.raises(UnsupportedSectionError) as excinfo:
        voxelizer_unit_box.create_grid(fibers, [], 64)

    message = str(excinfo.value)
    assert "superelliptical" in message
    assert "[2]" in message


def test_section_type_check():
    check_voxel_section_type('circular')
    with pytest.raises(UnsupportedSectionError):
        check_voxel_section_type('superelliptical')


def test_cli_refuses_superelliptical_before_generation(tmp_path):
    result = subprocess.run(
        [sys.executable, os.path.join(FPS_ROOT, "main.py"),
         "--section", "superelliptical", "--no_mesh", "--res_fft", "16"],
        cwd=tmp_path, capture_output=True, text=True, timeout=120,
    )

    assert result.returncode != 0
    assert "does not support the 'superelliptical' cross-section" in result.stderr
    assert "use --section circular" in result.stderr
    assert "PHASE 1" not in result.stderr
    assert list(tmp_path.iterdir()) == []
