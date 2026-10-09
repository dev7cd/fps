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


def test_circular_section_has_flat_cylindrical_ends(voxelizer_unit_box):
    fiber = make_fiber(1, CIRCULAR)
    grid = voxelizer_unit_box.create_grid([fiber], [], 64)

    assert grid.shape == (64, 64, 64)
    assert grid.dtype == np.uint8
    assert set(np.unique(grid)) == {0, TAG_FIBER}
    # Regression count for the flat-ended swept cylinder (no hemispherical caps).
    assert int(np.sum(grid == TAG_FIBER)) == 3040
    axial_centres = (np.flatnonzero(np.any(grid == TAG_FIBER, axis=(0, 1))) + 0.5) / 64
    assert axial_centres.min() >= fiber.centerline[0, 2]
    assert axial_centres.max() <= fiber.centerline[-1, 2]
    assert fiber.get_real_volume() == pytest.approx(np.pi * ENVELOPE_RADIUS**2 * 0.6)


def test_folded_fiber_keeps_voxels_covered_beyond_an_end_plane():
    # Hairpin polyline: up along x = 0.5, across, then back down along x = 0.58,
    # ending at z = 0.6. The probe voxel lies below that end plane: its nearest
    # segment is the end cap (rejected), but the first leg covers it.
    centerline = np.array([[0.5, 0.5, 0.2], [0.5, 0.5, 0.8],
                           [0.58, 0.5, 0.8], [0.58, 0.5, 0.6]])
    n, r = 100, 0.08
    grid = np.zeros((n, n, n), dtype=np.uint8)
    h = 1.0 / n
    voxelizer._raster_fiber_numba(grid, centerline, r * r, 0, n, 0, n, 0, n, h, h, h, TAG_FIBER)

    def centre(i, j, k):
        return np.array([i + 0.5, j + 0.5, k + 0.5]) * h

    covered, outside, inside_leg = (55, 49, 56), (60, 49, 56), (60, 49, 61)
    end = centerline[-1]
    first_leg_dist = np.hypot(*(centre(*covered) - [0.5, 0.5, 0.0])[:2])
    # The probe is closer to the end point than to the first leg, yet inside the first leg.
    assert np.linalg.norm(centre(*covered) - end) < first_leg_dist < r
    assert grid[covered] == TAG_FIBER
    # Beyond the end plane and covered by nothing else: not tagged (no cap).
    assert np.linalg.norm(centre(*outside) - end) < r
    assert grid[outside] == 0
    # Inside the last leg, above its end: tagged.
    assert grid[inside_leg] == TAG_FIBER


def test_curved_fiber_volume_matches_swept_disk():
    # Quarter torus: the swept-disk volume is pi r^2 * (R theta) (Pappus). Joins
    # must neither add caps at the ends nor lose volume in the bend.
    r, R = 0.08, 0.3
    theta = np.linspace(0.0, np.pi / 2, 100)
    centerline = np.ascontiguousarray(np.stack(
        [0.2 + R * np.cos(theta), 0.2 + R * np.sin(theta), np.full_like(theta, 0.5)], axis=1))
    n = 128
    grid = np.zeros((n, n, n), dtype=np.uint8)
    h = 1.0 / n
    voxelizer._raster_fiber_numba(grid, centerline, r * r, 0, n, 0, n, 0, n, h, h, h, TAG_FIBER)
    volume = np.count_nonzero(grid == TAG_FIBER) * h ** 3
    exact = np.pi * r * r * np.sum(np.linalg.norm(np.diff(centerline, axis=0), axis=1))
    assert abs(volume - exact) / exact < 0.025


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

    # D10 (R10): refusal with the validity diagnostic, exit code 1, before generation and exports.
    assert result.returncode == 1
    assert "does not support the 'superelliptical' cross-section" in result.stderr
    assert "use --section circular" in result.stderr
    assert "PHASE 1" not in result.stderr
    assert sorted(p.name for p in tmp_path.iterdir()) == ["RVE_Result_validity.json"]
