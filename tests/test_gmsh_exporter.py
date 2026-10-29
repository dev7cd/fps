# SPDX-FileCopyrightText: 2026 Devine Ngouloubi <exauce-devine.ngouloubi@unicaen.fr>
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Tests of the CAD and mesh exports (GmshExporter) on a curved fiber crossing a box face.

Regressions covered: free spline points and curves written to the STEP (empty bodies in
Abaqus), fiber pieces also assigned to MATRIX, and the export check record.
"""
import json

import numpy as np
import pytest

from core.config import FiberPackingConfig
from core.fiber import Fiber
from export.gmsh_exporter import GmshExporter

# Curved fiber leaving the box through x = 1: one root and one periodic image are cut.
POINTS = np.array([
    [0.55, 0.50, 0.45],
    [0.75, 0.55, 0.50],
    [0.95, 0.50, 0.55],
    [1.15, 0.45, 0.50],
    [1.35, 0.50, 0.45],
])
RADIUS = 0.08


@pytest.fixture(scope="module")
def export(tmp_path_factory):
    """Runs the export once and returns (check record, output prefix)."""
    config = FiberPackingConfig()
    config.box_dims = np.array([1.0, 1.0, 1.0])
    config.fiber_radius = RADIUS
    config.enable_adaptive_mesh = False
    config.export_abaqus = True
    fiber = Fiber(1, POINTS.copy(), RADIUS, {})
    prefix = str(tmp_path_factory.mktemp("gmsh") / "rve")
    record = GmshExporter(config).generate_mesh([fiber], [], prefix)
    return record, prefix


def test_export_check_passes(export):
    record, prefix = export
    assert record["status"] == "passed", record["cause"]
    with open(f"{prefix}_export_check.json") as fh:
        assert json.load(fh)["status"] == "passed"


def test_matrix_and_inclusions_partition_the_box(export):
    checks = export[0]["checks"]
    assert checks["n_matrix_solids"] == 1
    assert checks["n_inclusion_solids"] == 2
    assert abs(checks["mesh_volume_error"]) < 1e-9
    assert abs(checks["inclusion_volume_error"]) < 0.05


def test_step_holds_only_the_solids(export):
    items = export[0]["checks"]["step_items"]
    assert items["MANIFOLD_SOLID_BREP"] == 3
    assert items["GEOMETRIC_CURVE_SET"] == items["GEOMETRIC_SET"] == items["SHELL_BASED_SURFACE_MODEL"] == 0


def test_abaqus_input_has_the_phase_sets(export):
    with open(f"{export[1]}.inp") as fh:
        text = fh.read()
    assert "*ELSET,ELSET=MATRIX" in text and "*ELSET,ELSET=INCLUSIONS" in text
    assert "type=C3D4" in text
