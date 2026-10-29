# SPDX-FileCopyrightText: 2026 Devine Ngouloubi <exauce-devine.ngouloubi@unicaen.fr>
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Tests of the Abaqus voxel export: one C3D8 per voxel at the voxel position, phase and face sets.
"""
import re

import numpy as np
import pytest

from core.config import FiberPackingConfig
from export.abaqus_exporter import AbaqusVoxelExporter
from export.voxelizer import TAG_FIBER, TAG_MATRIX, TAG_VOID

SHAPE = (3, 4, 5)
DIMS = np.array([1.5, 2.0, 1.0])


def read_inp(path):
    """Returns nodes {label: xyz}, elements {label: nodes} and sets {name: labels}."""
    nodes, elements, sets, block = {}, {}, {}, None
    for line in open(path):
        line = line.strip()
        if line.startswith("*"):
            m = re.match(r"\*(Elset|Nset), (?:elset|nset)=(\w+)", line)
            block = ("set", m.group(2)) if m else line.split(",")[0].lower()
            if m:
                sets[m.group(2)] = []
            continue
        values = [v for v in line.split(",") if v.strip()]
        if block == "*node":
            nodes[int(values[0])] = np.array(values[1:], float)
        elif block == "*element":
            elements[int(values[0])] = [int(v) for v in values[1:]]
        elif isinstance(block, tuple):
            sets[block[1]] += [int(v) for v in values]
    return nodes, elements, sets


def read_equations(path):
    """Returns the *Equation blocks as lists of (node, dof, coefficient)."""
    equations, inside, todo = [], False, 0
    for line in open(path):
        line = line.strip()
        if line.startswith("*"):
            inside = line.lower().startswith("*equation")
            continue
        if not inside:
            continue
        values = [v.strip() for v in line.split(",") if v.strip()]
        if todo == 0:
            todo = int(values[0])
            equations.append([])
            continue
        for p in range(0, len(values), 3):
            equations[-1].append((int(values[p]), int(values[p + 1]), float(values[p + 2])))
            todo -= 1
    return equations


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    grid = np.full(SHAPE, TAG_MATRIX, dtype=np.uint8)
    grid[1, 2, 3] = TAG_FIBER
    grid[0, :, 4] = TAG_FIBER
    grid[2, 3, 0] = TAG_VOID
    config = FiberPackingConfig()
    config.box_dims = DIMS
    path = str(tmp_path_factory.mktemp("abaqus") / "rve.inp")
    summary = AbaqusVoxelExporter(config).write(grid, path)
    return grid, summary, read_inp(path), path


def test_counts_and_phase_sets(exported):
    grid, summary, (nodes, elements, sets), _ = exported
    nx, ny, nz = SHAPE
    assert summary["nodes"] == len(nodes) == (nx + 1) * (ny + 1) * (nz + 1)
    assert summary["elements"] == len(elements) == nx * ny * nz
    assert len(sets["INCLUSIONS"]) == np.sum(grid == TAG_FIBER) == 5
    assert len(sets["PORES"]) == 1
    assert sorted(sets["MATRIX"] + sets["INCLUSIONS"] + sets["PORES"]) == sorted(elements)


def test_element_sits_on_its_voxel_with_positive_volume(exported):
    grid, _, (nodes, elements, sets), _ = exported
    nx, ny, _ = SHAPE
    h = DIMS / np.array(SHAPE)
    for (i, j, k), tag in np.ndenumerate(grid):
        label = 1 + i + nx * (j + ny * k)
        X = np.array([nodes[n] for n in elements[label]])
        assert np.allclose(X.mean(axis=0), (np.array([i, j, k]) + 0.5) * h)
        # C3D8 ordering: bottom face counterclockwise seen from +z, then the top face
        assert np.dot(np.cross(X[1] - X[0], X[3] - X[0]), X[4] - X[0]) > 0
    assert 1 + 1 + nx * (2 + ny * 3) in sets["INCLUSIONS"]


def test_face_node_sets(exported):
    _, _, (nodes, _, sets), _ = exported
    for name, axis, value in (("X0", 0, 0.0), ("X1", 0, DIMS[0]), ("Y0", 1, 0.0),
                              ("Y1", 1, DIMS[1]), ("Z0", 2, 0.0), ("Z1", 2, DIMS[2])):
        on_face = {n for n, x in nodes.items() if abs(x[axis] - value) < 1e-12}
        assert set(sets[name]) == on_face


def test_part_and_assembly_structure(exported):
    text = open(exported[3]).read()
    assert "*Part, name=VER" in text and "*End Part" in text
    assert "*Instance, name=VER-1, part=VER" in text


def test_periodic_equations_hold_for_periodic_fields(exported):
    """u = E x + w(x), w periodic, satisfies every equation; dependent nodes appear once."""
    path = exported[3]
    nodes, _, sets = read_inp(path)
    equations = read_equations(path)
    nx, ny, nz = SHAPE
    tied = [n for n, x in nodes.items() if np.any(np.isclose(x, DIMS))]
    corners = {sets[c][0] for c in ("CX", "CY", "CZ")}
    assert len(equations) == 3 * (len(tied) - 3)

    rng = np.random.default_rng(0)
    E = rng.normal(size=(3, 3))
    w = rng.normal(size=(nx, ny, nz, 3))  # periodic fluctuation, one value per canonical node
    h = DIMS / np.array(SHAPE)

    def u(n):
        x = nodes[n]
        i, j, k = np.rint(x / h).astype(int)
        return E @ x + w[i % nx, j % ny, k % nz]

    dependent = [eq[0][0] for eq in equations]
    assert len(set(dependent)) * 3 == len(dependent)
    assert not corners & set(dependent)
    others = {n for eq in equations for n, _, _ in eq[1:]}
    assert not others & set(dependent)
    for eq in equations:
        assert abs(sum(c * u(n)[dof - 1] for n, dof, c in eq)) < 1e-9


def test_corner_sets(exported):
    nodes, _, sets = read_inp(exported[3])
    expected = {"C0": [0, 0, 0], "CX": [DIMS[0], 0, 0], "CY": [0, DIMS[1], 0], "CZ": [0, 0, DIMS[2]]}
    for name, x in expected.items():
        assert np.allclose(nodes[sets[name][0]], x)


def test_no_equation_without_pbc(tmp_path):
    config = FiberPackingConfig()
    config.box_dims = DIMS
    path = str(tmp_path / "rve.inp")
    summary = AbaqusVoxelExporter(config).write(np.zeros(SHAPE, np.uint8), path, periodic=False)
    assert summary["periodic_equations"] == 0 and "*Equation" not in open(path).read()
