# SPDX-FileCopyrightText: 2026 Devine Ngouloubi <exauce-devine.ngouloubi@unicaen.fr>
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
export/abaqus_exporter.py
Export of the voxel grid to an Abaqus input file (.inp): one C3D8 hexahedron per voxel.

The mesh is conforming by construction, its opposite faces carry matching nodes (ready for
periodic constraints), and it is the same discretisation as the FFT grid, so a RVE can be
computed by FFT and by Abaqus on identical voxels.
"""

import logging

import numpy as np

from export.voxelizer import TAG_FIBER, TAG_MATRIX, TAG_VOID

## @var logger
#  @brief Logger for the abaqus_exporter module.
logger = logging.getLogger(__name__)

## @var PHASE_SETS
#  @brief Element set name of each voxel tag.
PHASE_SETS = {TAG_MATRIX: "MATRIX", TAG_FIBER: "INCLUSIONS", TAG_VOID: "PORES"}


class AbaqusVoxelExporter:
    """!
    @class AbaqusVoxelExporter
    @brief Writes a voxel grid as an Abaqus part (C3D8 elements, phase and face sets).
    """

    def __init__(self, config):
        """!
        @brief Initialises the exporter.
        @param config FiberPackingConfig Configuration object holding box_dims.
        """
        self.config = config  ##< Simulation configuration.

    def write(self, grid: np.ndarray, inp_path: str) -> dict:
        """!
        @brief Writes the grid to inp_path.

        Voxel grid[i, j, k] spans [i*dx, (i+1)*dx] x [j*dy, (j+1)*dy] x [k*dz, (k+1)*dz], as in
        the voxeliser. Node (i, j, k) has label 1 + i + (nx+1)*(j + (ny+1)*k) and element
        (i, j, k) has label 1 + i + nx*(j + ny*k). The part "VER" holds the element sets
        MATRIX, INCLUSIONS and PORES (non-empty ones only) and the node sets X0, X1, Y0, Y1,
        Z0, Z1 of the box faces; the assembly holds one instance "VER-1". Materials, sections
        and steps are left to the user.

        @param grid np.ndarray uint8 grid (nx, ny, nz) with the voxeliser tags.
        @param inp_path str Path of the .inp file.
        @return dict Number of nodes, elements and elements per set.
        """
        nx, ny, nz = grid.shape
        dims = np.asarray(self.config.box_dims, dtype=float)
        dx, dy, dz = dims / np.array([nx, ny, nz])
        unknown = set(np.unique(grid).tolist()) - set(PHASE_SETS)
        if unknown:
            raise ValueError(f"Unknown voxel tags {sorted(unknown)}")

        # Nodes, x fastest
        k, j, i = np.meshgrid(np.arange(nz + 1), np.arange(ny + 1), np.arange(nx + 1), indexing="ij")
        node_ids = 1 + i + (nx + 1) * (j + (ny + 1) * k)
        nodes = np.column_stack([node_ids.ravel(), i.ravel() * dx, j.ravel() * dy, k.ravel() * dz])

        # Elements, x fastest: bottom face (z = k) counterclockwise, then top face
        k, j, i = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij")
        i, j, k = i.ravel(), j.ravel(), k.ravel()
        n = lambda a, b, c: 1 + a + (nx + 1) * (b + (ny + 1) * c)
        elements = np.column_stack([
            1 + i + nx * (j + ny * k),
            n(i, j, k), n(i + 1, j, k), n(i + 1, j + 1, k), n(i, j + 1, k),
            n(i, j, k + 1), n(i + 1, j, k + 1), n(i + 1, j + 1, k + 1), n(i, j + 1, k + 1),
        ])
        tags = grid[i, j, k]

        counts = {}
        with open(inp_path, "w") as fh:
            fh.write("*Heading\nFPS voxel RVE: %d x %d x %d C3D8, box %s\n" % (nx, ny, nz, dims.tolist()))
            fh.write("*Part, name=VER\n*Node\n")
            np.savetxt(fh, nodes, fmt=["%d", "%.10g", "%.10g", "%.10g"], delimiter=", ")
            fh.write("*Element, type=C3D8\n")
            np.savetxt(fh, elements, fmt="%d", delimiter=", ")
            for tag, name in PHASE_SETS.items():
                ids = elements[tags == tag, 0]
                if ids.size:
                    counts[name] = int(ids.size)
                    self._write_set(fh, f"*Elset, elset={name}", ids)
            I, J, K = (a.ravel() for a in np.meshgrid(np.arange(nx + 1), np.arange(ny + 1), np.arange(nz + 1), indexing="ij"))
            all_nodes = 1 + I + (nx + 1) * (J + (ny + 1) * K)
            for name, mask in (("X0", I == 0), ("X1", I == nx), ("Y0", J == 0), ("Y1", J == ny),
                               ("Z0", K == 0), ("Z1", K == nz)):
                self._write_set(fh, f"*Nset, nset={name}", np.sort(all_nodes[mask]))
            fh.write("*End Part\n*Assembly, name=Assembly\n*Instance, name=VER-1, part=VER\n"
                     "*End Instance\n*End Assembly\n")

        summary = {"nodes": int(nodes.shape[0]), "elements": int(elements.shape[0]), "sets": counts}
        logger.info(f"Abaqus voxel export: {inp_path} ({summary['nodes']} nodes, "
                    f"{summary['elements']} C3D8, {counts})")
        return summary

    @staticmethod
    def _write_set(fh, header, ids):
        """!
        @brief Writes a node or element set, 16 labels per line (Abaqus data line limit).
        @param fh file Open .inp file.
        @param header str Keyword line, e.g. "*Elset, elset=MATRIX".
        @param ids np.ndarray Labels.
        """
        fh.write(header + "\n")
        full = ids.size // 16 * 16
        if full:
            np.savetxt(fh, ids[:full].reshape(-1, 16), fmt="%d", delimiter=", ")
        if ids.size > full:
            fh.write(", ".join(str(v) for v in ids[full:]) + "\n")
