# SPDX-FileCopyrightText: 2026 Devine Ngouloubi <exauce-devine.ngouloubi@unicaen.fr>
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
export/abaqus_exporter.py
Export of the voxel grid to an Abaqus input file (.inp): one C3D8 hexahedron per voxel.

The mesh is conforming by construction, its opposite faces carry matching nodes, tied by
periodic equations, and it is the same discretisation as the FFT grid, so a RVE can be
computed by FFT and by Abaqus on identical voxels.

Periodicity uses the corner nodes, so no node is added (the Learning Edition is limited to
1000 nodes, i.e. 9 voxels per axis): C0 = (0,0,0), CX = (nx,0,0), CY = (0,ny,0), CZ = (0,0,nz).
Every node on an upper face (i = nx, j = ny or k = nz), except CX, CY and CZ, is tied to its
image (i mod nx, j mod ny, k mod nz):
    u(n) - u(image) - a [u(CX) - u(C0)] - b [u(CY) - u(C0)] - c [u(CZ) - u(C0)] = 0
with a, b, c = 1 on the upper face x, y, z. Each node is eliminated once, edges and corners
included. A macroscopic strain E is applied by fixing C0 and prescribing
u(CX) = Lx E e1, u(CY) = Ly E e2, u(CZ) = Lz E e3; the mean stress is
Sigma_ia = RF_i(Ca) L_a / V.
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

    def write(self, grid: np.ndarray, inp_path: str, periodic: bool = True) -> dict:
        """!
        @brief Writes the grid to inp_path.

        Voxel grid[i, j, k] spans [i*dx, (i+1)*dx] x [j*dy, (j+1)*dy] x [k*dz, (k+1)*dz], as in
        the voxeliser. Node (i, j, k) has label 1 + i + (nx+1)*(j + (ny+1)*k) and element
        (i, j, k) has label 1 + i + nx*(j + ny*k). The part "VER" holds the element sets
        MATRIX, INCLUSIONS and PORES (non-empty ones only) and the node sets X0, X1, Y0, Y1,
        Z0, Z1 of the box faces and C0, CX, CY, CZ of the corners; the assembly holds one
        instance "VER-1". Materials, sections and steps are left to the user.

        @param grid np.ndarray uint8 grid (nx, ny, nz) with the voxeliser tags.
        @param inp_path str Path of the .inp file.
        @param periodic bool Write the periodic equations (module docstring).
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
            corners = {"C0": (0, 0, 0), "CX": (nx, 0, 0), "CY": (0, ny, 0), "CZ": (0, 0, nz)}
            for name, (a, b, c) in corners.items():
                fh.write(f"*Nset, nset={name}\n{1 + a + (nx + 1) * (b + (ny + 1) * c)}\n")
            n_equations = self._write_periodic_equations(fh, nx, ny, nz) if periodic else 0
            fh.write("*End Part\n*Assembly, name=Assembly\n*Instance, name=VER-1, part=VER\n"
                     "*End Instance\n*End Assembly\n")

        summary = {"nodes": int(nodes.shape[0]), "elements": int(elements.shape[0]), "sets": counts,
                   "periodic_equations": n_equations}
        logger.info(f"Abaqus voxel export: {inp_path} ({summary['nodes']} nodes, "
                    f"{summary['elements']} C3D8, {counts}, {n_equations} periodic equations)")
        return summary

    @staticmethod
    def _write_periodic_equations(fh, nx, ny, nz):
        """!
        @brief Writes the periodic equations of the module docstring, three per tied node.
        @param fh file Open .inp file.
        @param nx int Voxels along x (same for ny, nz).
        @return int Number of equations written.
        """
        label = lambda i, j, k: 1 + i + (nx + 1) * (j + (ny + 1) * k)
        c0, cx, cy, cz = label(0, 0, 0), label(nx, 0, 0), label(0, ny, 0), label(0, 0, nz)
        lines = []
        for k in range(nz + 1):
            for j in range(ny + 1):
                for i in range(nx + 1):
                    a, b, c = int(i == nx), int(j == ny), int(k == nz)
                    node = label(i, j, k)
                    if not (a or b or c) or node in (cx, cy, cz):
                        continue
                    # node, image, corners; the image is C0 for the corner (nx, ny, nz) and the
                    # edges through it, so coefficients are summed per node
                    terms = {node: 1.0}
                    image = label(i % nx, j % ny, k % nz)
                    terms[image] = terms.get(image, 0.0) - 1.0
                    for flag, corner in ((a, cx), (b, cy), (c, cz)):
                        if flag:
                            terms[corner] = terms.get(corner, 0.0) - 1.0
                            terms[c0] = terms.get(c0, 0.0) + 1.0
                    terms = [(n, v) for n, v in terms.items() if v != 0.0]
                    for dof in (1, 2, 3):
                        lines.append(f"{len(terms)}\n")
                        # at most four terms per data line
                        for start in range(0, len(terms), 4):
                            lines.append(", ".join(f"{n}, {dof}, {v:g}" for n, v in terms[start:start + 4]) + "\n")
        if lines:
            fh.write("*Equation\n")
            fh.writelines(lines)
        return sum(1 for line in lines if "," not in line)

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
