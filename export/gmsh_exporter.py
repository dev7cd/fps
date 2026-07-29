# SPDX-FileCopyrightText: 2026 Devine Ngouloubi <exauce-devine.ngouloubi@unicaen.fr>
# SPDX-License-Identifier: AGPL-3.0-or-later

import gmsh
import numpy as np
import logging
from generation.periodicity import PeriodicManager

## @var logger
#  @brief Logger for the gmsh_exporter module.
logger = logging.getLogger(__name__)

class GmshExporter:
    """!
    @class GmshExporter
    @brief Class responsible for generating .step and .msh files via GMSH.
    """

    def __init__(self, config):
        """!
        @brief Initialises the exporter with the RVE configuration.
        @param config FiberPackingConfig Configuration object holding box_dims and the mesh parameters.
        """
        self.config = config ##< Simulation configuration

    def generate_mesh(self, fibers, voids, output_path) -> None:
        """!
        @brief Generates the CAD geometry (.step) and the mesh (.msh) robustly.

        Performs the following steps:
        1. Creation of the matrix cube.
        2. Creation of the fibers (sweep/pipe) and pores (spheres).
        3. Boolean fragmentation to ensure interface conformity.
        4. Fragment sorting (clipping) and definition of the physical groups.
        5. Volume meshing with adaptive size fields.

        @param fibers List[Fiber] List of fibers to mesh.
        @param voids List[Void] List of pores to mesh.
        @param output_path str Path (without extension) for the output files.
        @return None
        """
        # Clean initialisation
        try:
            gmsh.finalize()
        except:
            pass

        gmsh.initialize()
        gmsh.model.add("RVE_Composite_Final")
        occ = gmsh.model.occ # Shortcut to the kernel
        dims = self.config.box_dims

        logger.info("Starting the geometric construction...")

        # --- 1. CREATION OF THE MATRIX DOMAIN ---
        try:
            matrix_tag = occ.addBox(0, 0, 0, dims[0], dims[1], dims[2])
        except Exception as e:
            logger.error(f"Matrix creation error: {e}")
            return

        inclusion_volumes = []

        # --- 2. CREATION OF THE INCLUSIONS (root fibers + ghosts) ---
        count_f = 0
        for f in fibers:
            # Retrieve the periodic shift vectors
            shifts = PeriodicManager.generate_ghosts(f, dims)
            shifts.append(np.array([0, 0, 0], dtype=float))

            for s in shifts:
                try:
                    # Build the path
                    # Create the points and the spline directly
                    path_pts = [occ.addPoint(*(p + s)) for p in f.centerline]
                    if len(path_pts) < 2: continue

                    path_spline = occ.addSpline(path_pts)
                    path_wire = occ.addWire([path_spline])

                    # Section
                    section_face = self._create_section_face(f, occ, shift=s)

                    # Extrusion
                    # out is [(dim, tag)]
                    pipe_out = occ.addPipe([(2, section_face)], path_wire)
                    if pipe_out:
                        inclusion_volumes.append(pipe_out[0][1])

                    # Immediate cleanup of the temporary tools to relieve OCC
                    occ.remove([(2, section_face), (1, path_wire)], recursive=True)

                except Exception as e:
                    # Continue even if a fragment fails
                    continue
            count_f += 1

        logger.info(f"{len(inclusion_volumes)} fiber volumes generated (before cutting).")

        # --- 3. CREATION OF THE POROSITY ---
        for v in voids:
            try:
                sphere = occ.addSphere(v.center[0], v.center[1], v.center[2], v.radius)
                inclusion_volumes.append(sphere)
            except:
                pass

        # --- 4. FRAGMENTATION (boolean) ---
        occ.synchronize()
        logger.info("Boolean fragmentation in progress...")

        try:
            # fragment returns (new_objects, mapping parents->children)
            out, out_map = occ.fragment([(3, matrix_tag)], [(3, i) for i in inclusion_volumes])
        except Exception as e:
            logger.error(f"Critical GMSH fragmentation failure: {e}")
            gmsh.finalize()
            return

        occ.synchronize()

        # --- 5. ROBUST SORTING OF THE FRAGMENTS ---
        # This is where the "Unknown entity" error used to occur.
        # Strategy: query GMSH for what actually exists now.

        # Retrieve all live 3D entities after fragment
        all_live_entities_3d = set(tag for dim, tag in gmsh.model.getEntities(3))

        matrix_tags = []
        inclusion_tags = []
        to_remove = []

        eps = 1e-5

        # out_map[0]  -> children of the matrix (index 0 of fragment's first argument)
        # out_map[1:] -> children of the inclusions

        for input_index, child_list in enumerate(out_map):
            for dim, tag in child_list:
                # KEY SAFETY STEP: check whether the tag is live
                if tag not in all_live_entities_3d:
                    continue

                try:
                    # Compute the centre of mass
                    com = occ.getCenterOfMass(dim, tag)

                    # Clipping: is it inside the box [0, dims]?
                    is_inside = (
                        -eps <= com[0] <= dims[0] + eps and
                        -eps <= com[1] <= dims[1] + eps and
                        -eps <= com[2] <= dims[2] + eps
                    )

                    if is_inside:
                        if input_index == 0:
                            matrix_tags.append(tag)
                        else:
                            inclusion_tags.append(tag)
                    else:
                        to_remove.append((dim, tag))
                except Exception as e:
                    # If an error occurs here, the volume is probably corrupted/deleted
                    continue

        # Clean removal of what sticks out
        if to_remove:
            occ.remove(to_remove, recursive=True)

        occ.synchronize()

        # Physical groups
        if matrix_tags:
            gmsh.model.addPhysicalGroup(3, matrix_tags, tag=2, name="MATRIX")
        if inclusion_tags:
            gmsh.model.addPhysicalGroup(3, inclusion_tags, tag=1, name="INCLUSIONS")

        # Store the inclusion tags for the size fields
        self._inclusion_tags = inclusion_tags

        # --- 6. EXPORTS ---
        logger.info(f"Finalisation: {len(matrix_tags)} matrix fragments, {len(inclusion_tags)} inclusion fragments.")

        # CAD
        gmsh.write(f"{output_path}.step")

        # Mesh
        logger.info("Adaptive meshing...")

        # Base sizes
        min_size = self.config.fiber_radius / 6.0
        max_size = max(self.config.box_dims) / 8.0

        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", min_size)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", max_size)

        # Adaptive size fields (Distance + Threshold around the interfaces)
        if getattr(self.config, 'enable_adaptive_mesh', True):
            self._setup_size_fields(dims, min_size, max_size)
        else:
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 15)

        # 3D mesh optimiser
        gmsh.option.setNumber("Mesh.Algorithm3D", 10) # 1=Delaunay, 10=HXT (fast/parallel)

        try:
            gmsh.model.mesh.generate(3)

            # Periodic mesh (constraints on opposite faces)
            if getattr(self.config, 'enable_periodic_mesh', False):
                self._apply_periodic_constraints(dims)
                # Regenerate after the periodic constraints
                gmsh.model.mesh.generate(3)

            gmsh.write(f"{output_path}.msh")
        except Exception as e:
            logger.error(f"Meshing failed: {e}")

        gmsh.finalize()

    def _create_section_face(self, fiber, occ, shift):
        """!
        @brief Creates a 2D face representing the fiber cross-section.
        @param fiber Fiber Instance of the fiber.
        @param occ Any Shortcut to gmsh.model.occ.
        @param shift np.ndarray Translation vector (for the ghosts).
        @return int Tag of the created face.
        """
        p0 = fiber.centerline[0] + shift
        if fiber.section_type == 'circular':
            # Disk primitive: very stable
            # zAxis = tangent direction T, xAxis = normal N
            return occ.addDisk(p0[0], p0[1], p0[2], fiber.radius, fiber.radius,
                               zAxis=fiber.T[0], xAxis=fiber.N[0])
        else:
            # Complex spline
            pts = fiber.section_profile.generate_points(32)
            pt_tags = []
            for loc in pts:
                # P = P0 + u*N + v*B
                w = p0 + loc[0]*fiber.N[0] + loc[1]*fiber.B[0]
                pt_tags.append(occ.addPoint(*w))

            spl = occ.addSpline(pt_tags)
            lin = occ.addLine(pt_tags[-1], pt_tags[0])
            wire = occ.addWire([spl, lin])
            return occ.addPlaneSurface([wire])

    def _setup_size_fields(self, dims, min_size, max_size):
        """!
        @brief Configures adaptive Distance + Threshold size fields
        around the fiber/matrix interfaces.
        @param dims np.ndarray Box dimensions.
        @param min_size float Minimum mesh size (at the interfaces).
        @param max_size float Maximum mesh size (far from the fibers).
        """
        # Retrieve the inclusion surfaces
        inclusion_surfaces = []
        for tag in self._inclusion_tags:
            try:
                bounds = gmsh.model.getBoundary([(3, tag)], oriented=False)
                inclusion_surfaces.extend([b[1] for b in bounds])
            except:
                continue

        if not inclusion_surfaces:
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 15)
            return

        # Field 1: distance to the inclusion surfaces
        f_dist = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(f_dist, "SurfacesList", inclusion_surfaces)

        # Field 2: threshold based on the distance
        f_thresh = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(f_thresh, "InField", f_dist)
        gmsh.model.mesh.field.setNumber(f_thresh, "SizeMin", min_size)
        gmsh.model.mesh.field.setNumber(f_thresh, "SizeMax", max_size)
        gmsh.model.mesh.field.setNumber(f_thresh, "DistMin", self.config.fiber_radius * 0.5)
        gmsh.model.mesh.field.setNumber(f_thresh, "DistMax", self.config.fiber_radius * 3.0)

        # Use as the background mesh
        gmsh.model.mesh.field.setAsBackgroundMesh(f_thresh)

        # Disable the default size sources to avoid conflicts
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

    def _apply_periodic_constraints(self, dims):
        """!
        @brief Applies the periodic mesh constraints on the 3 pairs of faces.
        Face pairs: (X=0, X=Lx), (Y=0, Y=Ly), (Z=0, Z=Lz)
        @param dims np.ndarray Domain dimensions (Lx, Ly, Lz).
        """
        Lx, Ly, Lz = dims
        surfaces = gmsh.model.getEntities(2)

        eps = 1e-4

        # For each axis: identify master faces (position ~0) and slave faces (position ~L)
        face_pairs = [
            (0, Lx, [1, 0, 0, Lx,  0, 1, 0, 0,   0, 0, 1, 0,   0, 0, 0, 1]),  # X
            (1, Ly, [1, 0, 0, 0,   0, 1, 0, Ly,   0, 0, 1, 0,   0, 0, 0, 1]),  # Y
            (2, Lz, [1, 0, 0, 0,   0, 1, 0, 0,    0, 0, 1, Lz,  0, 0, 0, 1]),  # Z
        ]

        for axis_idx, length, transform in face_pairs:
            master_tags = []
            slave_tags = []

            for dim, tag in surfaces:
                try:
                    com = gmsh.model.occ.getCenterOfMass(dim, tag)
                except:
                    continue
                if abs(com[axis_idx]) < eps:
                    master_tags.append(tag)
                elif abs(com[axis_idx] - length) < eps:
                    slave_tags.append(tag)

            # Match each slave face with its corresponding master
            for st in slave_tags:
                try:
                    st_com = gmsh.model.occ.getCenterOfMass(2, st)
                except:
                    continue
                best_master = None
                best_dist = float('inf')

                for mt in master_tags:
                    try:
                        mt_com = gmsh.model.occ.getCenterOfMass(2, mt)
                    except:
                        continue
                    # The master must match geometrically (same position except on the axis)
                    diff = 0.0
                    for k in range(3):
                        if k == axis_idx:
                            continue
                        diff += (st_com[k] - mt_com[k]) ** 2
                    diff = diff ** 0.5
                    if diff < best_dist:
                        best_dist = diff
                        best_master = mt

                if best_master is not None and best_dist < eps * 10:
                    try:
                        gmsh.model.mesh.setPeriodic(2, [st], [best_master], transform)
                    except Exception as e:
                        logger.debug(f"setPeriodic failed for face {st}->{best_master}: {e}")
