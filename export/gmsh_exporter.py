# SPDX-FileCopyrightText: 2026 Devine Ngouloubi <exauce-devine.ngouloubi@unicaen.fr>
# SPDX-License-Identifier: AGPL-3.0-or-later

import gmsh
import json
import re
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

    def generate_mesh(self, fibers, voids, output_path) -> dict:
        """!
        @brief Generates the CAD geometry (.step) and the mesh (.msh) robustly, then checks them.

        Performs the following steps:
        1. Creation of the matrix cube.
        2. Creation of the fibers (sweep/pipe) and pores (spheres).
        3. Boolean fragmentation to ensure interface conformity.
        4. Fragment sorting (clipping) and definition of the physical groups.
        5. Volume meshing with adaptive size fields.
        6. Check of the exports, written to {output_path}_export_check.json (see check_exports).

        @param fibers List[Fiber] List of fibers to mesh.
        @param voids List[Void] List of pores to mesh.
        @param output_path str Path (without extension) for the output files.
        @return dict Export check record; its "status" is "passed" or "failed".
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
            gmsh.finalize()
            return self._write_check(output_path, {"status": "failed", "cause": f"matrix creation: {e}"})

        inclusion_volumes = []
        failed_tubes = []  # (fiber id, shift, error): a fiber missing from the exports must be reported

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
                    # (path_wire is a wire tag, not a curve tag: remove the spline itself)
                    occ.remove([(2, section_face), (1, path_spline)], recursive=True)
                    # The interior spline points are not on the wire boundary: without this they stay
                    # as free vertices, written to the STEP (Abaqus reads each one as an empty body)
                    occ.remove([(0, t) for t in path_pts[1:-1]])

                except Exception as e:
                    # Continue with the other tubes; the check marks the export as failed
                    failed_tubes.append((f.id, [float(c) for c in s], str(e)))
                    logger.warning(f"Tube of fiber {f.id} (shift {s}) failed: {e}")
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
            return self._write_check(output_path, {"status": "failed", "cause": f"fragmentation: {e}"})

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
        # A piece inside an inclusion is a child of both the box and the inclusion: it belongs to
        # the inclusions only, otherwise MATRIX would cover the whole box.
        inclusion_children = {tag for child_list in out_map[1:] for _, tag in child_list}

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
                            if tag not in inclusion_children:
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

        mesh_error = None
        try:
            gmsh.model.mesh.generate(3)

            # Periodic mesh (constraints on opposite faces)
            if getattr(self.config, 'enable_periodic_mesh', False):
                self._apply_periodic_constraints(dims)
                # Regenerate after the periodic constraints
                gmsh.model.mesh.generate(3)

            gmsh.write(f"{output_path}.msh")
            # Abaqus input file: nodes, elements and one element set per physical group
            if getattr(self.config, 'export_abaqus', False):
                gmsh.write(f"{output_path}.inp")
        except Exception as e:
            mesh_error = str(e)
            logger.error(f"Meshing failed: {e}")

        expected = sum(f.get_real_volume() for f in fibers) if not voids else None
        record = self.check_exports(output_path, dims, len(matrix_tags), len(inclusion_tags),
                                    failed_tubes, mesh_error, expected)
        gmsh.finalize()
        return self._write_check(output_path, record)

    def check_exports(self, output_path, dims, n_matrix, n_inclusions, failed_tubes,
                      mesh_error, expected_inclusion_volume, mesh_tolerance=1e-6, inclusion_tolerance=0.05):
        """!
        @brief Checks the exports of the current model: tubes built, STEP read back, mesh volumes.

        Volumes are measured on the tetrahedral mesh, not with occ.getMass: its integration over
        the trimmed swept faces is off by up to a few 1e-3 of the box volume, while the mesh of a
        conforming partition fills the box exactly.
        - every tube was built (no fiber silently missing);
        - the STEP holds one solid per exported volume and no loose geometry (curve or surface
          sets, which Abaqus imports as empty bodies), and reads back with the same solids;
        - mesh volume of MATRIX + INCLUSIONS equals the box volume (relative mesh_tolerance);
        - without pores, mesh volume of INCLUSIONS matches the real fiber volume (relative
          inclusion_tolerance): a coarse check against a lost piece, wide enough for the inscribed
          polygonal sections (about -3 % without the adaptive refinement).

        @param output_path str Path (without extension) of the exports.
        @param dims np.ndarray Box dimensions.
        @param n_matrix int Number of matrix solids written.
        @param n_inclusions int Number of inclusion solids written.
        @param failed_tubes list (fiber id, shift, error) of the tubes that could not be built.
        @param mesh_error Optional[str] Meshing error, None if the mesh was written.
        @param expected_inclusion_volume Optional[float] Real fiber volume, None if not checked.
        @param mesh_tolerance float Relative tolerance on the box volume.
        @param inclusion_tolerance float Relative tolerance on the inclusion volume.
        @return dict Check record with "status", "checks" and "cause".
        """
        box_volume = float(np.prod(dims))
        checks = {"failed_tubes": failed_tubes, "n_matrix_solids": n_matrix,
                  "n_inclusion_solids": n_inclusions}
        causes = []
        if failed_tubes:
            causes.append(f"{len(failed_tubes)} tube(s) not built")

        # Mesh volumes per physical group (current model)
        if mesh_error is None:
            node_tags, coords, _ = gmsh.model.mesh.getNodes()
            X = np.zeros((int(node_tags.max()) + 1, 3))
            X[node_tags.astype(int)] = coords.reshape(-1, 3)
            volumes = {}
            for dim, phys in gmsh.model.getPhysicalGroups(3):
                name = gmsh.model.getPhysicalName(dim, phys)
                volumes[name] = 0.0
                for tag in gmsh.model.getEntitiesForPhysicalGroup(dim, phys):
                    for etype, _, enodes in zip(*gmsh.model.mesh.getElements(3, tag)):
                        if etype != 4:  # linear tetrahedra only
                            causes.append(f"unsupported element type {etype} for the volume check")
                            continue
                        n = enodes.astype(int).reshape(-1, 4)
                        a, b, c, d = (X[n[:, i]] for i in range(4))
                        volumes[name] += float(np.abs(np.einsum('ij,ij->i', np.cross(b - a, c - a), d - a)).sum() / 6)
            checks["mesh_volumes"] = volumes
            total = sum(volumes.values())
            checks["mesh_volume_error"] = total / box_volume - 1
            if abs(checks["mesh_volume_error"]) > mesh_tolerance:
                causes.append(f"mesh volume {total:.9g} differs from the box volume {box_volume:.9g}")
            if expected_inclusion_volume:
                err = volumes.get("INCLUSIONS", 0.0) / expected_inclusion_volume - 1
                checks["inclusion_volume_error"] = err
                if abs(err) > inclusion_tolerance:
                    causes.append(f"inclusion volume off by {err:+.2%} from the real fiber volume")
        else:
            causes.append(f"meshing failed: {mesh_error}")

        # STEP: top-level items as written (Gmsh ignores loose curve sets when reading back)
        with open(f"{output_path}.step") as fh:
            text = fh.read()
        items = {k: len(re.findall(rf"=\s*{k}\s*\(", text))
                 for k in ("MANIFOLD_SOLID_BREP", "GEOMETRIC_CURVE_SET", "GEOMETRIC_SET", "SHELL_BASED_SURFACE_MODEL")}
        checks["step_items"] = items
        if items["MANIFOLD_SOLID_BREP"] != n_matrix + n_inclusions:
            causes.append(f"STEP holds {items['MANIFOLD_SOLID_BREP']} solids instead of {n_matrix + n_inclusions}")
        loose = sum(v for k, v in items.items() if k != "MANIFOLD_SOLID_BREP")
        if loose:
            causes.append(f"STEP holds {loose} loose geometric sets")

        # STEP read back in a separate model
        gmsh.model.add("step_check")
        try:
            gmsh.model.occ.importShapes(f"{output_path}.step")
            gmsh.model.occ.synchronize()
            checks["step_solids_read_back"] = len(gmsh.model.getEntities(3))
            if checks["step_solids_read_back"] != n_matrix + n_inclusions:
                causes.append(f"STEP reads back as {checks['step_solids_read_back']} solids")
        except Exception as e:
            causes.append(f"STEP read back failed: {e}")
        gmsh.model.remove()

        status = "failed" if causes else "passed"
        return {"status": status, "cause": "; ".join(causes), "checks": checks}

    @staticmethod
    def _write_check(output_path, record):
        """!
        @brief Writes the export check record to {output_path}_export_check.json and logs it.
        @param output_path str Path (without extension) of the exports.
        @param record dict Check record.
        @return dict The same record.
        """
        with open(f"{output_path}_export_check.json", "w") as fh:
            json.dump(record, fh, indent=2)
        if record["status"] == "passed":
            logger.info(f"Export check passed: {output_path}_export_check.json")
        else:
            logger.error(f"Export check failed ({record['cause']}): {output_path}_export_check.json")
        return record

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
            # Circle made of four arcs in the (N, B) plane. A disk would give a single periodic
            # lateral face; once trimmed by a box face it nearly grazes, Abaqus's STEP reader
            # keeps the wrong side of the seam and loses about half of the piece.
            r, N, B = fiber.radius, fiber.N[0], fiber.B[0]
            center = occ.addPoint(*p0)
            rim = [occ.addPoint(*(p0 + r * u)) for u in (N, B, -N, -B)]
            arcs = [occ.addCircleArc(rim[i], center, rim[(i + 1) % 4]) for i in range(4)]
            face = occ.addPlaneSurface([occ.addCurveLoop(arcs)])
            occ.remove([(0, center)])  # the center is not on the face boundary
            return face
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
            face = occ.addPlaneSurface([wire])
            occ.remove([(0, t) for t in pt_tags[1:-1]])  # free interior spline points
            return face

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
