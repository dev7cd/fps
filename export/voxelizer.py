# SPDX-FileCopyrightText: 2026 Devine Ngouloubi <exauce-devine.ngouloubi@unicaen.fr>
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
@file voxelizer.py
@brief Voxelised mesh generation (FFT) and PGM (Portable Gray Map) export.
@details Standard tags: matrix=0, fibers=128, pores=255.
The algorithm uses a bounding-box (BBox) rasterisation approach accelerated
by Numba to test whether voxels belong to the cylinders (fibers) or the
spheres (pores).

Supported domain: only fibers with a circular cross-section are voxelised.
The rasteriser uses the centerline and the scalar radius alone, so any other
section (e.g. superelliptical) would be exported as a circular tube; such
fibers are rejected with UnsupportedSectionError before any rasterisation.
"""

import numpy as np
import logging
from numba import njit # type: ignore

## @var logger
#  @brief Logger for the voxelisation module.
logger = logging.getLogger(__name__)

## @var TAG_MATRIX
#  @brief Tag for the matrix (background).
TAG_MATRIX = 0
## @var TAG_FIBER
#  @brief Tag for the fibers.
TAG_FIBER = 128
## @var TAG_VOID
#  @brief Tag for the pores (porosity).
TAG_VOID = 255

## @var SUPPORTED_SECTIONS
#  @brief Cross-section types that the voxel export represents faithfully.
SUPPORTED_SECTIONS = ("circular",)


class UnsupportedSectionError(ValueError):
    """!
    @class UnsupportedSectionError
    @brief Raised when the voxel export is requested for a cross-section it cannot represent.
    """


def check_voxel_section_type(section_type):
    """!
    @brief Rejects a cross-section type that the voxel export does not support.
    @param section_type str Declared section type (e.g. 'circular').
    @exception UnsupportedSectionError If the section type is not in SUPPORTED_SECTIONS.
    """
    if section_type not in SUPPORTED_SECTIONS:
        raise UnsupportedSectionError(
            f"Voxel export does not support the '{section_type}' cross-section: "
            f"only {', '.join(SUPPORTED_SECTIONS)} sections are voxelised."
        )


def check_voxel_fibers(fibers):
    """!
    @brief Rejects a fiber collection containing any unsupported cross-section.
    @param fibers list List of Fiber objects.
    @exception UnsupportedSectionError If at least one fiber has an unsupported section;
    the message lists the offending fiber ids.
    """
    unsupported = {}
    for f in fibers:
        if f.section_type not in SUPPORTED_SECTIONS:
            unsupported.setdefault(f.section_type, []).append(f.id)
    if unsupported:
        details = "; ".join(f"'{sec}' (fiber ids {ids})" for sec, ids in unsupported.items())
        raise UnsupportedSectionError(
            f"Voxel export does not support these cross-sections: {details}. "
            f"Only {', '.join(SUPPORTED_SECTIONS)} sections are voxelised; "
            f"no grid was produced."
        )


class Voxelizer:
    """!
    @class Voxelizer
    @brief Class responsible for converting the continuous geometry into a discrete voxel grid.
    """

    def __init__(self, config):
        """!
        @brief Initialises the voxelizer.
        @param config Configuration object holding the box dimensions (box_dims).
        """
        ## @var config
        #  @brief Problem configuration.
        self.config = config

    def create_grid(self, fibers, voids, res: int):
        """!
        @brief Generates a voxelised 3D grid with tags for FFT simulation or image analysis.

        Rasterises the fibers (including the periodic ghosts) and the pores.
        The tagging priority is: pores > fibers > matrix.

        @param fibers list List of Fiber objects (roots).
        @param voids list List of Void objects (porosity).
        @param res int Grid resolution (number of voxels per axis, e.g. 256 for 256^3).

        @return np.ndarray grid 3D numpy array of type uint8 containing the tags.
        @exception UnsupportedSectionError If a fiber has a non-circular section
        (checked before any rasterisation).
        """
        check_voxel_fibers(fibers)

        dims = np.array(self.config.box_dims)

        # Initialisation: everything is MATRIX (0)
        grid = np.full((res, res, res), TAG_MATRIX, dtype=np.uint8)

        # Compute voxel size
        voxel_size = dims / res
        dx, dy, dz = voxel_size

        logger.info(f"Accelerated voxelisation ({res}^3)...")

        from generation.periodicity import PeriodicManager

        # --- 1. RASTERISATION OF THE FIBERS (tag = 128) ---
        for f in fibers:
            shifts = PeriodicManager.generate_ghosts(f, self.config.box_dims)
            shifts.append(np.array([0., 0., 0.]))

            radius_sq = f.radius ** 2

            for s in shifts:
                centerline_shifted = f.centerline + s

                p_min = (f.bbox[0] + s) / voxel_size
                p_max = (f.bbox[1] + s) / voxel_size

                imin = max(0, int(np.floor(p_min[0]) - 1))
                imax = min(res, int(np.ceil(p_max[0]) + 1))
                jmin = max(0, int(np.floor(p_min[1]) - 1))
                jmax = min(res, int(np.ceil(p_max[1]) + 1))
                kmin = max(0, int(np.floor(p_min[2]) - 1))
                kmax = min(res, int(np.ceil(p_max[2]) + 1))

                _raster_fiber_numba(
                    grid, centerline_shifted, radius_sq,
                    imin, imax, jmin, jmax, kmin, kmax,
                    dx, dy, dz, tag=TAG_FIBER
                )

        # --- 2. RASTERISATION OF THE PORES (tag = 255) ---
        for v in voids:
            r_sq = v.radius ** 2
            p_min = (v.center - v.radius) / voxel_size
            p_max = (v.center + v.radius) / voxel_size

            imin = max(0, int(np.floor(p_min[0]) - 1))
            imax = min(res, int(np.ceil(p_max[0]) + 1))
            jmin = max(0, int(np.floor(p_min[1]) - 1))
            jmax = min(res, int(np.ceil(p_max[1]) + 1))
            kmin = max(0, int(np.floor(p_min[2]) - 1))
            kmax = min(res, int(np.ceil(p_max[2]) + 1))

            _raster_sphere_numba(
                grid, v.center, r_sq,
                imin, imax, jmin, jmax, kmin, kmax,
                dx, dy, dz, tag=TAG_VOID
            )

        n_fiber_voxels = np.sum(grid == TAG_FIBER)
        n_void_voxels = np.sum(grid == TAG_VOID)
        logger.info(f"Voxelisation finished: {n_fiber_voxels} fiber voxels, "
                     f"{n_void_voxels} pore voxels, {np.sum(grid == TAG_MATRIX)} matrix voxels")

        return grid

    def save_pgm(self, grid, output_prefix):
        """!
        @brief Saves the 3D grid as a series of 2D slices in PGM (Portable Gray Map) format.

        The format used is P2 (ASCII), one file per z index, named `<prefix>_slice0000.pgm`, ...
        next to the other exports, so that runs (and regenerations) never overwrite each other's
        slices. Grey levels are the voxel tags (matrix 0, fiber 128, pore 255) on a 0-255 scale.
        @param grid np.ndarray The 3D grid (numpy array) to export.
        @param output_prefix str Prefix for the file names.
        @return None
        """
        nx, ny, nz = grid.shape
        header = f"P2\n{nx} {ny}\n255\n"     # width = nx columns, height = ny rows

        logger.info(f"Exporting PGM slices to {output_prefix}_slice****.pgm")

        for z in range(nz):
            slice_data = grid[:, :, z].T      # (ny, nx): one row per y index

            fname = f"{output_prefix}_slice{z:04d}.pgm"
            try:
                with open(fname, 'w') as f:
                    f.write(header)
                    np.savetxt(f, slice_data, fmt='%d', delimiter=' ')
            except Exception as e:
                logger.error(f"PGM write error {fname}: {e}")


# --- NUMBA KERNELS (JIT) ---

@njit(cache=True, fastmath=True)
def _raster_fiber_numba(grid, centerline, r_sq, imin, imax, jmin, jmax, kmin, kmax, dx, dy, dz, tag):
    """!
    @brief Numba kernel for rasterising a segmented fiber.
    @details Rasterises the tube swept by a disk of radius r kept normal to the
    polyline (flat ends, no hemispherical caps), so that its volume matches
    Fiber.get_real_volume() = section area * centerline length.

    A point is inside when either
    - its projection falls *within* a segment (0 <= b <= 1, not clamped) at a
      distance <= r: a flat-ended cylinder per segment; or
    - it lies within r of an *internal* vertex and in the outer wedge of the
      bend, i.e. in front of the incoming segment's normal plane and behind
      the outgoing one's: the region the disk sweeps while turning.
    The two outer ends have no vertex contribution, so they stay flat however
    short the segments are (clamped capsules would rebuild an almost full cap
    from the joint next to each end). Every segment and vertex is tested on its
    own, so a portion of the same fiber passing nearby still covers a voxel
    beyond an end plane (folded fiber).

    @param grid Reference to the 3D grid.
    @param centerline Points of the fiber centerline.
    @param r_sq Square of the fiber radius.
    @param imin, imax, jmin, jmax, kmin, kmax Bounds of the BBox in voxel indices.
    @param dx, dy, dz Physical dimensions of a voxel.
    @param tag Tag value to apply.
    """
    n_pts = len(centerline)

    for i in range(imin, imax):
        x = (i + 0.5) * dx
        for j in range(jmin, jmax):
            y = (j + 0.5) * dy
            for k in range(kmin, kmax):
                z = (k + 0.5) * dz

                # If already tagged (by another fiber/inclusion), skip
                if grid[i, j, k] != 0:
                    continue

                is_inside = False

                # Flat-ended cylinders, one per segment.
                for idx in range(n_pts - 1):
                    p1 = centerline[idx]
                    p2 = centerline[idx + 1]
                    vx = p2[0] - p1[0]
                    vy = p2[1] - p1[1]
                    vz = p2[2] - p1[2]
                    c2 = vx * vx + vy * vy + vz * vz
                    if c2 <= 1e-12:
                        continue
                    wx = x - p1[0]
                    wy = y - p1[1]
                    wz = z - p1[2]
                    c1 = wx * vx + wy * vy + wz * vz
                    if c1 < 0.0 or c1 > c2:
                        continue
                    b = c1 / c2
                    d_sq = (wx - b * vx) ** 2 + (wy - b * vy) ** 2 + (wz - b * vz) ** 2
                    if d_sq <= r_sq:
                        is_inside = True
                        break

                # Outer wedges at internal vertices.
                if not is_inside:
                    for idx in range(1, n_pts - 1):
                        pv = centerline[idx]
                        wx = x - pv[0]
                        wy = y - pv[1]
                        wz = z - pv[2]
                        if wx * wx + wy * wy + wz * wz > r_sq:
                            continue
                        pin = centerline[idx - 1]
                        pout = centerline[idx + 1]
                        in_dot = wx * (pv[0] - pin[0]) + wy * (pv[1] - pin[1]) + wz * (pv[2] - pin[2])
                        out_dot = wx * (pout[0] - pv[0]) + wy * (pout[1] - pv[1]) + wz * (pout[2] - pv[2])
                        if in_dot >= 0.0 and out_dot <= 0.0:
                            is_inside = True
                            break

                if is_inside:
                    grid[i, j, k] = tag


@njit(cache=True, fastmath=True)
def _raster_sphere_numba(grid, center, r_sq, imin, imax, jmin, jmax, kmin, kmax, dx, dy, dz, tag):
    """!
    @brief Numba kernel for rasterising a sphere (pore).

    @param grid Reference to the 3D grid.
    @param center Coordinates (x, y, z) of the sphere centre.
    @param r_sq Square of the sphere radius.
    @param imin, imax, jmin, jmax, kmin, kmax Bounds of the BBox in voxel indices.
    @param dx, dy, dz Physical dimensions of a voxel.
    @param tag Tag value to apply.
    """
    for i in range(imin, imax):
        vx = (i + 0.5) * dx - center[0]
        for j in range(jmin, jmax):
            vy = (j + 0.5) * dy - center[1]
            for k in range(kmin, kmax):
                vz = (k + 0.5) * dz - center[2]

                dist_sq = vx * vx + vy * vy + vz * vz
                if dist_sq <= r_sq:
                    grid[i, j, k] = tag
