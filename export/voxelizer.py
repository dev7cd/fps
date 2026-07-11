"""
@file voxelizer.py
@brief Voxelised mesh generation (FFT) and PGM (Portable Gray Map) export.
@details Standard tags: matrix=0, fibers=128, pores=255.
The algorithm uses a bounding-box (BBox) rasterisation approach accelerated
by Numba to test whether voxels belong to the cylinders (fibers) or the
spheres (pores).
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
        """
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

        The format used is P2 (ASCII). Files are named slice0000.pgm, slice0001.pgm, etc.
        @param grid np.ndarray The 3D grid (numpy array) to export.
        @param output_prefix str Prefix for the file names.
        @return None
        """
        nx, ny, nz = grid.shape
        max_val = int(np.max(grid))

        logger.info(f"Exporting PGM slices to slice****.pgm (max_val={max_val})")

        header = f"P2\n{ny} {nx}\n{max_val}\n"

        for z in range(nz):
            slice_data = grid[:, :, z].T

            fname = f"slice{z:04d}.pgm"
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
    @details Computes the minimum distance between each voxel centre and the fiber segments.

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

                for idx in range(n_pts - 1):
                    p1 = centerline[idx]
                    p2 = centerline[idx + 1]

                    vx = p2[0] - p1[0]
                    vy = p2[1] - p1[1]
                    vz = p2[2] - p1[2]

                    wx = x - p1[0]
                    wy = y - p1[1]
                    wz = z - p1[2]

                    c1 = wx * vx + wy * vy + wz * vz
                    c2 = vx * vx + vy * vy + vz * vz

                    if c2 <= 1e-12:
                        b = 0.0
                    else:
                        b = c1 / c2

                    if b < 0.0:
                        b = 0.0
                    elif b > 1.0:
                        b = 1.0

                    proj_x = p1[0] + b * vx
                    proj_y = p1[1] + b * vy
                    proj_z = p1[2] + b * vz

                    d_sq = (x - proj_x) ** 2 + (y - proj_y) ** 2 + (z - proj_z) ** 2

                    if d_sq <= r_sq:
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
