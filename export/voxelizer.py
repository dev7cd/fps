"""
export/voxelizer.py
Génération de maillage voxelisé (FFT) et export PGM (Portable Gray Map).
Tags standard : Matrice=0, Fibres=1, Pores=2
Optimisé avec Numba (Scan BBox + Early Exit).
"""

import numpy as np
import logging
from numba import njit

logger = logging.getLogger(__name__)

TAG_MATRIX = 0
TAG_FIBER = 128
TAG_VOID = 255


class Voxelizer:
    def __init__(self, config):
        self.config = config

    def create_grid(self, fibers, voids, res: int):
        """
        Génère une grille 3D voxelisée tags pour simulation FFT.

        Tags : matrice=0, fibres=1, pores=2

        Args:
            fibers: Liste des fibres (racines uniquement, les ghosts sont gérés en interne)
            voids: Liste des porosités
            res: Résolution (ex: 256 pour 256^3)

        Returns:
            grid: Numpy array 3D (uint8)
        """
        dims = np.array(self.config.box_dims)

        # Initialisation : Tout est MATRICE (0)
        grid = np.full((res, res, res), TAG_MATRIX, dtype=np.uint8)

        # Calcul taille voxel
        voxel_size = dims / res
        dx, dy, dz = voxel_size

        logger.info(f"Voxélisation accélérée ({res}^3)...")

        from generation.periodicity import PeriodicManager

        # --- 1. RASTÉRISATION DES FIBRES (Tag = 1) ---
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

        # --- 2. RASTÉRISATION DES PORES (Tag = 2) ---
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
        logger.info(f"Voxélisation terminée : {n_fiber_voxels} voxels fibres, "
                     f"{n_void_voxels} voxels pores, {np.sum(grid == TAG_MATRIX)} voxels matrice")

        return grid

    def save_pgm(self, grid, output_prefix):
        """
        Sauvegarde en slices PGM (Portable Gray Map, format P2 ASCII).
        """
        nx, ny, nz = grid.shape
        max_val = int(np.max(grid))

        logger.info(f"Export PGM slices vers slice****.pgm (max_val={max_val})")

        header = f"P2\n{ny} {nx}\n{max_val}\n"

        for z in range(nz):
            slice_data = grid[:, :, z].T

            fname = f"slice{z:04d}.pgm"
            try:
                with open(fname, 'w') as f:
                    f.write(header)
                    np.savetxt(f, slice_data, fmt='%d', delimiter=' ')
            except Exception as e:
                logger.error(f"Erreur d'écriture PGM {fname}: {e}")


# --- KERNELS NUMBA (JIT) ---

@njit(cache=True, fastmath=True)
def _raster_fiber_numba(grid, centerline, r_sq, imin, imax, jmin, jmax, kmin, kmax, dx, dy, dz, tag):
    """
    Rastérisation précise d'une fibre.
    Teste chaque voxel de la BBox contre tous les segments de la fibre.
    """
    n_pts = len(centerline)

    for i in range(imin, imax):
        x = (i + 0.5) * dx
        for j in range(jmin, jmax):
            y = (j + 0.5) * dy
            for k in range(kmin, kmax):
                z = (k + 0.5) * dz

                # Si déjà taggé (par une autre fibre/inclusion), on passe
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
    for i in range(imin, imax):
        vx = (i + 0.5) * dx - center[0]
        for j in range(jmin, jmax):
            vy = (j + 0.5) * dy - center[1]
            for k in range(kmin, kmax):
                vz = (k + 0.5) * dz - center[2]

                dist_sq = vx * vx + vy * vy + vz * vz
                if dist_sq <= r_sq:
                    grid[i, j, k] = tag
