"""
@file voxelizer.py
@brief Génération de maillage voxelisé (FFT) et export PGM (Portable Gray Map).
@details Tags standard : Matrice=0, Fibres=128, Pores=255.
L'algorithme utilise une approche de rastérisation par Bounding Box (BBox) 
accélérée par Numba pour tester l'appartenance des voxels aux cylindres (fibres) 
ou aux sphères (pores).
"""

import numpy as np
import logging
from numba import njit # type: ignore

## @var logger
#  @brief Logger pour le module de voxélisation.
logger = logging.getLogger(__name__)

## @var TAG_MATRIX
#  @brief Tag pour la matrice (fond).
TAG_MATRIX = 0
## @var TAG_FIBER
#  @brief Tag pour les fibres.
TAG_FIBER = 128
## @var TAG_VOID
#  @brief Tag pour les pores (porosités).
TAG_VOID = 255


class Voxelizer:
    """
    @class Voxelizer
    @brief Classe responsable de la conversion de la géométrie continue en une grille de voxels discrète.
    """

    def __init__(self, config):
        """! @brief Placeholder.
        @param self, config 
        @return None
        """
        """
        @brief Initialise le voxélisateur.
        @param config Objet de configuration contenant les dimensions de la boîte (box_dims).
        """
        ## @var config
        #  @brief Configuration du problème.
        self.config = config

    def create_grid(self, fibers, voids, res: int):
        """! @brief Placeholder.
        @param self, fibers, voids, res: int 
        @return None
        """
        """
        @brief Génère une grille 3D voxelisée avec des tags pour simulation FFT ou analyse d'image.
        
        Réalise la rastérisation des fibres (incluant les ghosts périodiques) et des pores.
        La priorité de taggage est : Pores > Fibres > Matrice.

        @param fibers list Liste des objets Fiber (racines).
        @param voids list Liste des objets Void (porosités).
        @param res int Résolution de la grille (nombre de voxels par axe, ex: 256 pour 256^3).

        @return np.ndarray grid Numpy array 3D de type uint8 contenant les tags.
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
        """! @brief Placeholder.
        @param self, grid, output_prefix 
        @return None
        """
        """
        @brief Sauvegarde la grille 3D sous forme de série de coupes 2D au format PGM (Portable Gray Map).
        
        Le format utilisé est P2 (ASCII). Les fichiers sont nommés slice0000.pgm, slice0001.pgm, etc.
        @param grid np.ndarray La grille 3D (numpy array) à exporter.
        @param output_prefix str Préfixe pour le nom des fichiers.
        @return None
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
    """! @brief Placeholder.
        @param grid, centerline, r_sq, imin, imax, jmin, jmax, kmin, kmax, dx, dy, dz, tag 
        @return None
    """
    """
    @brief Kernel Numba pour la rastérisation d'une fibre segmentée.
    @details Calcule la distance minimale entre le centre de chaque voxel et les segments de la fibre.
    
    @param grid Référence vers la grille 3D.
    @param centerline Points de la ligne moyenne de la fibre.
    @param r_sq Carré du rayon de la fibre.
    @param imin, imax, jmin, jmax, kmin, kmax Bornes de la BBox en indices voxels.
    @param dx, dy, dz Dimensions physiques d'un voxel.
    @param tag Valeur du tag à appliquer.
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
    """! @brief Placeholder.
        @param grid, center, r_sq, imin, imax, jmin, jmax, kmin, kmax, dx, dy, dz, tag 
        @return None
    """
    """
    @brief Kernel Numba pour la rastérisation d'une sphère (pore).
    
    @param grid Référence vers la grille 3D.
    @param center Coordonnées (x, y, z) du centre de la sphère.
    @param r_sq Carré du rayon de la sphère.
    @param imin, imax, jmin, jmax, kmin, kmax Bornes de la BBox en indices voxels.
    @param dx, dy, dz Dimensions physiques d'un voxel.
    @param tag Valeur du tag à appliquer.
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
