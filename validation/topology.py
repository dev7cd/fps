"""
validation/topology.py
Validateur topologique unifié avec gestion de la périodicité.
Fusionne l'audit de clearance (Numba segment-segment + MIC) et la distribution des gaps (cKDTree).
"""

import numpy as np
import logging
from typing import List, Dict
from scipy.spatial import cKDTree

from collision.detector_math import min_dist_periodic_segments_numba

logger = logging.getLogger(__name__)


class TopologyValidator:
    def __init__(self, box_dims):
        """
        box_dims : Tuple/Array (Lx, Ly, Lz) pour gérer la distance périodique.
        """
        self.dims = np.array(box_dims, dtype=float)

    def check_clearance(self, fibers: List) -> float:
        """
        Calcule le 'clearance' (espace vide) minimal entre toutes les paires de fibres.
        Utilise la distance segment-à-segment exacte avec Minimum Image Convention.
        Un résultat > 0 signifie aucune intersection.
        """
        min_gap = float('inf')

        n_fibers = len(fibers)
        logger.info(f"Audit topologique sur {n_fibers} fibres (segment-à-segment)...")

        min_gap_pair = (-1, -1)

        for i in range(n_fibers):
            pts1 = fibers[i].centerline
            r1 = fibers[i].radius

            for j in range(i + 1, n_fibers):
                pts2 = fibers[j].centerline
                r2 = fibers[j].radius

                d_center = min_dist_periodic_segments_numba(pts1, pts2, self.dims)

                gap = d_center - (r1 + r2)

                if gap < min_gap:
                    min_gap = gap
                    min_gap_pair = (fibers[i].id, fibers[j].id)

        if min_gap < 0:
            logger.error(f"ECHEC AUDIT : Intersection détectée ({min_gap:.6f}).")
        else:
            logger.info(f"SUCCES AUDIT : Gap minimal = {min_gap:.6f} (Paire {min_gap_pair})")

        return min_gap

    def compute_gap_distribution(self, fibers: List, sample_points: int = 20) -> Dict:
        """
        Estime la distribution des distances minimales inter-fibres (Clearance).
        Approximation statistique rapide via KDTree sur nuage de points.
        """
        all_pts = []
        all_radii = []
        fiber_ids = []

        for idx, f in enumerate(fibers):
            pts = f.centerline[::2]  # 1 point sur 2
            all_pts.extend(pts)
            all_radii.extend([f.radius] * len(pts))
            fiber_ids.extend([idx] * len(pts))

        if not all_pts:
            return {}

        data = np.array(all_pts)
        radii = np.array(all_radii)
        ids = np.array(fiber_ids)

        tree = cKDTree(data)
        dists, idxs = tree.query(data, k=10, workers=-1)

        gaps = []

        for i in range(len(data)):
            my_r = radii[i]
            my_id = ids[i]

            for k in range(1, 10):
                neighbor_idx = idxs[i, k]
                if neighbor_idx >= len(ids):
                    break

                neighbor_id = ids[neighbor_idx]
                if neighbor_id != my_id:
                    d_center = dists[i, k]
                    neighbor_r = radii[neighbor_idx]
                    gap = d_center - (my_r + neighbor_r)
                    gaps.append(gap)
                    break

        if not gaps:
            return {}

        return {
            "min_gap": float(np.min(gaps)),
            "mean_gap": float(np.mean(gaps)),
            "gap_histogram": np.histogram(gaps, bins=20)[0].tolist(),
            "gap_bins": np.histogram(gaps, bins=20)[1].tolist()
        }
