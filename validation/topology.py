## @file topology.py
#  @brief Unified topological validator with periodicity management.
#
#  Merges clearance auditing (Numba segment-segment + MIC) and 
#  gap distribution analysis (cKDTree).

import numpy as np
import logging
from typing import List, Dict
from scipy.spatial import cKDTree

from collision.detector_math import min_dist_periodic_segments_numba

## @var logger
#  @brief Logger for the topology module.
logger = logging.getLogger(__name__)


class TopologyValidator:
    """!
    @brief Class for validating the topological integrity of fiber microstructures.
    
    Provides methods to check for intersections and analyze the distribution 
    of distances between fibers.
    """
    def __init__(self, box_dims):
        """!
        @brief Initializes the validator with domain dimensions.
        @param box_dims tuple or np.ndarray (Lx, Ly, Lz) for periodic distance handling.
        """
        self.dims = np.array(box_dims, dtype=float)  ##< Domain dimensions (Lx, Ly, Lz).

    def check_clearance(self, fibers: List) -> float:
        """!
        @brief Calculates the exact minimum clearance (empty space) between all fiber pairs.
        @details Uses exact segment-to-segment distance with Minimum Image Convention (MIC).
        A result > 0 indicates no intersections.
        @param fibers List List of Fiber objects to audit.
        @return float Minimum gap value (negative if fibers intersect).
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
        """!
        @brief Estimates the distribution of inter-fiber distances (Clearance).
        @details Fast statistical approximation using a cKDTree on the fiber centerlines.
        @param fibers List List of Fiber objects.
        @param sample_points int Number of points to sample (currently uses step slicing).
        @return Dict Dictionary containing min_gap, mean_gap, and histogram data.
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
