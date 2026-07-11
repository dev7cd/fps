## @file statistics.py
#  @brief Spatial statistical descriptors for UD microstructure validation.
#
#  This module provides tools to analyze the spatial distribution of fibers
#  using 2D projections of their centroids, which is standard for continuous
#  fiber composites. It includes NND, Ripley's K, Pair Correlation, and Voronoi
#  analysis.

import numpy as np
import logging
from typing import List, Dict, Optional
from scipy.spatial import cKDTree, Voronoi

## @var logger
#  @brief Logger for the statistics module.
logger = logging.getLogger(__name__)


def _extract_2d_centroids(fibers: List, projection: str = 'xy') -> np.ndarray:
    """!
    @brief Extracts 2D centroids of fibers by projecting them onto a plane.
    @param fibers List List of Fiber objects.
    @param projection str Projection plane ('xy', 'xz', 'yz').
    @return np.ndarray Array (N, 2) of projected centroids.
    """
    axis_map = {'xy': (0, 1), 'xz': (0, 2), 'yz': (1, 2)}
    ax1, ax2 = axis_map[projection]

    centroids = []
    for f in fibers:
        c = np.mean(f.centerline, axis=0)
        centroids.append([c[ax1], c[ax2]])

    return np.array(centroids)


def nearest_neighbor_distance(fibers: List, box_dims, projection: str = 'xy') -> Dict:
    """!
    @brief Calculates the Nearest Neighbor Distance (NND) distribution.
    @param fibers List List of fibers.
    @param box_dims tuple Domain dimensions (Lx, Ly, Lz).
    @param projection str 2D projection plane.
    @return Dict Dictionary containing nnd_mean, nnd_std, and nnd_values.
    """
    centroids = _extract_2d_centroids(fibers, projection)
    n = len(centroids)
    if n < 2:
        return {"nnd_mean": 0.0, "nnd_std": 0.0, "nnd_values": []}

    dims = np.array(box_dims)
    axis_map = {'xy': (0, 1), 'xz': (0, 2), 'yz': (1, 2)}
    ax1, ax2 = axis_map[projection]
    box_2d = np.array([dims[ax1], dims[ax2]])

    # 3x3 replication for edge-effect (toroidal) correction
    offsets = []
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            offsets.append(np.array([di * box_2d[0], dj * box_2d[1]]))

    replicated = []
    for offset in offsets:
        replicated.append(centroids + offset)
    all_pts = np.vstack(replicated)

    tree = cKDTree(all_pts)

    # For each original centroid, find the nearest neighbour
    # (the first neighbour is itself in the central replica)
    dists, _ = tree.query(centroids, k=2)
    nnd_values = dists[:, 1]  # distance to the 2nd nearest (1st = itself)

    return {
        "nnd_mean": float(np.mean(nnd_values)),
        "nnd_std": float(np.std(nnd_values)),
        "nnd_values": nnd_values.tolist()
    }


def ripley_k_function(fibers: List, box_dims, h_values: Optional[np.ndarray] = None,
                      projection: str = 'xy') -> Dict:
    """!
    @brief Computes Ripley's K-function with toroidal correction (periodic boundaries).
    @details K(h) = A/n^2 x Sum_i Sum_{j!=i} 1(d_ij <= h)
    For a CSR (Complete Spatial Randomness) process: K(h) = pi*h^2
    @param fibers List List of fibers.
    @param box_dims tuple Domain dimensions.
    @param h_values Optional[np.ndarray] Distance values to evaluate (auto-generated if None).
    @param projection str 2D projection plane.
    @return Dict Dictionary with h_values, K_h, K_poisson, and Besag's L-function (L_h).
    """
    centroids = _extract_2d_centroids(fibers, projection)
    n = len(centroids)
    if n < 2:
        return {"h_values": [], "K_h": [], "K_poisson": [], "L_h": []}

    dims = np.array(box_dims)
    axis_map = {'xy': (0, 1), 'xz': (0, 2), 'yz': (1, 2)}
    ax1, ax2 = axis_map[projection]
    L1, L2 = dims[ax1], dims[ax2]
    A = L1 * L2  # area of the 2D domain

    if h_values is None:
        h_max = min(L1, L2) / 2.0
        h_values = np.linspace(0, h_max, 50)

    # Distance matrix with the 2D Minimum Image Convention (MIC)
    diffs = centroids[:, np.newaxis, :] - centroids[np.newaxis, :, :]  # (n, n, 2)
    diffs[:, :, 0] -= L1 * np.round(diffs[:, :, 0] / L1)
    diffs[:, :, 1] -= L2 * np.round(diffs[:, :, 1] / L2)
    dist_matrix = np.sqrt(np.sum(diffs ** 2, axis=2))  # (n, n)

    K_h = np.zeros(len(h_values))
    for idx, h in enumerate(h_values):
        if h <= 0:
            continue
        # Count the pairs (i!=j) with d_ij <= h
        count = np.sum(dist_matrix <= h) - n  # remove the diagonal (d_ii=0)
        K_h[idx] = A / (n * n) * count

    K_poisson = np.pi * h_values ** 2
    # Besag's L-function: L(h) = sqrt(K(h)/pi) - h
    L_h = np.sqrt(np.maximum(K_h, 0) / np.pi) - h_values

    return {
        "h_values": h_values.tolist(),
        "K_h": K_h.tolist(),
        "K_poisson": K_poisson.tolist(),
        "L_h": L_h.tolist()
    }


def pair_correlation_function(fibers: List, box_dims, r_values: Optional[np.ndarray] = None,
                              projection: str = 'xy') -> Dict:
    """!
    @brief Computes the Pair Correlation Function g(r) (Radial Distribution Function).
    @details g(r) ~ K'(r) / (2*pi*r) estimated by annular counting.
    g(r) = 1 for a CSR process.
    g(r) > 1 indicates clustering, g(r) < 1 indicates regularity.
    @param fibers List List of fibers.
    @param box_dims tuple Domain dimensions.
    @param r_values Optional[np.ndarray] Ring centers (auto-generated if None).
    @param projection str 2D projection plane.
    @return Dict Dictionary with r_values and g_r.
    """
    centroids = _extract_2d_centroids(fibers, projection)
    n = len(centroids)
    if n < 2:
        return {"r_values": [], "g_r": []}

    dims = np.array(box_dims)
    axis_map = {'xy': (0, 1), 'xz': (0, 2), 'yz': (1, 2)}
    ax1, ax2 = axis_map[projection]
    L1, L2 = dims[ax1], dims[ax2]
    A = L1 * L2
    intensity = n / A  # lambda = density

    if r_values is None:
        r_max = min(L1, L2) / 2.0
        r_values = np.linspace(0.01 * r_max, r_max, 50)

    dr = r_values[1] - r_values[0] if len(r_values) > 1 else r_values[0]

    # Distance matrix with 2D MIC
    diffs = centroids[:, np.newaxis, :] - centroids[np.newaxis, :, :]
    diffs[:, :, 0] -= L1 * np.round(diffs[:, :, 0] / L1)
    diffs[:, :, 1] -= L2 * np.round(diffs[:, :, 1] / L2)
    dist_matrix = np.sqrt(np.sum(diffs ** 2, axis=2))

    # Set the diagonal to infinity to exclude it
    np.fill_diagonal(dist_matrix, np.inf)

    g_r = np.zeros(len(r_values))
    for idx, r in enumerate(r_values):
        # Annular counting [r - dr/2, r + dr/2]
        r_low = r - dr / 2
        r_high = r + dr / 2
        if r_low < 0:
            r_low = 0.0
        count = np.sum((dist_matrix >= r_low) & (dist_matrix < r_high))
        # Normalisation: CSR expectation in the ring = n x 2*pi*r*dr x lambda
        expected = n * 2 * np.pi * r * dr * intensity
        if expected > 0:
            g_r[idx] = count / expected

    return {
        "r_values": r_values.tolist(),
        "g_r": g_r.tolist()
    }


def voronoi_statistics(fibers: List, box_dims, projection: str = 'xy') -> Dict:
    """!
    @brief Computes Voronoi cell statistics (area distribution).
    @details Uses 3x3 replication to handle periodic boundaries.
    CV (coefficient of variation): ~0 = very regular, ~0.53 = Poisson (CSR).
    @param fibers List List of fibers.
    @param box_dims tuple Domain dimensions.
    @param projection str 2D projection plane.
    @return Dict Dictionary with areas_mean, areas_std, areas_cv, and areas_values.
    """
    centroids = _extract_2d_centroids(fibers, projection)
    n = len(centroids)
    if n < 3:
        return {"areas_mean": 0.0, "areas_std": 0.0, "areas_cv": 0.0, "areas_values": []}

    dims = np.array(box_dims)
    axis_map = {'xy': (0, 1), 'xz': (0, 2), 'yz': (1, 2)}
    ax1, ax2 = axis_map[projection]
    box_2d = np.array([dims[ax1], dims[ax2]])

    # 3x3 replication for periodicity
    replicated = []
    original_mask = []
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            offset = np.array([di * box_2d[0], dj * box_2d[1]])
            replicated.append(centroids + offset)
            if di == 0 and dj == 0:
                original_mask.extend([True] * n)
            else:
                original_mask.extend([False] * n)

    all_pts = np.vstack(replicated)
    original_mask = np.array(original_mask)

    try:
        vor = Voronoi(all_pts)
    except Exception as e:
        logger.warning(f"Voronoi failed: {e}")
        return {"areas_mean": 0.0, "areas_std": 0.0, "areas_cv": 0.0, "areas_values": []}

    # Extract the cell areas corresponding to the original points
    original_indices = np.where(original_mask)[0]
    areas = []

    for pt_idx in original_indices:
        region_idx = vor.point_region[pt_idx]
        region = vor.regions[region_idx]
        if -1 in region or len(region) == 0:
            continue
        vertices = vor.vertices[region]
        # Area via the shoelace formula
        area = _polygon_area(vertices)
        if area > 0:
            areas.append(area)

    if not areas:
        return {"areas_mean": 0.0, "areas_std": 0.0, "areas_cv": 0.0, "areas_values": []}

    areas = np.array(areas)
    mean_a = float(np.mean(areas))
    std_a = float(np.std(areas))
    cv = std_a / mean_a if mean_a > 0 else 0.0

    return {
        "areas_mean": mean_a,
        "areas_std": std_a,
        "areas_cv": cv,
        "areas_values": areas.tolist()
    }


def _polygon_area(vertices: np.ndarray) -> float:
    """!
    @brief Calculates the area of a 2D polygon using the shoelace formula.
    @param vertices np.ndarray Array of polygon vertices.
    @return float Area of the polygon.
    """
    n = len(vertices)
    if n < 3:
        return 0.0
    x = vertices[:, 0]
    y = vertices[:, 1]
    return 0.5 * abs(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]) + x[-1] * y[0] - x[0] * y[-1])
