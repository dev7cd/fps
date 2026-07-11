"""!
@file frames.py
@brief Local frame computation along 3D curves.
@details Implements Parallel Transport (Bishop Frame) to minimize torsion and
provides utilities for tangent vector calculation.
"""

import numpy as np
from typing import Tuple

def compute_tangents(points: np.ndarray) -> np.ndarray:
    """!
    @brief Computes normalized tangent vectors using centered finite differences.
    @param points Array of shape (N, 3) representing the curve points.
    @return Array of shape (N, 3) containing normalized tangent vectors.
    @exception ValueError If the input contains fewer than 2 points.
    """
    if len(points) < 2:
        raise ValueError("At least 2 points are required to compute tangents.")

    # Gradient (numerical derivative)
    tangents = np.gradient(points, axis=0)

    # Normalisation
    norms = np.linalg.norm(tangents, axis=1)
    # Guard against division by zero
    norms[norms < 1e-12] = 1.0

    return tangents / norms[:, np.newaxis]

def compute_bishop_frame(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """!
    @brief Computes the Bishop frame (Tangent, Normal, Binormal) along a curve.
    @details Uses the Parallel Transport method to propagate the normal vector
    along the curve, which avoids the "twisting" artifacts of Frenet-Serret
    frames at points of zero curvature.
    @param points Array of shape (N, 3) representing the curve points.
    @return A tuple (T, N, B) where each is an array of shape (N, 3).
    """
    n_points = len(points)
    T = compute_tangents(points)
    N = np.zeros_like(T)
    B = np.zeros_like(T)

    # --- 1. Initialisation at the first point ---
    # Pick an arbitrary vector not collinear with T[0]
    t0 = T[0]
    arbitrary = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(t0, arbitrary)) > 0.9:
        arbitrary = np.array([1.0, 0.0, 0.0])

    # First normal vector via cross product
    v_perp = np.cross(t0, arbitrary)
    n0 = v_perp / np.linalg.norm(v_perp)
    b0 = np.cross(t0, n0)

    N[0] = n0
    B[0] = b0

    # --- 2. Parallel transport (propagation) ---
    for i in range(1, n_points):
        t_prev = T[i-1]
        t_curr = T[i]
        n_prev = N[i-1]
        b_prev = B[i-1]

        # Rotation axis to align t_prev with t_curr
        # This is the local curvature binormal vector
        axis = np.cross(t_prev, t_curr)
        len_axis = np.linalg.norm(axis)

        if len_axis < 1e-9:
            # Straight curve, no rotation
            N[i] = n_prev
            B[i] = b_prev
        else:
            axis = axis / len_axis
            # Rotation angle
            # dot = cos(theta)
            cos_theta = np.clip(np.dot(t_prev, t_curr), -1.0, 1.0)
            theta = np.arccos(cos_theta)

            # Rotation matrix around 'axis' by angle 'theta'
            # Simplified Rodrigues formula for a vector:
            # V_rot = V*cos + (K x V)*sin + K*(K.V)*(1-cos), with K = axis

            # Rotate n_prev
            n_new = (n_prev * cos_theta +
                     np.cross(axis, n_prev) * np.sin(theta) +
                     axis * np.dot(axis, n_prev) * (1 - cos_theta))

            # Rotate b_prev
            b_new = (b_prev * cos_theta +
                     np.cross(axis, b_prev) * np.sin(theta) +
                     axis * np.dot(axis, b_prev) * (1 - cos_theta))

            N[i] = n_new / np.linalg.norm(n_new)
            B[i] = b_new / np.linalg.norm(b_new)

    return T, N, B
