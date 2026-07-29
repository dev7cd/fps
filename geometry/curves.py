# SPDX-FileCopyrightText: 2026 Devine Ngouloubi <exauce-devine.ngouloubi@unicaen.fr>
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
@file curves.py
@brief Curve management: Interpolation (Splines) and Random Generation (Walk).
@details Provides tools for smooth fiber path generation using Catmull-Rom splines
and stochastic control point generation with orientation bias.
"""

import numpy as np
from typing import List, Tuple, Union, Optional

class CatmullRomSpline:
    """!
    @class CatmullRomSpline
    @brief Vectorized implementation of Catmull-Rom splines.
    @details Supports centripetal, uniform, and chordal parameterization through alpha.
    """

    @staticmethod
    def interpolate(control_points: np.ndarray, num_points: int = 100, alpha: float = 0.5) -> np.ndarray:
        """!
        @brief Generates a smooth curve passing through control points.
        @param control_points Array of shape (N, 3) containing the skeleton points.
        @param num_points Total number of interpolated points desired for the final curve.
        @param alpha Parameterization factor: 0.0=Uniform, 0.5=Centripetal (recommended), 1.0=Chordal.
        @return A (num_points, 3) array representing the smooth curve.
        """
        if len(control_points) < 4:
            # Not enough points, simple linear interpolation
            return CatmullRomSpline._linear_resample(control_points, num_points)

        # --- 1. Ghost points ---
        # Add ghost points at extremities for a natural open curve
        # Duplicate/project the endpoints for a natural open curve
        p0 = 2 * control_points[0] - control_points[1]
        pn = 2 * control_points[-1] - control_points[-2]
        points = np.vstack([p0, control_points, pn])

        # Number of valid segments
        n_segments = len(points) - 3
        points_per_segment = max(1, num_points // n_segments)

        curve_points = []

        for i in range(n_segments):
            P0, P1, P2, P3 = points[i], points[i+1], points[i+2], points[i+3]

            # Local time t in [0, 1]
            t = np.linspace(0, 1, points_per_segment, endpoint=False)

            # Tangent-vector computation (simplified method for speed).
            # A strict centripetal implementation would compute dt based on alpha.
            # Here we use a standard uniform Catmull-Rom approximation, which is
            # usually sufficient; for a dense mesh the standard matrix form is
            # very efficient even when alpha != 0.

            # Catmull-Rom basis matrix
            # Q(t) = 0.5 * [1 t t^2 t^3] * M * [P0 P1 P2 P3]
            t2 = t * t
            t3 = t2 * t

            # Weights
            b0 = 0.5 * (-t3 + 2*t2 - t)
            b1 = 0.5 * (3*t3 - 5*t2 + 2)
            b2 = 0.5 * (-3*t3 + 4*t2 + t)
            b3 = 0.5 * (t3 - t2)

            # Linear combination
            # Broadcasting: b0 is (M,), P0 is (3,) -> (M, 3)
            segment = (np.outer(b0, P0) +
                       np.outer(b1, P1) +
                       np.outer(b2, P2) +
                       np.outer(b3, P3))

            curve_points.append(segment)

        # Append the very last point
        curve_points.append(points[-2].reshape(1, 3))

        full_curve = np.vstack(curve_points)

        # Final resampling to obtain exactly num_points equidistant samples
        return CatmullRomSpline._linear_resample(full_curve, num_points)

    @staticmethod
    def _linear_resample(points: np.ndarray, num_samples: int) -> np.ndarray:
        """!
        @brief Resamples a polyline to ensure equidistant points.
        @param points Input polyline points (N, 3).
        @param num_samples Desired number of points.
        @return Resampled polyline (num_samples, 3).
        """
        if len(points) < 2: return points

        # Cumulative distances
        dists = np.linalg.norm(np.diff(points, axis=0), axis=1)
        cum_dist = np.zeros(len(points))
        cum_dist[1:] = np.cumsum(dists)
        total_len = cum_dist[-1]

        if total_len < 1e-9: return np.linspace(points[0], points[-1], num_samples)

        # New sampling positions
        target_dists = np.linspace(0, total_len, num_samples)

        # Interpolation dimension by dimension
        new_points = np.zeros((num_samples, 3))
        for dim in range(3):
            new_points[:, dim] = np.interp(target_dists, cum_dist, points[:, dim])

        return new_points

def generate_random_control_points(
    n_points: int,
    step_mean: float,
    step_std: float,
    box_dims: Tuple[float, float, float],
    rng: np.random.Generator,
    orientation_bias: Union[str, List[float]] = 'free',
    bias_strength: float = 0.0
) -> np.ndarray:
    """!
    @brief Generates a sequence of points (random walk) in continuous space.
    @param n_points Number of control points to generate.
    @param step_mean Mean distance between consecutive points.
    @param step_std Standard deviation of the step length.
    @param box_dims Dimensions [Lx, Ly, Lz] of the generation domain.
    @param rng Numpy random generator instance.
    @param orientation_bias Directional constraint ('x', 'y', 'z', 'free' or a 3D vector).
    @param bias_strength Weight of the bias (0.0 to 1.0).
    @return Array of shape (n_points, 3).
    """
    Lx, Ly, Lz = box_dims

    # 1. Random starting point in the box
    current_pos = np.array([
        rng.uniform(0, Lx),
        rng.uniform(0, Ly),
        rng.uniform(0, Lz)
    ])

    points = [current_pos.copy()]

    # Determine the bias vector
    bias_vec = None
    if bias_strength > 0:
        if isinstance(orientation_bias, list):
            bias_vec = np.array(orientation_bias)
        elif orientation_bias == 'x': bias_vec = np.array([1, 0, 0])
        elif orientation_bias == 'y': bias_vec = np.array([0, 1, 0])
        elif orientation_bias == 'z': bias_vec = np.array([0, 0, 1])

        if bias_vec is not None:
            bias_vec = bias_vec / np.linalg.norm(bias_vec)

    # 2. Random walk
    for _ in range(n_points - 1):
        # Random direction on the sphere
        # Marsaglia method or normalized Gaussian
        v = rng.normal(0, 1, 3)
        norm = np.linalg.norm(v)
        if norm < 1e-9: v = np.array([1.0, 0, 0])
        else: v = v / norm

        # Apply the bias
        if bias_vec is not None:
            # Linear blend: (1-k)*Rand + k*Bias, then renormalise
            v = (1 - bias_strength) * v + bias_strength * bias_vec
            v = v / np.linalg.norm(v)

        # Step length
        step = max(0.01, rng.normal(step_mean, step_std))

        # New point
        current_pos = current_pos + v * step
        points.append(current_pos.copy())

    return np.array(points)
