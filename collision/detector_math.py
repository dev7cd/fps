# SPDX-FileCopyrightText: 2026 Devine Ngouloubi <exauce-devine.ngouloubi@unicaen.fr>
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
collision/detector_math.py
Collision detection via analytic segment-to-segment distance (Eberly/Geometric Tools).
Replaces the former point-to-point approach with an exact capsule-capsule solution.
"""

import numpy as np
from numba import njit


@njit(cache=True)
def _segment_segment_dist_sq(Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz, Dx, Dy, Dz):
    """
    Minimum squared distance between two 3D segments [A,B] and [C,D].
    Eberly's algorithm (Geometric Tools) - exact parametric resolution.

    Segments: P(s) = A + s*(B-A),  Q(t) = C + t*(D-C),  s,t in [0,1]
    """
    # Direction vectors and difference
    d1x = Bx - Ax;  d1y = By - Ay;  d1z = Bz - Az
    d2x = Dx - Cx;  d2y = Dy - Cy;  d2z = Dz - Cz
    rx  = Ax - Cx;   ry = Ay - Cy;   rz = Az - Cz

    a = d1x*d1x + d1y*d1y + d1z*d1z  # |d1|^2
    e = d2x*d2x + d2y*d2y + d2z*d2z  # |d2|^2
    f = d2x*rx  + d2y*ry  + d2z*rz   # dot(d2, r)

    EPS = 1e-12

    # Degenerate case: both segments are points
    if a <= EPS and e <= EPS:
        return rx*rx + ry*ry + rz*rz

    if a <= EPS:
        # Segment 1 degenerates to a point
        s = 0.0
        t = f / e
        if t < 0.0: t = 0.0
        elif t > 1.0: t = 1.0
    else:
        c = d1x*rx + d1y*ry + d1z*rz  # dot(d1, r)
        if e <= EPS:
            # Segment 2 degenerates to a point
            t = 0.0
            s = -c / a
            if s < 0.0: s = 0.0
            elif s > 1.0: s = 1.0
        else:
            # General case: two non-degenerate segments
            b = d1x*d2x + d1y*d2y + d1z*d2z  # dot(d1, d2)
            denom = a * e - b * b  # always >= 0

            # If segments are non-parallel, find the closest point on the infinite line
            if denom > EPS:
                s = (b * f - c * e) / denom
                if s < 0.0: s = 0.0
                elif s > 1.0: s = 1.0
            else:
                # Parallel segments: pick s=0 as the starting point
                s = 0.0

            # Compute t from s
            t = (b * s + f) / e

            # Clamp t and recompute s if necessary
            if t < 0.0:
                t = 0.0
                s = -c / a
                if s < 0.0: s = 0.0
                elif s > 1.0: s = 1.0
            elif t > 1.0:
                t = 1.0
                s = (b - c) / a
                if s < 0.0: s = 0.0
                elif s > 1.0: s = 1.0

    # Closest points
    px = Ax + s * d1x - (Cx + t * d2x)
    py = Ay + s * d1y - (Cy + t * d2y)
    pz = Az + s * d1z - (Cz + t * d2z)

    return px*px + py*py + pz*pz


@njit(cache=True)
def check_collision_numba(pts1, pts2, r1, r2):
    """!
    @brief Segment-to-segment collision detection between two polylines.
    @param pts1 np.ndarray Centerline (N, 3) of the first fiber.
    @param pts2 np.ndarray Centerline (M, 3) of the second fiber.
    @param r1 float Radius of the first fiber.
    @param r2 float Radius of the second fiber.
    @return bool True if collision (min distance < r1 + r2), False otherwise.
    """
    threshold_sq = (r1 + r2) ** 2
    n1 = len(pts1) - 1  # number of segments
    n2 = len(pts2) - 1

    for i in range(n1):
        Ax = pts1[i, 0];    Ay = pts1[i, 1];    Az = pts1[i, 2]
        Bx = pts1[i+1, 0];  By = pts1[i+1, 1];  Bz = pts1[i+1, 2]

        for j in range(n2):
            Cx = pts2[j, 0];    Cy = pts2[j, 1];    Cz = pts2[j, 2]
            Dx = pts2[j+1, 0];  Dy = pts2[j+1, 1];  Dz = pts2[j+1, 2]

            d_sq = _segment_segment_dist_sq(
                Ax, Ay, Az, Bx, By, Bz,
                Cx, Cy, Cz, Dx, Dy, Dz
            )

            if d_sq < threshold_sq:
                return True

    return False


@njit(cache=True)
def min_dist_segments_numba(pts1, pts2):
    """!
    @brief Exact minimum distance between two polylines (segment-to-segment).
    @param pts1 np.ndarray Centerline (N, 3) of the first polyline.
    @param pts2 np.ndarray Centerline (M, 3) of the second polyline.
    @return float Minimum distance (not squared).
    """
    min_sq = 1e20
    n1 = len(pts1) - 1
    n2 = len(pts2) - 1

    for i in range(n1):
        Ax = pts1[i, 0];    Ay = pts1[i, 1];    Az = pts1[i, 2]
        Bx = pts1[i+1, 0];  By = pts1[i+1, 1];  Bz = pts1[i+1, 2]

        for j in range(n2):
            Cx = pts2[j, 0];    Cy = pts2[j, 1];    Cz = pts2[j, 2]
            Dx = pts2[j+1, 0];  Dy = pts2[j+1, 1];  Dz = pts2[j+1, 2]

            d_sq = _segment_segment_dist_sq(
                Ax, Ay, Az, Bx, By, Bz,
                Cx, Cy, Cz, Dx, Dy, Dz
            )

            if d_sq < min_sq:
                min_sq = d_sq

    return np.sqrt(min_sq)


@njit(cache=True)
def min_dist_periodic_segments_numba(pts1, pts2, dims):
    """!
    @brief Minimum segment-to-segment distance with the Minimum Image Convention (MIC).
    @param pts1 np.ndarray Centerline (N, 3) of the first polyline.
    @param pts2 np.ndarray Centerline (M, 3) of the second polyline.
    @param dims np.ndarray Domain dimensions [Lx, Ly, Lz].
    @return float Minimum periodic distance.
    """
    min_sq = 1e20
    Lx = dims[0]; Ly = dims[1]; Lz = dims[2]
    n1 = len(pts1) - 1
    n2 = len(pts2) - 1

    for i in range(n1):
        Ax = pts1[i, 0];    Ay = pts1[i, 1];    Az = pts1[i, 2]
        Bx = pts1[i+1, 0];  By = pts1[i+1, 1];  Bz = pts1[i+1, 2]

        # Midpoint of segment 1
        m1x = 0.5 * (Ax + Bx)
        m1y = 0.5 * (Ay + By)
        m1z = 0.5 * (Az + Bz)

        for j in range(n2):
            Cx = pts2[j, 0];    Cy = pts2[j, 1];    Cz = pts2[j, 2]
            Dx = pts2[j+1, 0];  Dy = pts2[j+1, 1];  Dz = pts2[j+1, 2]

            # Midpoint of segment 2
            m2x = 0.5 * (Cx + Dx)
            m2y = 0.5 * (Cy + Dy)
            m2z = 0.5 * (Cz + Dz)

            # MIC shift based on the midpoints
            shift_x = -Lx * round((m2x - m1x) / Lx)
            shift_y = -Ly * round((m2y - m1y) / Ly)
            shift_z = -Lz * round((m2z - m1z) / Lz)

            # Apply the shift to segment 2
            Cx_s = Cx + shift_x;  Cy_s = Cy + shift_y;  Cz_s = Cz + shift_z
            Dx_s = Dx + shift_x;  Dy_s = Dy + shift_y;  Dz_s = Dz + shift_z

            d_sq = _segment_segment_dist_sq(
                Ax, Ay, Az, Bx, By, Bz,
                Cx_s, Cy_s, Cz_s, Dx_s, Dy_s, Dz_s
            )

            if d_sq < min_sq:
                min_sq = d_sq

    return np.sqrt(min_sq)
