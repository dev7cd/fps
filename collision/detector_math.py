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
def _segment_segment_closest(Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz, Dx, Dy, Dz):
    """
    Minimum squared distance between two 3D segments [A,B] and [C,D], with the
    parameters of the closest points.
    Eberly's algorithm (Geometric Tools) - exact parametric resolution.

    Segments: P(s) = A + s*(B-A),  Q(t) = C + t*(D-C),  s,t in [0,1]
    Returns (d_sq, s, t).
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
        return rx*rx + ry*ry + rz*rz, 0.0, 0.0

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

    return px*px + py*py + pz*pz, s, t


@njit(cache=True)
def _segment_segment_dist_sq(Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz, Dx, Dy, Dz):
    """
    Minimum squared distance between two 3D segments [A,B] and [C,D].
    """
    d_sq, s, t = _segment_segment_closest(Ax, Ay, Az, Bx, By, Bz,
                                          Cx, Cy, Cz, Dx, Dy, Dz)
    return d_sq


@njit(cache=True)
def check_collision_numba(pts1, pts2, r1, r2, clearance=0.0):
    """!
    @brief Segment-to-segment collision detection between two polylines.
    @param pts1 np.ndarray Centerline (N, 3) of the first fiber.
    @param pts2 np.ndarray Centerline (M, 3) of the second fiber.
    @param r1 float Radius of the first fiber.
    @param r2 float Radius of the second fiber.
    @param clearance float Minimum empty gap required between the two surfaces.
    @return bool True if collision (min distance < r1 + r2 + clearance), False otherwise.
    """
    threshold_sq = (r1 + r2 + clearance) ** 2
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
def _polyline_dist_sq_shifted(pts1, pts2, sx, sy, sz):
    """
    Minimum squared distance between polyline 1 and polyline 2 translated by (sx, sy, sz).
    """
    min_sq = 1e20
    n1 = len(pts1) - 1
    n2 = len(pts2) - 1

    for i in range(n1):
        Ax = pts1[i, 0];    Ay = pts1[i, 1];    Az = pts1[i, 2]
        Bx = pts1[i+1, 0];  By = pts1[i+1, 1];  Bz = pts1[i+1, 2]

        for j in range(n2):
            Cx = pts2[j, 0] + sx;    Cy = pts2[j, 1] + sy;    Cz = pts2[j, 2] + sz
            Dx = pts2[j+1, 0] + sx;  Dy = pts2[j+1, 1] + sy;  Dz = pts2[j+1, 2] + sz

            d_sq = _segment_segment_dist_sq(
                Ax, Ay, Az, Bx, By, Bz,
                Cx, Cy, Cz, Dx, Dy, Dz
            )

            if d_sq < min_sq:
                min_sq = d_sq

    return min_sq


@njit(cache=True)
def _interval_gap(min_a, max_a, min_b, max_b):
    """Distance between the intervals [min_a, max_a] and [min_b, max_b] (0 if they overlap)."""
    if min_b > max_a:
        return min_b - max_a
    if min_a > max_b:
        return min_a - max_b
    return 0.0


@njit(cache=True)
def min_dist_periodic_images_numba(pts1, pts2, dims, skip_zero_shift=False):
    """!
    @brief Exact minimum distance between two polylines under periodic boundaries.
    @details Scans every periodic image of the second polyline that can hold the
    minimum and keeps the smallest segment-to-segment distance. The image range
    is derived per axis from the two bounding boxes, so it stays valid when a
    fiber is longer than the domain (the Minimum Image Convention is not).
    A bounding-box test prunes the images that cannot beat the current best.
    @param pts1 np.ndarray Centerline (N, 3) of the first polyline.
    @param pts2 np.ndarray Centerline (M, 3) of the second polyline.
    @param dims np.ndarray Domain dimensions [Lx, Ly, Lz].
    @param skip_zero_shift bool Ignore the untranslated image. Set it when both
           polylines are the same fiber, to measure the distance to its own
           periodic images instead of the trivial zero.
    @return float Minimum periodic distance.
    """
    # Axis-aligned bounds of both polylines
    a_min0 = pts1[0, 0]; a_min1 = pts1[0, 1]; a_min2 = pts1[0, 2]
    a_max0 = a_min0;     a_max1 = a_min1;     a_max2 = a_min2
    for i in range(1, len(pts1)):
        if pts1[i, 0] < a_min0: a_min0 = pts1[i, 0]
        elif pts1[i, 0] > a_max0: a_max0 = pts1[i, 0]
        if pts1[i, 1] < a_min1: a_min1 = pts1[i, 1]
        elif pts1[i, 1] > a_max1: a_max1 = pts1[i, 1]
        if pts1[i, 2] < a_min2: a_min2 = pts1[i, 2]
        elif pts1[i, 2] > a_max2: a_max2 = pts1[i, 2]

    b_min0 = pts2[0, 0]; b_min1 = pts2[0, 1]; b_min2 = pts2[0, 2]
    b_max0 = b_min0;     b_max1 = b_min1;     b_max2 = b_min2
    for j in range(1, len(pts2)):
        if pts2[j, 0] < b_min0: b_min0 = pts2[j, 0]
        elif pts2[j, 0] > b_max0: b_max0 = pts2[j, 0]
        if pts2[j, 1] < b_min1: b_min1 = pts2[j, 1]
        elif pts2[j, 1] > b_max1: b_max1 = pts2[j, 1]
        if pts2[j, 2] < b_min2: b_min2 = pts2[j, 2]
        elif pts2[j, 2] > b_max2: b_max2 = pts2[j, 2]

    Lx = dims[0]; Ly = dims[1]; Lz = dims[2]
    min_sq = 1e20

    # Per-axis image range: every shift that brings the two boxes within one
    # period of each other. Beyond those bounds the gap only grows, so the
    # minimum is necessarily attained inside the range.
    kx_lo = int(np.floor((a_min0 - b_max0) / Lx)) - 1
    kx_hi = int(np.ceil((a_max0 - b_min0) / Lx)) + 1
    ky_lo = int(np.floor((a_min1 - b_max1) / Ly)) - 1
    ky_hi = int(np.ceil((a_max1 - b_min1) / Ly)) + 1
    kz_lo = int(np.floor((a_min2 - b_max2) / Lz)) - 1
    kz_hi = int(np.ceil((a_max2 - b_min2) / Lz)) + 1

    for kx in range(kx_lo, kx_hi + 1):
        sx = kx * Lx
        gx = _interval_gap(a_min0, a_max0, b_min0 + sx, b_max0 + sx)
        if gx * gx >= min_sq:
            continue
        for ky in range(ky_lo, ky_hi + 1):
            sy = ky * Ly
            gy = _interval_gap(a_min1, a_max1, b_min1 + sy, b_max1 + sy)
            if gx * gx + gy * gy >= min_sq:
                continue
            for kz in range(kz_lo, kz_hi + 1):
                if skip_zero_shift and kx == 0 and ky == 0 and kz == 0:
                    continue
                sz = kz * Lz
                gz = _interval_gap(a_min2, a_max2, b_min2 + sz, b_max2 + sz)
                if gx * gx + gy * gy + gz * gz >= min_sq:
                    continue

                d_sq = _polyline_dist_sq_shifted(pts1, pts2, sx, sy, sz)
                if d_sq < min_sq:
                    min_sq = d_sq

    return np.sqrt(min_sq)


@njit(cache=True)
def min_dist_periodic_segments_numba(pts1, pts2, dims):
    """!
    @brief Minimum segment-to-segment distance with the Minimum Image Convention (MIC).
    @warning The MIC is only valid when both polylines are smaller than half the
    domain in every direction. For fibers spanning (or exceeding) the box, the
    midpoint-based image choice can be wrong and the result is then an
    underestimate: use min_dist_periodic_images_numba() instead.
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


@njit(cache=True)
def closest_approach_periodic(pts1, pts2, dims, skip_zero_shift=False, cutoff=1e18):
    """!
    @brief Closest approach between two polylines under periodic boundaries.
    @details Scans the relevant periodic images of the second polyline (same
    range logic as min_dist_periodic_images_numba) and returns the contact
    geometry needed to push the two fibers apart.
    @param pts1 np.ndarray Centerline (N, 3) of the first polyline.
    @param pts2 np.ndarray Centerline (M, 3) of the second polyline.
    @param dims np.ndarray Domain dimensions [Lx, Ly, Lz].
    @param skip_zero_shift bool Ignore the untranslated image. Set it when both
           arguments are the same fiber, to get its contact with its own images.
    @param cutoff float Distance above which the caller does not care. Images that
           cannot beat it are skipped outright, which is what makes a full contact
           scan affordable: pass the required separation and read the result as
           "no contact" whenever the returned distance reaches the cutoff.
    @return tuple (d, u1, u2, ux, uy, uz) where d is the minimum distance,
            u1 the contact abscissa on polyline 1 as a fractional index,
            u2 the same on polyline 2 (in its own, unshifted frame), and
            (ux, uy, uz) the unit vector pointing from polyline 2 to polyline 1.
            The vector is (0, 0, 0) when the two curves are exactly coincident.
    """
    Lx = dims[0]; Ly = dims[1]; Lz = dims[2]
    n1 = len(pts1) - 1
    n2 = len(pts2) - 1

    min_sq = cutoff * cutoff
    best_i = 0; best_j = 0
    best_s = 0.0; best_t = 0.0
    best_sx = 0.0; best_sy = 0.0; best_sz = 0.0

    # Bounds of both polylines, to restrict the image range
    a_min0 = pts1[0, 0]; a_min1 = pts1[0, 1]; a_min2 = pts1[0, 2]
    a_max0 = a_min0;     a_max1 = a_min1;     a_max2 = a_min2
    for i in range(1, len(pts1)):
        if pts1[i, 0] < a_min0: a_min0 = pts1[i, 0]
        elif pts1[i, 0] > a_max0: a_max0 = pts1[i, 0]
        if pts1[i, 1] < a_min1: a_min1 = pts1[i, 1]
        elif pts1[i, 1] > a_max1: a_max1 = pts1[i, 1]
        if pts1[i, 2] < a_min2: a_min2 = pts1[i, 2]
        elif pts1[i, 2] > a_max2: a_max2 = pts1[i, 2]

    b_min0 = pts2[0, 0]; b_min1 = pts2[0, 1]; b_min2 = pts2[0, 2]
    b_max0 = b_min0;     b_max1 = b_min1;     b_max2 = b_min2
    for j in range(1, len(pts2)):
        if pts2[j, 0] < b_min0: b_min0 = pts2[j, 0]
        elif pts2[j, 0] > b_max0: b_max0 = pts2[j, 0]
        if pts2[j, 1] < b_min1: b_min1 = pts2[j, 1]
        elif pts2[j, 1] > b_max1: b_max1 = pts2[j, 1]
        if pts2[j, 2] < b_min2: b_min2 = pts2[j, 2]
        elif pts2[j, 2] > b_max2: b_max2 = pts2[j, 2]

    kx_lo = int(np.floor((a_min0 - b_max0) / Lx)) - 1
    kx_hi = int(np.ceil((a_max0 - b_min0) / Lx)) + 1
    ky_lo = int(np.floor((a_min1 - b_max1) / Ly)) - 1
    ky_hi = int(np.ceil((a_max1 - b_min1) / Ly)) + 1
    kz_lo = int(np.floor((a_min2 - b_max2) / Lz)) - 1
    kz_hi = int(np.ceil((a_max2 - b_min2) / Lz)) + 1

    for kx in range(kx_lo, kx_hi + 1):
        sx = kx * Lx
        gx = _interval_gap(a_min0, a_max0, b_min0 + sx, b_max0 + sx)
        if gx * gx >= min_sq:
            continue
        for ky in range(ky_lo, ky_hi + 1):
            sy = ky * Ly
            gy = _interval_gap(a_min1, a_max1, b_min1 + sy, b_max1 + sy)
            if gx * gx + gy * gy >= min_sq:
                continue
            for kz in range(kz_lo, kz_hi + 1):
                if skip_zero_shift and kx == 0 and ky == 0 and kz == 0:
                    continue
                sz = kz * Lz
                gz = _interval_gap(a_min2, a_max2, b_min2 + sz, b_max2 + sz)
                if gx * gx + gy * gy + gz * gz >= min_sq:
                    continue

                for i in range(n1):
                    Ax = pts1[i, 0];    Ay = pts1[i, 1];    Az = pts1[i, 2]
                    Bx = pts1[i+1, 0];  By = pts1[i+1, 1];  Bz = pts1[i+1, 2]

                    for j in range(n2):
                        Cx = pts2[j, 0] + sx;    Cy = pts2[j, 1] + sy;    Cz = pts2[j, 2] + sz
                        Dx = pts2[j+1, 0] + sx;  Dy = pts2[j+1, 1] + sy;  Dz = pts2[j+1, 2] + sz

                        d_sq, s, t = _segment_segment_closest(
                            Ax, Ay, Az, Bx, By, Bz,
                            Cx, Cy, Cz, Dx, Dy, Dz
                        )

                        if d_sq < min_sq:
                            min_sq = d_sq
                            best_i = i; best_j = j
                            best_s = s; best_t = t
                            best_sx = sx; best_sy = sy; best_sz = sz

    # Rebuild the contact points to get the separation direction
    p1x = pts1[best_i, 0] + best_s * (pts1[best_i+1, 0] - pts1[best_i, 0])
    p1y = pts1[best_i, 1] + best_s * (pts1[best_i+1, 1] - pts1[best_i, 1])
    p1z = pts1[best_i, 2] + best_s * (pts1[best_i+1, 2] - pts1[best_i, 2])

    p2x = pts2[best_j, 0] + best_t * (pts2[best_j+1, 0] - pts2[best_j, 0]) + best_sx
    p2y = pts2[best_j, 1] + best_t * (pts2[best_j+1, 1] - pts2[best_j, 1]) + best_sy
    p2z = pts2[best_j, 2] + best_t * (pts2[best_j+1, 2] - pts2[best_j, 2]) + best_sz

    d = np.sqrt(min_sq)
    if d > 1e-12:
        ux = (p1x - p2x) / d
        uy = (p1y - p2y) / d
        uz = (p1z - p2z) / d
    else:
        ux = 0.0; uy = 0.0; uz = 0.0

    return d, best_i + best_s, best_j + best_t, ux, uy, uz
