# SPDX-FileCopyrightText: 2026 Devine Ngouloubi <exauce-devine.ngouloubi@unicaen.fr>
# SPDX-License-Identifier: AGPL-3.0-or-later

## @file audit_distance.py
#  @brief Distance kernel of the final geometric audit (F03), independent of the
#  collision detector used during generation.
#  @details The generation detector (collision/detector_math.py) compares
#  scale-dependent quantities with an absolute threshold and can overestimate
#  the distance between short or nearly parallel segments. This module does not
#  call it. Its decisions are invariant under a common change of scale:
#
#  - Minimum of |P(s) - Q(t)|, P(s) = A + s d1, Q(t) = C + t d2, (s, t) in [0, 1]^2.
#    The minimum of this convex function lies either on an edge of the square
#    (one of the four endpoint-to-segment distances) or at the interior
#    stationary point. All five candidates are kept; the interior one is never
#    replaced by an endpoint fallback.
#  - Interior candidate, when the segments are not nearly parallel
#    (|n|^2 > NEAR_PARALLEL * a * e, n = d1 x d2 formed directly, a = |d1|^2,
#    e = |d2|^2, no a*e - b*b): s = ((C - A) x d2) . n / |n|^2,
#    t = ((C - A) x d1) . n / |n|^2 and, when both lie in [0, 1], the line-to-line
#    distance |(C - A) . n| / |n|. This distance never exceeds the true segment
#    distance, so a misclassified interior point can only make the audit more
#    conservative.
#  - Nearly parallel segments (0 < |n|^2 <= NEAR_PARALLEL * a * e): the direction
#    of n carries a relative error of order eps / sin(theta), which the line
#    distance multiplies by |C - A|. The interior minimum is then found from the
#    convex function g(s) = dist(A + s d1, [C, E]) (distance of an affine point to
#    a convex set) by golden-section search down to |delta s| <= S_TOL: the value
#    returned is attained by a point of the first segment, so it can exceed the
#    minimum by at most |d1| * S_TOL, plus the point-to-segment rounding.
#  - n = 0 exactly (parallel or degenerate segments): the edge candidates give
#    the exact minimum. Zero-length segments are treated as points.
#  - Interior point-to-segment distances use |w x d| / |d| (no cancellation in
#    |w|^2 - proj^2).
#
#  Periodic images: for two polylines with coordinate intervals [a_min, a_max]
#  and [b_min, b_max] on an axis of period L, the shift k minimising
#  |p - q - kL| for any pair of points lies in
#  [floor((a_min - b_max) / L), ceil((a_max - b_min) / L)]; one extra period is
#  scanned on each side against rounding. The axes are independent, so this box
#  of shifts contains the minimum over all images. A fiber is compared with its
#  own non-zero images only.

import math

import numpy as np
from numba import njit

NEAR_PARALLEL = 1e-4      # sin^2(theta) below which the cross-product interior formula is not used
S_TOL = 1e-14             # parameter tolerance of the golden-section search
GOLDEN = 0.6180339887498949


@njit(cache=True)
def point_segment_distance(px, py, pz, cx, cy, cz, dx, dy, dz):
    """!
    @brief Distance from point P to segment [C, C + d].
    """
    wx = px - cx
    wy = py - cy
    wz = pz - cz
    e = dx * dx + dy * dy + dz * dz
    if e == 0.0:
        return math.sqrt(wx * wx + wy * wy + wz * wz)
    proj = wx * dx + wy * dy + wz * dz
    if proj <= 0.0:
        return math.sqrt(wx * wx + wy * wy + wz * wz)
    if proj >= e:
        ux = wx - dx
        uy = wy - dy
        uz = wz - dz
        return math.sqrt(ux * ux + uy * uy + uz * uz)
    qx = wy * dz - wz * dy
    qy = wz * dx - wx * dz
    qz = wx * dy - wy * dx
    return math.sqrt(qx * qx + qy * qy + qz * qz) / math.sqrt(e)


@njit(cache=True)
def segment_segment_distance(ax, ay, az, bx, by, bz, cx, cy, cz, ex, ey, ez):
    """!
    @brief Distance between segments [A, B] and [C, E] (five candidates, see module).
    """
    d1x = bx - ax
    d1y = by - ay
    d1z = bz - az
    d2x = ex - cx
    d2y = ey - cy
    d2z = ez - cz
    best = point_segment_distance(ax, ay, az, cx, cy, cz, d2x, d2y, d2z)
    v = point_segment_distance(bx, by, bz, cx, cy, cz, d2x, d2y, d2z)
    if v < best:
        best = v
    v = point_segment_distance(cx, cy, cz, ax, ay, az, d1x, d1y, d1z)
    if v < best:
        best = v
    v = point_segment_distance(ex, ey, ez, ax, ay, az, d1x, d1y, d1z)
    if v < best:
        best = v
    nx = d1y * d2z - d1z * d2y
    ny = d1z * d2x - d1x * d2z
    nz = d1x * d2y - d1y * d2x
    nn = nx * nx + ny * ny + nz * nz
    a = d1x * d1x + d1y * d1y + d1z * d1z
    e = d2x * d2x + d2y * d2y + d2z * d2z
    if nn > 0.0 and nn <= NEAR_PARALLEL * a * e:
        v = _golden_interior(ax, ay, az, d1x, d1y, d1z, cx, cy, cz, d2x, d2y, d2z)
        if v < best:
            best = v
    elif nn > 0.0:
        rx = cx - ax
        ry = cy - ay
        rz = cz - az
        # (r x d2) . n and (r x d1) . n
        s = ((ry * d2z - rz * d2y) * nx + (rz * d2x - rx * d2z) * ny + (rx * d2y - ry * d2x) * nz) / nn
        t = ((ry * d1z - rz * d1y) * nx + (rz * d1x - rx * d1z) * ny + (rx * d1y - ry * d1x) * nz) / nn
        if 0.0 <= s <= 1.0 and 0.0 <= t <= 1.0:
            v = abs(rx * nx + ry * ny + rz * nz) / math.sqrt(nn)
            if v < best:
                best = v
    return best


@njit(cache=True)
def _golden_interior(ax, ay, az, d1x, d1y, d1z, cx, cy, cz, d2x, d2y, d2z):
    """min over s in [0, 1] of dist(A + s d1, [C, C + d2]) by golden-section search (convex in s)."""
    lo = 0.0
    hi = 1.0
    x1 = hi - GOLDEN * (hi - lo)
    x2 = lo + GOLDEN * (hi - lo)
    f1 = point_segment_distance(ax + x1 * d1x, ay + x1 * d1y, az + x1 * d1z, cx, cy, cz, d2x, d2y, d2z)
    f2 = point_segment_distance(ax + x2 * d1x, ay + x2 * d1y, az + x2 * d1z, cx, cy, cz, d2x, d2y, d2z)
    while hi - lo > S_TOL:
        if f1 <= f2:
            hi = x2
            x2 = x1
            f2 = f1
            x1 = hi - GOLDEN * (hi - lo)
            f1 = point_segment_distance(ax + x1 * d1x, ay + x1 * d1y, az + x1 * d1z, cx, cy, cz, d2x, d2y, d2z)
        else:
            lo = x1
            x1 = x2
            f1 = f2
            x2 = lo + GOLDEN * (hi - lo)
            f2 = point_segment_distance(ax + x2 * d1x, ay + x2 * d1y, az + x2 * d1z, cx, cy, cz, d2x, d2y, d2z)
    return min(f1, f2)


@njit(cache=True)
def _polyline_distance(p1, p2, sx, sy, sz, best):
    """Minimum segment distance between p1 and p2 + (sx, sy, sz); skips segment pairs whose boxes
    are farther apart than `best`. Returns (distance, i, j)."""
    bi = -1
    bj = -1
    for i in range(p1.shape[0] - 1):
        a0 = p1[i]
        a1 = p1[i + 1]
        for j in range(p2.shape[0] - 1):
            c0x = p2[j, 0] + sx
            c0y = p2[j, 1] + sy
            c0z = p2[j, 2] + sz
            c1x = p2[j + 1, 0] + sx
            c1y = p2[j + 1, 1] + sy
            c1z = p2[j + 1, 2] + sz
            g2 = 0.0
            lo = min(a0[0], a1[0]) - max(c0x, c1x)
            hi = min(c0x, c1x) - max(a0[0], a1[0])
            g = max(lo, hi, 0.0)
            g2 += g * g
            lo = min(a0[1], a1[1]) - max(c0y, c1y)
            hi = min(c0y, c1y) - max(a0[1], a1[1])
            g = max(lo, hi, 0.0)
            g2 += g * g
            lo = min(a0[2], a1[2]) - max(c0z, c1z)
            hi = min(c0z, c1z) - max(a0[2], a1[2])
            g = max(lo, hi, 0.0)
            g2 += g * g
            if g2 > best * best:
                continue
            d = segment_segment_distance(a0[0], a0[1], a0[2], a1[0], a1[1], a1[2],
                                         c0x, c0y, c0z, c1x, c1y, c1z)
            if d < best:
                best = d
                bi = i
                bj = j
    return best, bi, bj


@njit(cache=True)
def periodic_polyline_distance(p1, p2, Lx, Ly, Lz, same_fiber):
    """!
    @brief Minimum distance between polyline p1 and all periodic images of p2.
    @param same_fiber True when p1 and p2 are the same fiber: the zero shift is skipped.
    @return (distance, kx, ky, kz, segment i, segment j); distance = inf if no image was examined.
    """
    amin = np.empty(3)
    amax = np.empty(3)
    bmin = np.empty(3)
    bmax = np.empty(3)
    for a in range(3):
        amin[a] = p1[:, a].min()
        amax[a] = p1[:, a].max()
        bmin[a] = p2[:, a].min()
        bmax[a] = p2[:, a].max()
    L = (Lx, Ly, Lz)
    klo = np.empty(3, dtype=np.int64)
    khi = np.empty(3, dtype=np.int64)
    for a in range(3):
        klo[a] = int(math.floor((amin[a] - bmax[a]) / L[a])) - 1
        khi[a] = int(math.ceil((amax[a] - bmin[a]) / L[a])) + 1
    best = math.inf
    bk = (0, 0, 0)
    bi = -1
    bj = -1
    for kx in range(klo[0], khi[0] + 1):
        sx = kx * Lx
        gx = max(amin[0] - (bmax[0] + sx), (bmin[0] + sx) - amax[0], 0.0)
        if gx * gx > best * best:
            continue
        for ky in range(klo[1], khi[1] + 1):
            sy = ky * Ly
            gy = max(amin[1] - (bmax[1] + sy), (bmin[1] + sy) - amax[1], 0.0)
            if gx * gx + gy * gy > best * best:
                continue
            for kz in range(klo[2], khi[2] + 1):
                if same_fiber and kx == 0 and ky == 0 and kz == 0:
                    continue
                sz = kz * Lz
                gz = max(amin[2] - (bmax[2] + sz), (bmin[2] + sz) - amax[2], 0.0)
                if gx * gx + gy * gy + gz * gz > best * best:
                    continue
                d, i, j = _polyline_distance(p1, p2, sx, sy, sz, best)
                if i >= 0 and d < best:
                    best = d
                    bk = (kx, ky, kz)
                    bi = i
                    bj = j
    return best, bk[0], bk[1], bk[2], bi, bj
