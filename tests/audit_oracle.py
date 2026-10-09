# SPDX-FileCopyrightText: 2026 Devine Ngouloubi <exauce-devine.ngouloubi@unicaen.fr>
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Exact reference for the F03 audit tests (V1, V2), independent of the audit kernel.

The float coordinates actually given to the kernel are converted to exact rationals.
The squared distance f(s, t) = |r + s d1 - t d2|^2 (r = A - C) is minimised exactly
over [0, 1]^2: interior stationary point when det = a e - b^2 > 0, and the
minimiser of each of the four edges, clamped. Only the final square root uses
high-precision decimals (60 digits). Nothing here calls validation/audit_distance.py
or collision/detector_math.py.
"""
from decimal import Decimal, getcontext
from fractions import Fraction
import itertools

getcontext().prec = 60


def _q(p):
    return [Fraction(float(x)) for x in p]


def _dot(u, v):
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]


def _sub(u, v):
    return [u[0] - v[0], u[1] - v[1], u[2] - v[2]]


def _clamp(x):
    return min(max(x, Fraction(0)), Fraction(1))


def segment_distance_sq(A, B, C, D):
    """Exact squared distance between segments [A, B] and [C, D] (rational inputs)."""
    d1, d2, r = _sub(B, A), _sub(D, C), _sub(A, C)
    a, b, e = _dot(d1, d1), _dot(d1, d2), _dot(d2, d2)
    c, f, g = _dot(d1, r), _dot(d2, r), _dot(r, r)

    def F(s, t):
        return a * s * s - 2 * b * s * t + e * t * t + 2 * c * s - 2 * f * t + g

    cands = []
    det = a * e - b * b
    if det > 0:
        s = (b * f - c * e) / det
        t = (a * f - b * c) / det
        if 0 <= s <= 1 and 0 <= t <= 1:
            cands.append(F(s, t))
    for s in (Fraction(0), Fraction(1)):            # edges s = 0, 1 : minimise in t
        t = _clamp((b * s + f) / e) if e > 0 else Fraction(0)
        cands.append(F(s, t))
    for t in (Fraction(0), Fraction(1)):            # edges t = 0, 1 : minimise in s
        s = _clamp((b * t - c) / a) if a > 0 else Fraction(0)
        cands.append(F(s, t))
    return min(cands)


def sqrt_exact(q):
    return float(Decimal(q.numerator).sqrt() / Decimal(q.denominator).sqrt()) if q > 0 else 0.0


def segment_distance(A, B, C, D):
    return sqrt_exact(segment_distance_sq(_q(A), _q(B), _q(C), _q(D)))


def periodic_polyline_distance(p1, p2, L, same_fiber, K=3):
    """Brute force over shifts k in [-K, K]^3 (zero shift skipped for the same fiber), exact per segment pair.
    Shifted coordinates are formed exactly in rationals from the float inputs."""
    P1 = [_q(p) for p in p1]
    P2 = [_q(p) for p in p2]
    Lq = _q(L)
    best = None
    for k in itertools.product(range(-K, K + 1), repeat=3):
        if same_fiber and k == (0, 0, 0):
            continue
        s = [k[a] * Lq[a] for a in range(3)]
        Q = [[p[a] + s[a] for a in range(3)] for p in P2]
        for i in range(len(P1) - 1):
            for j in range(len(Q) - 1):
                d = segment_distance_sq(P1[i], P1[i + 1], Q[j], Q[j + 1])
                if best is None or d < best[0]:
                    best = (d, k)
    return sqrt_exact(best[0]), best[1]
