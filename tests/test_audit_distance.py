# SPDX-FileCopyrightText: 2026 Devine Ngouloubi <exauce-devine.ngouloubi@unicaen.fr>
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
V1 - Qualification of the audit distance kernel against the exact rational oracle.

Tolerance tau = 1e-12 * max(l1, l2) per pair (point-point distance for two points;
zero for coincident points). Two properties are checked: |d - d_ref| <= tau and,
reported separately, d - d_ref <= tau (an overestimate can allow a false success).
Each case is repeated after translation, swap of the segments and reversal of the
endpoints. Counter-test: the historical generation kernel fails the qualification on
the crossing of length 1e-4 (5e-5 instead of 0).
"""
import math

import numpy as np
import pytest

from collision.detector_math import _segment_segment_closest
from tests.audit_oracle import segment_distance
from validation.audit_distance import segment_segment_distance


def audit_kernel(A, B, C, D):
    return segment_segment_distance(*A, *B, *C, *D)


def historical_kernel(A, B, C, D):
    return math.sqrt(_segment_segment_closest(*A, *B, *C, *D)[0])


def tau(A, B, C, D):
    l1 = float(np.linalg.norm(np.subtract(B, A)))
    l2 = float(np.linalg.norm(np.subtract(D, C)))
    if l1 == 0.0 and l2 == 0.0:
        return 1e-12 * float(np.linalg.norm(np.subtract(C, A)))
    return 1e-12 * max(l1, l2)


def variants(A, B, C, D):
    """Translation, swap of the segments and reversal of the endpoints.

    The translation is proportional to the largest segment length, so that the translated
    coordinates stay in the precision domain of the inputs: a shift of order 1 applied to
    segments of length 1e-6 would round the coordinates by more than tau (fiche D10).
    """
    A, B, C, D = (np.asarray(x, dtype=float) for x in (A, B, C, D))
    ell = max(float(np.linalg.norm(B - A)), float(np.linalg.norm(D - C)), float(np.linalg.norm(C - A)))
    shift = np.array([0.37, -1.25, 2.5]) * ell
    yield A, B, C, D
    yield A + shift, B + shift, C + shift, D + shift
    yield C, D, A, B
    yield B, A, D, C
    yield D, C, B, A


def qualify(kernel, cases):
    """List of failures: (name, d, d_ref, tau, overestimate) for every case and variant."""
    failures = []
    for name, (A, B, C, D) in cases:
        for v in variants(A, B, C, D):
            d = kernel(*v)
            ref = segment_distance(*v)
            t = tau(*v)
            if not abs(d - ref) <= t:
                failures.append((name, d, ref, t, d - ref > t))
    return failures


def perpendicular_crossing(L):
    return (-L / 2, 0.0, 0.0), (L / 2, 0.0, 0.0), (0.0, -L / 2, 0.0), (0.0, L / 2, 0.0)


def near_parallel_crossing(L, theta, h=0.0):
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return ((-L / 2 * c, -L / 2 * s, 0.0), (L / 2 * c, L / 2 * s, 0.0),
            (-L / 2 * c, L / 2 * s, h), (L / 2 * c, -L / 2 * s, h))


SCALES = [10.0 ** (-k) for k in range(0, 7)]


def deterministic_cases():
    cases = []
    for L in SCALES:
        cases.append((f'croisement perpendiculaire L={L:g}', perpendicular_crossing(L)))
        cases.append((f'croisement aux extremites (L) L={L:g}', ((0, 0, 0), (L, 0, 0), (0, 0, 0), (0, L, 0))))
        cases.append((f'T L={L:g}', ((-L, 0, 0), (L, 0, 0), (0, 0, 0), (0, L, 0))))
        cases.append((f'gauche L={L:g}', ((0, 0, 0), (L, 0, 0), (L / 2, -L, L / 3), (L / 2, L, L / 3))))
        cases.append((f'gauche exterieur L={L:g}', ((0, 0, 0), (L, 0, 0), (2 * L, -L, L / 3), (2 * L, L, L / 3))))
        cases.append((f'paralleles decales L={L:g}', ((0, 0, 0), (L, 0, 0), (L / 3, L / 7, 0), (2 * L, L / 7, 0))))
        cases.append((f'colineaires recouvrants L={L:g}', ((0, 0, 0), (L, 0, 0), (L / 2, 0, 0), (2 * L, 0, 0))))
        cases.append((f'colineaires disjoints L={L:g}', ((0, 0, 0), (L, 0, 0), (2 * L, 0, 0), (3 * L, 0, 0))))
        cases.append((f'point-segment L={L:g}', ((L / 3, L / 5, L / 7), (L / 3, L / 5, L / 7), (0, 0, 0), (L, 0, 0))))
        cases.append((f'point-point L={L:g}', ((0, 0, 0), (0, 0, 0), (L, L / 2, 0), (L, L / 2, 0))))
        for theta in (1e-2, 1e-4, 1e-6, 1e-8):
            cases.append((f'presque paralleles croises L={L:g} theta={theta:g}', near_parallel_crossing(L, theta)))
            cases.append((f'presque paralleles ecartes L={L:g} theta={theta:g}',
                          near_parallel_crossing(L, theta, h=L * 1e-3)))
    cases.append(('points confondus', ((0.3, 0.3, 0.3), (0.3, 0.3, 0.3), (0.3, 0.3, 0.3), (0.3, 0.3, 0.3))))
    return cases


def random_cases(n=1500, seed=20261009):
    rng = np.random.default_rng(seed)
    cases = []
    for k in range(n):
        L = 10.0 ** rng.uniform(-6, 0)
        A = rng.normal(size=3) * L
        d1 = rng.normal(size=3) * L
        mode = k % 3
        if mode == 0:
            C, d2 = rng.normal(size=3) * L, rng.normal(size=3) * L
        elif mode == 1:                                   # nearly parallel through a common point
            theta = 10.0 ** rng.uniform(-9, -2)
            axis = np.cross(d1, rng.normal(size=3))
            axis /= np.linalg.norm(axis)
            u = d1 / np.linalg.norm(d1)
            d2 = (u * math.cos(theta) + np.cross(axis, u) * math.sin(theta)) * L * rng.uniform(0.5, 2)
            m = A + d1 * rng.uniform(0.1, 0.9)
            C = m - d2 * rng.uniform(0.1, 0.9)
        else:                                             # crossing through an interior point
            d2 = rng.normal(size=3) * L
            m = A + d1 * rng.uniform(0, 1)
            C = m - d2 * rng.uniform(0, 1)
        cases.append((f'aleatoire {k}', (tuple(A), tuple(A + d1), tuple(C), tuple(C + d2))))
    return cases


def test_v1_deterministic():
    failures = qualify(audit_kernel, deterministic_cases())
    assert failures == [], failures[:5]


def test_v1_random():
    failures = qualify(audit_kernel, random_cases())
    assert failures == [], failures[:5]


@pytest.mark.parametrize('case', [perpendicular_crossing(1e-4), near_parallel_crossing(1.0, 1e-8)],
                         ids=['croisement 1e-4', 'croisement theta 1e-8'])
def test_v1_imposed_crossings_are_zero(case):
    """The two imposed interior crossings: distance zero, hence no false success."""
    assert segment_distance(*case) == 0.0
    assert audit_kernel(*case) <= tau(*case)


def test_v1_decisions_with_radii():
    """Decision on the clearance d - (r1 + r2) against the requested margin, compared with the reference."""
    for name, case in deterministic_cases():
        ref = segment_distance(*case)
        for r, requested in ((1e-5, 0.0), (ref / 4, 0.0), (ref / 4, ref / 4)):
            if r <= 0:
                continue
            gap_ref = ref - 2 * r
            gap = audit_kernel(*case) - 2 * r
            if abs(gap_ref - requested) > 10 * tau(*case):     # cases away from the threshold
                assert (gap >= requested) == (gap_ref >= requested), (name, r, requested, gap, gap_ref)


def test_v1_counter_test_historical_kernel_fails():
    """The historical generation kernel overestimates the crossing of length 1e-4: the qualification fails."""
    case = perpendicular_crossing(1e-4)
    assert historical_kernel(*case) == pytest.approx(5e-5, rel=1e-12)
    failures = qualify(historical_kernel, [('contre-exemple historique', case)])
    assert failures and all(f[4] for f in failures)            # overestimates: false-success direction
