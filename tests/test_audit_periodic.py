# SPDX-FileCopyrightText: 2026 Devine Ngouloubi <exauce-devine.ngouloubi@unicaen.fr>
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
V2 - Periodic images of the audit kernel against an independent brute-force reference.

The reference scans every shift in [-3, 3]^3 (zero shift skipped for a fiber against
itself) with the exact rational segment distance, without the audit's image range.
Tolerance per pair: tau = 1e-12 * max segment length. Cases: pairs close across faces,
edges and corners of a non-cubic box, a fiber longer than the box meeting its own
non-zero image, and a control without collision. A decision test runs the full audit
on Fiber objects whose straight centerlines have known distances.
"""
import numpy as np
import pytest

from core.fiber import Fiber
from tests.audit_oracle import periodic_polyline_distance as reference
from validation.audit_distance import periodic_polyline_distance
from validation.geometric_audit import audit

L = (1.0, 2.0, 0.5)


def poly(*pts):
    return np.ascontiguousarray(pts, dtype=np.float64)


CASES = {
    'face x': (poly((0.05, 1.0, 0.25), (0.08, 1.3, 0.2), (0.06, 1.6, 0.3)),
               poly((0.97, 1.1, 0.22), (0.95, 1.4, 0.27)), False),
    'arete xy': (poly((0.02, 0.03, 0.25), (0.2, 0.3, 0.2)),
                 poly((0.98, 1.97, 0.26), (0.8, 1.7, 0.3)), False),
    'coin xyz': (poly((0.01, 0.02, 0.01), (0.1, 0.15, 0.08)),
                 poly((0.99, 1.99, 0.495), (0.9, 1.85, 0.42)), False),
    'image propre (fibre plus longue que la boite)': (poly((0.1, 0.5, 0.25), (0.7, 0.55, 0.25), (1.2, 0.6, 0.25)),
                                                       None, True),
    'image propre repliee': (poly((0.2, 0.2, 0.1), (0.6, 0.9, 0.3), (0.3, 1.7, 0.4), (0.25, 2.15, 0.12)), None, True),
    'temoin sans collision': (poly((0.4, 0.9, 0.2), (0.5, 1.1, 0.25)),
                              poly((0.1, 0.3, 0.05), (0.15, 0.4, 0.07)), False),
}


@pytest.mark.parametrize('name', list(CASES))
def test_v2_images_against_reference(name):
    p1, p2, same = CASES[name]
    p2 = p1 if p2 is None else p2
    d, kx, ky, kz, i, j = periodic_polyline_distance(p1, p2, *L, same)
    ref, kref = reference(p1, p2, L, same)
    ell = max(np.linalg.norm(np.diff(p1, axis=0), axis=1).max(), np.linalg.norm(np.diff(p2, axis=0), axis=1).max())
    tau = 1e-12 * ell
    assert abs(d - ref) <= tau, (d, ref)
    assert d - ref <= tau
    if same:
        assert (kx, ky, kz) != (0, 0, 0)


def straight_fiber(fid, a, b, r):
    """Fiber with two control points: the centerline is the linear resampling of [a, b]."""
    return Fiber(fid, np.array([a, b], dtype=float), r, {'fiber_section_type': 'circular', 'section_parameters': {}})


@pytest.mark.parametrize('gap, requested, expected', [
    (0.02, 0.0, 'passed'), (0.02, 0.019, 'passed'), (0.02, 0.021, 'failed'), (-0.01, 0.0, 'failed'),
    (0.0, 0.0, 'passed'),
])
def test_v2_decision_across_face(gap, requested, expected):
    """Two parallel straight fibers along y, close through the x face: axis distance known exactly.

    The contact case (gap 0, requested 0) is a numerical decision at the threshold: with these float
    positions the computed clearance is a few 1e-17, accepted as contact (contact allowed at zero margin).
    """
    r = 0.05
    d = 2 * r + gap                                   # axis distance through the x = 0 / x = 1 face
    f1 = straight_fiber(1, (d / 2, 0.5, 0.25), (d / 2, 1.5, 0.25), r)
    f2 = straight_fiber(2, (1.0 - d / 2, 0.5, 0.25), (1.0 - d / 2, 1.5, 0.25), r)
    res = audit([f1, f2], L, requested)
    assert res['status'] == expected, res
    assert res['min_clearance'] == pytest.approx(gap, abs=1e-12)
    assert res['responsible_pair']['fibers'] == [1, 2]
    assert abs(res['responsible_pair']['shift_periods'][0]) == 1


def test_v2_self_image_collision():
    """A straight fiber longer than the box along x meets its own image shifted by one period."""
    f = straight_fiber(7, (0.1, 1.0, 0.25), (1.3, 1.0, 0.25), 0.02)
    res = audit([f], L, 0.0)
    assert res['status'] == 'failed'
    assert res['responsible_pair']['fibers'] == [7, 7]
    assert res['min_clearance'] == pytest.approx(-0.04, abs=1e-12)
