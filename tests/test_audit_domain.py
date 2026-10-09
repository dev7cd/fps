# SPDX-FileCopyrightText: 2026 Devine Ngouloubi <exauce-devine.ngouloubi@unicaen.fr>
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
V3 - Domain of the final audit: no implicit success.

Every input the audit cannot conclude on is classified `undetermined`, with a cause:
invalid box, radius or requested margin, non-finite or malformed control points,
duplicate identifiers, a segment with squared length <= 1e-12 (skipped by the
voxelizer), a centerline that is not the interpolation of the control points,
porosity or a non-circular section, a fiber whose periodic image meeting the cell
needs a shift beyond -1/0/+1 (F05), and an error inside the audit.
"""
import json

import numpy as np
import pytest

import validation.geometric_audit as ga
from core.config import FiberPackingConfig
from core.fiber import Fiber

BOX = (1.0, 1.0, 1.0)
CFG = {'fiber_section_type': 'circular', 'section_parameters': {}}


def fiber(fid, pts, r=0.05):
    return Fiber(fid, np.array(pts, dtype=float), r, dict(CFG))


def two_good():
    return [fiber(1, [(0.2, 0.2, 0.5), (0.2, 0.8, 0.5)]), fiber(2, [(0.6, 0.2, 0.5), (0.6, 0.8, 0.5)])]


def undetermined(fibers, box=BOX, requested=0.0, **kw):
    res = ga.audit(fibers, box, requested, **kw)
    assert res['status'] == 'undetermined', res
    assert res['min_clearance'] is None and res['min_clearance_cause']
    return res['cause']


def test_v3_control_passes():
    assert ga.audit(two_good(), BOX, 0.0)['status'] == 'passed'


@pytest.mark.parametrize('box', [(1.0, 0.0, 1.0), (1.0, -1.0, 1.0), (1.0, float('nan'), 1.0),
                                 (1.0, float('inf'), 1.0), (1.0, 1.0), ('a', 1, 1)])
def test_v3_invalid_box(box):
    assert 'box' in undetermined(two_good(), box=box)


@pytest.mark.parametrize('requested', [-1e-3, float('nan'), float('inf'), None, True])
def test_v3_invalid_requested(requested):
    assert 'requested clearance' in undetermined(two_good(), requested=requested)


@pytest.mark.parametrize('r', [0.0, -0.05, float('nan'), float('inf')])
def test_v3_invalid_radius(r):
    fs = two_good()
    fs[0].radius = r
    assert 'radius' in undetermined(fs)


def test_v3_non_finite_control_points():
    fs = two_good()
    fs[1].control_points = np.array([(0.6, 0.2, 0.5), (np.nan, 0.8, 0.5)])
    assert 'control points' in undetermined(fs)


def test_v3_malformed_control_points():
    fs = two_good()
    fs[1].control_points = np.array([0.6, 0.2, 0.5])
    assert 'control points' in undetermined(fs)


def test_v3_duplicate_ids():
    fs = two_good()
    fs[1].id = 1
    assert 'duplicate' in undetermined(fs)


def test_v3_tiny_segment():
    """Two control points 5e-5 apart: 99 segments of squared length 2.6e-13 <= 1e-12."""
    fs = two_good() + [fiber(3, [(0.4, 0.5, 0.2), (0.4, 0.5, 0.2 + 5e-5)], r=1e-6)]
    assert 'squared length' in undetermined(fs)


def test_v3_stale_centerline():
    """Control points moved without refreshing the centerline: exports and adapter would differ."""
    fs = two_good()
    fs[0].control_points = fs[0].control_points + np.array([0.01, 0.0, 0.0])
    assert 'stale geometry' in undetermined(fs)


def test_v3_porosity_and_section():
    assert 'porosity' in undetermined(two_good(), generate_porosity=True)
    assert 'section' in undetermined(two_good(), section_type='superelliptical')


def test_v3_periodic_scope_f05():
    """A fiber spanning more than two periods along x: its image meeting the cell needs a shift of 2."""
    fs = [fiber(1, [(-1.4, 0.5, 0.5), (0.5, 0.5, 0.5)], r=0.01)]
    assert 'shift' in undetermined(fs)


def test_v3_audit_error(monkeypatch):
    def boom(*a, **k):
        raise FloatingPointError('injected')
    monkeypatch.setattr(ga, 'periodic_polyline_distance', boom)
    assert 'audit error' in undetermined(two_good())


def test_v3_record_is_strict_json(tmp_path):
    """Non-finite values become null with their cause; the record parses without NaN/Infinity."""
    cfg = FiberPackingConfig(fiber_section_type='circular', generate_porosity=False, min_clearance=float('nan'),
                             output_prefix=str(tmp_path / 'x'))
    res = ga.audit(two_good(), BOX, cfg.min_clearance)
    rec = ga.build_record(cfg, two_good(), res, 1)
    path = ga.write_record(tmp_path / 'x_validity.json', rec)

    def refuse(name):
        raise ValueError(f'non-strict JSON constant {name}')
    data = json.loads(path.read_text(), parse_constant=refuse)
    assert data['status'] == 'undetermined' and data['requested_clearance'] is None
    assert data['min_clearance'] is None and data['min_clearance_cause']
    assert data['contact_allowed'] is False
