# SPDX-FileCopyrightText: 2026 Devine Ngouloubi <exauce-devine.ngouloubi@unicaen.fr>
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Hashes of the validity record (fiche D10, "Regle d'empreinte"), recomputed independently.

fibers_sha256: SHA-256 of microstructure_data as written to the parametric JSON (ids, control
points, radii, real volumes), sorted keys, separators (',', ':'), list order kept.
audited_geometry_sha256: SHA-256 of the strict canonical JSON object holding the rule version,
fibers_sha256, box and unit, circular section, flat ends, interpolation convention, linear-fallback
fiber ids and FPS commit. Separate changes of points, radii, identifiers and interpolation
convention must change the hashes they cover.
"""
import hashlib
import json

import numpy as np

import validation.geometric_audit as ga
from core.config import FiberPackingConfig
from core.fiber import Fiber

CFG = {'fiber_section_type': 'circular', 'section_parameters': {}}
BOX = (1.0, 1.0, 1.0)


def fibers(points=None, radii=(0.05, 0.05), ids=(1, 2)):
    pts = points or [[(0.2, 0.2, 0.5), (0.25, 0.5, 0.5), (0.2, 0.8, 0.5)],
                     [(0.6, 0.2, 0.5), (0.6, 0.5, 0.5), (0.6, 0.8, 0.5)]]
    return [Fiber(i, np.array(p, dtype=float), r, dict(CFG)) for i, p, r in zip(ids, pts, radii)]


def independent_fibers_sha(fs):
    data = [{'id': int(f.id), 'control_points': np.asarray(f.control_points, dtype=float).tolist(),
             'radius': float(f.radius), 'geometry': {'real_volume': float(f.get_real_volume())}} for f in fs]
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def independent_geometry_sha(fib_sha, fs, commit, sampling):
    obj = {'rule_version': 'fps-audited-geometry-1', 'fibers_sha256': fib_sha, 'box': list(BOX),
           'length_unit': 'box length unit (code units)', 'section': 'circular', 'end_convention': 'flat',
           'interpolation': sampling,
           'linear_fallback_fiber_ids': sorted(int(f.id) for f in fs if len(f.control_points) < 4),
           'fps_commit': commit}
    canon = json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(canon.encode('utf-8')).hexdigest()


def record(fs):
    cfg = FiberPackingConfig(box_dims=BOX, fiber_section_type='circular', generate_porosity=False)
    return ga.build_record(cfg, fs, ga.audit(fs, BOX, 0.0), 0)


def test_hashes_recomputed_independently():
    fs = fibers()
    rec = record(fs)
    assert rec['status'] == 'passed'
    assert rec['fibers_sha256'] == independent_fibers_sha(fs)
    assert rec['audited_geometry_sha256'] == independent_geometry_sha(rec['fibers_sha256'], fs,
                                                                      rec['fps']['commit'], ga.SAMPLING)


def test_separate_changes_change_the_hashes(monkeypatch):
    base = record(fibers())
    moved = [[(0.2, 0.2, 0.5), (0.25, 0.5 + 1e-9, 0.5), (0.2, 0.8, 0.5)],
             [(0.6, 0.2, 0.5), (0.6, 0.5, 0.5), (0.6, 0.8, 0.5)]]
    variants = {'point': record(fibers(points=moved)), 'radius': record(fibers(radii=(0.05, 0.0500001))),
                'identifier': record(fibers(ids=(1, 3)))}
    for name, rec in variants.items():
        assert rec['fibers_sha256'] != base['fibers_sha256'], name
        assert rec['audited_geometry_sha256'] != base['audited_geometry_sha256'], name
    # Interpolation convention: same fibers, different declared sampling -> only the geometry hash changes.
    monkeypatch.setattr(ga, 'SAMPLING', dict(ga.SAMPLING, num_points=200))
    other = record(fibers())
    assert other['fibers_sha256'] == base['fibers_sha256']
    assert other['audited_geometry_sha256'] != base['audited_geometry_sha256']
