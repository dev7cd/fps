# SPDX-FileCopyrightText: 2026 Devine Ngouloubi <exauce-devine.ngouloubi@unicaen.fr>
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
V4 - Orchestration: deterministic final geometries injected into the real path of main
(`main.finalize`, called by `main.main` after generation and optimisation).

On `failed` or `undetermined`: exit code 1 before porosity, statistics and exporters,
`<prefix>_validity.json` written with the cause, earlier files of the same prefix left
byte-for-byte intact. On `passed`: exports written and the same validity record stored in
the parametric JSON, without changing the fiber data.
"""
import json

import numpy as np
import pytest

import main
from core.config import FiberPackingConfig
from core.fiber import Fiber

CFG = {'fiber_section_type': 'circular', 'section_parameters': {}}
OLD = {'_parametric.json': b'{"old": true}\n', '_roots.csv': b'old roots\n', '_full_periodic.csv': b'old\n',
       '_voxelmap.npy': b'old npy', '_validity.json': b'{"old validity": true}\n'}


def fiber(fid, a, b, r=0.05):
    return Fiber(fid, np.array([a, b], dtype=float), r, dict(CFG))


def config(tmp_path, **kw):
    base = dict(box_dims=(1.0, 1.0, 1.0), seed=1, fiber_section_type='circular', section_parameters={},
                generate_porosity=False, export_mesh=False, compute_spatial_stats=False, voxel_resolution=16,
                min_clearance=0.0, output_prefix=str(tmp_path / 'rve'))
    base.update(kw)
    return FiberPackingConfig(**base)


def seed_old_files(tmp_path):
    for suffix, data in OLD.items():
        (tmp_path / f'rve{suffix}').write_bytes(data)


def forbid_exports(monkeypatch):
    calls = []

    def forbidden(name):
        def f(*a, **k):
            calls.append(name)
            raise AssertionError(f'exporter called after a refused audit: {name}')
        return f
    monkeypatch.setattr(main, 'save_parametric_record_O1', forbidden('parametric'))
    monkeypatch.setattr(main.CSVFiberExporter, 'export', staticmethod(forbidden('csv')))
    monkeypatch.setattr(main, 'Voxelizer', forbidden('voxelizer'))
    monkeypatch.setattr(main, 'GmshExporter', forbidden('gmsh'))
    monkeypatch.setattr(main, 'PorosityGenerator', forbidden('porosity'))
    monkeypatch.setattr(main, 'AD_PCA_Analyzer', forbidden('statistics'))
    return calls


@pytest.mark.parametrize('kind', ['failed', 'undetermined'])
def test_v4_refusal_before_exports(tmp_path, monkeypatch, kind):
    seed_old_files(tmp_path)
    calls = forbid_exports(monkeypatch)
    if kind == 'failed':                                       # two crossing fibers
        fibers = [fiber(1, (0.2, 0.5, 0.5), (0.8, 0.5, 0.5)), fiber(2, (0.5, 0.2, 0.5), (0.5, 0.8, 0.5))]
        cfg = config(tmp_path)
    else:                                                      # requested clearance not finite
        fibers = [fiber(1, (0.2, 0.2, 0.5), (0.2, 0.8, 0.5))]
        cfg = config(tmp_path, min_clearance=float('nan'))
    with pytest.raises(SystemExit) as exc:
        main.finalize(cfg, fibers, 0.0)
    assert exc.value.code == 1
    assert calls == []
    rec = json.loads((tmp_path / 'rve_validity.json').read_text())
    assert rec['status'] == kind and rec['exit_code'] == 1 and rec['cause']
    for suffix, data in OLD.items():
        if suffix != '_validity.json':
            assert (tmp_path / f'rve{suffix}').read_bytes() == data, suffix


def test_v4_passed_exports_with_status(tmp_path):
    seed_old_files(tmp_path)
    fibers = [fiber(1, (0.2, 0.2, 0.5), (0.2, 0.8, 0.5)), fiber(2, (0.6, 0.2, 0.5), (0.6, 0.8, 0.5))]
    cps = [f.control_points.copy() for f in fibers]
    cfg = config(tmp_path)
    main.finalize(cfg, fibers, 0.0)
    rec = json.loads((tmp_path / 'rve_validity.json').read_text())
    param = json.loads((tmp_path / 'rve_parametric.json').read_text())
    assert rec['status'] == 'passed' and rec['exit_code'] == 0 and rec['contact_allowed'] is True
    assert param['validity'] == rec
    assert rec['fibers_sha256'] == main.geometric_audit.fibers_sha256(param['microstructure_data'])
    assert (tmp_path / 'rve_voxelmap.npy').read_bytes() != OLD['_voxelmap.npy']
    for f, cp in zip(fibers, cps):
        assert np.array_equal(f.control_points, cp)


def strict(path):
    def refuse(c):
        raise ValueError(f'non-strict JSON constant {c}')
    return json.loads(path.read_text(), parse_constant=refuse)


def test_v4_section_refused_by_the_real_launcher(tmp_path, monkeypatch):
    """R10: main.main with a non-circular section accepted by the parser: code 1, strict `undetermined`
    diagnostic, no generation and no exporter, earlier exports intact, unknown geometry left unknown."""
    seed_old_files(tmp_path)
    calls = forbid_exports(monkeypatch)

    def no_generation(*a, **k):
        raise AssertionError('generation started for a refused section')
    monkeypatch.setattr(main, 'FiberGenerator', no_generation)
    monkeypatch.setattr(main, 'CollisionDetector', no_generation)
    monkeypatch.setattr('sys.argv', ['main.py', '--section', 'superelliptical', '--no_mesh', '--res_fft', '16',
                                     '--output', str(tmp_path / 'rve')])
    with pytest.raises(SystemExit) as exc:
        main.main()
    assert exc.value.code == 1 and calls == []
    rec = strict(tmp_path / 'rve_validity.json')
    assert rec['status'] == 'undetermined' and rec['exit_code'] == 1 and 'superelliptical' in rec['cause']
    assert rec['stage'] == 'input validation (section)'
    assert rec['n_fibers'] is None and rec['fibers_sha256'] is None and rec['audited_geometry_sha256'] is None
    assert rec['hash_cause'] == 'geometry not produced' and rec['min_clearance'] is None
    for suffix, data in OLD.items():
        if suffix != '_validity.json':
            assert (tmp_path / f'rve{suffix}').read_bytes() == data, suffix


def nominal_fibers():
    return [fiber(1, (0.2, 0.2, 0.5), (0.2, 0.8, 0.5)), fiber(2, (0.6, 0.2, 0.5), (0.6, 0.8, 0.5))]


@pytest.mark.parametrize('bad', [float('nan'), float('inf')])
def test_v4_non_finite_self_distance_stops_exports(tmp_path, monkeypatch, bad):
    """R11 in finalize: non-finite self-image distance, finite inter-fiber distance -> refusal before exports."""
    seed_old_files(tmp_path)
    calls = forbid_exports(monkeypatch)
    real = main.geometric_audit.periodic_polyline_distance

    def fake(p1, p2, Lx, Ly, Lz, same):
        return (bad, 0, 0, 0, -1, -1) if same else real(p1, p2, Lx, Ly, Lz, same)
    monkeypatch.setattr(main.geometric_audit, 'periodic_polyline_distance', fake)
    with pytest.raises(SystemExit) as exc:
        main.finalize(config(tmp_path), nominal_fibers(), 0.0)
    assert exc.value.code == 1 and calls == []
    rec = strict(tmp_path / 'rve_validity.json')
    assert rec['status'] == 'undetermined' and 'own images' in rec['cause']


def test_v4_hash_failure_turns_success_into_refusal(tmp_path, monkeypatch):
    """R12: nominal geometry (audit passed) but the hash cannot be built -> undetermined, code 1, no export."""
    seed_old_files(tmp_path)
    calls = forbid_exports(monkeypatch)

    def broken(records):
        raise RuntimeError('injected hash failure')
    monkeypatch.setattr(main.geometric_audit, 'fibers_sha256', broken)
    with pytest.raises(SystemExit) as exc:
        main.finalize(config(tmp_path), nominal_fibers(), 0.0)
    assert exc.value.code == 1 and calls == []
    rec = strict(tmp_path / 'rve_validity.json')
    assert rec['status'] == 'undetermined' and rec['stage'] == 'hashes' and rec['exit_code'] == 1
    assert rec['fibers_sha256'] is None and 'injected hash failure' in rec['hash_cause']
    assert rec['min_clearance'] is not None                # the geometric result is kept in the diagnostic
    for suffix, data in OLD.items():
        if suffix != '_validity.json':
            assert (tmp_path / f'rve{suffix}').read_bytes() == data, suffix
