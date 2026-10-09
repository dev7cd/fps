# SPDX-FileCopyrightText: 2026 Devine Ngouloubi <exauce-devine.ngouloubi@unicaen.fr>
# SPDX-License-Identifier: AGPL-3.0-or-later

## @file geometric_audit.py
#  @brief Blocking final geometric audit (F03) and its validity record.
#  @details Status of an attempt, written to `<prefix>_validity.json`:
#
#  - `passed`: computed minimum clearance >= requested clearance, within the
#    declared scope (contact allowed when the requested clearance is zero);
#  - `failed`: envelope intersection or insufficient clearance;
#  - `undetermined`: unusable input, audit error or domain not covered.
#
#  Only `passed` allows the exports. Scope: a sufficient condition for the
#  tubes of the sampled centerline (flat-ended cylinders per segment plus
#  vertex sectors, as rasterised by export/voxelizer.py) to be disjoint, for the
#  audit kernel validation/audit_distance.py qualified by the V1 tests. The
#  continuous spline, self-intersections of a fiber with itself (zero shift),
#  the separation of fibers in the voxel grid, non-circular sections and pores
#  are not covered. Margins are fiber-axis distances minus the sum of radii,
#  in the length unit of the box.

import datetime
import hashlib
import json
import math
import os
import subprocess
import uuid
from pathlib import Path

import numpy as np

from geometry.curves import CatmullRomSpline
from validation.audit_distance import periodic_polyline_distance

SCHEMA_VERSION = 'fps-validity-1'
HASH_RULE_VERSION = 'fps-audited-geometry-1'
NUM_POINTS = 100
SEGMENT_C2_MIN = 1e-12          # squared length (code units); segments at or below are skipped by the voxelizer
FPS_ROOT = Path(__file__).resolve().parents[1]

METHOD = ('exact segment-to-segment distance between the sampled centerlines (validation/audit_distance.py: '
          'four endpoint-to-segment candidates and the interior stationary point from n = d1 x d2), minimised '
          'over every periodic image that can hold the minimum; clearance = distance - (r_i + r_j); a fiber '
          'against its own non-zero images: distance - 2 r_i')
SAMPLING = {'routine': 'geometry.curves.CatmullRomSpline.interpolate', 'num_points': NUM_POINTS, 'alpha': 0.5,
            'form': 'uniform Catmull-Rom basis per span with ghost end points, then the routine resampling; '
                    'alpha=0.5 does not make the parameterisation centripetal',
            'linear_fallback': 'fewer than 4 control points: linear resampling (CatmullRomSpline._linear_resample)'}
SCOPE = ('sufficient condition for the tubes of the sampled centerline (flat-ended cylinders per segment and '
         'vertex sectors) to be disjoint, for the audit kernel qualified by V1; contact allowed when the '
         'requested clearance is zero')
EXCLUDED = ['continuous spline (only the 100-point centerline is audited)',
            'self-intersections of a fiber with itself (zero shift not examined)',
            'exact intersection predicate at flat ends (capsule test: conservative refusals possible)',
            'separation or connectivity of fibers in the voxel grid',
            'periodic exports needing shifts beyond -1/0/+1 (refused as undetermined)',
            'non-circular sections', 'pores']
UNITS = {'lengths': 'box length unit (code units)', 'clearance': 'box length unit'}


class AuditInputError(ValueError):
    """Input that the audit cannot conclude on: the status is `undetermined`."""


def _finite(x):
    return x is not None and isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, bool) \
        and math.isfinite(float(x))


def json_safe(obj):
    """Strict JSON: numpy types converted, non-finite floats replaced by None (the caller records the cause)."""
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj) if math.isfinite(float(obj)) else None
    if isinstance(obj, (str, int, bool)) or obj is None:
        return obj
    return str(obj)


def microstructure_records(fibers):
    """Entries of `microstructure_data`, as written to the parametric JSON (main.save_parametric_record_O1)."""
    return [{'id': f.id, 'control_points': f.control_points, 'radius': f.radius,
             'geometry': {'real_volume': f.get_real_volume()}} for f in fibers]


def fibers_sha256(records):
    """D07 canonical hash: SHA-256 of microstructure_data as read back from JSON, sorted keys, compact."""
    data = json.loads(json.dumps(json_safe(records)))
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def fps_provenance():
    """Commit and state of the FPS working tree, or None with the cause."""
    try:
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=FPS_ROOT, capture_output=True, text=True,
                                check=True).stdout.strip()
        dirty = subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=FPS_ROOT,
                               capture_output=True, text=True, check=True).stdout.strip() != ''
        return {'commit': commit, 'dirty': dirty, 'cause': None}
    except (OSError, subprocess.CalledProcessError) as e:
        return {'commit': None, 'dirty': None, 'cause': f'git unavailable: {type(e).__name__}'}


def audited_geometry_sha256(fib_sha, box_dims, linear_fallback_ids, commit):
    """Hash binding the status to the audited geometry and to the interpolation that produced it."""
    obj = {'rule_version': HASH_RULE_VERSION, 'fibers_sha256': fib_sha,
           'box': [float(x) for x in box_dims], 'length_unit': UNITS['lengths'],
           'section': 'circular', 'end_convention': 'flat', 'interpolation': SAMPLING,
           'linear_fallback_fiber_ids': sorted(int(i) for i in linear_fallback_ids), 'fps_commit': commit}
    canon = json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(canon.encode('utf-8')).hexdigest(), obj


def _validate(fibers, box_dims, requested, section_type, generate_porosity):
    """Raises AuditInputError for any input the audit cannot conclude on; returns (L, centerlines)."""
    if generate_porosity:
        raise AuditInputError('porosity requested: pores are outside the audited domain')
    if section_type != 'circular':
        raise AuditInputError(f'section {section_type!r}: only circular sections are in the audited domain')
    try:
        L = [float(x) for x in box_dims]
    except (TypeError, ValueError):
        raise AuditInputError(f'box dimensions {box_dims!r} are not numbers') from None
    if len(L) != 3 or not all(math.isfinite(x) and x > 0 for x in L):
        raise AuditInputError(f'box dimensions {box_dims!r}: three positive finite values required')
    if not _finite(requested) or float(requested) < 0:
        raise AuditInputError(f'requested clearance {requested!r}: finite value >= 0 required')
    if not fibers:
        raise AuditInputError('no fiber to audit')
    ids = [f.id for f in fibers]
    if len(set(ids)) != len(ids):
        raise AuditInputError('duplicate fiber identifiers')
    lines = []
    for f in fibers:
        if not _finite(f.radius) or float(f.radius) <= 0:
            raise AuditInputError(f'fiber {f.id}: radius {f.radius!r} must be positive and finite')
        cp = np.asarray(f.control_points, dtype=float)
        if cp.ndim != 2 or cp.shape[1] != 3 or cp.shape[0] < 2 or not np.all(np.isfinite(cp)):
            raise AuditInputError(f'fiber {f.id}: control points must be a finite (n >= 2, 3) array')
        line = np.ascontiguousarray(CatmullRomSpline.interpolate(cp, num_points=NUM_POINTS), dtype=np.float64)
        cur = np.asarray(f.centerline, dtype=float) if f.centerline is not None else None
        if cur is None or cur.shape != line.shape or not np.array_equal(cur, line):
            raise AuditInputError(f'fiber {f.id}: centerline differs from the interpolation of its control points '
                                  f'(stale geometry); exports and the adapter would not see the same object')
        if line.ndim != 2 or line.shape[1] != 3 or line.shape[0] < 2 or not np.all(np.isfinite(line)):
            raise AuditInputError(f'fiber {f.id}: centerline must be a finite (n >= 2, 3) array')
        c2 = np.sum(np.diff(line, axis=0) ** 2, axis=1)
        if np.any(c2 <= SEGMENT_C2_MIN):
            k = int(np.argmax(c2 <= SEGMENT_C2_MIN))
            raise AuditInputError(f'fiber {f.id}: segment {k} has squared length {c2[k]:.3e} <= {SEGMENT_C2_MIN:g} '
                                  f'(skipped by the voxelizer)')
        lines.append(line)
    return L, lines


def _periodic_scope(fibers, lines, L):
    """F05 guard: every image of a fiber box (centerline +- radius) meeting the open cell must need a shift
    in {-1, 0, 1} on each axis, as PeriodicManager.generate_ghosts only scans these neighbours."""
    for f, line in zip(fibers, lines):
        lo = line.min(axis=0) - f.radius
        hi = line.max(axis=0) + f.radius
        ks = []
        for a in range(3):
            # shifts k with hi + kL > 0 and lo + kL < L
            k_min = math.floor(-hi[a] / L[a]) + 1
            k_max = math.ceil((L[a] - lo[a]) / L[a]) - 1
            ks.append((k_min, k_max))
        if any(k_max < k_min for k_min, k_max in ks):
            continue                                       # no image meets the cell
        for a, (k_min, k_max) in enumerate(ks):
            if k_min < -1 or k_max > 1:
                raise AuditInputError(f'fiber {f.id}: an image meeting the cell needs a shift of '
                                      f'{k_min if k_min < -1 else k_max} periods along axis {"xyz"[a]} '
                                      f'(exports only handle -1/0/+1)')


def audit(fibers, box_dims, requested, section_type='circular', generate_porosity=False):
    """!
    @brief Final audit of the fibers transmitted to the exports.
    @return dict with status, cause, stage, clearance, responsible pair and the hashes.
    """
    res = {'status': 'undetermined', 'cause': None, 'stage': 'input validation', 'min_clearance': None,
           'min_clearance_cause': 'not computed', 'responsible_pair': None}
    try:
        L, lines = _validate(fibers, box_dims, requested, section_type, generate_porosity)
        res['stage'] = 'periodic scope (F05)'
        _periodic_scope(fibers, lines, L)
        res['stage'] = 'clearance'
        best = math.inf
        pair = None
        for i, fi in enumerate(fibers):
            for j in range(i, len(fibers)):
                fj = fibers[j]
                d, kx, ky, kz, si, sj = periodic_polyline_distance(lines[i], lines[j], L[0], L[1], L[2], i == j)
                if not math.isfinite(d):
                    if i == j:
                        continue                               # no non-zero image of a fiber near itself
                    raise AuditInputError(f'no distance computed between fibers {fi.id} and {fj.id}')
                gap = d - (float(fi.radius) + float(fj.radius))
                if gap < best:
                    best = gap
                    pair = {'fibers': [fi.id, fj.id], 'shift_periods': [int(kx), int(ky), int(kz)],
                            'segments': [int(si), int(sj)], 'axis_distance': float(d)}
        if not math.isfinite(best):
            raise AuditInputError('no clearance computed')
        res.update(min_clearance=float(best), min_clearance_cause=None, responsible_pair=pair, stage='decision')
        if best >= float(requested):
            res.update(status='passed', cause=None)
        elif best < 0:
            res.update(status='failed', cause=f'envelope intersection: clearance {best:.6e} < 0')
        else:
            res.update(status='failed', cause=f'clearance {best:.6e} below the requested {float(requested):.6e}')
    except AuditInputError as e:
        res.update(status='undetermined', cause=str(e))
    except Exception as e:                                    # audit error: never a success
        res.update(status='undetermined', cause=f'audit error: {type(e).__name__}: {e}')
    return res


def build_record(config, fibers, result, exit_code):
    """Validity record of an attempt (strict JSON once passed through json_safe)."""
    prov = fps_provenance()
    try:
        records = microstructure_records(fibers)
        fib_sha = fibers_sha256(records)
        fallback = [f.id for f in fibers if len(np.asarray(f.control_points)) < 4]
        geo_sha, geo_obj = audited_geometry_sha256(fib_sha, config.box_dims, fallback, prov['commit'])
        hash_cause = None
    except Exception as e:
        fib_sha = geo_sha = geo_obj = None
        hash_cause = f'hash not computed: {type(e).__name__}: {e}'
    requested = getattr(config, 'min_clearance', None)
    return {
        'schema_version': SCHEMA_VERSION,
        'attempt': {'id': uuid.uuid4().hex, 'timestamp': datetime.datetime.now().isoformat(timespec='seconds')},
        'status': result['status'], 'cause': result['cause'], 'stage': result['stage'],
        'exit_code': exit_code,
        'fps': prov,
        'seed': getattr(config, 'seed', None),
        'config_effective': json_safe(vars(config)),
        'n_fibers': len(fibers),
        'fibers_sha256': fib_sha,
        'audited_geometry_sha256': geo_sha,
        'audited_geometry': geo_obj,
        'hash_cause': hash_cause,
        'method': METHOD, 'sampling': SAMPLING, 'scope': SCOPE, 'excluded_checks': EXCLUDED, 'units': UNITS,
        'requested_clearance': requested if _finite(requested) else None,
        'contact_allowed': bool(_finite(requested) and float(requested) == 0.0),
        'min_clearance': result['min_clearance'],
        'min_clearance_cause': result['min_clearance_cause'],
        'responsible_pair': result['responsible_pair'],
    }


def write_record(path, record):
    """Atomic strict-JSON write (no NaN/Infinity); returns the path."""
    path = Path(path)
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(json_safe(record), indent=2, allow_nan=False, ensure_ascii=False) + '\n')
    os.replace(tmp, path)
    return path
