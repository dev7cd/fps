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
#  the separation of fibers in the voxel grid and non-circular sections are not
#  covered. Margins are fiber-axis distances minus the sum of radii, in the
#  length unit of the box.
#
#  Spherical pores are generated after this audit; audit_pores() then checks,
#  as a second blocking step, that they overlap neither one another nor any
#  fiber (periodic images included) before anything is exported.

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

METHOD = ('segment-to-segment distance between the sampled centerlines (validation/audit_distance.py), in double '
          'precision: minimum of the four endpoint-to-segment distances and of the interior candidate; interior '
          'candidate from n = d1 x d2 (line distance |r.n|/|n|) when |n|^2 > 1e-4 a e, otherwise golden-section '
          'search of the convex function dist(P(s), segment) down to |ds| <= 1e-14; minimised over every periodic '
          'image that can hold the minimum; clearance = distance - (r_i + r_j), a fiber against its own non-zero '
          'images: distance - 2 r_i. Numerical qualification by the V1/V2 tests against an exact rational oracle '
          '(|d - d_ref| <= 1e-12 max segment length on the tested cases); not a certified floating-point bound')
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
    """Raises AuditInputError for any input the audit cannot conclude on; returns (L, centerlines).
    Pores do not exist yet at this stage: they are audited afterwards by audit_pores()."""
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
    if any(isinstance(i, bool) or not isinstance(i, (int, np.integer)) for i in ids):
        raise AuditInputError(f'fiber identifiers must be integers: {ids!r}')
    if len(set(int(i) for i in ids)) != len(ids):
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
        box = getattr(f, 'bbox', None)
        if box is None or not (np.array_equal(np.asarray(box[0], dtype=float), line.min(axis=0) - f.radius)
                               and np.array_equal(np.asarray(box[1], dtype=float), line.max(axis=0) + f.radius)):
            raise AuditInputError(f'fiber {f.id}: cached bounding box differs from the centerline +- radius '
                                  f'(stale geometry); the voxelizer uses it to choose images and raster bounds')
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
                if not math.isfinite(d):                       # self images included: they always exist
                    what = f'fiber {fi.id} and its own images' if i == j else f'fibers {fi.id} and {fj.id}'
                    raise AuditInputError(f'non-finite distance ({d!r}) between {what}')
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


PORE_METHOD = ('spherical pores grouped by identifier (periodic images must share centre modulo the cell and '
               'radius); pore-pore: centre distance under the minimum-image convention (exact for points, '
               'pore diameters below the smallest cell side required) minus the sum of radii; pore-fiber: exact '
               'point-to-segment distance from the pore centre to the 100-point sampled centerline, over every '
               'periodic shift whose fiber box can reach the pore, minus pore radius and fiber radius (capsule '
               'test: a sufficient condition for the flat-ended tube rasterised by the voxelizer)')
PORE_TOLERANCE = 1e-12          # relative to the largest cell side: round-off allowed on contact


def _point_polyline_distance(p, line):
    """Exact Euclidean distance from point p to the polyline `line` (n, 3)."""
    a = line[:-1]
    ab = line[1:] - a
    ab2 = np.einsum('ij,ij->i', ab, ab)
    t = np.clip(np.einsum('ij,ij->i', p - a, ab) / np.where(ab2 > 0, ab2, 1.0), 0.0, 1.0)
    q = a + t[:, None] * ab
    return float(np.sqrt(np.min(np.einsum('ij,ij->i', q - p, q - p))))


def audit_pores(fibers, voids, box_dims):
    """!
    @brief Blocking audit of the spherical pores, run after porosity generation.
    @details Every pore must be disjoint from every other pore and from every fiber, periodic images
    included (contact within a round-off tolerance is allowed).
    @param fibers Final fibers (roots), already audited by audit().
    @param voids Pores as exported (periodic images included, sharing the identifier of their pore).
    @param box_dims Cell dimensions.
    @return dict with status, cause, number of pores, minimum gaps and the responsible pair.
    """
    res = {'status': 'undetermined', 'cause': None, 'method': PORE_METHOD, 'n_pores': None,
           'min_pore_pore_gap': None, 'min_pore_fiber_gap': None, 'responsible': None, 'tolerance': None}
    try:
        L = np.array([float(x) for x in box_dims])
        tol = PORE_TOLERANCE * float(L.max())
        res['tolerance'] = tol
        pores = {}
        for v in voids:
            c = np.asarray(v.center, dtype=float)
            r = float(v.radius)
            if c.shape != (3,) or not np.all(np.isfinite(c)) or not math.isfinite(r) or r <= 0:
                raise AuditInputError(f'pore {v.id}: centre {c.tolist()} / radius {r!r} not finite and positive')
            wrapped = np.mod(c, L)
            if v.id in pores:
                c0, r0 = pores[v.id]
                delta = (wrapped - c0 + L / 2) % L - L / 2
                if r != r0 or np.max(np.abs(delta)) > tol:
                    raise AuditInputError(f'pore {v.id}: images are not translates of one sphere by whole periods')
            else:
                pores[v.id] = (wrapped, r)
        ids = list(pores)
        res['n_pores'] = len(ids)
        if not ids:
            res.update(status='passed')
            return res
        C = np.array([pores[i][0] for i in ids])
        R = np.array([pores[i][1] for i in ids])
        if np.any(2 * R >= L.min()):
            raise AuditInputError('a pore diameter reaches the smallest cell side (minimum image not valid)')

        best_pp, pair_pp = math.inf, None
        if len(ids) > 1:
            D = C[:, None, :] - C[None, :, :]
            D -= L * np.round(D / L)
            gap = np.sqrt(np.einsum('ijk,ijk->ij', D, D)) - (R[:, None] + R[None, :])
            np.fill_diagonal(gap, np.inf)
            i, j = np.unravel_index(np.argmin(gap), gap.shape)
            best_pp, pair_pp = float(gap[i, j]), [ids[i], ids[j]]

        best_pf, pair_pf = math.inf, None
        for f in fibers:
            line = np.asarray(f.centerline, dtype=float)
            rf = float(f.radius)
            lo, hi = line.min(axis=0), line.max(axis=0)
            reach = R[:, None] + rf
            k_min = np.ceil((C - reach - hi) / L).astype(int)
            k_max = np.floor((C + reach - lo) / L).astype(int)
            for n in np.nonzero(np.all(k_max >= k_min, axis=1))[0]:
                for kx in range(k_min[n, 0], k_max[n, 0] + 1):
                    for ky in range(k_min[n, 1], k_max[n, 1] + 1):
                        for kz in range(k_min[n, 2], k_max[n, 2] + 1):
                            shift = np.array([kx, ky, kz]) * L
                            g = _point_polyline_distance(C[n] - shift, line) - R[n] - rf
                            if g < best_pf:
                                best_pf = g
                                pair_pf = {'pore': ids[n], 'fiber': f.id, 'fiber_shift_periods': [kx, ky, kz]}

        res.update(min_pore_pore_gap=None if pair_pp is None else best_pp,
                   min_pore_fiber_gap=None if pair_pf is None else best_pf)
        if pair_pf is None:
            res['min_pore_fiber_gap_cause'] = 'no fiber box (periodic images included) within reach of any pore'
        if best_pp < -tol:
            res.update(status='failed', cause=f'pores {pair_pp} overlap (gap {best_pp:.6e})',
                       responsible={'pores': pair_pp})
        elif best_pf < -tol:
            res.update(status='failed', cause=f"pore {pair_pf['pore']} overlaps fiber {pair_pf['fiber']} "
                                              f"(gap {best_pf:.6e})", responsible=pair_pf)
        else:
            res.update(status='passed', responsible={'pores': pair_pp, 'pore_fiber': pair_pf})
    except AuditInputError as e:
        res.update(status='undetermined', cause=str(e))
    except Exception as e:                                    # audit error: never a success
        res.update(status='undetermined', cause=f'pore audit error: {type(e).__name__}: {e}')
    return res


def attach_pore_audit(validity, pore_result):
    """!
    @brief Adds the pore audit to a validity record; a refused pore audit turns the record into a refusal.
    @param validity Record built by build_record() for the fibers (status `passed`).
    @param pore_result Result of audit_pores().
    @return The updated record.
    """
    validity['pores'] = pore_result
    validity['excluded_checks'] = [c for c in validity.get('excluded_checks', []) if c != 'pores']
    if pore_result['status'] != 'passed':
        validity.update(status=pore_result['status'], stage='pores', cause=pore_result['cause'], exit_code=1)
    return validity


def refusal(stage, cause):
    """Audit result of an attempt refused before any geometry exists."""
    return {'status': 'undetermined', 'cause': cause, 'stage': stage, 'min_clearance': None,
            'min_clearance_cause': 'geometry not produced', 'responsible_pair': None}


def build_record(config, fibers, result, exit_code):
    """Validity record of an attempt (strict JSON once passed through json_safe).

    The hashes are required for a success: if they cannot be built, a `passed` result becomes
    `undetermined` (stage `hashes`, exit code 1). `fibers` is None when no geometry was produced.
    """
    prov = fps_provenance()
    status, cause, stage = result['status'], result['cause'], result['stage']
    if fibers is None:
        fib_sha = geo_sha = geo_obj = None
        hash_cause = 'geometry not produced'
    else:
        try:
            records = microstructure_records(fibers)
            fib_sha = fibers_sha256(records)
            fallback = [f.id for f in fibers if len(np.asarray(f.control_points)) < 4]
            geo_sha, geo_obj = audited_geometry_sha256(fib_sha, config.box_dims, fallback, prov['commit'])
            hash_cause = None
        except Exception as e:
            fib_sha = geo_sha = geo_obj = None
            hash_cause = f'hash not computed: {type(e).__name__}: {e}'
            if status == 'passed':
                status, stage, exit_code = 'undetermined', 'hashes', 1
                cause = f'audit passed but the geometry hashes could not be built ({hash_cause}): no export'
    requested = getattr(config, 'min_clearance', None)
    return {
        'schema_version': SCHEMA_VERSION,
        'attempt': {'id': uuid.uuid4().hex, 'timestamp': datetime.datetime.now().isoformat(timespec='seconds')},
        'status': status, 'cause': cause, 'stage': stage,
        'exit_code': exit_code,
        'fps': prov,
        'seed': getattr(config, 'seed', None),
        'config_effective': json_safe(vars(config)),
        'n_fibers': None if fibers is None else len(fibers),
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
