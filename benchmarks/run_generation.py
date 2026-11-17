# SPDX-FileCopyrightText: 2026 Devine Ngouloubi <exauce-devine.ngouloubi@unicaen.fr>
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Generation benchmark: wall-clock time of each phase versus the target fiber volume fraction.

Runs main.py (one single-threaded process per RVE) over a grid of target volume fractions and
seeds, and collects from each parametric record the phase timings, the reached volume fraction,
the number of fibers and the aspect ratio. Results: <out>/generation.csv and <out>/machine.json.

    python benchmarks/run_generation.py --out benchmarks/results --jobs 3

The RVEs are kept in <out>/rves/ (parametric records), so every point of the figure can be
regenerated or inspected with regenerate.py.
"""
import argparse
import csv
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.machine import machine_record  # noqa: E402

# Fixed geometry of the benchmark (code units: the cell is the unit cube)
BASE = ['--dims', '1', '1', '1', '--radius', '0.02', '--no_mesh', '--res_fft', '64', '--no_spatial_stats']
VFS = [0.10, 0.15, 0.20, 0.25, 0.30]
SEEDS = [1, 2, 3]


def run_one(out, vf, seed, timeout):
    prefix = out / 'rves' / f'vf{int(round(vf * 100)):02d}_s{seed}'
    prefix.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(ROOT / 'main.py')] + BASE + ['--vf', str(vf), '--seed', str(seed),
                                                          '--output', str(prefix)]
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        code, status = proc.returncode, 'ok' if proc.returncode == 0 else 'refused'
        Path(f'{prefix}.log').write_text(proc.stdout + proc.stderr)
    except subprocess.TimeoutExpired:
        code, status = None, 'timeout'
    wall = time.perf_counter() - t0
    row = {'target_vf': vf, 'seed': seed, 'status': status, 'exit_code': code, 'wall_s': wall}
    rec_path = Path(f'{prefix}_parametric.json')
    if status == 'ok' and rec_path.exists():
        rec = json.loads(rec_path.read_text())
        t = rec.get('timings_s', {})
        fibers = rec['microstructure_data']
        lengths = [float(np.sum(np.linalg.norm(np.diff(np.asarray(f['control_points']), axis=0), axis=1)))
                   for f in fibers]
        diam = 2 * float(np.median([f['radius'] for f in fibers]))
        row.update(reached_vf=rec['computed_statistics']['volume_fraction_real'], n_fibers=len(fibers),
                   aspect_ratio=float(np.mean(lengths)) / diam,
                   placement_s=t.get('placement'), densification_s=t.get('densification'),
                   fiber_audit_s=t.get('fiber_audit'),
                   generation_s=sum(t.get(k, 0.0) for k in ('placement', 'densification', 'fiber_audit')))
    print(f"Vf {vf:.2f} seed {seed}: {status} in {wall:.1f} s", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, default=ROOT / 'benchmarks' / 'results')
    ap.add_argument('--jobs', type=int, default=1, help='RVEs generated concurrently (each single-threaded)')
    ap.add_argument('--timeout', type=float, default=7200.0, help='seconds per RVE')
    ap.add_argument('--vf', type=float, nargs='+', default=VFS)
    ap.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'machine.json').write_text(json.dumps(machine_record(jobs=a.jobs, command=sys.argv), indent=2))

    # Heaviest runs first so that the concurrent pool finishes as early as possible
    tasks = sorted(((vf, s) for vf in a.vf for s in a.seeds), key=lambda t: -t[0])
    with ThreadPoolExecutor(max_workers=a.jobs) as pool:
        rows = list(pool.map(lambda t: run_one(a.out, t[0], t[1], a.timeout), tasks))
    rows.sort(key=lambda r: (r['target_vf'], r['seed']))
    keys = ['target_vf', 'seed', 'status', 'exit_code', 'reached_vf', 'n_fibers', 'aspect_ratio',
            'placement_s', 'densification_s', 'fiber_audit_s', 'generation_s', 'wall_s']
    with open(a.out / 'generation.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})
    print(f"Results: {a.out / 'generation.csv'}")


if __name__ == '__main__':
    main()
