# SPDX-FileCopyrightText: 2026 Devine Ngouloubi <exauce-devine.ngouloubi@unicaen.fr>
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Voxeliser benchmark: rasterisation time versus grid resolution N, on a stored RVE.

The RVE is rebuilt from its parametric record (no placement), then Voxelizer.create_grid is timed
alone for each resolution, after a warm-up call that loads the cached Numba kernels.

    python benchmarks/run_voxelizer.py benchmarks/results/rves/vf30_s1_parametric.json \
        --res 32 64 128 256 512 --out benchmarks/results
"""
import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.machine import machine_record  # noqa: E402
from cli import config_from_args  # noqa: E402
from export.voxelizer import Voxelizer  # noqa: E402
from regenerate import load_record, namespace_from_record, fibers_from_record, voids_from_record  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('record', type=Path)
    ap.add_argument('--res', type=int, nargs='+', default=[32, 64, 128, 256, 512])
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--out', type=Path, default=ROOT / 'benchmarks' / 'results')
    a = ap.parse_args()

    record = load_record(a.record)
    config = config_from_args(namespace_from_record(record, {}))
    fibers = fibers_from_record(record, config)
    voids = voids_from_record(record)
    vox = Voxelizer(config)
    vox.create_grid(fibers, voids, res=16)                     # warm-up: Numba kernels from the disk cache

    rows = []
    for n in a.res:
        times = []
        for _ in range(a.repeats):
            t0 = time.perf_counter()
            grid = vox.create_grid(fibers, voids, res=n)
            times.append(time.perf_counter() - t0)
        fiber_fraction = float(np.mean(grid == 128))
        rows.append({'resolution': n, 'voxels': n ** 3, 'median_s': statistics.median(times),
                     'min_s': min(times), 'max_s': max(times), 'repeats': a.repeats,
                     'fiber_voxel_fraction': fiber_fraction})
        print(f"N = {n:4d}: median {statistics.median(times):.3f} s (fiber voxels {fiber_fraction:.3f})",
              flush=True)

    a.out.mkdir(parents=True, exist_ok=True)
    with open(a.out / 'voxelizer.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (a.out / 'voxelizer_machine.json').write_text(json.dumps(machine_record(
        record=str(a.record), n_fibers=len(fibers), reached_vf=record['computed_statistics']['volume_fraction_real'],
        command=sys.argv), indent=2))
    print(f"Results: {a.out / 'voxelizer.csv'}")


if __name__ == '__main__':
    main()
