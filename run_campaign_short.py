"""
@file run_campaign_short.py
@brief Generation campaign for SHORT-fiber RVEs (FFT res <= 128).

@details
Two chained sub-campaigns sharing the same base geometry:

    Cubic box            : Lx = Ly = Lz = 1.0
    Fiber radius         : r  = 0.02   (-> ~2.56 voxels at res 128)
    Section              : circular
    Control points       : min = 19, max = 20   (short fibers)
    FFT resolution       : 128 (voxel = 0.0078, fiber diameter ~5 voxels)

  The volume fractions are DILUTE (already as a fraction):
     0.09% -> 0.0009, 0.179% -> 0.00179, ..., 0.447% -> 0.00447.
  The 1.0^3 box is chosen as a resolution/fidelity trade-off: at res 128
  we keep ~2.56 voxels per radius while obtaining real Vf close to the
  targets (0.10% -> 0.51%) with 2 to 10 short fibers per RVE.

  -- Part A: volume-fraction sweep --
     Vf = {0.0009, 0.00179, 0.00269, 0.00358, 0.00447}, fixed step = 0.02.
     Fiber length ~ 19 * 0.02 = 0.38 < 1.0 (box) -> discontinuous fibers.

  -- Part B: segment-length (step) sweep --
     Vf fixed at 0.00447 (last value, 0.447%), pts 19/20,
     step in {0.015, 0.020, 0.025}
     -> fiber lengths ~ {0.285, 0.380, 0.475}, all < 1.0.

Output tree:

    campaign_short/
    |-- A_vf_sweep/
    |   |-- vf_0090/   RVE_vf0090_*
    |   |-- vf_0179/
    |   |-- ...
    |-- B_step_sweep/
    |   |-- step_0015/ RVE_step0015_*
    |   |-- step_0020/
    |   |-- step_0025/

Usage:
    python run_campaign_short.py            # run everything
    python run_campaign_short.py --dry_run  # print the commands without executing
    python run_campaign_short.py --part A   # part A only
    python run_campaign_short.py --part B   # part B only

@author  Devine Ngouloubi -- LMNO, University of Caen
"""

import argparse
import subprocess
import sys
import time
import logging
from pathlib import Path

# -- Base geometry common to both sub-campaigns ------------------------------
BASE_PARAMS = {
    "--dims":      "1.0 1.0 1.0",
    "--radius":    "0.02",
    "--section":   "circular",
    "--clearance": "0.005",
    "--min_pts":   "19",
    "--max_pts":   "20",
    "--res_fft":   "128",
}
BASE_FLAGS = ["--no_mesh"]  # no GMSH mesh; keep the spatial stats

# Statistical realisations: one generation per seed and per point.
SEEDS = list(range(42, 52))  # 42..51 inclusive (10 realisations)

# Part A: Vf sweep (fixed step). Values already as a fraction (0.09% -> 0.0009).
A_STEP = "0.02"
A_VF_VALUES = ["0.0009", "0.00179", "0.00269", "0.00358", "0.00447"]

# Part B: step sweep (Vf fixed = last value of A = 0.447%)
B_VF = "0.00447"
B_STEP_VALUES = ["0.015", "0.020", "0.025"]

CAMPAIGN_ROOT = Path("campaign_short")
MAIN_SCRIPT = Path(__file__).parent / "main.py"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("CampaignShort")


def build_command(extra: dict, prefix: str) -> list[str]:
    """!
    @brief Builds the main.py command from the base params + overrides.
    @param extra  Dict of overrides (e.g. {'--vf': '0.09', '--step': '0.02'}).
    @param prefix Full output prefix (path included) of the output files.
    @return List of arguments.
    """
    params = dict(BASE_PARAMS)
    params.update(extra)
    cmd = [sys.executable, str(MAIN_SCRIPT)]
    for flag, value in params.items():
        cmd += [flag] + str(value).split()
    cmd += BASE_FLAGS
    cmd += ["--output", prefix]
    return cmd


def run_single(label: str, run_dir: Path, prefix: str, extra: dict, dry_run: bool) -> bool:
    """!
    @brief Runs (or simulates) a single generation.
    @return True on success or dry_run, False otherwise.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_command(extra, prefix)

    if dry_run:
        logger.info(f"DRY-RUN {label}  ->  {' '.join(cmd)}")
        return True

    logger.info(f"START   {label}  ->  {run_dir}")
    t0 = time.time()
    try:
        subprocess.run(cmd, check=True, text=True)
        logger.info(f"OK      {label}  ({time.time() - t0:.1f}s)")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"FAIL    {label}  (code={e.returncode}, {time.time() - t0:.1f}s)")
        return False
    except FileNotFoundError:
        logger.critical(f"main.py not found: {MAIN_SCRIPT}")
        sys.exit(1)


def jobs_part_a():
    """! @brief Builds the list of Part A jobs (Vf sweep x seeds)."""
    jobs = []
    for vf in A_VF_VALUES:
        tag = f"vf_{int(round(float(vf) * 100000)):04d}"  # 0.0009 -> vf_0090 (=0.090%)
        for seed in SEEDS:
            run_dir = CAMPAIGN_ROOT / "A_vf_sweep" / tag / f"seed_{seed:02d}"
            prefix = str(run_dir / f"RVE_{tag}_s{seed:02d}")
            jobs.append((f"[A | Vf={vf} | step={A_STEP} | seed={seed:02d}]",
                         run_dir, prefix,
                         {"--vf": vf, "--step": A_STEP, "--seed": str(seed)}))
    return jobs


def jobs_part_b():
    """! @brief Builds the list of Part B jobs (step sweep x seeds)."""
    jobs = []
    for step in B_STEP_VALUES:
        tag = f"step_{int(round(float(step) * 1000)):04d}"  # 0.015 -> step_0015
        for seed in SEEDS:
            run_dir = CAMPAIGN_ROOT / "B_step_sweep" / tag / f"seed_{seed:02d}"
            prefix = str(run_dir / f"RVE_{tag}_s{seed:02d}")
            jobs.append((f"[B | Vf={B_VF} | step={step} | seed={seed:02d}]",
                         run_dir, prefix,
                         {"--vf": B_VF, "--step": step, "--seed": str(seed)}))
    return jobs


def parse_campaign_args():
    """! @brief Parses the campaign arguments."""
    p = argparse.ArgumentParser(description="Short-fiber RVE campaign (res <= 128)")
    p.add_argument("--part", choices=["A", "B", "all"], default="all",
                   help="Sub-campaign to run")
    p.add_argument("--dry_run", action="store_true",
                   help="Print the commands without executing them")
    return p.parse_args()


def main():
    """! @brief Entry point of the campaign."""
    args = parse_campaign_args()

    jobs = []
    if args.part in ("A", "all"):
        jobs += jobs_part_a()
    if args.part in ("B", "all"):
        jobs += jobs_part_b()

    logger.info("=" * 64)
    logger.info("  SHORT-FIBER RVE CAMPAIGN -- LMNO / University of Caen")
    logger.info(f"  Part(s)     : {args.part}")
    logger.info(f"  Total runs  : {len(jobs)}")
    logger.info(f"  Root        : {CAMPAIGN_ROOT.resolve()}")
    if args.dry_run:
        logger.info("  MODE        : DRY-RUN")
    logger.info("=" * 64)

    t_start = time.time()
    ok, fail, fail_list = 0, 0, []
    for label, run_dir, prefix, extra in jobs:
        if run_single(label, run_dir, prefix, extra, args.dry_run):
            ok += 1
        else:
            fail += 1
            fail_list.append(label)

    logger.info("=" * 64)
    logger.info(f"  SUMMARY: {ok} success / {fail} failure(s)  "
                f"-- duration {time.time() - t_start:.1f}s")
    for label in fail_list:
        logger.warning(f"    FAILED: {label}")
    logger.info("=" * 64)

    if not args.dry_run:
        aggregate(args.part)

    sys.exit(0 if fail == 0 else 1)


def aggregate(part: str):
    """!
    @brief Aggregates the realisations (seeds) per point: mean and standard
    deviation of the real Vf (voxels) and of the fiber count. Writes
    campaign_short/summary_stats.csv.
    """
    import csv, glob, json
    import numpy as np

    point_dirs = []
    if part in ("A", "all"):
        point_dirs += sorted(glob.glob(str(CAMPAIGN_ROOT / "A_vf_sweep" / "*")))
    if part in ("B", "all"):
        point_dirs += sorted(glob.glob(str(CAMPAIGN_ROOT / "B_step_sweep" / "*")))

    rows = []
    for pdir in point_dirs:
        vfs, nfs, lens = [], [], []
        for j in sorted(glob.glob(f"{pdir}/seed_*/RVE_*_parametric.json")):
            d = json.load(open(j))
            npy = j.replace("_parametric.json", "_voxelmap.npy")
            g = np.load(npy)
            vfs.append(float((g > 0).mean()))
            md = d["microstructure_data"]
            nfs.append(len(md))
            for f in md:
                cp = np.array(f["control_points"])
                lens.append(float(np.linalg.norm(np.diff(cp, axis=0), axis=1).sum()))
        if not vfs:
            continue
        vfs, nfs = np.array(vfs), np.array(nfs)
        rows.append({
            "point": Path(pdir).name,
            "n_seeds": len(vfs),
            "vf_real_mean_pct": round(vfs.mean() * 100, 4),
            "vf_real_std_pct": round(vfs.std(ddof=1) * 100, 4) if len(vfs) > 1 else 0.0,
            "nfibers_mean": round(nfs.mean(), 2),
            "nfibers_std": round(nfs.std(ddof=1), 2) if len(nfs) > 1 else 0.0,
            "len_fiber_mean": round(float(np.mean(lens)), 4) if lens else 0.0,
        })

    out = CAMPAIGN_ROOT / "summary_stats.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    logger.info("  STATISTICS PER POINT (mean +/- standard deviation over seeds)")
    logger.info(f"  {'point':<12}{'n':>3}{'Vf_real %':>14}{'fibers':>14}{'L_fib':>9}")
    for r in rows:
        logger.info(f"  {r['point']:<12}{r['n_seeds']:>3}"
                    f"{r['vf_real_mean_pct']:>8.3f}+/-{r['vf_real_std_pct']:<5.3f}"
                    f"{r['nfibers_mean']:>7.2f}+/-{r['nfibers_std']:<5.2f}"
                    f"{r['len_fiber_mean']:>9.3f}")
    logger.info(f"  -> {out}")
    logger.info("=" * 64)


if __name__ == "__main__":
    main()
