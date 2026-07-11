"""
@file run_campaign.py
@brief Automation of the multi-seed / multi-resolution RVE generation campaign.

@details
Generates a matrix of configurations (N_seeds x N_resolutions) by calling
main.py through subprocess. Each run is isolated in its own output directory,
structured as follows:

    campaign/
    +-- cfg_42/
    |   +-- res_064/
    |   |   +-- RVE_cfg42_r064.*
    |   +-- res_128/
    |   +-- res_256/
    |   +-- res_512/
    |   +-- res_700/
    +-- cfg_43/
    |   +-- ...
    +-- ...

Fixed global parameters (identical for all configurations):
  - Cubic box             : Lx = Ly = Lz = 0.612
  - Volume fraction       : Vf = 0.18
  - Fiber radius          : r  = 0.02
  - Section               : circular
  - Inter-fiber clearance : 0.01
  - RSDA                  : enabled
  - Orientation           : uniaxial along (1, 0, 0), strength = 0.22
  - GMSH / Nastran export : disabled

Campaign variables:
  - Seeds  : 42 -> 51 inclusive (10 distinct geometric configurations)
  - FFT resolutions : 64, 128, 256, 512, 700

Usage:
    python run_campaign.py               # run everything
    python run_campaign.py --dry_run     # print the commands without executing them
    python run_campaign.py --seeds 42 43 --resolutions 64 128  # sub-campaign

@author  Devine Ngouloubi - LMNO, University of Caen
"""

import argparse
import subprocess
import sys
import os
import time
import logging
from itertools import product
from pathlib import Path

# -- Campaign configuration ----------------------------------------------------

# Geometric and physical parameters - INVARIANT
GLOBAL_PARAMS = {
    "--dims":       "0.612 0.612 0.612",
    "--vf":         "0.18",
    "--radius":     "0.02",
    "--section":    "circular",
    "--clearance":  "0.01",
    "--bias_type":  "uniaxial",
    "--bias_vec1":  "1.0 0.0 0.0",
    "--strength":   "0.22",
}

# Boolean flags (no associated value)
GLOBAL_FLAGS = [
    "--rsda",       # RSDA enabled
    "--no_mesh",    # No GMSH mesh
]
# --nastran is omitted (disabled by default in cli.py)

# Campaign variables
DEFAULT_SEEDS       = list(range(42, 52))          # 42..51 inclusive
DEFAULT_RESOLUTIONS = [64, 128, 256, 512, 700]

# Root directory of the campaign
CAMPAIGN_ROOT = Path("campaign")

# Path to main.py (relative to this script's location)
MAIN_SCRIPT = Path(__file__).parent / "main.py"

# -- Utilities -----------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Campaign")


def build_command(seed: int, resolution: int, output_prefix: str) -> list[str]:
    """
    Builds the argument list for a call to main.py.

    @param seed          Random seed of the configuration.
    @param resolution    FFT grid resolution (cube of side `resolution`).
    @param output_prefix Full output prefix (path included) of the output files.
    @return List of strings constituting the full command.
    """
    cmd = [sys.executable, str(MAIN_SCRIPT)]

    # Invariant key/value parameters
    for flag, value in GLOBAL_PARAMS.items():
        cmd += [flag] + value.split()

    # Boolean flags
    cmd += GLOBAL_FLAGS

    # Variable parameters
    cmd += ["--seed",    str(seed)]
    cmd += ["--res_fft", str(resolution)]
    cmd += ["--output",  output_prefix]

    return cmd


def run_single(seed: int, resolution: int, dry_run: bool = False) -> bool:
    """
    Runs (or simulates) the generation of a single configuration.

    @param seed       Seed of the configuration.
    @param resolution FFT resolution.
    @param dry_run    If True, print the command without executing it.
    @return True on success (or dry_run), False on a subprocess error.
    """
    # Build the output directory tree
    run_dir = CAMPAIGN_ROOT / f"cfg_{seed:02d}" / f"res_{resolution:04d}"
    run_dir.mkdir(parents=True, exist_ok=True)

    prefix = str(run_dir / f"RVE_cfg{seed:02d}_r{resolution:04d}")
    cmd    = build_command(seed, resolution, prefix)

    label = f"[seed={seed:02d} | res={resolution:4d}^3]"

    if dry_run:
        logger.info(f"DRY-RUN {label}  ->  {' '.join(cmd)}")
        return True

    logger.info(f"START   {label}  ->  {run_dir}")
    t0 = time.time()

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,   # let stdout/stderr display live
            text=True,
        )
        elapsed = time.time() - t0
        logger.info(f"OK      {label}  ({elapsed:.1f}s)")
        return True

    except subprocess.CalledProcessError as e:
        elapsed = time.time() - t0
        logger.error(f"FAIL    {label}  (code={e.returncode}, {elapsed:.1f}s)")
        return False

    except FileNotFoundError:
        logger.critical(f"main.py not found: {MAIN_SCRIPT}")
        sys.exit(1)


# -- Entry point ---------------------------------------------------------------

def parse_campaign_args():
    """!
    @brief Parses the command-line arguments of the campaign runner.
    @return The parsed argparse Namespace.
    """
    p = argparse.ArgumentParser(
        description="Multi-seed / multi-resolution RVE generation campaign",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
        help="List of seeds to use (e.g. 42 43 44)",
    )
    p.add_argument(
        "--resolutions", type=int, nargs="+", default=DEFAULT_RESOLUTIONS,
        help="List of FFT resolutions to generate (e.g. 64 128 256)",
    )
    p.add_argument(
        "--dry_run", action="store_true",
        help="Print the commands without executing them",
    )
    p.add_argument(
        "--sequential", action="store_true", default=True,
        help="Sequential execution (default). Reserved for a future parallel extension.",
    )
    return p.parse_args()


def main():
    """!
    @brief Main entry point: builds and runs the full campaign matrix.
    @return None
    """
    args = parse_campaign_args()

    seeds       = sorted(set(args.seeds))
    resolutions = sorted(set(args.resolutions))
    n_total     = len(seeds) * len(resolutions)

    logger.info("=" * 60)
    logger.info("  RVE CAMPAIGN - LMNO / University of Caen")
    logger.info(f"  Seeds       : {seeds}")
    logger.info(f"  Resolutions : {resolutions}")
    logger.info(f"  Total runs  : {n_total}")
    logger.info(f"  Root        : {CAMPAIGN_ROOT.resolve()}")
    if args.dry_run:
        logger.info("  MODE        : DRY-RUN (no file produced)")
    logger.info("=" * 60)

    t_campaign_start = time.time()
    results = {"ok": 0, "fail": 0, "fail_list": []}

    # Main loop: iterate first over the seeds, then over the resolutions.
    # This ensures that the 5 resolutions of the same geometric configuration
    # are generated consecutively - useful if one wants to inspect a config
    # before moving to the next one.
    for seed, res in product(seeds, resolutions):
        success = run_single(seed, res, dry_run=args.dry_run)
        if success:
            results["ok"] += 1
        else:
            results["fail"] += 1
            results["fail_list"].append((seed, res))

    elapsed_total = time.time() - t_campaign_start

    logger.info("=" * 60)
    logger.info(f"  SUMMARY: {results['ok']} success / {results['fail']} failure(s)")
    logger.info(f"  Total duration: {elapsed_total:.1f}s  "
                f"({elapsed_total / 60:.1f} min)")
    if results["fail_list"]:
        logger.warning("  Failed configurations:")
        for s, r in results["fail_list"]:
            logger.warning(f"    seed={s:02d}, res={r}")
    logger.info("=" * 60)

    sys.exit(0 if results["fail"] == 0 else 1)


if __name__ == "__main__":
    main()
