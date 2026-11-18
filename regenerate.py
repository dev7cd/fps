# SPDX-FileCopyrightText: 2026 Devine Ngouloubi <exauce-devine.ngouloubi@unicaen.fr>
# SPDX-License-Identifier: AGPL-3.0-or-later

## @file regenerate.py
#  @brief Regenerates every export of an RVE from its parametric record, without re-running placement.
#  @details The record `<prefix>_parametric.json` written by main.py stores the complete CLI argument
#  set, the control points and radius of every fiber, and the pores. This tool rebuilds the
#  configuration and the geometry from it, checks that the record is the audited one (SHA-256 of
#  `microstructure_data`) and that the rebuilt fibers have exactly its control points and radii,
#  re-runs the blocking geometric audit, and writes
#  the statistics, CSV, voxel field, PGM slices, CAD/mesh and optional Nastran/Abaqus files.
#
#  Only export options may differ from the original run, for example a finer voxel grid:
#
#      python regenerate.py RVE_parametric.json --res_fft 512 --no_mesh --output RVE_512
#
#  Arguments that shape the geometry (--vf, --radius, --seed, ...) are fixed by the record.

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np

from cli import build_parser, config_from_args, EXPORT_ARGUMENTS
from core.fiber import Fiber
from core.void import Void
from validation import geometric_audit

logger = logging.getLogger("RVE_Regenerate")

## Relative tolerance on the fiber volume recomputed from the spline. The volume is a derived
#  quantity: NumPy reductions round differently from one CPU to another (for instance with the
#  NEON instructions of arm64), so it can differ from the stored value in its last bits.
REAL_VOLUME_RTOL = 1e-9


class RecordError(ValueError):
    """Parametric record that cannot be regenerated."""


def load_record(path):
    """!
    @brief Reads a parametric record.
    @param path Path of `<prefix>_parametric.json`.
    @return dict
    """
    with open(path) as f:
        record = json.load(f)
    if record.get("cli_arguments") is None:
        raise RecordError(f"{path} has no 'cli_arguments' (record written before FPS 7.0): "
                          "the configuration cannot be rebuilt from it.")
    if "microstructure_data" not in record:
        raise RecordError(f"{path} has no 'microstructure_data'.")
    return record


def namespace_from_record(record, overrides):
    """!
    @brief Rebuilds the CLI namespace of the original run, with export overrides applied.
    @param record Parametric record.
    @param overrides dict of export arguments given on the command line.
    @return argparse.Namespace
    """
    args = build_parser().parse_args([])
    for key, value in record["cli_arguments"].items():
        setattr(args, key, value)
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def fibers_from_record(record, config):
    """!
    @brief Rebuilds the root fibers (centerline, frames, bounding box) from their control points.
    @param record Parametric record.
    @param config FiberPackingConfig of the original run.
    @return list of Fiber
    """
    return [Fiber(int(entry["id"]), np.array(entry["control_points"], dtype=float), float(entry["radius"]),
                  vars(config))
            for entry in record["microstructure_data"]]


def voids_from_record(record):
    """!
    @brief Rebuilds the pores (periodic images included) stored in the record.
    @param record Parametric record.
    @return list of Void
    """
    return [Void(id=int(v["id"]), center=np.array(v["center"], dtype=float), radius=float(v["radius"]))
            for v in record.get("voids", [])]


def check_same_geometry(record, fibers):
    """!
    @brief Checks that the rebuilt fibers are those audited in the original run.
    @details The canonical SHA-256 of the stored `microstructure_data` must equal the one of the
    validity block, which detects any change to the record. The rebuilt fibers must then carry
    exactly the stored ids, control points and radii. Their volume is recomputed from the spline
    and is compared with the stored value up to REAL_VOLUME_RTOL, since its last bits depend on
    the platform.
    @param record Parametric record.
    @param fibers Rebuilt fibers.
    @return str The hash of the audited geometry.
    """
    stored_data = record["microstructure_data"]
    sha = geometric_audit.fibers_sha256(stored_data)
    stored = (record.get("validity") or {}).get("fibers_sha256")
    if stored is not None and stored != sha:
        raise RecordError(f"microstructure_data differs from the audited geometry (sha256 {sha} != {stored}).")

    rebuilt = geometric_audit.microstructure_records(fibers)
    if len(rebuilt) != len(stored_data):
        raise RecordError(f"{len(rebuilt)} fibers rebuilt for {len(stored_data)} in the record.")
    for new, old in zip(rebuilt, stored_data):
        same = (int(new["id"]) == int(old["id"]) and float(new["radius"]) == float(old["radius"])
                and np.array_equal(np.asarray(new["control_points"], dtype=float),
                                   np.asarray(old["control_points"], dtype=float)))
        if not same:
            raise RecordError(f"fiber {old['id']}: the rebuilt fiber differs from the record.")
        volume = (old.get("geometry") or {}).get("real_volume")
        if volume is not None and not math.isclose(new["geometry"]["real_volume"], volume,
                                                   rel_tol=REAL_VOLUME_RTOL, abs_tol=0.0):
            raise RecordError(f"fiber {old['id']}: rebuilt volume {new['geometry']['real_volume']!r} "
                              f"differs from the stored {volume!r}.")
    return sha


def parse_overrides(argv):
    """!
    @brief Parses the record path and the export options given on the command line.
    @param argv Argument list.
    @return (record path, dict of explicitly given export arguments)
    """
    parser = argparse.ArgumentParser(
        description="Regenerate the exports of an FPS RVE from its parametric record "
                    "(no placement is run). Export options of main.py may be given; "
                    "geometry options are fixed by the record.")
    parser.add_argument("record", help="<prefix>_parametric.json written by main.py")
    ns, rest = parser.parse_known_args(argv)

    # Same option names and types as main.py; a sentinel default tells explicitly given options apart
    # (a non-string default is never converted by argparse).
    gen_parser = build_parser()
    unset = object()
    gen_parser.set_defaults(**{dest: unset for dest in vars(gen_parser.parse_args([]))})
    given = {k: v for k, v in vars(gen_parser.parse_args(rest)).items() if v is not unset}
    fixed = sorted(set(given) - set(EXPORT_ARGUMENTS))
    if fixed:
        parser.error(f"{', '.join('--' + k for k in fixed)}: fixed by the record (they shape the geometry); "
                     f"only export options can be changed: {', '.join('--' + k for k in EXPORT_ARGUMENTS)}")
    return ns.record, given


def main(argv=None):
    """!
    @brief Entry point: rebuild the RVE from its record and write all exports.
    @param argv Argument list (None: sys.argv[1:]).
    @return None (exit code 1 on refusal, as main.py)
    """
    import main as fps_main  # configures logging like a generation run

    t_start = time.time()
    record_path, overrides = parse_overrides(argv)
    try:
        record = load_record(record_path)
        if "output" not in overrides:
            stem = Path(record_path).name
            stem = stem[:-len("_parametric.json")] if stem.endswith("_parametric.json") else Path(stem).stem
            overrides["output"] = str(Path(record_path).with_name(f"{stem}_regenerated"))
        args = namespace_from_record(record, overrides)
        config = config_from_args(args)
        fibers = fibers_from_record(record, config)
        voids = voids_from_record(record)
        sha = check_same_geometry(record, fibers)
    except (OSError, json.JSONDecodeError, RecordError, KeyError, TypeError) as e:
        logger.error(f"Cannot regenerate from {record_path}: {e}")
        sys.exit(1)

    logger.info(f"=== REGENERATION from {record_path}: {len(fibers)} fibers, {len(voids)} pore objects, "
                f"seed {config.seed}, geometry sha256 {sha[:12]}... ===")
    logger.info(f"Output prefix: {config.output_prefix}")
    fps_main.finalize(config, fibers, t_start, args=args, voids=voids)


if __name__ == "__main__":
    main()
