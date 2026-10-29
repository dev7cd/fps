# SPDX-FileCopyrightText: 2026 Devine Ngouloubi <exauce-devine.ngouloubi@unicaen.fr>
# SPDX-License-Identifier: AGPL-3.0-or-later

## @file main.py
#  @brief Main orchestrator for the Fiber Packing System (FPS).
#  @details Coordinates the multi-phase generation process:
#  1. Constructive placement (CSAW), 2. Dynamic optimization,
#  3. Topological validation, 4. Porosity generation,
#  and 5. Multi-format exports (GMSH, Voxel, CSV, JSON).


import sys
import os
import json
import time
import logging
import numpy as np
import copy

# Internal modules
from cli import parse_args
from core.config import FiberPackingConfig
from collision.detector import CollisionDetector
from generation.generator import FiberGenerator
from generation.optimizer_dynamic import DynamicOptimizer
from generation.periodicity import PeriodicManager
from generation.porosity_gen import PorosityGenerator
from core.fiber import Fiber # Needed to instantiate the ghosts

from validation.topology import TopologyValidator
from validation import geometric_audit

from export.gmsh_exporter import GmshExporter
from export.voxelizer import Voxelizer, UnsupportedSectionError, check_voxel_section_type
from export.csv_exporter import CSVFiberExporter
from export.abaqus_exporter import AbaqusVoxelExporter
from visualization.descriptors import AD_PCA_Analyzer, MicroDescriptor

# Logging configuration
## @var level
#  @brief Logging level.
## @var format
#  @brief Logging format string.
## @var datefmt
#  @brief Logging date format string.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
## @var logger
#  @brief Main logger for the RVE orchestrator.
logger = logging.getLogger("RVE_Orchestrator")

def save_parametric_record_O1(config: FiberPackingConfig, fibers: list, stats_dict: dict, filename: str,
                              validity: dict = None):
    """!
    @brief Saves the RVE geometry and certified analysis results to a JSON file.
    @details This "O1" record contains metadata, simulation parameters,
    computed microstructural statistics, and the raw control points for every fiber.
    @param config The configuration object used for the generation.
    @param fibers List of Fiber objects (roots).
    @param stats_dict Dictionary containing AD-PCA and spatial statistics.
    @param filename Output filename without extension.
    @param validity Validity record of the final geometric audit (F03), stored under "validity".
    @return None
    """
    import datetime

    def np_encoder(o):
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        raise TypeError(f"Object of type {type(o)} is not JSON serializable")

    record = {
        "metadata": {
            "timestamp": str(datetime.datetime.now()),
            "rve_type": "Periodic High-Density Composite",
        },
        "simulation_parameters": {
            "random_seed": config.seed,
            "box_dims": config.box_dims,
            "target_vf": config.target_volume_fraction,
            "algo_params": config.generation_parameters
        },
        "computed_statistics": stats_dict,
        "microstructure_data": geometric_audit.microstructure_records(fibers)
    }
    if validity is not None:
        record["validity"] = geometric_audit.json_safe(validity)

    with open(f"{filename}.json", "w") as f:
        json.dump(record, f, default=np_encoder, indent=4)

def main():
    """!
    @brief Main execution loop of the RVE generator.
    @details Parses CLI arguments, runs the generation phases, performs
    statistical audits, and triggers all requested exporters.
    @return None
    """
    t_start = time.time()

    # --- STEP 0: configuration ---
    config, args = parse_args()

    # The voxel export below is unconditional and the audit only covers circular sections:
    # refuse other sections now, before the costly generation, with the validity diagnostic.
    try:
        check_voxel_section_type(config.fiber_section_type)
    except UnsupportedSectionError as e:
        result = geometric_audit.refusal('input validation (section)',
                                         f"{e} main.py always exports the voxel grid: use --section circular.")
        path = geometric_audit.write_record(f"{config.output_prefix}_validity.json",
                                            geometric_audit.build_record(config, None, result, 1))
        logger.error(f"{result['cause']} Diagnostic: {path}. No generation, no export.")
        sys.exit(1)

    logger.info(f"=== INITIALISATION: target Vf={config.target_volume_fraction:.1%} ===")

    detector = CollisionDetector(config)
    generator = FiberGenerator(config, detector)
    fibers = []
    box_vol = config.box_volume

    # --- STEP 1: PHASE 1 (CSAW) ---
    logger.info("--- PHASE 1: Constructive placement (CSAW) ---")
    attempts = 0
    # Was hardcoded, which made --max_attempts inert. Grinding sequential
    # adsorption to exhaustion is also wasteful when phase 2 is enabled: the
    # collective rearrangement places what phase 1 can no longer fit.
    max_fails = config.generation_parameters.get('max_packing_attempts', 500)
    vf_phase1_limit = config.target_volume_fraction

    while attempts < max_fails:
        new_id = len(fibers) + 1
        fiber = generator.generate_fiber(new_id)

        if fiber:
            ghost_shifts = PeriodicManager.generate_ghosts(fiber, config.box_dims)
            fiber_group = [fiber]

            for s in ghost_shifts:
                g_pts = fiber.control_points + s
                g = Fiber(fiber.id, g_pts, fiber.radius, vars(config), is_ghost=True, parent_id=fiber.id)
                fiber_group.append(g)

            if detector.is_group_valid(fiber_group):
                fibers.append(fiber)
                detector.add_fibers_group(fiber_group)
                attempts = 0

                current_vol = sum(f.get_real_volume() for f in fibers)
                current_vf = current_vol / box_vol

                if len(fibers) % 25 == 0:
                    logger.info(f"Placement: {len(fibers)} fibers | Vf: {current_vf:.3f}")

                if current_vf >= vf_phase1_limit:
                    break
            else:
                # RSDA: attempt a dynamic rearrangement after many failures
                if config.enable_rsda and attempts > 100:
                    rsda_fiber = generator.attempt_rsda_placement(new_id, fibers)
                    if rsda_fiber:
                        rsda_shifts = PeriodicManager.generate_ghosts(rsda_fiber, config.box_dims)
                        rsda_group = [rsda_fiber]
                        for s in rsda_shifts:
                            g_pts = rsda_fiber.control_points + s
                            g = Fiber(rsda_fiber.id, g_pts, rsda_fiber.radius, vars(config),
                                      is_ghost=True, parent_id=rsda_fiber.id)
                            rsda_group.append(g)
                        fibers.append(rsda_fiber)
                        detector.add_fibers_group(rsda_group)
                        attempts = 0

                        current_vol = sum(f.get_real_volume() for f in fibers)
                        current_vf = current_vol / box_vol
                        logger.info(f"[RSDA] Placement succeeded: {len(fibers)} fibers | Vf: {current_vf:.3f}")

                        if current_vf >= vf_phase1_limit:
                            break
                        continue
                attempts += 1
        else:
            attempts += 1

    # --- STEP 2: PHASE 2 (densification) ---
    if config.enable_optimizer:
        logger.info("--- PHASE 2: Dynamic optimisation ---")
        optimizer = DynamicOptimizer(config, detector)
        fibers = optimizer.optimize_and_fill(fibers)

    finalize(config, fibers, t_start)


def finalize(config, fibers, t_start):
    """!
    @brief Final audit, porosity, statistics and exports of the generated fibers.
    @details The blocking audit (F03) runs on the fibers transmitted to the exports, before any
    other output: on `failed` or `undetermined` it writes `<prefix>_validity.json` and exits
    with code 1, leaving earlier exports of the same prefix untouched.
    @param config The configuration object.
    @param fibers Final fibers (roots), not modified after the audit.
    @param t_start Start time of the run.
    @return None
    """
    # --- STEP 3: AUDIT (blocking, F03) ---
    logger.info("--- PHASE 3: Geometric audit (blocking) ---")
    requested = getattr(config, 'min_clearance', 0.0)
    result = geometric_audit.audit(fibers, config.box_dims, requested,
                                   section_type=config.fiber_section_type,
                                   generate_porosity=config.generate_porosity)
    validity = geometric_audit.build_record(config, fibers, result, 0 if result['status'] == 'passed' else 1)
    path = geometric_audit.write_record(f"{config.output_prefix}_validity.json", validity)
    # Decision on the final status of the record: a passed audit whose hashes cannot be built is
    # turned into `undetermined` by build_record (no export without its link to the geometry).
    if validity['status'] != 'passed':
        logger.error(f"Geometric audit {validity['status']} ({validity['stage']}): {validity['cause']}. "
                     f"Diagnostic: {path}. No export written.")
        sys.exit(1)
    logger.info(f"Geometric audit passed: min clearance = {result['min_clearance']:.6e} "
                f"(requested >= {float(requested):.6e}, pair {result['responsible_pair']['fibers']}).")

    # Historical statistic (generation detector kernel), kept as a separate measure.
    validator = TopologyValidator(config.box_dims)
    min_clearance = validator.check_clearance(fibers)

    # --- STEP 4: POROSITY ---
    voids = []
    if config.generate_porosity:
        logger.info("--- PHASE 4: Porosity generation ---")
        p_gen = PorosityGenerator(config)
        voids = p_gen.generate_voids(fibers)

    # --- STEP 5: EXPORTS AND STATS ---
    logger.info("--- PHASE 5: Statistics computation and exports ---")

    # 1. Compute the global statistics
    logger.info("Computing the final descriptors...")
    ad_pca = AD_PCA_Analyzer(fibers)
    ad_pca.compute_all()
    spectrum = ad_pca.get_spectrum()

    descriptor = MicroDescriptor(fibers)
    geo_stats = descriptor.compute_geometric_stats()

    final_stats = {
        "volume_fraction_real": sum(f.get_real_volume() for f in fibers) / config.box_volume,
        "orientation_tensor_axial": spectrum['A_axial'],
        "orientation_tensor_planar": spectrum['A_planar'],
        "hermans_factors": {
            "f_axial": spectrum['f_axial'],
            "f_planar": spectrum['f_planar']
        },
        "tortuosity": geo_stats['tortuosity'],
        "topology_gap_min": min_clearance
    }

    # 1b. Spatial statistical descriptors
    if config.compute_spatial_stats:
        from validation.statistics import (nearest_neighbor_distance, ripley_k_function,
                                           pair_correlation_function, voronoi_statistics)
        logger.info("Computing the spatial descriptors (NND, Ripley K, g(r), Voronoi)...")
        nnd = nearest_neighbor_distance(fibers, config.box_dims)
        ripley = ripley_k_function(fibers, config.box_dims)
        pcf = pair_correlation_function(fibers, config.box_dims)
        voronoi = voronoi_statistics(fibers, config.box_dims)
        final_stats["spatial"] = {
            "nnd": nnd,
            "ripley_k": ripley,
            "pair_correlation": pcf,
            "voronoi": voronoi
        }
        logger.info(f"  Mean NND = {nnd['nnd_mean']:.6f}, Voronoi CV = {voronoi['areas_cv']:.4f}")

    # 2. O1 save (parametric record)
    save_parametric_record_O1(config, fibers, final_stats, f"{config.output_prefix}_parametric", validity)

    # 3. CSV saves (DOUBLE EXPORT)

    # A. Roots only (for lightweight analysis)
    logger.info("CSV export: roots only")
    CSVFiberExporter.export(fibers, f"{config.output_prefix}_roots.csv")

    # B. Full periodic configuration (roots + explicit ghosts)
    logger.info("Generating the extended periodic configuration (ghosts)...")
    all_periodic_fibers = []
    # First add all the roots
    all_periodic_fibers.extend(fibers)

    # Generate and add all the ghosts
    for f in fibers:
        shifts = PeriodicManager.generate_ghosts(f, config.box_dims)
        for s in shifts:
            g_pts = f.control_points + s
            # Create a full Fiber object for the exporter
            g_fib = Fiber(
                fiber_id=f.id,
                control_points=g_pts,
                radius=f.radius,
                config_params=vars(config),
                is_ghost=True,
                parent_id=f.id
            )
            all_periodic_fibers.append(g_fib)

    logger.info(f"CSV export: periodic configuration ({len(all_periodic_fibers)} objects)")
    CSVFiberExporter.export(all_periodic_fibers, f"{config.output_prefix}_full_periodic.csv")

    # 4. GMSH & Voxel (handle the ghosts internally, so we pass only the roots)
    logger.info("FFT generation...")
    vox = Voxelizer(config)
    grid = vox.create_grid(fibers, voids, res=config.voxel_resolution)
    np.save(f"{config.output_prefix}_voxelmap.npy", grid)
    vox.save_pgm(grid, config.output_prefix)

    # Abaqus voxel mesh: written before the CAD export, which can take tens of minutes
    if config.export_abaqus:
        res = config.abaqus_resolution or config.voxel_resolution
        abq_grid = grid if res == config.voxel_resolution else vox.create_grid(fibers, voids, res=res)
        AbaqusVoxelExporter(config).write(abq_grid, f"{config.output_prefix}.inp", periodic=config.abaqus_periodic)

    if config.export_mesh:
        exporter = GmshExporter(config)
        export_check = exporter.generate_mesh(fibers, voids, config.output_prefix)

        # Nastran .bdf export if requested
        if config.export_nastran:
            from export.nastran_exporter import NastranExporter
            nastran = NastranExporter(config)
            nastran.convert_msh_to_bdf(
                f"{config.output_prefix}.msh",
                f"{config.output_prefix}.bdf"
            )

        if export_check["status"] != "passed":
            logger.error(f"=== FAILED export check in {time.time() - t_start:.2f}s ===")
            sys.exit(1)



    logger.info(f"=== FINISHED in {time.time() - t_start:.2f}s ===")

if __name__ == "__main__":
    main()
