import argparse
import numpy as np
from core.config import FiberPackingConfig

## @file cli.py
#  @brief Command Line Interface (CLI) for the Fiber Packing System.
#  @details Parses user arguments and maps them to the FiberPackingConfig object
#  to control RVE generation, optimization, and export.

def parse_args():
    """!
    @brief Parses command line arguments and initializes the configuration.
    @details Organizes arguments into logical groups (Domain, Geometry, Trajectory,
    Optimization, Porosity, and System) and performs necessary unit conversions
    (e.g., degrees to radians for curvature).
    @return A tuple containing (config, args) where config is a FiberPackingConfig
    instance and args is the raw argparse Namespace.
    """
    parser = argparse.ArgumentParser(
        description="High-performance fiber RVE generator (CSAW + compression)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # --- 1. BOX AND TARGETS ---
    g_box = parser.add_argument_group('1. Box and targets')
    g_box.add_argument("--dims", type=float, nargs=3, default=[1.0, 1.0, 1.0], help="Dimensions [Lx Ly Lz]")
    g_box.add_argument("--vf", type=float, default=0.05, help="Target (final) volume fraction")
    g_box.add_argument("--seed", type=int, default=None, help="Random seed (reproducibility)")

    # --- 2. FIBER GEOMETRY ---
    g_geo = parser.add_argument_group('2. Fiber geometry')
    g_geo.add_argument("--radius", type=float, default=0.02, help="Collision radius (safety-circle envelope)")
    g_geo.add_argument("--section", type=str, default="circular", choices=["circular", "superelliptical"], help="Real section type")

    # Fine section parameters
    g_geo.add_argument("--sec_major", type=float, default=0.019, help="Major semi-axis (real)")
    g_geo.add_argument("--sec_minor", type=float, default=0.015, help="Minor semi-axis (real)")
    g_geo.add_argument("--sec_n", type=float, default=2.5, help="Super-ellipse exponent (>2 = rounded square)")
    g_geo.add_argument("--clearance", type=float, default=0.0, help="Minimum inter-fiber clearance")

    # --- 3. TRAJECTORY (PHASE 1 - CSAW) ---
    g_traj = parser.add_argument_group('3. Trajectory & orientation')
    g_traj.add_argument("--min_pts", type=int, default=29, help="Min control points")
    g_traj.add_argument("--max_pts", type=int, default=30, help="Max control points")
    g_traj.add_argument("--step", type=float, default=0.053, help="Mean length of a segment")
    g_traj.add_argument("--max_attempts", type=int, default=500, help="Max attempts (RSA)")

    # Orientation bias
    g_traj.add_argument("--bias_type", type=str, default="uniaxial", choices=["free", "uniaxial", "planar"])
    g_traj.add_argument("--bias_vec1", type=float, nargs=3, default=[1.0, 0.0, 0.0], help="Direction vector 1")
    g_traj.add_argument("--bias_vec2", type=float, nargs=3, default=[0.0, 1.0, 0.0], help="Direction vector 2 (if planar)")
    g_traj.add_argument("--strength", type=float, default=0.22, help="Bias strength (0.0=random, 1.0=strict)")
    g_traj.add_argument("--curvature", type=float, default=70.0, help="Max curvature angle between segments (in degrees)")
    g_traj.add_argument("--rsda", action='store_true', help="Enable RSDA (dynamic rearrangement in Phase 1)")
    g_traj.add_argument("--rsda_perturb", type=float, default=0.05, help="RSDA perturbation intensity (fraction of radius)")

    # --- 4. OPTIMISATION (PHASE 2 - DYNAMIC) ---
    g_opt = parser.add_argument_group('4. Optimisation (Phase 2)')
    g_opt.add_argument("--optimize", action='store_true', default=True, help="Enable the dynamic optimiser")
    g_opt.add_argument("--no_optimize", action='store_false', dest='optimize', help="Disable the optimiser")
    g_opt.add_argument("--opt_iters", type=int, default=10, help="Number of optimisation cycles")
    g_opt.add_argument("--jitter", type=float, default=0.05, help="Jitter intensity (fraction of radius)")
    g_opt.add_argument("--compression", type=float, default=0.01, help="Centripetal compression intensity")
    g_opt.add_argument("--injection", type=int, default=50, help="Injection attempts per cycle")

    # --- 5. POROSITY ---
    g_void = parser.add_argument_group('5. Porosity')
    g_void.add_argument("--porosity", action='store_true', help="Generate pores/defects")
    g_void.add_argument("--void_vf", type=float, default=0.01, help="Pore volume fraction")
    g_void.add_argument("--void_mean", type=float, default=0.01, help="Mean pore radius")
    g_void.add_argument("--void_std", type=float, default=0.002, help="Pore radius standard deviation")

    # --- 6. SYSTEM & EXPORTS ---
    g_sys = parser.add_argument_group('6. System & exports')
    g_sys.add_argument("--output", type=str, default="RVE_Result", help="Output file prefix")
    g_sys.add_argument("--no_mesh", action='store_true', help="Disable the GMSH mesh")
    g_sys.add_argument("--res_mesh", type=int, default=20, help="Mesh curvature resolution")
    g_sys.add_argument("--res_fft", type=int, default=128, help="Voxel grid resolution (FFT)")
    g_sys.add_argument("--periodic_mesh", action='store_true', help="Enable the periodic GMSH mesh")
    g_sys.add_argument("--no_adaptive_mesh", action='store_true', help="Disable the adaptive refinement")
    g_sys.add_argument("--nastran", action='store_true', help="Export to Nastran .bdf format")
    g_sys.add_argument("--no_spatial_stats", action='store_true', help="Disable the spatial descriptors (NND, Ripley K, g(r), Voronoi)")

    args = parser.parse_args()

    # --- PROCESSING LOGIC ---

    # 1. Bias processing (CLI -> internal format conversion)
    bias_vectors = None
    if args.bias_type == "uniaxial":
        bias_vectors = [args.bias_vec1]
    elif args.bias_type == "planar":
        bias_vectors = [args.bias_vec1, args.bias_vec2]
    # 'free' leaves bias_vectors as None

    # 2. Construction of the Config object
    config = FiberPackingConfig(
        # RVE geometry
        box_dims=tuple(args.dims),
        target_volume_fraction=args.vf,
        seed=args.seed,

        # Fiber
        fiber_radius=args.radius,
        min_clearance=args.clearance,
        fiber_section_type=args.section,
        section_parameters={
            'major_radius': args.radius,
            'minor_radius': args.radius,
            'exponent': args.sec_n
        },

        # Phase 1
        min_control_points=args.min_pts,
        max_control_points=args.max_pts,
        step_length_mean=args.step,
        generation_parameters={
            'orientation_bias': args.bias_type,
            'bias_vectors': bias_vectors,
            'bias_strength': args.strength,
            'max_curvature_angle': np.deg2rad(args.curvature), # Deg -> Rad conversion
            'max_packing_attempts': args.max_attempts,
            'backtrack_depth': 200 # Fixed default value, can be exposed if needed
        },

        # RSDA
        enable_rsda=args.rsda,
        rsda_perturbation_radius=args.rsda_perturb,

        # Phase 2
        enable_optimizer=args.optimize,
        optimizer_iterations=args.opt_iters,
        jitter_intensity=args.jitter,
        compression_intensity=args.compression,
        injection_per_iteration=args.injection,

        # Porosity
        generate_porosity=args.porosity,
        target_void_fraction=args.void_vf,
        void_radius_mean=args.void_mean,
        void_radius_std=args.void_std,

        # Export
        export_mesh=(not args.no_mesh),
        enable_periodic_mesh=args.periodic_mesh,
        enable_adaptive_mesh=(not args.no_adaptive_mesh),
        mesh_curvature_resolution=args.res_mesh,
        voxel_resolution=args.res_fft,
        output_prefix=args.output,
        export_nastran=args.nastran,
        compute_spatial_stats=(not args.no_spatial_stats),
    )

    return config, args
