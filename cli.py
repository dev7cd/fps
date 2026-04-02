"""
cli.py
Interface CLI Complète - Expose tous les paramètres de config.py
"""

import argparse
import numpy as np
from core.config import FiberPackingConfig

def parse_args():
    parser = argparse.ArgumentParser(
        description="Générateur RVE Fibres Hautes Performances (CSAW + Compression)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # --- 1. BOITE ET CIBLES ---
    g_box = parser.add_argument_group('1. Boîte et Cibles')
    g_box.add_argument("--dims", type=float, nargs=3, default=[1.0, 1.0, 1.0], help="Dimensions [Lx Ly Lz]")
    g_box.add_argument("--vf", type=float, default=0.05, help="Fraction Volumique Cible (Finale)")
    g_box.add_argument("--seed", type=int, default=None, help="Graine aléatoire (Reproductibilité)")
    
    # --- 2. GÉOMÉTRIE FIBRE ---
    g_geo = parser.add_argument_group('2. Géométrie Fibre')
    g_geo.add_argument("--radius", type=float, default=0.02, help="Rayon de calcul (Cercle de sécurité collision)")
    g_geo.add_argument("--section", type=str, default="circular", choices=["circular", "superelliptical"], help="Type de section réelle")
    
    # Paramètres fins de section
    g_geo.add_argument("--sec_major", type=float, default=0.019, help="Demi-axe majeur (réel)")
    g_geo.add_argument("--sec_minor", type=float, default=0.015, help="Demi-axe mineur (réel)")
    g_geo.add_argument("--sec_n", type=float, default=2.5, help="Exposant Superellipse (>2 = carré arrondi)")
    g_geo.add_argument("--clearance", type=float, default=0.0, help="Clearance minimale inter-fibres")

    # --- 3. TRAJECTOIRE (PHASE 1 - CSAW) ---
    g_traj = parser.add_argument_group('3. Trajectoire & Orientation')
    g_traj.add_argument("--min_pts", type=int, default=15, help="Min points de contrôle")
    g_traj.add_argument("--max_pts", type=int, default=20, help="Max points de contrôle")
    g_traj.add_argument("--step", type=float, default=0.07, help="Longueur moyenne d'un segment")
    g_traj.add_argument("--max_attempts", type=int, default=500, help="Tentatives max (RSA)")
    
    # Biais d'orientation
    g_traj.add_argument("--bias_type", type=str, default="planar", choices=["free", "uniaxial", "planar"])
    g_traj.add_argument("--bias_vec1", type=float, nargs=3, default=[1.0, 0.0, 0.0], help="Vecteur directeur 1")
    g_traj.add_argument("--bias_vec2", type=float, nargs=3, default=[0.0, 1.0, 0.0], help="Vecteur directeur 2 (si planaire)")
    g_traj.add_argument("--strength", type=float, default=0.0, help="Force du biais (0.0=Random, 1.0=Strict)")
    g_traj.add_argument("--curvature", type=float, default=60.0, help="Angle de courbure max entre segments (en Degrés)")
    g_traj.add_argument("--rsda", action='store_true', help="Activer RSDA (réarrangement dynamique en Phase 1)")
    g_traj.add_argument("--rsda_perturb", type=float, default=0.05, help="Intensité perturbation RSDA (ratio du rayon)")

    # --- 4. OPTIMISATION (PHASE 2 - DYNAMIQUE) ---
    g_opt = parser.add_argument_group('4. Optimisation (Phase 2)')
    g_opt.add_argument("--optimize", action='store_true', default=True, help="Activer l'optimiseur dynamique")
    g_opt.add_argument("--no_optimize", action='store_false', dest='optimize', help="Désactiver l'optimiseur")
    g_opt.add_argument("--opt_iters", type=int, default=10, help="Nombre de cycles d'optimisation")
    g_opt.add_argument("--jitter", type=float, default=0.05, help="Intensité des secousses (ratio du rayon)")
    g_opt.add_argument("--compression", type=float, default=0.01, help="Intensité compression centripète")
    g_opt.add_argument("--injection", type=int, default=50, help="Tentatives d'injection par cycle")

    # --- 5. POROSITÉ ---
    g_void = parser.add_argument_group('5. Porosité')
    g_void.add_argument("--porosity", action='store_true', help="Générer des pores/défauts")
    g_void.add_argument("--void_vf", type=float, default=0.01, help="Fraction volumique des pores")
    g_void.add_argument("--void_mean", type=float, default=0.01, help="Rayon moyen des pores")
    g_void.add_argument("--void_std", type=float, default=0.002, help="Ecart-type rayon pores")

    # --- 6. SYSTEME & EXPORTS ---
    g_sys = parser.add_argument_group('6. Système & Exports')
    g_sys.add_argument("--output", type=str, default="RVE_Result", help="Préfixe des fichiers de sortie")
    g_sys.add_argument("--no_mesh", action='store_true', help="Désactiver le maillage GMSH")
    g_sys.add_argument("--res_mesh", type=int, default=20, help="Résolution courbure maillage")
    g_sys.add_argument("--res_fft", type=int, default=128, help="Résolution grille Voxels (FFT)")
    g_sys.add_argument("--periodic_mesh", action='store_true', help="Activer le maillage périodique GMSH")
    g_sys.add_argument("--no_adaptive_mesh", action='store_true', help="Désactiver le raffinement adaptatif")
    g_sys.add_argument("--nastran", action='store_true', help="Exporter en format Nastran .bdf")
    g_sys.add_argument("--no_spatial_stats", action='store_true', help="Désactiver les descripteurs spatiaux (NND, Ripley K, g(r), Voronoi)")
    
    args = parser.parse_args()
    
    # --- LOGIQUE DE TRAITEMENT ---

    # 1. Traitement du biais (conversion CLI -> Format interne)
    bias_vectors = None
    if args.bias_type == "uniaxial":
        bias_vectors = [args.bias_vec1]
    elif args.bias_type == "planar":
        bias_vectors = [args.bias_vec1, args.bias_vec2]
    # 'free' laisse bias_vectors à None

    # 2. Construction de l'objet Config
    config = FiberPackingConfig(
        # Géométrie RVE
        box_dims=tuple(args.dims),
        target_volume_fraction=args.vf,
        seed=args.seed,
        
        # Fibre
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
            'max_curvature_angle': np.deg2rad(args.curvature), # Conversion Deg -> Rad
            'max_packing_attempts': args.max_attempts,
            'backtrack_depth': 200 # Valeur par défaut fixe ou exposable si besoin
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
        
        # Porosité
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