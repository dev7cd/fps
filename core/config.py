# SPDX-FileCopyrightText: 2026 Devine Ngouloubi <exauce-devine.ngouloubi@unicaen.fr>
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
core/config.py
Centralised configuration v6.0 - High-density filling strategy.
"""

from dataclasses import dataclass, field
from typing import Tuple, Dict, Any, Optional, Union, List
import numpy as np

@dataclass
class FiberPackingConfig:
    """!
    @class FiberPackingConfig
    @brief Data structure centralising all simulation parameters.

    Manages the RVE dimensions, the fiber properties, the parameters of the
    generation (CSAW) and optimisation (RSDA) algorithms, and the export options.
    """

    # --- 1. DOMAIN GEOMETRY (RVE) ---
    box_dims: Tuple[float, float, float] = (1.0, 1.0, 1.0)  ##< Domain dimensions (Lx, Ly, Lz)
    target_volume_fraction: float = 0.45  ##< Target volume fraction (Vf) after phase 2
    seed: Optional[int] = None            ##< Random seed for reproducibility

    # --- 2. FIBER GEOMETRY ---
    fiber_radius: Optional[float] = None   ##< Collision radius (circular envelope for collisions)
    min_clearance: float = 0.0             ##< Minimum inter-fiber clearance (contact tolerance)
    fiber_section_type: str = 'superelliptical' ##< Section type: 'circular' or 'superelliptical'

    ## Real section parameters (for GMSH/Voxelizer export)
    section_parameters: Dict[str, Any] = field(default_factory=lambda: {
        'major_radius': 0.019,
        'minor_radius': 0.015,
        'exponent': 2.5,  # > 2 tends towards a rectangular section
    })

    # --- 3. TRAJECTORY GENERATION (PHASE 1: CSAW) ---
    min_control_points: int = 15    ##< Minimum number of control points per fiber
    max_control_points: int = 20    ##< Maximum number of control points per fiber
    step_length_mean: float = 0.07  ##< Mean length of each trajectory segment

    generation_parameters: Dict[str, Any] = field(default_factory=lambda: {
        'orientation_bias': 'planar',    # 'free', 'uniaxial' (x, y, z), or 'planar'
        'bias_vectors': [[1,0,0], [0,1,0]], # Vectors defining the bias (XY plane here)
        'bias_strength': 0.8,            # 1.0 = perfect alignment | 0.0 = random
        'max_curvature_angle': np.pi/6,  # 30 deg max between two segments
        'max_packing_attempts': 5000,    # Number of tolerated failures in Phase 1
    })

    # --- RSDA (Dynamic rearrangement in Phase 1) ---
    enable_rsda: bool = False              ##< Enable dynamic rearrangement (relaxation)
    rsda_perturbation_radius: float = 0.05 ##< Perturbation intensity (fraction of radius)
    rsda_max_neighbors: int = 5            ##< Max number of neighbours to perturb on a conflict

    # --- 4. OPTIMISATION AND COMPACTION (PHASE 2) ---
    enable_optimizer: bool = True          ##< Enable the compaction phase
    optimizer_iterations: int = 10         ##< Number of compaction/injection cycles

    jitter_intensity: float = 0.05       ##< Fraction of radius for random jitter
    compression_intensity: float = 0.01   ##< Displacement fraction towards the centre per cycle
    injection_per_iteration: int = 50    ##< Number of fiber-insertion attempts after compaction

    # --- 5. DEFECTS AND POROSITY (VOIDS) ---
    generate_porosity: bool = True       ##< Enable air-bubble (void) generation
    target_void_fraction: float = 0.01   ##< Target porosity volume fraction
    void_radius_mean: float = 0.01       ##< Mean bubble radius
    void_radius_std: float = 0.002       ##< Standard deviation of the bubble radius

    # --- 6. VALIDATION AND STATISTICS ---
    compute_spatial_stats: bool = True     ##< Compute NND, Ripley K, g(r), Voronoi

    # --- 7. EXPORTS AND POST-PROCESSING ---
    export_mesh: bool = True               ##< Generate the volume mesh (GMSH)
    enable_periodic_mesh: bool = False     ##< Enforce periodicity of opposite faces
    enable_adaptive_mesh: bool = True      ##< Use adaptive size fields
    export_nastran: bool = False           ##< Export to .bdf format via meshio
    mesh_curvature_resolution: int = 20    ##< Number of segments to discretise the section
    voxel_resolution: int = 256            ##< Grid resolution for FFT export (NxNxN)
    output_prefix: str = "RVE_Sample_60"   ##< Output file prefix

    @property
    def box_volume(self) -> float:
        """!
        @brief Computes the total volume of the RVE domain.
        @return float Total volume of the RVE domain.
        """
        return np.prod(self.box_dims)

    def estimate_radius_if_needed(self, estimated_num_fibers: int = 200):
        """!
        @brief Automatic radius estimation for CSAW if not specified.
        @param estimated_num_fibers int Estimated number of fibers used for the computation.
        """
        if self.fiber_radius is not None:
            return

        # Based on mean fiber length x num_fibers
        avg_len = (self.min_control_points + self.max_control_points) / 2 * self.step_length_mean
        total_fiber_vol = self.box_volume * (self.target_volume_fraction * 0.8) # 80% in phase 1
        vol_per_fiber = total_fiber_vol / max(1, estimated_num_fibers)
        area = vol_per_fiber / avg_len
        self.fiber_radius = np.sqrt(area / np.pi)

    def __post_init__(self):
        """!
        @brief Post-construction initialisation to estimate the radius if needed.
        """
        self.estimate_radius_if_needed()
