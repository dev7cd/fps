"""
core/config.py
Configuration centralisée v6.0 - Stratégie de remplissage Haute Densité.
"""

from dataclasses import dataclass, field
from typing import Tuple, Dict, Any, Optional, Union, List
import numpy as np

@dataclass
class FiberPackingConfig:
    # --- 1. GÉOMÉTRIE DU DOMAINE (RVE) ---
    box_dims: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    target_volume_fraction: float = 0.45  # Vf final après phase 2
    seed: Optional[int] = None

    # --- 2. GÉOMÉTRIE DE LA FIBRE ---
    fiber_radius: Optional[float] = None   # Rayon de calcul (enveloppe circulaire)
    min_clearance: float = 0.0             # Espace minimal inter-fibres pour la grille spatiale
    fiber_section_type: str = 'superelliptical' # 'circular' ou 'superelliptical'
    
    # Paramètres réels de section (pour l'export GMSH/Voxelizer)
    section_parameters: Dict[str, Any] = field(default_factory=lambda: {
        'major_radius': 0.019,
        'minor_radius': 0.015,
        'exponent': 2.5,  # > 2 tend vers une section rectangulaire
    })

    # --- 3. GÉNÉRATION DE TRAJECTOIRE (PHASE 1 : CSAW) ---
    # Paramètres de longueur et de forme
    min_control_points: int = 15
    max_control_points: int = 20
    step_length_mean: float = 0.07  # Longueur de chaque segment
    
    generation_parameters: Dict[str, Any] = field(default_factory=lambda: {
        'orientation_bias': 'planar',    # 'free', 'uniaxial' (x, y, z), ou 'planar'
        'bias_vectors': [[1,0,0], [0,1,0]], # Vecteurs définissant le biais (Plan XY ici)
        'bias_strength': 0.8,            # 1.0 = alignement parfait | 0.0 = aléatoire
        'max_curvature_angle': np.pi/6,  # 30° max entre deux segments
        'max_packing_attempts': 5000,    # Nombre d'échecs tolérés en Phase 1
    })

    # --- RSDA (Réarrangement dynamique en Phase 1) ---
    enable_rsda: bool = False              # Activer le réarrangement dynamique
    rsda_perturbation_radius: float = 0.05 # Intensité perturbation (ratio du rayon)
    rsda_max_neighbors: int = 5            # Nombre max de voisins à perturber

    # --- 4. OPTIMISATION ET TASSEMENT (PHASE 2) ---
    enable_optimizer: bool = True
    optimizer_iterations: int = 10
    
    # Paramètres de mouvement
    jitter_intensity: float = 0.05       # Ratio du rayon (secousses aléatoires)
    compression_intensity: float = 0.01   # Ratio vers le centre à chaque cycle
    injection_per_iteration: int = 50    # Nb de tentatives d'ajout après tassement

    # --- 5. DÉFAUTS ET POROSITÉ (VOIDS) ---
    generate_porosity: bool = True
    target_void_fraction: float = 0.01   # Fraction volumique de bulles d'air
    void_radius_mean: float = 0.01
    void_radius_std: float = 0.002

    # --- 6. VALIDATION ET STATISTIQUES ---
    compute_spatial_stats: bool = True     # Descripteurs NND, Ripley K, g(r), Voronoi

    # --- 7. EXPORTS ET POST-TRAITEMENT ---
    export_mesh: bool = True
    enable_periodic_mesh: bool = False     # Maillage périodique GMSH (faces opposées)
    enable_adaptive_mesh: bool = True      # Champs de taille adaptatifs Distance+Threshold
    export_nastran: bool = False           # Export Nastran .bdf via meshio
    mesh_curvature_resolution: int = 20  # Résolution circulaire pour GMSH
    voxel_resolution: int = 256          # Taille de grille FFT (NxNxN)
    output_prefix: str = "RVE_Sample_60"

    @property
    def box_volume(self) -> float:
        return np.prod(self.box_dims)

    def estimate_radius_if_needed(self, estimated_num_fibers: int = 200):
        """Calcul automatique du rayon pour CSAW si non spécifié."""
        if self.fiber_radius is not None:
            return
            
        # Basé sur L_fibre moyenne x num_fibers
        avg_len = (self.min_control_points + self.max_control_points) / 2 * self.step_length_mean
        total_fiber_vol = self.box_volume * (self.target_volume_fraction * 0.8) # 80% en phase 1
        vol_per_fiber = total_fiber_vol / max(1, estimated_num_fibers)
        area = vol_per_fiber / avg_len
        self.fiber_radius = np.sqrt(area / np.pi)

    def __post_init__(self):
        self.estimate_radius_if_needed()