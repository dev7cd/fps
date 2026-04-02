"""
core/config.py
Configuration centralisée v6.0 - Stratégie de remplissage Haute Densité.
"""

from dataclasses import dataclass, field
from typing import Tuple, Dict, Any, Optional, Union, List
import numpy as np

@dataclass
class FiberPackingConfig:
    """
    @class FiberPackingConfig
    @brief Structure de données centralisant tous les paramètres de simulation.
    
    Gère les dimensions du RVE, les propriétés des fibres, les paramètres des algorithmes
    de génération (CSAW) et d'optimisation (RSDA), ainsi que les options d'export.
    """

    # --- 1. GÉOMÉTRIE DU DOMAINE (RVE) ---
    box_dims: Tuple[float, float, float] = (1.0, 1.0, 1.0)  #< Dimensions (Lx, Ly, Lz) du domaine
    target_volume_fraction: float = 0.45  #< Fraction volumique cible (Vf) après phase 2
    seed: Optional[int] = None            #< Graine aléatoire pour la reproductibilité

    # --- 2. GÉOMÉTRIE DE LA FIBRE ---
    fiber_radius: Optional[float] = None   #< Rayon de calcul (enveloppe circulaire pour collisions)
    min_clearance: float = 0.0             #< Espace minimal inter-fibres (tolérance de contact)
    fiber_section_type: str = 'superelliptical' #< Type de section : 'circular' ou 'superelliptical'
    
    ## Paramètres réels de section (pour l'export GMSH/Voxelizer)
    section_parameters: Dict[str, Any] = field(default_factory=lambda: {
        'major_radius': 0.019,
        'minor_radius': 0.015,
        'exponent': 2.5,  # > 2 tend vers une section rectangulaire
    })

    # --- 3. GÉNÉRATION DE TRAJECTOIRE (PHASE 1 : CSAW) ---
    min_control_points: int = 15    #< Nombre minimum de points de contrôle par fibre
    max_control_points: int = 20    #< Nombre maximum de points de contrôle par fibre
    step_length_mean: float = 0.07  #< Longueur moyenne de chaque segment de la trajectoire
    
    generation_parameters: Dict[str, Any] = field(default_factory=lambda: {
        'orientation_bias': 'planar',    # 'free', 'uniaxial' (x, y, z), ou 'planar'
        'bias_vectors': [[1,0,0], [0,1,0]], # Vecteurs définissant le biais (Plan XY ici)
        'bias_strength': 0.8,            # 1.0 = alignement parfait | 0.0 = aléatoire
        'max_curvature_angle': np.pi/6,  # 30° max entre deux segments
        'max_packing_attempts': 5000,    # Nombre d'échecs tolérés en Phase 1
    })

    # --- RSDA (Réarrangement dynamique en Phase 1) ---
    enable_rsda: bool = False              #< Activer le réarrangement dynamique (Relaxation)
    rsda_perturbation_radius: float = 0.05 #< Intensité de perturbation (ratio du rayon)
    rsda_max_neighbors: int = 5            #< Nombre max de voisins à perturber lors d'un conflit

    # --- 4. OPTIMISATION ET TASSEMENT (PHASE 2) ---
    enable_optimizer: bool = True          #< Activer la phase de tassement
    optimizer_iterations: int = 10         #< Nombre de cycles de tassement/injection
    
    jitter_intensity: float = 0.05       #< Ratio du rayon pour les secousses aléatoires
    compression_intensity: float = 0.01   #< Ratio de déplacement vers le centre à chaque cycle
    injection_per_iteration: int = 50    #< Nb de tentatives d'ajout de fibres après tassement

    # --- 5. DÉFAUTS ET POROSITÉ (VOIDS) ---
    generate_porosity: bool = True       #< Activer la génération de bulles d'air
    target_void_fraction: float = 0.01   #< Fraction volumique cible de porosité
    void_radius_mean: float = 0.01       #< Rayon moyen des bulles
    void_radius_std: float = 0.002       #< Écart-type du rayon des bulles

    # --- 6. VALIDATION ET STATISTIQUES ---
    compute_spatial_stats: bool = True     #< Calculer NND, Ripley K, g(r), Voronoi

    # --- 7. EXPORTS ET POST-TRAITEMENT ---
    export_mesh: bool = True               #< Générer le maillage volumique (GMSH)
    enable_periodic_mesh: bool = False     #< Forcer la périodicité des faces opposées
    enable_adaptive_mesh: bool = True      #< Utiliser des champs de taille adaptatifs
    export_nastran: bool = False           #< Export au format .bdf via meshio
    mesh_curvature_resolution: int = 20    #< Nombre de segments pour discrétiser la section
    voxel_resolution: int = 256            #< Résolution de la grille pour export FFT (NxNxN)
    output_prefix: str = "RVE_Sample_60"   #< Préfixe des fichiers de sortie

    @property
    def box_volume(self) -> float:
        """@return Volume total du domaine RVE."""
        return np.prod(self.box_dims)

    def estimate_radius_if_needed(self, estimated_num_fibers: int = 200):
        """
        @brief Calcul automatique du rayon pour CSAW si non spécifié.
        @param estimated_num_fibers Nombre estimé de fibres pour le calcul.
        """
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