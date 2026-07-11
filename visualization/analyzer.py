## @file analyzer.py
#  @brief Microstructural viability auditor (Safety Check).
#
#  Coordinates geometric analysis, AD-PCA orientation tensors, and 
#  topological validation to ensure the RVE is valid for simulation.

from typing import List, Dict
import numpy as np
from .descriptors import AD_PCA_Analyzer, MicroDescriptor

class RVE_Analyzer:
    """!
    @brief High-level analyzer for Representative Volume Elements (RVE).

    Aggregates data from geometric descriptors, orientation tensors, 
    and clearance audits to provide a comprehensive viability report.
    """
    def __init__(self, fibers, config):
        """!
        @brief Initializes the analyzer.
        @param fibers List List of Fiber objects to analyze.
        @param config FiberPackingConfig Configuration object containing box dimensions and targets.
        """
        self.fibers = fibers  ##< Liste des fibres  ##< List of Fiber objects.
        self.config = config  ##< Configuration object.

    def perform_viability_audit(self) -> Dict:
        """!
        @brief Generates the final report for validation before simulation.
        @return Dict Dictionary containing status, volume fraction, orientation factors, and safety gaps.
        """
        logger = MicroDescriptor(self.fibers)
        ad_pca = AD_PCA_Analyzer(self.fibers)
        ad_pca.compute_all()

        geo = logger.compute_geometric_stats()
        spectrum = ad_pca.get_spectrum()

        # 1. Vérification Admissibilité (The Gap Audit)
        # Indispensable pour s'assurer qu'un maillage adaptatif passera
        min_gap = self._compute_min_clearance()

        # 2. Vf Réel vs Cible
        total_vol = sum(f.get_real_volume() for f in self.fibers)
        vf_achieved = total_vol / self.config.box_volume

        return {
            "status": "PASS" if min_gap > 0 else "FAIL_INTERSECTION",
            "volume_fraction": {
                "target": self.config.target_volume_fraction,
                "achieved": vf_achieved,
                "error": (vf_achieved - self.config.target_volume_fraction) / self.config.target_volume_fraction
            },
            "orientation": {
                "herman_f_axial": spectrum['f_axial'],
                "herman_f_planar": spectrum['f_planar']
            },
            "safety": {
                "min_interfiber_gap": min_gap,
                "tortuosity_avg": geo['tortuosity']['mean']
            }
        }

    def _compute_min_clearance(self) -> float:
        """!
        @brief Calculates the real minimum distance between fiber skins.
        @details Interfaces with TopologyValidator to perform segment-to-segment 
        distance checks with periodic boundary conditions.
        @return float Real minimum distance between fiber skins.
        """
        from validation.topology import TopologyValidator
        validator = TopologyValidator(self.config.box_dims)
        return validator.check_clearance(self.fibers)