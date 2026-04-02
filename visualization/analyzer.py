"""
visualization/analyzer.py
Auditeur de viabilité microstructurale (Safety Check).
"""

from typing import List, Dict
import numpy as np
from .descriptors import AD_PCA_Analyzer, MicroDescriptor

class RVE_Analyzer:
    def __init__(self, fibers, config):
        self.fibers = fibers
        self.config = config

    def perform_viability_audit(self) -> Dict:
        """Génère le rapport final pour validation avant simulation."""
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

    def _compute_min_clearance(self):
        """Calcule la distance minimale réelle entre les peaux de fibres."""
        from validation.topology import TopologyValidator
        validator = TopologyValidator(self.config.box_dims)
        return validator.check_clearance(self.fibers)