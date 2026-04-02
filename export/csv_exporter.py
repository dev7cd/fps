"""
export/csv_exporter.py
Export des fibres au format texte/CSV spécifique (set_tubes.csv).
Format: "tube" "disc" R R DirX DirY DirZ Pt1X Pt1Y Pt1Z Pt2X Pt2Y Pt2Z ...
"""

import logging
import numpy as np
from typing import List
from core.fiber import Fiber

logger = logging.getLogger(__name__)

class CSVFiberExporter:
    @staticmethod
    def export(fibers: List[Fiber], filename: str):
        """
        Génère un fichier .csv contenant les données des fibres.
        """
        # S'assurer de l'extension
        if not filename.endswith(".csv"):
            filename += ".csv"
            
        logger.info(f"Export CSV ({len(fibers)} entités) vers {filename}...")
        
        try:
            with open(filename, 'w') as f:
                for fib in fibers:
                    line_parts = []
                    
                    # 1. En-tête Fixe pour le format 'set_tubes'
                    line_parts.append('"tube"')
                    line_parts.append('"disc"')
                    
                    # 2. Rayons (Majeur/Mineur identiques ici car format "disc" attendu)
                    # Note : On exporte le rayon physique effectif
                    r = fib.radius
                    line_parts.append(f"{r:.6f}") # R1
                    line_parts.append(f"{r:.6f}") # R2
                    
                    # 3. Vecteur Directeur Global (Normalisé - Corde)
                    pts = fib.control_points
                    if len(pts) > 1:
                        vec = pts[-1] - pts[0]
                        norm = np.linalg.norm(vec)
                        if norm > 1e-9:
                            vec /= norm
                        else:
                            vec = np.array([1.0, 0.0, 0.0])
                    else:
                        vec = np.array([1.0, 0.0, 0.0])
                    
                    line_parts.append(f"{vec[0]:.10f}")
                    line_parts.append(f"{vec[1]:.10f}")
                    line_parts.append(f"{vec[2]:.10f}")
                    
                    # 4. Liste des points de contrôle
                    for p in pts:
                        line_parts.append(f"{p[0]:.6f}")
                        line_parts.append(f"{p[1]:.6f}")
                        line_parts.append(f"{p[2]:.6f}")
                    
                    # Écriture de la ligne (Séparateur Espace)
                    f.write(" ".join(line_parts) + "\n")
                    
        except Exception as e:
            logger.error(f"Echec écriture CSV {filename} : {e}")