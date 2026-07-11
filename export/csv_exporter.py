## @file csv_exporter.py
#  @brief Export des fibres au format texte/CSV spécifique (set_tubes.csv).
#  @details Format: "tube" "disc" R R DirX DirY DirZ Pt1X Pt1Y Pt1Z Pt2X Pt2Y Pt2Z ...

import logging
import numpy as np
from typing import List
from core.fiber import Fiber

## @var logger
#  @brief Logger for the csv_exporter module.
logger = logging.getLogger(__name__)

class CSVFiberExporter:
    """!
    @brief Classe utilitaire pour l'exportation des fibres vers un format CSV compatible avec les lecteurs de tubes.
    """

    @staticmethod
    def export(fibers: List[Fiber], filename: str):
        """!
        @brief Génère un fichier .csv contenant les données géométriques des fibres.
        @details Le format d'exportation suit une structure spécifique :
        Type, Section, Rayon1, Rayon2, DirX, DirY, DirZ, [Points de contrôle...]
        @param fibers List[Fiber] Liste d'objets Fiber à exporter.
        @param filename str Chemin du fichier de destination (ajoute .csv si manquant).
        @return None
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
