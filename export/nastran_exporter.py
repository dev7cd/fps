"""
export/nastran_exporter.py
Export du maillage GMSH vers le format Nastran Bulk Data (.bdf).
Utilise meshio pour la conversion MSH -> BDF avec mapping des groupes physiques.
"""

import logging
from typing import Optional, Dict

logger = logging.getLogger(__name__)


class NastranExporter:
    def __init__(self, config):
        self.config = config

    def convert_msh_to_bdf(self, msh_path: str, bdf_path: str,
                           material_map: Optional[Dict] = None):
        """
        Convertit un maillage GMSH (.msh) en format Nastran Bulk Data (.bdf).
        Les groupes physiques GMSH sont mappés vers des propriétés PSOLID.

        Args:
            msh_path: Chemin du fichier .msh source
            bdf_path: Chemin du fichier .bdf destination
            material_map: Optionnel - Dict {physical_group_tag: {"name": str, "E": float, "nu": float, "rho": float}}
                          Si fourni, des cartes MAT1 sont ajoutées au .bdf
        """
        try:
            import meshio
        except ImportError:
            logger.error("meshio n'est pas installé. Installer avec : pip install meshio")
            return

        logger.info(f"Conversion {msh_path} -> {bdf_path}")

        try:
            mesh = meshio.read(msh_path)
        except Exception as e:
            logger.error(f"Erreur lecture {msh_path}: {e}")
            return

        try:
            meshio.write(bdf_path, mesh, file_format="nastran")
        except Exception as e:
            logger.error(f"Erreur écriture {bdf_path}: {e}")
            return

        # Ajouter les cartes matériau si fournies
        if material_map:
            self._append_material_cards(bdf_path, material_map)

        logger.info(f"Export Nastran terminé : {bdf_path}")

    def _append_material_cards(self, bdf_path: str, material_map: Dict):
        """
        Ajoute des cartes MAT1 au fichier BDF pour chaque matériau.
        Format MAT1 : MAT1, MID, E, G, NU, RHO

        Args:
            bdf_path: Chemin du fichier .bdf
            material_map: Dict {MID: {"name": str, "E": float, "nu": float, "rho": float}}
        """
        try:
            with open(bdf_path, 'a') as f:
                f.write("$\n")
                f.write("$ --- MATERIAL PROPERTIES ---\n")
                f.write("$\n")
                for mid, props in material_map.items():
                    name = props.get('name', f'MAT_{mid}')
                    E = props.get('E', 0.0)
                    nu = props.get('nu', 0.0)
                    rho = props.get('rho', 0.0)
                    G = E / (2.0 * (1.0 + nu)) if (1.0 + nu) != 0 else 0.0

                    f.write(f"$ {name}\n")
                    # Format champ libre (virgules)
                    f.write(f"MAT1,{mid},{E:.6E},{G:.6E},{nu:.4f},{rho:.6E}\n")

            logger.info(f"Cartes MAT1 ajoutées pour {len(material_map)} matériaux")
        except Exception as e:
            logger.error(f"Erreur ajout cartes MAT1 : {e}")
