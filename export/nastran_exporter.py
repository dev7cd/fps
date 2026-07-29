# SPDX-FileCopyrightText: 2026 Devine Ngouloubi <exauce-devine.ngouloubi@unicaen.fr>
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
export/nastran_exporter.py
Export of the GMSH mesh to the Nastran Bulk Data format (.bdf).
Uses meshio for the MSH -> BDF conversion with physical-group mapping.
"""

import logging
from typing import Optional, Dict

## @var logger
#  @brief Logger for the nastran_exporter module.
logger = logging.getLogger(__name__)


class NastranExporter:
    """!
    @class NastranExporter
    @brief Class handling the export and conversion of meshes to the Nastran (.bdf) format.
    """

    def __init__(self, config):
        """!
        @brief Initialises the Nastran exporter.
        @param config FiberPackingConfig Configuration object holding the global parameters.
        """
        self.config = config  ##< Simulation configuration.

    def convert_msh_to_bdf(self, msh_path: str, bdf_path: str,
                           material_map: Optional[Dict] = None) -> None:
        """!
        @brief Converts a GMSH mesh (.msh) to the Nastran Bulk Data format (.bdf).
        The GMSH physical groups are mapped to PSOLID properties.

        @param msh_path str Path of the source .msh file.
        @param bdf_path str Path of the destination .bdf file.
        @param material_map Optional[Dict] Optional - dictionary mapping physical-group tags
                            to properties {name, E, nu, rho}.
                          If provided, MAT1 cards are added to the .bdf.
        @return None
        """
        try:
            import meshio # type: ignore
        except ImportError:
            logger.error("meshio is not installed. Install it with: pip install meshio")
            return

        logger.info(f"Conversion {msh_path} -> {bdf_path}")

        try:
            mesh = meshio.read(msh_path)
        except Exception as e:
            logger.error(f"Read error {msh_path}: {e}")
            return

        try:
            meshio.write(bdf_path, mesh, file_format="nastran")
        except Exception as e:
            logger.error(f"Write error {bdf_path}: {e}")
            return

        # Add the material cards if provided
        if material_map:
            self._append_material_cards(bdf_path, material_map)

        logger.info(f"Nastran export finished: {bdf_path}")

    def _append_material_cards(self, bdf_path: str, material_map: Dict):
        """!
        @brief Adds MAT1 cards to the BDF file for each material.
        MAT1 format: MAT1, MID, E, G, NU, RHO

        @param bdf_path Path of the .bdf file.
        @param material_map Dictionary {MID: {"name": str, "E": float, "nu": float, "rho": float}}.
        @exception Exception Logs the error on write failure.
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
                    # Free-field format (commas)
                    f.write(f"MAT1,{mid},{E:.6E},{G:.6E},{nu:.4f},{rho:.6E}\n")

            logger.info(f"MAT1 cards added for {len(material_map)} materials")
        except Exception as e:
            logger.error(f"Error adding MAT1 cards: {e}")
