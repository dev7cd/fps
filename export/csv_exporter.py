# SPDX-FileCopyrightText: 2026 Devine Ngouloubi <exauce-devine.ngouloubi@unicaen.fr>
# SPDX-License-Identifier: AGPL-3.0-or-later

## @file csv_exporter.py
#  @brief Export of fibers to a specific text/CSV format (set_tubes.csv).
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
    @brief Utility class for exporting fibers to a CSV format compatible with tube readers.
    """

    @staticmethod
    def export(fibers: List[Fiber], filename: str):
        """!
        @brief Generates a .csv file containing the geometric data of the fibers.
        @details The export format follows a specific structure:
        Type, Section, Radius1, Radius2, DirX, DirY, DirZ, [Control points...]
        @param fibers List[Fiber] List of Fiber objects to export.
        @param filename str Path of the destination file (adds .csv if missing).
        @return None
        """
        # Ensure the extension
        if not filename.endswith(".csv"):
            filename += ".csv"

        logger.info(f"CSV export ({len(fibers)} entities) to {filename}...")

        try:
            with open(filename, 'w') as f:
                for fib in fibers:
                    line_parts = []

                    # 1. Fixed header for the 'set_tubes' format
                    line_parts.append('"tube"')
                    line_parts.append('"disc"')

                    # 2. Radii (major/minor identical here since 'disc' format is expected)
                    # Note: export the effective physical radius
                    r = fib.radius
                    line_parts.append(f"{r:.6f}") # R1
                    line_parts.append(f"{r:.6f}") # R2

                    # 3. Global direction vector (normalised chord)
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

                    # 4. List of control points
                    for p in pts:
                        line_parts.append(f"{p[0]:.6f}")
                        line_parts.append(f"{p[1]:.6f}")
                        line_parts.append(f"{p[2]:.6f}")

                    # Write the line (space separator)
                    f.write(" ".join(line_parts) + "\n")

        except Exception as e:
            logger.error(f"CSV write failed {filename}: {e}")
