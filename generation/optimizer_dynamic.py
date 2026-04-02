"""
generation/optimizer_dynamic.py

"""

import numpy as np
import logging
import copy
from typing import List, Tuple, Set, Dict
from collision.detector import CollisionDetector
from generation.periodicity import PeriodicManager
from dataclasses import dataclass, asdict


logger = logging.getLogger(__name__)


class DynamicOptimizer:
    def __init__(self, config, detector):
        self.config = config
        self.detector = detector
        self.rng = np.random.default_rng(config.seed)

    def optimize_and_fill(self, fibers: List):
        """Cycle de densification sans dilatation (Compression et Shaking)."""
        target_vf = self.config.target_volume_fraction
        # Utilisation de la méthode de volume réel pour plus de précision
        current_vf = self._calculate_vf(fibers)
        
        logger.info(f"Début densification. Vf actuel: {current_vf:.3f}")

        iteration = 0
        max_iterations = 100
        while current_vf < target_vf and iteration < max_iterations:
            iteration += 1
            
            # 1. Agitation (Jitter)
            self._apply_global_jitter(fibers)
            
            # 2. Compression vers le centre
            self._apply_centripetal_compression(fibers, intensity=0.01)
            
            # 3. Résolution des collisions résiduelles
            self._resolve_all_collisions(fibers)
            
            # 4. Injection de nouvelles fibres
            added_fibers = self._inject_additional_fibers(fibers)
            if len(added_fibers) > 0:
                fibers.extend(added_fibers)
                logger.info(f"Injecté {len(added_fibers)} nouvelles fibres après tassement.")
            
            current_vf = self._calculate_vf(fibers)
            logger.info(f"Itération {iteration}: Vf = {current_vf:.4f}")
            
        return fibers

    def _resolve_all_collisions(self, fibers):
        """Cherche et repousse les paires en conflit."""
        for _ in range(3): # Tentatives de résolution
            collisions = self._detect_collision_pairs(fibers)
            if not collisions: 
                break
            
            for f1, f2 in collisions:
                # nudge déplace f1 loin de f2
                self._nudge_fibers(f1, f2, fibers)

    def _detect_collision_pairs(self, fibers) -> List[Tuple]:
        """Identifie les paires root-root ou root-ghost en collision."""
        colliding_pairs = []
        seen_collisions = set() # Pour ne pas traiter (A,B) et (B,A)

        for f1 in fibers:
            neighbors = self.detector.grid.query_neighbors(f1.bbox, exclude_id=f1.id)
            for nb in neighbors:
                # nb peut être un ghost. On compare les parent_id
                pair_id = tuple(sorted((f1.id, nb.parent_id)))
                if pair_id in seen_collisions: 
                    continue

                if self.detector.check_collision_fine(f1, nb):
                    # nb est l'objet physique en collision. On l'ajoute.
                    colliding_pairs.append((f1, nb))
                    seen_collisions.add(pair_id)
        return colliding_pairs

    def _nudge_fibers(self, f1, target, all_fibers):
        """Pousse la fibre f1 loin de 'target' (qui peut être un ghost)."""
        c1 = np.mean(f1.centerline, axis=0)
        c2 = np.mean(target.centerline, axis=0) # Fonctionne que ce soit root ou ghost
        
        vec = c1 - c2
        dist = np.linalg.norm(vec)
        if dist == 0: 
            vec = self.rng.standard_normal(3)
        
        push = (vec / np.linalg.norm(vec)) * (f1.radius * 0.05)
        
        old_pts = f1.control_points.copy()
        f1.control_points += push
        f1.refresh_geometry()
        
        # Validation : La fibre f1 ne doit pas être en collision dans sa nouvelle position
        if self._has_any_collision(f1):
            # Échec du mouvement
            f1.control_points = old_pts
            f1.refresh_geometry()

    def _has_any_collision(self, fiber):
        """Vérifie si une fibre seule est en collision dans la grille actuelle."""
        neighbors = self.detector.grid.query_neighbors(fiber.bbox, exclude_id=fiber.id)
        for nb in neighbors:
            if self.detector.check_collision_fine(fiber, nb):
                return True
        return False

    def _calculate_vf(self, fibers):
        if not fibers: return 0.0
        total_vol = sum(f.get_real_volume() for f in fibers)
        return total_vol / np.prod(self.config.box_dims)

    def _apply_global_jitter(self, fibers):
        for f in fibers:
            old_pts = f.control_points.copy()
            shift = self.rng.uniform(-0.01, 0.01, 3) * f.radius
            f.control_points += shift
            f.refresh_geometry()
            if self._has_any_collision(f):
                f.control_points = old_pts
                f.refresh_geometry()

    def _apply_centripetal_compression(self, fibers, intensity):
        center = np.array(self.config.box_dims) / 2
        for f in fibers:
            centroid = np.mean(f.control_points, axis=0)
            push = (center - centroid) * intensity
            old_pts = f.control_points.copy()
            f.control_points += push
            f.refresh_geometry()
            if self._has_any_collision(f):
                f.control_points = old_pts
                f.refresh_geometry()

    def _inject_additional_fibers(self, fibers):
        from generation.generator import FiberGenerator
        # Note : detector contient déjà les anciennes fibres indexées
        gen = FiberGenerator(self.config, self.detector)
        added = []
        for i in range(10): # Tente 10 injections
            new_f = gen.generate_fiber(len(fibers) + i + 1)
            if new_f:
                added.append(new_f)
                # Ajout immédiat à la grille pour les suivantes
                self.detector.add_fibers_group([new_f])
        return added