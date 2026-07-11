"""
generation/optimizer_dynamic.py
Optimiseur dynamique pour la densification du VER par agitation et compression.
Implémente des méthodes de type Monte-Carlo pour augmenter la fraction volumique.
"""

import numpy as np
import logging
import copy
from typing import List, Tuple, Set, Dict
from collision.detector import CollisionDetector
from generation.periodicity import PeriodicManager
from dataclasses import dataclass, asdict


## @var logger
#  @brief Logger pour le module d'optimisation dynamique.
logger = logging.getLogger(__name__)


class DynamicOptimizer:
    """
    @class DynamicOptimizer
    @brief Gère la densification post-génération des fibres.
    
    Cette classe permet d'augmenter la fraction volumique (Vf) en déplaçant 
    itérativement les fibres existantes pour créer de l'espace pour de nouvelles injections.
    """
    def __init__(self, config, detector):
        """! @brief Placeholder.
        @param self, config, detector 
        @return None
        """
        """
        @brief Initialise l'optimiseur avec la configuration et le détecteur de collisions.
        @param config FiberPackingConfig Configuration du problème.
        @param detector CollisionDetector Détecteur de collisions.
        """
        ## @var config
        #  @brief Configuration globale.
        self.config = config
        ## @var detector
        #  @brief Détecteur de collisions.
        self.detector = detector
        ## @var rng
        #  @brief Générateur de nombres aléatoires.
        self.rng = np.random.default_rng(config.seed)

    def optimize_and_fill(self, fibers: List):
        """! @brief Placeholder.
        @param self, fibers: List 
        @return None
        """
        """
        @brief Cycle principal de densification (Compression et Shaking).
        @param fibers List Liste des fibres initiales à densifier.
        @return List Liste des fibres mise à jour et complétée."""
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
        """! @brief Placeholder.
        @param self, fibers 
        @return None
        """
        """
        @brief Identifie et tente de résoudre les collisions par micro-déplacements.
        @param fibers Liste des fibres à traiter.
        """
        for _ in range(3): # Tentatives de résolution
            collisions = self._detect_collision_pairs(fibers)
            if not collisions: 
                break
            
            for f1, f2 in collisions:
                # nudge déplace f1 loin de f2
                self._nudge_fibers(f1, f2, fibers)

    def _detect_collision_pairs(self, fibers) -> List[Tuple]:
        """
        @brief Détecte les paires de fibres (ou ghosts) en intersection.
        @return Liste de tuples (fibre_1, fibre_2_ou_ghost).
        """
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
        """! @brief Placeholder.
        @param self, f1, target, all_fibers 
        @return None
        """
        """
        @brief Applique une force de répulsion à f1 par rapport à une cible.
        @param f1 La fibre à déplacer.
        @param target La fibre ou le ghost provoquant la collision.
        @param all_fibers Contexte global des fibres.
        """
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
        """! @brief Placeholder.
        @param self, fiber 
        @return None
        """
        """
        @brief Vérifie si une fibre donnée est en collision avec le reste du domaine.
        @return True si une collision est détectée.
        """
        neighbors = self.detector.grid.query_neighbors(fiber.bbox, exclude_id=fiber.id)
        for nb in neighbors:
            if self.detector.check_collision_fine(fiber, nb):
                return True
        return False

    def _calculate_vf(self, fibers):
        """! @brief Placeholder.
        @param self, fibers 
        @return None
        """
        """
        @brief Calcule la fraction volumique réelle actuelle.
        """
        if not fibers: return 0.0
        total_vol = sum(f.get_real_volume() for f in fibers)
        return total_vol / np.prod(self.config.box_dims)

    def _apply_global_jitter(self, fibers):
        """! @brief Placeholder.
        @param self, fibers 
        @return None
        """
        """
        @brief Applique un léger déplacement aléatoire à toutes les fibres (agitation thermique).
        @param fibers Liste des fibres à agiter.
        """
        for f in fibers:
            old_pts = f.control_points.copy()
            shift = self.rng.uniform(-0.01, 0.01, 3) * f.radius
            f.control_points += shift
            f.refresh_geometry()
            if self._has_any_collision(f):
                f.control_points = old_pts
                f.refresh_geometry()

    def _apply_centripetal_compression(self, fibers, intensity):
        """! @brief Placeholder.
        @param self, fibers, intensity 
        @return None
        """
        """
        @brief Déplace les fibres vers le centre du domaine pour libérer de l'espace aux parois.
        @param fibers Liste des fibres.
        @param intensity Force de la compression (0.0 à 1.0).
        """
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
        """! @brief Placeholder.
        @param self, fibers 
        @return None
        """
        """
        @brief Tente d'insérer de nouvelles fibres dans les espaces libérés.
        @param fibers Liste actuelle des fibres.
        @return Liste des nouvelles fibres acceptées.
        """
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