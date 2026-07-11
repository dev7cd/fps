"""
generation/generator.py
Générateur "CSAW" Optimisé v6.1
Correctif Critique : Récupération correcte des vecteurs de biais depuis la config v6.
"""

import numpy as np
import logging
import copy
from typing import Optional, List, Tuple
from core.fiber import Fiber
from generation.periodicity import PeriodicManager

## @var logger
#  @brief Logger pour le module de génération.
logger = logging.getLogger(__name__)

class FiberGenerator:
    """
    @class FiberGenerator
    @brief Optimized fiber generator using the CSAW (Constrained Self-Avoiding Walk) algorithm.
    """
    def __init__(self, config, detector):
        """! @brief Placeholder.
        @param self, config, detector 
        @return None
        """
        """
        @brief Initializes the generator with configuration and collision detector.
        @param config FiberPackingConfig Configuration object.
        @param detector CollisionDetector Detector for checking overlaps.
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
        
        # --- CORRECTION CRITIQUE DU CHARGEMENT DU BIAIS ---
        gen_params = config.generation_parameters
        
        # 1. On essaie de récupérer les vecteurs explicites (Nouvelle norme config v6)
        raw_vectors = gen_params.get('bias_vectors')
        
        # 2. Fallback : Si 'bias_vectors' est vide, on regarde 'orientation_bias' 
        # (Supporte les shortcuts 'x', 'y', 'z' de l'ancienne version)
        if raw_vectors is None:
            raw_vectors = gen_params.get('orientation_bias')

        ## @var bias_vectors
        #  @brief Liste des vecteurs de direction privilégiés.
        self.bias_vectors = self._initialize_bias(raw_vectors)
        ## @var bias_strength
        #  @brief Force de l'attraction vers les vecteurs de biais (0.0 à 1.0).
        self.bias_strength = gen_params.get('bias_strength', 0.0)
        
        # Debug Log pour confirmer le biais
        if self.bias_vectors:
            logger.debug(f"Biais activé. {len(self.bias_vectors)} vecteurs. Force={self.bias_strength}")
        else:
            logger.debug("Aucun biais détecté (Isotrope).")

    def generate_fiber(self, fiber_id: int) -> Optional[Fiber]:
        """
        @brief Main method: Snake Algorithm with Backtracking.
        @param fiber_id int Unique identifier for the new fiber.
        @return Optional[Fiber] A Fiber object if successful, None otherwise.
        @details Uses a stochastic walk with curvature constraints and collision checks.
        """
        radius = self.config.fiber_radius
        
        p0 = self._find_start_point(radius)
        if p0 is None:
            return None

        control_points = [p0]
        n_retry_max = 50
        n_fail = 0
        current_dir = self._initial_direction()

        while len(control_points) < self.config.max_control_points:
            p_cur = control_points[-1]
            
            p_next, new_dir = self._propose_next_point(p_cur, current_dir)
            
            # Collision & Auto-intersection
            if self.detector.is_segment_free(p_cur, p_next, radius):
                if not self._check_self_collision(p_next, control_points, radius):
                    control_points.append(p_next)
                    current_dir = new_dir
                    n_fail = 0
                    continue
            
            n_fail += 1
            if n_fail > n_retry_max:
                if len(control_points) > 1:
                    control_points.pop()
                    n_fail = 0
                    # Recalcul de direction post-backtrack
                    if len(control_points) > 1:
                        v = control_points[-1] - control_points[-2]
                        norm = np.linalg.norm(v)
                        current_dir = v / norm if norm > 1e-9 else self._initial_direction()
                else:
                    return None
                    
        return Fiber(fiber_id, np.array(control_points), radius, vars(self.config))

    def _initialize_bias(self, bias_input):
        """! @brief Placeholder.
        @param self, bias_input 
        @return None
        """
        """
        @brief Robust conversion of Config inputs to a list of Numpy Arrays.
        @param bias_input Raw input from configuration (string, list, or array).
        @return List of normalized direction vectors or None.
        """
        if bias_input is None or bias_input == 'free':
            return None
            
        # Shortcuts string (legacy)
        mapping = {'x': [1,0,0], 'y': [0,1,0], 'z': [0,0,1]}
        if isinstance(bias_input, str):
            if bias_input in mapping:
                return [np.array(mapping[bias_input], dtype=float)]
            return None # "uniaxial" seul sans vecteur est ignoré ici car traité en amont par CLI

        # Conversion Liste/Array
        try:
            data = np.array(bias_input, dtype=float)
        except:
            return None

        # Cas 1D : [1, 0, 0]
        if data.ndim == 1 and data.size == 3:
            n = np.linalg.norm(data)
            return [data/n] if n > 1e-9 else None
            
        # Cas 2D : [[1,0,0]] ou [[1,0,0], [0,1,0]]
        if data.ndim == 2 and data.shape[1] == 3:
            vectors = []
            for v in data:
                n = np.linalg.norm(v)
                if n > 1e-9:
                    vectors.append(v/n)
            
            # Orthogonalisation si planaire (pour propreté mathématique)
            if len(vectors) == 2:
                v1, v2 = vectors
                v2_ortho = v2 - np.dot(v2, v1) * v1
                n_o = np.linalg.norm(v2_ortho)
                if n_o > 1e-9:
                    vectors[1] = v2_ortho / n_o
                else:
                    return [v1] # Repli uniaxial si colinéaire
            return vectors if vectors else None

        return None

    def _get_target_direction(self, current_dir):
        """! @brief Placeholder.
        @param self, current_dir 
        @return None
        """
        """
        @brief Calculates the attractor vector based on orientation bias.
        @param current_dir The current movement vector.
        @return Normalized target direction vector.
        """
        if not self.bias_vectors:
            return current_dir

        if len(self.bias_vectors) == 1:
            target = self.bias_vectors[0]
            # On s'aligne dans le sens de la marche
            if np.dot(target, current_dir) < 0:
                target = -target
        else:
            # Projection Planaire
            v1, v2 = self.bias_vectors
            proj = np.dot(current_dir, v1)*v1 + np.dot(current_dir, v2)*v2
            n_p = np.linalg.norm(proj)
            target = proj/n_p if n_p > 1e-9 else v1

        # Mélange Fort (Style v5) : Inertie + Biais
        combined = (1.0 - self.bias_strength) * current_dir + self.bias_strength * target
        n_c = np.linalg.norm(combined)
        return combined / n_c if n_c > 1e-9 else current_dir

    def _propose_next_point(self, p_cur, current_dir):
        """! @brief Placeholder.
        @param self, p_cur, current_dir 
        @return None
        """
        """
        @brief Proposes the next control point using inertia, bias, and noise.
        @param p_cur Current last control point.
        @param current_dir Current direction of the fiber.
        @return Tuple of (next_point, next_direction).
        @details Implements Rodrigues' rotation formula for clamping curvature.
        """
        target_dir = self._get_target_direction(current_dir)
        max_angle = self.config.generation_parameters.get('max_curvature_angle', np.pi/6)
        
        best_dir = target_dir
        
        # On tente de perturber (bruit) pour la tortuosité
        for _ in range(10):
            # Le bruit est proportionnel à l'angle max permis
            noise = self.rng.normal(0, max_angle * 0.33, 3) 
            trial = target_dir + noise
            norm = np.linalg.norm(trial)
            
            if norm > 1e-9:
                trial /= norm
                # Contrainte cinématique (continuité) : angle vs PREV_DIR
                if np.dot(trial, current_dir) >= np.cos(max_angle):
                    best_dir = trial
                    break
        
        # Si le bruit nous a sorti de la contrainte, on clamp proprement target_dir
        if best_dir is target_dir: # Référence identique si boucle échouée
             cos_theta = np.dot(target_dir, current_dir)
             if cos_theta < np.cos(max_angle):
                 # Rotation de Rodrigues (Clamp) vers le target
                 axis = np.cross(current_dir, target_dir)
                 n_ax = np.linalg.norm(axis)
                 if n_ax > 1e-9:
                     k = axis / n_ax
                     # Rotation de max_angle autour de k
                     c, s = np.cos(max_angle), np.sin(max_angle)
                     best_dir = current_dir*c + np.cross(k, current_dir)*s + k*np.dot(k, current_dir)*(1-c)

        return p_cur + best_dir * self.config.step_length_mean, best_dir

    def _find_start_point(self, radius, max_attempts=1000):
        """! @brief Placeholder.
        @param self, radius, max_attempts=1000 
        @return None
        """
        """
        @brief Finds a random collision-free starting point in the domain.
        @param radius Radius of the fiber.
        @param max_attempts Maximum number of random trials.
        """
        for _ in range(max_attempts):
            p = self.rng.uniform([0,0,0], self.config.box_dims)
            if self.detector.is_point_free(p, radius): return p
        return None

    def _initial_direction(self):
        """! @brief Placeholder.
        @param self 
        @return None
        """
        """
        @brief Generates an initial direction vector.
        @details Forces alignment with bias if bias_strength is high.
        """
        if self.bias_strength > 0.8 and self.bias_vectors:
            # On part déjà un peu dans la bonne direction (+ bruit)
            # Sinon on perd du temps à tourner dès le 1er segment
            target = self.bias_vectors[0]
            v = target + self.rng.normal(0, 0.2, 3)
        else:
            v = self.rng.normal(0, 1, 3)
        return v / np.linalg.norm(v)

    def _check_self_collision(self, p_next, control_points, radius):
        """! @brief Placeholder.
        @param self, p_next, control_points, radius 
        @return None
        """
        """
        @brief Checks if the new point collides with the fiber's own body.
        @param p_next Proposed point.
        @param control_points Existing points in the fiber.
        @param radius Fiber radius.
        """
        if len(control_points) < 5: return False
        pts = np.array(control_points[:-3])
        dists_sq = np.sum((pts - p_next)**2, axis=1)
        return np.any(dists_sq < (radius * 2.1)**2)

    # --- RSDA : Réarrangement Dynamique pendant le placement ---

    def attempt_rsda_placement(self, fiber_id: int, fibers: List) -> Optional[Fiber]:
        """
        @brief RSDA (Randomized Sequential Dynamic Adsorption) placement.
        @param fiber_id int ID for the new fiber.
        @param fibers List List of existing fibers in the domain.
        @return Optional[Fiber] Fiber if placement succeeded after perturbation, None otherwise.
        @details When standard placement fails, this method perturbs neighboring 
        fibers to create space. It includes a rollback mechanism if a valid 
        configuration isn't found.
        """
        if not getattr(self.config, 'enable_rsda', False):
            return None

        radius = self.config.fiber_radius
        perturb_ratio = getattr(self.config, 'rsda_perturbation_radius', 0.05)
        max_neighbors = getattr(self.config, 'rsda_max_neighbors', 5)

        # Étape 1 : Générer un candidat
        candidate = self.generate_fiber(fiber_id)
        if candidate is None:
            return None

        # Étape 2 : Créer le groupe (candidat + ghosts)
        ghost_shifts = PeriodicManager.generate_ghosts(candidate, self.config.box_dims)
        candidate_group = [candidate]
        for s in ghost_shifts:
            g_pts = candidate.control_points + s
            g = Fiber(candidate.id, g_pts, candidate.radius, vars(self.config),
                      is_ghost=True, parent_id=candidate.id)
            candidate_group.append(g)

        # Étape 3 : Identifier les voisins en collision
        colliders = set()
        for fib in candidate_group:
            neighbors = self.detector.grid.query_neighbors(fib.bbox, fib.parent_id)
            for nb in neighbors:
                if self.detector.check_collision_fine(fib, nb):
                    pid = nb.parent_id
                    # Trouver la fibre racine correspondante
                    for f in fibers:
                        if f.id == pid:
                            colliders.add(id(f))
                            break

        if not colliders or len(colliders) > max_neighbors:
            return None

        # Étape 4 : Sauvegarder et perturber les voisins
        collider_fibers = [f for f in fibers if id(f) in colliders]
        saved_states = {}
        for f in collider_fibers:
            saved_states[f.id] = f.control_points.copy()

        perturb_amount = perturb_ratio * radius
        success = False

        for attempt in range(10):
            # Perturber chaque voisin en collision
            for f in collider_fibers:
                shift = self.rng.uniform(-perturb_amount, perturb_amount, f.control_points.shape)
                f.control_points = saved_states[f.id] + shift * (attempt + 1)
                f.refresh_geometry()

            # Retirer les voisins perturbés de la grille et les ré-ajouter
            for f in collider_fibers:
                self.detector.grid.remove_fiber(f.id)

            # Recréer les ghosts des voisins perturbés et ré-indexer
            perturbed_groups = {}
            for f in collider_fibers:
                shifts = PeriodicManager.generate_ghosts(f, self.config.box_dims)
                group = [f]
                for s in shifts:
                    g_pts = f.control_points + s
                    g = Fiber(f.id, g_pts, f.radius, vars(self.config),
                              is_ghost=True, parent_id=f.id)
                    group.append(g)
                perturbed_groups[f.id] = group

            # Ré-indexer les voisins perturbés
            for fid, group in perturbed_groups.items():
                for fib in group:
                    self.detector.grid.add_fiber(fib)

            # Valider : les voisins perturbés ne se collisionnent pas entre eux
            all_valid = True
            for fid, group in perturbed_groups.items():
                if not self.detector.is_group_valid(group):
                    all_valid = False
                    break

            # Valider : le candidat ne collisionne plus avec personne
            if all_valid and self.detector.is_group_valid(candidate_group):
                success = True
                break

            # Rollback de la grille pour réessayer
            for f in collider_fibers:
                self.detector.grid.remove_fiber(f.id)
            for f in collider_fibers:
                f.control_points = saved_states[f.id].copy()
                f.refresh_geometry()
            for fid in saved_states:
                for f in collider_fibers:
                    if f.id == fid:
                        shifts = PeriodicManager.generate_ghosts(f, self.config.box_dims)
                        group = [f]
                        for s in shifts:
                            g_pts = f.control_points + s
                            g = Fiber(f.id, g_pts, f.radius, vars(self.config),
                                      is_ghost=True, parent_id=f.id)
                            group.append(g)
                        for fib in group:
                            self.detector.grid.add_fiber(fib)

        if not success:
            # Rollback final complet
            for f in collider_fibers:
                self.detector.grid.remove_fiber(f.id)
                f.control_points = saved_states[f.id]
                f.refresh_geometry()
                shifts = PeriodicManager.generate_ghosts(f, self.config.box_dims)
                group = [f]
                for s in shifts:
                    g_pts = f.control_points + s
                    g = Fiber(f.id, g_pts, f.radius, vars(self.config),
                              is_ghost=True, parent_id=f.id)
                    group.append(g)
                for fib in group:
                    self.detector.grid.add_fiber(fib)
            return None

        return candidate