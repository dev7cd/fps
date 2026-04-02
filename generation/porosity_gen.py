"""
generation/porosity_gen.py
Générateur de porosité optimisé.
"""

import numpy as np
import logging
from typing import List, Tuple, Optional
from core.config import FiberPackingConfig
from core.void import Void
from core.fiber import Fiber

logger = logging.getLogger(__name__)

class PorosityGenerator:
    """!
    @class PorosityGenerator
    @brief Generates spherical void inclusions within the RVE.
    @details Implements an optimized RSA (Random Sequential Adsorption) algorithm 
    with periodic boundary conditions and collision detection against fibers.
    """
    
    def __init__(self, config: FiberPackingConfig):
        """!
        @brief Initializes the porosity generator.
        @param config Configuration object containing domain and porosity parameters.
        """
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.box_dims = np.array(config.box_dims)
        
    def generate_voids(self, existing_fibers: List[Fiber]) -> List[Void]:
        """!
        @brief Main generation loop using an optimized RSA algorithm.
        @param existing_fibers List of fibers already present in the domain.
        @return A list of generated Void objects.
        """
        if not self.config.generate_porosity:
            return []
            
        target_vf = self.config.target_void_fraction
        box_vol = self.config.box_volume
        target_vol = box_vol * target_vf
        
        current_vol = 0.0
        voids: List[Void] = []
        void_id_counter = 0
        
        logger.info(f"Génération porosité : Cible {target_vf*100:.1f}% ({target_vol:.4e} m3)")
        
        # Configuration
        mean_r = self.config.void_radius_mean
        std_r = self.config.void_radius_std
        
        # Pré-calcul des données des fibres pour optimisation
        fiber_data = []
        for fib in existing_fibers:
            # Utilise directement la centerline de la fibre
            centerline = fib.centerline
            radius = fib.radius
            
            # Calcule BBox pour optimisation
            min_pt = np.min(centerline, axis=0) - radius
            max_pt = np.max(centerline, axis=0) + radius
            
            fiber_data.append({
                'centerline': centerline,
                'radius': radius,
                'bbox_min': min_pt,
                'bbox_max': max_pt
            })
        
        # Paramètres de l'algorithme
        max_attempts = 30000
        attempts = 0
        consecutive_failures = 0
        
        while current_vol < target_vol and attempts < max_attempts:
            attempts += 1
            
            # 1. Génération Rayon adaptative
            if consecutive_failures > 50:
                # Réduction progressive du rayon après échecs
                radius_factor = max(0.3, 1.0 - consecutive_failures / 200.0)
                radius = self.rng.normal(mean_r * radius_factor, std_r * radius_factor)
            else:
                radius = self.rng.normal(mean_r, std_r)
            
            radius = max(radius, mean_r * 0.2)  # Borne inférieure
            
            # 2. Génération Position
            center = self.rng.uniform(radius, self.box_dims - radius)
            
            # 3. Gestion Périodicité optimisée
            candidates = self._create_periodic_images_fast(center, radius, void_id_counter)
            
            if not candidates:
                consecutive_failures += 1
                continue
                
            # 4. Vérification Collisions optimisée
            is_valid = True
            
            # A. Test Void-Void (vectorisé)
            if voids and not self._check_void_void_collision_fast(candidates, voids):
                is_valid = False
            
            # B. Test Void-Fibre (seulement si Void-Void OK)
            if is_valid and fiber_data:
                if not self._check_void_fiber_collision_fast(candidates, fiber_data):
                    is_valid = False
            
            # 5. Enregistrement
            if is_valid:
                voids.extend(candidates)
                vol_added = (4/3) * np.pi * (radius**3)
                current_vol += vol_added
                void_id_counter += 1
                consecutive_failures = 0
                
                if len(voids) % 100 == 0:
                    logger.debug(f"Porosité : {current_vol/box_vol:.2%} ({len(voids)} voids)...")
            else:
                consecutive_failures += 1
                
        if attempts >= max_attempts:
            logger.warning(f"Porosité : Limite tentatives atteinte. Vf final : {current_vol/box_vol:.2%}")
            
        logger.info(f"Porosité générée : {len(voids)} voids, Vf = {current_vol/box_vol:.2%}")
        return voids

    def _create_periodic_images_fast(self, center: np.ndarray, radius: float, uid: int) -> List[Void]:
        """!
        @brief Creates the primary void and its periodic replicas (ghosts).
        @param center Center coordinates of the primary void.
        @param radius Radius of the void.
        @param uid Unique identifier for the void group.
        @return List of Void objects (primary + periodic images).
        """
        Lx, Ly, Lz = self.box_dims
        images = []
        
        # Détermine quels décalages sont nécessaires
        shifts_x = [0]
        if center[0] - radius < 0: shifts_x.append(1)
        if center[0] + radius > Lx: shifts_x.append(-1)
        
        shifts_y = [0]
        if center[1] - radius < 0: shifts_y.append(1)
        if center[1] + radius > Ly: shifts_y.append(-1)
        
        shifts_z = [0]
        if center[2] - radius < 0: shifts_z.append(1)
        if center[2] + radius > Lz: shifts_z.append(-1)
        
        # Génération combinatoire
        shift_vectors = []
        for dx in shifts_x:
            for dy in shifts_y:
                for dz in shifts_z:
                    shift_vec = np.array([dx*Lx, dy*Ly, dz*Lz])
                    shift_vectors.append(shift_vec)
        
        # Vérification d'overlap vectorisée
        for shift_vec in shift_vectors:
            img_center = center + shift_vec
            
            # Vérification rapide d'overlap
            min_s = img_center - radius
            max_s = img_center + radius
            
            # Overlap avec la boîte [0, L]
            overlap = (
                min_s[0] < Lx and max_s[0] > 0 and
                min_s[1] < Ly and max_s[1] > 0 and
                min_s[2] < Lz and max_s[2] > 0
            )
            
            if overlap:
                v = Void(id=uid, center=img_center, radius=radius)
                images.append(v)
                        
        return images

    def _check_void_void_collision_fast(self, candidates: List[Void], existing_voids: List[Void]) -> bool:
        """!
        @brief Vectorized check for collisions between new candidates and existing voids.
        @param candidates List of new void images to check.
        @param existing_voids List of voids already accepted in the domain.
        @return True if no collision is detected, False otherwise.
        """
        if not existing_voids:
            return True
            
        # Extraction des arrays pour vectorisation
        cand_centers = np.array([v.center for v in candidates])  # (M, 3)
        cand_radii = np.array([v.radius for v in candidates])    # (M,)
        
        exist_centers = np.array([v.center for v in existing_voids])  # (N, 3)
        exist_radii = np.array([v.radius for v in existing_voids])    # (N,)
        
        # Calcul vectorisé des distances et seuils
        # (M, 1, 3) - (1, N, 3) -> (M, N, 3)
        diff = cand_centers[:, np.newaxis, :] - exist_centers[np.newaxis, :, :]
        dists_sq = np.sum(diff * diff, axis=2)  # (M, N)
        
        # Seuils de collision (r1 + r2)^2
        radii_sum = cand_radii[:, np.newaxis] + exist_radii[np.newaxis, :]  # (M, N)
        thresholds = radii_sum * radii_sum  # (M, N)
        
        # Vérification : aucune collision
        return not np.any(dists_sq < thresholds)

    def _check_void_fiber_collision_fast(self, candidates: List[Void], fiber_data: List[dict]) -> bool:
        """!
        @brief Optimized check for collisions between voids and fibers.
        @param candidates List of void images to check.
        @param fiber_data Pre-processed fiber geometric data (centerlines and BBoxes).
        @return True if no collision is detected, False otherwise.
        """
        for void in candidates:
            void_r = void.radius
            void_c = void.center
            void_r_sq = void_r * void_r
            
            # BBox du void pour optimisation
            void_min = void_c - void_r
            void_max = void_c + void_r
            
            for fib_data in fiber_data:
                # Test BBox rapide
                fib_min = fib_data['bbox_min']
                fib_max = fib_data['bbox_max']
                
                if (void_max[0] < fib_min[0] or void_min[0] > fib_max[0] or
                    void_max[1] < fib_min[1] or void_min[1] > fib_max[1] or
                    void_max[2] < fib_min[2] or void_min[2] > fib_max[2]):
                    continue  # Pas d'intersection BBox
                
                # Vérification précise
                fib_pts = fib_data['centerline']
                fib_r = fib_data['radius']
                min_dist_sq = (void_r + fib_r) ** 2
                
                # Calcul vectorisé de la distance au carré
                diff = fib_pts - void_c
                dists_sq = np.sum(diff * diff, axis=1)
                
                if np.any(dists_sq < min_dist_sq):
                    return False  # Collision détectée
                        
        return True

    # --- Méthodes de compatibilité (gardées pour interface) ---
    
    def _check_void_void_collision(self, candidates: List[Void], existing_voids: List[Void]) -> bool:
        """!
        @brief Compatibility wrapper for void-void collision check.
        @param candidates New void candidates.
        @param existing_voids Existing voids.
        @return Boolean result of the check.
        """
        return self._check_void_void_collision_fast(candidates, existing_voids)
    
    def _check_void_fiber_collision(self, candidates: List[Void], fibers: List[Fiber]) -> bool:
        """!
        @brief Compatibility wrapper for void-fiber collision check.
        @details Converts Fiber objects to optimized dict format before calling the fast checker.
        @param candidates New void candidates.
        @param fibers List of Fiber objects.
        @return Boolean result of the check.
        """
        # Convertit les fibres en format optimisé
        fiber_data = []
        for fib in fibers:
            centerline = fib.centerline
            radius = fib.radius
            min_pt = np.min(centerline, axis=0) - radius
            max_pt = np.max(centerline, axis=0) + radius
            fiber_data.append({
                'centerline': centerline,
                'radius': radius,
                'bbox_min': min_pt,
                'bbox_max': max_pt
            })
        return self._check_void_fiber_collision_fast(candidates, fiber_data)
    
    # --- Méthodes géométriques de base ---
    
    def _sq_dist_point_polyline(self, point: np.ndarray, polyline: np.ndarray) -> float:
        """!
        @brief Calculates the minimum squared distance between a point and a polyline.
        @param point 3D coordinates of the point.
        @param polyline Array of points forming the polyline.
        @return Minimum squared distance.
        """
        # Calcul vectorisé
        diff = polyline - point
        dists_sq = np.sum(diff * diff, axis=1)
        return np.min(dists_sq)
    
    def _sq_dist_point_segment(self, p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        """!
        @brief Calculates the squared distance between a point and a line segment.
        @param p The point.
        @param a Start point of the segment.
        @param b End point of the segment.
        @return Squared distance.
        """
        ab = b - a
        ap = p - a
        
        denom = np.dot(ab, ab)
        if denom < 1e-12:
            return np.dot(ap, ap)
            
        t = np.dot(ap, ab) / denom
        t_clamped = np.clip(t, 0.0, 1.0)
        
        closest = a + t_clamped * ab
        dist_vec = p - closest
        
        return np.dot(dist_vec, dist_vec)