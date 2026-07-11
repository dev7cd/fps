## @file descriptors.py
#  @brief Implementation of the AD-PCA framework and advanced geometric descriptors.
#
#  Provides tools for Anisotropic Decomposition via PCA (AD-PCA) to analyze 
#  curved fiber orientations and microstructural tortuosity.


import numpy as np
import logging
from typing import List, Dict, Tuple, Any
from core.fiber import Fiber

## @var logger
#  @brief Logger for the descriptors module.
logger = logging.getLogger(__name__)


class AD_PCA_Analyzer:
    """!
    @brief Microstructural analyzer using the AD-PCA framework.
    @details Implements the EF-RSA method for decomposing curved fiber 
    trajectories into axial and planar orientation tensors.
    """
    def __init__(self, fibers: List[Fiber]):
        """!
        @brief Initializes the AD-PCA analyzer.
        @param fibers List List of Fiber objects to analyze.
        """
        self.fibers = fibers  ##< List of Fiber objects.
        self.num_fibers = len(fibers)  ##< Number of fibers.
        
        # Cached results
        self.fiber_metrics = []  ##< List storing (tau, B_k, w_k, axes) for each fiber.
        self.A_axial = None  ##< Axial orientation tensor.
        self.A_planar = None  ##< Planar orientation tensor.
        
    def compute_all(self) -> None:
        """!
        @brief Executes the full AD-PCA analysis pipeline.
        @details Performs individual fiber PCA followed by global tensor construction.
        @return None
        """
        if self.num_fibers == 0: return

        # 1. Individual analysis (geometric descriptor & PCA per fiber)
        for fib in self.fibers:
            res = self._analyze_single_fiber(fib)
            if res:
                self.fiber_metrics.append(res)

        # 2. Construction of the global tensors
        self._compute_global_tensors()
        
    def _analyze_single_fiber(self, fib: Fiber) -> Dict:
        """!
        @brief Calculates tortuosity and local PCA for a single fiber.
        @param fib Fiber The Fiber object to analyze.
        @return Dict Dictionary containing tortuosity (tau), biaxiality (B_k), efficiency (w_k), and principal axes (u, v, w).
        """
        pts = fib.centerline # (M, 3) dense discrete points
        if len(pts) < 2: return None

        # --- A. Geometric Descriptors ---
        # 1. Curvilinear Length Lk
        # Sum of the segments
        vecs = pts[1:] - pts[:-1]
        L_k = np.sum(np.linalg.norm(vecs, axis=1))
        
        # 2. Chord Vector Length |pk|
        p_k_vec = pts[-1] - pts[0]
        chord_len = np.linalg.norm(p_k_vec)
        
        # 3. Tortuosity tau (EQ 1)
        if chord_len < 1e-9: tau = 1.0 # Degenerate case (loop closed?) or dot
        else: tau = L_k / chord_len
        
        # --- B. AD-PCA Core ---
        # 1. Centroid
        centroid = np.mean(pts, axis=0)
        
        # 2. Covariance Matrix C_k (EQ 2)
        # Centering
        centered_pts = pts - centroid # (M, 3)
        # C_k = (X.T @ X) / M
        M_k = len(pts)
        C_k = (centered_pts.T @ centered_pts) / M_k
        
        # 3. Eigen Decomposition
        eigvals, eigvecs = np.linalg.eigh(C_k) 
        # eigh returns sorted ascending, we need descending sigma1 >= sigma2 >= sigma3
        # Indices: [0,1,2] -> We want [2, 1, 0]
        
        sigma_1 = eigvals[2]
        sigma_2 = eigvals[1]
        sigma_3 = eigvals[0]
        
        u_k = eigvecs[:, 2] # Major Principal Axis (axial)
        v_k = eigvecs[:, 1] # Principal Bending Axis
        w_n_k = eigvecs[:, 0] # Normal to Plane
        
        # --- C. Biaxiality & Weights ---
        # B_k (EQ 3)
        B_k = sigma_3 / sigma_2 if sigma_2 > 1e-12 else 0.0
        # Ensure [0, 1] range (numerical noise can do funny things)
        B_k = max(0.0, min(1.0, B_k))
        
        # w_k = 1/tau (Efficiency)
        w_k = 1.0 / tau if tau > 0 else 0.0
        
        return {
            'tau': tau,
            'B_k': B_k,
            'w_k': w_k,
            'u': u_k,
            'v': v_k,
            'w': w_n_k
        }

    def _compute_global_tensors(self) -> None:
        """!
        @brief Constructs the global Axial and Planar orientation tensors.
        @details Normalizes the accumulated dyadic products by efficiency and biaxiality weights.
        @return None
        """
        sum_w = 0.0
        sum_B = 0.0
        
        A_ax_accum = np.zeros((3,3))
        A_pl_accum = np.zeros((3,3))
        
        for m in self.fiber_metrics:
            u = m['u']
            v = m['v']
            w_k = m['w_k']
            B_k = m['B_k']
            
            # Dyadic product: outer(u, u)
            A_ax_accum += w_k * np.outer(u, u)
            sum_w += w_k
            
            A_pl_accum += B_k * np.outer(v, v)
            sum_B += B_k
            
        # Normalization
        if sum_w > 0: self.A_axial = A_ax_accum / sum_w
        else: self.A_axial = np.eye(3)/3.0 # Fallback isotrope
            
        if sum_B > 0: self.A_planar = A_pl_accum / sum_B
        else: self.A_planar = np.eye(3)/3.0

    def get_spectrum(self) -> Dict:
        """!
        @brief Computes the Anisotropy Spectrum and Herman's orientation factors.
        @return Dict Dictionary containing f_axial, f_planar, and the full tensors.
        """
        # Eigenvalues of global tensors
        eig_ax, _ = np.linalg.eigh(self.A_axial)
        eig_pl, _ = np.linalg.eigh(self.A_planar)
        
        # Principal (Max) Eigenvalue
        lambda_max_ax = np.max(eig_ax)
        lambda_max_pl = np.max(eig_pl)
        
        # Herman's Orientation Factor (f = 3/2 * (lambda - 1/3))
        f_axial = 1.5 * (lambda_max_ax - (1.0/3.0))
        f_planar = 1.5 * (lambda_max_pl - (1.0/3.0))
        
        return {
            'f_axial': f_axial,
            'f_planar': f_planar,
            'A_axial': self.A_axial,
            'A_planar': self.A_planar
        }

    def compute_chord_tensor(self) -> Dict[str, Any]:
        """!
        @brief Computes the standard orientation tensor based on the end-to-end chord vector.
        @details Used for benchmarking AD-PCA against traditional linear orientation metrics.
        @return Dict[str, Any] Dictionary with the chord tensor and its Herman's factor.
        """
        chord_accum = np.zeros((3, 3))
        count = 0
        
        for fib in self.fibers:
            pts = fib.centerline
            if len(pts) < 2: continue
            
            # Normalised chord vector (p_end - p_start)
            vec = pts[-1] - pts[0]
            norm = np.linalg.norm(vec)
            if norm > 1e-9:
                u = vec / norm
                # Dyadic product u (x) u
                chord_accum += np.outer(u, u)
                count += 1
                
        if count == 0: return {}
        
        A_chord = chord_accum / count
        
        # Compute Hermans' factor
        evals, _ = np.linalg.eigh(A_chord)
        f_chord = 1.5 * (np.max(evals) - (1/3))
        
        return {
            "A_chord": A_chord.tolist(),
            "f_chord": f_chord
        }


class MicroDescriptor:
    """!
    @brief Utility class for basic geometric statistics of the fiber population.
    """
    def __init__(self, fibers: List[Fiber]):
        """!
        @brief Initializes the descriptor.
        @param fibers List List of Fiber objects.
        """
        self.fibers = fibers  ##< List of Fiber objects.

    def compute_geometric_stats(self) -> Dict[str, Any]:
        """!
        @brief Analyzes basic geometric statistics.
        @details Calculates mean and standard deviation for tortuosity, length, and biaxiality.
        @return Dict[str, Any] Dictionary of statistical distributions.
        """
        taus = []
        lengths = []
        biaxiality = []
        
        for f in self.fibers:
            pts = f.centerline
            # Curvilinear length
            L_k = np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))
            # End-to-end length
            chord = np.linalg.norm(pts[-1] - pts[0])
            
            tau = L_k / chord if chord > 1e-9 else 1.0
            taus.append(tau)
            lengths.append(L_k)
            
            # Shape analysis via local PCA (C_k)
            centroid = np.mean(pts, axis=0)
            C_k = np.cov(pts.T)
            eigvals = np.sort(np.linalg.eigvals(C_k))[::-1] # Descending

            # B_k: flattening ratio of the curved fiber (0 = straight, 1 = helix/circle)
            bk = eigvals[2] / eigvals[1] if eigvals[1] > 1e-12 else 0.0
            biaxiality.append(bk)

        return {
            "tortuosity": {"mean": np.mean(taus), "std": np.std(taus), "all": taus},
            "length": {"mean": np.mean(lengths), "total": np.sum(lengths)},
            "biaxiality": {"mean": np.mean(biaxiality), "std": np.std(biaxiality)}
        }
