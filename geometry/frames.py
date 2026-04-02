"""
geometry/frames.py
Calcul des repères locaux le long d'une courbe 3D.
Implémente le Transport Parallèle (Bishop Frame) pour minimiser la torsion.
"""

import numpy as np
from typing import Tuple

def compute_tangents(points: np.ndarray) -> np.ndarray:
    """Calcule les vecteurs tangents normalisés (différences finies centrées)."""
    if len(points) < 2:
        raise ValueError("Besoin d'au moins 2 points")
        
    # Gradient (dérivée numérique)
    tangents = np.gradient(points, axis=0)
    
    # Normalisation
    norms = np.linalg.norm(tangents, axis=1)
    # Protection div par zero
    norms[norms < 1e-12] = 1.0 
    
    return tangents / norms[:, np.newaxis]

def compute_bishop_frame(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcule le repère de Bishop (Tangente, Normale, Binormale).
    
    Returns:
        T, N, B : Arrays de forme (N_points, 3)
    """
    n_points = len(points)
    T = compute_tangents(points)
    N = np.zeros_like(T)
    B = np.zeros_like(T)
    
    # --- 1. Initialisation au premier point ---
    # On cherche un vecteur arbitraire non colinéaire à T[0]
    t0 = T[0]
    arbitrary = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(t0, arbitrary)) > 0.9:
        arbitrary = np.array([1.0, 0.0, 0.0])
        
    # Premier vecteur normal via produit vectoriel
    v_perp = np.cross(t0, arbitrary)
    n0 = v_perp / np.linalg.norm(v_perp)
    b0 = np.cross(t0, n0)
    
    N[0] = n0
    B[0] = b0
    
    # --- 2. Transport Parallèle (Propagation) ---
    for i in range(1, n_points):
        t_prev = T[i-1]
        t_curr = T[i]
        n_prev = N[i-1]
        b_prev = B[i-1]
        
        # Axe de rotation pour aligner t_prev sur t_curr
        # C'est le vecteur binormal de courbure locale
        axis = np.cross(t_prev, t_curr)
        len_axis = np.linalg.norm(axis)
        
        if len_axis < 1e-9:
            # Courbe droite, pas de rotation
            N[i] = n_prev
            B[i] = b_prev
        else:
            axis = axis / len_axis
            # Angle de rotation
            # dot = cos(theta)
            cos_theta = np.clip(np.dot(t_prev, t_curr), -1.0, 1.0)
            theta = np.arccos(cos_theta)
            
            # Matrice de rotation autour de 'axis' d'angle 'theta'
            # Formule de Rodrigues simplifiée pour vecteur
            # V_rot = V*cos + (K x V)*sin + K*(K.V)*(1-cos)
            # Ici K = axis
            
            # Rotation de n_prev
            n_new = (n_prev * cos_theta + 
                     np.cross(axis, n_prev) * np.sin(theta) + 
                     axis * np.dot(axis, n_prev) * (1 - cos_theta))
            
            # Rotation de b_prev
            b_new = (b_prev * cos_theta + 
                     np.cross(axis, b_prev) * np.sin(theta) + 
                     axis * np.dot(axis, b_prev) * (1 - cos_theta))
            
            N[i] = n_new / np.linalg.norm(n_new)
            B[i] = b_new / np.linalg.norm(b_new)
            
    return T, N, B