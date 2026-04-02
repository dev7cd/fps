"""
visualization/plotter.py
Outils de visualisation scientifiques.
- Rendu 3D (Wireframe)
- Figures de Pôles (AD-PCA Orientation)
- Histogrammes statistiques
"""

import matplotlib.pyplot as plt
import numpy as np
import logging
from typing import List, Dict, Optional
from mpl_toolkits.mplot3d import Axes3D

# On évite le backend interactif si lancé en headless (serveur)
try:
    import matplotlib
    matplotlib.use('Agg') # Force non-interactive backend for file saving
except:
    pass

logger = logging.getLogger(__name__)

class Visualizer:
    @staticmethod
    def plot_rve(fibers: List, box_dims, filename: Optional[str] = None):
        """Visualisation 3D géométrique simple (Squelettes)."""
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Boite
        lx, ly, lz = box_dims
        # Arêtes box
        lines = [
            ([0,lx], [0,0], [0,0]), ([0,lx], [ly,ly], [0,0]),
            ([0,lx], [0,0], [lz,lz]), ([0,lx], [ly,ly], [lz,lz]),
            ([0,0], [0,ly], [0,0]), ([lx,lx], [0,ly], [0,0]),
            ([0,0], [0,ly], [lz,lz]), ([lx,lx], [0,ly], [lz,lz]),
            ([0,0], [0,0], [0,lz]), ([lx,lx], [0,0], [0,lz]),
            ([0,0], [ly,ly], [0,lz]), ([lx,lx], [ly,ly], [0,lz])
        ]
        for x,y,z in lines:
            ax.plot(x,y,z, 'k-', lw=0.5, alpha=0.3)

        # Fibres
        for f in fibers:
            pts = f.centerline
            if len(pts) < 2: continue
            
            style = '--' if f.is_ghost else '-'
            alpha = 0.4 if f.is_ghost else 1.0
            color = 'gray' if f.is_ghost else None
            
            # Couleur orientation (si non ghost)
            if not f.is_ghost:
                vec = pts[-1] - pts[0]
                n = np.linalg.norm(vec)
                color = np.abs(vec/n) if n>0 else (0,0,1)
                
            ax.plot(pts[:,0], pts[:,1], pts[:,2], 
                   linestyle=style, color=color, alpha=alpha, lw=1.5)

        ax.set_aspect('equal')
        ax.set_axis_off()
        
        if filename:
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            logger.info(f"Rendu 3D sauvé : {filename}")
        plt.close(fig)

    @staticmethod
    def plot_analysis_report(stats_results: Dict, output_prefix: str):
        """
        Génère une planche de graphiques pour l'analyse scientifique (AD-PCA).
        - Distribution des Tortuosités.
        - Figure de pôles (Orientation Axiale et Planaire).
        """
        if not stats_results: return
        
        logger.info(f"Génération des graphiques d'analyse vers {output_prefix}_*.png ...")
        
        # Récupération des données brutes stockées temporairement dans le dictionnaire 
        # (Nécessite que RVEAnalyzer passe les listes brutes, pas juste les moyennes)
        # S'ils ne sont pas dans stats_results (JSON-ready), on ne peut pas plotter.
        # Idéalement, RVEAnalyzer doit retourner les arrays 'taus', 'u_vecs', 'v_vecs'.
        
        # Pour simplifier ici, on visualise ce qu'on a : le tenseur Global.
        
        # --- GRAPH 1 : Tenseurs d'Orientation (Heatmap) ---
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        
        # Axial Tensor
        ax_mat = np.array(stats_results['ad_pca']['tensors']['A_axial'])
        im1 = axes[0].imshow(ax_mat, cmap='viridis', vmin=0, vmax=1)
        axes[0].set_title(f"Tenseur Axial ($f_a={stats_results['ad_pca']['anisotropy_spectrum']['f_axial']:.2f}$)")
        for (i, j), z in np.ndenumerate(ax_mat):
            axes[0].text(j, i, f'{z:.2f}', ha='center', va='center', color='w' if z<0.7 else 'k')
        
        # Planar Tensor
        pl_mat = np.array(stats_results['ad_pca']['tensors']['A_planar'])
        im2 = axes[1].imshow(pl_mat, cmap='plasma', vmin=0, vmax=1)
        axes[1].set_title(f"Tenseur Planaire ($f_p={stats_results['ad_pca']['anisotropy_spectrum']['f_planar']:.2f}$)")
        for (i, j), z in np.ndenumerate(pl_mat):
            axes[1].text(j, i, f'{z:.2f}', ha='center', va='center', color='w' if z<0.7 else 'k')
            
        plt.colorbar(im1, ax=axes[0])
        plt.colorbar(im2, ax=axes[1])
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_tensors.png")
        plt.close()

        # --- GRAPH 2 : Ellipsoïde d'orientation (Visualisation 3D du tenseur) ---
        # On visualise l'ellipsoide représentant A_axial
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # Eigen decomposition
        evals, evecs = np.linalg.eigh(ax_mat)
        
        # Sphère unitaire déformée
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 30)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        
        # Transformation
        for i in range(len(x)):
            for j in range(len(x)):
                [x[i,j], y[i,j], z[i,j]] = np.dot(evecs, [x[i,j]*evals[0], y[i,j]*evals[1], z[i,j]*evals[2]])

        # Plot
        ax.plot_surface(x, y, z, color='b', alpha=0.3, edgecolor='k')
        
        # Axes principaux
        origin = np.zeros(3)
        for i in range(3):
            vec = evecs[:, i] * evals[i] * 1.2
            ax.quiver(*origin, *vec, color=['r','g','b'][i], lw=2)
            
        ax.set_title("Ellipsoïde d'Orientation Axiale")
        ax.set_aspect('equal')
        plt.savefig(f"{output_prefix}_ellipsoid.png")
        plt.close()

def plot_rve(fibers, dims, filename=None):
    Visualizer.plot_rve(fibers, dims, filename)

def plot_analysis(stats, prefix):
    Visualizer.plot_analysis_report(stats, prefix)