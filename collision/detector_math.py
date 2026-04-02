"""
collision/detector_math.py
Détection de collision par distance segment-à-segment analytique (Eberly/Geometric Tools).
Remplace l'ancienne approche point-à-point par une solution exacte capsule-capsule.
"""

import numpy as np
from numba import njit


@njit(cache=True)
def _segment_segment_dist_sq(Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz, Dx, Dy, Dz):
    """
    Distance² minimale entre deux segments 3D [A,B] et [C,D].
    Algorithme d'Eberly (Geometric Tools) — résolution paramétrique exacte.

    Segments : P(s) = A + s*(B-A),  Q(t) = C + t*(D-C),  s,t ∈ [0,1]
    """
    # Vecteurs directeurs et différence
    d1x = Bx - Ax;  d1y = By - Ay;  d1z = Bz - Az
    d2x = Dx - Cx;  d2y = Dy - Cy;  d2z = Dz - Cz
    rx  = Ax - Cx;   ry = Ay - Cy;   rz = Az - Cz

    a = d1x*d1x + d1y*d1y + d1z*d1z  # |d1|²
    e = d2x*d2x + d2y*d2y + d2z*d2z  # |d2|²
    f = d2x*rx  + d2y*ry  + d2z*rz   # dot(d2, r)

    EPS = 1e-12

    # Cas dégénéré : les deux segments sont des points
    if a <= EPS and e <= EPS:
        return rx*rx + ry*ry + rz*rz

    if a <= EPS:
        # Segment 1 dégénéré en point
        s = 0.0
        t = f / e
        if t < 0.0: t = 0.0
        elif t > 1.0: t = 1.0
    else:
        c = d1x*rx + d1y*ry + d1z*rz  # dot(d1, r)
        if e <= EPS:
            # Segment 2 dégénéré en point
            t = 0.0
            s = -c / a
            if s < 0.0: s = 0.0
            elif s > 1.0: s = 1.0
        else:
            # Cas général : deux segments non-dégénérés
            b = d1x*d2x + d1y*d2y + d1z*d2z  # dot(d1, d2)
            denom = a * e - b * b  # toujours >= 0

            # Si segments non-parallèles, calculer le point le plus proche sur la ligne infinie
            if denom > EPS:
                s = (b * f - c * e) / denom
                if s < 0.0: s = 0.0
                elif s > 1.0: s = 1.0
            else:
                # Segments parallèles : choisir s=0 comme point de départ
                s = 0.0

            # Calculer t à partir de s
            t = (b * s + f) / e

            # Clamper t et recalculer s si nécessaire
            if t < 0.0:
                t = 0.0
                s = -c / a
                if s < 0.0: s = 0.0
                elif s > 1.0: s = 1.0
            elif t > 1.0:
                t = 1.0
                s = (b - c) / a
                if s < 0.0: s = 0.0
                elif s > 1.0: s = 1.0

    # Points les plus proches
    px = Ax + s * d1x - (Cx + t * d2x)
    py = Ay + s * d1y - (Cy + t * d2y)
    pz = Az + s * d1z - (Cz + t * d2z)

    return px*px + py*py + pz*pz


@njit(cache=True)
def check_collision_numba(pts1, pts2, r1, r2):
    """
    Détection de collision segment-à-segment entre deux polylignes.
    Signature identique à l'ancienne version pour compatibilité.

    Args:
        pts1, pts2: Centerlines (N, 3) — chaque paire de points consécutifs forme un segment
        r1, r2: Rayons des fibres
    Returns:
        True si collision (distance min < r1 + r2)
    """
    threshold_sq = (r1 + r2) ** 2
    n1 = len(pts1) - 1  # nombre de segments
    n2 = len(pts2) - 1

    for i in range(n1):
        Ax = pts1[i, 0];    Ay = pts1[i, 1];    Az = pts1[i, 2]
        Bx = pts1[i+1, 0];  By = pts1[i+1, 1];  Bz = pts1[i+1, 2]

        for j in range(n2):
            Cx = pts2[j, 0];    Cy = pts2[j, 1];    Cz = pts2[j, 2]
            Dx = pts2[j+1, 0];  Dy = pts2[j+1, 1];  Dz = pts2[j+1, 2]

            d_sq = _segment_segment_dist_sq(
                Ax, Ay, Az, Bx, By, Bz,
                Cx, Cy, Cz, Dx, Dy, Dz
            )

            if d_sq < threshold_sq:
                return True

    return False


@njit(cache=True)
def min_dist_segments_numba(pts1, pts2):
    """
    Distance minimale exacte entre deux polylignes (segment-à-segment).
    Utilisé par TopologyValidator pour l'audit de clearance.

    Returns:
        Distance minimale (float, non au carré)
    """
    min_sq = 1e20
    n1 = len(pts1) - 1
    n2 = len(pts2) - 1

    for i in range(n1):
        Ax = pts1[i, 0];    Ay = pts1[i, 1];    Az = pts1[i, 2]
        Bx = pts1[i+1, 0];  By = pts1[i+1, 1];  Bz = pts1[i+1, 2]

        for j in range(n2):
            Cx = pts2[j, 0];    Cy = pts2[j, 1];    Cz = pts2[j, 2]
            Dx = pts2[j+1, 0];  Dy = pts2[j+1, 1];  Dz = pts2[j+1, 2]

            d_sq = _segment_segment_dist_sq(
                Ax, Ay, Az, Bx, By, Bz,
                Cx, Cy, Cz, Dx, Dy, Dz
            )

            if d_sq < min_sq:
                min_sq = d_sq

    return np.sqrt(min_sq)


@njit(cache=True)
def min_dist_periodic_segments_numba(pts1, pts2, dims):
    """
    Distance minimale segment-à-segment avec Minimum Image Convention (MIC).
    Pour chaque paire de segments, applique la correction périodique au midpoint
    puis calcule la distance exacte segment-segment sur les segments shiftés.

    Args:
        pts1, pts2: Centerlines (N, 3)
        dims: Dimensions du domaine [Lx, Ly, Lz]
    Returns:
        Distance minimale (float)
    """
    min_sq = 1e20
    Lx = dims[0]; Ly = dims[1]; Lz = dims[2]
    n1 = len(pts1) - 1
    n2 = len(pts2) - 1

    for i in range(n1):
        Ax = pts1[i, 0];    Ay = pts1[i, 1];    Az = pts1[i, 2]
        Bx = pts1[i+1, 0];  By = pts1[i+1, 1];  Bz = pts1[i+1, 2]

        # Midpoint du segment 1
        m1x = 0.5 * (Ax + Bx)
        m1y = 0.5 * (Ay + By)
        m1z = 0.5 * (Az + Bz)

        for j in range(n2):
            Cx = pts2[j, 0];    Cy = pts2[j, 1];    Cz = pts2[j, 2]
            Dx = pts2[j+1, 0];  Dy = pts2[j+1, 1];  Dz = pts2[j+1, 2]

            # Midpoint du segment 2
            m2x = 0.5 * (Cx + Dx)
            m2y = 0.5 * (Cy + Dy)
            m2z = 0.5 * (Cz + Dz)

            # Shift MIC basé sur les midpoints
            shift_x = -Lx * round((m2x - m1x) / Lx)
            shift_y = -Ly * round((m2y - m1y) / Ly)
            shift_z = -Lz * round((m2z - m1z) / Lz)

            # Appliquer le shift au segment 2
            Cx_s = Cx + shift_x;  Cy_s = Cy + shift_y;  Cz_s = Cz + shift_z
            Dx_s = Dx + shift_x;  Dy_s = Dy + shift_y;  Dz_s = Dz + shift_z

            d_sq = _segment_segment_dist_sq(
                Ax, Ay, Az, Bx, By, Bz,
                Cx_s, Cy_s, Cz_s, Dx_s, Dy_s, Dz_s
            )

            if d_sq < min_sq:
                min_sq = d_sq

    return np.sqrt(min_sq)
