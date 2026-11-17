# SPDX-FileCopyrightText: 2026 Devine Ngouloubi <exauce-devine.ngouloubi@unicaen.fr>
# SPDX-License-Identifier: AGPL-3.0-or-later

"""3D images of an RVE before and after periodisation, from its parametric record (no generation).

* before: the root fibers as generated, continuous, extending beyond the cell;
* after: every fiber and its periodic images cut by the cell faces (flat caps), i.e. the
  periodic RVE that the exports describe; a fiber leaving through one face re-enters
  through the opposite one with the same colour.

Requires PyVista (``pip install pyvista``), which is not a dependency of FPS.

    python benchmarks/render_rve.py RVE_parametric.json --out figures/
    python benchmarks/render_rve.py RVE_05_parametric.json RVE_UD_30_parametric.json \
        --labels "dilute, Vf = 5 %" "dense, Vf = 30 %" --out figures/ --panel fig_rve
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from geometry.curves import CatmullRomSpline  # noqa: E402
from geometry.frames import compute_bishop_frame  # noqa: E402

PORE_COLOR = '#e34948'
CELL_COLOR = '#2b2b2b'


def load(record_path):
    rec = json.loads(Path(record_path).read_text())
    L = np.asarray(rec['simulation_parameters']['box_dims'], dtype=float)
    fibers = [(CatmullRomSpline.interpolate(np.asarray(f['control_points'], dtype=float), num_points=100),
               float(f['radius'])) for f in rec['microstructure_data']]
    pores = {}
    for v in rec.get('voids', []):
        pores.setdefault(v['id'], (np.mod(np.asarray(v['center'], dtype=float), L), float(v['radius'])))
    return L, fibers, list(pores.values())


def tube(pv, line, r, n_sides=24):
    """Closed, manifold tube around a polyline: rings oriented by the Bishop frame (as for the CAD
    sweep), quads between consecutive rings and a triangle fan on each end."""
    _, N, B = compute_bishop_frame(line)
    theta = np.linspace(0.0, 2.0 * np.pi, n_sides, endpoint=False)
    ring = np.cos(theta)[None, :, None] * N[:, None, :] + np.sin(theta)[None, :, None] * B[:, None, :]
    pts = (line[:, None, :] + r * ring).reshape(-1, 3)
    n = len(line)
    pts = np.vstack([pts, line[0], line[-1]])
    c0, c1 = n * n_sides, n * n_sides + 1
    faces = []
    for i in range(n - 1):
        for j in range(n_sides):
            a, b = i * n_sides + j, i * n_sides + (j + 1) % n_sides
            faces += [4, a, b, b + n_sides, a + n_sides]
    for j in range(n_sides):
        faces += [3, c0, (j + 1) % n_sides, j]
        last = (n - 1) * n_sides
        faces += [3, c1, last + j, last + (j + 1) % n_sides]
    return pv.PolyData(pts, np.array(faces)).triangulate()


def clip_to_cell(surface, L):
    """Closed surface clipped by the six faces of the cell [0, L]; the cuts are capped."""
    for axis in range(3):
        for sign, origin in ((1.0, 0.0), (-1.0, L[axis])):
            if surface.n_points == 0:
                return surface
            normal = np.zeros(3)
            normal[axis] = sign
            o = np.zeros(3)
            o[axis] = origin
            surface = surface.clip_closed_surface(normal=normal, origin=o)
    return surface


def periodic_pieces(pv, line, r, L):
    """Pieces of a fiber and of its periodic images inside the cell."""
    lo, hi = line.min(axis=0) - r, line.max(axis=0) + r
    ranges = [range(int(np.floor(-hi[a] / L[a])) + 1, int(np.ceil((L[a] - lo[a]) / L[a]))) for a in range(3)]
    base = tube(pv, line, r)
    pieces = []
    for k in itertools.product(*ranges):
        piece = clip_to_cell(base.translate(np.array(k) * L, inplace=False), L)
        if piece.n_points:
            pieces.append(piece)
    return pieces


def render(record_path, out, prefix, window=(1600, 1300)):
    import pyvista as pv
    from matplotlib import colormaps
    pv.OFF_SCREEN = True
    L, fibers, pores = load(record_path)
    cmap = colormaps['viridis']
    colors = [cmap(0.08 + 0.84 * i / max(1, len(fibers) - 1))[:3] for i in range(len(fibers))]
    rng = np.random.default_rng(0)
    rng.shuffle(colors)                                        # neighbours get distinct shades

    allpts = np.vstack([line for line, _ in fibers])
    rmax = max(r for _, r in fibers)
    lo = np.minimum(allpts.min(axis=0) - rmax, 0.0)
    hi = np.maximum(allpts.max(axis=0) + rmax, L)
    centre = (lo + hi) / 2
    extent = float(np.linalg.norm(hi - lo))

    paths = {}
    for state in ('before', 'after'):
        p = pv.Plotter(off_screen=True, window_size=window)
        p.set_background('white')
        for (line, r), c in zip(fibers, colors):
            meshes = [tube(pv, line, r)] if state == 'before' else periodic_pieces(pv, line, r, L)
            for m in meshes:
                p.add_mesh(m, color=c, smooth_shading=True, specular=0.25, specular_power=20)
        for centre_p, rp in pores:
            p.add_mesh(pv.Sphere(radius=rp, center=centre_p, theta_resolution=32, phi_resolution=32),
                       color=PORE_COLOR, smooth_shading=True, specular=0.3)
        p.add_mesh(pv.Box(bounds=(0, L[0], 0, L[1], 0, L[2])).extract_feature_edges(),
                   color=CELL_COLOR, line_width=5)
        # Same camera for both states: the cell stays in place, only its content changes
        direction = np.array([1.35, -1.1, 0.95])
        p.camera.focal_point = centre
        p.camera.position = centre + direction / np.linalg.norm(direction) * 3.2 * extent
        p.camera.up = (0.0, 0.0, 1.0)
        p.enable_parallel_projection()
        p.camera.parallel_scale = 0.40 * extent
        p.enable_anti_aliasing('ssaa')
        path = Path(out) / f'{prefix}_{state}.png'
        path.parent.mkdir(parents=True, exist_ok=True)
        p.screenshot(str(path))
        p.close()
        paths[state] = path
        print(f'Written {path}')
    return paths


def panel(rows, labels, out, name):
    """Figure with one row per RVE: before | after periodisation."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(rows), 2, figsize=(7.2, 3.3 * len(rows)), constrained_layout=True,
                             squeeze=False)
    letters = iter('abcdefghij')
    for (paths, label), ax_row in zip(zip(rows, labels), axes):
        imgs = {state: plt.imread(paths[state])[..., :3] for state in ('before', 'after')}
        # Common crop of the two states (same camera): keeps the cell at the same place and scale
        ink = np.zeros(imgs['before'].shape[:2], dtype=bool)
        for img in imgs.values():
            ink |= img.min(axis=2) < 0.97
        ys, xs = np.nonzero(ink)
        pad = 10
        y0, y1 = max(ys.min() - pad, 0), ys.max() + pad
        x0, x1 = max(xs.min() - pad, 0), xs.max() + pad
        for state, ax in zip(('before', 'after'), ax_row):
            ax.imshow(imgs[state][y0:y1, x0:x1])
            ax.set_axis_off()
            what = 'as generated' if state == 'before' else 'periodised'
            ax.set_title(f'({next(letters)}) {label}, {what}', fontsize=8.5)
    for ext in ('pdf', 'png'):
        fig.savefig(Path(out) / f'{name}.{ext}', dpi=340, bbox_inches='tight')   # > 2244 px wide
    print(f"Written {Path(out) / name}.pdf")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('records', type=Path, nargs='+', help='<prefix>_parametric.json files')
    ap.add_argument('--out', type=Path, default=Path('.'))
    ap.add_argument('--labels', nargs='+', help='one label per record (for --panel)')
    ap.add_argument('--panel', help='also write a combined figure <panel>.pdf/.png')
    a = ap.parse_args()
    rows = []
    for rec in a.records:
        stem = rec.name[:-len('_parametric.json')] if rec.name.endswith('_parametric.json') else rec.stem
        rows.append(render(rec, a.out, stem))
    if a.panel:
        panel(rows, a.labels or [r.stem for r in a.records], a.out, a.panel)


if __name__ == '__main__':
    main()
