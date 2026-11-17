# SPDX-FileCopyrightText: 2026 Devine Ngouloubi <exauce-devine.ngouloubi@unicaen.fr>
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Figures of the SoftwareX article, rebuilt from the benchmark results and a stored RVE.

    python benchmarks/plot_figures.py performance --results benchmarks/results --out figures/
    python benchmarks/plot_figures.py rve RVE_parametric.json --out figures/

`performance` reads generation.csv and voxelizer.csv (run_generation.py, run_voxelizer.py);
`rve` renders the fiber centerlines and the pores of a parametric record (no generation).
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from geometry.curves import CatmullRomSpline  # noqa: E402

INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
BLUE, ORANGE, RED = '#2a78d6', '#eb6834', '#e34948'
BLUES = ['#86b6ef', '#6da7ec', '#5598e7', '#3987e5', '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281']

plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.titlesize': 9.5, 'legend.frameon': False,
                     'savefig.bbox': 'tight', 'savefig.dpi': 300})


def _plain_log_y(ax):
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:g}'))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def _axes(ax):
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.grid(True, which='major', color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def performance(results, out):
    gen = [r for r in _read(results / 'generation.csv') if r['status'] == 'ok']
    vox = _read(results / 'voxelizer.csv')
    by_vf = defaultdict(list)
    for r in gen:
        by_vf[float(r['target_vf'])].append(r)
    vfs = sorted(by_vf)
    col = lambda k: [np.array([float(r[k]) for r in by_vf[v]]) for v in vfs]  # noqa: E731
    total, place = col('generation_s'), col('placement_s')
    nfib, lod = col('n_fibers'), col('aspect_ratio')
    x = 100 * np.array(vfs)

    # One chart per file (journal artwork rule): generation time, then voxeliser scaling
    fig_a, a = plt.subplots(figsize=(4.6, 3.2), constrained_layout=True)
    fig_b, b = plt.subplots(figsize=(4.6, 3.2), constrained_layout=True)
    _axes(a)
    a.errorbar(x, [t.mean() for t in total], yerr=[t.std() for t in total], color=ORANGE, lw=2, marker='o',
               ms=5, capsize=3, label='total generation')
    a.errorbar(x, [t.mean() for t in place], yerr=[t.std() for t in place], color=BLUE, lw=2, marker='s',
               ms=5, capsize=3, label='sequential placement (CSAW)')
    a.set_yscale('log')
    _plain_log_y(a)
    a.set_xticks(x)
    a.set_xticklabels([f'{v:.0f}\n({n.mean():.0f})' for v, n in zip(x, nfib)])
    a.set_xlabel(r'target $V_f$ [%]  (mean number of fibers)')
    a.set_ylabel('wall-clock time [s]')
    a.set_title(f'Generation, $L/D \\approx {np.mean([l.mean() for l in lod]):.0f}$, '
                f'{max(len(t) for t in total)} seeds per point')
    # Volume fraction at which sequential placement stalls (runs whose target it could not reach)
    stalls = [float(r['placement_vf']) for r in gen
              if r.get('placement_vf') and float(r['placement_vf']) < float(r['target_vf']) - 1e-3]
    if stalls:
        xs = 100 * float(np.mean(stalls))
        a.axvline(xs, color=INK2, lw=1, ls=':')
        a.text(xs + 0.6, a.get_ylim()[0] * 1.6, f'CSAW alone\nstalls at\n$V_f \\approx {xs:.0f}\\%$', ha='left',
               va='bottom', fontsize=7.5, color=INK2)
    a.legend(loc='upper left', fontsize=8)

    _axes(b)
    n = np.array([int(r['resolution']) for r in vox])
    t = np.array([float(r['median_s']) for r in vox])
    b.plot(n, t, color=BLUE, lw=2, marker='s', ms=5, label='voxeliser (measured)')
    b.plot(n, t[0] * (n / n[0]) ** 3, color=INK2, lw=1.2, ls='--', label=r'$\propto N^3$ reference')
    b.set_xscale('log', base=2)
    b.set_yscale('log')
    _plain_log_y(b)
    b.set_xticks(n)
    b.set_xticklabels([str(v) for v in n])
    b.set_xlabel(r'voxel grid resolution $N$ (grid $N^3$)')
    b.set_ylabel('wall-clock time [s]')
    b.set_title('Voxeliser scaling')
    b.legend(loc='upper left', fontsize=8)

    out.mkdir(parents=True, exist_ok=True)
    for fig, name in ((fig_a, 'fig_generation'), (fig_b, 'fig_voxelizer')):
        for ext in ('pdf', 'png'):
            fig.savefig(out / f'{name}.{ext}')
        print(f"Written {out / name}.pdf")


def _sphere(ax, c, r, color):
    u, v = np.mgrid[0:2 * np.pi:16j, 0:np.pi:10j]
    ax.plot_surface(c[0] + r * np.cos(u) * np.sin(v), c[1] + r * np.sin(u) * np.sin(v), c[2] + r * np.cos(v),
                    color=color, linewidth=0, antialiased=True, shade=True, alpha=0.95)


def _box(ax, L):
    corners = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)]) * L
    for p in range(8):
        for q in range(p + 1, 8):
            if np.sum(np.abs(corners[p] - corners[q]) > 0) == 1:
                ax.plot(*zip(corners[p], corners[q]), color=INK2, lw=0.8)


def rve(record_path, out):
    rec = json.loads(Path(record_path).read_text())
    L = np.asarray(rec['simulation_parameters']['box_dims'], dtype=float)
    lines = [CatmullRomSpline.interpolate(np.asarray(f['control_points'], dtype=float), num_points=100)
             for f in rec['microstructure_data']]
    pores = {}
    for v in rec.get('voids', []):
        pores.setdefault(v['id'], (np.mod(np.asarray(v['center']), L), v['radius']))

    pts = np.vstack(lines)
    lo = np.minimum(pts.min(axis=0), 0.0)
    hi = np.maximum(pts.max(axis=0), L)
    fig = plt.figure(figsize=(7.2, 3.4), constrained_layout=True)
    views = [('(a) Perspective view', 22, -60, 'persp'), ('(b) Top view (fiber alignment)', 90, -90, 'ortho')]
    for k, (title, elev, azim, proj) in enumerate(views):
        ax = fig.add_subplot(1, 2, k + 1, projection='3d')
        ax.set_proj_type(proj)
        _box(ax, L)
        for i, line in enumerate(lines):
            ax.plot(line[:, 0], line[:, 1], line[:, 2], color=BLUES[i % len(BLUES)], lw=1.4,
                    solid_capstyle='round')
        for c, r in pores.values():
            _sphere(ax, c, r, RED)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_zlim(lo[2], hi[2])
        ax.set_box_aspect(hi - lo)
        ax.set_axis_off()
        ax.set_title(title)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(out / f'fig_rve.{ext}')
    print(f"Written {out / 'fig_rve.pdf'}  ({len(lines)} fibers, {len(pores)} pores)")


def architecture(out):
    """Pipeline diagram (Fig. 1): five stages, shared engine, outputs and regeneration."""
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(7.0, 7.0))
    ax.set_xlim(0, 100)
    ax.set_ylim(6, 106)
    ax.axis('off')

    def box(x, y, w, h, title, sub, face, edge):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.4,rounding_size=1.6',
                                    facecolor=face, edgecolor=edge, linewidth=1.4))
        ax.text(x + w / 2, y + h * 0.66, title, ha='center', va='center', fontsize=8.9, weight='bold', color=INK)
        ax.text(x + w / 2, y + h * 0.27, sub, ha='center', va='center', fontsize=8, color=INK2,
                family='monospace', linespacing=1.3)

    def arrow(x0, y0, x1, y1, color=INK2, ls='-', conn=None):
        props = dict(arrowstyle='-|>', color=color, lw=1.4, linestyle=ls, shrinkA=0, shrinkB=0)
        if conn:
            props['connectionstyle'] = conn
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=props)

    stage_face, stage_edge = '#e6f0fc', BLUE
    x, w, h = 10, 57, 9.6
    box(x, 94, w, h, 'Inputs: CLI, GUI or parametric record', 'box  Vf  radius  bias  curvature  seed',
        '#e3f4ec', '#1baf7a')
    stages = [('Stage 1 - Constructive placement (CSAW)', 'generation/generator.py', stage_face, stage_edge),
              ('Stage 2 - Collective rearrangement', 'generation/optimizer_dynamic.py', stage_face, stage_edge),
              ('Stage 3 - Blocking geometric audit', 'validation/geometric_audit.py',
               stage_face, stage_edge),
              ('Stage 4 - Pores + pore audit (optional)', 'generation/porosity_gen.py', '#fdeee6', ORANGE),
              ('Stage 5 - Statistics and exports', 'main.py', stage_face, stage_edge)]
    ys = [80, 67, 54, 41, 28]
    for (title, sub, face, edge), y in zip(stages, ys):
        box(x, y, w, h, title, sub, face, edge)
    for y0, y1 in zip([94] + ys[:-1], ys):
        arrow(x + w / 2, y0 - 0.6, x + w / 2, y1 + h + 0.6)
    box(x, 9, w, 13.5, 'Outputs', 'parametric JSON  validity JSON  STEP\n'
        'Gmsh .msh  voxel .npy/PGM  Nastran .bdf\nAbaqus .inp (periodic)', '#eeeeec', INK2)
    arrow(x + w / 2, 27.4, x + w / 2, 23.1)

    # Regeneration loop: record -> exports without placement
    arrow(x - 0.6, 15.75, x - 0.6, 98.8, color=BLUE, conn='bar,fraction=-0.065')
    ax.text(x - 7.6, 57, 'regenerate.py: parametric JSON -> all exports, no placement', rotation=90,
            ha='center', va='center', fontsize=8, color=BLUE)

    # Shared engine
    ex, ew = 74, 24.5
    ax.add_patch(FancyBboxPatch((ex, 28), ew, 61.6, boxstyle='round,pad=0.4,rounding_size=1.6',
                                facecolor='#f1effc', edgecolor='#6250d6', linewidth=1.4))
    ax.text(ex + ew / 2, 86.3, 'Shared engine', ha='center', fontsize=10, weight='bold', color=INK)
    items = [('core/', 'Fiber, Void, config,\nspatial-hash grid'), ('geometry/', 'Catmull-Rom spline,\nBishop frame'),
             ('collision/', 'segment-segment\ndistance, Numba JIT'), ('periodicity', 'ghost images,\nshifts -1/0/+1'),
             ('validation/', 'independent audit\nkernel')]
    for k, (name, desc) in enumerate(items):
        yy = 80 - k * 11.0
        ax.text(ex + 2, yy, name, fontsize=8.8, weight='bold', family='monospace', color='#4a3aa7')
        ax.text(ex + 2, yy - 4.3, desc, fontsize=7.8, color=INK2, va='center')
    for y in ys:
        arrow(ex - 0.6, y + h / 2, x + w + 0.8, y + h / 2, color='#6250d6', ls='--')

    out.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(out / f'fig_architecture.{ext}')
    print(f"Written {out / 'fig_architecture.pdf'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='figure', required=True)
    g = sub.add_parser('architecture')
    g.add_argument('--out', type=Path, default=Path('.'))
    p = sub.add_parser('performance')
    p.add_argument('--results', type=Path, default=ROOT / 'benchmarks' / 'results')
    p.add_argument('--out', type=Path, default=Path('.'))
    r = sub.add_parser('rve')
    r.add_argument('record', type=Path)
    r.add_argument('--out', type=Path, default=Path('.'))
    a = ap.parse_args()
    if a.figure == 'architecture':
        architecture(a.out)
    elif a.figure == 'performance':
        performance(a.results, a.out)
    else:
        rve(a.record, a.out)


if __name__ == '__main__':
    main()
