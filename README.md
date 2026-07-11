# Fiber Packing System (FPS)

*Language: **English** · [Français](README.fr.md)*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.compstruct.2026.120524-blue.svg)](https://doi.org/10.1016/j.compstruct.2026.120524)

**FPS** is an open-source Python tool that generates **periodic 3D representative
volume elements (RVEs)** of fiber-reinforced composites with **long, strongly
curved, non-parallel fibers**, optionally combined with **spherical porosity**.
It handles collisions with an analytic segment-to-segment distance test, enforces
strict periodic non-overlap, and exports to multiple formats for simulation
(FEM, FFT, surrogate models).

This repository is the software companion of the article:

> D. Ngouloubi, D. Choï, P. Karamian-Surville,
> *A geometric algorithm for dense 3D RVEs of long curved fiber composites*,
> **Composite Structures** 391 (2026) 120524.
> [doi:10.1016/j.compstruct.2026.120524](https://doi.org/10.1016/j.compstruct.2026.120524)

## Highlights

- **Smooth curved fibers** — centripetal Catmull–Rom spline centrelines with a
  no-twist **Bishop frame**; circular and super-elliptical cross-sections.
- **Dense admissible packing** — a **Constructive Self-Avoiding Walk (CSAW)**
  followed by a non-destructive **Compress & Fill** densification stage reaches
  fiber volume fractions up to **V_f ≈ 48 %** at aspect ratio *L/D = 150*,
  without altering fiber radii.
- **Multi-phase microstructures (fibers + spherical pores)** — an optional
  porosity generator inserts **spherical voids** that do not overlap either the
  fibers or one another, producing a three-phase RVE (matrix / fiber / pore).
- **Strict periodicity** — explicit ghost images (up to 26 neighbours) with the
  minimum-image convention, applied consistently to fibers and pores.
- **Fast collision detection** — spatial-hash broad phase + analytic
  segment-to-segment narrow phase (Eberly), compiled with **Numba JIT**.
- **Orientation control & descriptors** — directional/planar bias and **AD–PCA**
  orientation tensors valid for arbitrarily bent fibers.
- **Multi-format export** — parametric JSON, CAD (STEP), conformal FE mesh
  (Gmsh, optionally periodic), voxel field for FFT, and Nastran `.bdf`.

## Architecture

The pipeline runs in 5 sequential stages:

```
Stage 1 : Constructive placement (CSAW)      -> generation/generator.py
Stage 2 : Non-destructive densification       -> generation/optimizer_dynamic.py   (Compress & Fill)
Stage 3 : Topological audit                    -> validation/topology.py
Stage 4 : Porosity generation (optional)       -> generation/porosity_gen.py
Stage 5 : Statistics and export                -> main.py
```

### Directory tree

```
fps/
├── core/
│   ├── config.py              # Centralised configuration (dataclass, all parameters)
│   ├── fiber.py               # Fiber object (control points, centerline, Bishop frames)
│   ├── void.py                # Void object (spherical porosity)
│   └── grid_structure.py      # Spatial-hash grid (broad-phase collision, reverse-mapping)
├── geometry/
│   ├── curves.py              # Centripetal Catmull–Rom spline + resampling
│   ├── frames.py              # Bishop frame (parallel transport, Rodrigues)
│   └── sections.py            # Cross-sections (circular, super-elliptical, factory)
├── generation/
│   ├── generator.py           # CSAW algorithm (self-avoiding walk + backtracking) + RSDA
│   ├── periodicity.py         # PeriodicManager (wrap, 26-neighbour ghosts)
│   ├── optimizer_dynamic.py   # Compress & Fill: jitter + compression + soft-push + injection
│   └── porosity_gen.py        # Vectorised RSA for periodic spherical pores
├── collision/
│   ├── detector.py            # CollisionDetector (API: group_valid, segment_free, periodic)
│   └── detector_math.py       # Numba kernels: segment-to-segment distance (Eberly), MIC
├── validation/
│   ├── topology.py            # Unified clearance audit (segment-segment + gap distribution)
│   └── statistics.py          # Spatial descriptors (NND, Ripley K, g(r), Voronoi)
├── visualization/
│   ├── descriptors.py         # AD–PCA (orientation tensors), MicroDescriptor
│   ├── analyzer.py            # RVE_Analyzer (viability audit)
│   └── plotter.py             # 3D wireframe visualisation + analysis reports
├── export/
│   ├── gmsh_exporter.py       # CAD (.step) + FEM (.msh), periodic + adaptive mesh
│   ├── voxelizer.py           # Voxelised FFT grid (matrix=0, fiber=1, pore=2) + PGM
│   ├── csv_exporter.py        # CSV export (control points + direction)
│   └── nastran_exporter.py    # .msh -> .bdf conversion (Nastran via meshio)
├── utils/
│   └── logger.py              # Centralised logging configuration
├── main.py                    # 5-stage orchestrator (CSAW -> Densify -> Audit -> Pores -> Export)
├── cli.py                     # Command-line interface (argparse, 6 parameter groups)
└── gui.py                     # PyQt6 desktop GUI (parameters, 3D view, console)
```

## Installation

Requires **Python ≥ 3.9**.

```bash
git clone https://github.com/dev7cd/fps.git
cd fps
pip install -r requirements.txt
```

Or, as a package (from `pyproject.toml`):

```bash
pip install -e .            # core dependencies
pip install -e ".[gui,dev]" # + PyQt6 GUI and pytest
```

| Dependency | Role |
|-----------|------|
| **numpy** | vectorised computation |
| **scipy** | cKDTree, Voronoi, special functions |
| **numba** | JIT compilation of critical loops (collision, voxelisation) |
| **gmsh** | finite-element mesh generation |
| **meshio** | multi-format conversion (required for `--nastran`) |
| **matplotlib** | visualisation (optional) |
| **PyQt6** | desktop GUI (required for `gui.py`) |

## Usage

### Minimal example

```bash
python main.py --dims 1 1 1 --vf 0.30 --radius 0.02 --seed 42
```

### Advanced example (dense, oriented, porous RVE with full export)

```bash
python main.py \
    --dims 1 1 1 \
    --vf 0.55 \
    --radius 0.015 \
    --seed 42 \
    --bias_type planar --bias_vec1 1 0 0 --bias_vec2 0 1 0 --strength 0.8 \
    --curvature 30 \
    --rsda \
    --optimize --opt_iters 20 --injection 100 \
    --porosity --void_vf 0.01 \
    --res_fft 256 \
    --periodic_mesh \
    --nastran \
    --output RVE_UD_55
```

## CLI parameters

### 1. Domain and targets

| Parameter | Type | Default | Description |
|-----------|------|--------|-------------|
| `--dims` | float x3 | 1.0 1.0 1.0 | RVE dimensions (Lx Ly Lz) |
| `--vf` | float | 0.05 | Target fiber volume fraction |
| `--seed` | int | None | Random seed for reproducibility |

### 2. Fiber geometry

| Parameter | Type | Default | Description |
|-----------|------|--------|-------------|
| `--radius` | float | 0.02 | Collision radius (circular envelope) |
| `--section` | str | circular | Cross-section type (circular, superelliptical) |
| `--sec_major` | float | 0.019 | Major semi-axis (real section) |
| `--sec_minor` | float | 0.015 | Minor semi-axis (real section) |
| `--sec_n` | float | 2.5 | Super-ellipse exponent (>2 = rounded corners) |
| `--clearance` | float | 0.0 | Minimum inter-fiber clearance |

### 3. Trajectory and orientation

| Parameter | Type | Default | Description |
|-----------|------|--------|-------------|
| `--min_pts` | int | 15 | Min control points per fiber |
| `--max_pts` | int | 20 | Max control points per fiber |
| `--step` | float | 0.07 | Segment length |
| `--max_attempts` | int | 500 | Max attempts before giving up |
| `--bias_type` | str | planar | Bias mode (free, uniaxial, planar) |
| `--bias_vec1` | float x3 | 1 0 0 | Primary direction vector |
| `--bias_vec2` | float x3 | 0 1 0 | Secondary direction vector (planar mode) |
| `--strength` | float | 0.0 | Bias strength (0 = random, 1 = strict) |
| `--curvature` | float | 60.0 | Maximum turning angle (degrees) |
| `--rsda` | flag | off | Enable dynamic rearrangement (RSDA) |
| `--rsda_perturb` | float | 0.05 | RSDA perturbation intensity (fraction of radius) |

### 4. Densification (Stage 2 — Compress & Fill)

| Parameter | Type | Default | Description |
|-----------|------|--------|-------------|
| `--optimize` / `--no_optimize` | flag | on | Enable/disable the densifier |
| `--opt_iters` | int | 10 | Optimisation cycles |
| `--jitter` | float | 0.05 | Jitter intensity (fraction of radius) |
| `--compression` | float | 0.01 | Centripetal compression intensity |
| `--injection` | int | 50 | Injection attempts per cycle |

### 5. Porosity (spherical pores)

| Parameter | Type | Default | Description |
|-----------|------|--------|-------------|
| `--porosity` | flag | off | Generate spherical pores |
| `--void_vf` | float | 0.01 | Target pore volume fraction |
| `--void_mean` | float | 0.01 | Mean pore radius |
| `--void_std` | float | 0.002 | Pore radius standard deviation |

### 6. Exports

| Parameter | Type | Default | Description |
|-----------|------|--------|-------------|
| `--output` | str | RVE_Result | Output file prefix |
| `--no_mesh` | flag | off | Disable Gmsh export |
| `--res_mesh` | int | 20 | Mesh curvature resolution |
| `--res_fft` | int | 128 | Voxel grid resolution (NxNxN) |
| `--periodic_mesh` | flag | off | Periodic mesh (identical nodes on opposite faces) |
| `--no_adaptive_mesh` | flag | off | Disable Distance+Threshold adaptive refinement |
| `--nastran` | flag | off | Export to Nastran .bdf format |
| `--no_spatial_stats` | flag | off | Disable spatial descriptors |

## Output files

| File | Description |
|------|-------------|
| `{prefix}_parametric.json` | metadata, parameters, statistics, fiber geometry |
| `{prefix}_roots.csv` | control points of the root fibers |
| `{prefix}_full_periodic.csv` | root fibers + all periodic ghosts |
| `{prefix}_voxelmap.npy` | 3D uint8 grid (matrix=0, fiber=128, pore=255) |
| `slice****.pgm` | 2D slices of the voxel grid (PGM P2) |
| `{prefix}.step` | CAD geometry (OpenCASCADE) |
| `{prefix}.msh` | finite-element mesh (Gmsh v4) |
| `{prefix}.bdf` | Nastran Bulk Data (with `--nastran`) |

## Algorithms

### CSAW (Constructive Self-Avoiding Walk)

Constructive placement of curved fibers by a biased self-avoiding random walk
with backtracking:

1. Search for a free starting point in the domain.
2. Grow the fiber segment-by-segment under a maximum curvature (turning angle)
   constraint.
3. Orientation bias (free, uniaxial, planar) via directional mixing.
4. Backtracking when too many consecutive collisions occur.
5. Periodic validation (fiber + ghost images) before accepting each segment.

### RSDA (Randomised Sequential Dynamic Adsorption)

An extension of the standard placement that perturbs already-placed fibers to
free up space (`--rsda`):

- Triggered when standard placement fails (>100 attempts).
- Identifies colliding fibers and displaces them slightly.
- Full rollback if the perturbation creates new collisions.
- Helps exceed the jamming limit of pure RSA (~54.7 %).

### Compress & Fill (non-destructive densification, Stage 2)

Increases the volume fraction by admissibly reorganising the existing
configuration (jitter, centripetal drift, soft-push repulsion,
bounded-curvature morphogenesis) and refilling freed void space through
additional CSAW passes — **without changing any fiber radius**. Each move is
transactional: it is rolled back if it breaks admissibility.

### Collision detection

Analytic segment-to-segment distance (Eberly's algorithm):

- Exact parametric resolution on the unit square [0,1] × [0,1].
- Handling of the 9 edge-clamping cases and parallel segments.
- Compiled with Numba JIT for performance.
- Accelerated by a spatial-hash grid (broad phase).

### Periodic boundary conditions

Toroidal topology on the 3 axes:

- Each fiber crossing a face generates ghost images on the opposite face.
- Up to 26 periodic images per fiber (6 faces + 12 edges + 8 corners).
- Minimum Image Convention (MIC) for distance computations.

### Spherical porosity and sphere–fiber association

Optional generation of a **second inclusion phase** (spherical pores) by
Random Sequential Adsorption (`--porosity`):

- Pore radii drawn from a truncated normal law (`--void_mean`, `--void_std`),
  with adaptive radius shrinking after repeated rejections.
- **Mutual non-overlap** is enforced on both sides: pore–pore checks
  (vectorised) *and* pore–fiber checks (bounding-box broad phase +
  point-to-centerline distance against the fiber tube).
- Pores are periodic (their own ghost images), consistently with the fibers.
- The result is a three-phase RVE (matrix / fiber / pore) exported as voxel tags
  **0 / 128 / 255** (uint8), directly usable for FFT and FE homogenisation of
  porous fiber-reinforced composites.

### Statistical descriptors

Validation of the spatial distribution (2D projection of centroids):

- **NND**: nearest-neighbour distance distribution (toroidal correction).
- **Ripley's K**: K(h) compared to the CSR process (K_poisson = π·h²).
- **g(r)**: pair-correlation function by annular counting.
- **Voronoi**: area distribution, coefficient of variation (CV ≈ 0 = regular,
  CV ≈ 0.53 = Poisson).

### Orientation (AD–PCA)

Anisotropic decomposition via principal component analysis:

- Axial orientation tensor A_axial (weighted by efficiency 1/tortuosity).
- Planar orientation tensor A_planar (weighted by biaxiality).
- Hermans-type factors f_axial, f_planar in [0, 1] (0 = isotropic, 1 = aligned).

### Voxelisation

3D rasterisation on a regular grid:

- Tags: matrix = 0, fiber = 128, pore = 255 (uint8, chosen for direct PGM greyscale export).
- Exact point-segment distance per voxel (Numba JIT).
- Accelerated by local BBox + early exit.
- PGM (Portable Gray Map) export by Z-slices.

### Gmsh meshing

CAD + finite-element mesh generation:

- Pipe extrusion (section swept along the spline centerline).
- Boolean fragmentation (OpenCASCADE) for interfaces.
- Adaptive size fields (Distance + Threshold around interfaces).
- Optional periodic mesh (identical nodes on opposite faces via `setPeriodic`).
- 3D algorithm: HXT (parallel).

## Graphical interface

A PyQt6 desktop GUI is available to drive the generator without the command line.

### Launch

```bash
python gui.py
```

### Features

- **Parameter panel**: 6 collapsible sections grouping all CLI parameters, with
  explanatory tooltips.
- **3D view**: wireframe visualisation of the box and generated fibers
  (embedded matplotlib).
- **Real-time console**: displays the generation log as it runs.
- **Progress bar**: tracks the stages (CSAW, densification, validation,
  porosity, export).
- **Non-blocking execution**: generation runs in a subprocess; the interface
  stays responsive.
- **Stop at any time**: a button to interrupt an ongoing generation.

| Section | Content |
|---------|---------|
| Domain and Targets | RVE dimensions, volume fraction, random seed |
| Fiber Geometry | Radius, section type, clearance, super-ellipse parameters |
| Orientation | Bias mode (free/uniaxial/planar), alignment strength, curvature, RSDA |
| Densification | Enable Stage 2, iterations, jitter, compression, injection |
| Porosity | Enable pores, volume fraction, mean radius and standard deviation |
| Export | Output prefix, FFT resolution, Gmsh mesh, periodic, Nastran, spatial stats |

## Reproducibility

Every stochastic draw is controlled by an explicit `--seed`, and the full
parameter set is stored in `{prefix}_parametric.json`. Re-running the same
command reproduces an identical RVE; the parametric record regenerates all
downstream exports without re-running the placement engine.

## Documentation

- API reference (Doxygen): `docs/html/index.html`
- French README: [README.fr.md](README.fr.md)

## Citing

If you use FPS in your research, please cite the associated article:

```bibtex
@article{Ngouloubi2026FPS,
  title   = {A geometric algorithm for dense 3D RVEs of long curved fiber composites},
  author  = {Ngouloubi, Devine and Cho{\"\i}, Daniel and Karamian-Surville, Philippe},
  journal = {Composite Structures},
  volume  = {391},
  pages   = {120524},
  year    = {2026},
  doi     = {10.1016/j.compstruct.2026.120524}
}
```

## License

Distributed under the **MIT License**. See [LICENSE](LICENSE) for details.

## Author

**Devine Ngouloubi** — sole developer (concept, design and implementation).
Nicolas Oresme Mathematics Laboratory (LMNO, UMR 6139), Normandy University,
UNICAEN, CNRS, Caen, France.

The underlying geometric and homogenisation theory was developed jointly with
Daniel Choï and Philippe Karamian-Surville in the associated *Composite
Structures* article (see [Citing](#citing)).
