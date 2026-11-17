# Changelog

## v7.0.0 — 2026-11

Version described in the SoftwareX article. It replaces v6.1.0, whose placement
could produce interpenetrating fibers; RVEs generated with v6.x are not comparable
with RVEs generated with v7 (the same seed now yields a different, admissible
microstructure).

### Correctness

- Collision detection: the periodic ghost of a fiber no longer hides its own root
  fiber from the neighbour query; `--clearance` reaches the collision kernel; the
  free-point test checks a real segment; the spatial index is rebuilt after
  densification moves fibers.
- **Blocking geometric audit** (`validation/geometric_audit.py`) with its own
  distance kernel (`validation/audit_distance.py`), qualified against an exact
  rational oracle: no export unless every pair of fibers, and every fiber against
  its own periodic images, is disjoint. Status, scope, measured clearance,
  responsible pair and SHA-256 hashes in `{prefix}_validity.json`.
- **Pores**: the generator tested pores against the root fibers only, so a pore
  near a face could overlap the periodic image of a fiber crossing the opposite
  face; every useful image is now tested with an exact point-to-segment distance,
  and a blocking **pore audit** checks pore–pore and pore–fiber non-overlap before
  any export (`--porosity` was refused altogether by the first version of the audit).
- Voxeliser: flat-ended swept tubes, robust to folded fibers; non-circular
  sections are refused instead of being rasterised as circles.
- CAD export readable by Abaqus (no loose geometry), `MATRIX` and `INCLUSIONS`
  physical groups disjoint (the matrix group covered the whole box), and a
  post-export check (`{prefix}_export_check.json`).

### Densification

- Stage 2 is now a force-biased **collective rearrangement**: batches of fibers
  are inserted with overlaps allowed and relaxed away; fibers bend, their step
  length and turning angles are restored after each sweep; rounds that cannot
  be relaxed are rolled back. CLI: `--dens_sweeps` and `--dens_damping` added,
  `--compression` removed; `--opt_iters`, `--jitter` and `--injection` are now
  honoured (they were ignored in v6.1.0).

### Reproducibility

- The parametric record stores the complete CLI argument set, the software
  versions, the duration of each phase and the pores.
- Without `--seed`, a seed is drawn, logged and recorded.
- `regenerate.py` / `fps-regenerate` rebuilds every export from a parametric
  record without re-running placement (only export options may change).
- PGM slices are prefixed (`{prefix}_slice****.pgm`).

### Exports

- Abaqus voxel mesh (`--abaqus`, C3D8) with periodic equations.

### Packaging

- Console scripts `fps`, `fps-regenerate`, `fps-gui` (`fps` used to point to the
  argument parser and generated nothing).
- NumPy 2 supported; tested on Python 3.9–3.13 (Linux) and 3.12 (macOS, Windows).
- `benchmarks/`: scripts, results and machine description of the article figures.
- License: AGPL-3.0-or-later only (no dual licensing); `Licence.txt`.

## v6.1.0 — 2026-07

Initial public version (MIT, then AGPL-3.0-or-later).
