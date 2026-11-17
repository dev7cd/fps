# Example RVEs

Parametric records of the two example RVEs of the SoftwareX article, generated
with FPS v7.0.0 on a laptop (Intel Core i5-1335U, Python 3.13).

| File | Command | Content |
|------|---------|---------|
| `RVE_05_parametric.json` | `python main.py --dims 1 1 1 --vf 0.05 --radius 0.02 --seed 42 --output RVE_05` | 26 fibers, V_f = 5 % |
| `RVE_UD_30_parametric.json` | `python main.py --dims 1 1 1 --vf 0.30 --radius 0.02 --seed 7 --bias_type uniaxial --bias_vec1 1 0 0 --strength 0.3 --curvature 60 --porosity --void_vf 0.02 --void_mean 0.03 --res_fft 256 --abaqus --periodic_mesh --output RVE_UD_30` | 156 fibers, V_f = 30 %, 242 pores |

The dense record was written before `--res_abaqus 64` was added to the command of
the article; this option only sets the resolution of the Abaqus export and has no
effect on the geometry. The FPS commit stored in the records predates a rewrite of
the commit messages of the repository; its code is identical to tag `v7.0.0`.

Each record holds the complete argument set, the software versions, the phase
timings, the control points and radius of every fiber, the pores and the result
of the geometric audits. The exports can be rebuilt from it without running the
placement again, for instance:

```bash
python regenerate.py examples/RVE_UD_30_parametric.json --res_fft 256 --no_mesh --output RVE_UD_30
python regenerate.py examples/RVE_UD_30_parametric.json --abaqus --res_abaqus 64 --no_mesh --output RVE_UD_30
```

The 3D views of the article (before and after periodisation) are produced from
the records with `benchmarks/render_rve.py`, which needs PyVista:

```bash
python benchmarks/render_rve.py examples/RVE_05_parametric.json examples/RVE_UD_30_parametric.json \
    --labels "\$V_f\$ = 5 %" "\$V_f\$ = 30 % with pores" --out figures/ --panel fig_rve
```
