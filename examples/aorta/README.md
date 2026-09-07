# Aorta

Patient-specific aorta with four RCR Windkessel outlets.

| | |
|---|---|
| Mesh | ~110k nodes, ~640k tetrahedra (not in the repository, see `data/aorta/README.md`) |
| Units | CGS: cm/s, poise, g/cm³, barye |
| Time step | 8×10⁻⁴ s, cardiac cycle 0.8 s |
| Cost | hours; run under MPI |

```bash
mpirun -n 8 python run.py --config Ao11mmrest.yaml
```

Every parameter is in the YAML file. The `Run:` section holds the output directory, the element pair, how many steps to take and which of them are written to XDMF; set `Run.MaxSteps` to a small number for a smoke check.

## Outlet coupling

The four RCR outlets are coupled implicitly. The coupling tangent `dP_k/dQ_l` is obtained by perturbing the 0D model rather than by differentiating it, so any lumped-parameter network satisfying the small interface in `docs/development.md` can be attached without deriving a derivative by hand -- including networks whose outlets share state, which give a full tangent matrix. `Solver.Scheme: "newton"` is what applies it; `"picard"` lags the coupling instead.

## Configurations

| file | mesh | inflow | notes |
|---|---|---|---|
| `Ao9mmrest.yaml` | 9 mm | 1 cycle, scale 0.7 | wall tag 4 |
| `Ao11mmrest.yaml` | 11 mm | 1 cycle | wall tag 4; the default |
| `Ao11mmrest-12cycles.yaml` | 11 mm | 12 cycles | for a periodic Windkessel state |
| `Ao13mmrest.yaml` | 13 mm | 1 cycle, scale 0.8 | **wall tag 1**, tighter tolerance |

The wall tag differs between meshes. That is correct per-mesh metadata, but pointing a configuration at the wrong mesh yields no boundary conditions at all rather than an error, so the solver validates the tags at construction and refuses to run.

## Output

- `aorta_u.xdmf` / `aorta_p.xdmf` — velocity and pressure (separate files: they live on different spaces). Use `--store-every`; a full run writes 12000 fields, tens of GB.
- `aorta_diagnostics.csv` — one row per step: per-cap flow and split, mass defect, divergence norm, Windkessel pressures in mmHg, Picard/convergence data, wall time. Written on rank 0 and flushed every step.

## Interpreting the pressures

Pressures start at `Pd_init` and charge over several cardiac cycles, so the physiological warnings will fire during the first cycle — that is expected, not a failure. Judge systolic/diastolic values only from a run that has reached a periodic state.
