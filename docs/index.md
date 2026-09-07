# fenicsx-navier-stokes

Stabilized incompressible Navier–Stokes solvers in [FEniCSx](https://fenicsproject.org) for vascular haemodynamics.

- **[Theory](theory.md)** — the weak form, the stabilization, and the block solver.
- **[Examples](examples.md)** — aorta, DFG cylinder, 3D vessel segment.
- **[Verification](verification.md)** — what is tested, against what, and how accurate it is.
- **[API reference](api.md)** — every public function and class.

## Quick start

```bash
make image                     # build the pinned dolfinx container
make test                      # fast test suite
python examples/turek/run.py       # settings come from examples/turek/turek2d.yaml
```

## Requirements

dolfinx `>=0.10,<0.11`, PETSc with MUMPS and HYPRE, MPI, and `numpy`, `pyyaml`, `pint`. `gmsh` is needed by the geometry generators, `scipy` and `matplotlib` by parts of the test suite.

!!! warning "The dolfinx version pin is not cosmetic"
    The solver uses the `kind="nest"` block-assembly API and
    `V.element.interpolation_points` as a *property*.  Neither exists in the 0.8 series
    that is still tagged `dolfinx/dolfinx:stable`, so the code will not import there.
