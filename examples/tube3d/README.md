# 3D tube with a Windkessel outlet

A straight vessel segment (SimVascular-style): a cylinder of radius `R` and length `L` along the z axis, driven by a prescribed inflow waveform, with a three-element RCR Windkessel on the outlet.

```bash
python run.py --radius 2.0 --length 30.0 --resolution 0.5
python tube_mesh.py --radius 2.0 --length 30.0 --output ../../data/tube3d/tube  # mesh only
```

Facet tags are `wall=1`, `inlet=2`, `outlet=3`, matching `tube3d.yaml`.

## This is a verification case

With the shipped parameters the outlet pressure has a closed form. The waveform in `data/inflow/tube.flow` is `sin²(2πt)` and the configuration sets `Rd*C = 1/(4π)`, so `sin²(τ/2) = sin²(2πt)` with `τ = t/(Rd C)`, and

```
P(t) = Rd*Q0*[ (Rp/Rd + 1/2) sin²(τ/2) + (1/4)(1 − exp(−τ) − sin τ) ]
```

`tests/benchmarks/test_tube.py` asserts against this.

**Change the Windkessel parameters and the waveform together, or not at all.** Changing one alone leaves the case running perfectly well while silently ceasing to verify anything.

## Accuracy

The limiting factor is the coupling, not the ODE integrator. The flow rate is held frozen across a time step, which makes the scheme first order in `dt` however large `NbIters` is. Refining `dt` improves the answer; refining `NbIters` past a few tens does not.

## Mesh grading

`--wall-resolution` refines toward the wall. Anything sensitive to a boundary layer needs it — the Womersley solution, for instance, has a Stokes layer of thickness `sqrt(2 mu/(rho omega))` that must span several cells.

## Sign convention

`VelocityProfileScale` is **negative**. The inlet profile is built along the *outward* normal of the inlet cap, so a negative scale drives flow into the domain.
