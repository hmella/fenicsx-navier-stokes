# Development notes

Stabilized incompressible Navier–Stokes in FEniCSx for vascular haemodynamics. See `README.md` for usage and `docs/theory.md` for the scheme.

## Tech stack (verified)

- **dolfinx 0.10.0**, ufl 2025.2.0, basix 0.10.0, PETSc 3.24 — image `dolfinx/dolfinx:v0.10.0`.
- The image ships **MPICH 4.3.1**, not OpenMPI: `mpirun -n N` needs no `--oversubscribe` or `--allow-run-as-root`, and `OMPI_*` env vars are no-ops.
- Host `python3` has no dolfinx. Everything runs in Docker (`make image`, `make shell`).
- Pin is load-bearing: the code uses `create_matrix(..., kind="nest")`, `assemble_vector(..., kind="nest")` and `V.element.interpolation_points` as a *property*. None exist in 0.8 (`dolfinx/dolfinx:stable`).
- The image carries the environment only. The source is mounted at `/work` and reached through `PYTHONPATH`, so the image is never stale. `PYTHONPATH` must be *prepended* to the base image's value rather than assigned, since the base image uses it to expose dolfinx and PETSc; overwriting it breaks `import dolfinx`. Dependencies are in `docker/requirements.txt`, and `.dockerignore` keeps the build context near zero.

## dolfinx 0.10 API notes

- `dolfinx.io.gmshio` → `dolfinx.io.gmsh`; `model_to_mesh` returns a `MeshData` object (`.mesh`, `.cell_tags`, `.facet_tags`), not a 3-tuple.
- `Function.vector` is gone → `Function.x.petsc_vec` (or `dolfinx.la.petsc.create_vector_wrap`).
- `Expression` cannot infer the communicator from an expression with no domain (e.g. UFL simplifies `gamma**(n-1)` to a literal when `n == 1`); pass `comm=` explicitly.

## Commands

```bash
make help                  # every target
make image / shell / test / test-slow / test-mpi / lint / docs / examples
docker run --rm -u "$(id -u):$(id -g)" -v "$PWD":/work -w /work fenicsx-ns-dev bash -c "..."
```

Always pass `-u "$(id -u):$(id -g)"`. Without it the container runs as root and everything it writes into the mounted tree — simulation output, caches, built docs — is owned by root and cannot be removed by the host user. The Makefile passes it by default, and `make fix-permissions` reclaims earlier leftovers. The image sets `PYTHONDONTWRITEBYTECODE=1` and points the FFCx, matplotlib and gmsh caches at `/tmp` so it works under any UID.

## Testing gotchas

- A `-m` expression on the command line **replaces** the one in `addopts`, it does not combine with it. `-m "not mpi"` silently re-enables the slow tier; write `-m "not mpi and not slow and not benchmark"`.
- Never compare summed local counts of mesh entities across MPI ranks. A facet shared between ranks is counted more than once, so the difference between two such sums is meaningless and can come out negative. Compare index sets rank-locally and reduce the result.

## Conventions

- Standard PEP8, 4-space indent, ~100 columns. `ruff check` is what CI enforces. `ruff format` is deliberately not run: it explodes multi-argument calls into one argument per line, which reads worse than hand-wrapping them. Prose in comments and docstrings is written as **continuous lines**, never hard-wrapped to a column; `E501` is off in `pyproject.toml` for that reason, and `ruff format` does not touch comment content.
- `set_problem` is deliberately one long method with labelled sections, mirroring the original script. Splitting it into `_setup_time` / `_assemble_operators` / `_setup_solver` made it harder to follow; the only helper kept is `_assemble_system`, which runs once per Picard iteration.

- Package layout `src/fenicsx_navier_stokes/`, lowercase module names, NumPy-style docstrings.
- Tolerance tiers live in `tests/conftest.py`: `TOL_EXACT` 1e-13, `TOL_SOLVE` 1e-7, `TOL_DISCRETE` 5e-2. Name the tier in the assertion.
- pytest markers: `slow`, `mpi`, `benchmark`, `regression`. `addopts` excludes `slow`/`benchmark`. `filterwarnings = error::RuntimeWarning` is deliberate — it is what keeps a silently-`nan` convergence criterion from returning.
- Convergence-rate assertions carry both a lower *and* an upper bound.

## Gotchas

- **Example module names must be unique across examples.** `examples/*/mesh.py` twice collided in `sys.modules`: the DFG tests silently meshed a tube and still passed. Hence `turek_mesh.py` / `tube_mesh.py`, and `load_example_module()` in `conftest.py`.
- **Facet tags differ per mesh.** Aorta `WallID` is 4 (9/11 mm) but 1 (13 mm) — correct metadata, but a config/mesh mismatch gives *no* BCs rather than an error. `PicardNSProblem._check_tags` refuses at construction; `tests/unit/test_meshdata.py` covers it. In parallel, gather tag values across ranks before checking — a rank need not own facets of every tag.
- **`inlet_lap_paraboloid` limits run opposite to the name.** `lam → 0` is a *plug*, `lam → ∞` is the paraboloid. Shape factor runs 0.5 → 1.
- **`VelocityProfileScale` is negative for the tube**: the inlet profile follows the *outward* cap normal.
- **Global mass balance is exact by construction** (a constant pressure test function annihilates every PSPG term), so it confirms assembly, not solution quality. Use `postprocess.divergence_norm` for the local defect.
- **The tube case is a designed verification**: `Rd*C = 1/(4π)` matches the `sin²(2πt)` waveform. Change them together or the case stops verifying anything.
- **The Windkessel coupling is first order in `dt`** (measured 0.994–0.999) because `Q` is frozen across a step. Refining `NbIters` buys nothing.
- Docker-run containers write root-owned files into `output/`; the original `output/` from the pre-repo folder is root-owned and gitignored.

## Defects fixed while packaging this code (do not reintroduce)

- **`A11_star` staleness.** The PSPG (1,1) block depends on the iterate through `tau_M` and must be re-assembled every nonlinear iteration. Treating it as constant leaves it at its `tau_M(up = 0)` value, so the continuity row stops weighting a single consistent strong residual. The effect is masked at small time steps, where the transient term of `tau_M` dominates, and shows up on the physical cases where the convective term is comparable.
- **PETSc options are global.** Every problem instance takes its own `ns<N>_` options prefix; without one, two problems in the same process silently share solver settings.
- SUPG momentum block carried `rho²` while its RHS counterpart carried `rho` — inconsistent at any `rho != 1`. Caught by the patch test **on a perturbed mesh only**: on a structured mesh `tau_M` is constant per cell and the error telescopes away.
- `tau_M` viscous term used `Ck*mu²/h⁴` (dimensionally wrong) → `Ck*(mu/(rho*h²))²`.
- Windkessel residual divided by `Pd`, which is exactly 0 at cold start → `nan`, and `nan > tol` is False, silently dropping the criterion. Now `|ΔPd|/(P_ATOL + |Pd|)`.
- `Windkessel.RK4` used `arange(0, dt+h, h)`, overshooting to `dt+h` for many `(dt, Niter)` pairs (~9e-5 relative error at `dt=0.3, Niter=1000`).
- `BackflowStab` class was unusable (4 independent bugs) → pure-UFL `backflow_stab()`.
- `ParameterHandler.__getattr__` recursed on `deepcopy`.
- `PowerLaw` defaulted to `n=0` (i.e. `mu = m/gamma`, not Newtonian) → `n=1`.

## Left as-is, deliberately

- `sigma_BDF = 2.0` vs the 3/2 BDF2 coefficient. Not a bug: `tau_M` multiplies a residual that vanishes on the exact solution, so it cannot affect consistency or order. Exposed as a documented parameter.
- Velocity BCs passed to the `a10`/`a11` assembly. For `a10` (Q×V) dolfinx zeroes the *columns* — correct and necessary. For `a11` (Q×Q) they match neither space and are dropped; the list is still passed because a genuine *pressure* BC in it does apply.
- `supg_viscous=True`. Measured to have **no** effect for P1 (identical MMS errors to four significant figures with it on and off); its coefficient is `tau_M*mu ~ dt*mu`.

## Measured results (regenerate with `make test-slow`)

- Newton's Jacobian against finite differences: relative error falls linearly with the step, 4.007e-05 → 4.023e-09, then round-off. That linear scaling is the proof it is exact.
- **Picard is faster than Newton on the aorta as configured.** 250 steps through systole, 8 ranks each: Picard 523 iterations / 2522 s, Newton 516 / 3289 s. Both converge in **2 iterations** on 240 of 250 steps, so Newton saves 1.3% of iterations while costing 32% more per iteration (6.37 s against 4.82 s). The cause is `NonlinearTolerance: 1.0e-2` with `dt = 8e-4`: there is almost no nonlinear work to save. Newton wins during the start-up transient from rest (steps 1-4: 3,3,3 against 7,5,4) and on genuinely harder problems -- 29% faster on a 2D 60x60 case at `Re ~ 1000`. Do not generalise from a short run near `t = 0`; a four-step measurement suggested Newton was 37% faster and that was an artefact of the transient.
- MMS at `rho=1.06`, 8/16/32 crossed: L²(u) rate **2.00**, H¹(u) **1.06**, L²(p) 3.08 (pre-asymptotic). P2-P1: 3 and 2.
- Inlet profile vs modified-Bessel shape factor: error ≤0.011 for `beta ≤ 13.3` on a 90k-cell graded cylinder; `beta = 40` needs ~290k cells.
- Aorta 11 mm (640k tets, iterative solver): ~18 KSP iterations, ~113 s/step serial.

## Not covered (and why)

- **Poiseuille**: with the stress-divergence viscous term the do-nothing outlet is inconsistent with a fully developed profile, so even P2-P1 is not exact — the discrepancy is a boundary-condition artefact, not a scheme error. Would need an imposed outlet traction.
- **Womersley**: the inlet is `fixed spatial profile × scalar(t)`, but the Womersley profile changes *shape* over the cycle.
- **Aorta regression baseline**: no trusted reference solution exists; ships diagnostics only.
