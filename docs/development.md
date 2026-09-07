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
- **`Windkessel.flow_rate` compiled its form once and then ignored its argument.** A compiled form is bound to the coefficients it was built from, so reusing one form for a different function silently returns the first function's flux. Latent until `coupling_residual` began asking for the flux of the solution after `update` had bound the form to the iterate: the residual then came out identically zero and the Windkessel term dropped out of the convergence test without failing. Cache is now keyed on the function.
- `BackflowStab` class was unusable (4 independent bugs) → pure-UFL `backflow_stab()`.
- `ParameterHandler.__getattr__` recursed on `deepcopy`.
- `PowerLaw` defaulted to `n=0` (i.e. `mu = m/gamma`, not Newtonian) → `n=1`.

## Left as-is, deliberately

- `sigma_BDF = 2.0` vs the 3/2 BDF2 coefficient. Not a bug: `tau_M` multiplies a residual that vanishes on the exact solution, so it cannot affect consistency or order. Exposed as a documented parameter.
- Velocity BCs passed to the `a10`/`a11` assembly. For `a10` (Q×V) dolfinx zeroes the *columns* — correct and necessary. For `a11` (Q×Q) they match neither space and are dropped; the list is still passed because a genuine *pressure* BC in it does apply.
- `supg_viscous=True`. Measured to have **no** effect for P1 (identical MMS errors to four significant figures with it on and off); its coefficient is `tau_M*mu ~ dt*mu`.

## Measured results (regenerate with `make test-slow`)

- Newton's Jacobian against finite differences: relative error falls linearly with the step, 4.007e-05 → 4.023e-09, then round-off. That linear scaling is the proof it is exact.
- **Newton is faster than Picard on the aorta, and is the default there.** 40 steps at 8 ranks: Newton 5.12 s/step against Picard 7.48. Newton carries the exact rank-one Windkessel term `Z_k a_k a_k^T`, and once the convergence test measures the real 0D/3D coupling residual that term earns its keep.

  This reverses an earlier measurement recorded here, and the reason it was wrong is worth keeping. Under the old convergence test both schemes were pinned at exactly two sweeps a step -- the test could not pass on the first sweep and the coupling error it reported was already below tolerance on the second -- so Newton had nothing to accelerate and lost on its 32% higher per-iteration cost. The old numbers (250 systolic steps: Picard 523 iterations / 2522 s, Newton 516 / 3289 s) were correct measurements of a comparison that the stopping test had rendered meaningless. Newton also wins on genuinely harder problems: 29% faster on a 2D 60x60 case at `Re ~ 1000`.
- MMS at `rho=1.06`, 8/16/32 crossed: L²(u) rate **2.00**, H¹(u) **1.06**, L²(p) 3.08 (pre-asymptotic). P2-P1: 3 and 2.
- Inlet profile vs modified-Bessel shape factor: error ≤0.011 for `beta ≤ 13.3` on a 90k-cell graded cylinder; `beta = 40` needs ~290k cells.
- Aorta 11 mm (643k tets, iterative solver): ~113 s/step serial before the performance work; see the Performance section for the current parallel numbers.

## Performance (aorta, 8 ranks, 643k tets / 445k dofs)

Regenerate the profile with `make profile STEPS=30 NP=8`; the `fxns_*` events and the `fxns_time_loop` stage in `output/profile.txt` are what to read.

**Where the time goes now** (30 steps, per cent of the time-loop stage):

| phase | share |
|---|---|
| `fxns_assemble_A` — the four Jacobian/Oseen blocks | 59% |
| `KSPSolve` | 21% |
| `fxns_assemble_b` — the right-hand side | 15% |
| `PCSetUp` + `PCSetUpOnBlocks` | 3.5% |
| `fxns_callback` — the diagnostics | 0.9% |

**Assembly, not the linear solver, is now the bottleneck.** Before the solver work it was the other way round: `KSPSolve` was 31% and two BoomerAMG hierarchies were rebuilt on every nonlinear iteration for another 13%. Anyone continuing this should start from `fxns_assemble_A`.

**Measured effect** (100 steps, production configuration, against the commit before this work). Compare like windows: the first ten steps run 9, 7, 6, 4 nonlinear iterations while the flow starts from rest, so a mean over all 100 differs from the steady-state mean.

| | all 100 steps | steps 10+ | s/nonlinear iteration |
|---|---|---|---|
| baseline | 8.96 | 8.02 | 4.02 |
| new code, old AMG settings and no reuse | 5.11 | — | — |
| new code | 3.45 | 3.05 | 1.53 |

**2.6x either way**, and both configurations take exactly 2.00 nonlinear iterations per step from step 10 on, so the per-iteration figure is a clean comparison. Re-measured back to back on an idle machine, both numbers reproduce to better than 0.5%.

That middle row splits the credit: everything other than the AMG settings and preconditioner reuse (quadrature degree, nonzero initial guess, the assembly items) is worth 8.96 → 5.11, and the AMG settings plus reuse are worth a further 5.11 → 3.45. Note that the tuned AMG runs at *more* Krylov iterations, not fewer -- 11.7 against 9.3 per solve -- trading iteration count for a much cheaper setup and V-cycle.

**These changes do not alter the solution.** Verified by running both codes at `NonlinearTolerance: 1e-6` and `SolverATol: 1e-12` for 20 steps: every outlet flow, flow split, pressure and the divergence norm agree to **6.4e-6**, i.e. to the nonlinear tolerance.

**At the production `NonlinearTolerance: 1.0e-2` the same comparison shows up to 6% difference in outlet pressure, and that is not caused by any of this work.** The check that establishes it: take the *same* code and vary only settings that provably cannot change the converged answer -- preconditioner reuse off, AMG back to `strong_threshold 0.25` with Falgout and classical interpolation. Those two runs differ by **10.7%** in `P_out_6_mmHg`, more than the 6.0% between old code and new. Stopping the Picard iteration at a 1% relative change simply does not pin the outlet pressures better than about ten per cent during the start-up transient. Any A/B of the discretization has to be run at a tight nonlinear tolerance, and two aorta runs should not be treated as comparable at the percent level.

**Nothing else got slower.** The tube3d verification case pays 0.6% for its tighter tolerances (20.29 s against 20.42 s over 25 steps) and gains a factor of 66 in the mass defect. The Turek 2D case uses the direct solver and is untouched, at 5.6 s for its 20-step smoke size. `make test` went from 60.5 s to 56.7 s.

### Things that were tried and did not work

- **`pc_fieldsplit_schur_precondition a11`.** Tempting, because PSPG puts a `tau_M`-weighted pressure Laplacian in the (1,1) block and Cahouet–Chabard says that is the right Schur preconditioner at this operating point (`rho*1.5/dt ~ 2000` against a viscous `mu/h^2 ~ 16`). It diverges on the first solve (`KSP_DIVERGED_DTOL`): that block is a pure-Neumann Laplacian, singular on constants, where `selfp`'s approximation is not. A non-singular variant would shift it by a pressure mass matrix.
- **A Cauchy-in-`dt` study of the temporal order.** Invalid for this scheme: `tau_M` contains `(sigma/dt)^2`, and the transient term accounts for 99.7% of it at the settings tested, so `tau_M` is proportional to `dt` and refining the time step changes the *spatial* stabilization too. Run on the Picard scheme, which should show a clean BDF2 rate, it measures 0.72.

### The nonlinear convergence test measures a residual, not a change

Both halves of the criterion used to be posed on the change between successive iterates, and that made them unable to pass on a first sweep no matter how good the solution already was.

`nl_error` is `||up - u_h|| / ||u_h||`. With `up` seeded from `u^n` the first sweep reports how far the flow moved in one time step -- around 6e-2 on the aorta against a 1e-2 tolerance. `PicardNSProblem._extrapolate_iterate` now seeds it with `2u^n - u^{n-1}` instead, which makes that first number the actual extrapolation error: measured 1.6e-3, so the velocity half of the test passes on the first sweep. This changes only where the iteration starts, never the fixed point it converges to.

`wk_error` was `|Pd_nl - Pd_nl_prev|`, the difference between two successive *impositions* of the outlet pressure. `Windkessel.coupling_residual` replaces it with the mismatch between the traction that was imposed and the pressure the returned velocity implies, which is zero only at the fixed point of the coupled system.

**That reveals the old test was optimistic.** The old measure never saw the `Rp*Q` term, which responds instantly to velocity with none of the RCR's damping, so it was blind to the fast half of the coupling. Measured per sweep in steady state:

| sweep | `nl_error` | `wk_error` (true residual) |
|---|---|---|
| 0 | 1.63e-3 | 3.36e-2 |
| 1 | 1.81e-3 | 3.97e-2 (rises) |
| 2 | 1.17e-3 | 1.18e-2 |
| 3 | 5.84e-4 | 6.50e-3 |

Four sweeps to reach 1e-2, where the old proxy stopped at two. **Aorta results produced before this change were converged to roughly 1-4% in the coupling, not the 1% their configuration asked for.** The honest test costs 1.7x with Newton (5.12 s/step against 3.04), which is the price of the tolerance meaning what it says.

The coupling residual contracts at only 0.3-0.55 and is not monotonic, so Aitken relaxation on the outlet pressure is the obvious next lever if this cost matters. It looked pointless while the proxy criterion was hiding the iteration it would accelerate.

### Tolerances: check which one is binding

`SolverRTol` and `SolverATol` are both in play and the absolute one wins more often than expected. On the aorta **every accepted solve stops on `CONVERGED_ATOL`**, so `SolverRTol: 1.0e-8` is decoration and changing it does nothing. The `ksp_reason` column in the diagnostics CSV records this (2 = ATOL, 3 = RTOL); read it before tuning either.

The same effect bit the tube verification case. `mass_balance` is a near-total cancellation of `Q_in` against the outflows, so it amplifies whatever residual the linear solve leaves behind, and the old configuration passed its `< 1e-6` assertion only because a zero initial guess made every solve overshoot well past its tolerance. With a nonzero initial guess the solve stops where it is told to, `SolverATol: 1.0e-10` binds first, and the defect rises to 6.7e-6. Tightening both tolerances for that case (it is 6k cells and half a second a step) puts it at 2.4e-9, twenty times better than the original.

## Not covered (and why)

- **Poiseuille**: with the stress-divergence viscous term the do-nothing outlet is inconsistent with a fully developed profile, so even P2-P1 is not exact — the discrepancy is a boundary-condition artefact, not a scheme error. Would need an imposed outlet traction.
- **Womersley**: the inlet is `fixed spatial profile × scalar(t)`, but the Womersley profile changes *shape* over the cycle.
- **Aorta regression baseline**: no trusted reference solution exists; ships diagnostics only.
