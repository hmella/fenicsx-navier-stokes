# Theory

## Governing equations

Incompressible Navier–Stokes on a domain $\Omega$:

$$\rho\bigl(\partial_t u + (u\cdot\nabla)u\bigr)
  - \nabla\cdot\bigl(2\mu\,\varepsilon(u)\bigr) + \nabla p = f, \qquad \nabla\cdot u = 0,$$

with $\varepsilon(u) = \tfrac12(\nabla u + \nabla u^\mathsf{T})$.  The viscous term is in **stress-divergence** form, not the Laplacian form.  That choice matters at outflow boundaries: the natural ("do-nothing") condition it induces is $(-pI + 2\mu\varepsilon(u))\cdot n = 0$, which is *not* satisfied by a fully developed Poiseuille profile, so a do-nothing outlet perturbs the solution near the exit.

## Time discretization

BDF2, with the convective term linearized about the previous nonlinear iterate $u^{*}$:

$$\rho\,\frac{3u^{n+1} - 4u^{n} + u^{n-1}}{2\Delta t}
  + \rho\,(u^{*}\cdot\nabla)u^{n+1}
  - \nabla\cdot\bigl(2\mu\varepsilon(u^{n+1})\bigr) + \nabla p^{n+1} = f.$$

Both history levels start at zero, so a cold start is BDF2 applied to two identical zero levels — consistent for cases that ramp from rest, but first-order on the very first step. [`initial_condition`][fenicsx_navier_stokes.problems.PicardNSProblem.initial_condition] seeds them when that matters.

## Stabilization

Equal-order P1-P1 spaces violate the inf-sup condition. The scheme adds residual-based SUPG and PSPG terms plus a grad–div term, with

$$\tau_M = \Bigl[\Bigl(\frac{\sigma}{\Delta t}\Bigr)^2
                 + \Bigl(\frac{|u^{*}|}{h}\Bigr)^2
                 + C_k\Bigl(\frac{\mu}{\rho h^2}\Bigr)^2\Bigr]^{-1/2}, \qquad C_k = 60\cdot 2^{k-2},\qquad \tau_C = 0.4\,\frac{\mu}{\rho}.$$

All three contributions to $\tau_M$ have units of inverse time. The viscous one uses the *kinematic* viscosity $\mu/\rho$; writing it with the dynamic viscosity is dimensionally inconsistent, and happens to be masked whenever the transient term dominates.

Every stabilization term weights the strong residual, which vanishes on the exact solution. Consistency therefore hinges on the implicit and explicit halves of that residual carrying **the same power of $\rho$** — a mismatch is invisible at $\rho = 1$ and shows up as a loss of accuracy at any other density. The uniform-flow patch test on a *non-uniform* mesh is the cheapest detector; on a structured mesh the error telescopes into a boundary term and hides completely.

!!! note "$\sigma$ is a free parameter"
    $\sigma = 2$ is the conventional Tezduyar value, and is what the physical cases were
    run with. Since $\tau_M$ multiplies a residual that vanishes on the exact solution,
    changing it cannot affect consistency or the order of convergence — only the size of
    the added terms, and hence quantitative outputs such as drag.

## Linear algebra

The $2\times2$ saddle-point system is a PETSc `MatNest`:

$$\begin{bmatrix} A & B^\mathsf{T} \\ B & -C \end{bmatrix}
  \begin{bmatrix} u \\ p \end{bmatrix} = \begin{bmatrix} f \\ g \end{bmatrix}$$

with $B = -\nabla\cdot$ and $C \succeq 0$ from PSPG. Only the blocks depending on $u^{*}$ are re-assembled each Picard iteration; the BDF2 mass-plus-viscous block and the two divergence blocks are assembled once and added back with `axpy`.

Two solver paths, chosen by `Solver.Kind`:

`direct` : `preonly` + `lu` with MUMPS. The right choice for 2D and small 3D cases.

`iterative` : FGMRES preconditioned by `pc_fieldsplit_type schur`, `schur_fact_type lower`,
  `schur_precondition selfp`, with one BoomerAMG V-cycle on each block. Required for the patient-specific meshes. The sign convention above is what `selfp` assumes; flipping it still assembles and still solves directly, but silently destroys the preconditioner.

## Boundary conditions

**Inlet.** Rather than assuming a circular cap, the profile solves a screened-Poisson problem $u - \lambda^2\Delta u = 1$ with $u = 0$ on the wall, normalises by the global maximum, restricts to the cap and multiplies by the discrete facet normal. The limits run opposite to the intuition the name "paraboloid" suggests:

- $\lambda \to 0$ gives a **plug** with a boundary layer of width $\lambda$;
- $\lambda \to \infty$ gives the **Poiseuille paraboloid**.

On a circular cap the exact profile is $\hat p(r) = [1 - I_0(r/\lambda)/I_0(\beta)] / [1 - 1/I_0(\beta)]$ with $\beta = R/\lambda$, and its mean-to-peak shape factor runs from $1/2$ (paraboloid) to $1$ (plug).

**Outlets.** A three-element RCR Windkessel per outlet,

$$C\,\frac{\mathrm{d}P_d}{\mathrm{d}t} = Q - \frac{P_d}{R_d},
  \qquad P_{\mathrm{out}} = R_pQ + P_d,$$

integrated with RK4 across each step and imposed as a constant normal traction. The coupling is explicit within one linear solve but iterated to consistency inside the Picard loop. Accuracy is limited by that coupling, not the integrator: $Q$ is frozen across the step, so the scheme is **first order in $\Delta t$** regardless of the number of RK4 sub-steps.

Optionally a backflow traction $s(u) = \tfrac12\rho\beta(u\cdot n - |u\cdot n|)\,u$ is added, exactly zero during outflow and dissipative during inflow.

## Mass conservation

Two different statements, easily confused:

- **Global.** $\int_\Omega \nabla\cdot u = 0$ to round-off, *by construction*: testing the continuity row with the constant pressure function annihilates every stabilization term, since each carries $\nabla q$. This confirms the assembly and the facet tags, nothing more.
- **Local.** P1-P1 with PSPG is not pointwise divergence-free. [`divergence_norm`][fenicsx_navier_stokes.postprocess.divergence_norm] measures it. The grad–div term here is a bulk viscosity that does not scale with $h|u|$, so its control weakens as the Reynolds number grows.
