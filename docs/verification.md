# Verification

What is actually checked, against what, and how accurate it is.

## Manufactured solution

On $(0,1)^2$ with $\rho = 1.06$, $\mu = 0.04$:

$$u = g(t)\bigl(\sin^2\pi x\,\sin 2\pi y,\; -\sin 2\pi x\,\sin^2\pi y\bigr),\quad
  p = g(t)\sin\pi x\,\cos\pi y,\quad g = \cos 2\pi t,$$

with the body force derived symbolically. The solution vanishes on all four sides (no boundary interpolation error in the rate), is divergence-free analytically, and has zero mean pressure.

Measured on 8/16/32 crossed meshes:

| norm | errors | rate | theory |
|---|---|---|---|
| $L^2(u)$ | 2.46e-2, 6.12e-3, 1.53e-3 | **2.00** | 2 |
| $H^1(u)$ | 9.69e-1, 4.67e-1, 2.25e-1 | **1.06** | 1 |
| $L^2(p)$ | 6.16e-1, 8.31e-2, 8.61e-3 | 3.08 | 1 |

The velocity rates match theory. The pressure figure is pre-asymptotic: its coarse-mesh error is still of the same order as the solution itself, so it is not a genuine third-order result.

Taylor–Hood (P2-P1) on the same problem gives $L^2(u)$ rate 3 and $H^1(u)$ rate 2.

!!! note "The density is deliberately not 1"
    A mismatch in the power of $\rho$ between the two halves of the stabilized residual is
    invisible at $\rho = 1$. Any benchmark run at unit density would miss it.

## Patch test

A uniform flow $u = c$ at both history levels and on the whole boundary is an exact solution with an identically zero strong residual, so every stabilization term must contribute nothing. Recovering $c$ to $10^{-11}$ verifies the $3/2 : -2 : 1/2$ BDF2 coefficients, the consistency of each stabilization right-hand side with its matrix term, the grad–div term and the Dirichlet lifting — in about two seconds.

The variant on a **perturbed** mesh is the one with teeth. On a structured mesh $\tau_M$ is identical in every cell, so a density mismatch telescopes into a boundary term and vanishes for interior test functions; the test passes with the bug present. Perturbing the interior vertices removes that cancellation.

## Windkessel

The RK4 integration is checked against closed-form solutions: a constant flow rate (agrees to $8\times10^{-15}$, fourth order in the sub-step count) and a $\sin^2$ drive against

$$P(t) = R_dQ_0\Bigl[\bigl(\tfrac{R_p}{R_d}+\tfrac12\bigr)\sin^2\tfrac{\tau}{2}
        + \tfrac14\bigl(1 - e^{-\tau} - \sin\tau\bigr)\Bigr],\quad \tau = t/(R_dC),$$

which was verified independently against a stiff ODE integrator to $2\times10^{-11}$.

Separately, the *coupling* is measured to be **first order in $\Delta t$** (fitted rates 0.994–0.999), because the flow rate is frozen across a step. Refining `Niter` does not help; refining `dt` does.

The full 3D tube case asserts both that the recorded pressure/flow series satisfies the RCR ODE, and that it matches the closed form to within 10 % of $R_dQ_0$.

## Inlet profile

The screened-Poisson profile is compared against its modified-Bessel solution through the mean-to-peak shape factor $S(\beta) = [1 - (2/\beta)I_1(\beta)/I_0(\beta)]/[1 - 1/I_0(\beta)]$.

Measured on a graded cylinder (~90k cells, $R = 0.5$):

| $\beta = R/\lambda$ | exact | numeric | error |
|---|---|---|---|
| 0.025 | 0.5000 | 0.4964 | 0.004 |
| 1 | 0.5102 | 0.5059 | 0.004 |
| 4 | 0.6234 | 0.6146 | 0.009 |
| 13.3 | 0.8558 | 0.8446 | 0.011 |
| 40 | 0.9506 | 0.8269 | 0.124 |

The last row is the honest limit of this mesh: at $\beta = 40$ the boundary layer is $R/40$ wide and spans about one cell. Refining to ~290k cells brings every error below 0.01.

## Assembly

Structural checks that are exact rather than approximate: the constant-block + `axpy` shortcut against a from-scratch assembly of the combined form; $A_{10} = A_{01}^\mathsf{T}$ before and after boundary conditions; $A_{00}$ symmetric; the PSPG block negative semidefinite; the BDF2 history coefficients pinned individually against the mass matrix; and $\int\nabla\cdot u = \sum_\text{tags}\int u\cdot n$ to $10^{-12}$.

## Benchmarks

**DFG 2D-3.** Reference values (John 2004): $c_{D,\max} = 2.9509$, $c_{L,\max} = 0.478$, $\Delta p(8\,\text{s}) = -0.1116$. Reaching them needs millions of degrees of freedom — a first-order Crouzeix–Raviart computation needed ~785k dofs and $\Delta t = 0.0025$ to land within 0.8 % on drag and 9 % on $\Delta p$, and stabilized P1-P1 is not a better element. The automated tests therefore assert a physical envelope, agreement between the two drag formulations, and that refinement moves the answer toward the reference. The full comparison is marked `benchmark` and is not run automatically.

## What is not covered

- **Poiseuille flow.** With the stress-divergence viscous term the do-nothing outlet is not consistent with a fully developed profile, so the exact solution is not reproduced even by P2-P1, and the discrepancy is an artefact of the boundary condition rather than of the scheme. A meaningful test would need an explicitly imposed outlet traction.
- **Womersley flow.** The inlet machinery is `fixed spatial profile × scalar(t)`, but the Womersley profile changes *shape* over the cycle, so it cannot be represented without a time-dependent analytic inlet.
- **A regression baseline for the aorta.** No trusted reference solution exists; the case ships physical diagnostics instead.
