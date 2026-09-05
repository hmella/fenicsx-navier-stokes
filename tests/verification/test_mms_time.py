r"""The extrapolated nonlinear scheme, measured against the Picard one.

``Solver.NonlinearScheme = "extrapolated"`` replaces the Picard iteration with a single linear solve per step, linearizing the convection about :math:`2u^n - u^{n-1}`. It is off by default; this file is what there is to justify turning it on, and it is deliberately explicit about what it does *not* establish.

**What is measured here.** That the scheme runs, that it agrees closely with the converged Picard solution on the same mesh and time step, and that the gap between them shrinks when ``dt`` is halved. That is enough to say the two are the same discretization to within the extrapolation error, and it would catch a sign error, a wrong extrapolation, or a ``tau_M`` left inconsistent between terms.

**What is not measured, and why.** The obvious study -- refine ``dt`` and fit the order of the change in the solution -- does not work for this scheme, and it is worth recording why so nobody re-derives it. ``tau_M`` contains :math:`(\\sigma/\\Delta t)^2`, and at the operating points that matter that term dominates: on the mesh below, the transient part of ``tau_M`` accounts for 99.7 % of it at ``dt = 2e-3``, so ``tau_M`` is proportional to ``dt`` to three digits. Refining ``dt`` therefore changes the *spatial* stabilization operator as well as the time discretization, and the difference between two solutions at different ``dt`` mixes the two effects. Running that study on the Picard scheme -- which is fully implicit in the convection and should show a clean BDF2 rate -- gives a measured rate of 0.72, which is the study failing, not BDF2 failing.

So the temporal order of the extrapolated scheme is **not verified**. Establishing it would need a reference solution at a much smaller ``dt`` on the same mesh with ``tau_M`` held fixed, which is a different experiment from anything the test suite currently runs.
"""

import numpy as np
import pytest
from basix.ufl import element
from conftest import default_params, tag_all_boundaries
from dolfinx.fem import Constant, Expression, Function, assemble_scalar, form, functionspace
from dolfinx.mesh import CellType, DiagonalType, create_unit_square
from mpi4py import MPI
from ufl import SpatialCoordinate, as_vector, cos, div, grad, inner, pi, sin, sym
from ufl import dx as ufl_dx

from fenicsx_navier_stokes import PicardNSProblem, pressure_pin_bc

RHO, MU = 1.06, 0.04
N_CELLS = 16
T_END = 2.4e-2
STEP_COUNTS = (12, 24)


def manufactured(mesh, t_const):
    """The same divergence-free, boundary-vanishing solution ``test_mms_space`` uses."""
    x = SpatialCoordinate(mesh)
    g = cos(2 * pi * t_const)
    dg = -2 * pi * sin(2 * pi * t_const)
    shape = as_vector(
        [
            sin(pi * x[0]) ** 2 * sin(2 * pi * x[1]),
            -sin(2 * pi * x[0]) * sin(pi * x[1]) ** 2,
        ]
    )
    u = g * shape
    p = g * sin(pi * x[0]) * cos(pi * x[1])
    f = RHO * dg * shape + RHO * grad(u) * u - div(2 * MU * sym(grad(u))) + grad(p)
    return u, p, f


def solve_to_T(comm, n_steps, scheme):
    """Integrate to ``T_END`` in ``n_steps`` steps; return the problem and the final velocity.

    Both history levels are seeded from the exact solution, so the extrapolation is second order from the very first step it is used on rather than starting from two identical levels.
    """
    dt = T_END / n_steps
    mesh = create_unit_square(comm, N_CELLS, N_CELLS, CellType.triangle,
                              diagonal=DiagonalType.crossed)
    ft = tag_all_boundaries(mesh, wall=1, inlet=2)

    t_const = Constant(mesh, 0.0)
    u_ex, _, f = manufactured(mesh, t_const)

    pars = default_params(density=RHO, viscosity=MU, kind="direct", wall=1, inlet=2,
                          outlets=(), nl_tol=1e-11, max_nl=40)
    pars.Solver.NonlinearScheme = scheme

    VE = element("Lagrange", mesh.topology.cell_name(), 1, shape=(2,))
    zero = Function(functionspace(mesh, VE))
    zero.x.array[:] = 0.0

    problem = PicardNSProblem(
        parameters=pars, mesh=mesh, XDMF=False, boundaries=ft, element="P1-P1",
        dt=dt, num_steps=n_steps, f=f,
        inlet_profile=zero, inlet_scale=lambda k, t: 0.0,
        extra_bcs=lambda V, Q: [pressure_pin_bc(Q, 0.0)],
    )

    interp = problem.V.element.interpolation_points
    for target, time in ((problem.u0, 0.0), (problem.u00, -dt)):
        t_const.value = time
        target.interpolate(Expression(u_ex, interp))
        target.x.scatter_forward()
    problem.up.x.array[:] = problem.u0.x.array

    def advance_time(_prob, _step, t):
        t_const.value = t

    history = problem.solve(store_after=None, verbose=False, pre_step=advance_time)
    assert all(h["converged"] for h in history), f"{scheme} did not converge"
    return problem, problem.u_h.x.array.copy(), history


def l2_norm(problem, values):
    """``||values||_L2`` interpreted on the problem's velocity space."""
    w = Function(problem.V)
    w.x.array[:] = values
    w.x.scatter_forward()
    dx = ufl_dx(domain=problem.mesh, metadata={"quadrature_degree": 4})
    return np.sqrt(problem.mesh.comm.allreduce(
        assemble_scalar(form(inner(w, w) * dx)), op=MPI.SUM
    ))


@pytest.mark.slow
def test_extrapolated_matches_picard_and_costs_one_solve(comm):
    """One linear solve per step, and a solution close to the converged Picard one.

    The tolerance is absolute and generous, because what is being asserted is that the two schemes solve the same problem, not that they agree to solver precision -- they cannot, since one iterates the convection to convergence and the other does not.
    """
    prob_p, u_p, hist_p = solve_to_T(comm, STEP_COUNTS[1], "picard")
    prob_e, u_e, hist_e = solve_to_T(comm, STEP_COUNTS[1], "extrapolated")

    # Step 0 has no second history level to extrapolate from and falls back to Picard;
    # every step after it must be a single solve.
    assert [h["method"] for h in hist_e][0] == "picard"
    assert all(h["method"] == "extrapolated" for h in hist_e[1:])
    assert all(h["iterations"] == 1 for h in hist_e[1:])
    assert sum(h["iterations"] for h in hist_p) > 2 * len(hist_p), (
        "Picard converged in one iteration a step, so this comparison proves nothing"
    )

    rel = l2_norm(prob_p, u_p - u_e) / l2_norm(prob_e, u_e)
    assert rel < 1.0e-5, f"schemes disagree by {rel:.2e} relative"


@pytest.mark.slow
def test_gap_to_picard_shrinks_with_dt(comm):
    """Halving ``dt`` brings the two schemes closer together.

    They differ only through the extrapolation error, so refining must close the gap. This is the property that says the option is a cheaper approximation of the same problem: a sign error in the extrapolation, or a ``tau_M`` built on a different field from the one the convection uses, leaves a gap that does not shrink.

    Only the direction is asserted, not a rate. See the module docstring: ``tau_M`` is proportional to ``dt`` at this operating point, so refining ``dt`` also changes the spatial operator, and a fitted order would not be measuring what it appears to.
    """
    gaps = []
    for n_steps in STEP_COUNTS:
        prob_p, u_p, _ = solve_to_T(comm, n_steps, "picard")
        _, u_e, _ = solve_to_T(comm, n_steps, "extrapolated")
        gaps.append(l2_norm(prob_p, u_p - u_e))

    report = f"\n  gaps {gaps} at dt {[T_END / n for n in STEP_COUNTS]}"
    assert gaps[1] < gaps[0], f"refining dt did not close the gap{report}"
