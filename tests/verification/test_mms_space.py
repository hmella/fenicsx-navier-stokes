r"""Spatial convergence against a manufactured solution.

This is the test that actually verifies the discretization, as opposed to checking that a particular exact solution is reproduced.  On ``(0, 1)^2``:

.. math::

    u = g(t)\\bigl(\\sin^2(\\pi x)\\sin(2\\pi y),\; -\\sin(2\\pi x)\\sin^2(\\pi y)\\bigr), \\qquad p = g(t)\\sin(\\pi x)\\cos(\\pi y), \\qquad g(t) = \\cos(2\\pi t),

with the body force ``f`` derived symbolically from the strong form.

Three properties make this choice better than Taylor-Green or Kovasznay here:

* ``u`` vanishes on all four sides, so no boundary interpolation error contaminates the
  measured rate;
* ``div u == 0`` analytically;
* ``integral(p) == 0`` over the domain, so the mean-subtracted pressure comparison is exact.

``f`` is built in UFL from ``SpatialCoordinate`` and never interpolated, so it carries no discretization error of its own.

The density is deliberately ``1.06`` rather than ``1``.  A mismatch in the power of ``rho`` between the implicit and explicit halves of the stabilized residual is invisible at ``rho == 1`` -- every benchmark that runs at unit density would miss it.
"""

import numpy as np
import pytest
from basix.ufl import element
from conftest import default_params, rate, tag_all_boundaries
from dolfinx.fem import Constant, Function, assemble_scalar, form, functionspace
from dolfinx.mesh import CellType, DiagonalType, create_unit_square
from mpi4py import MPI
from ufl import SpatialCoordinate, as_vector, cos, div, grad, inner, pi, sin, sym
from ufl import dx as ufl_dx

from fenicsx_navier_stokes import PicardNSProblem, pressure_pin_bc

RHO, MU = 1.06, 0.04
DT = 1.0e-4
N_STEPS = 3


def manufactured(mesh, t_const):
    """Return ``(u_exact, p_exact, f)`` as UFL expressions in space and ``t_const``."""
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

    # Strong residual of the momentum equation, with the same stress form the solver uses.
    f = RHO * dg * shape + RHO * grad(u) * u - div(2 * MU * sym(grad(u))) + grad(p)
    return u, p, f


def errors(problem, u_ex, p_ex):
    """Velocity L2 and H1 errors and the mean-subtracted pressure L2 error."""
    mesh = problem.mesh
    comm = mesh.comm
    dx = ufl_dx(domain=mesh, metadata={"quadrature_degree": 6})

    def integrate(expr):
        return comm.allreduce(assemble_scalar(form(expr)), op=MPI.SUM)

    eu = problem.u_h - u_ex
    l2_u = np.sqrt(integrate(inner(eu, eu) * dx))
    h1_u = np.sqrt(integrate(inner(grad(eu), grad(eu)) * dx))

    # The discrete pressure is fixed only up to a constant, so compare mean-free fields.
    area = integrate(Constant(mesh, 1.0) * dx)
    mean_h = integrate(problem.p_h * dx) / area
    mean_e = integrate(p_ex * dx) / area
    ep = (problem.p_h - mean_h) - (p_ex - mean_e)
    l2_p = np.sqrt(integrate(ep * ep * dx))
    return l2_u, h1_u, l2_p


def run_mms(comm, n, element_pair="P1-P1", dt=DT, n_steps=N_STEPS, supg_viscous=True):
    """Solve the manufactured problem on an ``n x n`` mesh and return the errors."""
    mesh = create_unit_square(comm, n, n, CellType.triangle, diagonal=DiagonalType.crossed)
    ft = tag_all_boundaries(mesh, wall=1, inlet=2)

    t_const = Constant(mesh, 0.0)
    u_ex, p_ex, f = manufactured(mesh, t_const)

    pars = default_params(
        density=RHO,
        viscosity=MU,
        kind="direct",
        wall=1,
        inlet=2,
        outlets=(),
        nl_tol=1e-11,
        max_nl=40,
    )

    deg = 2 if element_pair == "P2-P1" else 1
    VE = element("Lagrange", mesh.topology.cell_name(), deg, shape=(2,))
    zero = Function(functionspace(mesh, VE))
    zero.x.array[:] = 0.0

    problem = PicardNSProblem(
        parameters=pars,
        mesh=mesh,
        XDMF=False,
        boundaries=ft,
        element=element_pair,
        dt=dt,
        num_steps=n_steps,
        f=f,
        # u vanishes on the entire boundary, so both the "inlet" and the wall are zero.
        inlet_profile=zero,
        inlet_scale=lambda k, t: 0.0,
        extra_bcs=lambda V, Q: [pressure_pin_bc(Q, 0.0)],
        supg_viscous=supg_viscous,
    )

    # Seed both history levels from the exact solution: a cold start would make the first step first-order and pollute the measurement.
    t0 = 0.0
    interp = problem.V.element.interpolation_points
    from dolfinx.fem import Expression

    for target, time in ((problem.u0, t0), (problem.u00, t0 - dt)):
        t_const.value = time
        target.interpolate(Expression(u_ex, interp))
        target.x.scatter_forward()
    problem.up.x.array[:] = problem.u0.x.array

    def advance_time(_prob, _step, t):
        t_const.value = t

    history = problem.solve(store_after=None, verbose=False, pre_step=advance_time)
    assert all(h["converged"] for h in history), "Picard did not converge; rates are meaningless"

    t_const.value = t0 + n_steps * dt
    return errors(problem, u_ex, p_ex)


@pytest.mark.slow
def test_p1p1_spatial_rates(comm):
    """P1-P1 with SUPG/PSPG: second order in L2(u), first order in H1(u).

    Measured on 8/16/32 crossed meshes at ``rho = 1.06``:

    =======  ==========================  ======
    norm     errors                      rate
    =======  ==========================  ======
    L2(u)    2.46e-2, 6.12e-3, 1.53e-3   2.00 H1(u)    9.69e-1, 4.67e-1, 2.25e-1   1.06 L2(p)    6.16e-1, 8.31e-2, 8.61e-3   3.08
    =======  ==========================  ======

    The velocity rates match the theory exactly.  The pressure converges much faster than the ``O(h)`` the theory guarantees, but its coarse-mesh error is still of the same order as the solution itself, so that figure is pre-asymptotic rather than a genuine third-order result -- hence a lower bound that bites and an upper bound that is only a sanity guard.  A rate far above the theoretical one usually means the exact solution has landed in the finite element space and the test has stopped measuring anything.
    """
    sizes = [8, 16, 32]
    results = [run_mms(comm, n) for n in sizes]
    h = [1.0 / n for n in sizes]

    l2_u = [r[0] for r in results]
    h1_u = [r[1] for r in results]
    l2_p = [r[2] for r in results]

    r_l2u, r_h1u, r_l2p = rate(l2_u, h), rate(h1_u, h), rate(l2_p, h)
    report = (
        f"\n  L2(u) {l2_u} rate {r_l2u:.2f}"
        f"\n  H1(u) {h1_u} rate {r_h1u:.2f}"
        f"\n  L2(p) {l2_p} rate {r_l2p:.2f}"
    )

    assert all(a > b for a, b in zip(l2_u, l2_u[1:], strict=False)), f"L2(u) not decreasing{report}"
    assert 1.80 <= r_l2u <= 2.40, f"L2(u) rate {r_l2u:.2f}{report}"
    assert 0.88 <= r_h1u <= 1.40, f"H1(u) rate {r_h1u:.2f}{report}"
    assert 0.88 <= r_l2p <= 3.60, f"L2(p) rate {r_l2p:.2f}{report}"


@pytest.mark.slow
def test_p2p1_spatial_rates(comm):
    """Taylor-Hood: third order in L2(u), second in H1(u) and L2(p)."""
    sizes = [4, 8, 16]
    results = [run_mms(comm, n, element_pair="P2-P1") for n in sizes]
    h = [1.0 / n for n in sizes]

    r_l2u = rate([r[0] for r in results], h)
    r_h1u = rate([r[1] for r in results], h)
    assert 2.6 <= r_l2u <= 3.5, f"L2(u) rate {r_l2u:.2f}"
    assert 1.7 <= r_h1u <= 2.5, f"H1(u) rate {r_h1u:.2f}"


def test_mms_error_is_small_on_a_fine_mesh(comm):
    """A single-resolution smoke check, cheap enough for the pull-request job.

    The convergence studies above are marked slow; this keeps the manufactured solution in the fast suite so a gross consistency regression is caught on every change.
    """
    l2_u, h1_u, l2_p = run_mms(comm, 16)
    assert l2_u < 2.0e-2, f"L2(u) = {l2_u:.3e}"
    assert h1_u < 5.0e-1, f"H1(u) = {h1_u:.3e}"
    assert l2_p < 2.0e-1, f"L2(p) = {l2_p:.3e}"
