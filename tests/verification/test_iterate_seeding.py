r"""Starting the nonlinear iteration from a second-order extrapolation.

`PicardNSProblem._extrapolate_iterate` seeds `up` with :math:`2u^n - u^{n-1}` instead of :math:`u^n` at the top of each time step. That fixes where the fixed-point iteration starts, not where it ends, so the two properties tested here are **fewer sweeps** and **the same answer**.

`nl_error` is :math:`\\|up - u_h\\| / \\|u_h\\|`, the relative change from the starting iterate, so the first sweep reports the extrapolation error.

`Windkessel.coupling_residual` supplies the other half of the criterion: the imposed outlet traction against the pressure the returned velocity implies, which is zero only at the fixed point of the coupled system.
"""

import numpy as np
import pytest
from basix.ufl import element
from conftest import default_params, tag_all_boundaries, tag_square_boundaries
from dolfinx.fem import Constant, Expression, Function, assemble_scalar, form, functionspace
from dolfinx.mesh import CellType, DiagonalType, create_unit_square
from mpi4py import MPI
from petsc4py import PETSc
from ufl import SpatialCoordinate, as_vector, cos, div, grad, inner, pi, sin, sym
from ufl import dx as ufl_dx

from fenicsx_navier_stokes import PicardNSProblem, Windkessel, pressure_pin_bc

RHO, MU = 1.06, 0.04
N_CELLS = 16
DT = 2.0e-3
N_STEPS = 12


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


def run(comm, seeded, nl_tol=1e-11):
    """Integrate the manufactured problem, optionally with the seeding disabled.

    The seeding is not a configurable option, so it is disabled by replacing the method on
    the instance; the unseeded path exists only as a comparison for this test.
    """
    mesh = create_unit_square(comm, N_CELLS, N_CELLS, CellType.triangle,
                              diagonal=DiagonalType.crossed)
    ft = tag_all_boundaries(mesh, wall=1, inlet=2)

    t_const = Constant(mesh, 0.0)
    u_ex, _, f = manufactured(mesh, t_const)

    pars = default_params(density=RHO, viscosity=MU, kind="direct", wall=1, inlet=2,
                          outlets=(), nl_tol=nl_tol, max_nl=40)

    VE = element("Lagrange", mesh.topology.cell_name(), 1, shape=(2,))
    zero = Function(functionspace(mesh, VE))
    zero.x.array[:] = 0.0

    problem = PicardNSProblem(
        parameters=pars, mesh=mesh, XDMF=False, boundaries=ft, element="P1-P1",
        dt=DT, num_steps=N_STEPS, f=f,
        inlet_profile=zero, inlet_scale=lambda k, t: 0.0,
        extra_bcs=lambda V, Q: [pressure_pin_bc(Q, 0.0)],
    )
    if not seeded:
        problem._extrapolate_iterate = lambda n_time: None

    # Seed both history levels from the exact solution, so the extrapolation is second order
    # from the first step it is used on rather than starting from two identical levels
    interp = problem.V.element.interpolation_points
    for target, time in ((problem.u0, 0.0), (problem.u00, -DT)):
        t_const.value = time
        target.interpolate(Expression(u_ex, interp))
        target.x.scatter_forward()
    problem.up.x.array[:] = problem.u0.x.array

    def advance_time(_prob, _step, t):
        t_const.value = t

    history = problem.solve(store_after=None, verbose=False, pre_step=advance_time)
    assert all(h["converged"] for h in history)
    return problem, problem.u_h.x.array.copy(), history


def l2_norm(problem, values):
    w = Function(problem.V)
    w.x.array[:] = values
    w.x.scatter_forward()
    dx = ufl_dx(domain=problem.mesh, metadata={"quadrature_degree": 4})
    return np.sqrt(problem.mesh.comm.allreduce(
        assemble_scalar(form(inner(w, w) * dx)), op=MPI.SUM
    ))


@pytest.mark.slow
def test_seeding_does_not_move_the_fixed_point(comm):
    """Same converged solution, to solver precision.

    The seeding sets only the starting iterate; the Oseen operator, the stabilization and the convergence criterion are unchanged, so at a tight nonlinear tolerance the two runs agree to far better than that tolerance. A looser agreement means the starting point has leaked into the answer, which happens when `tau_M` or the SUPG weight reads a different velocity from the convection.
    """
    prob_a, u_seeded, _ = run(comm, seeded=True)
    _, u_plain, _ = run(comm, seeded=False)

    rel = l2_norm(prob_a, u_seeded - u_plain) / l2_norm(prob_a, u_plain)
    assert rel < 1.0e-10, f"seeding moved the converged solution by {rel:.2e}"


@pytest.mark.slow
def test_seeding_costs_fewer_sweeps(comm):
    """Strictly fewer nonlinear iterations, at the same tolerance.

    The tolerance has to sit between the two starting errors for the comparison to measure anything. Over one step the manufactured solution moves by about ``2*pi*dt``, roughly 1.3e-2 here, while the second-order extrapolation starts within ``O(dt^2)`` of the solution. A tolerance of 1e-3 falls between them, so the unseeded run needs a second sweep and the seeded one does not.
    """
    _, _, hist_seeded = run(comm, seeded=True, nl_tol=1e-3)
    _, _, hist_plain = run(comm, seeded=False, nl_tol=1e-3)

    seeded = sum(h["iterations"] for h in hist_seeded)
    plain = sum(h["iterations"] for h in hist_plain)
    assert seeded < plain, f"seeded {seeded} iterations, unseeded {plain}"


class TestCouplingResidual:
    """``Windkessel.coupling_residual`` measures the 0D/3D mismatch, not an iterate difference."""

    @staticmethod
    def make(comm):
        mesh = create_unit_square(comm, 6, 6, CellType.triangle, diagonal=DiagonalType.crossed)
        ft = tag_square_boundaries(mesh)
        pars = default_params(density=1.06, viscosity=0.04, kind="direct",
                              nl_tol=1e-12, max_nl=40)
        wksl = Windkessel(Pd_prev=0.0, Rd=1.0, Rp=0.1, C=1.0 / (4.0 * np.pi), cap_id=3,
                          Niter=50, facet_tags=ft, dt_sim=1.0e-2,
                          P_out=Constant(mesh, PETSc.ScalarType(0.0)))
        prob = PicardNSProblem(parameters=pars, mesh=mesh, XDMF=False, boundaries=ft,
                               windkessels=[wksl], element="P1-P1", dt=1.0e-2, num_steps=3,
                               inlet_scale=lambda k, t: 1.0)
        return prob, wksl

    def test_zero_when_the_models_agree(self, comm):
        """Imposing the pressure a velocity implies leaves no residual.

        ``update`` computes the outlet pressure from ``up`` and imposes it; the pressure that same field implies is the same number.
        """
        prob, wksl = self.make(comm)
        prob.up.interpolate(lambda x: np.vstack([0.4 + 0.0 * x[0], 0.1 * x[0]]))
        prob.up.x.scatter_forward()

        wksl.update(prob.up, verbose=False)
        assert wksl.coupling_residual(prob.up) == pytest.approx(0.0, abs=1e-14)

    def test_nonzero_for_a_different_function(self, comm):
        """The residual must read the function it is handed, not the one before it.

        This mirrors the solver: ``update`` is called on the iterate ``up``, the system is solved into ``u_h``, and the residual is asked about ``u_h``. A flux form cached against ``up`` and reused for ``u_h`` returns ``Q(up)`` for both, giving an identically zero residual and a Windkessel term that drops out of the convergence criterion.
        """
        prob, wksl = self.make(comm)
        prob.up.interpolate(lambda x: np.vstack([0.4 + 0.0 * x[0], 0.1 * x[0]]))
        prob.up.x.scatter_forward()
        wksl.update(prob.up, verbose=False)

        # A genuinely different velocity, in a different Function, as after a solve
        prob.u_h.x.array[:] = 2.0 * prob.up.x.array
        prob.u_h.x.scatter_forward()

        assert wksl.flow_rate(prob.u_h) == pytest.approx(2.0 * wksl.flow_rate(prob.up), rel=1e-12)
        assert wksl.coupling_residual(prob.u_h) > 1.0e-3

    def test_finite_at_cold_start(self, comm):
        """Zero flow and ``Pd_init = 0`` must give 0, not nan.

        The absolute floor in the denominator keeps the result finite, as it does in ``residual``.
        """
        prob, wksl = self.make(comm)
        prob.up.x.array[:] = 0.0
        prob.up.x.scatter_forward()
        wksl.update(prob.up, verbose=False)

        r = wksl.coupling_residual(prob.up)
        assert np.isfinite(r) and r == 0.0
