"""Uniform-flow patch test.

If the velocity is a constant vector ``c`` at both previous time levels and on the whole boundary, and there is no body force, then ``u = c``, ``p = 0`` solves the problem exactly and the strong momentum residual vanishes identically:

* BDF2       ``rho/dt * (1.5*c - 2*c + 0.5*c) == 0``
* convection ``(c . grad) c == 0``
* viscous    ``div(2 mu eps(c)) == 0``
* pressure   ``grad(0) == 0``
* continuity ``div(c) == 0``

Every SUPG/PSPG/grad-div term therefore weights a zero argument and must contribute nothing.  Recovering ``c`` to round-off is one cheap assertion that simultaneously verifies the ``3/2 : -2 : 1/2`` BDF2 coefficients, that each stabilization right-hand-side term is the exact consistency counterpart of its matrix term -- including the density scaling, which is where the two halves are easiest to get out of step -- the grad-div term, and the Dirichlet lifting.

The test is deliberately run at ``rho != 1``.  A density mismatch between the implicit and explicit halves of the stabilized residual is invisible at ``rho == 1``.
"""

import numpy as np
import pytest
from conftest import default_params, tag_all_boundaries
from dolfinx.fem import Constant, Function
from dolfinx.mesh import CellType, DiagonalType, create_unit_square
from mpi4py import MPI
from ufl import as_vector

from fenicsx_navier_stokes import PicardNSProblem, pressure_pin_bc

C_VEC = (0.7, -0.3)


def uniform_function(V, c):
    """A :class:`~dolfinx.fem.Function` equal to the constant vector ``c`` everywhere."""
    fn = Function(V)
    fn.interpolate(lambda x: np.vstack([np.full(x.shape[1], ci) for ci in c]))
    fn.x.scatter_forward()
    return fn


def global_max_abs(comm, arr):
    local = float(np.abs(arr).max()) if arr.size else 0.0
    return comm.allreduce(local, op=MPI.MAX)


def build_patch_problem(mesh, elem, dt, rho, mu, c=C_VEC):
    """A problem whose exact solution is the uniform flow ``u = c``, ``p = 0``."""
    ft = tag_all_boundaries(mesh, wall=1, inlet=2)
    pars = default_params(
        density=rho,
        viscosity=mu,
        kind="direct",
        wall=1,
        inlet=2,
        outlets=(),
        nl_tol=1e-12,
        max_nl=12,
    )
    uniform = as_vector([Constant(mesh, float(ci)) for ci in c])

    prob = PicardNSProblem(
        parameters=pars,
        mesh=mesh,
        XDMF=False,
        boundaries=ft,
        element=elem,
        dt=dt,
        num_steps=1,
        # The inlet profile is the constant field and the scale is 1, so the inlet carries the same value as the wall: u = c on the entire boundary.
        inlet_profile=uniform,
        inlet_scale=lambda n, t: 1.0,
        wall_velocity=uniform,
        extra_bcs=lambda V, Q: [pressure_pin_bc(Q, 0.0)],
    )
    return prob


@pytest.mark.parametrize("rho,mu", [(1.0, 0.001), (1.06, 0.04)])
@pytest.mark.parametrize("dt", [1.0e-3, 1.0e-1])
@pytest.mark.parametrize("elem", ["P1-P1", "P2-P1"])
def test_uniform_flow_is_reproduced_exactly(comm, rho, mu, dt, elem):
    mesh = create_unit_square(comm, 6, 6, CellType.triangle, diagonal=DiagonalType.crossed)
    prob = build_patch_problem(mesh, elem, dt, rho, mu)

    exact = uniform_function(prob.V, C_VEC)
    prob.initial_condition(u0=exact, u00=exact)
    prob.up.x.array[:] = exact.x.array

    history = prob.solve(store_after=None, verbose=False)
    assert history[0]["converged"]

    u_err = global_max_abs(comm, prob.u_h.x.array - exact.x.array)
    p_err = global_max_abs(comm, prob.p_h.x.array)

    assert u_err < 1e-11, f"velocity patch error {u_err:.3e}"
    assert p_err < 1e-9, f"pressure patch error {p_err:.3e}"


def test_patch_holds_over_several_steps(comm):
    """A uniform state must be stationary, not merely correct on the first step."""
    mesh = create_unit_square(comm, 4, 4, CellType.triangle, diagonal=DiagonalType.crossed)
    prob = build_patch_problem(mesh, "P1-P1", 5.0e-3, rho=1.06, mu=0.04)

    exact = uniform_function(prob.V, C_VEC)
    prob.initial_condition(u0=exact, u00=exact)
    prob.up.x.array[:] = exact.x.array

    prob.solve(store_after=None, verbose=False, max_steps=5)

    assert global_max_abs(comm, prob.u_h.x.array - exact.x.array) < 1e-11


@pytest.mark.parametrize("rho", [1.0, 1.06])
def test_uniform_flow_on_perturbed_mesh(comm, rho):
    """The patch test on a mesh with non-uniform cell sizes.

    On a structured mesh ``tau_M`` is identical in every cell, so a stabilization term with the wrong density scaling contributes ``const * integral(tau_M * ((c.grad) v) . c)``, which telescopes into a boundary integral and vanishes for every interior test function -- the defect hides completely. Perturbing the interior vertices makes ``tau_M`` vary from cell to cell and removes that cancellation, so this variant fails if the implicit and explicit halves of the stabilized residual carry different powers of ``rho``.
    """
    from conftest import perturb_interior_vertices

    mesh = create_unit_square(comm, 6, 6, CellType.triangle, diagonal=DiagonalType.crossed)
    perturb_interior_vertices(mesh, amplitude=0.25, seed=3)

    prob = build_patch_problem(mesh, "P1-P1", 1.0e-2, rho=rho, mu=0.04)
    exact = uniform_function(prob.V, C_VEC)
    prob.initial_condition(u0=exact, u00=exact)
    prob.up.x.array[:] = exact.x.array

    prob.solve(store_after=None, verbose=False)

    u_err = global_max_abs(comm, prob.u_h.x.array - exact.x.array)
    assert u_err < 1e-11, f"velocity patch error {u_err:.3e} on a non-uniform mesh"
