"""The backflow stabilization traction.

The term must be *exactly* inactive during outflow -- not merely small -- and dissipative during inflow.  Both are checked against closed-form values on a uniform field, where the surface integrals can be written down by hand.
"""

import numpy as np
import pytest
from basix.ufl import element
from conftest import tag_square_boundaries
from dolfinx.fem import Constant, Function, assemble_scalar, form, functionspace
from dolfinx.fem.petsc import assemble_vector
from dolfinx.mesh import CellType, DiagonalType, create_unit_square
from mpi4py import MPI
from petsc4py import PETSc
from ufl import FacetNormal, Measure, TestFunction, dot

from fenicsx_navier_stokes import backflow_stab

RHO, BETA, OUTLET = 1.06, 0.2, 3


@pytest.fixture(scope="module")
def setup(comm):
    mesh = create_unit_square(comm, 8, 8, CellType.triangle, diagonal=DiagonalType.crossed)
    ft = tag_square_boundaries(mesh)
    V = functionspace(mesh, element("Lagrange", mesh.topology.cell_name(), 1, shape=(2,)))
    ds = Measure("ds", domain=mesh, subdomain_data=ft, metadata={"quadrature_degree": 6})
    return mesh, V, ds


def uniform(V, vec):
    u = Function(V)
    u.interpolate(lambda x: np.vstack([np.full(x.shape[1], c) for c in vec]))
    u.x.scatter_forward()
    return u


def stab(mesh, u):
    return backflow_stab(u, FacetNormal(mesh), Constant(mesh, RHO), Constant(mesh, BETA))


def test_exactly_zero_for_outflow(setup):
    """At the x=1 outlet the outward normal is +e_x, so u = +c e_x is pure outflow."""
    mesh, V, ds = setup
    u = uniform(V, (1.3, 0.0))
    v = TestFunction(V)
    b = assemble_vector(form(dot(stab(mesh, u), v) * ds(OUTLET)))
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    assert b.norm(PETSc.NormType.INFINITY) == 0.0
    b.destroy()


def test_closed_form_value_for_inflow(setup):
    """For ``u = -c e_x`` the traction is ``rho*beta*c**2 e_x`` over a unit-length edge.

    ``dot(u, n) = -c``, so the prefactor ``0.5*(u.n - |u.n|)`` is ``-c`` and ``s = rho*beta*(-c)*u = rho*beta*c**2 e_x``.  Testing against ``u`` itself gives the energy contribution ``integral(dot(s, u)) = -rho*beta*c**3 * |edge|``.
    """
    mesh, V, ds = setup
    c = 1.3
    u = uniform(V, (-c, 0.0))
    val = mesh.comm.allreduce(assemble_scalar(form(dot(stab(mesh, u), u) * ds(OUTLET))), op=MPI.SUM)
    assert val == pytest.approx(-RHO * BETA * c**3 * 1.0, rel=1e-10)


def test_is_dissipative(setup):
    """Any field with inflow gives a negative energy contribution."""
    mesh, V, ds = setup
    u = Function(V)
    u.interpolate(lambda x: np.vstack([-0.5 - x[1], 0.2 * np.ones_like(x[0])]))
    u.x.scatter_forward()
    val = mesh.comm.allreduce(assemble_scalar(form(dot(stab(mesh, u), u) * ds(OUTLET))), op=MPI.SUM)
    assert val < 0.0


def test_beta_zero_disables_the_term(setup):
    mesh, V, ds = setup
    u = uniform(V, (-1.3, 0.0))
    s = backflow_stab(u, FacetNormal(mesh), Constant(mesh, RHO), Constant(mesh, 0.0))
    val = mesh.comm.allreduce(assemble_scalar(form(dot(s, u) * ds(OUTLET))), op=MPI.SUM)
    assert val == 0.0


def test_only_the_inflow_part_contributes(setup):
    """With half the boundary flowing in and half out, only the inflow half contributes.

    ``u_x = y - 0.5`` reverses at the mid-height of the outlet edge.  The exact value is ``integral over y in [0, 0.5] of -rho*beta*(0.5-y)^3 dy = -rho*beta*0.5^4/4``.
    """
    mesh, V, ds = setup
    u = Function(V)
    u.interpolate(lambda x: np.vstack([x[1] - 0.5, np.zeros_like(x[0])]))
    u.x.scatter_forward()
    val = mesh.comm.allreduce(assemble_scalar(form(dot(stab(mesh, u), u) * ds(OUTLET))), op=MPI.SUM)
    expected = -RHO * BETA * 0.5**4 / 4.0
    assert val == pytest.approx(expected, rel=2e-2)


def test_quadrature_degree_is_adequate(setup):
    """``|u.n|`` is not smooth across the reversal line, so check the integral is settled.

    If a low surface quadrature degree changed this materially, the production setting (``SurfaceQuadratureDegree: 3``) would be under-resolving the term.
    """
    mesh, V, _ = setup
    ft = tag_square_boundaries(mesh)
    u = Function(V)
    u.interpolate(lambda x: np.vstack([x[1] - 0.5, np.zeros_like(x[0])]))
    u.x.scatter_forward()

    vals = []
    for deg in (3, 8):
        ds = Measure("ds", domain=mesh, subdomain_data=ft, metadata={"quadrature_degree": deg})
        vals.append(
            mesh.comm.allreduce(
                assemble_scalar(form(dot(stab(mesh, u), u) * ds(OUTLET))), op=MPI.SUM
            )
        )
    assert abs(vals[0] - vals[1]) <= 0.02 * abs(vals[1])
