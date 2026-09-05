"""The SUPG/PSPG stabilization parameters.

``tau_M`` is checked against a value computed by hand on a mesh whose element diameter is known exactly, and against its three asymptotic limits.  The scaling test is the one that matters most: it is what distinguishes a dimensionally consistent viscous term from the plausible-looking but wrong ``Ck*mu**2/h**4``.
"""

import numpy as np
import pytest
from conftest import TOL_EXACT
from dolfinx.fem import Constant, assemble_scalar, form
from mpi4py import MPI
from ufl import dx as ufl_dx

from fenicsx_navier_stokes.stabilization import Ck_constant, tau_C, tau_M

# On create_unit_square(1, 1, diagonal=right) both cells are right triangles with legs of length 1, so CellDiameter (the longest vertex-to-vertex distance) is exactly sqrt(2).
#
# tau_M = ((2/0.1)^2 + 0 + 30*(1.0/(1.0*2))^2)^(-1/2) = (400 + 7.5)^(-1/2)
TAU_M_REFERENCE = 0.04953774046180699


def integrate(mesh, expr):
    """Integrate a scalar expression over the whole mesh (area 1 for the unit square)."""
    return mesh.comm.allreduce(assemble_scalar(form(expr * ufl_dx)), op=MPI.SUM)


def zero_velocity(mesh):
    from basix.ufl import element
    from dolfinx.fem import Function, functionspace

    V = functionspace(
        mesh, element("Lagrange", mesh.topology.cell_name(), 1, shape=(mesh.geometry.dim,))
    )
    u = Function(V)
    u.x.array[:] = 0.0
    return u


def test_Ck_values():
    """``Ck = 60 * 2**(k-2)``."""
    assert Ck_constant(1) == 30.0
    assert Ck_constant(2) == 60.0


def test_tau_M_reference_value(unit_square_1):
    """Reproduce a hand-computed value exactly."""
    mesh = unit_square_1
    up = zero_velocity(mesh)
    tM = tau_M(
        up, Constant(mesh, 0.1), Constant(mesh, 1.0), Constant(mesh, 1.0), mesh=mesh, degree=1
    )
    assert integrate(mesh, tM) == pytest.approx(TAU_M_REFERENCE, rel=1e-13)


def test_tau_M_transient_limit(unit_square_1):
    """As ``dt -> 0`` the transient term dominates and ``tau_M -> dt/sigma``."""
    mesh = unit_square_1
    up = zero_velocity(mesh)
    dt = 1.0e-8
    tM = tau_M(up, Constant(mesh, dt), Constant(mesh, 1.0), Constant(mesh, 1.0), mesh=mesh)
    assert integrate(mesh, tM * (2.0 / dt)) == pytest.approx(1.0, abs=1e-9)


def test_tau_M_convective_limit(unit_square_1):
    """With a large velocity, ``tau_M -> h/|u|``."""
    mesh = unit_square_1
    up = zero_velocity(mesh)
    up.x.array[::2] = 1.0e6  # x-component only
    up.x.scatter_forward()
    tM = tau_M(up, Constant(mesh, 1.0), Constant(mesh, 1.0), Constant(mesh, 1.0), mesh=mesh)
    h = np.sqrt(2.0)
    assert integrate(mesh, tM * (1.0e6 / h)) == pytest.approx(1.0, abs=1e-7)


def test_tau_M_viscous_limit(unit_square_1):
    """With ``dt -> inf`` and no convection, ``tau_M -> h^2 rho / (mu sqrt(Ck))``."""
    mesh = unit_square_1
    up = zero_velocity(mesh)
    rho, mu = 1.06, 0.04
    tM = tau_M(up, Constant(mesh, 1.0e12), Constant(mesh, rho), Constant(mesh, mu), mesh=mesh)
    h2 = 2.0
    expected = h2 * rho / (mu * np.sqrt(Ck_constant(1)))
    assert integrate(mesh, tM) == pytest.approx(expected, rel=1e-9)


def test_tau_M_depends_only_on_kinematic_viscosity(unit_square_1):
    """Scaling ``mu`` and ``rho`` together must leave ``tau_M`` unchanged.

    In the viscous limit ``tau_M`` depends on the fluid only through ``nu = mu/rho``, so doubling both leaves it alone.  Writing that term with the dynamic viscosity instead (``Ck*mu**2/h**4``) is dimensionally inconsistent with the other two contributions and makes ``tau_M`` change by the square of the scale factor -- which this catches.
    """
    mesh = unit_square_1
    up = zero_velocity(mesh)
    big_dt = Constant(mesh, 1.0e12)

    base = integrate(mesh, tau_M(up, big_dt, Constant(mesh, 1.06), Constant(mesh, 0.04), mesh=mesh))
    scaled = integrate(
        mesh, tau_M(up, big_dt, Constant(mesh, 10.6), Constant(mesh, 0.4), mesh=mesh)
    )
    assert scaled == pytest.approx(base, rel=1e-10)


def test_tau_M_sigma_is_a_free_parameter(unit_square_1):
    """``sigma_BDF`` changes tau_M by exactly the predicted amount.

    It is a parameter of the stabilization, not a discretization coefficient: ``tau_M`` multiplies a residual that vanishes on the exact solution, so changing it cannot affect consistency or the order of convergence, only the magnitude of the added terms.
    """
    mesh = unit_square_1
    up = zero_velocity(mesh)
    dt = 0.1
    dtc = Constant(mesh, dt)
    t2 = integrate(
        mesh, tau_M(up, dtc, Constant(mesh, 1.0), Constant(mesh, 1.0), mesh=mesh, sigma_BDF=2.0)
    )
    t15 = integrate(
        mesh, tau_M(up, dtc, Constant(mesh, 1.0), Constant(mesh, 1.0), mesh=mesh, sigma_BDF=1.5)
    )
    visc = Ck_constant(1) * (1.0 / (1.0 * 2.0)) ** 2
    assert t2 == pytest.approx(((2.0 / dt) ** 2 + visc) ** -0.5, rel=1e-13)
    assert t15 == pytest.approx(((1.5 / dt) ** 2 + visc) ** -0.5, rel=1e-13)
    assert t15 > t2


def test_tau_M_requires_mesh_or_h():
    with pytest.raises(ValueError, match="element size"):
        tau_M(None, 1.0, 1.0, 1.0)


def test_tau_C_is_viscous_scaled():
    """``tau_C * rho == coefficient * mu`` exactly."""
    assert tau_C(1.06, 0.04) == pytest.approx(0.4 * 0.04 / 1.06, rel=TOL_EXACT)
    assert tau_C(2.0, 1.0, coefficient=1.0) * 2.0 == pytest.approx(1.0, rel=TOL_EXACT)
