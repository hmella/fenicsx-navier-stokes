"""Strain-rate and viscosity models."""

import numpy as np
import pytest
from basix.ufl import element
from dolfinx.fem import Constant, Function, assemble_scalar, form, functionspace
from dolfinx.mesh import CellType, DiagonalType, create_unit_square
from mpi4py import MPI
from ufl import dx, inner

from fenicsx_navier_stokes.constitutive import Newtonian, PowerLaw, eps, powerLaw, shear_rate


@pytest.fixture(scope="module")
def mesh(comm):
    return create_unit_square(comm, 6, 6, CellType.triangle, diagonal=DiagonalType.crossed)


@pytest.fixture(scope="module")
def V(mesh):
    return functionspace(mesh, element("Lagrange", mesh.topology.cell_name(), 2, shape=(2,)))


@pytest.fixture(scope="module")
def Qs(mesh):
    return functionspace(mesh, element("Lagrange", mesh.topology.cell_name(), 1))


def norm2(mesh, expr):
    return mesh.comm.allreduce(assemble_scalar(form(inner(expr, expr) * dx)), op=MPI.SUM)


def field(V, fn):
    u = Function(V)
    u.interpolate(fn)
    u.x.scatter_forward()
    return u


class TestEps:
    def test_is_symmetric(self, mesh, V):
        u = field(V, lambda x: np.vstack([np.sin(x[0]) * x[1], x[0] ** 2 - x[1]]))
        assert norm2(mesh, eps(u) - eps(u).T) < 1e-24

    def test_trace_equals_divergence(self, mesh, V):
        from ufl import div, tr

        u = field(V, lambda x: np.vstack([x[0] ** 2, x[0] * x[1]]))
        a = mesh.comm.allreduce(assemble_scalar(form(tr(eps(u)) * dx)), op=MPI.SUM)
        b = mesh.comm.allreduce(assemble_scalar(form(div(u) * dx)), op=MPI.SUM)
        assert a == pytest.approx(b, rel=1e-13)

    def test_rigid_rotation_has_no_strain(self, mesh, V):
        """``u = (-y, x)`` is a rigid body rotation, so eps(u) == 0."""
        u = field(V, lambda x: np.vstack([-x[1], x[0]]))
        assert norm2(mesh, eps(u)) < 1e-22

    def test_shear_rate_of_simple_shear(self, mesh, V):
        """For ``u = (g*y, 0)`` the shear rate is exactly ``|g|``."""
        g = 2.0
        u = field(V, lambda x: np.vstack([g * x[1], np.zeros_like(x[0])]))
        val = mesh.comm.allreduce(assemble_scalar(form(shear_rate(u) * dx)), op=MPI.SUM)
        assert val == pytest.approx(abs(g), rel=1e-12)


class TestNewtonian:
    def test_viscosity_is_constant(self):
        model = Newtonian(0.04)
        assert model.viscosity() == 0.04
        assert model.viscosity(u="ignored") == 0.04

    def test_repr(self):
        assert "0.04" in repr(Newtonian(0.04))


class TestPowerLaw:
    def test_n_equals_one_is_newtonian(self, V, Qs):
        """``n = 1`` must reproduce a constant viscosity ``m`` regardless of the flow."""
        u = field(V, lambda x: np.vstack([np.sin(3 * x[0]) * x[1], x[0] * x[1]]))
        mu = powerLaw(u, Qs, m=0.04, n=1.0)
        assert np.allclose(mu.x.array, 0.04, rtol=1e-12)

    def test_simple_shear_value(self, V, Qs):
        """``mu = m * gamma**(n-1)`` with ``gamma = |g|`` for simple shear."""
        g, m, n = 2.0, 0.04, 0.6
        u = field(V, lambda x: np.vstack([g * x[1], np.zeros_like(x[0])]))
        mu = powerLaw(u, Qs, m=m, n=n)
        expected = m * g ** (n - 1.0)
        assert np.allclose(mu.x.array, expected, rtol=1e-10), (
            f"expected {expected}, got {mu.x.array.min()}..{mu.x.array.max()}"
        )

    def test_zero_shear_is_clamped_not_infinite(self, V, Qs):
        """A constant velocity gives zero shear rate; for ``n < 1`` the formula diverges.

        Relying on comparisons alone would leak non-finite values through, since ``nan > x`` is False.
        """
        m = 0.04
        u = field(V, lambda x: np.vstack([np.full(x.shape[1], 1.5), np.full(x.shape[1], -0.5)]))
        mu = powerLaw(u, Qs, m=m, n=0.6)
        assert np.isfinite(mu.x.array).all()
        assert np.allclose(mu.x.array, 1.0e4 * m, rtol=1e-12)

    def test_clamp_bounds_are_respected(self, V, Qs):
        m = 0.04
        u = field(V, lambda x: np.vstack([1e3 * x[0] ** 3, np.zeros_like(x[0])]))
        mu = powerLaw(u, Qs, m=m, n=0.3)
        assert (mu.x.array >= m / 1e4 - 1e-18).all()
        assert (mu.x.array <= m * 1e4 + 1e-18).all()

    def test_bounds_accepts_float_or_constant(self, mesh, V):
        u = Function(V)
        assert PowerLaw(u, m=0.04).bounds == pytest.approx((0.04 / 1e4, 0.04 * 1e4))
        c = PowerLaw(u, m=Constant(mesh, 0.04)).bounds
        assert c == pytest.approx((0.04 / 1e4, 0.04 * 1e4))

    def test_default_index_is_newtonian(self, V, Qs):
        """The default ``n`` must be 1.

        An earlier revision defaulted to ``n = 0``, which silently gives ``mu = m/gamma``:
        not Newtonian, and not a sensible default for anyone who omits the argument.
        """
        u = field(V, lambda x: np.vstack([np.sin(x[0]) * x[1], x[0] * x[1]]))
        mu = powerLaw(u, Qs, m=0.04)
        assert np.allclose(mu.x.array, 0.04, rtol=1e-12)

    def test_invalid_clamp_ratio(self, V):
        with pytest.raises(ValueError, match="clamp_ratio"):
            PowerLaw(Function(V), clamp_ratio=0.5)

    def test_update_copies_dofs(self, V):
        a, b = Function(V), Function(V)
        b.x.array[:] = 3.0
        model = PowerLaw(a)
        model.update(b)
        assert np.allclose(a.x.array, 3.0)
