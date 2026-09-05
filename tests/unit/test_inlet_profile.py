"""The screened-Poisson inlet velocity profile.

``inlet_lap_paraboloid`` solves ``u - lam^2 laplacian(u) = 1`` over the domain with ``u = 0`` on the wall, normalises by the global maximum, restricts the result to the inlet cap and multiplies by the discrete facet normal.

Note the direction of the limits, which the argument name ``plateau_lam`` reflects:

* ``lam -> 0`` gives a **plug** (``u -> 1`` in the interior with a thin boundary layer),
* ``lam -> infinity`` gives the **Poiseuille paraboloid** (``lam^2 (-laplacian u) -> 1``
  gives ``u = (R^2 - r^2)/(4 lam^2)``, which normalised is ``1 - r^2/R^2``).

On a cylinder the cap carries a natural (homogeneous Neumann) condition, so the solution is independent of the axial coordinate and coincides exactly with the two-dimensional radial solution on a disc:

.. math::

    \\hat p(r) = \\frac{1 - I_0(r/\\lambda)/I_0(\\beta)}{1 - 1/I_0(\\beta)}, \\qquad \\beta = R/\\lambda .

The single most informative quantity is its mean-to-peak **shape factor**

.. math::

    S(\\beta) = \\frac{1 - (2/\\beta) I_1(\\beta)/I_0(\\beta)}{1 - 1/I_0(\\beta)},

which tends to ``1/2`` for a paraboloid and to ``1`` for a plug.  Asserting it pins the auxiliary solve, the normalisation, the restriction to the cap and the surface integration in one number.
"""

import numpy as np
import pytest
from basix.ufl import element
from dolfinx.fem import Constant, assemble_scalar, form, functionspace
from mpi4py import MPI
from scipy.special import iv
from ufl import FacetNormal, Measure, dot

from fenicsx_navier_stokes import inlet_lap_paraboloid

WALL, INLET = 1, 2
RADIUS = 0.5


def shape_factor_exact(beta):
    """Mean-to-peak ratio of the exact radial profile."""
    return (1.0 - (2.0 / beta) * iv(1, beta) / iv(0, beta)) / (1.0 - 1.0 / iv(0, beta))


def cap_integrals(mesh, par):
    """Return ``(integral of par . n over the cap, cap area)``."""
    ds = Measure("ds", domain=mesh, subdomain_data=par._ft, metadata={"quadrature_degree": 4})
    n = FacetNormal(mesh)
    flux = mesh.comm.allreduce(assemble_scalar(form(dot(par, n) * ds(INLET))), op=MPI.SUM)
    area = mesh.comm.allreduce(assemble_scalar(form(Constant(mesh, 1.0) * ds(INLET))), op=MPI.SUM)
    return flux, area


def build_profile(cylinder_graded, lam):
    mesh, _, ft = cylinder_graded
    V = functionspace(mesh, element("Lagrange", mesh.topology.cell_name(), 1, shape=(3,)))
    par = inlet_lap_paraboloid(V, ft, INLET, WALL, plateau_lam=lam)
    par._ft = ft
    return mesh, par


@pytest.mark.parametrize("beta", [0.025, 1.0, 4.0, 13.3333])
def test_shape_factor_matches_bessel(cylinder_graded, beta):
    """The mean-to-peak ratio must follow the modified-Bessel prediction."""
    lam = RADIUS / beta
    mesh, par = build_profile(cylinder_graded, lam)
    flux, area = cap_integrals(mesh, par)

    numeric = abs(flux) / area
    exact = shape_factor_exact(beta)
    # Measured errors on this mesh run from 0.004 (beta = 0.025) to 0.011 (beta = 13.3), and shrink under refinement; 0.02 leaves headroom without being vacuous.
    assert numeric == pytest.approx(exact, abs=0.02), (
        f"beta={beta}: shape factor {numeric:.4f}, expected {exact:.4f}"
    )


def test_large_lambda_is_a_paraboloid(cylinder_graded):
    """``lam >> R`` recovers the Poiseuille paraboloid, whose shape factor is exactly 1/2."""
    mesh, par = build_profile(cylinder_graded, 20.0 * RADIUS)
    flux, area = cap_integrals(mesh, par)
    assert abs(flux) / area == pytest.approx(0.5, abs=0.01)


def test_small_lambda_is_a_plug(cylinder_graded):
    """``lam << R`` gives a plug: the mean approaches the peak."""
    mesh, par = build_profile(cylinder_graded, RADIUS / 40.0)
    flux, area = cap_integrals(mesh, par)
    # The exact shape factor at beta = 40 is 0.951, but the boundary layer is only lam = R/40 wide and this mesh spans it with roughly one cell, so the measured value is about 0.83 and climbs toward the exact one under refinement.  Assert the qualitative behaviour -- clearly plug-like, clearly not a paraboloid -- rather than a number the mesh cannot deliver.
    assert abs(flux) / area > 0.80


def test_peak_value_is_one(cylinder_graded):
    """The profile is normalised by its global maximum, so the peak is exactly 1."""
    mesh, par = build_profile(cylinder_graded, RADIUS / 4.0)
    peak = mesh.comm.allreduce(float(np.abs(par.x.array).max()), op=MPI.MAX)
    assert peak == pytest.approx(1.0, abs=1e-12)


def test_profile_is_normal_to_the_cap(cylinder_graded):
    """On the ``z = 0`` cap the velocity must be purely axial.

    The profile is built by multiplying a scalar by the discretely approximated facet normal, so a tangential component would mean the normal approximation is wrong.
    """
    mesh, par = build_profile(cylinder_graded, RADIUS / 4.0)
    vals = par.x.array.reshape(-1, 3)
    active = np.linalg.norm(vals, axis=1) > 1e-10
    if active.any():
        tangential = float(np.abs(vals[active][:, :2]).max())
    else:
        tangential = 0.0
    assert mesh.comm.allreduce(tangential, op=MPI.MAX) < 1e-6


def test_flow_direction_is_outward(cylinder_graded):
    """``integral(par . n)`` over the inlet is positive, i.e. par follows the outward normal.

    This is the convention that makes ``VelocityProfileScale: -1.0`` in the tube configuration drive flow *into* the domain.  Flipping it would reverse every case that uses this profile.
    """
    mesh, par = build_profile(cylinder_graded, RADIUS / 4.0)
    flux, _ = cap_integrals(mesh, par)
    assert flux > 0.0


@pytest.mark.parametrize("beta", [1.0, 4.0])
def test_matches_the_exact_radial_profile(cylinder_graded, beta):
    """Compare the profile dof-by-dof against the closed-form radial solution.

    Stronger than a symmetry check: a profile that was axisymmetric but radially wrong would pass the latter.  Comparing pointwise against

        p(r) = [1 - I0(r/lam)/I0(beta)] / [1 - 1/I0(beta)]

    tests the shape and the axisymmetry together.
    """
    lam = RADIUS / beta
    mesh, par = build_profile(cylinder_graded, lam)

    coords = par.function_space.tabulate_dof_coordinates()
    vals = par.x.array.reshape(-1, 3)
    magnitude = np.linalg.norm(vals, axis=1)

    on_cap = np.isclose(coords[:, 2], 0.0) & (magnitude > 0.0)
    worst = 0.0
    if on_cap.any():
        r = np.linalg.norm(coords[on_cap, :2], axis=1)
        exact = (1.0 - iv(0, r / lam) / iv(0, beta)) / (1.0 - 1.0 / iv(0, beta))
        worst = float(np.abs(magnitude[on_cap] - exact).max())
    assert mesh.comm.allreduce(worst, op=MPI.MAX) < 0.05
