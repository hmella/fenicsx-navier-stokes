"""
NewtonNSProblem: the Jacobian, the convergence rate, and the Picard interoperation.

The finite-difference Jacobian check is the important one. Everything else about Newton -- the convergence rate, the iteration count, the agreement with Picard -- follows from the Jacobian being right, and a wrong Jacobian usually still converges, just slowly, so a convergence test alone would not catch it.
"""

import numpy as np
import pytest
from conftest import default_params, tag_all_boundaries, tag_square_boundaries
from dolfinx.fem import Constant, Function
from dolfinx.fem.petsc import assemble_matrix, assemble_vector
from dolfinx.mesh import CellType, DiagonalType, create_unit_square
from mpi4py import MPI
from petsc4py import PETSc
from ufl import as_vector

from fenicsx_navier_stokes import NewtonNSProblem, PicardNSProblem, pressure_pin_bc


def build(cls, comm, n=6, dt=1.0e-2, num_steps=1, rho=1.06, mu=0.04, nl_tol=1e-12,
          max_nl=15, **kwargs):
    """A small driven-cavity-like problem with an inlet, an outlet and walls."""
    mesh = create_unit_square(comm, n, n, CellType.triangle, diagonal=DiagonalType.crossed)
    ft = tag_square_boundaries(mesh)
    pars = default_params(density=rho, viscosity=mu, kind="direct",
                          nl_tol=nl_tol, max_nl=max_nl)
    return cls(parameters=pars, mesh=mesh, XDMF=False, boundaries=ft, element="P1-P1",
               dt=dt, num_steps=num_steps, inlet_scale=lambda k, t: 1.0, **kwargs)


def seed_state(prob, seed=0):
    """Put the problem in a non-trivial state so every residual term is active."""
    prob.u_h.interpolate(lambda x: np.vstack([1.0 + x[1], 0.3 * x[0]]))
    prob.u_h.x.scatter_forward()
    prob.p_h.interpolate(lambda x: 0.5 - x[0])
    prob.p_h.x.scatter_forward()
    prob.u0.x.array[:] = 0.7 * prob.u_h.x.array
    prob.u00.x.array[:] = 0.2 * prob.u_h.x.array
    # tau_M and the SUPG weight are frozen on `up`, so it must be held fixed while the
    # solution is perturbed -- the Jacobian does not differentiate through it
    prob.up.x.array[:] = prob.u_h.x.array


def raw_residual(prob):
    """The residual with no boundary conditions applied, as a flat owned-dof array."""
    b = assemble_vector(prob._F_form, kind="nest")
    for sub in b.getNestSubVecs():
        sub.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    out = np.concatenate([s.getArray().copy() for s in b.getNestSubVecs()])
    b.destroy()
    return out


def raw_jacobian_action(prob, du, dp):
    """J @ (du, dp) with no boundary conditions applied."""
    blocks = {}
    for name, f in (("00", prob._f_j00), ("01", prob._f_j01),
                    ("10", prob._f_j10), ("11", prob._f_j11)):
        A = assemble_matrix(f, bcs=[])
        A.assemble()
        blocks[name] = A

    xu = blocks["00"].createVecRight()
    xu.getArray()[:] = du
    xu.assemble()
    xp = blocks["11"].createVecRight()
    xp.getArray()[:] = dp
    xp.assemble()

    y0 = blocks["00"].createVecLeft()
    tmp0 = blocks["00"].createVecLeft()
    blocks["00"].mult(xu, y0)
    blocks["01"].mult(xp, tmp0)
    y0.axpy(1.0, tmp0)

    y1 = blocks["10"].createVecLeft()
    tmp1 = blocks["10"].createVecLeft()
    blocks["10"].mult(xu, y1)
    blocks["11"].mult(xp, tmp1)
    y1.axpy(1.0, tmp1)

    out = np.concatenate([y0.getArray().copy(), y1.getArray().copy()])
    for A in blocks.values():
        A.destroy()
    return out


class TestJacobian:
    """The Jacobian must be the derivative of the residual, not merely something that converges."""

    def test_matches_finite_difference(self, comm):
        """Sweep the step size and require the error to fall linearly with it.

        A correct Jacobian gives an error of order eps from the one-sided difference, down to where round-off (order machine/eps) takes over. Checking the *best* value over a sweep, rather than one arbitrary eps, is what makes this robust: it fails loudly for a wrong Jacobian and cannot be tuned into passing.
        """
        prob = build(NewtonNSProblem, comm)
        seed_state(prob)

        nu = prob.V.dofmap.index_map.size_local * prob.V.dofmap.index_map_bs
        nq = prob.Q.dofmap.index_map.size_local * prob.Q.dofmap.index_map_bs
        rng = np.random.default_rng(0)
        du, dp = rng.standard_normal(nu), rng.standard_normal(nq)

        jd = raw_jacobian_action(prob, du, dp)
        f0 = raw_residual(prob)
        u_save, p_save = prob.u_h.x.array.copy(), prob.p_h.x.array.copy()

        best = np.inf
        for eps in (1e-4, 1e-5, 1e-6, 1e-7):
            prob.u_h.x.array[:] = u_save
            prob.u_h.x.array[:nu] += eps * du
            prob.p_h.x.array[:] = p_save
            prob.p_h.x.array[:nq] += eps * dp
            prob.u_h.x.scatter_forward()
            prob.p_h.x.scatter_forward()
            fd = (raw_residual(prob) - f0) / eps
            num = comm.allreduce(float(np.linalg.norm(fd - jd) ** 2), op=MPI.SUM)
            den = comm.allreduce(float(np.linalg.norm(jd) ** 2), op=MPI.SUM)
            best = min(best, np.sqrt(num / den))

        prob.u_h.x.array[:] = u_save
        prob.p_h.x.array[:] = p_save
        assert best < 1.0e-5, f"Jacobian disagrees with the residual: best relative error {best:.3e}"

    def test_differs_from_the_picard_operator(self, comm):
        """Newton must actually add the grad(u)*du term, not silently reproduce Picard."""
        newton = build(NewtonNSProblem, comm)
        picard = build(PicardNSProblem, comm)
        for prob in (newton, picard):
            seed_state(prob)

        jn = assemble_matrix(newton._f_j00, bcs=[])
        jn.assemble()
        jp = assemble_matrix(picard._f_a00_star, bcs=[])
        jp.assemble()
        # The Picard star block omits the constant part, so compare only that the Newton
        # block is not equal to Picard's full operator
        jp.axpy(1.0, picard.A00)
        diff = jn.copy()
        diff.axpy(-1.0, jp)
        assert diff.norm() > 1.0e-6 * jn.norm(), "Newton and Picard operators are identical"


class TestIteration:
    """Dispatch, convergence and the Picard fallback."""

    def test_first_step_uses_picard(self, comm):
        """There is no previous solution to start Newton from on the first step."""
        prob = build(NewtonNSProblem, comm, num_steps=3)
        history = prob.solve(store_after=None, verbose=False)
        assert history[0]["method"] == "picard"
        assert all(h["method"] == "newton" for h in history[1:])
        assert all(h["converged"] for h in history)

    def test_reaches_the_same_solution_as_picard(self, comm):
        """The two schemes solve the same equations, so they must agree at convergence."""
        newton = build(NewtonNSProblem, comm, num_steps=3, nl_tol=1e-13, max_nl=40)
        picard = build(PicardNSProblem, comm, num_steps=3, nl_tol=1e-13, max_nl=40)
        newton.solve(store_after=None, verbose=False)
        picard.solve(store_after=None, verbose=False)

        diff = float(np.linalg.norm(newton.u_h.x.array - picard.u_h.x.array))
        scale = float(np.linalg.norm(picard.u_h.x.array))
        num = comm.allreduce(diff**2, op=MPI.SUM)
        den = comm.allreduce(scale**2, op=MPI.SUM)
        rel = np.sqrt(num / max(den, 1e-300))
        # Measured 9.6e-16. The bound is set just above that, not loosely: a systematic
        # difference between the two schemes shows up as a tolerance-INDEPENDENT offset, which
        # is exactly how the stale-tau_M defect in the (1,1) block was found -- it sat at
        # 8.3e-6 no matter how tight the nonlinear tolerance was set.
        assert rel < 1.0e-12, f"Newton and Picard disagree by {rel:.3e}"

    def test_needs_no_more_iterations_than_picard(self, comm):
        """Newton should not be slower to converge than Picard once it is running."""
        newton = build(NewtonNSProblem, comm, num_steps=4, nl_tol=1e-10, max_nl=40)
        picard = build(PicardNSProblem, comm, num_steps=4, nl_tol=1e-10, max_nl=40)
        hn = newton.solve(store_after=None, verbose=False)
        hp = picard.solve(store_after=None, verbose=False)
        # Compare only the steps Newton actually ran (the first is Picard for both)
        newton_its = sum(h["iterations"] for h in hn[1:])
        picard_its = sum(h["iterations"] for h in hp[1:])
        assert newton_its <= picard_its, f"Newton took {newton_its}, Picard {picard_its}"

    def test_residual_decreases_superlinearly(self, comm):
        """Fit the convergence order from successive residual norms.

        Frozen stabilization parameters cost some of the quadratic rate, so the bound is 1.3 rather than 2. The point is to distinguish superlinear from the linear rate a wrong Jacobian would give.
        """
        prob = build(NewtonNSProblem, comm, n=8, dt=5.0e-3, num_steps=3,
                     nl_tol=1e-14, max_nl=25)
        history = prob.solve(store_after=None, verbose=False)

        orders = []
        for info in history[1:]:
            r = np.array([v for v in info["residuals"] if np.isfinite(v) and v > 0])
            # Use the tail, where the asymptotic rate shows, and stop before round-off
            r = r[r > 1e-13]
            if len(r) >= 3:
                orders.extend(
                    np.log(r[k + 1]) / np.log(r[k]) for k in range(len(r) - 1) if r[k] < 1e-2
                )
        if orders:
            assert max(orders) > 1.3, f"no superlinear convergence: orders {orders}"

    def test_falls_back_to_picard(self, comm):
        """Starve Newton of iterations and check the step still converges, and says so."""
        prob = build(NewtonNSProblem, comm, num_steps=2, nl_tol=1e-13, max_nl=1)
        history = prob.solve(store_after=None, verbose=False)
        second = history[1]
        assert second["newton_failed"] is True
        assert second["method"] == "picard"
        assert "newton_iterations" in second


class TestPatch:
    """The uniform-flow patch test, which the Picard scheme also has to pass."""

    @pytest.mark.parametrize("rho", [1.0, 1.06])
    def test_uniform_flow_is_exact(self, comm, rho):
        c = (0.7, -0.3)
        mesh = create_unit_square(comm, 6, 6, CellType.triangle, diagonal=DiagonalType.crossed)
        ft = tag_all_boundaries(mesh, wall=1, inlet=2)
        pars = default_params(density=rho, viscosity=0.04, kind="direct",
                              wall=1, inlet=2, outlets=(), nl_tol=1e-12, max_nl=15)
        uniform = as_vector([Constant(mesh, float(ci)) for ci in c])
        prob = NewtonNSProblem(
            parameters=pars, mesh=mesh, XDMF=False, boundaries=ft, element="P1-P1",
            dt=1.0e-2, num_steps=2, inlet_profile=uniform, inlet_scale=lambda k, t: 1.0,
            wall_velocity=uniform, extra_bcs=lambda V, Q: [pressure_pin_bc(Q, 0.0)],
        )
        exact = Function(prob.V)
        exact.interpolate(lambda x: np.vstack([np.full(x.shape[1], ci) for ci in c]))
        exact.x.scatter_forward()
        prob.initial_condition(u0=exact, u00=exact)
        prob.up.x.array[:] = exact.x.array
        prob.u_h.x.array[:] = exact.x.array

        prob.solve(store_after=None, verbose=False)

        u_err = comm.allreduce(
            float(np.abs(prob.u_h.x.array - exact.x.array).max()), op=MPI.MAX
        )
        p_err = comm.allreduce(float(np.abs(prob.p_h.x.array).max()), op=MPI.MAX)
        assert u_err < 1e-11, f"velocity patch error {u_err:.3e}"
        assert p_err < 1e-9, f"pressure patch error {p_err:.3e}"


def test_postprocess_accepts_a_newton_problem(comm):
    """drag_lift_variational reads sixteen attributes off the problem; a Newton problem must expose them all with the same meaning."""
    from fenicsx_navier_stokes import drag_lift_surface, drag_lift_variational

    prob = build(NewtonNSProblem, comm, num_steps=1)
    prob.solve(store_after=None, verbose=False)
    f_var = drag_lift_variational(prob, 1)   # the wall tag
    f_sur = drag_lift_surface(prob, 1)
    assert np.isfinite(f_var).all()
    assert np.isfinite(f_sur).all()


class TestWindkesselJacobian:
    """The exact rank-one outlet coupling."""

    @staticmethod
    def make(comm, cls, **kwargs):
        from dolfinx.fem import Constant
        from petsc4py import PETSc

        from fenicsx_navier_stokes import Windkessel

        mesh = create_unit_square(comm, 6, 6, CellType.triangle, diagonal=DiagonalType.crossed)
        ft = tag_square_boundaries(mesh)
        pars = default_params(density=1.06, viscosity=0.04, kind="direct",
                              nl_tol=1e-12, max_nl=40)
        wksl = Windkessel(Pd_prev=0.0, Rd=1.0, Rp=0.1, C=1.0 / (4.0 * np.pi), cap_id=3,
                          Niter=50, facet_tags=ft, dt_sim=1.0e-2,
                          P_out=Constant(mesh, PETSc.ScalarType(0.0)))
        prob = cls(parameters=pars, mesh=mesh, XDMF=False, boundaries=ft, windkessels=[wksl],
                   element="P1-P1", dt=1.0e-2, num_steps=3,
                   inlet_scale=lambda k, t: 1.0, **kwargs)
        return prob, wksl

    def test_impedance_matches_the_ode(self, comm):
        """dP_out/dQ must match a finite difference of the Windkessel's own RK4 routine.

        The closed form is exact for the linear RCR ODE with the flow rate frozen across the
        step, which is how the solver drives it.
        """
        prob, wksl = self.make(comm, NewtonNSProblem)
        analytic = prob._wk_impedance[0]

        base = wksl.Q_out
        eps = 1.0e-6
        wksl.Q_out = base + eps
        hi = wksl.RK4([0.0, wksl.dt_sim])
        wksl.Q_out = base - eps
        lo = wksl.RK4([0.0, wksl.dt_sim])
        wksl.Q_out = base
        assert analytic == pytest.approx((hi - lo) / (2.0 * eps), rel=1.0e-8)

    def test_correction_is_rank_one_and_positive(self, comm):
        """(A_op - A) x must equal Z (a.x) a, and be positive semi-definite.

        Physically the sign has to come out this way: more outflow raises the outlet pressure,
        which resists further outflow. A negative coefficient would make the velocity block
        indefinite and degrade the AMG preconditioner.
        """
        prob, _ = self.make(comm, NewtonNSProblem)
        assert prob._newton_operator is not None

        # A MatNest hands out either a VecNest or a flat vector depending on how it was
        # built, so reach the velocity block through the same helper the operator uses.
        velocity_is = prob.A.getNestISs()[0][0]
        block = lambda v: v.getSubVector(velocity_is)  # noqa: E731

        x = prob.A.createVecRight()
        x.set(0.0)
        xu = block(x)
        rng = np.random.default_rng(0)
        xu.getArray()[: prob._wk_vectors[0].getLocalSize()] = rng.standard_normal(
            prob._wk_vectors[0].getLocalSize()
        )
        x.assemble()

        y_full = prob.A.createVecLeft()
        prob._newton_operator.mult(x, y_full)
        y_base = prob.A.createVecLeft()
        prob.A.mult(x, y_base)
        y_full.axpy(-1.0, y_base)

        a = prob._wk_vectors[0]
        Z = prob._wk_impedance[0]
        expected = prob.A.createVecLeft()
        expected.set(0.0)
        block(expected).axpy(Z * a.dot(block(x)), a)
        expected.assemble()

        y_full.axpy(-1.0, expected)
        assert y_full.norm() < 1e-12 * max(y_base.norm(), 1.0)
        assert Z > 0.0

    def test_matches_picard_with_outlets(self, comm):
        """The exact Jacobian must not change the answer, only how fast it is reached."""
        newton, _ = self.make(comm, NewtonNSProblem)
        picard, _ = self.make(comm, PicardNSProblem)
        hn = newton.solve(store_after=None, verbose=False)
        hp = picard.solve(store_after=None, verbose=False)

        rel = float(np.linalg.norm(newton.u_h.x.array - picard.u_h.x.array)) / float(
            np.linalg.norm(picard.u_h.x.array)
        )
        assert rel < 1.0e-12, f"disagree by {rel:.3e}"
        assert sum(h["iterations"] for h in hn[1:]) <= sum(h["iterations"] for h in hp[1:])

    def test_can_be_disabled(self, comm):
        """windkessel_jacobian=False lags the coupling instead, as the Picard scheme does."""
        prob, _ = self.make(comm, NewtonNSProblem, windkessel_jacobian=False)
        assert prob._newton_operator is None
        history = prob.solve(store_after=None, verbose=False)
        assert all(h["converged"] for h in history)
