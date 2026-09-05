"""Unit tests for the three-element Windkessel outlet model.

Everything here exercises the ODE integration and the convergence bookkeeping, none of which touches the mesh, so these are the cheapest meaningful tests in the suite.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from fenicsx_navier_stokes.boundaries import Windkessel

_spec = importlib.util.spec_from_file_location(
    "rcr", Path(__file__).resolve().parents[1] / "_ref" / "rcr.py"
)
rcr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rcr)

# The parameter set the shipped tube case was designed around: Rd*C = 1/(4*pi), so the sin^2 inflow waveform lines up exactly with the analytic reference.
RD, RP, C = 1.0, 0.1, 1.0 / (4.0 * np.pi)


def make_wk(**kwargs):
    """A Windkessel with no dolfinx coupling, for pure-ODE tests."""
    params = dict(Rd=RD, Rp=RP, C=C, Niter=100, dt_sim=1.0e-2, cap_id=1)
    params.update(kwargs)
    return Windkessel(**params)


class TestRK4:
    """The RK4 integration of the distal-pressure ODE."""

    def test_constant_Q_matches_exponential(self):
        """With Q held constant the ODE is linear and has a closed-form solution."""
        dt, Pd0, Q = 1.0e-2, 3.0, 7.0
        wk = make_wk(Pd_prev=Pd0, Q_out=Q, Niter=100, dt_sim=dt)
        p_out = wk.RK4([0.0, dt])

        expected_Pd = RD * Q + (Pd0 - RD * Q) * np.exp(-dt / (RD * C))
        assert wk.Pd_nl == pytest.approx(expected_Pd, abs=1e-12)
        assert p_out == pytest.approx(RP * Q + expected_Pd, rel=1e-14)

    @pytest.mark.parametrize("niter", [4, 10, 40])
    def test_fourth_order_in_substeps(self, niter):
        """RK4 error decays as Niter**-4; check the errors are ordered and small."""
        dt, Pd0, Q = 1.0e-2, 0.0, 7.0
        exact = RD * Q * (1.0 - np.exp(-dt / (RD * C)))
        wk = make_wk(Pd_prev=Pd0, Q_out=Q, Niter=niter, dt_sim=dt)
        wk.RK4([0.0, dt])
        assert abs(wk.Pd_nl - exact) < 1.0e-8

    def test_substep_convergence_rate(self):
        """The fitted order in Niter is 4, not merely 'small'."""
        dt, Q = 1.0e-2, 7.0
        exact = RD * Q * (1.0 - np.exp(-dt / (RD * C)))
        niters = np.array([2, 4, 8])
        errs = []
        for n in niters:
            wk = make_wk(Pd_prev=0.0, Q_out=Q, Niter=int(n), dt_sim=dt)
            wk.RK4([0.0, dt])
            errs.append(abs(wk.Pd_nl - exact))
        order = -np.polyfit(np.log(niters), np.log(errs), 1)[0]
        assert 3.7 < order < 4.3, f"expected 4th order in Niter, measured {order:.2f}"

    def test_steady_state(self):
        """Held at constant Q for many time constants, Pd tends to Rd*Q."""
        Q = 2.5
        wk = make_wk(Pd_prev=0.0, Q_out=Q, Niter=200, dt_sim=20.0 * RD * C)
        wk.RK4([0.0, wk.dt_sim])
        assert wk.Pd_nl == pytest.approx(RD * Q, rel=1e-8)

    def test_integrates_to_dt_not_dt_plus_h(self):
        """The sub-step grid must land exactly on dt.

        Building it as ``arange(0, dt + h, h)`` admits one extra point for many (dt, Niter) pairs because of floating-point round-off, integrating the ODE to ``dt + h``.  The parameters below are one such pair: the overshoot is ~9e-5 relative, which is far larger than the RK4 truncation error it hides behind.
        """
        dt, Q, niter = 0.3, 7.0, 1000
        wk = make_wk(Pd_prev=0.0, Q_out=Q, Niter=niter, dt_sim=dt)
        wk.RK4([0.0, dt])

        exact_dt = RD * Q * (1.0 - np.exp(-dt / (RD * C)))
        h = dt / niter
        exact_overshoot = RD * Q * (1.0 - np.exp(-(dt + h) / (RD * C)))

        assert wk.Pd_nl == pytest.approx(exact_dt, rel=1e-10)
        assert wk.Pd_nl != pytest.approx(exact_overshoot, rel=1e-8)

    def test_niter_one_is_valid(self):
        """A single sub-step must work rather than tripping over loop-variable leakage."""
        wk = make_wk(Pd_prev=0.0, Q_out=1.0, Niter=1, dt_sim=1.0e-3)
        wk.RK4([0.0, wk.dt_sim])
        assert np.isfinite(wk.Pd_nl)
        assert wk.Pd_nl > 0.0

    def test_niter_zero_rejected(self):
        with pytest.raises(ValueError, match="Niter"):
            make_wk(Niter=0)

    @pytest.mark.parametrize("bad", [dict(Rd=0.0), dict(C=-1.0)])
    def test_nonphysical_parameters_rejected(self, bad):
        with pytest.raises(ValueError):
            make_wk(**bad)


class TestCoupling:
    """Accuracy of the zero-order-hold coupling to the flow solver."""

    @staticmethod
    def _march(dt, n_steps, Q0, niter=100):
        """Reproduce exactly how the solver drives the model: freeze Q over the step."""
        wk = make_wk(Pd_prev=0.0, Niter=niter, dt_sim=dt)
        ts, ps = [], []
        for k in range(1, n_steps + 1):
            t = k * dt
            wk.Q_out = Q0 * np.sin(t / (RD * C) / 2.0) ** 2
            ps.append(wk.RK4([0.0, dt]))
            wk.advance()
            ts.append(t)
        return np.array(ts), np.array(ps)

    def test_sin2_drive_matches_closed_form(self):
        """Two periods of the sin^2 drive stay within 0.3% of Rd*Q0 of the analytic curve."""
        Q0, dt = 10.0, 1.0e-3
        ts, ps = self._march(dt, 500, Q0)
        exact = rcr.p_out_sin2(ts, Q0, RD, RP, C)
        err = np.abs(ps - exact).max() / (RD * Q0)
        assert err < 3.0e-3, f"normalized max error {err:.2e}"

    def test_coupling_is_first_order_in_dt(self):
        """Freezing Q across the step caps the scheme at first order, whatever Niter is.

        This is the number that actually governs Windkessel accuracy in a simulation:
        refining ``Niter`` buys nothing, refining ``dt`` is what helps.
        """
        Q0, T = 10.0, 0.5
        dts = np.array([4.0e-3, 2.0e-3, 1.0e-3])
        errs = []
        for dt in dts:
            ts, ps = self._march(dt, int(round(T / dt)), Q0)
            errs.append(np.abs(ps - rcr.p_out_sin2(ts, Q0, RD, RP, C)).max())
        order = np.polyfit(np.log(dts), np.log(errs), 1)[0]
        assert 0.85 < order < 1.20, f"expected first order in dt, measured {order:.3f}"


class TestResidual:
    """The nonlinear convergence measure."""

    def test_finite_at_cold_start(self):
        """Pd == 0 must not produce nan.

        With ``Pd_init: 0.0`` -- the default in every aorta configuration -- and a zero initial velocity, the first nonlinear iteration has ``Pd_nl == Pd_nl_prev == 0``. A purely relative residual computes 0/0 and yields nan, and because ``nan > tol`` is False the Windkessel term silently vanishes from the convergence criterion instead of failing.
        """
        wk = make_wk(Pd_prev=0.0, Q_out=0.0)
        wk.RK4([0.0, wk.dt_sim])
        assert wk.Pd_nl == 0.0
        r = wk.residual()
        assert np.isfinite(r), "residual must stay finite when Pd is exactly zero"
        assert r == 0.0

    def test_detects_change(self):
        wk = make_wk(Pd_prev=0.0, Q_out=5.0)
        wk.RK4([0.0, wk.dt_sim])
        assert wk.residual() > 1.0e-3

    def test_zero_after_convergence(self):
        wk = make_wk(Pd_prev=1.0, Q_out=5.0)
        wk.RK4([0.0, wk.dt_sim])
        wk.Pd_nl_prev = wk.Pd_nl
        assert wk.residual() == 0.0

    def test_advance_accepts_state(self):
        wk = make_wk(Pd_prev=0.0, Q_out=5.0)
        wk.RK4([0.0, wk.dt_sim])
        pd = wk.Pd_nl
        wk.advance()
        assert wk.Pd_prev == pd
        assert wk.Pd_nl_prev == pd
        assert wk.residual() == 0.0


def test_p_out_definition():
    """P_out is exactly Rp*Q + Pd, with no rounding slack."""
    wk = make_wk(Pd_prev=0.4, Q_out=3.0)
    p_out = wk.RK4([0.0, wk.dt_sim])
    assert p_out == pytest.approx(RP * wk.Q_out + wk.Pd_nl, rel=1e-14)


def test_repr_is_informative():
    assert "cap_id=1" in repr(make_wk())
