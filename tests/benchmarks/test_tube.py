r"""3D tube with a three-element Windkessel outlet.

The shipped configuration is a designed verification case: with :math:`R_dC = 1/4\pi` and the :math:`\sin^2(2\pi t)` inflow waveform, :math:`\sin^2(\tau/2) = \sin^2(2\pi t)` with :math:`\tau = t/(R_dC)`, so the outlet pressure has the closed form in ``tests/_ref/rcr.py``.

Two assertions, in order of robustness:

1. **ODE consistency** on the recorded series -- the pressure the solver produced must
   satisfy the RCR ODE driven by the flow rate the solver produced.  This is independent of whatever flow amplitude the CFD happens to deliver, so it does not go flaky when the mesh changes.
2. **The closed form**, with :math:`Q_0` measured from the simulation rather than assumed.

The accuracy budget is dominated by the *coupling*, not by the ODE integrator: the flow rate is frozen across a time step, which makes the scheme first order in ``dt`` no matter how many RK4 sub-steps are used (``tests/unit/test_windkessel.py`` measures that directly).
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from conftest import load_example_module

_spec = importlib.util.spec_from_file_location(
    "rcr", Path(__file__).resolve().parents[1] / "_ref" / "rcr.py"
)
rcr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rcr)

RD, RP, C = 1.0, 0.1, 1.0 / (4.0 * np.pi)


def run_tube(repo_root, radius=0.5, length=5.0, resolution=0.15, max_steps=25):
    """Run the shipped tube example and return its per-step records."""
    tube_run = load_example_module("tube3d", "run.py")
    return tube_run.main(
        [
            "--config",
            str(repo_root / "examples" / "tube3d" / "tube3d.yaml"),
            "--output",
            "output/test_tube",
            "--radius",
            str(radius),
            "--length",
            str(length),
            "--resolution",
            str(resolution),
            "--max-steps",
            str(max_steps),
            "--quiet",
        ]
    )


@pytest.fixture(scope="module")
def short_run(repo_root):
    return run_tube(repo_root, max_steps=25)


def test_solver_converges_every_step(short_run):
    assert all(r["converged"] for r in short_run)
    assert all(r["picard_iterations"] <= 25 for r in short_run)


def test_global_mass_balance(short_run):
    """Exact by construction for this scheme; see ``postprocess.mass_balance``."""
    assert max(r["mass_defect"] for r in short_run) < 1.0e-6


def test_fields_are_finite(short_run):
    for r in short_run:
        assert np.isfinite(r["Q_in"]) and np.isfinite(r["div_norm"])
        assert np.isfinite(r["P_out_3"]) and np.isfinite(r["Q_out_3"])


def test_pressure_rises_with_flow(short_run):
    """The waveform is increasing over the first quarter period, so the pressure must be.

    A sign error anywhere in the chain -- the inlet profile orientation, the negative ``VelocityProfileScale``, the flow-rate normal -- shows up here as a pressure that falls while the pump is ramping up.
    """
    p = np.array([r["P_out_3"] for r in short_run])
    assert p[-1] > p[0]
    assert (p >= -1.0e-12).all(), "outlet pressure went negative during forward flow"


def test_flow_enters_the_domain(short_run):
    """``Q_in`` is negative: flow enters through the inlet, whose normal points outward."""
    q = np.array([r["Q_in"] for r in short_run])
    assert q[-1] < 0.0


@pytest.mark.slow
def test_rcr_ode_consistency(repo_root):
    """The recorded (Q, Pd) series must satisfy ``C dPd/dt + Pd/Rd == Q``.

    The primary Windkessel assertion.  It compares the solver against its own flow rate, so it is insensitive to the mesh and to whatever amplitude the CFD produces.
    """
    records = run_tube(repo_root, max_steps=250)
    dt = records[1]["time"] - records[0]["time"]
    pd = np.array([r["Pd_3"] for r in records])
    q = np.array([r["Q_out_3"] for r in records])

    residual = C * np.diff(pd) / dt + pd[1:] / RD - q[1:]
    scale = np.abs(q).max()
    assert scale > 0.0, "no flow through the outlet; the case did not drive anything"
    assert np.abs(residual).max() < 0.02 * scale, (
        f"max ODE residual {np.abs(residual).max():.3e} against flow scale {scale:.3e}"
    )


@pytest.mark.slow
def test_rcr_matches_the_closed_form(repo_root):
    """The outlet pressure must follow the analytic RCR solution.

    ``Q0`` is taken from the simulation rather than assumed, so this measures the Windkessel coupling rather than the CFD's flow amplitude.
    """
    records = run_tube(repo_root, max_steps=250)
    t = np.array([r["time"] for r in records])
    p_sim = np.array([r["P_out_3"] for r in records])
    q0 = float(np.abs([r["Q_out_3"] for r in records]).max())

    p_exact = rcr.p_out_sin2(t, q0, RD, RP, C)
    err = np.abs(p_sim - p_exact).max() / (RD * q0)
    assert err < 0.10, f"normalized max error {err:.3e}"
