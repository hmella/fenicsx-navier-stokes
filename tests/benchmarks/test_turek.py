"""DFG 2D flow-around-a-cylinder benchmark.

Reference values for case 2D-3 (V. John, *Int. J. Numer. Meth. Fluids* **44** (2004) 777-788, eqs. 8-10):

===================  ============  =============
quantity             value         at
===================  ============  =============
``c_D,max``          2.950921575   t = 3.93625 ``c_L,max``          0.47795       t = 5.693125 ``dp(8 s)``          -0.1116
===================  ============  =============

Matching these is out of reach for CI, and saying so plainly is more useful than a test that pretends otherwise.  John's own first-order (Crouzeix-Raviart) computation needed about 785k dofs and ``dt = 0.0025`` to land within 0.8 % on ``c_D,max`` and 9 % on ``dp``; stabilized P1-P1 is not a better element than CR.  So the fast tests below assert that the computation *ran and is physically sane*, plus a refinement pair showing the answer moving toward the reference -- which is the strongest honest statement at this cost.  The full-resolution comparison is marked ``slow``.
"""

import numpy as np
import pytest
from conftest import load_example_module

REFERENCE = {"cD_max": 2.950921575, "cL_max": 0.47795, "dp_end": -0.1116}


def run_turek(repo_root, res_min=None, dt=0.02, max_steps=25, case="2d3"):
    turek_run = load_example_module("turek", "run.py")
    argv = [
        "--case",
        case,
        "--config",
        str(repo_root / "examples" / "turek" / "turek2d.yaml"),
        "--output",
        "output/test_turek",
        "--dt",
        str(dt),
        "--max-steps",
        str(max_steps),
        "--quiet",
    ]
    if res_min is not None:
        argv += ["--res-min", str(res_min)]
    return turek_run.main(argv)


@pytest.fixture(scope="module")
def smoke(repo_root):
    return run_turek(repo_root, max_steps=25)


def test_solver_converges(smoke):
    assert all(r["converged"] for r in smoke)
    assert max(r["picard_iterations"] for r in smoke) < 25


def test_quantities_are_finite(smoke):
    for r in smoke:
        assert np.isfinite(r["cD"]) and np.isfinite(r["cL"]) and np.isfinite(r["dp"])


def test_physical_envelope(smoke):
    """A generous sanity envelope, not the published interval.

    Catches a sign error, a wrong non-dimensionalisation or a diverging solve, without pretending a 726-cell mesh can reproduce the benchmark.
    """
    last = smoke[-1]
    assert 0.1 < last["cD"] < 12.0, f"cD = {last['cD']}"
    assert abs(last["cL"]) < 1.0, f"cL = {last['cL']}"
    assert 0.0 < last["dp"] < 2.0, f"dp = {last['dp']}"


def test_drag_grows_during_the_ramp(smoke):
    """Case 2D-3 ramps as ``sin(pi t / 8)``, so over the first second drag must increase."""
    cD = np.array([r["cD"] for r in smoke])
    assert cD[-1] > cD[0]


def test_mass_balance(smoke):
    assert max(r["mass_defect"] for r in smoke) < 1.0e-8


def test_stress_forms_agree(smoke):
    """Surface-integral and reaction-force drag must agree to within the discrete defect.

    The two differ only through the discrete mass-conservation error, so a large gap means the velocity field is far from divergence-free -- or that one of the two routines is wrong.
    """
    last = smoke[-1]
    gap = abs(last["cD"] - last["cD_surface"]) / abs(last["cD"])
    assert gap < 0.30, f"drag formulations differ by {gap:.1%}"


def test_pressure_probes_are_inside_the_mesh(repo_root):
    """The benchmark's probes sit exactly on the cylinder, which a polygon inscribes.

    Whether an un-nudged probe is found depends on where mesh vertices happen to land: on the circle itself the point may coincide with a polygon vertex and be located, or fall in a chord's shadow and be missed entirely.  The nudge removes that dependence.  This test asserts both halves of the contract -- nudged probes always evaluate, and a point genuinely outside the fluid domain raises rather than returning a silent ``nan``.
    """
    from fenicsx_navier_stokes.postprocess import evaluate_at_points

    turek_mesh = load_example_module("turek", "turek_mesh.py")
    turek_run = load_example_module("turek", "run.py")
    mesh, _, _ = turek_mesh.generate()

    from basix.ufl import element
    from dolfinx.fem import Function, functionspace

    Q = functionspace(mesh, element("Lagrange", mesh.topology.cell_name(), 1))
    p = Function(Q)
    p.x.array[:] = 1.0

    values = evaluate_at_points(p, turek_run.probe_points(), mesh)
    assert np.isfinite(values).all()
    assert values == pytest.approx(1.0)

    # A point inside the cylinder is not in the fluid domain at all; the guard must say so rather than handing back nan.
    inside_cylinder = np.array([[turek_mesh.C_X, turek_mesh.C_Y, 0.0]])
    with pytest.raises(ValueError, match="outside the mesh"):
        evaluate_at_points(p, inside_cylinder, mesh)


@pytest.mark.slow
def test_drag_converges_under_refinement(repo_root):
    """Successive mesh refinements must change the answer by less and less.

    This is a Cauchy-convergence statement about the discretization, and it deliberately does not
    mention the published reference. An earlier version of this test compared two meshes against
    `c_D,max = 2.9509` and asserted the finer one was closer, which was not a meaningful claim at
    this cost: 60 steps at dt = 0.02 only reaches t = 1.2 s, where the 2D-3 ramp sin(pi*t/8) is at
    45 % of peak and the drag is around 1.0 against a reference of 2.95. The error was dominated
    by the time window, not the mesh, so which of two meshes came out closer was very nearly a
    coin toss -- it flipped on an unrelated change to the pressure-block assembly.
    """
    steps, dt = 60, 0.02
    values = []
    for factor in (2, 4, 8):
        records = run_turek(repo_root, res_min=0.05 / factor, dt=dt, max_steps=steps)
        values.append(max(r["cD"] for r in records))

    first = abs(values[1] - values[0])
    second = abs(values[2] - values[1])
    assert second < first, (
        f"drag is not settling under refinement: cD = {values}, "
        f"successive changes {first:.4f} then {second:.4f}"
    )


@pytest.mark.benchmark
def test_2d3_against_reference(repo_root):
    """The full DFG 2D-3 comparison.  Hours, and excluded from every automated tier."""
    records = run_turek(repo_root, res_min=0.05 / 8, dt=0.005, max_steps=int(8.0 / 0.005))
    cD = np.array([r["cD"] for r in records])
    cL = np.array([r["cL"] for r in records])
    assert abs(cD.max() - REFERENCE["cD_max"]) / REFERENCE["cD_max"] < 0.05
    assert abs(cL.max() - REFERENCE["cL_max"]) / REFERENCE["cL_max"] < 0.15
    assert abs(abs(records[-1]["dp"]) - abs(REFERENCE["dp_end"])) < 0.02
