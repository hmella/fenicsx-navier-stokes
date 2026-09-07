"""
DFG 2D flow around a cylinder -- the Turek benchmark.

WHAT THIS CASE IS
-----------------
Flow through a narrow channel with a circular obstacle in it. It is the standard verification problem for incompressible flow solvers: several groups have computed it to high accuracy, so the drag and lift on the cylinder are known numbers that a new solver can be checked against.

The geometry is a channel 2.2 long and 0.41 high with a cylinder of radius 0.05 at (0.2, 0.2), slightly off the channel centreline (0.2 against 0.205). The offset produces the alternating vortex shedding the benchmark measures.

THE THREE CASES
---------------
- 2d1: steady, Re = 20. Low enough that the flow settles to a steady state.
- 2d2: Re = 100 held constant. Produces a periodic vortex street, but needs 25-30 s of physical time before the street is fully developed.
- 2d3: Re ramps up as U(t) = 1.5*sin(pi*t/8) over t in [0, 8], starting from rest. This is the default.

REFERENCE VALUES (case 2d3)
---------------------------
From V. John, Int. J. Numer. Meth. Fluids 44 (2004) 777-788:
- max drag coefficient: 2.950921575 at t = 3.93625
- max lift coefficient: 0.47795 at t = 5.693125
- pressure drop at 8 s: -0.1116

COST. Matching those numbers takes millions of unknowns and hours of computing: a first-order method needs about 785000 unknowns and a time step of 0.0025 to get within 1% on drag and 9% on the pressure drop. The shipped mesh is coarse, sized for checking that the case runs.

WHAT IT WRITES
--------------
<Run.Output>/turek_<case>.csv, one row per time step, with drag, lift, pressure drop and mass balance, plus XDMF fields when Run.StoreAfter is set.

CONFIGURATION
-------------
Every parameter lives in the YAML file; the only command-line argument is which file to read. The case, mesh resolution and time step are Run.Case, Run.ResMin and Run.TimeStep.

Usage:
    python examples/turek/run.py --config examples/turek/turek2d.yaml
"""

# ruff: noqa: E402

# The example scripts are run directly rather than installed, so the package directory and this directory are put on the import path before the imports below.
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from mpi4py import MPI

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import turek_mesh
from basix.ufl import element as basix_element
from dolfinx.fem import Function, functionspace

from fenicsx_navier_stokes import (
    NewtonNSProblem,
    ParameterHandler,
    PicardNSProblem,
    drag_lift_coefficients,
    drag_lift_surface,
    drag_lift_variational,
    evaluate_at_points,
    mass_balance,
)

# Nonlinear solvers selectable through Solver.Scheme in the configuration file
SOLVERS = {"picard": PicardNSProblem, "newton": NewtonNSProblem}


# Geometry constants, taken from the mesh generator so the two cannot drift apart
H, L, R = turek_mesh.H, turek_mesh.L, turek_mesh.R
C_X, C_Y = turek_mesh.C_X, turek_mesh.C_Y
D = turek_mesh.D  # cylinder diameter, the length used to non-dimensionalise drag and lift

# Peak centreline inflow velocity, final time, and whether the inflow ramps up
CASES = {"2d1": dict(u_max=0.3, t_end=8.0, ramp=False),
         "2d2": dict(u_max=1.5, t_end=30.0, ramp=False),
         "2d3": dict(u_max=1.5, t_end=8.0, ramp=True)}

# Published reference values for case 2d3 (John 2004, eqs. 8-10)
REFERENCE_2D3 = {"cD_max": 2.950921575, "t_cD_max": 3.93625,
                 "cL_max": 0.47795, "t_cL_max": 5.693125,
                 "dp_end": -0.1116}

# The benchmark measures the pressure drop between the front and back of the cylinder, at points that lie exactly on its surface. A straight-sided mesh approximates the circle by a polygon inscribed inside it, so those points can fall marginally outside the mesh and the evaluation finds nothing. Nudging them a hair inwards removes the problem.
PROBE_EPS = 1.0e-6


def probe_points():
    """The two pressure probes, moved just inside the cylinder surface."""
    centre = np.array([C_X, C_Y, 0.0])
    front = np.array([C_X - R, C_Y, 0.0])  # upstream stagnation point
    back = np.array([C_X + R, C_Y, 0.0])   # downstream stagnation point
    return np.vstack([centre + (front - centre) * (1.0 - PROBE_EPS),
                      centre + (back - centre) * (1.0 - PROBE_EPS)])


def parabolic_inflow(V, u_max):
    """The benchmark inlet profile: a parabola across the channel, flowing in +x.

    u_x = 4*U*y*(H - y)/H^2 is zero at both walls and peaks at U in the middle. Its average across the channel is 2/3*U, and that average is the velocity the Reynolds number and the drag and lift coefficients are defined with.
    """
    fn = Function(V)
    fn.interpolate(lambda x: np.vstack([4.0 * u_max * x[1] * (H - x[1]) / H**2,
                                        np.zeros_like(x[0])]))
    fn.x.scatter_forward()
    return fn


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--config", default=str(Path(__file__).parent / "turek2d.yaml"),
                   help="parameter file; every setting is read from it")
    args = p.parse_args(argv)

    # Read the configuration. The facet tags are inlet = 2, outlet = 3, channel walls = 4, cylinder = 5. Walls and cylinder are tagged separately so drag can be integrated over the cylinder alone, but both are no-slip, which turek2d.yaml expresses as WallID: [4, 5].
    comm = MPI.COMM_WORLD
    pars = ParameterHandler(args.config)
    run = pars.Run
    obstacle = pars.Geometry.ObstacleID

    case = CASES[run.Case]
    t_end = run.TEnd if run.TEnd is not None else case["t_end"]
    num_steps = int(round(t_end / run.TimeStep))

    # Build the mesh with gmsh
    mesh, cell_tags, facet_tags = turek_mesh.generate(res_min=run.ResMin, comm=comm)

    # The inlet velocity is a fixed spatial profile times a scalar that changes each step. Case 2d3 ramps that scalar as sin(pi*t/8); the other two hold it at 1.
    if case["ramp"]:
        def scale(step, t):
            return float(np.sin(np.pi * t / 8.0))
    else:
        def scale(step, t):
            return 1.0

    # Build the parabolic profile on the velocity space. Passing it explicitly bypasses the solver's default inlet profile, which is meant for irregular anatomical caps.
    degree = 2 if run.Element == "P2-P1" else 1
    VE = basix_element("Lagrange", mesh.topology.cell_name(), degree, shape=(2,))
    inflow = parabolic_inflow(functionspace(mesh, VE), case["u_max"])

    problem = SOLVERS[pars.Solver.Scheme](parameters=pars, mesh=mesh,
                                          XDMF=run.StoreAfter is not None,
                                          domains=cell_tags, boundaries=facet_tags,
                                          element=run.Element, dt=run.TimeStep,
                                          num_steps=num_steps,
                                          inlet_profile=inflow, inlet_scale=scale)

    # Mean inflow velocity, used to non-dimensionalise the forces
    u_bar = 2.0 / 3.0 * case["u_max"]
    rho = float(problem.rho.value)

    out = Path(run.Output)
    if comm.rank == 0:
        out.mkdir(parents=True, exist_ok=True)

    records = []
    points = probe_points()

    def record(prob, step, t, info):
        """Called after every converged time step, to measure the forces on the cylinder."""
        # Drag and lift are computed two different ways.
        #
        # The variational (reaction force) method evaluates the discrete equations against a test function that equals a unit vector on the cylinder. What comes out is the force the no-slip constraint exerts there. It is the accurate one, because it inherits the accuracy of the whole solution rather than of a surface integral.
        #
        # The surface method integrates the stress over the cylinder directly. It is the textbook definition and is kept as a cross-check: the two agree only to the extent that the discrete velocity is divergence-free, so the gap between them is itself a useful diagnostic.
        f_variational = drag_lift_variational(prob, obstacle)
        f_surface = drag_lift_surface(prob, obstacle)

        # Turn forces into dimensionless coefficients: c = 2*F/(rho*u_bar^2*D)
        c_variational = drag_lift_coefficients(f_variational, rho, u_bar, D)
        c_surface = drag_lift_coefficients(f_surface, rho, u_bar, D)

        # Pressure difference across the cylinder, the benchmark's third quantity
        pressures = evaluate_at_points(prob.p_h, points)

        balance = mass_balance(prob.u_h, prob.boundaries, pars.Geometry.InletID,
                               pars.Geometry.OutletIDs)

        records.append({"step": step, "time": t,
                        "cD": c_variational[0], "cL": c_variational[1],
                        "cD_surface": c_surface[0], "cL_surface": c_surface[1],
                        "dp": pressures[0] - pressures[1],
                        "mass_defect": balance["defect"],
                        "picard_iterations": info["iterations"],
                        "converged": int(info["converged"])})

    # Run the time loop
    problem.solve(xdmf_path=str(out / f"turek_{run.Case}.xdmf"),
                  store_after=run.StoreAfter, store_every=run.StoreEvery,
                  max_steps=run.MaxSteps, verbose=run.Verbose, callback=record)

    # Write the diagnostics and compare against the reference where that is meaningful
    if comm.rank == 0 and records:
        csv_path = out / f"turek_{run.Case}.csv"
        with open(csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)

        cD = np.array([r["cD"] for r in records])
        cL = np.array([r["cL"] for r in records])
        ts = np.array([r["time"] for r in records])
        n_cells = mesh.topology.index_map(2).size_global
        print(f"\n[turek {run.Case}] {len(records)} steps, {n_cells} cells, dt={run.TimeStep}")
        print(f"  cD max  = {cD.max():.6f} at t = {ts[cD.argmax()]:.5f}")
        print(f"  cL max  = {cL.max():.6f} at t = {ts[cL.argmax()]:.5f}")
        print(f"  dp end  = {records[-1]['dp']:.6f}")
        print(f"  mass defect (max) = {max(r['mass_defect'] for r in records):.3e}")

        # The reference values describe the complete 8 second ramp, so only report the comparison when the run actually covered it
        if run.Case == "2d3" and ts[-1] >= 7.9:
            ref = REFERENCE_2D3
            print(f"  reference (John 2004): cD_max={ref['cD_max']:.6f}, "
                  f"cL_max={ref['cL_max']:.5f}, dp(8s)={ref['dp_end']:.4f}")
            print(f"  relative error: cD {abs(cD.max() - ref['cD_max']) / ref['cD_max']:.2%}, "
                  f"cL {abs(cL.max() - ref['cL_max']) / ref['cL_max']:.2%}")
        print(f"  wrote {csv_path}")
    return records


if __name__ == "__main__":
    main()
