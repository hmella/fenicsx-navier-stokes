"""
Straight 3D tube with a three-element Windkessel outlet.

WHAT THIS CASE IS
-----------------
A SimVascular-style vessel segment: blood flowing down a straight cylindrical pipe. The inlet is driven by a prescribed flow waveform, and the outlet is not simply left open but coupled to a lumped (0D) model of everything downstream -- the rest of the arterial tree, represented by a resistance-compliance circuit. That is the RCR Windkessel.

THE OUTLET MODEL
----------------
A traction-free outlet pins the pressure there to zero. The Windkessel replaces it: given the flow rate Q leaving the cap, it returns a pressure P that is imposed back on the fluid as a normal traction.

VERIFICATION
------------
The shipped tube3d.yaml sets Rd*C = 1/(4*pi), matching the sin^2(2*pi*t) waveform in data/inflow/tube.flow. With that pairing the outlet pressure has a closed form, which tests/benchmarks/test_tube.py checks against. Changing the parameters or the waveform on their own leaves a case that runs but verifies nothing.

WHAT IT WRITES
--------------
<Run.Output>/tube3d.csv, one row per time step, with the flow rates, the Windkessel pressure, the mass balance and the divergence norm, plus XDMF fields when Run.StoreAfter is set.

CONFIGURATION
-------------
Every parameter lives in the YAML file; the only command-line argument is which file to read. The cylinder geometry and mesh resolution are in its Run: section.

Usage:
    python examples/tube3d/run.py --config examples/tube3d/tube3d.yaml
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

import tube_mesh
from dolfinx.fem import Constant
from petsc4py import PETSc

from fenicsx_navier_stokes import (
    NewtonNSProblem,
    ParameterHandler,
    PicardNSProblem,
    Windkessel,
    divergence_norm,
    mass_balance,
)

# Nonlinear solvers selectable through Solver.Scheme in the configuration file
SOLVERS = {"picard": PicardNSProblem, "newton": NewtonNSProblem}


def build(config, comm=None):
    """Put together everything the solver needs: mesh, parameters and outlet models."""
    comm = comm if comm is not None else MPI.COMM_WORLD

    # Read the YAML configuration. Values are reached with dot notation, so for example pars.Problem.Viscosity is the dynamic viscosity in poise.
    pars = ParameterHandler(config)
    run = pars.Run

    # Element size at the wall and in the interior. Run.Resolution defaults to a quarter of the radius, and Run.WallResolution to Run.Resolution.
    resolution = run.Resolution if run.Resolution is not None else run.Radius / 4.0

    # Build the cylinder with gmsh. The returned tags label the boundary facets: wall = 1, inlet = 2, outlet = 3. These numbers must agree with tube3d.yaml, or the solver would find no facets to apply boundary conditions to.
    mesh, cell_tags, facet_tags = tube_mesh.generate(radius=run.Radius, length=run.Length,
                                                     resolution=resolution,
                                                     wall_resolution=run.WallResolution,
                                                     comm=comm)

    # The time step is not a free parameter here: it is the spacing between rows of the inflow waveform file, since the solver advances one row per step. The Windkessel ODE is integrated over exactly this interval at every step.
    inflow = np.loadtxt(pars.Problem.VelocityProfileFile)
    dt = float(inflow[1, 0] - inflow[0, 0])

    # Create one Windkessel model per outlet. Each holds its own circuit parameters:
    # - Rp: proximal resistance, felt immediately (pressure jumps with flow)
    # - Rd: distal resistance, felt through the compliance
    # - C: compliance, the elasticity of the downstream vessels
    #
    # P_out is a dolfinx Constant that the momentum equation references. The model writes the current pressure into it each iteration, so the fluid problem sees the update without any form having to be rebuilt.
    wk_cfg = pars.Problem.Windkessels
    windkessels = []
    for i, cap in enumerate(pars.Geometry.OutletIDs):
        windkessels.append(Windkessel(Pd_prev=wk_cfg.Pd_init,
                                      Rd=wk_cfg.Rd[i], Rp=wk_cfg.Rp[i], C=wk_cfg.C[i],
                                      cap_id=cap, Niter=wk_cfg.NbIters,
                                      facet_tags=facet_tags, dt_sim=dt,
                                      P_out=Constant(mesh, PETSc.ScalarType(wk_cfg.Pd_init))))

    # Assemble the Navier-Stokes problem. 'inlet_plateau_lam' shapes the inlet velocity profile: small values give a flat plug, large values a parabola. 0.15 on a radius of a few centimetres is close to a plug, which is what a real vessel inlet looks like.
    problem = SOLVERS[pars.Solver.Scheme](parameters=pars, mesh=mesh,
                                          XDMF=run.StoreAfter is not None,
                                          domains=cell_tags, boundaries=facet_tags,
                                          windkessels=windkessels, element=run.Element,
                                          inlet_plateau_lam=run.InletPlateauLambda)
    return problem, pars, windkessels


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--config", default=str(Path(__file__).parent / "tube3d.yaml"),
                   help="parameter file; every setting is read from it")
    args = p.parse_args(argv)

    comm = MPI.COMM_WORLD
    problem, pars, windkessels = build(args.config, comm=comm)
    run = pars.Run

    out = Path(run.Output)
    if comm.rank == 0:
        out.mkdir(parents=True, exist_ok=True)

    records = []

    def record(prob, step, t, info):
        """Called by the solver after every converged time step, to log diagnostics.

        'prob' is the problem itself (so prob.u_h is the current velocity), 'step' and 't' locate us in time, and 'info' carries the Picard iteration count and errors.
        """
        # How much mass is lost between the inlet and the outlets. For this scheme the global balance is exact by construction, so a large value here means something is wrong with the tags or the assembly, not with the flow field.
        balance = mass_balance(prob.u_h, prob.boundaries, pars.Geometry.InletID,
                               pars.Geometry.OutletIDs)

        row = {"step": step,
               "time": t,
               "Q_in": balance["Q_in"],
               "mass_defect": balance["defect"],
               # ||div u|| / ||grad u||: the pointwise mass error, which for P1-P1 is genuinely nonzero. This is the number that says how good the field is.
               "div_norm": divergence_norm(prob.u_h),
               "picard_iterations": info["iterations"],
               "converged": int(info["converged"])}

        # Flow rate and pressure at each Windkessel outlet
        for wk in windkessels:
            row[f"Q_out_{wk.cap_id}"] = wk.Q_out
            row[f"P_out_{wk.cap_id}"] = float(wk.P_out.value)
            row[f"Pd_{wk.cap_id}"] = wk.Pd_nl
        records.append(row)

    # Run the time loop. Every step performs Picard iterations until the velocity and the Windkessel pressures stop changing, then advances to the next waveform row.
    problem.solve(xdmf_path=str(out / "tube3d.xdmf"), store_after=run.StoreAfter,
                  store_every=run.StoreEvery, max_steps=run.MaxSteps,
                  verbose=run.Verbose, callback=record)

    # Write the diagnostics and print a short summary on rank 0
    if comm.rank == 0 and records:
        csv_path = out / "tube3d.csv"
        with open(csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)

        cap = windkessels[0].cap_id
        p_out = np.array([r[f"P_out_{cap}"] for r in records])
        n_cells = problem.mesh.topology.index_map(3).size_global
        print(f"\n[tube3d] {len(records)} steps, {n_cells} cells")
        print(f"  P_out range              : {p_out.min():.4f} .. {p_out.max():.4f}")
        print(f"  max mass defect          : {max(r['mass_defect'] for r in records):.3e}")
        print(f"  max ||div u||/||grad u|| : {max(r['div_norm'] for r in records):.3e}")
        print(f"  wrote {csv_path}")
    return records


if __name__ == "__main__":
    main()
