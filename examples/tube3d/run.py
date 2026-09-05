"""
Straight 3D tube with a three-element Windkessel outlet.

WHAT THIS CASE IS
-----------------
A SimVascular-style vessel segment: blood flowing down a straight cylindrical pipe. The inlet is driven by a prescribed flow waveform, and the outlet is not simply left open but coupled to a lumped (0D) model of everything downstream -- the rest of the arterial tree, represented by a resistance-compliance circuit. That is the RCR Windkessel.

WHY THE WINDKESSEL MATTERS
--------------------------
If you just let the outlet be traction-free, the pressure there is pinned to zero and the simulation cannot produce physiological pressures at all. Real vessels feel the resistance and elasticity of the vasculature downstream. The Windkessel supplies that: given the flow rate Q leaving the cap, it returns a pressure P that is imposed back on the fluid.

WHY THIS PARTICULAR SETUP IS A VERIFICATION CASE
------------------------------------------------
The shipped tube3d.yaml chooses the circuit parameters so that Rd*C = 1/(4*pi), which matches the sin^2(2*pi*t) waveform in data/inflow/tube.flow. With that pairing the outlet pressure has an exact closed-form solution, so the simulation can be checked against it (tests/benchmarks/test_tube.py does exactly that). Change the parameters or the waveform on their own and the case still runs, but it stops verifying anything.

WHAT IT WRITES
--------------
output/tube3d/tube3d.csv, one row per time step, with the flow rates, the Windkessel pressure, the mass balance and the divergence norm; plus XDMF fields if --store-after is given.

Usage:
    python examples/tube3d/run.py --radius 0.5 --length 5 --resolution 0.1 --max-steps 200
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
    ParameterHandler,
    PicardNSProblem,
    Windkessel,
    divergence_norm,
    mass_balance,
)


def build(config, radius, length, resolution, wall_resolution=None, element="P1-P1",
          comm=None, plateau_lam=0.15):
    """Put together everything the solver needs: mesh, parameters and outlet models."""
    comm = comm if comm is not None else MPI.COMM_WORLD

    # Read the YAML configuration. Values are reached with dot notation, so for example pars.Problem.Viscosity is the dynamic viscosity in poise.
    pars = ParameterHandler(config)

    # Build the cylinder with gmsh. The returned tags label the boundary facets: wall = 1, inlet = 2, outlet = 3. These numbers must agree with tube3d.yaml, or the solver would find no facets to apply boundary conditions to.
    mesh, cell_tags, facet_tags = tube_mesh.generate(radius=radius, length=length,
                                                     resolution=resolution,
                                                     wall_resolution=wall_resolution,
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
    problem = PicardNSProblem(parameters=pars, mesh=mesh, XDMF=True, domains=cell_tags,
                              boundaries=facet_tags, windkessels=windkessels,
                              element=element, inlet_plateau_lam=plateau_lam)
    return problem, pars, windkessels


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--config", default=str(Path(__file__).parent / "tube3d.yaml"))
    p.add_argument("--output", default="output/tube3d")
    p.add_argument("--radius", type=float, default=2.0, help="cylinder radius")
    p.add_argument("--length", type=float, default=30.0, help="cylinder length")
    p.add_argument("--resolution", type=float, default=None, help="element size (default R/4)")
    p.add_argument("--wall-resolution", type=float, default=None,
                   help="element size at the wall; set it smaller to resolve a boundary layer")
    p.add_argument("--element", default="P1-P1", choices=["P1-P1", "P2-P1"])
    p.add_argument("--plateau-lam", type=float, default=0.15,
                   help="inlet profile shape: small gives a plug, large a parabola")
    p.add_argument("--max-steps", type=int, default=None, help="stop early (for a quick check)")
    p.add_argument("--store-after", type=int, default=None,
                   help="first step to write to XDMF; omit to write no fields at all")
    p.add_argument("--store-every", type=int, default=10, help="write every n-th step")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    comm = MPI.COMM_WORLD
    problem, pars, windkessels = build(args.config, args.radius, args.length,
                                       args.resolution,
                                       wall_resolution=args.wall_resolution,
                                       element=args.element,
                                       plateau_lam=args.plateau_lam, comm=comm)

    out = Path(args.output)
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
    problem.solve(xdmf_path=str(out / "tube3d.xdmf"), store_after=args.store_after,
                  store_every=args.store_every, max_steps=args.max_steps,
                  verbose=not args.quiet, callback=record)

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
