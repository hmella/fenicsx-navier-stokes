"""
Patient-specific aorta with four Windkessel outlets.

WHAT THIS CASE IS
-----------------
Blood flow through an aorta reconstructed from medical imaging. Flow enters at the aortic root from a measured waveform, and leaves through four branch vessels, each coupled to a lumped model of the circulation downstream of it.

This is the production case the library was built for. It is large -- roughly 110000 nodes and 640000 tetrahedra, 1000 time steps per heartbeat -- so it is a multi-hour parallel run, not something to try casually. Use --max-steps first to check everything assembles.

UNITS
-----
Everything is CGS, which is conventional in haemodynamics:
- length: cm
- velocity: cm/s (peak inflow here is about 160 cm/s)
- viscosity: poise (blood is about 0.04 P)
- density: g/cm^3 (blood is about 1.06)
- pressure: barye (divide by 1333.22 to get mmHg)
The diagnostics below convert pressures to mmHg, because that is what a clinician reads.

WHY THERE ARE NO PASS/FAIL CHECKS
---------------------------------
There is no trusted reference solution for this geometry, so the script reports physical diagnostics and warns when they look unphysiological, rather than asserting anything. Note in particular that the Windkessel pressures start from zero and take several heartbeats to charge up to physiological values, so the warnings WILL fire during the first cycle. That is expected, not a failure.

WHAT IT WRITES
--------------
output/aorta/aorta_diagnostics.csv, one row per time step, flushed as it goes so that a job killed by a scheduler still leaves usable data; plus XDMF fields if --store-after is given.

Usage:
    mpirun -n 8 python examples/aorta/run.py --config examples/aorta/Ao11mmrest.yaml \\
        --output output/aorta11 --store-after 0
    python examples/aorta/run.py --max-steps 5      # quick check that it runs
"""

# ruff: noqa: E402

# The example scripts are run directly rather than installed, so the package directory is put on the import path before the imports below.
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from mpi4py import MPI

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from dolfinx import io
from dolfinx.fem import Constant
from petsc4py import PETSc

from fenicsx_navier_stokes import (
    ParameterHandler,
    PicardNSProblem,
    Windkessel,
    divergence_norm,
    mass_balance,
)

# CGS pressure (barye, dyn/cm^2) per mmHg
BARYE_PER_MMHG = 1333.22

# Ranges a healthy adult would fall in. Used only to print warnings, never to fail.
SYSTOLIC_RANGE = (60.0, 180.0)
DIASTOLIC_RANGE = (40.0, 110.0)
PULSE_PRESSURE_RANGE = (30.0, 60.0)


def read_mesh(pars, comm):
    """Read the volume mesh and the boundary markers produced by the segmentation."""
    # The volume file carries the tetrahedra and their region markers
    with io.XDMFFile(comm, pars.Geometry.MeshFile, "r") as xdmf:
        mesh = xdmf.read_mesh(name="Grid")
        cell_tags = xdmf.read_meshtags(mesh, name="Grid")

    # Facet markers live in a separate file. dolfinx needs to know which cell each facet belongs to before it can attach them, hence the connectivity call.
    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    with io.XDMFFile(comm, pars.Geometry.BoundariesFile, "r") as xdmf:
        facet_tags = xdmf.read_meshtags(mesh, name="Grid")
    return mesh, cell_tags, facet_tags


def build_windkessels(pars, mesh, facet_tags, dt):
    """Create one RCR circuit per outlet branch.

    Each branch gets its own resistances and compliance, tuned so that the flow divides between the branches the way it does in the patient. P_out is a Constant the momentum equation holds a reference to, so writing into it updates the boundary condition without rebuilding any form.
    """
    cfg = pars.Problem.Windkessels
    models = []
    for i, cap in enumerate(pars.Geometry.OutletIDs):
        models.append(Windkessel(Pd_prev=cfg.Pd_init,
                                 Rd=cfg.Rd[i], Rp=cfg.Rp[i], C=cfg.C[i],
                                 cap_id=cap, Niter=cfg.NbIters, facet_tags=facet_tags,
                                 dt_sim=dt,
                                 P_out=Constant(mesh, PETSc.ScalarType(cfg.Pd_init))))
    return models


class Diagnostics:
    """Collects one row of physical diagnostics per time step and writes it to CSV.

    An instance is passed to solve() as the callback, so it is called after every converged step. Rows are flushed immediately rather than at the end, because these runs are long enough that they are often cut short by a scheduler.
    """

    def __init__(self, path, pars, windkessels, comm, cycle_steps=None):
        self.pars = pars
        self.windkessels = windkessels
        self.comm = comm
        self.cycle_steps = cycle_steps  # steps per heartbeat, for the summary window
        self.rows = []
        self.path = Path(path)
        self._fh = None
        self._writer = None

    def __call__(self, problem, step, t, info):
        pars = self.pars

        # Flow in at the root against flow out through the branches
        balance = mass_balance(problem.u_h, problem.boundaries, pars.Geometry.InletID,
                               pars.Geometry.OutletIDs)

        row = {"step": step,
               "time": t,
               "Q_in": balance["Q_in"],
               "mass_defect": balance["defect"],
               # The pointwise mass error. Unlike mass_defect, which is exact by construction here, this one genuinely measures solution quality.
               "div_norm": divergence_norm(problem.u_h),
               "picard_iterations": info["iterations"],
               "nl_error": info["nl_error"],
               "wk_error": info["wk_error"],
               "converged": int(info["converged"]),
               "wall_time": info["wall_time"]}

        # Per-branch flow, the fraction of total outflow it takes, and its pressure in mmHg. The flow split is worth watching: it should settle to the same values every cycle once the circuits have charged.
        q_total = sum(balance["Q_out"])
        for wk, q in zip(self.windkessels, balance["Q_out"], strict=True):
            row[f"Q_out_{wk.cap_id}"] = q
            row[f"split_{wk.cap_id}"] = q / q_total if q_total != 0.0 else np.nan
            row[f"P_out_{wk.cap_id}_mmHg"] = float(wk.P_out.value) / BARYE_PER_MMHG

        self.rows.append(row)
        self._write(row)

        # A step that runs out of Picard iterations is worth knowing about immediately
        if self.comm.rank == 0 and not info["converged"]:
            print(f"  [warn] step {step}: Picard did not converge "
                  f"({info['iterations']} iterations)", flush=True)

    def _write(self, row):
        """Append one row, opening the file and writing the header on the first call."""
        if self.comm.rank != 0:
            return
        if self._fh is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(self.path, "w", newline="")
            self._writer = csv.DictWriter(self._fh, fieldnames=list(row))
            self._writer.writeheader()
        self._writer.writerow(row)
        self._fh.flush()

    def close(self):
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def summary(self):
        """Print a sanity report at the end of the run. Warnings only, nothing asserted."""
        if self.comm.rank != 0 or not self.rows:
            return

        print(f"\n[aorta] {len(self.rows)} steps")
        print(f"  max mass defect          : {max(r['mass_defect'] for r in self.rows):.3e}")
        print(f"  max ||div u||/||grad u|| : {max(r['div_norm'] for r in self.rows):.3e}")
        print(f"  Picard iterations        : "
              f"{min(r['picard_iterations'] for r in self.rows)}-"
              f"{max(r['picard_iterations'] for r in self.rows)}")
        n_bad = sum(1 for r in self.rows if not r["converged"])
        if n_bad:
            print(f"  [warn] {n_bad} step(s) did not reach the nonlinear tolerance")

        # Report blood pressure per branch over the last heartbeat: the peak is systolic, the trough diastolic, and the difference is the pulse pressure.
        for wk in self.windkessels:
            key = f"P_out_{wk.cap_id}_mmHg"
            p = np.array([r[key] for r in self.rows])
            window = p[-self.cycle_steps:] if self.cycle_steps else p
            systolic, diastolic = float(window.max()), float(window.min())
            pulse = systolic - diastolic
            print(f"  cap {wk.cap_id}: systolic {systolic:7.1f} mmHg, "
                  f"diastolic {diastolic:7.1f} mmHg, pulse {pulse:6.1f} mmHg")
            self._warn(f"cap {wk.cap_id} systolic", systolic, SYSTOLIC_RANGE)
            self._warn(f"cap {wk.cap_id} diastolic", diastolic, DIASTOLIC_RANGE)
            self._warn(f"cap {wk.cap_id} pulse pressure", pulse, PULSE_PRESSURE_RANGE)

        print("  Note: pressures are only physiological once the Windkessels have reached a "
              "periodic state, which takes several cardiac cycles (see "
              "Ao11mmrest-12cycles.yaml).")

    @staticmethod
    def _warn(label, value, bounds):
        lo, hi = bounds
        if not (lo <= value <= hi):
            print(f"  [warn] {label} = {value:.1f} mmHg is outside the expected "
                  f"[{lo:.0f}, {hi:.0f}] mmHg", flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--config", default=str(Path(__file__).parent / "Ao11mmrest.yaml"),
                   help="which aorta and which inflow waveform to run")
    p.add_argument("--output", default="output/aorta")
    p.add_argument("--element", default="P1-P1", choices=["P1-P1", "P2-P1"])
    p.add_argument("--plateau-lam", type=float, default=0.15,
                   help="inlet profile shape: small gives a plug, large a parabola")
    p.add_argument("--max-steps", type=int, default=None, help="stop early (for a quick check)")
    p.add_argument("--store-after", type=int, default=None,
                   help="first step to write to XDMF; omit to write no fields at all")
    p.add_argument("--store-every", type=int, default=10,
                   help="write every n-th step; a full run is 12000 steps, so keep this high")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    comm = MPI.COMM_WORLD

    # Configuration, mesh and inflow waveform. The waveform's row spacing sets the time step: 8e-4 s here, with a heartbeat lasting 0.8 s, so 1000 steps per cycle.
    pars = ParameterHandler(args.config)
    mesh, cell_tags, facet_tags = read_mesh(pars, comm)
    inflow = np.loadtxt(pars.Problem.VelocityProfileFile)
    dt = float(inflow[1, 0] - inflow[0, 0])

    windkessels = build_windkessels(pars, mesh, facet_tags, dt)

    # Assemble the problem. The inlet profile is computed from the geometry of the cap itself rather than assumed to be round, since a segmented aortic root is not.
    problem = PicardNSProblem(parameters=pars, mesh=mesh,
                              XDMF=args.store_after is not None,
                              domains=cell_tags, boundaries=facet_tags,
                              windkessels=windkessels, element=args.element,
                              inlet_plateau_lam=args.plateau_lam)

    # Run, logging diagnostics as we go. The try/finally makes sure the CSV is closed even if the run is interrupted.
    out = Path(args.output)
    cycle_steps = int(round(0.8 / dt))  # one heartbeat
    diagnostics = Diagnostics(out / "aorta_diagnostics.csv", pars, windkessels, comm,
                              cycle_steps)
    try:
        problem.solve(xdmf_path=str(out / "aorta.xdmf"), store_after=args.store_after,
                      store_every=args.store_every, max_steps=args.max_steps,
                      verbose=not args.quiet, callback=diagnostics)
    finally:
        diagnostics.close()

    diagnostics.summary()
    return diagnostics.rows


if __name__ == "__main__":
    main()
