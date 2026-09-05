"""
Stabilized incompressible Navier-Stokes solvers in FEniCSx for vascular haemodynamics.

Monolithic velocity-pressure formulation with BDF2 time stepping and Picard (Oseen) linearization of the convection. Equal-order P1-P1 pairs are stabilized with SUPG/PSPG plus a grad-div term; the inf-sup stable P2-P1 pair is also supported. The saddle-point system is a PETSc MatNest solved either directly (MUMPS) or with FGMRES preconditioned by a Schur-complement fieldsplit with BoomerAMG on both blocks. Outlets can be coupled to three-element RCR Windkessel models.

Requires dolfinx >= 0.10, < 0.11.
"""

from .boundaries import (
    Windkessel,
    backflow_stab,
    facet_vector_approximation,
    inlet_lap_paraboloid,
    inlet_paraboloid,
    pressure_pin_bc,
    tangential_projection,
)
from .constitutive import Newtonian, PowerLaw, eps, powerLaw, shear_rate
from .helpers import Projector, inflow_profile, project
from .parameters import DotDict, ParameterHandler, magnitude
from .postprocess import (
    cross_section_average,
    divergence_norm,
    drag_lift_coefficients,
    drag_lift_surface,
    drag_lift_variational,
    evaluate_at_points,
    flow_rate,
    mass_balance,
)
from .problems import BaseProblem, NewtonNSProblem, PicardNSProblem
from .stabilization import Ck_constant, tau_C, tau_M

__version__ = "0.1.0"

__all__ = [
    "BaseProblem",
    "Ck_constant",
    "DotDict",
    "NewtonNSProblem",
    "Newtonian",
    "ParameterHandler",
    "PicardNSProblem",
    "PowerLaw",
    "Projector",
    "Windkessel",
    "__version__",
    "backflow_stab",
    "cross_section_average",
    "divergence_norm",
    "drag_lift_coefficients",
    "drag_lift_surface",
    "drag_lift_variational",
    "eps",
    "evaluate_at_points",
    "facet_vector_approximation",
    "flow_rate",
    "inflow_profile",
    "inlet_lap_paraboloid",
    "inlet_paraboloid",
    "magnitude",
    "mass_balance",
    "powerLaw",
    "pressure_pin_bc",
    "project",
    "shear_rate",
    "tangential_projection",
    "tau_C",
    "tau_M",
]
