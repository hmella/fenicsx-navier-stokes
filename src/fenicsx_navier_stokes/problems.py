"""
Monolithic stabilized incompressible Navier-Stokes solvers.

Momentum and mass are solved together as one saddle-point system per nonlinear iteration (no projection or fractional-step splitting):

    rho*(3*u^{n+1} - 4*u^n + u^{n-1})/(2*dt) + rho*(u.grad)u^{n+1}
      - div(2*mu*eps(u^{n+1})) + grad(p^{n+1}) = f,      div(u^{n+1}) = 0

Time stepping is BDF2. Two treatments of the nonlinearity are available:

  PicardNSProblem  linearizes the convection about the previous iterate (an Oseen problem). Converges linearly but from anywhere, including from rest.

  NewtonNSProblem  solves the true nonlinear residual with its exact Jacobian, including the rank-one Windkessel coupling term. Converges quadratically from a good enough starting guess, which for a transient problem the previous time step supplies.

Equal-order P1-P1 pairs are stabilized with SUPG/PSPG plus a grad-div term; P2-P1 is also available and is stabilized identically, the terms being consistent.

The 2x2 block system is a PETSc MatNest. The blocks that do not change during the nonlinear iteration are assembled once and added back with axpy each iteration; only the convection and stabilization blocks are re-assembled.
"""

import time
from pathlib import Path

import numpy as np
from basix.ufl import element
from dolfinx import io
from dolfinx.fem import (
    Constant,
    Expression,
    Function,
    assemble_scalar,
    bcs_by_block,
    dirichletbc,
    extract_function_spaces,
    form,
    functionspace,
    locate_dofs_topological,
)
from dolfinx.fem.petsc import apply_lifting, assemble_matrix, assemble_vector, create_matrix, set_bc
from dolfinx.la.petsc import create_vector_wrap
from mpi4py import MPI
from petsc4py import PETSc
from ufl import (
    FacetNormal,
    Measure,
    TestFunction,
    TrialFunction,
    derivative,
    div,
    dot,
    grad,
    inner,
    nabla_grad,
)

from .boundaries import WindkesselNetwork, backflow_stab, inlet_lap_paraboloid, numerical_tangent
from .constitutive import eps
from .helpers import inflow_profile
from .stabilization import tau_C, tau_M

# BDF2 coefficient of the new time level, i.e. 3/2
BDF2_THETA = 1.5

# PETSc log events for the phases that are not themselves PETSc calls, so that a -log_view
# profile accounts for matrix assembly, the Windkessel update and the user callback. Entering an
# event costs a few hundred nanoseconds when logging is off.
_EV_ASSEMBLE_A = PETSc.Log.Event("fxns_assemble_A")
_EV_ASSEMBLE_B = PETSc.Log.Event("fxns_assemble_b")
_EV_WINDKESSEL = PETSc.Log.Event("fxns_windkessel")
_EV_CONVERGENCE = PETSc.Log.Event("fxns_convergence")
_EV_CALLBACK = PETSc.Log.Event("fxns_callback")
_EV_IO = PETSc.Log.Event("fxns_io")

# Separates the time loop from one-off setup: mesh reading, the inlet-profile solve and form
# compilation
_STAGE_TIME_LOOP = PETSc.Log.Stage("fxns_time_loop")

# The *_star matrices are allocated from the union of the constant and iterate-dependent forms,
# so a constant block's pattern is a strict subset of theirs and MatAXPY can add them without
# rebuilding. Not SAME: the constant blocks are allocated from their own form with
# IGNORE_ZERO_ENTRIES on, so their stored pattern is smaller.
_SUBSET_PATTERN = PETSc.Mat.Structure.SUBSET_NONZERO_PATTERN


# Writes PETSc options under a per-instance prefix
class _PrefixedOptions:
    """
    Maps opts["ksp_rtol"] onto the prefixed key PETSc actually reads, so the option-setting code below reads exactly as it did when it wrote into the global database.
    """

    def __init__(self, prefix):
        self.prefix = prefix
        self.db = PETSc.Options()

    def __setitem__(self, key, value):
        self.db[self.prefix + key] = value


# Machinery shared by every problem
class BaseProblem:
    """
    Everything that is not a variational form: the function spaces, the boundary conditions, the coefficients, the block matrices, the linear solver and the time loop.

    Subclasses supply `set_problem()`, which builds their forms and then calls the helpers below, and `step()`, which advances one time step.

    Arguments:

      parameters - a ParameterHandler (or anything with the same Problem / Solver / Geometry layout).

      mesh, domains, boundaries - the mesh and its cell and facet tags.

      windkessels - sequence of Windkessel outlet models, or None.

      element - 'P1-P1' (stabilized equal order) or 'P2-P1' (Taylor-Hood).

      XDMF, HDF5 - output flags.

      ds_quad_deg, dx_quad_deg - fallback quadrature degrees, used only when the parameter file omits them.

      inlet_plateau_lam - length scale of the screened-Poisson inlet profile.

      f - body force (ufl expression or Function); None means no body force. Needed for manufactured-solution verification, not by the physical cases.

      dt, num_steps - time step and number of steps; default to the inflow file's row spacing and row count.

      inlet_profile - spatial shape of the inlet velocity, as a Function on the velocity space. Defaults to the screened-Poisson profile. The benchmarks pass an analytic profile here instead.

      inlet_scale - callable (step, t) -> float multiplying inlet_profile; defaults to the inflow waveform.

      wall_velocity - Dirichlet value on the wall; defaults to no slip.

      extra_bcs - additional DirichletBCs, or a callable (V, Q) -> list returning them. Prefer the callable: a condition must be built on this problem's own spaces. Pressure conditions belong here (see pressure_pin_bc).

      sigma_BDF - transient coefficient of tau_M, default 2.0.

      tau_C_coefficient - multiplier of the grad-div parameter, default 0.4.

      supg_viscous - include the streamline-viscous SUPG term. It has no right-hand-side counterpart and for P1 the viscous part of the residual vanishes elementwise, so it is formally inconsistent. Measured effect: none, the manufactured-solution errors being identical to four significant figures with it on and off, its coefficient being tau_M*mu ~ dt*mu. Left on so the physical cases keep the behaviour they were run with.
    """

    def __init__(
        self,
        parameters=None,
        mesh=None,
        XDMF=True,
        HDF5=False,
        domains=None,
        boundaries=None,
        windkessels=None,
        element="P2-P1",
        ds_quad_deg=3,
        dx_quad_deg=4,
        inlet_plateau_lam=None,
        f=None,
        dt=None,
        num_steps=None,
        inlet_profile=None,
        inlet_scale=None,
        wall_velocity=None,
        extra_bcs=None,
        sigma_BDF=2.0,
        tau_C_coefficient=0.4,
        supg_viscous=True,
    ):
        self.parameters = parameters
        self.mesh = mesh
        self.XDMF = XDMF
        self.HDF5 = HDF5
        self.domains = domains
        self.boundaries = boundaries
        # The solver talks to one 0D network, not to a list of outlets. A plain list of
        # Windkessel objects is wrapped so that callers can keep passing one.
        if windkessels is None or isinstance(windkessels, WindkesselNetwork):
            self.outlet_model = windkessels
        else:
            self.outlet_model = WindkesselNetwork(windkessels)
        self.windkessels = list(self.outlet_model) if self.outlet_model is not None else None
        self.element = element
        self.ds_quad_deg = ds_quad_deg
        self.dx_quad_deg = dx_quad_deg
        self.inlet_plateau_lam = inlet_plateau_lam
        self.f = f
        self.sigma_BDF = sigma_BDF
        self.tau_C_coefficient = tau_C_coefficient
        self.supg_viscous = supg_viscous
        self._dt_arg = dt
        self._num_steps_arg = num_steps
        self._inlet_profile_arg = inlet_profile
        self._inlet_scale_arg = inlet_scale
        self._wall_velocity_arg = wall_velocity
        self._extra_bcs_arg = extra_bcs
        self.V, self.Q = self.function_space()
        self.set_problem()

    # Distinguishes each instance's PETSc options prefix
    _ksp_counter = 0

    # Polynomial degree of the velocity space
    @property
    def velocity_degree(self):
        return 2 if self.element == "P2-P1" else 1

    # Create function spaces
    def function_space(self):
        cell = self.mesh.topology.cell_name()
        gdim = self.mesh.geometry.dim
        if self.element == "P2-P1":
            VE = element("Lagrange", cell, 2, shape=(gdim,))
            FE = element("Lagrange", cell, 1)
        elif self.element == "P1-P1":
            VE = element("Lagrange", cell, 1, shape=(gdim,))
            FE = element("Lagrange", cell, 1)
        else:
            raise ValueError(f"unknown element pair {self.element!r}; expected 'P1-P1' or 'P2-P1'")
        V = functionspace(self.mesh, VE)
        Q = functionspace(self.mesh, FE)

        ndofs = (
            V.dofmap.index_map.size_global * V.dofmap.index_map_bs
            + Q.dofmap.index_map.size_global * Q.dofmap.index_map_bs
        )
        if self.mesh.comm.rank == 0:
            print(f"[Function space] {self.element}, {ndofs:d} global dofs", flush=True)
        return V, Q

    # Set the initial condition
    def initial_condition(self, u0=None, u00=None):
        """
        Set the two previous velocity levels u^n and u^{n-1}; None leaves a level at zero.

        Both start at zero, so a cold start is BDF2 applied to two identical zero levels. That is consistent for cases that ramp up from rest, but it makes the first step first-order, so a temporal convergence study must seed both levels here.
        """
        for target, source in ((self.u0, u0), (self.u00, u00)):
            if source is None:
                continue
            if isinstance(source, Function):
                target.x.array[:] = source.x.array
            else:
                target.interpolate(Expression(source, self.V.element.interpolation_points))
            target.x.scatter_forward()
        return self

    # Build the forms, matrices and solver. Supplied by the subclass.
    def set_problem(self):
        raise NotImplementedError("subclasses must build their own variational forms")

    # Advance one time step. Supplied by the subclass.
    def step(self, n_time, t, verbose=True, pre_step=None):
        raise NotImplementedError("subclasses must implement their own nonlinear iteration")

    # ---------------------------------------------------------------------------------
    # Setup helpers, called from the subclasses' set_problem()
    # ---------------------------------------------------------------------------------

    # Define measures for integrals
    def _setup_measures(self):
        pars = self.parameters
        ds_deg = getattr(pars.Problem, "SurfaceQuadratureDegree", self.ds_quad_deg)
        dx_deg = getattr(pars.Problem, "VolumeQuadratureDegree", self.dx_quad_deg)
        self.ds = Measure("ds", domain=self.mesh, subdomain_data=self.boundaries,
                          metadata={"quadrature_degree": ds_deg})
        self.dx = Measure("dx", domain=self.mesh, subdomain_data=self.domains,
                          metadata={"quadrature_degree": dx_deg})

    # Time stepping data: taken from the arguments, else from the inflow waveform file
    def _setup_time(self):
        pars = self.parameters
        self.inflow = None
        profile_file = getattr(pars.Problem, "VelocityProfileFile", None)
        if profile_file is not None:
            self.inflow = inflow_profile(
                profile_file, scale_factor=pars.Problem.VelocityProfileScale
            )

        if self._dt_arg is not None:
            self.dt_value = float(self._dt_arg)
        elif self.inflow is not None:
            self.dt_value = float(self.inflow[1, 0] - self.inflow[0, 0])
        else:
            raise ValueError("provide dt explicitly, or a Problem.VelocityProfileFile")

        if self._num_steps_arg is not None:
            self.num_steps = int(self._num_steps_arg)
        elif self.inflow is not None:
            self.num_steps = int(self.inflow.shape[0])
        else:
            raise ValueError("provide num_steps explicitly, or a Problem.VelocityProfileFile")

        # Scalar multiplying the inlet profile at each step
        if self._inlet_scale_arg is not None:
            self.inlet_scale_fn = self._inlet_scale_arg
        elif self.inflow is not None:
            waveform = self.inflow[:, 1]
            self.inlet_scale_fn = lambda step, t, _w=waveform: float(_w[min(step, _w.shape[0] - 1)])
        else:
            self.inlet_scale_fn = lambda step, t: 0.0

    # Find inlet, outlet and wall dofs, then build the boundary conditions
    def _setup_bcs(self):
        pars = self.parameters
        mesh = self.mesh
        fdim = mesh.topology.dim - 1

        # WallID may be a single tag or a list: the DFG cylinder benchmark needs the channel walls and the obstacle tagged separately (so drag can be integrated over the obstacle alone) while both carry no slip
        wall_id = pars.Geometry.WallID
        wall_ids = [wall_id] if np.isscalar(wall_id) else list(wall_id)
        inlet_id = pars.Geometry.InletID
        outlet_ids = list(pars.Geometry.OutletIDs)
        self.wall_ids, self.outlet_ids, self.inlet_id = wall_ids, outlet_ids, inlet_id

        wall_facets = (
            np.unique(np.hstack([self.boundaries.find(w) for w in wall_ids]))
            if wall_ids
            else np.zeros(0, dtype=np.int32)
        )
        wall_dofs = locate_dofs_topological(self.V, fdim, wall_facets)
        inlet_dofs = locate_dofs_topological(self.V, fdim, self.boundaries.find(inlet_id))

        # A configuration pointed at the wrong mesh gives empty dof sets, hence no boundary conditions at all and a silently meaningless solve. The aorta meshes do not agree on the wall tag (4 for the 9 and 11 mm cases, 1 for 13 mm), so refuse loudly here
        tags_present = sorted(set(np.unique(self.boundaries.values).tolist()))
        for name, tag, dofs in (("WallID", wall_ids, wall_dofs), ("InletID", inlet_id, inlet_dofs)):
            if mesh.comm.allreduce(len(dofs), op=MPI.SUM) == 0:
                raise ValueError(
                    f"{name}={tag} matches no facets on this mesh; tags present: {tags_present}. The parameter file "
                    "and the mesh most likely do not correspond."
                )
        for oid in outlet_ids:
            if mesh.comm.allreduce(len(self.boundaries.find(oid)), op=MPI.SUM) == 0:
                raise ValueError(
                    f"OutletID {oid} matches no facets on this mesh; tags present: {tags_present}."
                )

        # Spatial shape of the inlet velocity
        if self._inlet_profile_arg is not None:
            par = self._inlet_profile_arg
        else:
            par = inlet_lap_paraboloid(
                self.V, self.boundaries, inlet_id, wall_id, plateau_lam=self.inlet_plateau_lam
            )
        self.inlet_shape = par

        # Wall velocity (no slip unless one was given)
        u_wall = Function(self.V)
        u_wall.x.array[:] = 0.0
        if self._wall_velocity_arg is not None:
            if isinstance(self._wall_velocity_arg, Function):
                u_wall.x.array[:] = self._wall_velocity_arg.x.array
            else:
                u_wall.interpolate(
                    Expression(self._wall_velocity_arg, self.V.element.interpolation_points)
                )
            u_wall.x.scatter_forward()
        self.u_wall = u_wall

        # Inlet velocity: a fixed spatial profile times a scalar updated every step
        self.u_inlet = Function(self.V)
        self.u_inlet_scale = Constant(mesh, PETSc.ScalarType(0.0))
        self.u_inlet_expr = Expression(
            par * self.u_inlet_scale, self.V.element.interpolation_points
        )
        self.u_inlet.interpolate(self.u_inlet_expr)

        self.bcs = [dirichletbc(self.u_inlet, inlet_dofs), dirichletbc(u_wall, wall_dofs)]
        if self._extra_bcs_arg is not None:
            # A callable is resolved here, once V and Q exist: a DirichletBC must be built on the very spaces the forms use, or bcs_by_block will not match it and it is silently ignored
            extra = self._extra_bcs_arg
            self.bcs += list(extra(self.V, self.Q) if callable(extra) else extra)

    # Solution functions, previous time levels and the physical coefficients
    def _setup_state(self):
        mesh = self.mesh
        pars = self.parameters

        # Initial conditions and the function the convection is linearized about
        self.u0 = Function(self.V)
        self.u00 = Function(self.V)
        self.up = Function(self.V)
        for fn in (self.u0, self.u00, self.up):
            fn.x.array[:] = 0.0

        # Solution functions. Their vectors are combined into a nested vector, so that the solver writes straight into u_h and p_h
        self.u_h, self.p_h = Function(self.V), Function(self.Q)
        self.u_h.name = "u"
        self.p_h.name = "p"
        self.x = PETSc.Vec().createNest(
            [create_vector_wrap(self.u_h.x), create_vector_wrap(self.p_h.x)]
        )

        # Time step, density and viscosity
        self.dt = Constant(mesh, PETSc.ScalarType(self.dt_value))
        self.rho = Constant(mesh, PETSc.ScalarType(pars.Problem.Density))
        self.mu = Constant(mesh, PETSc.ScalarType(pars.Problem.Viscosity))

        # Backflow stabilization strength
        self.beta = Constant(mesh, PETSc.ScalarType(getattr(pars.Problem, "BackflowBeta", 0.0)))

        # Stabilization parameters. tau_M is built on `up`, not on the solution, so it stays frozen through a Newton iteration; ufl.derivative steps over it rather than differentiating sqrt(dot(u,u)), which is singular at u = 0
        self.tau_M = tau_M(
            self.up, self.dt, self.rho, self.mu, mesh=mesh,
            sigma_BDF=self.sigma_BDF, degree=self.velocity_degree,
        )
        self.tau_C = tau_C(self.rho, self.mu, coefficient=self.tau_C_coefficient)

    # Allocate the block matrices and compose the nest
    def _allocate_blocks(self, a00, a01, a10, alloc=None):
        """
        `a00`, `a01`, `a10` are the constant (iteration-independent) Galerkin forms. The iteration-dependent parts must already be on self.a**_star and compiled into self._f_a**_star.

        `alloc` optionally gives three extra forms whose stencils are unioned into the allocation. A subclass that assembles a *different* operator into these matrices must pass it, because NEW_NONZERO_LOCATIONS is switched off afterwards and any entry outside the pattern would then be silently dropped.
        """
        extra00, extra01, extra10 = alloc if alloc is not None else (None, None, None)
        alloc00 = a00 + self.a00_star + (extra00 if extra00 is not None else 0 * a00)
        alloc01 = a01 + self.a01_star + (extra01 if extra01 is not None else 0 * a01)
        alloc10 = a10 + self.a10_star + (extra10 if extra10 is not None else 0 * a10)

        # The *_star matrices are allocated against the COMBINED sparsity pattern, so that the later axpy with the constant block never needs a new nonzero location
        self.A00_star = create_matrix(form(alloc00))
        self.A00_star.setOption(PETSc.Mat.Option.IGNORE_ZERO_ENTRIES, False)
        assemble_matrix(self.A00_star, self._f_a00_star, bcs=self.bcs, diag=0.0)
        self.A00_star.assemble()
        self.A00_star.setOption(PETSc.Mat.Option.NEW_NONZERO_LOCATIONS, False)

        self.A01_star = create_matrix(form(alloc01))
        self.A01_star.setOption(PETSc.Mat.Option.IGNORE_ZERO_ENTRIES, False)
        assemble_matrix(self.A01_star, self._f_a01_star, bcs=self.bcs, diag=0.0)
        self.A01_star.assemble()
        self.A01_star.setOption(PETSc.Mat.Option.NEW_NONZERO_LOCATIONS, False)

        self.A10_star = create_matrix(form(alloc10))
        self.A10_star.setOption(PETSc.Mat.Option.IGNORE_ZERO_ENTRIES, False)
        assemble_matrix(self.A10_star, self._f_a10_star, bcs=self.bcs, diag=0.0)
        self.A10_star.assemble()
        self.A10_star.setOption(PETSc.Mat.Option.NEW_NONZERO_LOCATIONS, False)

        # The (1,1) PSPG block, allocated here and re-assembled every iteration like the others: tau_M is built on `up`, so this block depends on the iterate too, and the continuity row must weight one consistent strong residual across its (1,0) and (1,1) parts. The full bc list is passed: both spaces here are the pressure space, so dolfinx discards the velocity conditions, while a pressure condition in the list is applied and makes the block non-singular in the fully-Dirichlet case
        self.A11_star = create_matrix(form(self.a11_star))
        self.A11_star.setOption(PETSc.Mat.Option.SYMMETRIC, True)
        self.A11_star.setOption(PETSc.Mat.Option.SYMMETRY_ETERNAL, True)
        self.A11_star.setOption(PETSc.Mat.Option.IGNORE_ZERO_ENTRIES, True)
        assemble_matrix(self.A11_star, form(self.a11_star), bcs=self.bcs, diag=1.0)
        self.A11_star.assemble()
        self.A11_star.setOption(PETSc.Mat.Option.NEW_NONZERO_LOCATIONS, False)

        # Mass + viscous block, constant in time
        self.A00 = create_matrix(form(a00))
        self.A00.setOption(PETSc.Mat.Option.SYMMETRIC, True)
        self.A00.setOption(PETSc.Mat.Option.SYMMETRY_ETERNAL, True)
        self.A00.setOption(PETSc.Mat.Option.IGNORE_ZERO_ENTRIES, True)
        assemble_matrix(self.A00, form(a00), bcs=self.bcs, diag=1.0)
        self.A00.assemble()
        self.A00.setOption(PETSc.Mat.Option.NEW_NONZERO_LOCATIONS, False)

        # Divergence blocks, also constant. For these rectangular blocks the boundary conditions zero the COLUMNS belonging to constrained velocity dofs -- the counterpart of apply_lifting on the right-hand side. The diag argument would be ignored here, since the test and trial spaces differ
        self.A01 = create_matrix(form(a01))
        self.A01.setOption(PETSc.Mat.Option.IGNORE_ZERO_ENTRIES, True)
        assemble_matrix(self.A01, form(a01), bcs=self.bcs)
        self.A01.assemble()
        self.A01.setOption(PETSc.Mat.Option.NEW_NONZERO_LOCATIONS, False)

        self.A10 = create_matrix(form(a10))
        self.A10.setOption(PETSc.Mat.Option.IGNORE_ZERO_ENTRIES, True)
        assemble_matrix(self.A10, form(a10), bcs=self.bcs)
        self.A10.assemble()
        self.A10.setOption(PETSc.Mat.Option.NEW_NONZERO_LOCATIONS, False)

        # Add the constant blocks onto the star blocks
        self.A00_star.axpy(1.0, self.A00, structure=_SUBSET_PATTERN)
        self.A01_star.axpy(1.0, self.A01, structure=_SUBSET_PATTERN)
        self.A10_star.axpy(1.0, self.A10, structure=_SUBSET_PATTERN)

        # Create nested matrix
        self.A = create_matrix(self.a, kind="nest")
        self.A.setOption(PETSc.Mat.Option.IGNORE_ZERO_ENTRIES, True)
        self.A.createNest([[self.A00_star, self.A01_star], [self.A10_star, self.A11_star]])
        self.A.assemble()
        self.A.setOption(PETSc.Mat.Option.NEW_NONZERO_LOCATIONS, False)

    # Create and configure solver
    def _setup_solver(self, operator=None):
        """
        `operator` is the matrix the Krylov method applies. It defaults to the assembled nest; a subclass may pass a different one, in which case the nest is still used as the preconditioner.
        """
        pars = self.parameters
        self.ksp = PETSc.KSP().create(self.mesh.comm)
        if operator is None or operator is self.A:
            self.ksp.setOperators(self.A)
        else:
            # Krylov sees the exact operator, the preconditioner keeps the sparse nest
            self.ksp.setOperators(operator, self.A)

        # PETSc's options database is global and persists for the life of the process, so each
        # instance writes under its own prefix
        BaseProblem._ksp_counter += 1
        prefix = f"ns{BaseProblem._ksp_counter}_"
        self._guess_nonzero = False
        self._is_iterative = pars.Solver.Kind == "iterative"

        # Preconditioner reuse policy, see _prepare_preconditioner. A refresh period of 0
        # disables reuse and rebuilds on every solve, which is the behaviour this code had
        # before the policy existed.
        self._pc_period = int(getattr(pars.Solver, "PreconditionerRefresh", 5))
        self._pc_guard = float(getattr(pars.Solver, "PreconditionerGuard", 1.5))
        self._pc_steps_since_rebuild = 0
        self._pc_its_at_rebuild = None
        self._pc_last_its = 0
        self._pc_rebuilt_this_solve = False

        # Relative step used to perturb the 0D model when building the coupling tangent
        self._tangent_epsilon = float(getattr(pars.Solver, "TangentPerturbation", 1.0e-6))

        # Two-level linear tolerance, see _linear_rtol
        self._rtol_final = pars.Solver.SolverRTol
        nl_tol = pars.Solver.NonlinearTolerance
        self._rtol_first = float(
            getattr(pars.Solver, "SolverRTolFirst", max(self._rtol_final, 0.1 * nl_tol))
        )
        self.ksp.setOptionsPrefix(prefix)
        opts = _PrefixedOptions(prefix)

        if pars.Solver.Kind == "iterative":
            # FGMRES preconditioned by a Schur-complement fieldsplit, with one BoomerAMG V-cycle on each block. Required for the 3D patient-specific meshes, where a direct factorization does not fit
            self.ksp.setType("fgmres")
            self.ksp.getPC().setType("fieldsplit")
            opts["ksp_gmres_cgs_refinement_type"] = "refine_ifneeded"
            opts["ksp_max_it"] = pars.Solver.SolverIterations
            opts["ksp_converged_reason"] = None
            opts["ksp_rtol"] = pars.Solver.SolverRTol
            opts["ksp_atol"] = pars.Solver.SolverATol
            opts["ksp_gmres_restart"] = pars.Solver.SolverGMRESRestart
            opts["ksp_pc_side"] = "right"

            # Start each solve from the current contents of the solution vector: the previous
            # Picard iterate, or on the first iteration of a step the converged previous time
            # level. The Newton path turns this off again, since it solves for a correction.
            self.ksp.setInitialGuessNonzero(True)
            self._guess_nonzero = True

            # Define the velocity and pressure blocks of the preconditioner from the index sets of the nested matrix
            nested_IS = self.A.getNestISs()
            self.ksp.getPC().setFieldSplitIS(("u", nested_IS[0][0]), ("p", nested_IS[0][1]))

            opts["pc_fieldsplit_type"] = "schur"
            opts["pc_fieldsplit_schur_fact_type"] = getattr(
                pars.Solver, "SchurFactType", "lower"
            )

            # `selfp` approximates the Schur complement by A11 - A10 diag(A00)^-1 A01. The
            # (1,1) block on its own is not usable here: PSPG makes it a pure-Neumann
            # Laplacian, singular on constants, and the cold start diverges with
            # KSP_DIVERGED_DTOL on the first solve.
            opts["pc_fieldsplit_schur_precondition"] = getattr(
                pars.Solver, "SchurPrecondition", "selfp"
            )

            # BoomerAMG settings for tetrahedra: HMIS coarsening, ext+i interpolation and one
            # level of aggressive coarsening, with a strong threshold of 0.5. PETSc's default
            # threshold of 0.25 is the two-dimensional value.
            amg = {
                "strong_threshold": getattr(pars.Solver, "AMGStrongThreshold", 0.5),
                "coarsen_type": getattr(pars.Solver, "AMGCoarsenType", "HMIS"),
                "interp_type": getattr(pars.Solver, "AMGInterpType", "ext+i"),
                "agg_nl": getattr(pars.Solver, "AMGAggressiveLevels", 1),
            }
            for blk in ("u", "p"):
                opts[f"fieldsplit_{blk}_ksp_type"] = "preonly"
                opts[f"fieldsplit_{blk}_pc_type"] = "hypre"
                opts[f"fieldsplit_{blk}_pc_hypre_type"] = "boomeramg"
                for key, value in amg.items():
                    opts[f"fieldsplit_{blk}_pc_hypre_boomeramg_{key}"] = value

        elif pars.Solver.Kind == "direct":
            # Direct LU with MUMPS: robust, and the right choice for the small 2D cases.
            #
            # When the operator differs from the preconditioner, `preonly` would apply only the LU of the preconditioner and so silently solve the WRONG system. Iterate instead: preconditioned by an exact factorization of the sparse part, GMRES converges in a couple of iterations.
            if operator is None or operator is self.A:
                self.ksp.setType("preonly")
            else:
                self.ksp.setType("gmres")
                opts["ksp_rtol"] = getattr(pars.Solver, "SolverRTol", 1.0e-10)
                opts["ksp_max_it"] = 100
            self.ksp.getPC().setType("lu")
            self.ksp.getPC().setFactorSolverType("mumps")
            self.ksp.getPC().setReusePreconditioner(False)
            opts["mat_mumps_icntl_14"] = 80  # increase MUMPS working memory
            opts["ksp_error_if_not_converged"] = 1

        else:
            raise ValueError(
                f"unknown Solver.Kind {pars.Solver.Kind!r}; expected 'iterative' or 'direct'"
            )

        # Apply options and set up solver
        self.ksp.setFromOptions()

    # Decide whether this solve rebuilds the preconditioner or reuses the previous one
    def _prepare_preconditioner(self, first_of_step):
        """
        FGMRES applies the true, current operator in every matrix-vector product and the preconditioner only steers the search, so reuse changes the Krylov iteration count and nothing else.

        Two tiers. Within a time step, iterations after the first reuse unconditionally. Across time steps a rebuild is forced every `PreconditionerRefresh` steps, or sooner once the Krylov count has grown by more than a factor `PreconditionerGuard` since the last rebuild.

        KSPGetIterationNumber is collective and returns the same value on every rank, so the guard reaches the same decision everywhere.
        """
        if not self._is_iterative or self._pc_period <= 0:
            return
        rebuild = False
        if first_of_step:
            if self._pc_its_at_rebuild is None:
                rebuild = True  # first solve of the run
            elif self._pc_steps_since_rebuild >= self._pc_period:
                rebuild = True
            elif self._pc_last_its > self._pc_guard * self._pc_its_at_rebuild:
                rebuild = True
        self.ksp.getPC().setReusePreconditioner(not rebuild)
        self._pc_rebuilt_this_solve = rebuild
        if rebuild:
            self._pc_steps_since_rebuild = 0

    # Record what the last solve cost, for the reuse guard
    def _record_linear_iterations(self, its):
        if self._pc_rebuilt_this_solve:
            # The reference count is the one a fresh preconditioner achieves
            self._pc_its_at_rebuild = its
            self._pc_rebuilt_this_solve = False
        self._pc_last_its = its

    # Relative tolerance for this nonlinear iteration
    def _linear_rtol(self, it):
        """
        The first nonlinear iteration supplies the linearization point and tau_M for the next one and gets a looser tolerance: by default a tenth of the nonlinear tolerance, and never looser than the final one. Every iteration after it uses the full tolerance.
        """
        return self._rtol_first if it == 0 else self._rtol_final

    # Forms used by the convergence test, compiled once
    def _setup_convergence_forms(self):
        self._diff = Function(self.V)
        self._err_num = form(inner(self._diff, self._diff) * self.dx)
        self._err_den = form(inner(self.u_h, self.u_h) * self.dx)

    # Update the inlet velocity to the new time level
    def _update_inlet(self, n_time, t):
        self.u_inlet_scale.value = self.inlet_scale_fn(n_time, t)
        self.u_inlet.interpolate(self.u_inlet_expr)

    # Relative L2 difference between the current solution and `other`
    def _relative_velocity_change(self, other):
        self._diff.x.array[:] = other.x.array - self.u_h.x.array
        num = self.mesh.comm.allreduce(assemble_scalar(self._err_num), op=MPI.SUM)
        den = self.mesh.comm.allreduce(assemble_scalar(self._err_den), op=MPI.SUM)
        return float(np.sqrt(num / den)) if den > 0.0 else 0.0

    # Roll the solution into the previous time levels
    def _advance_time_levels(self):
        self._pc_steps_since_rebuild += 1
        if self.outlet_model is not None:
            self.outlet_model.advance()
        self.u00.x.array[:] = self.u0.x.array
        self.u0.x.array[:] = self.u_h.x.array

    # Time stepping loop
    def solve(
        self,
        xdmf_path="output/solution.xdmf",
        store_after=None,
        store_every=1,
        max_steps=None,
        verbose=True,
        callback=None,
        pre_step=None,
    ):
        """
        Run the time loop and return one dict per step.

        Velocity and pressure are written to <stem>_u.xdmf and <stem>_p.xdmf, since they live on different spaces. 'store_after' is the first step index written (None writes nothing) and 'store_every' thins the output -- a full aorta run is 12000 steps, which is tens of gigabytes if every one is written.

        'callback(problem, step, t, info)' runs after each step, and is what the examples use to record diagnostics.
        """
        comm = self.mesh.comm
        rank = comm.rank
        dt = self.dt_value
        num_steps = self.num_steps if max_steps is None else min(self.num_steps, int(max_steps))

        self.u_h.x.scatter_forward()
        self.p_h.x.scatter_forward()

        # Output files
        file_u = file_p = None
        if self.XDMF and store_after is not None:
            out = Path(xdmf_path)
            if rank == 0:
                out.parent.mkdir(parents=True, exist_ok=True)
            comm.Barrier()
            file_u = io.XDMFFile(comm, out.parent / (out.stem + "_u.xdmf"), "w")
            file_u.write_mesh(self.mesh)
            file_p = io.XDMFFile(comm, out.parent / (out.stem + "_p.xdmf"), "w")
            file_p.write_mesh(self.mesh)

        history = []
        t = 0.0

        # Pushed rather than used as a context manager so the loop body keeps its indentation
        _STAGE_TIME_LOOP.push()
        try:
            for n_time in range(num_steps):
                # Update time
                start_solver = time.perf_counter()
                t += dt
                if rank == 0:
                    print(f"Solving time step {n_time:d} (time: {t:.4f})", flush=True)

                info = self.step(n_time, t, verbose=verbose, pre_step=pre_step)
                info["time"] = t
                info["step"] = n_time
                info["wall_time"] = time.perf_counter() - start_solver
                history.append(info)

                # Print time required to solve the current time step
                if rank == 0:
                    status = "converged" if info["converged"] else "NOT converged"
                    method = info.get("method", "picard")
                    print(
                        "  Time step {} in {:.1f} seconds ({} {} iterations)".format(
                            status, info["wall_time"], info["iterations"], method
                        ),
                        flush=True,
                    )

                if callback is not None:
                    with _EV_CALLBACK:
                        callback(self, n_time, t, info)

                # Write to file
                if (
                    file_u is not None
                    and n_time >= store_after
                    and (n_time - store_after) % store_every == 0
                ):
                    with _EV_IO:
                        file_u.write_function(self.u_h, t)
                        file_u.flush()
                        file_p.write_function(self.p_h, t)
                        file_p.flush()
        finally:
            _STAGE_TIME_LOOP.pop()
            # Close files
            if file_u is not None:
                file_u.close()
            if file_p is not None:
                file_p.close()

        return history


# BDF2 / Picard stabilized Navier-Stokes problem
class PicardNSProblem(BaseProblem):
    """
    Convection is linearized about the previous iterate, giving an Oseen problem at every nonlinear iteration. Robust from any starting point, including rest, but only linearly convergent.

    See BaseProblem for the constructor arguments.
    """

    # Build boundary conditions, forms, matrices and the solver
    def set_problem(self):
        self._setup_measures()
        self._setup_time()
        self._setup_bcs()
        self._setup_state()
        a00, a01, a10 = self._build_oseen_forms()
        self._allocate_blocks(a00, a01, a10)
        self._setup_solver()
        self._setup_convergence_forms()

    # The Oseen (Picard) forms.
    #
    # Shared with NewtonNSProblem, which runs a Picard step on the first time level and again
    # whenever a Newton step fails to converge.
    def _build_oseen_forms(self):
        """
        Sets self.a, self.L and the four a**_star forms, and returns the constant Galerkin blocks (a00, a01, a10) for the matrix allocation.
        """
        mesh = self.mesh
        dx, ds = self.dx, self.ds
        dt, rho, mu, beta = self.dt, self.rho, self.mu, self.beta
        tM, tC = self.tau_M, self.tau_C
        n = FacetNormal(mesh)

        # Trial and test functions
        u, v = TrialFunction(self.V), TestFunction(self.V)
        p, q = TrialFunction(self.Q), TestFunction(self.Q)

        # ---------------------------------------------------------------------------------
        # Variational formulation (Galerkin part)
        # ---------------------------------------------------------------------------------
        a00_conv = rho * inner(grad(u) * self.up, v) * dx
        # The grad-div term sits with the mass and viscous terms in the constant block: tau_C
        # is 0.4*mu/rho, with no dependence on the iterate or on the element size
        a00 = (rho / dt * inner(BDF2_THETA * u, v) * dx
               + 2 * mu * inner(eps(u), eps(v)) * dx
               + tC * rho * div(u) * div(v) * dx)
        a01 = -p * div(v) * dx
        a10 = -q * div(u) * dx
        L0 = rho / dt * inner(2 * self.u0 - 0.5 * self.u00, v) * dx
        L1 = Constant(mesh, PETSc.ScalarType(0.0)) * q * dx

        # Body force
        if self.f is not None:
            L0 += inner(self.f, v) * dx

        # Add backflow stabilization force at the outlets (explicit: built on the previous iterate, so it only ever reaches the right-hand side)
        stab_force = backflow_stab(self.up, n, rho, beta)
        for oid in self.outlet_ids:
            L0 += dot(stab_force, v) * ds(oid)

        # Windkessel outlet boundary condition, as a constant normal traction
        if self.windkessels is not None:
            for wksl in self.windkessels:
                L0 += -dot(wksl.P_out * n, v) * ds(wksl.cap_id)

        # ---------------------------------------------------------------------------------
        # SUPG/PSPG stabilization
        # ---------------------------------------------------------------------------------
        # The two halves of the strong momentum residual: the terms in the unknown u, and the known data at the previous time levels. Both carry a SINGLE factor of rho, so that together they form one residual that vanishes on the exact solution. The viscous part is omitted; it vanishes elementwise for P1 velocities
        res_lhs = rho * (BDF2_THETA / dt * u + grad(u) * self.up)
        res_rhs = rho / dt * (2.0 * self.u0 - 0.5 * self.u00)
        if self.f is not None:
            res_rhs = res_rhs + self.f

        # Momentum row: streamline weight (up.grad)v against the residual
        a_SUPG_00 = tM * inner(grad(v) * self.up, res_lhs) * dx
        if self.supg_viscous:
            a_SUPG_00 += tM * mu * inner(nabla_grad(grad(v) * self.up), nabla_grad(u)) * dx

        # Pressure gradient in the momentum row (SUPG part)
        a_SUPG_01 = tM * inner(grad(p), grad(v) * self.up) * dx

        # Continuity row weighted by -grad(q) (PSPG), and the pressure-pressure coupling
        a_PSPG_10 = tM * inner(-grad(q), res_lhs) * dx
        a_PSPG_11 = tM * inner(-grad(q), grad(p)) * dx

        # Consistent right-hand side counterparts
        L0 += tM * inner(grad(v) * self.up, res_rhs) * dx
        L1 += tM * inner(-grad(q), res_rhs) * dx

        # Variational forms of the full 2x2 system
        self.a = form([[a00 + a00_conv + a_SUPG_00, a01 + a_SUPG_01], [a10 + a_PSPG_10, a_PSPG_11]])
        self.L = form([L0, L1])

        # The blocks that change during the Picard iteration, i.e. everything depending on up
        self.a00_star = a00_conv + a_SUPG_00
        self.a01_star = a_SUPG_01
        self.a10_star = a_PSPG_10
        self.a11_star = a_PSPG_11

        # Compile them once. Calling form() inside the Picard loop is not a recompilation (FFCx caches on disk) but it re-hashes the ufl signature and re-imports the module on every call, which is milliseconds times four blocks times every iteration
        self._f_a00_star = form(self.a00_star)
        self._f_a01_star = form(self.a01_star)
        self._f_a10_star = form(self.a10_star)
        self._f_a11_star = form(self.a11_star)

        # Per-block boundary-condition lists, grouped by function space once here rather than
        # on every nonlinear iteration
        self._bcs_lift = bcs_by_block(extract_function_spaces(self.a), self.bcs)
        self._bcs_set = bcs_by_block(extract_function_spaces(self.L), self.bcs)

        return a00, a01, a10

    # Re-assemble the iteration-dependent blocks and the right-hand side
    def _assemble_system(self):
        with _EV_ASSEMBLE_B:
            # Assemble right-hand side vector
            b = assemble_vector(self.L, kind="nest")

            # Modify ('lift') the RHS for the Dirichlet boundary conditions
            apply_lifting(b, self.a, bcs=self._bcs_lift)

            # Sum contributions for entries shared across parallel processes
            for b_sub in b.getNestSubVecs():
                b_sub.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

            # Set the Dirichlet values in the RHS vector
            set_bc(b, self._bcs_set)

        with _EV_ASSEMBLE_A:
            # Re-assemble convection and stabilization, then add the stored constant blocks back
            for A_star, f_star, A_const in ((self.A00_star, self._f_a00_star, self.A00),
                                            (self.A01_star, self._f_a01_star, self.A01),
                                            (self.A10_star, self._f_a10_star, self.A10)):
                A_star.zeroEntries()
                assemble_matrix(A_star, f_star, bcs=self.bcs, diag=0.0)
                A_star.assemble()
                A_star.axpy(1.0, A_const, structure=_SUBSET_PATTERN)

            # The (1,1) block has no constant counterpart, but it is iterate-dependent through
            # tau_M and must be refreshed too
            self.A11_star.zeroEntries()
            assemble_matrix(self.A11_star, self._f_a11_star, bcs=self.bcs, diag=1.0)
            self.A11_star.assemble()

            self.A.assemble()
        return b

    # Picard-iterate to convergence at the current time level
    def _oseen_step(self, n_time, verbose=True):
        """
        Does not touch the inlet velocity or the previous time levels; the caller owns those, so that a Newton step can fall back to this one and redo the same time step.
        """
        pars = self.parameters
        max_nl_iter = pars.Solver.MaxNonlinearIterations
        nl_tol = pars.Solver.NonlinearTolerance
        rank = self.mesh.comm.rank

        nl_error = np.inf
        wk_error = np.inf if self.outlet_model is not None else 0.0
        it = 0
        converged = False
        lin_its, lin_its_max, reason = 0, 0, 0

        # Nonlinear iterations
        while it < max_nl_iter:
            if verbose and rank == 0:
                print(f"  [Picard] Nonlinear iteration {it:d}", flush=True)

            # Impose the outlet pressures the 0D model gives for the current iterate
            if self.outlet_model is not None:
                with _EV_WINDKESSEL:
                    self.outlet_model.impose(self.outlet_model.flow_rates(self.up),
                                             verbose=verbose, comm=self.mesh.comm)

            # Assemble and solve
            b = self._assemble_system()
            self.ksp.setOperators(self.A)
            self.ksp.setInitialGuessNonzero(self._guess_nonzero)
            if self._is_iterative:
                self.ksp.setTolerances(rtol=self._linear_rtol(it))
            self._prepare_preconditioner(first_of_step=(it == 0))
            self.ksp.solve(b, self.x)
            self.u_h.x.scatter_forward()
            self.p_h.x.scatter_forward()

            # Linear-solver diagnostics. getIterationNumber is collective and returns the same
            # value on every rank, so recording it needs no reduction
            its = self.ksp.getIterationNumber()
            lin_its += its
            lin_its_max = max(lin_its_max, its)
            self._record_linear_iterations(its)
            reason = self.ksp.getConvergedReason()
            if reason < 0:
                raise RuntimeError(
                    f"linear solve diverged at step {n_time} iteration {it} (KSPConvergedReason={reason})"
                )

            with _EV_CONVERGENCE:
                # Relative L2 error between successive Picard iterates
                nl_error = self._relative_velocity_change(self.up)

                # Mismatch between the 0D and 3D models at the outlets, measured against the
                # traction that was imposed; see Windkessel.coupling_residual
                if self.outlet_model is not None:
                    wk_error = self.outlet_model.coupling_residual(self.u_h)

            # Update the iterate
            self.up.x.array[:] = self.u_h.x.array

            if verbose and rank == 0:
                print(f"      Nonlinear error: {nl_error:.2e}"
                      f"  ({its:d} linear iterations)", flush=True)
                if self.windkessels is not None:
                    print(f"      Windkessel iteration error: {wk_error:.2e}", flush=True)

            b.destroy()
            it += 1
            if nl_error < nl_tol and wk_error < nl_tol:
                converged = True
                break

        return {
            "iterations": it,
            "nl_error": nl_error,
            "wk_error": wk_error,
            "converged": converged,
            "method": "picard",
            "linear_iterations": lin_its,
            "linear_iterations_max": lin_its_max,
            "ksp_reason": int(reason),
        }

    # Second-order extrapolation of the velocity to the new time level
    def _extrapolate_iterate(self, n_time):
        """
        Sets the starting iterate to `2 u^n - u^{n-1}`, a second-order extrapolation consistent with the BDF2 time derivative. This fixes only where the iteration starts, not the fixed point it converges to, so it changes the number of sweeps a step takes and nothing else.

        `nl_error` is the relative change from `up` to the solution, so the first sweep now reports the extrapolation error. Every term that reads `up` -- tau_M, the SUPG weight, the backflow traction -- sees the same field.

        At the first step the two history levels hold the same initial condition and the extrapolation reduces to `u^n`.
        """
        if n_time == 0:
            return
        self.up.x.array[:] = 2.0 * self.u0.x.array - self.u00.x.array
        self.up.x.scatter_forward()

    # Advance one time step by Picard-iterating to convergence
    def step(self, n_time, t, verbose=True, pre_step=None):
        """
        Returns a dict with the iteration count, the two error measures and whether the step converged. 'pre_step(problem, n_time, t)' is called first, which is where time-dependent data held by reference in the forms (a manufactured body force, say) must be advanced to the new time level.
        """
        if pre_step is not None:
            pre_step(self, n_time, t)

        self._update_inlet(n_time, t)
        self._extrapolate_iterate(n_time)
        info = self._oseen_step(n_time, verbose=verbose)
        self._advance_time_levels()
        return info


# Applies the 0D/3D coupling on top of the assembled blocks
class _WindkesselJacobian:
    """
    Shell operator adding the outlet coupling terms to the sparse Jacobian.

    Each cap pressure depends on every cap flow rate through the 0D model, `P_k = P_k(Q_1..Q_m)` with `Q_l = integral(u.n)` over cap l, so the Jacobian carries

        sum_kl M_kl a_k a_l^T,   (a_k)_i = integral(phi_i.n) over cap k,   M_kl = dP_k/dQ_l

    `M` comes from the 0D model itself, so it is diagonal for independent outlets and full for a network whose outlets share state.

    That block is dense on the cap dofs. Inserting it into the sparse matrix would cost a few million extra nonzeros on the aorta and gives BoomerAMG dense rows to work on. Applying it as a shell leaves the assembled nest untouched, so the nest still serves as the preconditioner while the Krylov method sees the exact operator.
    """

    def __init__(self, base, vectors, tangent, velocity_is):
        self.base = base
        self.vectors = vectors
        self.tangent = tangent
        # Index set of the velocity block within the nest. Going through it rather than
        # getNestSubVecs() is what makes this work for both layouts a MatNest may hand out:
        # sometimes a VecNest, sometimes a flat vector, depending on how the nest was built.
        self.velocity_is = velocity_is

    def _apply(self, x, y, tangent):
        self.base.mult(x, y)

        # The correction lives entirely in the velocity block
        xu = x.getSubVector(self.velocity_is)
        yu = y.getSubVector(self.velocity_is)
        try:
            dots = np.array([vec.dot(xu) for vec in self.vectors], dtype=float)
            for vec, coeff in zip(self.vectors, tangent @ dots, strict=True):
                yu.axpy(float(coeff), vec)
        finally:
            x.restoreSubVector(self.velocity_is, xu)
            y.restoreSubVector(self.velocity_is, yu)

    def mult(self, mat, x, y):
        self._apply(x, y, self.tangent)

    def multTranspose(self, mat, x, y):
        # The correction is symmetric only while M is; a network whose outlets share state
        # gives a non-symmetric tangent
        self._apply(x, y, self.tangent.T)


# BDF2 / Newton stabilized Navier-Stokes problem
class NewtonNSProblem(PicardNSProblem):
    """
    Solves the true nonlinear residual with its exact Jacobian, obtained from the residual by `ufl.derivative`.

    Compared with the Picard scheme the (0,0) block gains the term `rho*(grad(u).du, v)` -- the part Picard drops by freezing the convecting velocity -- and the backflow traction becomes implicit. Convergence is quadratic rather than linear, but only from a starting guess close enough to the solution.

    Two things supply that guess. The first time step runs Picard, since there is no previous solution to start from; every later step starts from the previous step's converged solution, which is `O(dt)` away. If a Newton step still fails to converge, the step is redone with Picard and the fact recorded in the step's info dictionary, so a long run cannot be lost to one bad step.

    `tau_M` is deliberately *not* differentiated. It is built on `up` rather than on the solution, so `ufl.derivative` steps over it. That is the standard frozen-parameter Newton, and it is not optional here: `tau_M` contains `sqrt(dot(u,u))`, whose derivative is singular at `u = 0`, and differentiating through it produces `nan` on a cold start.

    Extra argument beyond BaseProblem:

      windkessel_jacobian - include the exact rank-one outlet term in the Jacobian (default True). Setting it False lags the coupling instead, as the Picard scheme does, which gives a quasi-Newton method that is cheaper per iteration but loses the quadratic rate on Windkessel-dominated problems.
    """

    def __init__(self, windkessel_jacobian=True, **kwargs):
        self.windkessel_jacobian = windkessel_jacobian
        super().__init__(**kwargs)

    # Build boundary conditions, forms, matrices and the solver
    def set_problem(self):
        self._setup_measures()
        self._setup_time()
        self._setup_bcs()
        self._setup_state()

        # The Oseen forms are built too: they carry the first time step and any fallback
        a00, a01, a10 = self._build_oseen_forms()
        j00, j01, j10 = self._build_newton_forms()

        # Allocate against the union of both stencils, since both operators are assembled
        # into the same matrices
        self._allocate_blocks(a00, a01, a10, alloc=(j00, j01, j10))
        self._setup_windkessel_jacobian()
        self._setup_solver(operator=self._newton_operator)
        self._setup_convergence_forms()

        # Increment functions. Newton solves for a correction, so the solution vector
        # cannot be written into directly the way the Picard iteration does.
        # Named dx_vec, not dx: self.dx is the volume integration measure
        self.du_h, self.dp_h = Function(self.V), Function(self.Q)
        self.dx_vec = PETSc.Vec().createNest(
            [create_vector_wrap(self.du_h.x), create_vector_wrap(self.dp_h.x)]
        )

    # The nonlinear residual and its Jacobian
    def _build_newton_forms(self):
        """
        Sets self._F_form (the residual) and self._J_form (the Jacobian), and returns the three Jacobian blocks whose stencils the matrix allocation must cover.

        The residual is written in terms of the solution functions u_h and p_h rather than trial functions, which is what lets ufl.derivative produce the Jacobian.
        """
        mesh = self.mesh
        dx, ds = self.dx, self.ds
        dt, rho, mu, beta = self.dt, self.rho, self.mu, self.beta
        tM, tC = self.tau_M, self.tau_C
        n = FacetNormal(mesh)
        u_h, p_h = self.u_h, self.p_h

        v, q = TestFunction(self.V), TestFunction(self.Q)
        du, dp = TrialFunction(self.V), TrialFunction(self.Q)

        # Strong residual of the momentum equation, now fully nonlinear: the convecting
        # velocity is the solution itself, not a frozen iterate
        R = rho / dt * (BDF2_THETA * u_h - 2.0 * self.u0 + 0.5 * self.u00) + rho * grad(u_h) * u_h
        if self.f is not None:
            R = R - self.f

        # Momentum row. Signs follow from the Picard system a*u = L: the residual is
        # a*u - L, so every term that appears on the right-hand side there enters here
        # with the opposite sign.
        F0 = (
            rho / dt * inner(BDF2_THETA * u_h - 2.0 * self.u0 + 0.5 * self.u00, v) * dx
            + rho * inner(grad(u_h) * u_h, v) * dx
            + 2 * mu * inner(eps(u_h), eps(v)) * dx
            - p_h * div(v) * dx
        )
        if self.f is not None:
            F0 -= inner(self.f, v) * dx

        # Backflow traction, implicit here rather than lagged. ufl.derivative handles the
        # Abs() it contains; the result is not smooth at u.n = 0, which can slow Newton
        # during flow reversal but does not stop it.
        stab_force = backflow_stab(u_h, n, rho, beta)
        for oid in self.outlet_ids:
            F0 -= dot(stab_force, v) * ds(oid)

        # Windkessel outlet traction. P_out is a Constant, so ufl.derivative sees no
        # dependence on u_h; the exact coupling is added separately as a rank-one term.
        if self.windkessels is not None:
            for wksl in self.windkessels:
                F0 += dot(wksl.P_out * n, v) * ds(wksl.cap_id)

        # Stabilization. The weight grad(v)*up and tau_M stay frozen on `up`.
        F0 += tM * inner(grad(v) * self.up, R) * dx
        if self.supg_viscous:
            F0 += tM * mu * inner(nabla_grad(grad(v) * self.up), nabla_grad(u_h)) * dx
        F0 += tM * inner(grad(p_h), grad(v) * self.up) * dx
        F0 += tC * rho * div(u_h) * div(v) * dx

        # Continuity row, weighted by -grad(q) for the PSPG part
        F1 = (
            -q * div(u_h) * dx
            + tM * inner(-grad(q), R) * dx
            + tM * inner(-grad(q), grad(p_h)) * dx
        )

        # Jacobian, block by block
        j00 = derivative(F0, u_h, du)
        j01 = derivative(F0, p_h, dp)
        j10 = derivative(F1, u_h, du)
        j11 = derivative(F1, p_h, dp)

        self._F_form = form([F0, F1])
        self._J_form = form([[j00, j01], [j10, j11]])
        self._f_j00, self._f_j01 = form(j00), form(j01)
        self._f_j10, self._f_j11 = form(j10), form(j11)

        # Per-block boundary-condition lists, cached for the same reason as the Oseen ones
        self._bcs_lift_newton = bcs_by_block(extract_function_spaces(self._J_form), self.bcs)
        self._bcs_set_newton = bcs_by_block(extract_function_spaces(self._F_form), self.bcs)
        return j00, j01, j10

    # Cap vectors and coupling tangent for the exact outlet coupling
    def _setup_windkessel_jacobian(self):
        self._wk_vectors = []
        self._wk_tangent = np.zeros((0, 0))
        self._newton_operator = None

        if not self.windkessels or not self.windkessel_jacobian:
            return

        v = TestFunction(self.V)
        n = FacetNormal(self.mesh)
        n_owned = self.V.dofmap.index_map.size_local * self.V.dofmap.index_map_bs

        for wksl in self.windkessels:
            # a_k, the vector of integrals of each basis function against the cap normal, in
            # the velocity space layout. The shell applies it to the velocity sub-block, so it
            # needs no padding over the pressure dofs.
            cap = assemble_vector(form(dot(v, n) * self.ds(wksl.cap_id)))
            cap.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

            # Zero the constrained dofs. Every outlet rim meets the no-slip wall, and a
            # constrained row of the Jacobian must stay a unit row or the increment leaks
            # through the wall there.
            constrained = [
                bc._cpp_object.dof_indices()[0]
                for bc in self.bcs
                if bc.function_space == self.V._cpp_object
            ]
            if constrained:
                idx = np.unique(np.hstack(constrained))
                idx = idx[idx < n_owned]
                cap.getArray()[idx] = 0.0
            cap.assemble()

            self._wk_vectors.append(cap)

        # The tangent is rebuilt from the model at every nonlinear iteration; this is the
        # array the shell holds a reference to, so it is filled in place afterwards
        self._wk_tangent = np.zeros((len(self._wk_vectors), len(self._wk_vectors)))

        velocity_is = self.A.getNestISs()[0][0]
        ctx = _WindkesselJacobian(self.A, self._wk_vectors, self._wk_tangent, velocity_is)
        self._newton_operator = PETSc.Mat().createPython(
            self.A.getSizes(), ctx, comm=self.mesh.comm
        )
        self._newton_operator.setUp()

    # Rebuild the coupling tangent from the 0D model at the current cap flow rates
    def _refresh_tangent(self, flow_rates):
        """
        Fills M_kl = dP_k/dQ_l in place, so the shell operator keeps its reference.

        The tangent is obtained by perturbing the model rather than by differentiating a formula, which is what makes the coupling independent of what the 0D model is. See boundaries.numerical_tangent.
        """
        if self._wk_tangent.size == 0:
            return
        self._wk_tangent[:, :] = numerical_tangent(
            self.outlet_model, flow_rates, epsilon=self._tangent_epsilon
        )

    # Assemble the residual and the Jacobian at the current iterate
    def _assemble_newton_system(self):
        """
        Returns the right-hand side for `J dx = b`, from which the update is `u -= dx`.

        The boundary conditions are applied in increment form: `set_bc` with `x0` and `alpha=-1` puts `u_h - g` in the constrained entries, so the correction drives the solution onto the Dirichlet data. The inhomogeneous inlet profile and the pressure pin are handled the same way, with no separate homogeneous conditions.
        """
        # self.x is the nest wrapping u_h and p_h, i.e. the current iterate. The lifting
        # and boundary helpers take the nest itself and split it internally.
        with _EV_ASSEMBLE_B:
            b = assemble_vector(self._F_form, kind="nest")
            apply_lifting(b, self._J_form, bcs=self._bcs_lift_newton, x0=self.x, alpha=-1)
            for b_sub in b.getNestSubVecs():
                b_sub.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            set_bc(b, self._bcs_set_newton, x0=self.x, alpha=-1)

        # Jacobian. Unlike the Picard path there is no constant-plus-axpy split: every
        # block depends on the iterate, so each is assembled whole.
        with _EV_ASSEMBLE_A:
            for A_blk, f_blk, square in (
                (self.A00_star, self._f_j00, True),
                (self.A01_star, self._f_j01, False),
                (self.A10_star, self._f_j10, False),
                (self.A11_star, self._f_j11, True),
            ):
                A_blk.zeroEntries()
                if square:
                    assemble_matrix(A_blk, f_blk, bcs=self.bcs, diag=1.0)
                else:
                    assemble_matrix(A_blk, f_blk, bcs=self.bcs)
                A_blk.assemble()

            self.A.assemble()
        return b

    # Newton-iterate to convergence at the current time level
    def _newton_step(self, n_time, verbose=True):
        pars = self.parameters
        max_nl_iter = pars.Solver.MaxNonlinearIterations
        nl_tol = pars.Solver.NonlinearTolerance
        rank = self.mesh.comm.rank

        nl_error = np.inf
        wk_error = np.inf if self.outlet_model is not None else 0.0
        res_norm = np.inf
        residuals = []
        it = 0
        converged = False
        lin_its, lin_its_max, reason = 0, 0, 0

        while it < max_nl_iter:
            if verbose and rank == 0:
                print(f"  [Newton] Nonlinear iteration {it:d}", flush=True)

            # The stabilization is frozen on `up`, so track the current iterate with it
            self.up.x.array[:] = self.u_h.x.array

            # Impose the outlet pressures for the current iterate and rebuild the coupling
            # tangent from the same flow rates
            if self.outlet_model is not None:
                with _EV_WINDKESSEL:
                    flow_rates = self.outlet_model.flow_rates(self.u_h)
                    self.outlet_model.impose(flow_rates, verbose=verbose, comm=self.mesh.comm)
                    self._refresh_tangent(flow_rates)

            b = self._assemble_newton_system()
            res_norm = b.norm()
            residuals.append(res_norm)

            operator = self._newton_operator if self._newton_operator is not None else self.A
            self.ksp.setOperators(operator, self.A)

            # Newton solves for a correction, so the guess is zero. The iterative path
            # enables a nonzero guess for the Picard iteration and it is switched off here.
            self.ksp.setInitialGuessNonzero(False)
            self._prepare_preconditioner(first_of_step=(it == 0))
            self.ksp.solve(b, self.dx_vec)

            its = self.ksp.getIterationNumber()
            lin_its += its
            lin_its_max = max(lin_its_max, its)
            self._record_linear_iterations(its)
            reason = self.ksp.getConvergedReason()
            if reason < 0:
                b.destroy()
                raise RuntimeError(
                    f"linear solve diverged at step {n_time} iteration {it} (KSPConvergedReason={reason})"
                )

            # Newton update: the shifted right-hand side above means the correction is
            # subtracted, not added
            self.u_h.x.array[:] -= self.du_h.x.array
            self.p_h.x.array[:] -= self.dp_h.x.array
            self.u_h.x.scatter_forward()
            self.p_h.x.scatter_forward()

            with _EV_CONVERGENCE:
                # Relative size of the correction, the same measure the Picard path reports
                nl_error = self._relative_velocity_change(self.up)

                if self.outlet_model is not None:
                    wk_error = self.outlet_model.coupling_residual(self.u_h)

            if verbose and rank == 0:
                print(f"      Nonlinear error: {nl_error:.2e}  residual: {res_norm:.2e}", flush=True)

            b.destroy()
            it += 1
            if not np.isfinite(nl_error) or not np.isfinite(res_norm):
                break
            if nl_error < nl_tol and wk_error < nl_tol:
                converged = True
                break

        return {
            "iterations": it,
            "nl_error": nl_error,
            "wk_error": wk_error,
            "residual": res_norm,
            "residuals": residuals,
            "converged": converged,
            "method": "newton",
            "linear_iterations": lin_its,
            "linear_iterations_max": lin_its_max,
            "ksp_reason": int(reason),
        }

    # Advance one time step
    def step(self, n_time, t, verbose=True, pre_step=None):
        """
        Picard on the first time level, Newton afterwards, falling back to Picard for any step Newton cannot converge.

        The previous time levels are rolled forward only once a step has actually converged, so a fallback genuinely redoes the same step rather than advancing on a failed one.
        """
        if pre_step is not None:
            pre_step(self, n_time, t)

        self._update_inlet(n_time, t)

        if n_time == 0:
            # No previous solution to start Newton from
            info = self._oseen_step(n_time, verbose=verbose)
        else:
            info = self._newton_step(n_time, verbose=verbose)
            if not info["converged"]:
                if verbose and self.mesh.comm.rank == 0:
                    print(
                        f"  [Newton] did not converge in {info['iterations']} iterations; "
                        "redoing the step with Picard",
                        flush=True,
                    )
                # Restart from the last converged state rather than from the failed iterate
                self.u_h.x.array[:] = self.u0.x.array
                self.u_h.x.scatter_forward()
                self.up.x.array[:] = self.u0.x.array
                newton_info = info
                info = self._oseen_step(n_time, verbose=verbose)
                info["newton_failed"] = True
                info["newton_iterations"] = newton_info["iterations"]

        info.setdefault("newton_failed", False)
        self._advance_time_levels()
        return info
