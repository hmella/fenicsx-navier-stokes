"""
Monolithic stabilized incompressible Navier-Stokes solver.

Momentum and mass are solved together as one saddle-point system per nonlinear iteration (no projection or fractional-step splitting):

    rho*(3*u^{n+1} - 4*u^n + u^{n-1})/(2*dt) + rho*(u*.grad)u^{n+1}
      - div(2*mu*eps(u^{n+1})) + grad(p^{n+1}) = f,      div(u^{n+1}) = 0

Time stepping is BDF2 and the convection is linearized about the previous Picard iterate u* (an Oseen problem). Equal-order P1-P1 pairs are stabilized with SUPG/PSPG plus a grad-div term; P2-P1 is also available and is stabilized identically, the terms being consistent.

The 2x2 block system is a PETSc MatNest. The blocks that do not change during the Picard iteration are assembled once and added back with axpy each iteration; only the convection and stabilization blocks are re-assembled.
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
from ufl import FacetNormal, Measure, TestFunction, TrialFunction, div, dot, grad, inner, nabla_grad

from .boundaries import backflow_stab, inlet_lap_paraboloid
from .constitutive import eps
from .helpers import inflow_profile
from .stabilization import tau_C, tau_M

# BDF2 coefficient of the new time level, i.e. 3/2
BDF2_THETA = 1.5


# Configuration shared by every problem
class BaseProblem:
    """
    Holds the objects a problem is built from: the parameter file, the mesh and its tags, the outlet models, the element pair and the output flags.
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
    ):
        self.parameters = parameters
        self.mesh = mesh
        self.XDMF = XDMF
        self.HDF5 = HDF5
        self.domains = domains
        self.boundaries = boundaries
        self.windkessels = windkessels
        self.element = element
        self.ds_quad_deg = ds_quad_deg
        self.dx_quad_deg = dx_quad_deg
        self.inlet_plateau_lam = inlet_plateau_lam


# BDF2 / Picard stabilized Navier-Stokes problem
class PicardNSProblem(BaseProblem):
    """
    Extra arguments beyond BaseProblem:

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
        **kwargs,
    ):
        super().__init__(**kwargs)
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

    # Build boundary conditions, forms, matrices and the solver
    def set_problem(self):
        pars = self.parameters
        mesh = self.mesh

        # Define measures for integrals
        ds_deg = getattr(pars.Problem, "SurfaceQuadratureDegree", self.ds_quad_deg)
        dx_deg = getattr(pars.Problem, "VolumeQuadratureDegree", self.dx_quad_deg)
        ds = Measure("ds", domain=mesh, subdomain_data=self.boundaries,
                     metadata={"quadrature_degree": ds_deg})
        dx = Measure("dx", domain=mesh, subdomain_data=self.domains,
                     metadata={"quadrature_degree": dx_deg})
        self.ds, self.dx = ds, dx

        # ---------------------------------------------------------------------------------
        # Time stepping data: taken from the arguments, else from the inflow waveform file
        # ---------------------------------------------------------------------------------
        self.inflow = None
        profile_file = getattr(pars.Problem, "VelocityProfileFile", None)
        if profile_file is not None:
            self.inflow = inflow_profile(
                profile_file, scale_factor=pars.Problem.VelocityProfileScale
            )

        if self._dt_arg is not None:
            dt_value = float(self._dt_arg)
        elif self.inflow is not None:
            dt_value = float(self.inflow[1, 0] - self.inflow[0, 0])
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

        # ---------------------------------------------------------------------------------
        # Find inlet, outlets, and wall dofs
        # ---------------------------------------------------------------------------------
        fdim = mesh.topology.dim - 1

        # WallID may be a single tag or a list: the DFG cylinder benchmark needs the channel walls and the obstacle tagged separately (so drag can be integrated over the obstacle alone) while both carry no slip
        wall_id = pars.Geometry.WallID
        wall_ids = [wall_id] if np.isscalar(wall_id) else list(wall_id)
        inlet_id = pars.Geometry.InletID
        outlet_ids = list(pars.Geometry.OutletIDs)
        self.wall_ids, self.outlet_ids = wall_ids, outlet_ids

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

        # ---------------------------------------------------------------------------------
        # Boundary conditions
        # ---------------------------------------------------------------------------------
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

        # Trial and test functions
        u, v = TrialFunction(self.V), TestFunction(self.V)
        p, q = TrialFunction(self.Q), TestFunction(self.Q)

        # Initial conditions and the function holding the Picard iterate
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
        dt = Constant(mesh, PETSc.ScalarType(dt_value))
        rho = Constant(mesh, PETSc.ScalarType(pars.Problem.Density))
        mu = Constant(mesh, PETSc.ScalarType(pars.Problem.Viscosity))
        self.dt, self.rho, self.mu = dt, rho, mu
        self.dt_value = dt_value

        # Normal vector to facets and backflow stabilization strength
        n = FacetNormal(mesh)
        beta = Constant(mesh, PETSc.ScalarType(getattr(pars.Problem, "BackflowBeta", 0.0)))
        self.beta = beta

        # ---------------------------------------------------------------------------------
        # Variational formulation (Galerkin part)
        # ---------------------------------------------------------------------------------
        a00_conv = rho * inner(grad(u) * self.up, v) * dx
        a00 = rho / dt * inner(BDF2_THETA * u, v) * dx + 2 * mu * inner(eps(u), eps(v)) * dx
        a01 = -p * div(v) * dx
        a10 = -q * div(u) * dx
        L0 = rho / dt * inner(2 * self.u0 - 0.5 * self.u00, v) * dx
        L1 = Constant(mesh, PETSc.ScalarType(0.0)) * q * dx

        # Body force
        if self.f is not None:
            L0 += inner(self.f, v) * dx

        # Add backflow stabilization force at the outlets (explicit: built on the previous iterate, so it only ever reaches the right-hand side)
        stab_force = backflow_stab(self.up, n, rho, beta)
        for oid in outlet_ids:
            L0 += dot(stab_force, v) * ds(oid)

        # Windkessel outlet boundary condition, as a constant normal traction
        if self.windkessels is not None:
            for wksl in self.windkessels:
                L0 += -dot(wksl.P_out * n, v) * ds(wksl.cap_id)

        # ---------------------------------------------------------------------------------
        # SUPG/PSPG stabilization
        # ---------------------------------------------------------------------------------
        tM = tau_M(
            self.up, dt, rho, mu, mesh=mesh, sigma_BDF=self.sigma_BDF, degree=self.velocity_degree
        )
        tC = tau_C(rho, mu, coefficient=self.tau_C_coefficient)
        self.tau_M, self.tau_C = tM, tC

        # The two halves of the strong momentum residual: the terms in the unknown u, and the known data at the previous time levels. Both carry a SINGLE factor of rho -- they are two halves of the same residual, and a mismatch makes the stabilization inconsistent, i.e. it no longer vanishes on the exact solution. An earlier revision squared rho on the implicit half only, which is invisible at rho = 1 but not at the rho = 1.06 of the physical cases. The viscous part is omitted because it vanishes elementwise for P1 velocities
        res_lhs = rho * (BDF2_THETA / dt * u + grad(u) * self.up)
        res_rhs = rho / dt * (2.0 * self.u0 - 0.5 * self.u00)
        if self.f is not None:
            res_rhs = res_rhs + self.f

        # Momentum row: streamline weight (up.grad)v against the residual, plus grad-div
        a_SUPG_00 = tM * inner(grad(v) * self.up, res_lhs) * dx
        if self.supg_viscous:
            a_SUPG_00 += tM * mu * inner(nabla_grad(grad(v) * self.up), nabla_grad(u)) * dx
        a_SUPG_00 += tC * rho * div(u) * div(v) * dx

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

        # ---------------------------------------------------------------------------------
        # Assemble the matrices
        # ---------------------------------------------------------------------------------
        # The *_star matrices are allocated against the COMBINED sparsity pattern, so that the later axpy with the constant block never needs a new nonzero location
        self.A00_star = create_matrix(form(a00 + self.a00_star))
        self.A00_star.setOption(PETSc.Mat.Option.IGNORE_ZERO_ENTRIES, False)
        assemble_matrix(self.A00_star, self._f_a00_star, bcs=self.bcs, diag=0.0)
        self.A00_star.assemble()
        self.A00_star.setOption(PETSc.Mat.Option.NEW_NONZERO_LOCATIONS, False)

        self.A01_star = create_matrix(form(a01 + self.a01_star))
        self.A01_star.setOption(PETSc.Mat.Option.IGNORE_ZERO_ENTRIES, False)
        assemble_matrix(self.A01_star, self._f_a01_star, bcs=self.bcs, diag=0.0)
        self.A01_star.assemble()
        self.A01_star.setOption(PETSc.Mat.Option.NEW_NONZERO_LOCATIONS, False)

        self.A10_star = create_matrix(form(a10 + self.a10_star))
        self.A10_star.setOption(PETSc.Mat.Option.IGNORE_ZERO_ENTRIES, False)
        assemble_matrix(self.A10_star, self._f_a10_star, bcs=self.bcs, diag=0.0)
        self.A10_star.assemble()
        self.A10_star.setOption(PETSc.Mat.Option.NEW_NONZERO_LOCATIONS, False)

        # The (1,1) block is the PSPG pressure Laplacian. It carries no convective velocity, but it is still weighted by tau_M, which is built on `up` and therefore changes with every Picard iterate -- so it is re-assembled in _assemble_system alongside the other three. Passing the full bc list is deliberate: both spaces here are the pressure space, so VELOCITY conditions match neither and dolfinx discards them (which is why passing them used to be a silent no-op), but a PRESSURE condition in the list is applied, and that is what makes the block non-singular in the fully-Dirichlet case
        self.A11_star = create_matrix(form(self.a11_star))
        self.A11_star.setOption(PETSc.Mat.Option.SYMMETRIC, True)
        self.A11_star.setOption(PETSc.Mat.Option.SYMMETRY_ETERNAL, True)
        self.A11_star.setOption(PETSc.Mat.Option.IGNORE_ZERO_ENTRIES, True)
        assemble_matrix(self.A11_star, self._f_a11_star, bcs=self.bcs, diag=1.0)
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
        self.A00_star.axpy(1.0, self.A00)
        self.A01_star.axpy(1.0, self.A01)
        self.A10_star.axpy(1.0, self.A10)

        # Create nested matrix
        self.A = create_matrix(self.a, kind="nest")
        self.A.setOption(PETSc.Mat.Option.IGNORE_ZERO_ENTRIES, True)
        self.A.createNest([[self.A00_star, self.A01_star], [self.A10_star, self.A11_star]])
        self.A.assemble()
        self.A.setOption(PETSc.Mat.Option.NEW_NONZERO_LOCATIONS, False)

        # ---------------------------------------------------------------------------------
        # Create and configure solver
        # ---------------------------------------------------------------------------------
        self.ksp = PETSc.KSP().create(mesh.comm)
        self.ksp.setOperators(self.A)
        opts = PETSc.Options()

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

            # Define the velocity and pressure blocks of the preconditioner from the index sets of the nested matrix
            nested_IS = self.A.getNestISs()
            self.ksp.getPC().setFieldSplitIS(("u", nested_IS[0][0]), ("p", nested_IS[0][1]))

            opts["pc_fieldsplit_type"] = "schur"
            opts["pc_fieldsplit_schur_fact_type"] = "lower"
            opts["pc_fieldsplit_schur_precondition"] = "selfp"
            for blk in ("u", "p"):
                opts[f"fieldsplit_{blk}_ksp_type"] = "preonly"
                opts[f"fieldsplit_{blk}_pc_type"] = "hypre"
                opts[f"fieldsplit_{blk}_pc_hypre_type"] = "boomeramg"
                opts[f"fieldsplit_{blk}_pc_hypre_boomeramg_relax_weight_all"] = 0.0

        elif pars.Solver.Kind == "direct":
            # Direct LU with MUMPS: robust, and the right choice for the small 2D cases
            self.ksp.setType("preonly")
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

        # Forms used by the Picard convergence test, also compiled once
        self._diff = Function(self.V)
        self._err_num = form(inner(self._diff, self._diff) * dx)
        self._err_den = form(inner(self.u_h, self.u_h) * dx)

    # Re-assemble the iteration-dependent blocks and the right-hand side
    def _assemble_system(self):
        # Assemble right-hand side vector
        b = assemble_vector(self.L, kind="nest")

        # Modify ('lift') the RHS for the Dirichlet boundary conditions
        bcs1 = bcs_by_block(extract_function_spaces(self.a), self.bcs)
        apply_lifting(b, self.a, bcs=bcs1)

        # Sum contributions for entries shared across parallel processes
        for b_sub in b.getNestSubVecs():
            b_sub.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

        # Set the Dirichlet values in the RHS vector
        bcs0 = bcs_by_block(extract_function_spaces(self.L), self.bcs)
        set_bc(b, bcs0)

        # Re-assemble convection and stabilization, then add the stored constant blocks back
        for A_star, f_star, A_const in ((self.A00_star, self._f_a00_star, self.A00),
                                        (self.A01_star, self._f_a01_star, self.A01),
                                        (self.A10_star, self._f_a10_star, self.A10)):
            A_star.zeroEntries()
            assemble_matrix(A_star, f_star, bcs=self.bcs, diag=0.0)
            A_star.assemble()
            A_star.axpy(1.0, A_const)

        # The (1,1) block has no constant counterpart to add back, but it is iterate-dependent through tau_M and must be refreshed too. Leaving it at its cold-start value, tau_M(up = 0), makes the continuity row weight a different strong residual from the momentum row, so the two stop being consistent parts of one stabilized system
        self.A11_star.zeroEntries()
        assemble_matrix(self.A11_star, self._f_a11_star, bcs=self.bcs, diag=1.0)
        self.A11_star.assemble()

        self.A.assemble()
        return b

    # Advance one time step by Picard-iterating to convergence
    def step(self, n_time, t, verbose=True, pre_step=None):
        """
        Returns a dict with the iteration count, the two error measures and whether the step converged. 'pre_step(problem, n_time, t)' is called first, which is where time-dependent data held by reference in the forms (a manufactured body force, say) must be advanced to the new time level.
        """
        if pre_step is not None:
            pre_step(self, n_time, t)

        pars = self.parameters
        max_nl_iter = pars.Solver.MaxNonlinearIterations
        nl_tol = pars.Solver.NonlinearTolerance
        rank = self.mesh.comm.rank

        # Update inlet velocity profile
        self.u_inlet_scale.value = self.inlet_scale_fn(n_time, t)
        self.u_inlet.interpolate(self.u_inlet_expr)

        nl_error = np.inf
        wk_error = np.inf if self.windkessels else 0.0
        it = 0
        converged = False

        # Nonlinear iterations
        while it < max_nl_iter:
            if verbose and rank == 0:
                print(f"  [Picard] Nonlinear iteration {it:d}", flush=True)

            # Update the windkessels from the current iterate
            if self.windkessels is not None:
                for wksl in self.windkessels:
                    wksl.update(self.up, verbose=verbose)

            # Assemble and solve
            b = self._assemble_system()
            self.ksp.setOperators(self.A)
            self.ksp.solve(b, self.x)
            self.u_h.x.scatter_forward()
            self.p_h.x.scatter_forward()

            reason = self.ksp.getConvergedReason()
            if reason < 0:
                raise RuntimeError(
                    f"linear solve diverged at step {n_time} iteration {it} (KSPConvergedReason={reason})"
                )

            # Relative L2 error between successive Picard iterates
            self._diff.x.array[:] = self.up.x.array - self.u_h.x.array
            num = self.mesh.comm.allreduce(assemble_scalar(self._err_num), op=MPI.SUM)
            den = self.mesh.comm.allreduce(assemble_scalar(self._err_den), op=MPI.SUM)
            nl_error = float(np.sqrt(num / den)) if den > 0.0 else 0.0

            # Windkessel convergence measure
            if self.windkessels is not None:
                wk_error = max(w.residual() for w in self.windkessels)

            # Update the iterate
            self.up.x.array[:] = self.u_h.x.array
            if self.windkessels is not None:
                for wksl in self.windkessels:
                    wksl.Pd_nl_prev = wksl.Pd_nl

            if verbose and rank == 0:
                print(f"      Nonlinear error: {nl_error:.2e}", flush=True)
                if self.windkessels is not None:
                    print(f"      Windkessel iteration error: {wk_error:.2e}", flush=True)

            b.destroy()
            it += 1
            if nl_error < nl_tol and wk_error < nl_tol:
                converged = True
                break

        # Update previous time step
        if self.windkessels is not None:
            for wksl in self.windkessels:
                wksl.advance()
        self.u00.x.array[:] = self.u0.x.array
        self.u0.x.array[:] = self.u_h.x.array

        return {
            "iterations": it,
            "nl_error": nl_error,
            "wk_error": wk_error,
            "converged": converged,
        }

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
                    print(
                        "  Time step {} in {:.1f} seconds ({} Picard iterations)".format(
                            status, info["wall_time"], info["iterations"]
                        ),
                        flush=True,
                    )

                if callback is not None:
                    callback(self, n_time, t, info)

                # Write to file
                if (
                    file_u is not None
                    and n_time >= store_after
                    and (n_time - store_after) % store_every == 0
                ):
                    file_u.write_function(self.u_h, t)
                    file_u.flush()
                    file_p.write_function(self.p_h, t)
                    file_p.flush()
        finally:
            # Close files
            if file_u is not None:
                file_u.close()
            if file_p is not None:
                file_p.close()

        return history
