"""
Boundary conditions: facet normals, inlet profiles and outlet models.

The inlet profile solves an auxiliary problem on the domain rather than assuming a circular cap, so it works on the irregular caps produced by patient-specific segmentation. Outlets carry a three-element Windkessel model and, optionally, a backflow traction.
"""

import dolfinx.cpp as cpp
import numpy as np
from basix.ufl import element
from dolfinx import default_scalar_type
from dolfinx.fem import (
    Constant,
    Expression,
    Function,
    assemble_scalar,
    create_sparsity_pattern,
    dirichletbc,
    form,
    functionspace,
    locate_dofs_topological,
)
from dolfinx.fem.petsc import apply_lifting, assemble_matrix, assemble_vector, set_bc
from dolfinx.la.petsc import create_vector_wrap
from mpi4py import MPI
from petsc4py import PETSc
from ufl import (
    FacetNormal,
    Identity,
    Measure,
    SpatialCoordinate,
    TestFunction,
    TrialFunction,
    as_vector,
    conditional,
    dot,
    dx,
    grad,
    gt,
    inner,
    outer,
    sqrt,
)
from ufl.algebra import Abs


# Project a vector onto the plane of a facet
def tangential_projection(u, n):
    """
    Remove the component of 'u' along 'n'. See for instance https://link.springer.com/content/pdf/10.1023/A:1022235512626.pdf
    """
    return (Identity(u.ufl_shape[0]) - outer(n, n)) * u


# Discrete facet normal (or tangent)
def facet_vector_approximation(
    V,
    mt=None,
    mt_id="everywhere",
    tangent=False,
    interior=False,
    rtol=1.0e-8,
    jit_options=None,
    form_compiler_options=None,
):
    """
    Approximate the facet normal (or tangent) on a set of facets by projecting it into the function space 'V'. Based on a gist by @hherlyng, with a guard against division by zero in the normalisation.

    'V' is nominally meant to be discontinuous. It is used here with the continuous velocity space, which is fine only because the caps it is applied to are planar; on a rim where two tagged faces meet, a continuous space averages their normals.
    """
    jit_options = jit_options if jit_options is not None else {}
    form_compiler_options = form_compiler_options if form_compiler_options is not None else {}

    comm = V.mesh.comm
    n = FacetNormal(V.mesh)
    u, v = TrialFunction(V), TestFunction(V)

    # Measure over the facets of interest
    if interior:
        dS = (
            Measure("dS", domain=V.mesh)
            if mt is None
            else Measure("dS", domain=V.mesh, subdomain_data=mt, subdomain_id=mt_id)
        )
    else:
        ds = (
            Measure("ds", domain=V.mesh)
            if mt is None
            else Measure("ds", domain=V.mesh, subdomain_data=mt, subdomain_id=mt_id)
        )

    # Mass matrix on the facets, with the normal (or tangent) as the source term
    if tangent:
        if V.mesh.geometry.dim == 1:
            raise ValueError("Tangent not defined for 1D problem")
        elif V.mesh.geometry.dim == 2:
            # In 2D the tangent is the normal rotated by 90 degrees
            if interior:
                a = (inner(u("+"), v("+")) + inner(u("-"), v("-"))) * dS
                L = (
                    inner(as_vector([-n("+")[1], n("+")[0]]), v("+")) * dS
                    + inner(as_vector([-n("-")[1], n("-")[0]]), v("-")) * dS
                )
            else:
                a = inner(u, v) * ds
                L = inner(as_vector([-n[1], n[0]]), v) * ds
        else:
            # In 3D the tangent is not unique: project a fixed vector onto the facet plane
            c = Constant(V.mesh, (1.0, 1.0, 1.0))
            if interior:
                a = (inner(u("+"), v("+")) + inner(u("-"), v("-"))) * dS
                L = (
                    inner(tangential_projection(c, n("+")), v("+")) * dS
                    + inner(tangential_projection(c, n("-")), v("-")) * dS
                )
            else:
                a = inner(u, v) * ds
                L = inner(tangential_projection(c, n), v) * ds
    else:
        if interior:
            a = (inner(u("+"), v("+")) + inner(u("-"), v("-"))) * dS
            L = (inner(n("+"), v("+")) + inner(n("-"), v("-"))) * dS
        else:
            a = inner(u, v) * ds
            L = inner(n, v) * ds

    # The facet mass matrix is singular away from the tagged facets, so those dofs must be deactivated. Assembling the test function over the measure marks them: the entries that come out zero belong to dofs the measure does not touch
    ones = Constant(V.mesh, default_scalar_type((1,) * V.mesh.geometry.dim))
    if interior:
        local_val = form((dot(ones, v("+")) + dot(ones, v("-"))) * dS)
    else:
        local_val = form(dot(ones, TestFunction(V)) * ds)
    local_vec = assemble_vector(local_val)
    bdry_dofs_zero_val = np.flatnonzero(np.isclose(local_vec.array, 0))
    deac_blocks = np.unique(bdry_dofs_zero_val // V.dofmap.bs).astype(np.int32)

    # Build the sparsity pattern with a diagonal entry for the deactivated blocks, and pin them with a zero Dirichlet condition
    bilinear_form = form(a, jit_options=jit_options, form_compiler_options=form_compiler_options)
    pattern = create_sparsity_pattern(bilinear_form)
    pattern.insert_diagonal(deac_blocks)
    pattern.finalize()
    u_0 = Function(V)
    u_0.x.scatter_forward()
    bc_deac = dirichletbc(u_0, deac_blocks)

    # Assemble the matrix
    A = cpp.la.petsc.create_matrix(comm, pattern)
    A.zeroEntries()
    form_coeffs = cpp.fem.pack_coefficients(bilinear_form._cpp_object)
    form_consts = cpp.fem.pack_constants(bilinear_form._cpp_object)
    assemble_matrix(A, bilinear_form, constants=form_consts, coeffs=form_coeffs, bcs=[bc_deac])

    # Put ones on the diagonal of the deactivated blocks
    if bilinear_form.function_spaces[0] is bilinear_form.function_spaces[1]:
        A.assemblyBegin(PETSc.Mat.AssemblyType.FLUSH)
        A.assemblyEnd(PETSc.Mat.AssemblyType.FLUSH)
        cpp.fem.petsc.insert_diagonal(
            A=A, V=bilinear_form.function_spaces[0], bcs=[bc_deac._cpp_object], diagonal=1.0
        )
    A.assemble()

    # Assemble the right-hand side and apply the boundary condition
    linear_form = form(L, jit_options=jit_options, form_compiler_options=form_compiler_options)
    b = assemble_vector(linear_form)
    apply_lifting(b, [bilinear_form], [[bc_deac]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, [bc_deac])

    # Solve with conjugate gradients (the facet mass matrix is symmetric positive definite)
    solver = PETSc.KSP().create(MPI.COMM_WORLD)
    solver.setType("cg")
    solver.rtol = rtol
    solver.setOperators(A)
    nh = Function(V)
    solver.solve(b, create_vector_wrap(nh.x))
    nh.x.scatter_forward()

    # Normalize to unit length, guarding the dofs that were left at zero
    nh_norm = sqrt(inner(nh, nh))
    cond_norm = conditional(gt(nh_norm, 1e-10), nh_norm, 1.0)
    gdim = V.mesh.geometry.dim
    nh_norm_vec = as_vector([nh[i] / cond_norm for i in range(gdim)])

    n_out = Function(V)
    n_out.interpolate(Expression(nh_norm_vec, V.element.interpolation_points))
    return n_out


# Inlet paraboloid (analytic)
def inlet_paraboloid(V, st=None, st_id="everywhere", tol=1.0e-4):
    """
    Analytic paraboloid on the cap 'st_id', built from the radius of the equivalent circle. Kept for reference; inlet_lap_paraboloid() is what the examples use, since it does not assume the cap is round.
    """
    ds = Measure("ds", domain=V.mesh, subdomain_data=st)

    # Radius of a circle with the same area as the cap
    x = SpatialCoordinate(V.mesh)
    A = assemble_scalar(form(Constant(V.mesh, default_scalar_type(1.0)) * ds(st_id)))
    R_cap = np.sqrt(A / np.pi) + tol

    # Centroid of the cap
    coords = Function(V)
    coords.interpolate(Expression(x, V.element.interpolation_points))
    c0 = np.array(
        [
            V.mesh.comm.allreduce(
                assemble_scalar(form(Constant(V.mesh, 1.0) * x[i] * ds(st_id))), op=MPI.SUM
            )
            for i in range(len(x))
        ]
    )
    c1 = np.array(
        [
            V.mesh.comm.allreduce(
                assemble_scalar(form(Constant(V.mesh, 1.0) * ds(st_id))), op=MPI.SUM
            )
            for i in range(len(x))
        ]
    )
    c = c0 / c1

    # Paraboloid (R^2 - r^2)/R^2, clipped at zero outside the cap radius
    gdim = V.mesh.geometry.dim
    r = sum((x[i] - c[i]) ** 2 for i in range(gdim))
    R_cap = Constant(V.mesh, R_cap**2)
    p_ = (R_cap - r) / R_cap
    p = Function(V)
    p.interpolate(Expression(as_vector([p_] * gdim), V.element.interpolation_points))
    p.x.array[p.x.array < 0.0] = 0.0

    # Multiply by the discrete normal so the profile points across the cap
    inlet = st.find(st_id)
    nh = facet_vector_approximation(V, st, st_id)
    par = Function(V)
    for i in range(gdim):
        dofs = locate_dofs_topological(V.sub(i), V.mesh.topology.dim - 1, inlet)
        par.x.array[dofs] = p.x.array[dofs] * nh.x.array[dofs]

    return par


# Inlet profile from a screened Poisson problem
def inlet_lap_paraboloid(V, ft=None, cap_id="everywhere", wall_id="everywhere", plateau_lam=0.0):
    """
    Build the inlet velocity profile on the cap 'cap_id' by solving

        u - lam^2*laplacian(u) = 1,   u = 0 on the wall,

    normalising by the global maximum, restricting to the cap and multiplying by the discrete facet normal. Solving on the domain rather than fitting a shape means the profile follows whatever cap the segmentation produced.

    Note which way the limits run, which is what 'plateau_lam' refers to:
      - lam -> 0 gives a PLUG, with a boundary layer of width lam
      - lam -> infinity gives the Poiseuille PARABOLOID
    Passing plateau_lam=None solves the pure Laplace problem instead.

    On a circular cap the exact profile is
        p(r) = [1 - I0(r/lam)/I0(beta)] / [1 - 1/I0(beta)],   beta = R/lam,
    whose mean-to-peak ratio runs from 1/2 (paraboloid) to 1 (plug).
    """
    ds = Measure("ds", domain=V.mesh, subdomain_data=ft, subdomain_id=cap_id)

    # Scalar space for the auxiliary problem
    q_cg1 = element("Lagrange", V.mesh.topology.cell_name(), 1)
    Q = functionspace(V.mesh, q_cg1)
    u = TrialFunction(Q)
    v = TestFunction(Q)

    # Screened Poisson (or pure Laplace when no length scale is given)
    one = Constant(Q.mesh, PETSc.ScalarType(1))
    if plateau_lam is not None:
        lam = Constant(Q.mesh, PETSc.ScalarType(plateau_lam))
        a = u * v * dx + lam**2 * inner(grad(u), grad(v)) * dx
    else:
        a = inner(grad(u), grad(v)) * dx
    L = one * v * dx

    # No slip on the wall; the cap itself carries a natural (zero Neumann) condition, so on a cylinder the solution is independent of the axial coordinate
    zero = Constant(Q.mesh, PETSc.ScalarType(0))
    wall_dofs = locate_dofs_topological(Q, Q.mesh.topology.dim - 1, ft.find(wall_id))
    bc_wall = dirichletbc(zero, wall_dofs, Q)

    # Assemble and apply the boundary condition
    A = assemble_matrix(form(a), bcs=[bc_wall])
    A.assemble()
    b = assemble_vector(form(L))
    apply_lifting(b, [form(a)], [[bc_wall]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, [bc_wall])

    # Direct solve
    solver = PETSc.KSP().create(MPI.COMM_WORLD)
    solver.setType("preonly")
    solver.getPC().setType("lu")
    solver.getPC().setFactorSolverType("mumps")
    solver.setOperators(A)
    p = Function(Q)
    solver.solve(b, create_vector_wrap(p.x))
    p.x.scatter_forward()

    # Keep only the cap values and normalize the peak to one
    inlet = ft.find(cap_id)
    inlet_dofs = locate_dofs_topological(Q, Q.mesh.topology.dim - 1, inlet)
    inlet_values = p.x.array[inlet_dofs]
    p.x.array[:] = 0.0
    p.x.array[inlet_dofs] = inlet_values
    p.x.array[:] /= Q.mesh.comm.allreduce(p.x.array.max(), op=MPI.MAX)
    p.x.scatter_forward()

    # Report the cap area and the mean-to-peak shape factor
    volume = Q.mesh.comm.allreduce(assemble_scalar(form(p * ds)), op=MPI.SUM)
    area = Q.mesh.comm.allreduce(assemble_scalar(form(one * ds)), op=MPI.SUM)
    if Q.mesh.comm.rank == 0:
        print(f"[Boundary] Area of cap {cap_id:d}: {area:.4g}", flush=True)
        print(
            f"[Boundary] Normalized volume under the profile on cap {cap_id:d}: {volume:.4g} "
            f"(shape factor {volume / area:.4f})",
            flush=True,
        )

    # Multiply the scalar profile by the discrete normal to get a vector field on the cap
    gdim = V.mesh.geometry.dim
    p_vec = Function(V)
    p_vec.interpolate(Expression(as_vector([p] * gdim), V.element.interpolation_points))
    nh = facet_vector_approximation(V, ft, cap_id)
    par = Function(V)
    for i in range(gdim):
        dofs = locate_dofs_topological(V.sub(i), V.mesh.topology.dim - 1, inlet)
        par.x.array[dofs] = p_vec.x.array[dofs] * nh.x.array[dofs]

    return par


# Backflow stabilization force
def backflow_stab(u, n, rho, beta):
    """
    Traction 0.5*rho*beta*(u.n - |u.n|)*u, to be added to the momentum right-hand side as dot(s, v)*ds(outlet_id).

    The prefactor equals u.n where flow enters the domain and is exactly zero where it leaves, so the term is inactive during forward flow and dissipative during backflow. This is what keeps convection-dominated open outlets from diverging during flow reversal (Bertoglio & Caiazzo 2014; Moghadam et al. 2011).

    'u' is the previous Picard iterate, which makes the term explicit (right-hand side only).
    """
    un = dot(u, n)
    return 0.5 * rho * beta * (un - Abs(un)) * u


# Three-element (RCR) Windkessel outlet
class Windkessel:
    """
    Proximal resistance Rp in series with a parallel Rd/C pair:

        C*dPd/dt = Q - Pd/Rd,    P_out = Rp*Q + Pd

    The flow rate Q is computed over the outlet cap, the ODE is advanced across one time step with RK4, and P_out is imposed on the momentum equation as a constant normal traction -dot(P_out*n, v)*ds(cap_id).

    Because P_out enters only the right-hand side, the coupling is explicit within a single linear solve. It is nonetheless driven to consistency by calling update() at every Picard iteration, with residual() taking part in the nonlinear convergence test.

    Accuracy is limited by that coupling, not by the integrator: Q is frozen across the step, so the scheme is first order in dt no matter how large Niter is. Refining dt helps, refining Niter does not.
    """

    # Absolute pressure floor used to keep residual() finite at start-up
    P_ATOL = 1.0e-8

    def __init__(
        self,
        Q_out=1.0,
        Pd_prev=0.0,
        Rd=1.0,
        Rp=1.0,
        C=1.0,
        dt_sim=1e-2,
        facet_tags=None,
        cap_id=0,
        Niter=100,
        P_out=None,
    ):
        if Niter < 1:
            raise ValueError(f"Niter must be >= 1, got {Niter}")
        if C <= 0.0 or Rd <= 0.0:
            raise ValueError(f"Rd and C must be positive, got Rd={Rd}, C={C}")
        self.Q_out = Q_out
        self.Pd_prev = Pd_prev
        self.Rd = Rd
        self.Rp = Rp
        self.C = C
        self.dt_sim = dt_sim
        self.facet_tags = facet_tags
        self.cap_id = cap_id
        self.Niter = Niter
        self.P_out = P_out
        self.Pd_nl = Pd_prev  # distal pressure at the current nonlinear iterate
        self.Pd_nl_prev = Pd_prev  # ... and at the previous one, used by residual()
        # The outlet pressure last written into P_out, i.e. the one the momentum equation was
        # solved against; coupling_residual() measures against it
        self.P_out_imposed = Rp * Q_out + Pd_prev
        # Compiled flux forms, one per function they were built for; see flow_rate
        self._flow_forms = {}

    def ODE(self, t, Pd):
        """
        Right-hand side (Q - Pd/Rd)/C of the distal pressure equation.
        """
        return (1.0 / self.C) * (self.Q_out - Pd / self.Rd)

    def RK4(self, t_range, h=None):
        """
        Advance the distal pressure across 't_range' with classical RK4 and return P_out = Rp*Q + Pd. 'h' is ignored; the sub-step follows from Niter.

        The flow rate is frozen across the interval, so the equation is the linear, autonomous `y' = -lambda*(y - y_inf)` with `lambda = 1/(Rd*C)` and `y_inf = Rd*Q`. RK4 applied to it is the affine map

            y <- y_inf + A*(y - y_inf),   A = 1 - z + z^2/2 - z^3/6 + z^4/24,   z = lambda*step

        with A the RK4 stability polynomial, so n sub-steps multiply the offset by A^n and the integration is evaluated in closed form. The result matches sub-stepping to 2e-15 relative, including at Niter = 4, and carries the same fourth-order truncation error in Niter.
        """
        self.Pd_nl = self._distal_pressure(self.Q_out, t_range[1] - t_range[0])
        return self.Rp * self.Q_out + self.Pd_nl

    # The closed-form map, with no side effects
    def _distal_pressure(self, Q, span):
        """
        Distal pressure after integrating across 'span' with the flow rate held at 'Q', starting from the accepted previous time level Pd_prev. Leaves the model's state untouched.
        """
        n = int(self.Niter)
        step = span / n

        # RK4's stability polynomial minus one, by Horner in z so that no significance is lost
        # when z is tiny
        z = step / (self.Rd * self.C)
        Am1 = z * (-1.0 + z * (0.5 + z * (-1.0 / 6.0 + z / 24.0)))

        # A**n - 1, again without cancellation. The branch covers A outside (0, 1], which is
        # where the sub-step is large enough for RK4 to be unstable
        A = 1.0 + Am1
        powm1 = np.expm1(n * np.log1p(Am1)) if A > 0.0 else A**n - 1.0

        # y_n = y_inf + A^n (y_0 - y_inf), rearranged so the update is added to Pd_prev
        return float(self.Pd_prev + powm1 * (self.Pd_prev - self.Rd * Q))

    def flow_rate(self, u):
        """
        Flow rate through the cap. Positive means flow leaving the domain, since n is the outward normal.

        A compiled form is bound to the coefficients it was built from, so the cache is keyed on 'u'; the function is stored alongside its form so an id cannot be reused while the entry is live. In practice the dictionary holds at most two entries per outlet, the iterate and the solution.
        """
        mesh = u.function_space.mesh
        entry = self._flow_forms.get(id(u))
        if entry is None or entry[0] is not u:
            ds = Measure("ds", domain=mesh, subdomain_data=self.facet_tags)
            n = FacetNormal(mesh)
            entry = (u, form(dot(u, n) * ds(self.cap_id)))
            self._flow_forms[id(u)] = entry
        return mesh.comm.allreduce(assemble_scalar(entry[1]), op=MPI.SUM)

    def residual(self):
        """
        Relative change of the distal pressure between nonlinear iterates,
        |Pd_nl - Pd_nl_prev| / (P_ATOL + |Pd_nl|).

        The absolute floor keeps the result finite when Pd_nl is exactly zero, which it is on the first iteration of the first step whenever the initial velocity and Pd_init both vanish.
        """
        return abs(self.Pd_nl - self.Pd_nl_prev) / (self.P_ATOL + abs(self.Pd_nl))

    def coupling_residual(self, u):
        """
        Mismatch between the 0D and 3D models at the outlet, given the velocity 'u'.

        The momentum equation was solved against the fixed traction P_out_imposed, computed from the flow rate of the previous iterate. Feeding 'u' back through the outlet model gives the pressure it implies; the two are equal at the fixed point of the coupled system, so

            |Rp*Q(u) + Pd(Q(u)) - P_out_imposed| / (P_ATOL + |Rp*Q(u) + Pd(Q(u))|)

        is zero only there. This is the measure the solver's convergence test uses; residual() reports the change between successive impositions instead.

        The absolute floor keeps the result finite at a cold start with Pd_init = 0 and zero flow.
        """
        q = self.flow_rate(u)
        p_implied = self.Rp * q + self._distal_pressure(q, self.dt_sim)
        return abs(p_implied - self.P_out_imposed) / (self.P_ATOL + abs(p_implied))

    def advance(self):
        """
        Accept the current nonlinear iterate as the converged state of this time step.
        """
        self.Pd_prev = self.Pd_nl
        self.Pd_nl_prev = self.Pd_nl

    def update(self, u, verbose=True):
        """
        Recompute the flow rate from 'u', integrate the ODE over one time step, and store the result in the P_out constant that the momentum form references.
        """
        mesh = u.function_space.mesh
        self.Q_out = self.flow_rate(u)
        p_out = self.RK4([0.0, self.dt_sim])
        self.P_out.value = p_out
        self.P_out_imposed = p_out
        if verbose and mesh.comm.rank == 0:
            print(
                f"    [Windkessel] Pressure on cap {self.cap_id:d}: {p_out:.2e}",
                flush=True,
            )
        return p_out

    def impedance(self):
        """
        dP_out/dQ for this outlet over one time step, in closed form.

        The distal pressure is affine in Q with slope Rd*(1 - A^n), A being the RK4 stability polynomial, so

            dP_out/dQ = Rp + Rd*(1 - A^n)

        This is the tangent the coupled Newton iteration needs for a single independent RCR. It is not what the solver uses -- that comes from numerical_tangent(), which asks the model rather than assuming its form -- and it is kept as the reference the numerical tangent is checked against.
        """
        span = self.dt_sim
        n = int(self.Niter)
        z = (span / n) / (self.Rd * self.C)
        Am1 = z * (-1.0 + z * (0.5 + z * (-1.0 / 6.0 + z / 24.0)))
        A = 1.0 + Am1
        powm1 = np.expm1(n * np.log1p(Am1)) if A > 0.0 else A**n - 1.0
        return self.Rp - self.Rd * powm1

    def __repr__(self):
        return f"Windkessel(cap_id={self.cap_id}, Rp={self.Rp!r}, Rd={self.Rd!r}, C={self.C!r}, Niter={self.Niter})"


# One lumped-parameter network coupled to the 3D domain
class WindkesselNetwork:
    """
    Adapts a list of independent Windkessel outlets to the coupling interface.

    The interface a 0D model presents to the solver is a map from the cap flow rates to the cap pressures:

        cap_ids               ordered; fixes the row and column index of the tangent
        pressures(Q)          the map itself, with no side effects
        impose(Q)             evaluate it and write the result into the P_out constants
        coupling_residual(u)  how far the imposed pressures are from the ones u implies
        advance()             accept the state at the end of a converged step

    `pressures` is called m+1 times per nonlinear iteration to build the tangent numerically, so it MUST leave the model's state untouched: it integrates from the accepted previous time level every time, never from the current iterate. Here that is free, since Windkessel._distal_pressure always starts from Pd_prev. A network holding an internal integrator would have to save and restore around the call.

    Nothing about this class is specific to an RCR, and nothing about it assumes the outlets are independent. It happens to be diagonal because these outlets are; a network whose outlets share a compliance would return a full matrix from the same interface and the solver would not know the difference.
    """

    def __init__(self, outlets):
        self.outlets = list(outlets)
        self.cap_ids = [o.cap_id for o in self.outlets]

    def __len__(self):
        return len(self.outlets)

    def __iter__(self):
        return iter(self.outlets)

    def flow_rates(self, u):
        """Cap flow rates from the velocity field, positive leaving the domain."""
        return np.array([o.flow_rate(u) for o in self.outlets], dtype=float)

    def pressures(self, flow_rates):
        """
        Cap pressures for the given flow rates. No side effects.
        """
        return np.array(
            [o.Rp * q + o._distal_pressure(q, o.dt_sim)
             for o, q in zip(self.outlets, flow_rates, strict=True)],
            dtype=float,
        )

    def impose(self, flow_rates, verbose=False, comm=None):
        """
        Evaluate the model and write the pressures into the constants the momentum form references. Also records what was imposed, which coupling_residual() measures against.
        """
        pressures = self.pressures(flow_rates)
        for o, q, p_out in zip(self.outlets, flow_rates, pressures, strict=True):
            o.Q_out = float(q)
            o.Pd_nl = float(p_out) - o.Rp * float(q)
            o.P_out.value = p_out
            o.P_out_imposed = float(p_out)
        if verbose and (comm is None or comm.rank == 0):
            for o, p_out in zip(self.outlets, pressures, strict=True):
                print(f"    [Windkessel] Pressure on cap {o.cap_id:d}: {p_out:.2e}", flush=True)
        return pressures

    def coupling_residual(self, u):
        """
        Largest mismatch between the imposed pressures and the ones 'u' implies.
        """
        return max(o.coupling_residual(u) for o in self.outlets)

    def advance(self):
        for o in self.outlets:
            o.advance()

    def analytic_tangent(self):
        """
        The closed-form tangent, diagonal because these outlets are independent. Used to check numerical_tangent(); not used by the solver.
        """
        return np.diag([o.impedance() for o in self.outlets])


# Numerical Dirichlet-to-Neumann tangent of a 0D model
def numerical_tangent(model, flow_rates, epsilon=1.0e-6, scale=None):
    """
    M[k, l] = dP_k / dQ_l, by perturbing one cap flow rate at a time and re-evaluating the model.

    This is the coupling tangent of Esmaily Moghadam et al., J. Comput. Phys. 244 (2013) 63-79. The 0D model is treated as a black box: the only thing asked of it is the map from flow rates to pressures, so an arbitrary lumped-parameter network -- nonlinear, valved, closed-loop -- couples implicitly without anyone deriving its derivative by hand. Networks whose outlets share state give a dense M, which is the case a per-outlet impedance cannot express.

    Costs m+1 evaluations of the model. For a network expensive enough that this matters, the tangent can be held over several iterations; the coupling is a preconditioning of the nonlinear iteration, so a stale M costs iterations rather than accuracy.

    The perturbation is relative with an absolute floor: cap flow rates pass through zero during flow reversal, and a purely relative step would vanish there.
    """
    flow_rates = np.asarray(flow_rates, dtype=float)
    m = flow_rates.size
    if scale is None:
        scale = max(float(np.max(np.abs(flow_rates))), 1.0) if m else 1.0

    base = np.asarray(model.pressures(flow_rates), dtype=float)
    tangent = np.zeros((m, m), dtype=float)
    for col in range(m):
        delta = epsilon * max(abs(flow_rates[col]), scale)
        perturbed = flow_rates.copy()
        perturbed[col] += delta
        tangent[:, col] = (np.asarray(model.pressures(perturbed), dtype=float) - base) / delta
    return tangent


# Pin a single pressure degree of freedom
def pressure_pin_bc(Q, value=0.0, point=None):
    """
    Fix the hydrostatic constant by constraining one pressure dof.

    Needed only when the velocity is prescribed on the whole boundary, where the pressure is determined up to a constant. A traction boundary -- an outflow, or a Windkessel -- removes that nullspace, so the physical cases do not use this.

    The dof closest to 'point' (default: the origin corner) is chosen, deterministically, so that the result does not depend on the partitioning.
    """
    coords = Q.tabulate_dof_coordinates()
    gdim = Q.mesh.geometry.dim
    comm = Q.mesh.comm

    # Closest owned dof on this rank
    n_owned = Q.dofmap.index_map.size_local
    local = coords[:n_owned, :gdim]
    target = np.zeros(gdim) if point is None else np.asarray(point, dtype=float)[:gdim]
    if local.shape[0] > 0:
        d2 = np.sum((local - target) ** 2, axis=1)
        best_local = int(np.argmin(d2))
        best_dist = float(d2[best_local])
    else:
        best_local, best_dist = -1, np.inf

    # Pick a single global winner, breaking ties by rank so every process agrees
    _, owner = comm.allreduce((best_dist, comm.rank), op=MPI.MINLOC)
    dofs = (
        np.array([best_local], dtype=np.int32)
        if comm.rank == owner
        else np.zeros(0, dtype=np.int32)
    )
    return dirichletbc(PETSc.ScalarType(value), dofs, Q)
