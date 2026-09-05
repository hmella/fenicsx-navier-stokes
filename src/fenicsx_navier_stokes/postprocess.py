"""
Derived quantities: flow rates, mass balance, drag and lift, point evaluation.
"""

import numpy as np
from dolfinx.fem import Function, assemble_scalar, form, locate_dofs_topological
from mpi4py import MPI
from ufl import FacetNormal, Identity, Measure, div, dot, grad, inner, nabla_grad, sym

from .constitutive import eps


# Flow rate through a tagged surface
def flow_rate(u, facet_tags, tag, quadrature_degree=4):
    """
    Returns the integral of u.n over the facets carrying 'tag'. Positive means flow leaving the domain, since n is the outward normal.
    """
    mesh = u.function_space.mesh
    ds = Measure("ds", domain=mesh, subdomain_data=facet_tags,
                 metadata={"quadrature_degree": quadrature_degree})
    n = FacetNormal(mesh)
    return mesh.comm.allreduce(assemble_scalar(form(dot(u, n) * ds(int(tag)))), op=MPI.SUM)


# Mass balance between the inlet and the outlets
def mass_balance(u, facet_tags, inlet_tag, outlet_tags, quadrature_degree=4):
    """
    Returns {"Q_in", "Q_out", "defect"} with the relative defect |Q_in + sum(Q_out)|/|Q_in|.

    For this scheme the GLOBAL balance holds to round-off, and that is a property of the formulation rather than an accident: testing the continuity row with the constant pressure function kills every stabilization term in it, each carrying grad(q), leaving the integral of div(u) equal to zero exactly. By the divergence theorem that is precisely Q_in + sum(Q_out).

    So a small value here confirms the assembly and the facet tags, not the quality of the velocity field. P1-P1 with PSPG does NOT conserve mass pointwise: the local defect is what divergence_norm() measures. The property also does not survive a pressure Dirichlet condition, which removes the constant from the test space.
    """
    q_in = flow_rate(u, facet_tags, inlet_tag, quadrature_degree)
    q_out = [flow_rate(u, facet_tags, t, quadrature_degree) for t in outlet_tags]
    total = q_in + sum(q_out)
    denom = abs(q_in) if abs(q_in) > 0 else 1.0
    return {"Q_in": q_in, "Q_out": q_out, "defect": abs(total) / denom}


# Scale-free measure of the local mass defect
def divergence_norm(u, quadrature_degree=4):
    """
    Returns ||div u||_L2 / ||grad u||_L2. This is the quantity to judge a solution by, since the global balance above is exact by construction.
    """
    mesh = u.function_space.mesh
    dx = Measure("dx", domain=mesh, metadata={"quadrature_degree": quadrature_degree})
    num = mesh.comm.allreduce(assemble_scalar(form(div(u) ** 2 * dx)), op=MPI.SUM)
    den = mesh.comm.allreduce(assemble_scalar(form(inner(grad(u), grad(u)) * dx)), op=MPI.SUM)
    return float(np.sqrt(num / den)) if den > 0 else 0.0


# Test function equal to a unit vector on a body
def _body_test_function(V, facet_tags, body_tag, component):
    # Only the dofs on the body are set, so the function is supported in the single element layer around it and the residual assembly touches only those cells
    w = Function(V)
    w.x.array[:] = 0.0
    fdim = V.mesh.topology.dim - 1
    dofs = locate_dofs_topological(V.sub(component), fdim, facet_tags.find(int(body_tag)))
    w.x.array[dofs] = 1.0
    w.x.scatter_forward()
    return w


# Force on a body from the discrete residual (reaction-force method)
def drag_lift_variational(problem, body_tag):
    """
    The discrete solution satisfies R(u_h, p_h; v) = 0 for every test function v vanishing on the Dirichlet boundary. Choosing a w that does NOT vanish on the body makes R(u_h, p_h; w) the reaction force the constraint exerts there.

    This is the form to trust. It converges roughly an order faster than the surface integral, because it inherits the Galerkin orthogonality of the discretization rather than the accuracy of a boundary flux, and it avoids the facet-orientation noise of a polygonal approximation to a smooth body.

    R below is the COMPLETE residual, stabilization included: dropping the SUPG/PSPG and grad-div terms would break exactly the orthogonality the method relies on. Boundary terms on the outlets are omitted because w vanishes there.
    """
    u, p = problem.u_h, problem.p_h
    up, u0, u00 = problem.up, problem.u0, problem.u00
    rho, mu, dt = problem.rho, problem.mu, problem.dt
    tM, tC = problem.tau_M, problem.tau_C
    dx = problem.dx
    gdim = problem.mesh.geometry.dim

    # Strong residual of the momentum equation. The viscous contribution is omitted because it vanishes elementwise for P1 velocities, matching the stabilization the solver actually assembles
    strong = rho / dt * (1.5 * u - 2.0 * u0 + 0.5 * u00) + rho * grad(u) * up
    if problem.f is not None:
        strong = strong - problem.f

    forces = np.zeros(gdim)
    for i in range(gdim):
        w = _body_test_function(problem.V, problem.boundaries, body_tag, i)
        R = (
            rho / dt * inner(1.5 * u - 2.0 * u0 + 0.5 * u00, w) * dx
            + rho * inner(grad(u) * up, w) * dx
            + 2 * mu * inner(eps(u), eps(w)) * dx
            - p * div(w) * dx
            + tM * inner(grad(w) * up, strong) * dx
            + tM * inner(grad(p), grad(w) * up) * dx
            + tC * rho * div(u) * div(w) * dx
        )
        if problem.f is not None:
            R -= inner(problem.f, w) * dx
        if problem.supg_viscous:
            R += tM * mu * inner(nabla_grad(grad(w) * up), nabla_grad(u)) * dx
        forces[i] = -problem.mesh.comm.allreduce(assemble_scalar(form(R)), op=MPI.SUM)
    return forces


# Force on a body by integrating the traction over its surface
def drag_lift_surface(problem, body_tag, symmetric_stress=True, quadrature_degree=4):
    """
    Cross-check for drag_lift_variational(). With symmetric_stress=True the physical Cauchy stress -p*I + 2*mu*sym(grad u) is used; False gives the -p*I + mu*grad(u) form of the DFG benchmark definition.

    The two stress forms agree for the exact solution: on a no-slip boundary all tangential derivatives of u vanish, so grad(u).T.n reduces to n*div(u) = 0. They differ discretely by exactly mu*integral(div(u_h)*(n.v)), so the gap between them measures the discrete mass-conservation defect at the wall.
    """
    mesh = problem.mesh
    n = FacetNormal(mesh)
    ds = Measure("ds", domain=mesh, subdomain_data=problem.boundaries,
                 metadata={"quadrature_degree": quadrature_degree})
    ident = Identity(mesh.geometry.dim)
    u, p, mu = problem.u_h, problem.p_h, problem.mu

    sigma = -p * ident + (2 * mu * sym(grad(u)) if symmetric_stress else mu * grad(u))
    traction = dot(sigma, n)
    return np.array(
        [
            -mesh.comm.allreduce(assemble_scalar(form(traction[i] * ds(int(body_tag)))), op=MPI.SUM)
            for i in range(mesh.geometry.dim)
        ]
    )


# Non-dimensionalise a force
def drag_lift_coefficients(force, rho, u_ref, length):
    """
    Returns c = 2*F/(rho*u_ref^2*L).
    """
    return np.asarray(force) * 2.0 / (float(rho) * float(u_ref) ** 2 * float(length))


# Evaluate a scalar function at physical points
def evaluate_at_points(fn, points, mesh=None):
    """
    Evaluate at points of shape (n, 3), in parallel, raising if a point lies outside the mesh on every rank.

    Worth guarding explicitly: the DFG benchmark's pressure probes sit exactly on the cylinder surface, and a first-order mesh inscribes the circle, so whether a probe is found depends on where mesh vertices happen to land. Without this check a missed point returns a silent nan that propagates into the reported pressure drop.
    """
    from dolfinx.geometry import bb_tree, compute_colliding_cells, compute_collisions_points

    mesh = mesh if mesh is not None else fn.function_space.mesh
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)

    # Find the cell containing each point on this rank
    tree = bb_tree(mesh, mesh.topology.dim)
    candidates = compute_collisions_points(tree, points)
    colliding = compute_colliding_cells(mesh, candidates, points)

    values = np.full(points.shape[0], np.nan)
    for i in range(points.shape[0]):
        cells = colliding.links(i)
        if len(cells) > 0:
            values[i] = fn.eval(points[i], cells[:1])[0]

    # A point is only missing if no rank found it
    gathered = np.vstack(mesh.comm.allgather(values))
    missing = np.isnan(gathered).all(axis=0)
    if missing.any():
        raise ValueError(
            f"evaluation points outside the mesh: {points[missing].tolist()}. On a first-order mesh a point on a "
            "curved boundary lies just outside the polygonal domain; move it inward "
            "slightly."
        )
    with np.errstate(invalid="ignore"):
        return np.nanmax(gathered, axis=0)


# Average of a scalar function over a measure
def cross_section_average(fn, dx_measure, subdomain=None):
    """
    Returns the integral of 'fn' over the measure divided by the measure of the region, e.g. the mean pressure over a tagged cross-section.
    """
    mesh = fn.function_space.mesh
    m = dx_measure if subdomain is None else dx_measure(subdomain)
    num = mesh.comm.allreduce(assemble_scalar(form(fn * m)), op=MPI.SUM)
    den = mesh.comm.allreduce(assemble_scalar(form(1.0 * m)), op=MPI.SUM)
    return num / den
