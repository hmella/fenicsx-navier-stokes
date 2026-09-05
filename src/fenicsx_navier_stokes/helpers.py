"""
Small utilities: inflow waveform loading and L2 projection.
"""

import numpy as np
from dolfinx.fem import Function, form
from dolfinx.fem.petsc import apply_lifting, assemble_matrix, assemble_vector, set_bc
from petsc4py import PETSc
from ufl import Measure, TestFunction, TrialFunction, inner


# Import inflow profile
def inflow_profile(path, scale_factor=1.0):
    """
    Load a two-column (time, velocity) waveform. Only column 1 is scaled: scaling the time column too would silently change the time step, which the solver infers from the row spacing of this file.
    """
    data = np.loadtxt(path)

    # Basic shape checks
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(
            f"{path}: expected a 2-column (time, velocity) file, got shape {data.shape}"
        )
    if data.shape[0] < 2:
        raise ValueError(f"{path}: need at least two rows to infer a time step")

    # BDF2 and the Windkessel sub-stepping both assume a constant dt
    t = data[:, 0]
    dt = np.diff(t)
    if not np.all(dt > 0):
        raise ValueError(f"{path}: time column must be strictly increasing")
    if not np.allclose(dt, dt[0], rtol=1e-6, atol=0.0):
        raise ValueError(
            f"{path}: time column must be uniformly spaced (dt varies between {dt.min():.6e} and {dt.max():.6e})"
        )

    # Scale the velocity column only
    out = data[:, :2].copy()
    out[:, 1] *= scale_factor
    return out


# Project function
def project(expr, function_space, lumped=False, rtol=1.0e-8):
    """
    Project a ufl expression into 'function_space' by solving a mass matrix system. With lumped=True a vertex quadrature rule is used, which diagonalises the mass matrix.
    """
    dx = Measure("dx", metadata={"quadrature_rule": "vertex"}) if lumped else Measure("dx")

    # Mass matrix and right-hand side
    u = TrialFunction(function_space)
    v = TestFunction(function_space)
    A = assemble_matrix(form(inner(u, v) * dx))
    A.assemble()
    b = assemble_vector(form(inner(expr, v) * dx))
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    # Solve with conjugate gradients (the mass matrix is symmetric positive definite)
    solver = PETSc.KSP().create(function_space.mesh.comm)
    solver.setType("cg")
    solver.rtol = rtol
    solver.setOperators(A)

    uh = Function(function_space)
    solver.solve(b, uh.x.petsc_vec)  # Function.vector was removed in dolfinx 0.9
    uh.x.scatter_forward()

    solver.destroy()
    A.destroy()
    b.destroy()
    return uh


# Reusable projector
class Projector:
    """
    L2 projector that assembles the mass matrix once. Use this instead of project() when the same projection is repeated every time step, e.g. for a derived field written to file.
    """

    def __init__(self, function, space, bcs=None, lumped=False, metadata=None, rtol=1.0e-8):
        self.bcs = list(bcs) if bcs is not None else []
        dx = Measure("dx", metadata={"quadrature_rule": "vertex"}) if lumped else Measure("dx")

        # Assemble the mass matrix once
        u = TrialFunction(space)
        v = TestFunction(space)
        self.lhs = form(inner(u, v) * dx(metadata=metadata))
        self.A = assemble_matrix(self.lhs, bcs=self.bcs)
        self.A.assemble()

        # Compile the right-hand side form and allocate the vectors
        self.rhs = form(inner(function, v) * dx(metadata=metadata))
        self.x = Function(space)
        self.b = Function(space)

        # Krylov solver
        self.ksp = PETSc.KSP().create(space.mesh.comm)
        self.ksp.setOperators(self.A)
        self.ksp.setType("cg")
        self.ksp.rtol = rtol
        self.ksp.setFromOptions()

    def assemble_rhs(self):
        """
        Re-assemble the right-hand side and re-apply the boundary conditions.
        """
        self.b.x.array[:] = 0.0
        assemble_vector(self.b.x.petsc_vec, self.rhs)
        apply_lifting(self.b.x.petsc_vec, [self.lhs], bcs=[self.bcs])
        self.b.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(self.b.x.petsc_vec, self.bcs)
        self.b.x.scatter_forward()

    def solve(self, assemble_rhs=True):
        """
        Compute the projection. Pass assemble_rhs=False only if nothing the expression depends on has changed since the last call.
        """
        if assemble_rhs:
            self.assemble_rhs()
        self.ksp.solve(self.b.x.petsc_vec, self.x.x.petsc_vec)
        self.x.x.scatter_forward()
        return self.x
