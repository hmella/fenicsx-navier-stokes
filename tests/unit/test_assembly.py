"""Structural tests on the assembled block system.

These are cheap, exact and target the parts of the assembly that are easy to get subtly wrong and hard to notice: the constant-block plus ``axpy`` shortcut, the asymmetric way dolfinx applies boundary conditions to rectangular blocks, the sign convention of the saddle-point system, and the BDF2 coefficients.
"""

import numpy as np
import pytest
from conftest import TOL_EXACT, default_params, tag_square_boundaries
from dolfinx.fem import Function, form
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, create_matrix
from dolfinx.mesh import CellType, DiagonalType, create_unit_square
from mpi4py import MPI
from petsc4py import PETSc
from ufl import FacetNormal, Measure, TestFunction, TrialFunction, div, dot, dx, grad, inner

from fenicsx_navier_stokes import PicardNSProblem


@pytest.fixture
def problem(comm):
    """A small, fully built problem whose matrices can be inspected."""
    mesh = create_unit_square(comm, 6, 6, CellType.triangle, diagonal=DiagonalType.crossed)
    ft = tag_square_boundaries(mesh)
    pars = default_params(density=1.06, viscosity=0.04, kind="direct")
    prob = PicardNSProblem(
        parameters=pars,
        mesh=mesh,
        XDMF=False,
        boundaries=ft,
        element="P1-P1",
        dt=1.0e-2,
        num_steps=1,
        inlet_scale=lambda n, t: 1.0,
    )
    # Give the convecting field a non-trivial value so the *_star blocks are not zero.
    prob.up.interpolate(lambda x: np.vstack([1.0 + x[1], 0.3 * x[0]]))
    prob.up.x.scatter_forward()
    # The blocks were assembled during construction with up == 0; refresh them so they correspond to the convecting field above.
    prob._assemble_system().destroy()
    return prob


def rel_inf_diff(A, B):
    """``||A - B||_inf`` relative to ``||B||_inf``."""
    D = A.copy()
    D.axpy(-1.0, B)
    scale = B.norm(PETSc.NormType.INFINITY)
    out = D.norm(PETSc.NormType.INFINITY) / (scale if scale > 0 else 1.0)
    D.destroy()
    return out


def constrained_velocity_dofs(problem):
    """Global indices of velocity dofs carrying a Dirichlet condition, owned by this rank."""
    idx = [
        bc._cpp_object.dof_indices()[0]
        for bc in problem.bcs
        if bc.function_space == problem.V._cpp_object
    ]
    if not idx:
        return np.zeros(0, dtype=np.int32)
    return np.unique(np.hstack(idx))


class TestAxpyShortcut:
    """The solver assembles constant blocks once and adds them back each iteration."""

    def test_reproduces_from_scratch_assembly(self, problem):
        """``A_star + A_const`` must equal a direct assembly of the combined form.

        This is the single assertion that justifies the whole optimisation.  It also implicitly checks that the sparsity pattern allocated for the combined form is a superset of the constant block's, which is what makes it safe to forbid new nonzero locations afterwards.
        """
        u, v = TrialFunction(problem.V), TestFunction(problem.V)
        p, q = TrialFunction(problem.Q), TestFunction(problem.Q)
        rho, mu, dt = problem.rho, problem.mu, problem.dt
        from fenicsx_navier_stokes.constitutive import eps

        dxm = problem.dx
        a00 = rho / dt * inner(1.5 * u, v) * dxm + 2 * mu * inner(eps(u), eps(v)) * dxm
        a01 = -p * div(v) * dxm
        a10 = -q * div(u) * dxm

        for combined, block in (
            (a00 + problem.a00_star, problem.A00_star),
            (a01 + problem.a01_star, problem.A01_star),
            (a10 + problem.a10_star, problem.A10_star),
        ):
            A_full = create_matrix(form(combined))
            A_full.setOption(PETSc.Mat.Option.IGNORE_ZERO_ENTRIES, False)
            diag = 1.0 if combined is a00 + problem.a00_star else None
            if diag is None:
                assemble_matrix(A_full, form(combined), bcs=problem.bcs)
            else:
                assemble_matrix(A_full, form(combined), bcs=problem.bcs, diag=diag)
            A_full.assemble()

            assert rel_inf_diff(block, A_full) < TOL_EXACT
            A_full.destroy()

    def test_idempotent_across_iterations(self, problem):
        """Re-running the assembly cycle with an unchanged ``up`` must be reproducible.

        Catches a missing ``zeroEntries`` or a doubled ``axpy``, either of which would make the operator drift from one Picard iteration to the next.
        """
        snapshots = []
        for _ in range(3):
            b = problem._assemble_system()
            snapshots.append(problem.A00_star.copy())
            b.destroy()
        assert rel_inf_diff(snapshots[1], snapshots[0]) < TOL_EXACT
        assert rel_inf_diff(snapshots[2], snapshots[0]) < TOL_EXACT
        for s in snapshots:
            s.destroy()


class TestBoundaryConditions:
    """dolfinx treats square and rectangular blocks differently; pin that down."""

    def test_velocity_rows_are_identity(self, problem):
        """Constrained rows of the (0,0) block are unit rows."""
        dofs = constrained_velocity_dofs(problem)
        assert dofs.size > 0
        rstart, rend = problem.A00_star.getOwnershipRange()
        checked = 0
        for d in dofs:
            g = problem.V.dofmap.index_map.local_to_global(
                np.array([d // problem.V.dofmap.index_map_bs], dtype=np.int32)
            )[0] * problem.V.dofmap.index_map_bs + (d % problem.V.dofmap.index_map_bs)
            if not (rstart <= g < rend):
                continue
            cols, vals = problem.A00_star.getRow(int(g))
            diag = vals[cols == g]
            off = vals[cols != g]
            assert diag.size == 1 and abs(diag[0] - 1.0) < TOL_EXACT
            assert np.abs(off).max(initial=0.0) < TOL_EXACT
            checked += 1
        assert problem.mesh.comm.allreduce(checked, op=MPI.SUM) > 0

    def test_a11_ignores_velocity_conditions(self, problem):
        """Velocity conditions passed to the pressure-pressure block are a no-op.

        Both spaces of that form are the pressure space, so dolfinx discards conditions belonging to the velocity space.  Passing them is harmless -- and the list must still be passed, because a genuine *pressure* condition in it does apply.
        """
        A_with = create_matrix(form(problem.a11_star))
        assemble_matrix(A_with, form(problem.a11_star), bcs=problem.bcs, diag=1.0)
        A_with.assemble()

        A_without = create_matrix(form(problem.a11_star))
        assemble_matrix(A_without, form(problem.a11_star), diag=1.0)
        A_without.assemble()

        assert rel_inf_diff(A_with, A_without) < TOL_EXACT
        A_with.destroy()
        A_without.destroy()


class TestStructure:
    """Symmetry and sign conventions of the saddle-point blocks."""

    def test_a10_is_transpose_of_a01(self, problem):
        """``-q div(u)`` and ``-p div(v)`` are transposes, before and after BCs.

        Applying velocity conditions zeroes the rows of ``A01`` and the columns of ``A10`` for the same dof set, so the relation survives.  A sign slip or a wrong measure in either form breaks it immediately.
        """
        A01T = problem.A01.transpose()
        assert rel_inf_diff(A01T, problem.A10) < 1.0e-12
        A01T.destroy()

    def test_a00_is_symmetric(self, problem):
        """Mass plus symmetric-gradient viscous term, with no convection, is symmetric."""
        A00T = problem.A00.transpose()
        assert rel_inf_diff(A00T, problem.A00) < 1.0e-12
        A00T.destroy()

    def test_a11_is_negative_semidefinite(self, problem):
        """The PSPG block is ``tau_M <-grad q, grad p>``, i.e. a negative Laplacian.

        The system is therefore ``[[A, B^T], [B, -C]]`` with ``C >= 0``.  Flipping this sign would still assemble and still be solvable by a direct method, but it makes the ``selfp`` Schur-complement approximation meaningless, so the iterative path would silently degrade rather than fail.
        """
        A = problem.A11_star
        x, y = A.createVecs()
        rng = np.random.default_rng(0)
        for _ in range(10):
            with x.localForm() as loc:
                loc.array[:] = rng.standard_normal(loc.array.shape)
            A.mult(x, y)
            assert x.dot(y) <= 1.0e-12
        x.destroy()
        y.destroy()

    def test_nest_index_sets_partition_the_system(self, problem):
        """The fieldsplit index sets must cover the whole system exactly once."""
        iss = problem.A.getNestISs()
        u_is, p_is = iss[0][0], iss[0][1]
        nu = problem.V.dofmap.index_map.size_global * problem.V.dofmap.index_map_bs
        nq = problem.Q.dofmap.index_map.size_global * problem.Q.dofmap.index_map_bs
        comm = problem.mesh.comm
        assert comm.allreduce(u_is.getLocalSize(), op=MPI.SUM) == nu
        assert comm.allreduce(p_is.getLocalSize(), op=MPI.SUM) == nq
        overlap = np.intersect1d(u_is.getIndices(), p_is.getIndices())
        assert overlap.size == 0


class TestBDF2:
    """The BDF2 right-hand side carries the coefficients 2 and -1/2."""

    @pytest.mark.parametrize("which,coeff", [("u0", 2.0), ("u00", -0.5)])
    def test_history_coefficients(self, problem, which, coeff):
        """Set one history level to a known field and check the assembled load vector.

        Pinning the coefficients individually matters: the patch test only constrains their sum (``1.5 - 2 + 0.5 == 0``), so a compensating pair of errors would slip through it.
        """
        V = problem.V
        rho = float(problem.rho.value)
        dt = problem.dt_value

        e = Function(V)
        e.interpolate(lambda x: np.vstack([np.sin(x[0]), np.cos(x[1])]))
        e.x.scatter_forward()

        problem.u0.x.array[:] = e.x.array if which == "u0" else 0.0
        problem.u00.x.array[:] = e.x.array if which == "u00" else 0.0
        problem.up.x.array[:] = 0.0  # kill convection and the stabilization weight

        b = assemble_vector(form(problem.L[0]))
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

        v = TestFunction(V)
        expected = assemble_vector(form(rho / dt * coeff * inner(e, v) * problem.dx))
        expected.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

        b.axpy(-1.0, expected)
        assert b.norm(PETSc.NormType.INFINITY) < 1.0e-12 * max(
            expected.norm(PETSc.NormType.INFINITY), 1.0
        )
        b.destroy()
        expected.destroy()


def test_divergence_theorem(problem):
    """``integral(div u) == sum over tags of integral(u . n)``, to round-off.

    Exact by the divergence theorem, so it is a sharp check on the facet tags covering the whole boundary, on the normal orientation, and on the MPI reduction.  Every mass-balance assertion elsewhere in the suite depends on this holding.
    """
    from dolfinx.fem import assemble_scalar

    mesh, comm = problem.mesh, problem.mesh.comm
    u = Function(problem.V)
    u.interpolate(lambda x: np.vstack([x[0] ** 2 + x[1], np.sin(x[0]) - 0.5 * x[1]]))
    u.x.scatter_forward()

    n = FacetNormal(mesh)
    ds = Measure("ds", domain=mesh, subdomain_data=problem.boundaries)

    volume = comm.allreduce(assemble_scalar(form(div(u) * dx)), op=MPI.SUM)
    tags = sorted(set(np.unique(problem.boundaries.values).tolist()))
    surface = sum(
        comm.allreduce(assemble_scalar(form(dot(u, n) * ds(int(t)))), op=MPI.SUM) for t in tags
    )
    assert volume == pytest.approx(surface, rel=1.0e-12, abs=1.0e-13)


def test_global_mass_balance_is_exact(problem):
    """``integral(div u) == 0`` to round-off, for any converged solution.

    Testing the continuity row with the constant pressure function annihilates every stabilization term in it, since each carries ``grad(q)``.  What remains is ``-integral(div u) == 0``.  The global mass balance of this scheme is therefore exact by construction -- which is worth pinning, and worth not mistaking for evidence that the velocity field is pointwise divergence-free.  It is not; see ``postprocess.divergence_norm``.
    """
    from dolfinx.fem import assemble_scalar

    from fenicsx_navier_stokes.postprocess import mass_balance

    problem.solve(store_after=None, verbose=False, max_steps=1)
    comm = problem.mesh.comm

    total = comm.allreduce(assemble_scalar(form(div(problem.u_h) * problem.dx)), op=MPI.SUM)
    scale = comm.allreduce(
        assemble_scalar(form(inner(grad(problem.u_h), grad(problem.u_h)) * problem.dx)),
        op=MPI.SUM,
    )
    assert abs(total) < 1e-12 * max(np.sqrt(scale), 1.0)

    balance = mass_balance(problem.u_h, problem.boundaries, 2, [3])
    assert balance["defect"] < 1e-10
