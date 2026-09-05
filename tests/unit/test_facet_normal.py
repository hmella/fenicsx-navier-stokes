"""Discrete facet-normal approximation.

``facet_vector_approximation`` L2-projects the facet normal into a function space.  On a flat, singly-tagged face the result is exact; on a curved boundary it inherits the O(h) error of the piecewise-linear geometry, and asserting anything better would be wrong.
"""

import numpy as np
from basix.ufl import element
from conftest import TOL_SOLVE
from dolfinx.fem import functionspace
from dolfinx.mesh import create_unit_cube, create_unit_square, locate_entities_boundary, meshtags
from mpi4py import MPI

from fenicsx_navier_stokes import facet_vector_approximation

FACE = 7


def tag_one_face(mesh, marker):
    fdim = mesh.topology.dim - 1
    facets = locate_entities_boundary(mesh, fdim, marker)
    mesh.topology.create_connectivity(fdim, mesh.topology.dim)
    order = np.argsort(facets)
    return meshtags(mesh, fdim, facets[order], np.full(facets.size, FACE, dtype=np.int32)[order])


def vector_space(mesh, degree=1):
    return functionspace(
        mesh, element("Lagrange", mesh.topology.cell_name(), degree, shape=(mesh.geometry.dim,))
    )


def dofs_on_face(V, mt):
    from dolfinx.fem import locate_dofs_topological

    return locate_dofs_topological(V, V.mesh.topology.dim - 1, mt.find(FACE))


def max_over_ranks(comm, value):
    return comm.allreduce(value, op=MPI.MAX)


def test_flat_edge_in_2d(comm):
    """On the x=0 edge of the unit square the normal is exactly ``(-1, 0)``."""
    mesh = create_unit_square(comm, 4, 4)
    mt = tag_one_face(mesh, lambda x: np.isclose(x[0], 0.0))
    V = vector_space(mesh)
    nh = facet_vector_approximation(V, mt, FACE)

    dofs = dofs_on_face(V, mt)
    vals = nh.x.array.reshape(-1, 2)[dofs]
    err = float(np.abs(vals - np.array([-1.0, 0.0])).max()) if dofs.size else 0.0
    assert max_over_ranks(comm, err) < TOL_SOLVE


def test_flat_face_in_3d(comm):
    """On the z=0 face of the unit cube the normal is exactly ``(0, 0, -1)``."""
    mesh = create_unit_cube(comm, 3, 3, 3)
    mt = tag_one_face(mesh, lambda x: np.isclose(x[2], 0.0))
    V = vector_space(mesh)
    nh = facet_vector_approximation(V, mt, FACE)

    dofs = dofs_on_face(V, mt)
    vals = nh.x.array.reshape(-1, 3)[dofs]
    err = float(np.abs(vals - np.array([0.0, 0.0, -1.0])).max()) if dofs.size else 0.0
    assert max_over_ranks(comm, err) < TOL_SOLVE


def test_dofs_away_from_the_face_are_zero(comm):
    """The block-deactivation must leave every unrelated dof untouched.

    Those dofs are not constrained by the projection at all, so if the deactivation were wrong they would carry arbitrary solver noise rather than zero.
    """
    mesh = create_unit_square(comm, 4, 4)
    mt = tag_one_face(mesh, lambda x: np.isclose(x[0], 0.0))
    V = vector_space(mesh)
    nh = facet_vector_approximation(V, mt, FACE)

    dofs = dofs_on_face(V, mt)
    mask = np.ones(nh.x.array.size, dtype=bool)
    blocked = np.concatenate([2 * dofs, 2 * dofs + 1]) if dofs.size else np.array([], int)
    mask[blocked] = False
    err = float(np.abs(nh.x.array[mask]).max()) if mask.any() else 0.0
    assert max_over_ranks(comm, err) < 1e-14


def test_result_is_normalized(comm):
    """Every non-trivial dof value must be a unit vector.

    This is the only property the ``cond_norm`` guard actually guarantees -- it says nothing about direction, which is why the flat-face tests above exist separately.
    """
    mesh = create_unit_cube(comm, 3, 3, 3)
    mt = tag_one_face(mesh, lambda x: np.isclose(x[2], 0.0))
    V = vector_space(mesh)
    nh = facet_vector_approximation(V, mt, FACE)

    vals = nh.x.array.reshape(-1, 3)
    norms = np.linalg.norm(vals, axis=1)
    active = norms > 1e-10
    err = float(np.abs(norms[active] - 1.0).max()) if active.any() else 0.0
    assert max_over_ranks(comm, err) < 1e-12


def test_tangent_is_orthogonal_to_the_normal(comm):
    """With ``tangent=True`` the result must be perpendicular to the face normal."""
    mesh = create_unit_cube(comm, 3, 3, 3)
    mt = tag_one_face(mesh, lambda x: np.isclose(x[2], 0.0))
    V = vector_space(mesh)
    th = facet_vector_approximation(V, mt, FACE, tangent=True)

    dofs = dofs_on_face(V, mt)
    vals = th.x.array.reshape(-1, 3)[dofs]
    err = float(np.abs(vals[:, 2]).max()) if dofs.size else 0.0
    assert max_over_ranks(comm, err) < TOL_SOLVE


def test_averages_across_a_rim(comm):
    """A continuous space averages the normal where two tagged faces meet.

    Tagging two perpendicular edges of the square makes the shared corner dof the mean of ``(-1, 0)`` and ``(0, -1)``, normalized.  This is a genuine limitation of using a continuous space -- the docstring calls for a discontinuous one -- and it is harmless for the inlet profile only because a cap is planar.  Pinning it here keeps the behaviour honest rather than surprising.
    """
    mesh = create_unit_square(comm, 4, 4)
    mt = tag_one_face(mesh, lambda x: np.isclose(x[0], 0.0) | np.isclose(x[1], 0.0))
    V = vector_space(mesh)
    nh = facet_vector_approximation(V, mt, FACE)

    coords = V.tabulate_dof_coordinates()
    corner = np.flatnonzero(np.isclose(coords[:, 0], 0.0) & np.isclose(coords[:, 1], 0.0))
    local_err = 0.0
    if corner.size:
        val = nh.x.array.reshape(-1, 2)[corner[0]]
        expected = np.array([-1.0, -1.0]) / np.sqrt(2.0)
        local_err = float(np.abs(val - expected).max())
    assert max_over_ranks(comm, local_err) < 1e-9
