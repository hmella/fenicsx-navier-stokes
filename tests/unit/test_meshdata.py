"""Consistency between mesh facet tags and the configurations that reference them.

The failure this guards against is quiet rather than loud.  A configuration pointed at the wrong mesh -- or a mesh whose tags were renumbered -- makes ``locate_dofs_topological`` return an empty array, so no boundary condition is applied at all and the solve produces a plausible-looking but meaningless field.

Two real instances motivated these checks: the aorta meshes disagree on the wall tag (4 for the 9 and 11 mm cases, 1 for 13 mm -- correct per-mesh metadata, but easy to cross), and the tube mesh generator originally assigned ``inlet=1, outlet=2, wall=3`` while its configuration expected ``wall=1, inlet=2, outlet=3``.
"""

import numpy as np
import pytest
from basix.ufl import element
from dolfinx.fem import Constant, assemble_scalar, form, functionspace, locate_dofs_topological
from dolfinx.mesh import exterior_facet_indices
from mpi4py import MPI
from ufl import FacetNormal, Measure

from fenicsx_navier_stokes import ParameterHandler

AORTA_CONFIGS = ["Ao9mmrest.yaml", "Ao11mmrest.yaml", "Ao13mmrest.yaml"]


def check_tags(mesh, facet_tags, wall_ids, inlet_id, outlet_ids):
    """Assert the tag groups are disjoint, non-empty, and cover the whole boundary."""
    comm = mesh.comm
    wall_ids = [wall_ids] if np.isscalar(wall_ids) else list(wall_ids)
    groups = {"wall": wall_ids, "inlet": [inlet_id], "outlet": list(outlet_ids)}

    # Gather the tag values across ranks: on a partitioned mesh a rank need not own any facet of a given tag, so the local set is not the mesh's set.
    local = set(np.unique(facet_tags.values).tolist())
    present = sorted(set().union(*comm.allgather(local)))
    for name, ids in groups.items():
        for tag in ids:
            assert tag in present, f"{name} tag {tag} is not in the mesh (present: {present})"

    flat = [t for ids in groups.values() for t in ids]
    assert len(flat) == len(set(flat)), f"tag groups overlap: {groups}"

    # Every tag group must own at least one facet somewhere in the partitioning.
    for name, ids in groups.items():
        n = sum(comm.allreduce(len(facet_tags.find(t)), op=MPI.SUM) for t in ids)
        assert n > 0, f"{name} tags {ids} match no facets"

    # The tags must cover the whole exterior boundary; an untagged facet silently becomes a free-traction surface. The comparison is made rank-locally on facet indices, not on summed counts: a facet shared between ranks appears in more than one local set, so summing the two counts and subtracting them compares unequal quantities and can even come out negative.
    fdim = mesh.topology.dim - 1
    mesh.topology.create_connectivity(fdim, mesh.topology.dim)
    exterior = set(exterior_facet_indices(mesh.topology).tolist())
    tagged = set(np.unique(np.hstack([facet_tags.find(t) for t in flat])).tolist())
    untagged = comm.allreduce(len(exterior - tagged), op=MPI.SUM)
    assert untagged == 0, f"{untagged} exterior facets carry no tag"


def check_bcs_are_not_empty(mesh, facet_tags, wall_ids, inlet_id):
    """The velocity space must actually receive constrained dofs."""
    V = functionspace(
        mesh, element("Lagrange", mesh.topology.cell_name(), 1, shape=(mesh.geometry.dim,))
    )
    fdim = mesh.topology.dim - 1
    wall_ids = [wall_ids] if np.isscalar(wall_ids) else list(wall_ids)
    facets = np.unique(np.hstack([facet_tags.find(t) for t in wall_ids]))
    n_wall = mesh.comm.allreduce(len(locate_dofs_topological(V, fdim, facets)), op=MPI.SUM)
    n_inlet = mesh.comm.allreduce(
        len(locate_dofs_topological(V, fdim, facet_tags.find(inlet_id))), op=MPI.SUM
    )
    assert n_wall > 0, "no wall dofs: the configuration does not match this mesh"
    assert n_inlet > 0, "no inlet dofs: the configuration does not match this mesh"


def check_closed_surface(mesh, facet_tags):
    """``integral(n) == 0`` over a closed surface; catches inverted or duplicated facets."""
    ds = Measure("ds", domain=mesh, subdomain_data=facet_tags, metadata={"quadrature_degree": 2})
    n = FacetNormal(mesh)
    comm = mesh.comm
    area = comm.allreduce(assemble_scalar(form(Constant(mesh, 1.0) * ds)), op=MPI.SUM)
    total = np.array(
        [
            comm.allreduce(assemble_scalar(form(n[i] * ds)), op=MPI.SUM)
            for i in range(mesh.geometry.dim)
        ]
    )
    assert np.abs(total).max() < 1e-10 * area, f"integral of n is {total}, area {area}"


class TestTube:
    """The generated tube mesh against its shipped configuration."""

    def test_tags_are_consistent(self, cylinder, repo_root):
        mesh, _, ft = cylinder
        pars = ParameterHandler(repo_root / "examples" / "tube3d" / "tube3d.yaml")
        check_tags(mesh, ft, pars.Geometry.WallID, pars.Geometry.InletID, pars.Geometry.OutletIDs)

    def test_boundary_conditions_are_populated(self, cylinder, repo_root):
        mesh, _, ft = cylinder
        pars = ParameterHandler(repo_root / "examples" / "tube3d" / "tube3d.yaml")
        check_bcs_are_not_empty(mesh, ft, pars.Geometry.WallID, pars.Geometry.InletID)

    def test_surface_is_closed(self, cylinder):
        mesh, _, ft = cylinder
        check_closed_surface(mesh, ft)

    def test_cap_area(self, cylinder):
        """The inlet cap area must approach ``pi R^2``."""
        mesh, _, ft = cylinder
        ds = Measure("ds", domain=mesh, subdomain_data=ft, metadata={"quadrature_degree": 2})
        area = mesh.comm.allreduce(assemble_scalar(form(Constant(mesh, 1.0) * ds(2))), op=MPI.SUM)
        assert area == pytest.approx(np.pi * 0.5**2, rel=0.03)


class TestTurek:
    """The generated DFG mesh against its shipped configuration."""

    @pytest.fixture(scope="class")
    def turek(self):
        from conftest import load_example_module

        return load_example_module("turek", "turek_mesh.py").generate()

    def test_tags_are_consistent(self, turek, repo_root):
        mesh, _, ft = turek
        pars = ParameterHandler(repo_root / "examples" / "turek" / "turek2d.yaml")
        check_tags(mesh, ft, pars.Geometry.WallID, pars.Geometry.InletID, pars.Geometry.OutletIDs)

    def test_obstacle_is_part_of_the_wall(self, turek, repo_root):
        """The obstacle carries its own tag but must also be no-slip."""
        pars = ParameterHandler(repo_root / "examples" / "turek" / "turek2d.yaml")
        assert pars.Geometry.ObstacleID in pars.Geometry.WallID

    def test_obstacle_perimeter(self, turek):
        """The obstacle perimeter approaches ``2 pi R``; a polygon inscribes it."""
        mesh, _, ft = turek
        ds = Measure("ds", domain=mesh, subdomain_data=ft, metadata={"quadrature_degree": 2})
        length = mesh.comm.allreduce(assemble_scalar(form(Constant(mesh, 1.0) * ds(5))), op=MPI.SUM)
        assert length == pytest.approx(2 * np.pi * 0.05, rel=0.05)
        assert length <= 2 * np.pi * 0.05, "an inscribed polygon cannot be longer"


@pytest.mark.slow
@pytest.mark.parametrize("name", AORTA_CONFIGS)
def test_aorta_configs_match_their_meshes(repo_root, name, comm):
    """Each aorta configuration must match the mesh it names.

    Marked slow: these meshes are ~640k tetrahedra. They are not kept in the repository, so this skips on a fresh clone and the suite still comes out green; see data/aorta/README.md.
    """
    from dolfinx import io

    pars = ParameterHandler(repo_root / "examples" / "aorta" / name)
    mesh_file = repo_root / pars.Geometry.MeshFile
    bnd_file = repo_root / pars.Geometry.BoundariesFile
    if not mesh_file.exists() or not bnd_file.exists():
        pytest.skip(f"{mesh_file} not present; see data/aorta/README.md")

    with io.XDMFFile(comm, str(mesh_file), "r") as xdmf:
        mesh = xdmf.read_mesh(name="Grid")
    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    with io.XDMFFile(comm, str(bnd_file), "r") as xdmf:
        ft = xdmf.read_meshtags(mesh, name="Grid")

    check_tags(mesh, ft, pars.Geometry.WallID, pars.Geometry.InletID, pars.Geometry.OutletIDs)
    check_bcs_are_not_empty(mesh, ft, pars.Geometry.WallID, pars.Geometry.InletID)
    check_closed_surface(mesh, ft)
