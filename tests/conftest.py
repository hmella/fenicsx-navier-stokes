"""Shared pytest fixtures.

Mesh generation is the dominant fixed cost of this suite, so every mesh fixture is session-scoped and cached.  Anything that needs gmsh is built once into a temporary directory and re-read from XDMF.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from mpi4py.MPI import MAX as MPI_MAX
from mpi4py.MPI import MIN as MPI_MIN

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

# --- tolerance tiers ---------------------------------------------------------------
# Named so that every assertion states which tier it belongs to, and so that loosening one is a deliberate, greppable act.

# : Pure algebra: transposes, the divergence theorem, coefficient identities.
TOL_EXACT = 1.0e-13
# : Downstream of a Krylov solve.  Bounded by the solver's own rtol, not by the maths.
TOL_SOLVE = 1.0e-7
# : Physical quantities on a realistic mesh: mass defect, pressure drop, drag.
TOL_DISCRETE = 5.0e-2


def rate(errors, sizes):
    """Least-squares convergence rate of ``errors`` against ``sizes``.

    Parameters
    ----------
    errors, sizes
        Sequences of the same length; ``sizes`` is ``h`` or ``dt``.

    Returns
    -------
    float
        The slope of ``log(error)`` against ``log(size)``.
    """
    e = np.asarray(errors, dtype=float)
    s = np.asarray(sizes, dtype=float)
    if np.any(e <= 0.0):
        raise ValueError(f"non-positive error values, cannot fit a rate: {e}")
    return float(np.polyfit(np.log(s), np.log(e), 1)[0])


@pytest.fixture(scope="session")
def comm():
    from mpi4py import MPI

    return MPI.COMM_WORLD


@pytest.fixture(scope="session")
def unit_square_1(comm):
    """A single-cell-pair unit square: two right triangles, each of diameter sqrt(2).

    Used where a hand-computed reference value is wanted, since ``CellDiameter`` is exactly ``sqrt(2)`` on both cells.
    """
    from dolfinx.mesh import CellType, DiagonalType, create_unit_square

    return create_unit_square(comm, 1, 1, CellType.triangle, diagonal=DiagonalType.right)


@pytest.fixture(scope="session")
def unit_square_8(comm):
    """An 8x8 crossed-diagonal unit square.

    Crossed rather than right: the right-diagonal mesh is structurally biased and shows superconvergence artefacts that inflate measured convergence rates.
    """
    from dolfinx.mesh import CellType, DiagonalType, create_unit_square

    return create_unit_square(comm, 8, 8, CellType.triangle, diagonal=DiagonalType.crossed)


@pytest.fixture(scope="session")
def unit_cube_4(comm):
    from dolfinx.mesh import create_unit_cube

    return create_unit_cube(comm, 4, 4, 4)


@pytest.fixture(scope="session")
def repo_root():
    return REPO_ROOT


class Params:
    """A minimal stand-in for :class:`ParameterHandler` built from nested dicts."""

    def __init__(self, **sections):
        for name, values in sections.items():
            setattr(self, name, _Section(values))


class _Section:
    def __init__(self, values):
        self.__dict__.update(values)


def default_params(
    density=1.06,
    viscosity=0.04,
    kind="direct",
    wall=1,
    inlet=2,
    outlets=(3,),
    backflow_beta=0.0,
    nl_tol=1e-10,
    max_nl=30,
):
    """Build a parameter object for the small test problems.

    Defaults to the direct (MUMPS) solver: the test meshes are tiny, and a direct solve removes the Krylov tolerance from every downstream assertion.
    """
    return Params(
        Problem={
            "Density": density,
            "Viscosity": viscosity,
            "BackflowBeta": backflow_beta,
            "SurfaceQuadratureDegree": 4,
            "VolumeQuadratureDegree": 4,
            "VelocityProfileFile": None,
            "VelocityProfileScale": 1.0,
        },
        Solver={
            "Kind": kind,
            "SolverIterations": 500,
            "SolverRTol": 1e-10,
            "SolverATol": 1e-12,
            "SolverGMRESRestart": 100,
            "NonlinearTolerance": nl_tol,
            "MaxNonlinearIterations": max_nl,
        },
        Geometry={
            "WallID": wall,
            "InletID": inlet,
            "OutletIDs": list(outlets),
            "MeshFile": None,
            "BoundariesFile": None,
        },
    )


def tag_square_boundaries(mesh, wall=1, inlet=2, outlet=3):
    """Tag the unit square: inlet at x=0, outlet at x=1, walls at y=0 and y=1.

    Returns
    -------
    dolfinx.mesh.MeshTags
        Facet tags covering the whole exterior boundary.
    """
    from dolfinx.mesh import locate_entities_boundary, meshtags

    fdim = mesh.topology.dim - 1
    specs = [
        (inlet, lambda x: np.isclose(x[0], 0.0)),
        (outlet, lambda x: np.isclose(x[0], 1.0)),
        (wall, lambda x: np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0)),
    ]
    indices, values = [], []
    for tag, marker in specs:
        facets = locate_entities_boundary(mesh, fdim, marker)
        indices.append(facets)
        values.append(np.full_like(facets, tag))
    indices = np.hstack(indices)
    values = np.hstack(values).astype(np.int32)
    order = np.argsort(indices)
    mesh.topology.create_connectivity(fdim, mesh.topology.dim)
    return meshtags(mesh, fdim, indices[order], values[order])


def tag_all_boundaries(mesh, wall=1, inlet=2):
    """Tag x=0 as the inlet and every other exterior facet as wall.

    Used by the fully-Dirichlet cases (the patch test, the manufactured solution), where there is no outflow boundary at all.
    """
    from dolfinx.mesh import exterior_facet_indices, locate_entities_boundary, meshtags

    fdim = mesh.topology.dim - 1
    mesh.topology.create_connectivity(fdim, mesh.topology.dim)
    all_facets = exterior_facet_indices(mesh.topology)
    inlet_facets = locate_entities_boundary(mesh, fdim, lambda x: np.isclose(x[0], 0.0))
    values = np.full(all_facets.shape, wall, dtype=np.int32)
    values[np.isin(all_facets, inlet_facets)] = inlet
    return meshtags(mesh, fdim, all_facets, values)


def perturb_interior_vertices(mesh, amplitude=0.25, seed=0):
    """Randomly displace interior vertices of a box mesh, making cell sizes non-uniform.

    Several stabilization defects are invisible on a structured mesh because ``tau_M`` is then identical in every cell, so the spurious element contributions telescope into a boundary term that vanishes for interior test functions.  Perturbing the geometry breaks that cancellation and gives the patch test real diagnostic power.

    Vertices on the bounding box are left in place, so the domain and any boundary data imposed on it are unchanged.  The displacement is deterministic and bounded by a fraction of the smallest cell diameter, so cells stay valid.

    Parameters
    ----------
    mesh
        An affine mesh of a box-shaped domain.  Modified in place.
    amplitude
        Displacement as a fraction of the smallest cell diameter.  Must stay below ~0.3 to keep cells positively oriented.
    seed
        Seed of the random generator, fixed so runs are reproducible.

    Returns
    -------
    dolfinx.mesh.Mesh
        The same mesh object.
    """
    tdim = mesh.topology.dim
    gdim = mesh.geometry.dim
    n_cells = mesh.topology.index_map(tdim).size_local
    hmin = float(np.min(mesh.h(tdim, np.arange(n_cells, dtype=np.int32))))

    x = mesh.geometry.x
    rng = np.random.default_rng(seed)
    shift = rng.uniform(-1.0, 1.0, size=(x.shape[0], gdim)) * amplitude * hmin

    # Pin the bounding box.  Done geometrically to avoid relying on the correspondence between topological vertices and geometry nodes.
    on_boundary = np.zeros(x.shape[0], dtype=bool)
    for d in range(gdim):
        lo = mesh.comm.allreduce(x[:, d].min(), op=MPI_MIN)
        hi = mesh.comm.allreduce(x[:, d].max(), op=MPI_MAX)
        on_boundary |= np.isclose(x[:, d], lo) | np.isclose(x[:, d], hi)

    interior = ~on_boundary
    x[interior, :gdim] += shift[interior]
    return mesh


def example_config(tmp_path, example, filename, **overrides):
    """Copy a shipped example configuration, apply ``overrides``, and return the new path.

    Overrides are given as ``Section__Key=value`` and are applied to the parsed YAML before it is written back out.  The examples take no parameters on the command line, so this is how a test asks one of them for a smaller mesh or a shorter run.
    """
    import yaml

    src = REPO_ROOT / "examples" / example / filename
    with open(src) as fh:
        data = yaml.safe_load(fh)
    for key, value in overrides.items():
        section, _, name = key.partition("__")
        data.setdefault(section, {})[name] = value

    dest = tmp_path / filename
    with open(dest, "w") as fh:
        yaml.safe_dump(data, fh, sort_keys=False)
    return str(dest)


def load_example_module(example, filename):
    """Import a module from ``examples/<example>/`` under a unique name.

    Example scripts are not on the import path, and loading them by path keeps each one in its own module entry.  Module names are kept distinct across examples (``turek_mesh.py``, ``tube_mesh.py`` rather than ``mesh.py`` twice) because a shared name collides in ``sys.modules``: the first import wins and every later one silently returns it.  That is not hypothetical -- with both called ``mesh.py`` the DFG cylinder tests meshed a tube instead, and still passed.
    """
    import importlib.util

    path = REPO_ROOT / "examples" / example / filename
    name = f"_example_{example}_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def cylinder_factory():
    """Return a cached generator for cylindrical meshes.

    gmsh generation dominates the cost of the geometry-dependent tests, so meshes are built once per session and keyed on their arguments.
    """
    tube_mesh = load_example_module("tube3d", "tube_mesh.py")

    cache = {}

    def make(radius=0.5, length=5.0, resolution=0.15, wall_resolution=None):
        key = (radius, length, resolution, wall_resolution)
        if key not in cache:
            cache[key] = tube_mesh.generate(
                radius=radius,
                length=length,
                resolution=resolution,
                wall_resolution=wall_resolution,
            )
        return cache[key]

    return make


@pytest.fixture(scope="session")
def cylinder(cylinder_factory):
    """A short cylinder, R = 0.5, L = 5, tags wall=1, inlet=2, outlet=3."""
    return cylinder_factory()


@pytest.fixture(scope="session")
def cylinder_graded(cylinder_factory):
    """A radially graded cylinder, R = 0.5, L = 5, ~90k cells (about 4 s to generate).

    The grading matters for anything sensitive to a boundary layer.  The screened-Poisson inlet profile has a layer of width ``lam`` at the wall, and its shape factor is only accurate once that layer spans a few cells.
    """
    return cylinder_factory(radius=0.5, length=5.0, resolution=0.08, wall_resolution=0.02)
