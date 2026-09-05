"""
Generate the 3D tube (straight cylinder) mesh with gmsh.

A SimVascular-style vessel segment: a cylinder of radius R and length L aligned with the z axis, with the inlet cap at z = 0 and the outlet cap at z = L.

Facet tags are wall=1, inlet=2, outlet=3, matching tube3d.yaml. The script this was derived from assigned inlet=1, outlet=2, wall=3 while its configuration expected the opposite, a mismatch that produces no boundary conditions at all rather than an error. The tags are asserted here and cross-checked by tests/unit/test_meshdata.py.

Usage:
    python examples/tube3d/tube_mesh.py --radius 2.0 --length 30.0 --resolution 0.5 --output data/tube3d/tube
"""

import argparse
from pathlib import Path

import numpy as np
from mpi4py import MPI

# Physical group tags
WALL_ID, INLET_ID, OUTLET_ID = 1, 2, 3
FLUID_ID = 11


# Build the mesh
def generate(
    radius=2.0,
    length=30.0,
    resolution=None,
    wall_resolution=None,
    comm=None,
    model_rank=0,
    verbose=False,
):
    """
    Returns (mesh, cell_tags, facet_tags) for a cylinder of the given radius and length.

    'resolution' is the element size away from the wall (default radius/4) and 'wall_resolution' the size at the wall (default: the same). Setting the latter smaller grades the mesh radially, which any boundary-layer-sensitive case needs: the Womersley solution, for instance, has a Stokes layer of thickness sqrt(2*mu/(rho*omega)) that must span several cells.
    """
    import gmsh
    from dolfinx.io.gmsh import model_to_mesh

    comm = comm if comm is not None else MPI.COMM_WORLD
    resolution = resolution if resolution is not None else radius / 4.0
    wall_resolution = wall_resolution if wall_resolution is not None else resolution

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)

        # Under MPI, only one process builds the geometry and meshes it. The others wait and receive their share of the mesh when it is distributed further down.
        if comm.rank == model_rank:
            # Generate a cylinder of length L and radius R, aligned with the Z axis
            gmsh.model.add("tube3d")
            gmsh.model.occ.addCylinder(0.0, 0.0, 0.0, 0.0, 0.0, length, radius)
            gmsh.model.occ.synchronize()

            # Mark the volume as a gmsh "physical group". Only entities belonging to a physical group are exported, and the group's number is the tag that comes back as a cell or facet marker on the dolfinx side.
            volumes = gmsh.model.getEntities(dim=3)
            assert len(volumes) == 1, f"expected one volume, got {len(volumes)}"
            gmsh.model.addPhysicalGroup(3, [volumes[0][1]], FLUID_ID)
            gmsh.model.setPhysicalName(3, FLUID_ID, "Fluid")

            # Identify the two flat caps by their centre of mass: the inlet cap sits at z = 0 and the outlet at z = length. Whatever is left is the curved wall.
            walls = []
            inlet = outlet = None
            for surface in gmsh.model.occ.getEntities(dim=2):
                com = gmsh.model.occ.getCenterOfMass(surface[0], surface[1])
                if np.isclose(com[2], 0.0):
                    inlet = surface[1]
                elif np.isclose(com[2], length):
                    outlet = surface[1]
                else:
                    walls.append(surface[1])
            assert inlet is not None and outlet is not None and walls, (
                "failed to identify the cylinder caps"
            )

            # Create the physical surfaces
            gmsh.model.addPhysicalGroup(2, walls, WALL_ID)
            gmsh.model.setPhysicalName(2, WALL_ID, "Wall")
            gmsh.model.addPhysicalGroup(2, [inlet], INLET_ID)
            gmsh.model.setPhysicalName(2, INLET_ID, "Inlet")
            gmsh.model.addPhysicalGroup(2, [outlet], OUTLET_ID)
            gmsh.model.setPhysicalName(2, OUTLET_ID, "Outlet")

            if wall_resolution != resolution:
                # Grade toward the wall: fine within one wall layer of the surface, coarsening to 'resolution' by a quarter of the radius inward
                dist = gmsh.model.mesh.field.add("Distance")
                gmsh.model.mesh.field.setNumbers(dist, "SurfacesList", walls)
                thr = gmsh.model.mesh.field.add("Threshold")
                gmsh.model.mesh.field.setNumber(thr, "InField", dist)
                gmsh.model.mesh.field.setNumber(thr, "SizeMin", wall_resolution)
                gmsh.model.mesh.field.setNumber(thr, "SizeMax", resolution)
                gmsh.model.mesh.field.setNumber(thr, "DistMin", 0.0)
                gmsh.model.mesh.field.setNumber(thr, "DistMax", 0.25 * radius)
                gmsh.model.mesh.field.setAsBackgroundMesh(thr)
                gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
                gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
                gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
            else:
                gmsh.option.setNumber("Mesh.MeshSizeMin", resolution)
                gmsh.option.setNumber("Mesh.MeshSizeMax", resolution)

            gmsh.model.mesh.generate(3)

        # Load the mesh and its markers. First-order tetrahedra: the P1 residual and CellDiameter both assume an affine geometry, so the mesh order is left at 1. dolfinx 0.10 renamed dolfinx.io.gmshio to dolfinx.io.gmsh and returns a MeshData object rather than a 3-tuple
        data = model_to_mesh(gmsh.model, comm, model_rank, gdim=3)
        return data.mesh, data.cell_tags, data.facet_tags
    finally:
        gmsh.finalize()


# Generate and write to file
def write(output, **kwargs):
    """
    Write <output>_volume.xdmf and <output>_surface.xdmf.
    """
    from dolfinx.io import XDMFFile

    mesh, ct, ft = generate(**kwargs)
    comm = mesh.comm
    out = Path(output)
    if comm.rank == 0:
        out.parent.mkdir(parents=True, exist_ok=True)
    comm.Barrier()

    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    with XDMFFile(comm, out.with_name(out.name + "_volume.xdmf"), "w") as f:
        f.write_mesh(mesh)
        if ct is not None:
            ct.name = "Grid"
            f.write_meshtags(ct, mesh.geometry)
    with XDMFFile(comm, out.with_name(out.name + "_surface.xdmf"), "w") as f:
        f.write_mesh(mesh)
        ft.name = "Grid"
        f.write_meshtags(ft, mesh.geometry)
    return mesh, ct, ft


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--radius", type=float, default=2.0)
    p.add_argument("--length", type=float, default=30.0)
    p.add_argument("--resolution", type=float, default=None)
    p.add_argument("--wall-resolution", type=float, default=None)
    p.add_argument("--output", default="data/tube3d/tube")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    mesh, _, _ = write(
        args.output,
        radius=args.radius,
        length=args.length,
        resolution=args.resolution,
        wall_resolution=args.wall_resolution,
        verbose=args.verbose,
    )
    if mesh.comm.rank == 0:
        n_cells = mesh.topology.index_map(mesh.topology.dim).size_global
        print(
            f"[tube3d] {n_cells} cells; tags wall={WALL_ID} inlet={INLET_ID} outlet={OUTLET_ID}; wrote {args.output}_{{volume,surface}}"
            ".xdmf"
        )


if __name__ == "__main__":
    main()
