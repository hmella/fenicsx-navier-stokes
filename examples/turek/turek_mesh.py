"""
Generate the DFG 2D flow-around-a-cylinder benchmark mesh with gmsh.

Geometry (Schaefer & Turek, 1996): a channel [0, 2.2] x [0, 0.41] with a circular obstacle of radius 0.05 centred at (0.2, 0.2). The centre is deliberately offset from the channel mid-height (0.2 against 0.205), which is what triggers vortex shedding rather than a symmetric wake.

Facet tags: inlet=2, outlet=3, wall=4 (the channel top and bottom), obstacle=5 (the cylinder). Walls and obstacle are tagged separately so that drag and lift can be integrated over the cylinder alone; both carry no slip, which the configuration expresses as WallID: [4, 5].

The mesh is first-order triangles. The script this was derived from recombined into quadrilaterals and raised the geometry to order 2, which is wrong for the stabilized P1-P1 path: CellDiameter and the elementwise P1 residual both assume an affine simplex.

Usage:
    python examples/turek/turek_mesh.py --res-min 0.00625 --output data/turek/turek
"""

import argparse
from pathlib import Path

import numpy as np
from mpi4py import MPI

# Physical group tags
FLUID_ID = 1
INLET_ID, OUTLET_ID, WALL_ID, OBSTACLE_ID = 2, 3, 4, 5

# Channel length and height, obstacle centre and radius (the benchmark definition)
L, H = 2.2, 0.41
C_X, C_Y, R = 0.2, 0.2, 0.05

# Characteristic length used to non-dimensionalise drag, lift and the Strouhal number
D = 2.0 * R


# Build the mesh
def generate(res_min=None, res_max=None, comm=None, model_rank=0, verbose=False):
    """
    Returns (mesh, cell_tags, facet_tags).

    'res_min' is the element size at the cylinder, defaulting to R/2, which is a smoke-test mesh: the published reference values need something closer to R/40. 'res_max' is the size far from the cylinder, defaulting to 0.25*H.
    """
    import gmsh
    from dolfinx.io.gmsh import model_to_mesh

    comm = comm if comm is not None else MPI.COMM_WORLD
    res_min = res_min if res_min is not None else R / 2.0
    res_max = res_max if res_max is not None else 0.25 * H

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)

        # Under MPI, only one process builds the geometry and meshes it. The others wait and receive their share of the mesh when it is distributed further down.
        if comm.rank == model_rank:
            # Cut the obstacle out of the channel, leaving the fluid region behind
            gmsh.model.add("turek2d")
            rectangle = gmsh.model.occ.addRectangle(0, 0, 0, L, H)
            obstacle = gmsh.model.occ.addDisk(C_X, C_Y, 0, R, R)
            gmsh.model.occ.cut([(2, rectangle)], [(2, obstacle)])
            gmsh.model.occ.synchronize()

            # Mark the fluid region as a gmsh "physical group". Only entities in a physical group are exported, and the group's number becomes the tag that dolfinx sees as a cell or facet marker.
            volumes = gmsh.model.getEntities(dim=2)
            assert len(volumes) == 1, f"expected one fluid surface, got {len(volumes)}"
            gmsh.model.addPhysicalGroup(2, [volumes[0][1]], FLUID_ID)
            gmsh.model.setPhysicalName(2, FLUID_ID, "Fluid")

            # Sort the boundary curves by where their midpoint lies: left edge is the inlet, right edge the outlet, top and bottom the channel walls, and anything else must be the cylinder.
            inflow, outflow, walls, obstacle_edges = [], [], [], []
            for boundary in gmsh.model.getBoundary(volumes, oriented=False):
                com = gmsh.model.occ.getCenterOfMass(boundary[0], boundary[1])
                if np.allclose(com, [0, H / 2, 0]):
                    inflow.append(boundary[1])
                elif np.allclose(com, [L, H / 2, 0]):
                    outflow.append(boundary[1])
                elif np.allclose(com, [L / 2, H, 0]) or np.allclose(com, [L / 2, 0, 0]):
                    walls.append(boundary[1])
                else:
                    obstacle_edges.append(boundary[1])
            assert inflow and outflow and walls and obstacle_edges, "boundary tagging failed"

            # Create the physical curves
            for tag, entities, name in (
                (INLET_ID, inflow, "Inlet"),
                (OUTLET_ID, outflow, "Outlet"),
                (WALL_ID, walls, "Walls"),
                (OBSTACLE_ID, obstacle_edges, "Obstacle"),
            ):
                gmsh.model.addPhysicalGroup(1, entities, tag)
                gmsh.model.setPhysicalName(1, tag, name)

            # Refine toward the cylinder using a distance field, coarsening to res_max over two channel heights:
            #
            # res_max -                  /-------- / res_min -o---------/
            # |         |       | cylinder  DistMin DistMax
            dist = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(dist, "CurvesList", obstacle_edges)
            thr = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(thr, "InField", dist)
            gmsh.model.mesh.field.setNumber(thr, "SizeMin", res_min)
            gmsh.model.mesh.field.setNumber(thr, "SizeMax", res_max)
            gmsh.model.mesh.field.setNumber(thr, "DistMin", R)
            gmsh.model.mesh.field.setNumber(thr, "DistMax", 2 * H)
            gmsh.model.mesh.field.setAsBackgroundMesh(thr)
            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

            # Frontal-Delaunay triangles, order 1. No recombination: quadrilaterals and curved geometry both break the assumptions of the P1 stabilization
            gmsh.option.setNumber("Mesh.Algorithm", 6)
            gmsh.model.mesh.generate(2)
            gmsh.model.mesh.optimize("Netgen")

        data = model_to_mesh(gmsh.model, comm, model_rank, gdim=2)
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
    p.add_argument(
        "--res-min", type=float, default=None, help="element size at the cylinder (default R/2)"
    )
    p.add_argument("--res-max", type=float, default=None)
    p.add_argument("--output", default="data/turek/turek")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    mesh, _, _ = write(
        args.output, res_min=args.res_min, res_max=args.res_max, verbose=args.verbose
    )
    if mesh.comm.rank == 0:
        n = mesh.topology.index_map(mesh.topology.dim).size_global
        print(
            f"[turek] {n} cells; tags inlet={INLET_ID} outlet={OUTLET_ID} wall={WALL_ID} obstacle={OBSTACLE_ID}; wrote {args.output}_*.xdmf"
        )


if __name__ == "__main__":
    main()
