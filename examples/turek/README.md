# DFG flow around a cylinder (Turek benchmark)

Channel `[0, 2.2] × [0, 0.41]` with a cylinder of radius 0.05 at `(0.2, 0.2)`. The centre is offset from the mid-height (0.2 against 0.205), which is what triggers shedding rather than a symmetric wake.

```bash
python run.py --case 2d3 --res-min 0.00625 --dt 0.005
python turek_mesh.py --res-min 0.00625 --output ../../data/turek/turek   # mesh only
```

| case | inflow | Re | notes |
|---|---|---|---|
| `2d1` | steady, U = 0.3 | 20 | driven to steady state by time marching |
| `2d2` | steady, U = 1.5 | 100 | needs 25–30 s for a developed vortex street |
| `2d3` | `1.5 sin(πt/8)`, t ∈ [0,8] | ≤100 | the default; starts from rest |

## Reference values (case 2D-3)

V. John, *Int. J. Numer. Meth. Fluids* **44** (2004) 777–788:

| quantity | value | at |
|---|---|---|
| `c_D,max` | 2.950921575 | t = 3.93625 |
| `c_L,max` | 0.47795 | t = 5.693125 |
| `Δp(8 s)` | −0.1116 | |

Featflow reports Δp with the opposite sign; the magnitudes agree. Compare `|Δp|` and check the sign against John's `p_front − p_back` convention separately.

## What it actually costs

Reaching those values is a multi-million-dof, multi-hour computation. A *first-order* Crouzeix–Raviart discretisation needed ~785k dofs and Δt = 0.0025 to land within 0.8 % on drag and 9 % on Δp, and stabilized P1-P1 is not a better element than CR. The default mesh (`res_min = R/2`, ~700 cells) is a smoke-test mesh and will not come close.

## Drag and lift

Computed two ways, both reported:

- **variational** (`cD`, `cL`) — the discrete reaction force, evaluating the full residual against a test function equal to a unit vector on the cylinder. Trust this one: it inherits the Galerkin orthogonality of the discretization and converges roughly an order faster than a boundary flux.
- **surface** (`cD_surface`, `cL_surface`) — the traction integral, as a cross-check. The two differ only through the discrete mass-conservation defect.

## Pressure probes

The benchmark evaluates Δp at `(0.15, 0.2)` and `(0.25, 0.2)`, which lie exactly on the cylinder. A first-order mesh inscribes the circle, so whether a probe is located depends on where mesh vertices fall; the script nudges them inward by 1e-6 and the evaluator raises rather than returning a silent `nan`.
