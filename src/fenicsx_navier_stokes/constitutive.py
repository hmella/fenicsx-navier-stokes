"""
Constitutive models for the fluid viscosity.

The validated path is Newtonian (a constant viscosity). PowerLaw is provided and tested but is not wired into the solver, and no example uses it.
"""

import numpy as np
from dolfinx.fem import Constant, Expression, Function
from ufl import grad, inner, sqrt
from ufl.algebra import Power


# Fluid strain rate tensor
def eps(u):
    """
    Symmetric gradient 0.5*(grad(u) + grad(u).T).
    """
    return 0.5 * (grad(u) + grad(u).T)


# Scalar shear rate
def shear_rate(u):
    """
    Returns gamma = sqrt(2*eps(u):eps(u)). For simple shear u = (g*y, 0) this is exactly |g|, which is the normalisation used in the rheology literature.
    """
    e = eps(u)
    return sqrt(2.0 * inner(e, e))


# Constant viscosity
class Newtonian:
    """
    Newtonian fluid with a constant dynamic viscosity 'mu'. A thin wrapper so that a problem can be handed a constitutive object regardless of the rheology in use.
    """

    def __init__(self, mu):
        self.mu = mu

    def viscosity(self, u=None):
        # The velocity is ignored; the argument exists for interface compatibility
        return self.mu

    def __repr__(self):
        return f"Newtonian(mu={self.mu!r})"


# Power-law viscosity
class PowerLaw:
    """
    Generalized-Newtonian viscosity mu = m*gamma^(n-1), with gamma the shear rate.

    n = 1 recovers a Newtonian fluid with mu = m, n < 1 is shear thinning (blood-like) and n > 1 shear thickening. The result is clipped to [m/clamp_ratio, m*clamp_ratio]: this is not cosmetic, since for n < 1 the shear rate vanishes wherever the velocity is locally constant and the formula returns +inf.

    The default n = 1.0 is the Newtonian case, mu = m.
    """

    def __init__(self, u, m=0.04, n=1.0, clamp_ratio=1.0e4):
        if clamp_ratio <= 1.0:
            raise ValueError(f"clamp_ratio must be > 1, got {clamp_ratio}")
        self.u = u
        self.m = m
        self.n = n
        self.clamp_ratio = clamp_ratio
        self.gamma = shear_rate(u)
        self.expr = m * Power(self.gamma, n - 1.0)

    @property
    def bounds(self):
        # Lower and upper clamp, resolving 'm' if it is a Constant
        m = self.m.value if isinstance(self.m, Constant) else self.m
        m = float(np.asarray(m).reshape(-1)[0])
        return m / self.clamp_ratio, m * self.clamp_ratio

    def update(self, u):
        # Copy the degrees of freedom of 'u' into the stored velocity
        self.u.x.array[:] = u.x.array

    def filter(self, mu):
        """
        Clip a viscosity Function in place. Non-finite entries are mapped to the upper bound; comparisons alone would leak them through, since nan > x is False.
        """
        lo, hi = self.bounds
        a = mu.x.array
        a[~np.isfinite(a)] = hi
        np.clip(a, lo, hi, out=a)
        return mu

    def interpolate(self, function_space):
        """
        Interpolate the apparent viscosity into 'function_space' and clamp it.
        """
        mu = Function(function_space)

        # The communicator is passed explicitly: for n == 1 ufl simplifies gamma^(n-1) to the literal 1, leaving an expression with no domain for dolfinx to infer it from
        mu.interpolate(
            Expression(
                self.expr,
                function_space.element.interpolation_points,
                comm=function_space.mesh.comm,
            )
        )
        return self.filter(mu)

    def __repr__(self):
        return f"PowerLaw(m={self.m!r}, n={self.n!r})"


# Power-law viscosity (functional interface)
def powerLaw(u, function_space, m=1.0, n=1.0, clamp_ratio=1.0e4):
    """
    Interpolate a power-law apparent viscosity into 'function_space'. Wrapper around PowerLaw, kept for compatibility with the original script-based code.
    """
    return PowerLaw(u, m=m, n=n, clamp_ratio=clamp_ratio).interpolate(function_space)
