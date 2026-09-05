"""
SUPG/PSPG/grad-div stabilization parameters.

Equal-order (P1-P1) velocity-pressure pairs do not satisfy the inf-sup condition, so the momentum and continuity equations are augmented with terms weighting the strong residual. Because that residual vanishes for the exact solution, the added terms do not change the order of accuracy of the underlying method.
"""

from ufl import CellDiameter, dot, sqrt


# Viscous coefficient of the SUPG parameter
def Ck_constant(degree=1):
    """
    Returns Ck = 60*2^(k-2) for a velocity space of polynomial degree k. Gives 30 for P1 and 60 for P2.
    """
    return 60.0 * 2.0 ** (degree - 2)


# SUPG/PSPG stabilization parameter
def tau_M(u_conv, dt, rho, mu, mesh=None, h=None, sigma_BDF=2.0, degree=1):
    """
    Build the stabilization parameter

        tau_M = ((sigma/dt)^2 + (|u|/h)^2 + Ck*(mu/(rho*h^2))^2)^(-1/2)

    as a ufl expression. 'u_conv' is the convecting velocity (the previous Picard iterate), and 'h' defaults to the element diameter of 'mesh'.

    The three terms have units of 1/time, so tau_M has units of time. Note the viscous term uses the kinematic viscosity mu/rho: writing it with the dynamic viscosity is dimensionally inconsistent, and stays hidden whenever the transient term dominates.

    'sigma_BDF' is a free parameter of the stabilization (2.0 is the usual Tezduyar value). Since tau_M multiplies a residual that vanishes on the exact solution, changing it cannot affect consistency or the convergence order, only the size of the added terms.
    """
    # Element size
    if h is None:
        if mesh is None:
            raise ValueError("tau_M requires either an element size 'h' or a 'mesh'")
        h = CellDiameter(mesh)

    # Velocity norm and viscous coefficient
    vnorm = sqrt(dot(u_conv, u_conv))
    Ck = Ck_constant(degree)

    # Transient + convective + viscous contributions
    return ((sigma_BDF / dt) ** 2 + (vnorm / h) ** 2 + Ck * (mu / (rho * h**2)) ** 2) ** (-0.5)


# Grad-div (LSIC) stabilization parameter
def tau_C(rho, mu, coefficient=0.4):
    """
    Build tau_C = coefficient*mu/rho, so that tau_C*rho*div(u)*div(v)*dx has the same units as the viscous term.

    This is a bulk-viscosity form: tau_C*rho = 0.4*mu does not scale with the local velocity or the mesh size, unlike the usual choices tau_M*|u|^2 or h^2/tau_M. Its ability to enforce discrete mass conservation therefore weakens as the Reynolds number grows.
    """
    return coefficient * mu / rho
