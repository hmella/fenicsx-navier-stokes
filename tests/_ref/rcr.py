"""Analytic reference solutions for the three-element (RCR) Windkessel model.

The model is

.. math::

    C\\,\\frac{\\mathrm{d}P_d}{\\mathrm{d}t} = Q(t) - \\frac{P_d}{R_d}, \\qquad P_{out} = R_p Q + P_d ,

which in the dimensionless time :math:`\\tau = t/(R_d C)` becomes :math:`\\mathrm{d}P_d/\\mathrm{d}\\tau = R_d Q - P_d`.
"""

import numpy as np

__all__ = ["p_out_constant_Q", "p_out_sin2"]


def p_out_constant_Q(t, Q0, Rd, Rp, C, Pd0=0.0):
    """Outlet pressure for a constant flow rate ``Q0``.

    The distal pressure relaxes exponentially toward ``Rd*Q0`` with time constant ``Rd*C``:

    .. math:: P_d(t) = R_d Q_0 + (P_{d,0} - R_d Q_0)\\,e^{-t/(R_d C)} .
    """
    t = np.asarray(t, dtype=float)
    Pd = Rd * Q0 + (Pd0 - Rd * Q0) * np.exp(-t / (Rd * C))
    return Rp * Q0 + Pd


def p_out_sin2(t, Q0, Rd, Rp, C):
    """Outlet pressure for the drive :math:`Q(t) = Q_0 \\sin^2(\\tau/2)`, with
    :math:`P_d(0) = 0`.

    .. math::

        P_{out} = R_d Q_0 \\left[
            \\left(\\frac{R_p}{R_d} + \\frac12\\right)\\sin^2\\frac{\\tau}{2}
            + \\frac14\\bigl(1 - e^{-\\tau} - \\sin\\tau\\bigr)\\right],
        \\qquad \\tau = \\frac{t}{R_d C}.

    Notes
    -----
    This is the reference the shipped ``data/inflow/tube.flow`` waveform was built for:
    that file is :math:`\\sin^2(2\\pi t)` and the tube configuration uses :math:`R_d C = 1/(4\\pi)`, so :math:`\\sin^2(\\tau/2) = \\sin^2(2\\pi t)` exactly.
    """
    tau = np.asarray(t, dtype=float) / (Rd * C)
    return (
        Rd
        * Q0
        * ((Rp / Rd + 0.5) * np.sin(tau / 2.0) ** 2 + 0.25 * (1.0 - np.exp(-tau) - np.sin(tau)))
    )
