"""
Third-body gravitational perturbations (Sun and Moon).

Uses the standard formulation that avoids loss of significance:
    a = -mu_body * [ (r_sat - r_body)/|r_sat - r_body|^3 + r_body/|r_body|^3 ]

The first term is the direct attraction of the satellite by the body.
The second term is the indirect acceleration (the central body, Earth,
is also accelerated by the third body; since we integrate in an
Earth-centered frame, this must be subtracted).

Reference: Montenbruck & Gill, "Satellite Orbits", Eq. 3.57
"""

from __future__ import annotations

import numpy as np
from ..core.constants import MU_SUN, MU_MOON


def third_body_acceleration(r_sat: np.ndarray, r_body: np.ndarray,
                            mu_body: float
                            ) -> tuple[np.ndarray, np.ndarray]:
    """Point-mass third-body gravitational perturbation acceleration.

    Args:
        r_sat: Satellite ECI position [km], shape (3,).
        r_body: Third body ECI position [km], shape (3,).
        mu_body: Third body gravitational parameter [km^3/s^2].

    Returns:
        a: Perturbation acceleration [km/s^2], shape (3,).
        da_dr: Jacobian da/dr w.r.t. satellite position, shape (3,3).
    """
    # Vector from satellite to body
    d = r_body - r_sat
    d_mag = np.linalg.norm(d)
    d3 = d_mag ** 3

    # Vector from Earth center to body
    rb_mag = np.linalg.norm(r_body)
    rb3 = rb_mag ** 3

    # Acceleration: a = mu * [d/|d|^3 - r_body/|r_body|^3]
    a = mu_body * (d / d3 - r_body / rb3)

    # Jacobian da/dr_sat
    # Only the first term depends on r_sat (the indirect term is constant).
    # d(d/|d|^3)/dr_sat = d(-d)/dr_sat ... d = r_body - r_sat, dd/dr_sat = -I
    # d(d_i/d^3)/dr_sat_j = (-delta_ij * d^3 - d_i * 3*d^2 * dd/dr_sat_j * d_hat_j ...)
    # = (-delta_ij / d^3 + 3*d_i*d_j/d^5) * (-1) ... from chain rule with dd/dr = -I
    #
    # Simpler: the direct term is f(d) = mu*d/|d|^3 where d = r_body - r_sat
    # df/dr_sat = df/dd * dd/dr_sat = mu * [I/d^3 - 3*d*d^T/d^5] * (-I)
    #           = -mu * [I/d^3 - 3*d*d^T/d^5]

    d5 = d_mag ** 5
    da_dr = -mu_body * (np.eye(3) / d3 - 3.0 * np.outer(d, d) / d5)

    return a, da_dr


def solar_gravity(r_sat: np.ndarray, r_sun: np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Solar gravitational perturbation.

    Args:
        r_sat: Satellite ECI position [km].
        r_sun: Sun ECI position [km].

    Returns:
        Acceleration [km/s^2] and Jacobian.
    """
    return third_body_acceleration(r_sat, r_sun, MU_SUN)


def lunar_gravity(r_sat: np.ndarray, r_moon: np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Lunar gravitational perturbation.

    Args:
        r_sat: Satellite ECI position [km].
        r_moon: Moon ECI position [km].

    Returns:
        Acceleration [km/s^2] and Jacobian.
    """
    return third_body_acceleration(r_sat, r_moon, MU_MOON)
