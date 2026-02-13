"""
Solar Radiation Pressure — cannonball model.

Provides SRP acceleration and Jacobian for STM integration,
plus cylindrical and conical shadow (eclipse) functions.

The cannonball model assumes a spherical spacecraft with constant
cross-section. The acceleration is:

    a = -nu * Cr * (A/m) * P_sun * (AU/d_sun)^2 * s_hat

where nu is the shadow factor, Cr is reflectivity, A/m is area-to-mass,
P_sun is solar pressure at 1 AU, d_sun is satellite-sun distance, and
s_hat is the satellite-to-sun unit vector.

Reference: Montenbruck & Gill, "Satellite Orbits", Sec. 3.4
"""

from __future__ import annotations

import numpy as np
from ..core.constants import SOLAR_PRESSURE_1AU, AU_KM, R_EARTH
from ..core.types import ShadowModel


def cannonball_srp(r_sat: np.ndarray, r_sun: np.ndarray,
                   cr: float, area_over_mass_m2_kg: float,
                   shadow_factor: float = 1.0
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Cannonball SRP acceleration and Jacobian.

    Args:
        r_sat: Satellite ECI position [km], shape (3,).
        r_sun: Sun ECI position [km], shape (3,).
        cr: Reflectivity coefficient (typically 1.2-1.8).
        area_over_mass_m2_kg: Area-to-mass ratio [m^2/kg].
        shadow_factor: Eclipse factor, 0=full shadow, 1=full sunlight.

    Returns:
        a: SRP acceleration [km/s^2], shape (3,).
        da_dr: Jacobian da/dr w.r.t. satellite position, shape (3,3).
    """
    if shadow_factor <= 0.0:
        return np.zeros(3), np.zeros((3, 3))

    # Vector from satellite to sun
    d = r_sun - r_sat
    d_mag = np.linalg.norm(d)
    if d_mag < 1e-6:
        return np.zeros(3), np.zeros((3, 3))
    d_hat = d / d_mag

    # Solar pressure scaled by inverse-square distance
    # SOLAR_PRESSURE_1AU is in N/m^2; AU_KM in km
    # Acceleration needs to be in km/s^2
    # a = -shadow * Cr * (A/m) * P_1AU * (AU/d)^2 * d_hat
    # P_1AU [N/m^2] = P_1AU [kg/(m*s^2)]
    # A/m [m^2/kg]
    # => Cr * (A/m) * P_1AU gives [1/s^2] ... but we need [km/s^2]
    # d in km, AU in km, so (AU/d)^2 is dimensionless. Good.
    # Cr * (A/m) * P_1AU has units m/s^2; convert to km/s^2: /1000

    p_scale = SOLAR_PRESSURE_1AU * (AU_KM / d_mag) ** 2  # N/m^2 at satellite
    accel_mag = shadow_factor * cr * area_over_mass_m2_kg * p_scale / 1000.0  # km/s^2

    # Direction: push AWAY from sun => along d_hat (sat→sun is d_hat, so
    # force from sun on satellite is along -d_hat for absorption,
    # but SRP pushes satellite away from sun, so a is along -d_hat?
    # Convention: a = -Cr*(A/m)*P * (AU/d)^2 * (-s_hat) where s_hat = r_sun - r_sat
    # Actually: the force is AWAY from sun, so along -(r_sun - r_sat) direction.
    # Wait: radiation comes FROM sun TO satellite.
    # Radiation pressure pushes satellite AWAY from sun.
    # Direction away from sun = -(r_sun - r_sat) / |...| = (r_sat - r_sun) / |...|
    # So a = accel_mag * (r_sat - r_sun) / |r_sat - r_sun| = -accel_mag * d_hat

    a = -accel_mag * d_hat

    # Jacobian da/dr_sat
    # a = -C * (r_sat - r_sun) / |r_sat - r_sun|^3  where C absorbs constants
    # This has the same form as gravity: a = -C * (-d) / d^3 = C * d / d^3
    # Actually: a_i = -accel_0 * d_i / d_mag where accel_0 = shadow*Cr*(A/m)*P*(AU)^2
    # and d = r_sun - r_sat, so dd/dr_sat = -I
    #
    # Let's be precise:
    # a = K * (r_sat - r_sun) / |r_sat - r_sun|^3
    # where K = shadow * Cr * (A/m) * P_1AU * AU^2 / 1000
    #
    # da/dr_sat = K * [I/d^3 - 3*(r_sat-r_sun)(r_sat-r_sun)^T / d^5]
    # But d = r_sun - r_sat, so r_sat - r_sun = -d
    # da/dr_sat = K * [I/d^3 - 3*d*d^T/d^5]  ... same magnitude

    K = shadow_factor * cr * area_over_mass_m2_kg * SOLAR_PRESSURE_1AU * AU_KM ** 2 / 1000.0
    d3 = d_mag ** 3
    d5 = d_mag ** 5

    # a = -K * d / d^3 = K * (r_sat - r_sun) / d^3
    # da/dr_sat = K * (-1) * [I / d^3 - 3 * d * d^T / d^5] * (dd/dr_sat = -I)
    # = K * [I/d^3 - 3*d*d^T/d^5]

    da_dr = K * (np.eye(3) / d3 - 3.0 * np.outer(d, d) / d5)

    return a, da_dr


def shadow_function_cylindrical(r_sat: np.ndarray, r_sun: np.ndarray,
                                r_earth: float = R_EARTH) -> float:
    """Cylindrical shadow model.

    Binary: 0.0 (full shadow) or 1.0 (full sunlight).
    The satellite is in shadow if it is behind the Earth (anti-sun side)
    and its perpendicular distance to the Earth-Sun line is less than
    the Earth radius.

    Args:
        r_sat: Satellite ECI position [km], shape (3,).
        r_sun: Sun ECI position [km], shape (3,).
        r_earth: Earth radius [km].

    Returns:
        Shadow factor: 0.0 or 1.0.
    """
    sun_hat = r_sun / np.linalg.norm(r_sun)

    # Project satellite position along sun direction
    along = np.dot(r_sat, sun_hat)

    # Must be on anti-sun side
    if along >= 0.0:
        return 1.0

    # Perpendicular distance to Earth-Sun line
    perp = r_sat - along * sun_hat
    if np.linalg.norm(perp) < r_earth:
        return 0.0
    return 1.0


def shadow_function_conical(r_sat: np.ndarray, r_sun: np.ndarray,
                            r_earth: float = R_EARTH,
                            r_sun_radius_km: float = 696000.0
                            ) -> float:
    """Conical (penumbra/umbra) shadow model.

    Models the smooth penumbral transition using the apparent angular
    sizes of the Sun and Earth as seen from the satellite.

    Args:
        r_sat: Satellite ECI position [km], shape (3,).
        r_sun: Sun ECI position [km], shape (3,).
        r_earth: Earth equatorial radius [km].
        r_sun_radius_km: Solar radius [km].

    Returns:
        Shadow factor in [0, 1]. 1=full sun, 0=full umbra.
    """
    d_sun = np.linalg.norm(r_sun - r_sat)
    d_earth = np.linalg.norm(r_sat)

    if d_sun < 1.0 or d_earth < 1.0:
        return 1.0

    # Apparent angular radii
    alpha_sun = np.arcsin(min(r_sun_radius_km / d_sun, 1.0))
    alpha_earth = np.arcsin(min(r_earth / d_earth, 1.0))

    # Angular separation between Sun and Earth centers as seen from satellite
    s_dir = r_sun - r_sat
    e_dir = -r_sat
    cos_sep = np.dot(s_dir, e_dir) / (d_sun * d_earth)
    sep = np.arccos(np.clip(cos_sep, -1.0, 1.0))

    # No eclipse
    if sep >= alpha_sun + alpha_earth:
        return 1.0

    # Full umbra
    if sep <= alpha_earth - alpha_sun:
        return 0.0

    # Penumbral transition — linear interpolation
    pen_width = 2.0 * alpha_sun
    if pen_width < 1e-15:
        return float(sep > alpha_earth)

    frac = (sep - (alpha_earth - alpha_sun)) / pen_width
    return np.clip(frac, 0.0, 1.0)


def compute_shadow_factor(r_sat: np.ndarray, r_sun: np.ndarray,
                          r_earth: float,
                          model: ShadowModel) -> float:
    """Compute shadow factor using the configured model.

    Args:
        r_sat: Satellite ECI position [km].
        r_sun: Sun ECI position [km].
        r_earth: Earth radius [km].
        model: Shadow model type.

    Returns:
        Shadow factor in [0, 1].
    """
    if model == ShadowModel.NONE:
        return 1.0
    elif model == ShadowModel.CYLINDRICAL:
        return shadow_function_cylindrical(r_sat, r_sun, r_earth)
    elif model == ShadowModel.CONICAL:
        return shadow_function_conical(r_sat, r_sun, r_earth)
    return 1.0
