"""
Analytical ephemeris for Sun and Moon positions in ECI (J2000).

Sun: Low-precision formula from Meeus/Montenbruck, ~0.01 deg accuracy.
Moon: Simplified Brown theory from Montenbruck & Gill, ~0.1 deg accuracy.

Both are sufficient for SRP direction, lighting geometry, and third-body
perturbation computation at GEO.

References:
    Montenbruck & Gill, "Satellite Orbits", Sec. 3.3.2 (Sun), 3.3.3 (Moon)
    Meeus, "Astronomical Algorithms", Ch. 25
"""

from __future__ import annotations

import numpy as np
from ..core.constants import MJD_J2000, AU_KM, DEG2RAD


def sun_position_eci(mjd_tt: float) -> np.ndarray:
    """Compute Sun position in ECI (J2000) frame.

    Low-precision analytical model. Accuracy ~0.01 deg in ecliptic
    longitude, sufficient for SRP and lighting at GEO.

    Based on the simplified solar coordinates from Montenbruck & Gill,
    "Satellite Orbits", Sec. 3.3.2.

    Args:
        mjd_tt: Modified Julian Date in Terrestrial Time.

    Returns:
        r_sun: Sun ECI position [km], shape (3,).
    """
    # Julian centuries from J2000.0
    T = (mjd_tt - MJD_J2000) / 36525.0

    # Mean anomaly of the Sun [deg]
    M = 357.5256 + 35999.049 * T
    M_rad = np.deg2rad(M % 360.0)

    # Ecliptic longitude of the Sun [deg]
    lam = 280.460 + 36000.770 * T
    lam += 1.9146 * np.sin(M_rad) + 0.0200 * np.sin(2.0 * M_rad)
    lam_rad = np.deg2rad(lam % 360.0)

    # Distance from Earth to Sun [AU]
    r_au = 1.00014 - 0.01671 * np.cos(M_rad) - 0.00014 * np.cos(2.0 * M_rad)

    # Obliquity of the ecliptic [deg]
    eps = 23.4393 - 0.0130 * T
    eps_rad = np.deg2rad(eps)

    # Sun position in ECI (equatorial)
    cos_lam = np.cos(lam_rad)
    sin_lam = np.sin(lam_rad)
    cos_eps = np.cos(eps_rad)
    sin_eps = np.sin(eps_rad)

    r_km = r_au * AU_KM

    x = r_km * cos_lam
    y = r_km * sin_lam * cos_eps
    z = r_km * sin_lam * sin_eps

    return np.array([x, y, z])


def moon_position_eci(mjd_tt: float) -> np.ndarray:
    """Compute Moon position in ECI (J2000) frame.

    Simplified Brown theory from Montenbruck & Gill, "Satellite Orbits",
    Sec. 3.3.3. Accuracy ~0.1-0.3 deg, ~500 km in position.

    Args:
        mjd_tt: Modified Julian Date in Terrestrial Time.

    Returns:
        r_moon: Moon ECI position [km], shape (3,).
    """
    # Julian centuries from J2000.0
    T = (mjd_tt - MJD_J2000) / 36525.0

    # Fundamental arguments [deg]
    # Mean longitude of the Moon
    L0 = 218.3165 + 481267.8813 * T
    # Mean anomaly of the Moon
    l = 134.9634 + 477198.8676 * T
    # Mean anomaly of the Sun
    lp = 357.5291 + 35999.0503 * T
    # Mean elongation of the Moon
    D = 297.8502 + 445267.1115 * T
    # Mean longitude of ascending node
    F = 93.2720 + 483202.0175 * T

    # Convert to radians
    l_r = np.deg2rad(l % 360.0)
    lp_r = np.deg2rad(lp % 360.0)
    D_r = np.deg2rad(D % 360.0)
    F_r = np.deg2rad(F % 360.0)

    # Ecliptic longitude [deg]
    dL = (6.2888 * np.sin(l_r)
          + 1.2740 * np.sin(2.0 * D_r - l_r)
          + 0.6583 * np.sin(2.0 * D_r)
          + 0.2136 * np.sin(2.0 * l_r)
          - 0.1851 * np.sin(lp_r)
          - 0.1143 * np.sin(2.0 * F_r)
          + 0.0588 * np.sin(2.0 * (D_r - l_r))
          + 0.0572 * np.sin(2.0 * D_r - lp_r - l_r)
          + 0.0533 * np.sin(2.0 * D_r + l_r)
          + 0.0459 * np.sin(2.0 * D_r - lp_r)
          + 0.0410 * np.sin(lp_r - l_r)  # original has lp_r sign
          - 0.0348 * np.sin(D_r)
          - 0.0305 * np.sin(lp_r + l_r))

    lam = np.deg2rad((L0 + dL) % 360.0)

    # Ecliptic latitude [deg]
    dB = (5.1282 * np.sin(F_r)
          + 0.2806 * np.sin(l_r + F_r)
          + 0.2777 * np.sin(l_r - F_r)
          + 0.1733 * np.sin(2.0 * D_r - F_r))

    beta = np.deg2rad(dB)

    # Parallax [deg] → distance
    dP = (0.9508
          + 0.0518 * np.cos(l_r)
          + 0.0095 * np.cos(2.0 * D_r - l_r)
          + 0.0078 * np.cos(2.0 * D_r)
          + 0.0028 * np.cos(2.0 * l_r))

    # Earth radii to km: parallax in deg, sin(parallax) ≈ Re/distance
    parallax_rad = np.deg2rad(dP)
    R_EARTH_KM = 6378.137
    r_moon_km = R_EARTH_KM / np.sin(parallax_rad)

    # Position in ecliptic Cartesian
    cos_lam = np.cos(lam)
    sin_lam = np.sin(lam)
    cos_beta = np.cos(beta)
    sin_beta = np.sin(beta)

    x_ecl = r_moon_km * cos_beta * cos_lam
    y_ecl = r_moon_km * cos_beta * sin_lam
    z_ecl = r_moon_km * sin_beta

    # Rotate from ecliptic to equatorial (J2000 obliquity)
    eps = 23.4393 - 0.0130 * T
    eps_rad = np.deg2rad(eps)
    cos_eps = np.cos(eps_rad)
    sin_eps = np.sin(eps_rad)

    x = x_ecl
    y = y_ecl * cos_eps - z_ecl * sin_eps
    z = y_ecl * sin_eps + z_ecl * cos_eps

    return np.array([x, y, z])
