"""
Imaging geometry calculations.

Computes the geometric relationships between Sun, target, and chaser
that determine imaging feasibility and quality:
    - Phase angle (Sun-Target-Chaser)
    - Solar illumination state (sunlit/eclipse)
    - Target angular size at range
    - Sun-sensor angle for stray light assessment

All position inputs are ECI [km].
"""

from __future__ import annotations

import numpy as np
from ..core.constants import AU_KM, R_EARTH


def phase_angle(r_sun_eci: np.ndarray,
                r_target_eci: np.ndarray,
                r_chaser_eci: np.ndarray) -> float:
    """Sun-Target-Chaser phase angle.

    The angle at the target vertex between the Sun direction and the
    chaser direction. Phase angle of 0 = target is backlit (sun behind
    chaser, full illumination). Phase angle of pi = target between sun
    and chaser (silhouette).

    For resolved imaging, favorable phase angles are typically 20-120 deg.

    Args:
        r_sun_eci: Sun ECI position [km], shape (3,).
        r_target_eci: Target ECI position [km], shape (3,).
        r_chaser_eci: Chaser ECI position [km], shape (3,).

    Returns:
        Phase angle [rad] in [0, pi].
    """
    # Vectors from target to sun and from target to chaser
    to_sun = r_sun_eci - r_target_eci
    to_chaser = r_chaser_eci - r_target_eci

    sun_mag = np.linalg.norm(to_sun)
    chaser_mag = np.linalg.norm(to_chaser)

    if sun_mag < 1e-10 or chaser_mag < 1e-10:
        return 0.0

    cos_alpha = np.dot(to_sun, to_chaser) / (sun_mag * chaser_mag)
    cos_alpha = np.clip(cos_alpha, -1.0, 1.0)
    return np.arccos(cos_alpha)


def solar_incidence_angle(r_sun_eci: np.ndarray,
                           r_target_eci: np.ndarray) -> float:
    """Angle between the sun direction and the target's position vector.

    Relevant for understanding illumination geometry relative to the
    target's orbital position.

    Args:
        r_sun_eci: Sun ECI position [km].
        r_target_eci: Target ECI position [km].

    Returns:
        Incidence angle [rad].
    """
    to_sun = r_sun_eci - r_target_eci
    sun_mag = np.linalg.norm(to_sun)
    tgt_mag = np.linalg.norm(r_target_eci)

    if sun_mag < 1e-10 or tgt_mag < 1e-10:
        return 0.0

    cos_angle = np.dot(to_sun / sun_mag, r_target_eci / tgt_mag)
    return np.arccos(np.clip(cos_angle, -1.0, 1.0))


def is_target_sunlit_cylindrical(r_target_eci: np.ndarray,
                                  r_sun_eci: np.ndarray,
                                  r_earth: float = R_EARTH) -> bool:
    """Check if the target is in sunlight using a cylindrical shadow model.

    Simple and fast. The target is in shadow if it is behind the Earth
    (relative to the Sun) and within the Earth's shadow cylinder.

    Args:
        r_target_eci: Target ECI position [km], shape (3,).
        r_sun_eci: Sun ECI position [km], shape (3,).
        r_earth: Earth radius [km].

    Returns:
        True if target is sunlit (not in shadow).
    """
    # Sun direction unit vector
    sun_dir = r_sun_eci / np.linalg.norm(r_sun_eci)

    # Component of target position along sun direction
    along_sun = np.dot(r_target_eci, sun_dir)

    # Target must be on the anti-sun side of Earth to be in shadow
    if along_sun > 0:
        return True  # On the sun-side, always lit

    # Perpendicular distance from the target to the sun-Earth line
    perp = r_target_eci - along_sun * sun_dir
    perp_dist = np.linalg.norm(perp)

    # In shadow if perpendicular distance < Earth radius
    return perp_dist > r_earth


def is_target_sunlit_conical(r_target_eci: np.ndarray,
                              r_sun_eci: np.ndarray,
                              r_earth: float = R_EARTH,
                              r_sun_radius_km: float = 696000.0
                              ) -> tuple[bool, float]:
    """Check if target is sunlit using conical (penumbra/umbra) model.

    Returns both a boolean and a shadow fraction for smooth transitions.

    Args:
        r_target_eci: Target ECI position [km].
        r_sun_eci: Sun ECI position [km].
        r_earth: Earth equatorial radius [km].
        r_sun_radius_km: Solar radius [km].

    Returns:
        is_sunlit: True if any sunlight reaches the target.
        shadow_fraction: 1.0 = full sun, 0.0 = full umbra,
            intermediate = penumbra.
    """
    r_sat = r_target_eci
    r_sun = r_sun_eci

    # Apparent angular radii as seen from the satellite
    d_sun = np.linalg.norm(r_sun - r_sat)
    d_earth = np.linalg.norm(r_sat)  # Earth is at origin

    if d_sun < 1e-10 or d_earth < 1e-10:
        return True, 1.0

    alpha_sun = np.arcsin(min(r_sun_radius_km / d_sun, 1.0))
    alpha_earth = np.arcsin(min(r_earth / d_earth, 1.0))

    # Angle between sun and Earth centers as seen from satellite
    sun_dir = (r_sun - r_sat)
    earth_dir = -r_sat
    cos_sep = np.dot(sun_dir, earth_dir) / (d_sun * d_earth)
    cos_sep = np.clip(cos_sep, -1.0, 1.0)
    sep = np.arccos(cos_sep)

    # No eclipse if Earth is not between satellite and sun
    if sep > alpha_sun + alpha_earth:
        return True, 1.0

    # Full umbra
    if sep < alpha_earth - alpha_sun:
        return False, 0.0

    # Penumbral transition: linear interpolation for simplicity
    # Full sun when sep = alpha_sun + alpha_earth
    # Full shadow when sep = alpha_earth - alpha_sun
    penumbra_width = 2.0 * alpha_sun
    if penumbra_width < 1e-15:
        return sep > alpha_earth, float(sep > alpha_earth)

    fraction = (sep - (alpha_earth - alpha_sun)) / penumbra_width
    fraction = np.clip(fraction, 0.0, 1.0)

    return fraction > 0.0, fraction


def sun_sensor_angle(r_chaser_eci: np.ndarray,
                      r_sun_eci: np.ndarray,
                      boresight_eci: np.ndarray) -> float:
    """Angle between the sensor boresight and the Sun direction.

    Used for stray light assessment. Imaging is typically precluded when
    the sun is within the baffle exclusion zone.

    Args:
        r_chaser_eci: Chaser ECI position [km].
        r_sun_eci: Sun ECI position [km].
        boresight_eci: Sensor boresight unit vector in ECI, shape (3,).

    Returns:
        Angle between boresight and sun direction [rad].
    """
    sun_dir = r_sun_eci - r_chaser_eci
    sun_mag = np.linalg.norm(sun_dir)
    if sun_mag < 1e-10:
        return 0.0

    cos_angle = np.dot(boresight_eci, sun_dir / sun_mag)
    return np.arccos(np.clip(cos_angle, -1.0, 1.0))


def target_heliocentric_distance_au(r_target_eci: np.ndarray,
                                     r_sun_eci: np.ndarray) -> float:
    """Target's distance from the Sun in AU.

    For GEO targets this is always ~1 AU, but included for generality.

    Args:
        r_target_eci: Target ECI position [km].
        r_sun_eci: Sun ECI position [km].

    Returns:
        Distance [AU].
    """
    d_km = np.linalg.norm(r_sun_eci - r_target_eci)
    return d_km / AU_KM


def compute_geometry_at_epoch(r_chaser_eci: np.ndarray,
                               r_target_eci: np.ndarray,
                               r_sun_eci: np.ndarray
                               ) -> dict:
    """Compute all imaging geometry quantities at a single epoch.

    Convenience function that packages phase angle, range, shadow state,
    and heliocentric distance into a dict.

    Args:
        r_chaser_eci: Chaser position [km], shape (3,).
        r_target_eci: Target position [km], shape (3,).
        r_sun_eci: Sun position [km], shape (3,).

    Returns:
        Dict with keys:
            'range_km', 'phase_angle_rad', 'target_sunlit',
            'shadow_fraction', 'helio_distance_au'
    """
    rel = r_target_eci - r_chaser_eci
    rng = np.linalg.norm(rel)

    pa = phase_angle(r_sun_eci, r_target_eci, r_chaser_eci)
    sunlit, shadow_frac = is_target_sunlit_conical(r_target_eci, r_sun_eci)
    helio_au = target_heliocentric_distance_au(r_target_eci, r_sun_eci)

    return {
        'range_km': rng,
        'phase_angle_rad': pa,
        'target_sunlit': sunlit,
        'shadow_fraction': shadow_frac,
        'helio_distance_au': helio_au,
    }
