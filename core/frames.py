"""
Reference frame transformations.

Provides rotation matrices and their time derivatives for conversions between:
    - ECI (J2000/GCRF)
    - ECEF (ITRF, simplified)
    - LVLH / RSW (target-centered)

CRITICAL: velocity transformations must account for frame rotation.
"""

from __future__ import annotations

import numpy as np
from ..core.constants import OMEGA_EARTH, MJD_J2000, SECONDS_PER_DAY, TWO_PI
from ..core.types import RelativeState


def _skew(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric cross-product matrix [v×].

    Args:
        v: 3-vector.

    Returns:
        3x3 skew-symmetric matrix such that [v×]·w = v × w.
    """
    return np.array([
        [0., -v[2], v[1]],
        [v[2], 0., -v[0]],
        [-v[1], v[0], 0.]
    ])


def gmst_from_mjd_tt(mjd_tt: float) -> float:
    """Greenwich Mean Sidereal Time from MJD in Terrestrial Time.

    Simplified IAU expression. Accuracy ~0.1 arcsec, sufficient for
    tesseral harmonics and operational longitude context.

    Args:
        mjd_tt: Modified Julian Date in TT.

    Returns:
        GMST in radians, wrapped to [0, 2π).
    """
    # Julian centuries from J2000.0
    T = (mjd_tt - MJD_J2000) / 36525.0

    # GMST in seconds of time (simplified, ignoring UT1-UTC for now)
    gmst_sec = (67310.54841
                + (876600.0 * 3600.0 + 8640184.812866) * T
                + 0.093104 * T**2
                - 6.2e-6 * T**3)

    # Convert to radians
    gmst_rad = (gmst_sec / 86400.0) * TWO_PI
    return gmst_rad % TWO_PI


def eci_to_ecef(mjd_tt: float) -> tuple[np.ndarray, np.ndarray]:
    """ECI (J2000) to ECEF rotation matrix and its time derivative.

    Simplified model using only Earth rotation (no precession/nutation).
    For GEO longitude accuracy, add UT1-UTC correction.

    Args:
        mjd_tt: Epoch in MJD TT.

    Returns:
        R: 3x3 rotation matrix, ECEF = R · ECI.
        dR: 3x3 time derivative of R.
    """
    theta = gmst_from_mjd_tt(mjd_tt)
    c, s = np.cos(theta), np.sin(theta)

    R = np.array([
        [c,  s, 0.],
        [-s, c, 0.],
        [0., 0., 1.]
    ])

    # dR/dt = ω_earth × R expressed as matrix
    dR = OMEGA_EARTH * np.array([
        [-s, c, 0.],
        [-c, -s, 0.],
        [0., 0., 0.]
    ])

    return R, dR


def eci_to_lvlh(r: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Construct LVLH (RSW) frame rotation matrix from ECI state.

    Frame definition:
        R-hat: radial outward (r / |r|)
        C-hat: orbit normal  (r × v / |r × v|)  — cross-track
        I-hat: completes right-hand triad (C × R) — approximately along-track

    Convention: LVLH vector components are ordered [R, I, C] (radial,
    in-track, cross-track) to align with HCW conventions.

    Args:
        r: ECI position vector [km], shape (3,).
        v: ECI velocity vector [km/s], shape (3,).

    Returns:
        R_lvlh: 3x3 rotation matrix, v_LVLH = R_lvlh · v_ECI.
        dR_lvlh: 3x3 time derivative of R_lvlh.
    """
    r_mag = np.linalg.norm(r)
    h = np.cross(r, v)
    h_mag = np.linalg.norm(h)

    # Unit vectors
    r_hat = r / r_mag                   # radial
    c_hat = h / h_mag                   # cross-track (orbit normal)
    i_hat = np.cross(c_hat, r_hat)      # in-track

    # Rotation matrix: rows are the LVLH unit vectors expressed in ECI
    # v_LVLH = R · v_ECI  where R rows = [r_hat, i_hat, c_hat]
    R_lvlh = np.array([r_hat, i_hat, c_hat])

    # Angular velocity of the LVLH frame (= orbit angular velocity)
    omega_lvlh_eci = h / r_mag**2   # ω = h / r² (ECI components)

    # dR/dt = -[ω×] · R  (rotation matrix derivative)
    omega_skew = _skew(omega_lvlh_eci)
    dR_lvlh = -omega_skew @ R_lvlh

    return R_lvlh, dR_lvlh


def relative_state_eci_to_lvlh(
    r_chaser: np.ndarray, v_chaser: np.ndarray,
    r_target: np.ndarray, v_target: np.ndarray,
    epoch_mjd_tt: float
) -> RelativeState:
    """Compute relative state of chaser w.r.t. target in LVLH frame.

    Correctly accounts for frame rotation in the velocity transformation:
        Δv_LVLH = R · (Δv_ECI − ω × Δr_ECI)

    Args:
        r_chaser, v_chaser: Chaser ECI state [km, km/s].
        r_target, v_target: Target ECI state [km, km/s].
        epoch_mjd_tt: Epoch.

    Returns:
        RelativeState in target LVLH frame.
    """
    R_lvlh, _ = eci_to_lvlh(r_target, v_target)

    # Relative position and velocity in ECI
    dr_eci = r_chaser - r_target
    dv_eci = v_chaser - v_target

    # LVLH angular velocity
    h = np.cross(r_target, v_target)
    r_mag = np.linalg.norm(r_target)
    omega_eci = h / r_mag**2

    # Transform to LVLH (accounting for frame rotation)
    dr_lvlh = R_lvlh @ dr_eci
    dv_lvlh = R_lvlh @ (dv_eci - np.cross(omega_eci, dr_eci))

    return RelativeState(
        epoch_mjd_tt=epoch_mjd_tt,
        position_lvlh=dr_lvlh,
        velocity_lvlh=dv_lvlh
    )


def lvlh_to_eci_vector(v_lvlh: np.ndarray,
                        r_target: np.ndarray,
                        v_target: np.ndarray) -> np.ndarray:
    """Transform a vector from LVLH to ECI.

    Useful for converting HCW ΔV vectors to ECI for the corrector.

    Args:
        v_lvlh: Vector in LVLH frame, shape (3,).
        r_target: Target ECI position [km], shape (3,).
        v_target: Target ECI velocity [km/s], shape (3,).

    Returns:
        Vector in ECI frame, shape (3,).
    """
    R_lvlh, _ = eci_to_lvlh(r_target, v_target)
    return R_lvlh.T @ v_lvlh  # R^T = R_inverse for rotation matrices
