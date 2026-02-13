"""
Gravitational acceleration models.

Each function returns the acceleration vector AND its Jacobian (da/dr)
for STM integration. The Jacobian is the 3x3 matrix of partial derivatives
of acceleration with respect to position.

All accelerations computed in ECI (J2000).

J2 uses a fully analytical Jacobian. J3, J4, and J22 tesseral use the
acceleration from a Legendre/Cartesian formulation with finite-difference
Jacobians (O(h^2) ~ 1e-20 at GEO, negligible error for STM integration).

References:
    Montenbruck & Gill, "Satellite Orbits", Ch. 3
    Vallado, "Fundamentals of Astrodynamics and Applications", Ch. 8
"""

from __future__ import annotations

import numpy as np
from ..core.constants import MU_EARTH, R_EARTH, J2, J3, J4, C22, S22
from ..core.config import ForceModelConfig


# ===================================================================
# Two-body
# ===================================================================

def two_body(r: np.ndarray, mu: float = MU_EARTH
             ) -> tuple[np.ndarray, np.ndarray]:
    """Central body point-mass gravitational acceleration.

    Args:
        r: Position vector [km], shape (3,).
        mu: Gravitational parameter [km^3/s^2].

    Returns:
        a: Acceleration vector [km/s^2], shape (3,).
        da_dr: Jacobian da/dr, shape (3,3).
    """
    r_mag = np.linalg.norm(r)
    r3 = r_mag ** 3

    a = -mu * r / r3
    da_dr = -mu / r3 * (np.eye(3) - 3.0 * np.outer(r, r) / r_mag ** 2)

    return a, da_dr


# ===================================================================
# J2 — analytical acceleration + analytical Jacobian
# ===================================================================

def zonal_j2(r: np.ndarray, mu: float = MU_EARTH,
             re: float = R_EARTH, j2: float = J2
             ) -> tuple[np.ndarray, np.ndarray]:
    """J2 zonal harmonic perturbation acceleration and analytical Jacobian.

    Uses the Cartesian form:
        a_i = -(3/2)*mu*J2*re^2 * r_i / r^5 * f_i
    where f_xy = (5*z^2/r^2 - 1), f_z = (5*z^2/r^2 - 3).

    The Jacobian is derived analytically using the chain rule on these
    expressions — critical for accurate STM integration.

    Args:
        r: ECI position [km], shape (3,).
        mu: Gravitational parameter [km^3/s^2].
        re: Earth equatorial radius [km].
        j2: J2 coefficient.

    Returns:
        a: Perturbation acceleration [km/s^2], shape (3,).
        da_dr: Jacobian da/dr, shape (3,3).
    """
    x, y, z = r
    r_mag = np.linalg.norm(r)
    r2 = r_mag ** 2
    r5 = r_mag ** 5
    r7 = r_mag ** 7
    z2 = z ** 2
    S = z2 / r2  # sin^2(latitude)

    k = 1.5 * mu * j2 * re ** 2

    # Acceleration
    fxy = 5.0 * S - 1.0
    fz = 5.0 * S - 3.0

    a = np.array([
        -k * x / r5 * fxy,
        -k * y / r5 * fxy,
        -k * z / r5 * fz,
    ])

    # Analytical Jacobian
    # a_i = -k * r_i / r^5 * f_i(S)
    # where S = z^2/r^2
    # dS/dr_j = -2*z^2*r_j/r^4  for j=x,y;  dS/dz = 2*z*(r^2-z^2)/r^4

    dS = np.array([
        -2.0 * z2 * x / (r2 * r2),
        -2.0 * z2 * y / (r2 * r2),
        2.0 * z * (r2 - z2) / (r2 * r2),
    ])

    da_dr = np.zeros((3, 3))
    for i in range(3):
        f_i = fxy if i < 2 else fz
        for j in range(3):
            delta_ij = 1.0 if i == j else 0.0
            da_dr[i, j] = -k * (
                delta_ij / r5 * f_i
                + r[i] * (-5.0 * r[j] / r7) * f_i
                + r[i] / r5 * 5.0 * dS[j]
            )

    return a, da_dr


# ===================================================================
# J3, J4 — Legendre recursion acceleration + finite-diff Jacobian
# ===================================================================

def zonal_j3(r: np.ndarray, mu: float = MU_EARTH,
             re: float = R_EARTH, j3: float = J3
             ) -> tuple[np.ndarray, np.ndarray]:
    """J3 zonal harmonic perturbation acceleration and Jacobian.

    Args:
        r: ECI position [km], shape (3,).
        mu: Gravitational parameter [km^3/s^2].
        re: Earth equatorial radius [km].
        j3: J3 coefficient.

    Returns:
        a: Perturbation acceleration [km/s^2], shape (3,).
        da_dr: Jacobian da/dr, shape (3,3).
    """
    return _zonal_with_fd_jacobian(r, mu, re, j3, n=3)


def zonal_j4(r: np.ndarray, mu: float = MU_EARTH,
             re: float = R_EARTH, j4: float = J4
             ) -> tuple[np.ndarray, np.ndarray]:
    """J4 zonal harmonic perturbation acceleration and Jacobian.

    Args:
        r: ECI position [km], shape (3,).
        mu: Gravitational parameter [km^3/s^2].
        re: Earth equatorial radius [km].
        j4: J4 coefficient.

    Returns:
        a: Perturbation acceleration [km/s^2], shape (3,).
        da_dr: Jacobian da/dr, shape (3,3).
    """
    return _zonal_with_fd_jacobian(r, mu, re, j4, n=4)


def _zonal_with_fd_jacobian(r, mu, re, jn, n):
    """Zonal acceleration + central finite-difference Jacobian.

    Args:
        r: Position [km], shape (3,).
        mu, re, jn, n: Harmonic parameters.

    Returns:
        a, da_dr.
    """
    a = _zonal_accel(r, mu, re, jn, n)

    h = max(1e-8 * np.linalg.norm(r), 1e-10)
    da_dr = np.zeros((3, 3))
    for j in range(3):
        rp = r.copy()
        rm = r.copy()
        rp[j] += h
        rm[j] -= h
        da_dr[:, j] = (_zonal_accel(rp, mu, re, jn, n)
                       - _zonal_accel(rm, mu, re, jn, n)) / (2.0 * h)
    return a, da_dr


def _zonal_accel(r, mu, re, jn, n):
    """Zonal harmonic acceleration in Cartesian via Legendre recursion.

    Gradient of U_n = -(mu/r)(re/r)^n * Jn * P_n(sin phi).

    The gradient is computed in spherical components then converted
    to Cartesian ECI.

    Args:
        r: Position [km], shape (3,).
        mu, re, jn, n: Parameters.

    Returns:
        Acceleration [km/s^2], shape (3,).
    """
    x, y, z = r
    r_mag = np.linalg.norm(r)
    r2 = r_mag ** 2
    s = z / r_mag  # sin(geocentric latitude)

    Pn = _legendre_P(n, s)
    dPn = _legendre_dP(n, s)

    re_over_r_n = (re / r_mag) ** n
    coeff = mu / r2 * re_over_r_n * jn

    # Radial gradient: (n+1)*Pn
    dudr = coeff * (n + 1.0) * Pn

    # Latitudinal gradient: -cos(phi)*dPn/ds
    cos_phi = np.sqrt(max(1.0 - s * s, 0.0))
    dudlat = -coeff * cos_phi * dPn

    # Convert to Cartesian
    rho = np.sqrt(x * x + y * y)

    ax = -dudr * x / r_mag
    ay = -dudr * y / r_mag
    az = -dudr * z / r_mag

    if rho > 1e-15:
        lat_fac = dudlat / r2
        ax += z * x / (rho * r2) * dudlat  # sign: -(-z*x/(r^2*rho)) * dudlat
        ay += z * y / (rho * r2) * dudlat
        az -= rho / r2 * dudlat

    return np.array([ax, ay, az])


def _legendre_P(n, s):
    """Legendre polynomial P_n(s) via recursion."""
    if n == 0:
        return 1.0
    if n == 1:
        return s
    P_prev, P_curr = 1.0, s
    for k in range(2, n + 1):
        P_next = ((2 * k - 1) * s * P_curr - (k - 1) * P_prev) / k
        P_prev, P_curr = P_curr, P_next
    return P_curr


def _legendre_dP(n, s):
    """Derivative dP_n/ds via the identity n*(s*Pn - P_{n-1})/(s^2-1)."""
    if n == 0:
        return 0.0
    if n == 1:
        return 1.0
    Pn = _legendre_P(n, s)
    Pn1 = _legendre_P(n - 1, s)
    denom = s * s - 1.0
    if abs(denom) < 1e-14:
        return n * (n + 1.0) / 2.0 * (1.0 if s > 0 else (-1.0) ** (n + 1))
    return n * (s * Pn - Pn1) / denom


# ===================================================================
# J22 tesseral
# ===================================================================

def tesseral_j22(r: np.ndarray, gmst: float,
                 mu: float = MU_EARTH, re: float = R_EARTH,
                 c22: float = C22, s22: float = S22
                 ) -> tuple[np.ndarray, np.ndarray]:
    """J22 tesseral harmonic perturbation acceleration.

    Dominant tesseral term at GEO. Computed analytically in ECEF
    then rotated to ECI. Jacobian via finite differencing in ECEF.

    Potential in ECEF Cartesian:
        U_22 = 3*mu*re^2/r^5 * (C22*(x^2-y^2) + 2*S22*x*y)

    Args:
        r: ECI position [km], shape (3,).
        gmst: Greenwich Mean Sidereal Time [rad].
        mu, re, c22, s22: Parameters.

    Returns:
        a_eci: Acceleration in ECI [km/s^2], shape (3,).
        da_dr_eci: Jacobian in ECI, shape (3,3).
    """
    cg, sg = np.cos(gmst), np.sin(gmst)
    R_ei = np.array([[cg, sg, 0.], [-sg, cg, 0.], [0., 0., 1.]])
    R_ie = R_ei.T

    r_ef = R_ei @ r
    a_ef = _tesseral_j22_accel_ecef(r_ef, mu, re, c22, s22)

    # Finite-diff Jacobian in ECEF
    h = max(1e-8 * np.linalg.norm(r), 1e-10)
    da_ef = np.zeros((3, 3))
    for j in range(3):
        rp, rm = r_ef.copy(), r_ef.copy()
        rp[j] += h
        rm[j] -= h
        da_ef[:, j] = (_tesseral_j22_accel_ecef(rp, mu, re, c22, s22)
                       - _tesseral_j22_accel_ecef(rm, mu, re, c22, s22)) / (2.0 * h)

    return R_ie @ a_ef, R_ie @ da_ef @ R_ei


def _tesseral_j22_accel_ecef(r_bf, mu, re, c22, s22):
    """J22 acceleration in ECEF (analytical gradient of U_22).

    U_22 = 3*mu*re^2/r^5 * Q,  Q = C22*(x^2-y^2) + 2*S22*x*y

    Args:
        r_bf: ECEF position [km], shape (3,).

    Returns:
        Acceleration [km/s^2], shape (3,).
    """
    x, y, z = r_bf
    r_mag = np.linalg.norm(r_bf)
    r5 = r_mag ** 5
    r7 = r_mag ** 7
    k = 3.0 * mu * re ** 2
    Q = c22 * (x ** 2 - y ** 2) + 2.0 * s22 * x * y

    ax = k * (-5.0 * x * Q / r7 + (2.0 * c22 * x + 2.0 * s22 * y) / r5)
    ay = k * (-5.0 * y * Q / r7 + (-2.0 * c22 * y + 2.0 * s22 * x) / r5)
    az = k * (-5.0 * z * Q / r7)
    return np.array([ax, ay, az])


# ===================================================================
# Combined gravity assembler
# ===================================================================

def combined_gravity(r: np.ndarray, epoch_mjd_tt: float,
                     config: ForceModelConfig
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Assemble total gravitational acceleration from all enabled models.

    Args:
        r: ECI position [km], shape (3,).
        epoch_mjd_tt: Current epoch for Earth rotation (tesserals).
        config: Force model configuration with toggles.

    Returns:
        a_total: Total gravitational acceleration [km/s^2], shape (3,).
        da_dr_total: Total Jacobian da/dr, shape (3,3).
    """
    a_total, da_dr_total = two_body(r)

    if config.enable_j2:
        a, da = zonal_j2(r)
        a_total = a_total + a
        da_dr_total = da_dr_total + da

    if config.enable_j3:
        a, da = zonal_j3(r)
        a_total = a_total + a
        da_dr_total = da_dr_total + da

    if config.enable_j4:
        a, da = zonal_j4(r)
        a_total = a_total + a
        da_dr_total = da_dr_total + da

    if config.enable_j22_tesseral:
        from ..core.frames import gmst_from_mjd_tt
        gmst = gmst_from_mjd_tt(epoch_mjd_tt)
        a, da = tesseral_j22(r, gmst)
        a_total = a_total + a
        da_dr_total = da_dr_total + da

    return a_total, da_dr_total
