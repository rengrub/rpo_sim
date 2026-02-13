"""
Equations of motion assembler.

Constructs the full state derivative vector for numerical integration:
    y = [x, y, z, vx, vy, vz, mass, Phi_11, ..., Phi_66]  (43 elements)

The STM variational equations are integrated alongside the state:
    dPhi/dt = A(t) * Phi
where A(t) is the 6x6 Jacobian of the equations of motion.

During finite burns, thrust acceleration and mass depletion are included.
"""

from __future__ import annotations

import numpy as np
from ..core.config import SimConfig
from ..core.constants import (
    MU_EARTH, R_EARTH, G0, SECONDS_PER_DAY
)
from .gravity import combined_gravity
from .srp import cannonball_srp, compute_shadow_factor
from .thirdbody import solar_gravity, lunar_gravity
from .ephemeris import sun_position_eci, moon_position_eci


def eom_full(t: float, y: np.ndarray,
             config: SimConfig,
             cr: float,
             area_over_mass_m2_kg: float,
             epoch_ref_mjd_tt: float,
             thrust_func=None
             ) -> np.ndarray:
    """Full equations of motion with STM variational equations.

    Computes the time derivative of the 43-element augmented state vector.

    Args:
        t: Integration time [seconds since epoch_ref].
        y: Augmented state vector, shape (43,).
            y[0:3]  = position [km]
            y[3:6]  = velocity [km/s]
            y[6]    = mass [kg]
            y[7:43] = STM elements (row-major 6x6)
        config: Simulation configuration with force model toggles.
        cr: SRP reflectivity coefficient.
        area_over_mass_m2_kg: Area-to-mass ratio [m^2/kg].
        epoch_ref_mjd_tt: Reference epoch [MJD TT] — t=0 corresponds to this.
        thrust_func: Optional callable(t, y) -> (thrust_accel_km_s2, mass_flow_kg_s).
            Returns thrust acceleration [km/s^2] shape (3,) and mass flow rate [kg/s].
            None during coast arcs.

    Returns:
        dy_dt: Time derivative of augmented state, shape (43,).
    """
    # Unpack state
    r = y[0:3]
    v = y[3:6]
    mass = y[6]
    stm_flat = y[7:43]
    stm = stm_flat.reshape(6, 6)

    # Current epoch
    epoch_mjd_tt = epoch_ref_mjd_tt + t / SECONDS_PER_DAY

    # --- Gravitational acceleration + Jacobian ---
    a_grav, da_dr_grav = combined_gravity(r, epoch_mjd_tt, config.force_model)

    # Total acceleration and Jacobian accumulators
    a_total = a_grav.copy()
    da_dr_total = da_dr_grav.copy()
    da_dv_total = np.zeros((3, 3))  # No velocity-dependent forces at GEO

    # --- SRP ---
    if config.force_model.enable_srp:
        r_sun = sun_position_eci(epoch_mjd_tt)
        shadow = compute_shadow_factor(
            r, r_sun, R_EARTH, config.force_model.shadow_model
        )
        a_srp, da_dr_srp = cannonball_srp(
            r, r_sun, cr, area_over_mass_m2_kg, shadow
        )
        a_total += a_srp
        da_dr_total += da_dr_srp

    # --- Third-body: Sun ---
    if config.force_model.enable_solar_gravity:
        r_sun = sun_position_eci(epoch_mjd_tt)
        a_sun, da_dr_sun = solar_gravity(r, r_sun)
        a_total += a_sun
        da_dr_total += da_dr_sun

    # --- Third-body: Moon ---
    if config.force_model.enable_lunar_gravity:
        r_moon = moon_position_eci(epoch_mjd_tt)
        a_moon, da_dr_moon = lunar_gravity(r, r_moon)
        a_total += a_moon
        da_dr_total += da_dr_moon

    # --- Thrust (finite burn) ---
    dm_dt = 0.0
    if thrust_func is not None:
        a_thrust, mdot = thrust_func(t, y)
        a_total += a_thrust
        dm_dt = mdot  # negative value (mass decreasing)
        # Note: thrust Jacobian w.r.t. position is zero for fixed-direction burns.
        # For steered burns, the steering law introduces position dependence
        # that should be added to da_dr_total. At medium-high fidelity,
        # we neglect this (consistent with impulsive-like correction approach).

    # --- Assemble state Jacobian A(t) for STM ---
    # A = [[  0_3x3,   I_3x3  ],
    #      [da/dr_3x3, da/dv_3x3]]
    A = np.zeros((6, 6))
    A[0:3, 3:6] = np.eye(3)
    A[3:6, 0:3] = da_dr_total
    A[3:6, 3:6] = da_dv_total

    # STM derivative: dPhi/dt = A * Phi
    dstm = A @ stm

    # --- Assemble full derivative ---
    dy_dt = np.zeros(43)
    dy_dt[0:3] = v                  # dr/dt = v
    dy_dt[3:6] = a_total            # dv/dt = a
    dy_dt[6] = dm_dt                # dm/dt
    dy_dt[7:43] = dstm.flatten()    # dPhi/dt

    return dy_dt


def make_thrust_func(burn_profile, epoch_ref_mjd_tt: float):
    """Create a thrust function for use in eom_full during finite burns.

    Args:
        burn_profile: FiniteBurnProfile object.
        epoch_ref_mjd_tt: Reference epoch for time conversion.

    Returns:
        Callable(t, y) -> (a_thrust_km_s2, mdot_kg_s) or None if outside burn window.
    """
    from ..core.constants import G0

    t_start_s = (burn_profile.t_start_mjd_tt - epoch_ref_mjd_tt) * SECONDS_PER_DAY
    t_end_s = t_start_s + burn_profile.duration_s

    def thrust_func(t, y):
        """Compute thrust acceleration and mass flow rate.

        Args:
            t: Time since epoch_ref [seconds].
            y: Augmented state vector.

        Returns:
            a_thrust: Thrust acceleration [km/s^2], shape (3,).
            mdot: Mass flow rate [kg/s] (negative).
        """
        if t < t_start_s or t > t_end_s:
            return np.zeros(3), 0.0

        mass = y[6]
        if mass <= 0:
            return np.zeros(3), 0.0

        # Thrust direction
        if burn_profile.steering_law is not None:
            direction = burn_profile.steering_law(t, y)
        else:
            direction = burn_profile.direction_eci

        # Thrust acceleration: T/m, converted from N/kg = m/s² to km/s²
        a_thrust = (burn_profile.thrust_n / mass) * direction / 1000.0  # km/s²
        mdot = -burn_profile.thrust_n / (burn_profile.isp_s * G0)  # kg/s (negative)

        return a_thrust, mdot

    return thrust_func
