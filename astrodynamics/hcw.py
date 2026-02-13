"""
Hill-Clohessy-Wiltshire (HCW) relative motion model.

Used as the trajectory design oracle: provides initial guess topology
for approach waypoints, maneuver timing, and delta-V vectors.
All solutions are subsequently corrected in the high-fidelity ECI model.

Convention: relative state is [x, y, z, dx, dy, dz] where
    x = radial (R-bar, positive outward)
    y = in-track (V-bar, positive in velocity direction)
    z = cross-track (H-bar, positive along orbit normal)
"""

from __future__ import annotations

import numpy as np
from ..core.types import RelativeState, ManeuverPlan, Maneuver, ManeuverType
from ..core.constants import MU_EARTH, R_GEO, N_GEO


def hcw_stm(n: float, dt: float) -> np.ndarray:
    """Analytical HCW state transition matrix.

    Args:
        n: Mean motion of the target orbit [rad/s].
        dt: Time interval [seconds].

    Returns:
        Phi: 6x6 HCW state transition matrix.
    """
    nt = n * dt
    c = np.cos(nt)
    s = np.sin(nt)

    phi = np.array([
        [4 - 3*c,      0, 0,  s/n,      2*(1-c)/n,   0],
        [6*(s - nt),   1, 0, -2*(1-c)/n, (4*s - 3*nt)/n, 0],
        [0,            0, c,  0,         0,           s/n],
        [3*n*s,        0, 0,  c,         2*s,         0],
        [-6*n*(1-c),   0, 0, -2*s,       4*c - 3,     0],
        [0,            0, -n*s, 0,       0,           c]
    ])

    return phi


def hcw_free_motion(rel_state_0: np.ndarray, n: float,
                    t_array: np.ndarray) -> np.ndarray:
    """Propagate relative state analytically via HCW.

    Args:
        rel_state_0: Initial relative state [x,y,z,dx,dy,dz], shape (6,).
            Position [km], velocity [km/s].
        n: Target orbit mean motion [rad/s].
        t_array: Array of times [seconds] at which to evaluate, shape (N,).

    Returns:
        states: Relative state history, shape (N, 6).
    """
    states = np.zeros((len(t_array), 6))
    for i, t in enumerate(t_array):
        phi = hcw_stm(n, t)
        states[i] = phi @ rel_state_0
    return states


def hcw_two_impulse_transfer(rel_0: np.ndarray, rel_f: np.ndarray,
                             n: float, dt: float
                             ) -> tuple[np.ndarray, np.ndarray]:
    """Solve the two-impulse relative transfer problem (HCW Lambert).

    Given initial and final relative positions and a transfer time,
    find the two impulsive delta-V vectors.

    Args:
        rel_0: Initial relative state [x,y,z,dx,dy,dz] [km, km/s], shape (6,).
        rel_f: Desired final relative state [x,y,z,dx,dy,dz] [km, km/s], shape (6,).
        n: Target orbit mean motion [rad/s].
        dt: Transfer time [seconds].

    Returns:
        dv1: Departure delta-V [km/s], shape (3,).
        dv2: Arrival delta-V [km/s], shape (3,).
    """
    phi = hcw_stm(n, dt)

    # Partition STM: Phi = [[Phi_rr, Phi_rv], [Phi_vr, Phi_vv]]
    phi_rr = phi[0:3, 0:3]
    phi_rv = phi[0:3, 3:6]
    phi_vr = phi[3:6, 0:3]
    phi_vv = phi[3:6, 3:6]

    r0 = rel_0[0:3]
    v0 = rel_0[3:6]
    rf = rel_f[0:3]
    vf = rel_f[3:6]

    # Required initial velocity to reach rf from r0 in time dt:
    #   rf = Phi_rr * r0 + Phi_rv * v0_required
    #   v0_required = Phi_rv^-1 * (rf - Phi_rr * r0)
    v0_required = np.linalg.solve(phi_rv, rf - phi_rr @ r0)

    # Delta-V at departure
    dv1 = v0_required - v0

    # Velocity at arrival from the transfer orbit
    v_arrive = phi_vr @ r0 + phi_vv @ v0_required

    # Delta-V at arrival to match desired final velocity
    dv2 = vf - v_arrive

    return dv1, dv2


def hcw_single_impulse_transfer(r0: np.ndarray, rf: np.ndarray,
                                v0: np.ndarray,
                                n: float, dt: float) -> np.ndarray:
    """Single-impulse transfer: find delta-V at departure to reach rf.

    No arrival maneuver — the velocity at arrival is whatever the
    transfer orbit produces.

    Args:
        r0: Initial relative position [km], shape (3,).
        rf: Desired final relative position [km], shape (3,).
        v0: Current relative velocity [km/s], shape (3,).
        n: Target orbit mean motion [rad/s].
        dt: Transfer time [seconds].

    Returns:
        dv: Departure delta-V [km/s], shape (3,).
    """
    phi = hcw_stm(n, dt)
    phi_rr = phi[0:3, 0:3]
    phi_rv = phi[0:3, 3:6]

    v0_req = np.linalg.solve(phi_rv, rf - phi_rr @ r0)
    return v0_req - v0


def design_vbar_approach(range_initial_km: float,
                         range_final_km: float,
                         n: float,
                         n_hops: int,
                         hop_duration_orbits: float = 0.5
                         ) -> ManeuverPlan:
    """Design a multi-hop V-bar approach (along-track).

    Generates a sequence of impulsive maneuvers to step the chaser
    along the V-bar from range_initial to range_final.

    V-bar approach is passively safe: if a maneuver is missed,
    the chaser drifts along the V-bar without collision risk.

    Each hop transfers from one V-bar station to the next using a
    two-impulse HCW transfer. The waypoints are equally spaced in
    the in-track (y) direction.

    Args:
        range_initial_km: Starting in-track distance [km] (positive = behind target).
        range_final_km: Final in-track distance [km].
        n: Target orbit mean motion [rad/s].
        n_hops: Number of hops.
        hop_duration_orbits: Duration of each hop in orbital periods.

    Returns:
        ManeuverPlan with the approach maneuver sequence.
    """
    period_s = 2.0 * np.pi / n
    dt = hop_duration_orbits * period_s

    # Waypoint in-track distances (equally spaced)
    y_stations = np.linspace(range_initial_km, range_final_km, n_hops + 1)

    maneuvers = []
    cumulative_time = 0.0

    for i in range(n_hops):
        # Current and next waypoints on V-bar (x=0, y=station, z=0)
        r0 = np.array([0.0, y_stations[i], 0.0])
        rf = np.array([0.0, y_stations[i + 1], 0.0])
        v0 = np.zeros(3)  # at rest on V-bar

        # Current state
        state0 = np.concatenate([r0, v0])
        # Desired final state (at rest on V-bar)
        statef = np.concatenate([rf, np.zeros(3)])

        dv1, dv2 = hcw_two_impulse_transfer(state0, statef, n, dt)

        # Departure maneuver
        maneuvers.append(Maneuver(
            epoch_mjd_tt=cumulative_time,  # placeholder; real epochs set by caller
            dv_vector_eci=dv1,  # in LVLH; caller converts to ECI
            maneuver_type=ManeuverType.IMPULSIVE,
        ))

        # Arrival maneuver
        maneuvers.append(Maneuver(
            epoch_mjd_tt=cumulative_time + dt,
            dv_vector_eci=dv2,
            maneuver_type=ManeuverType.IMPULSIVE,
        ))

        cumulative_time += dt

    return ManeuverPlan(maneuvers=maneuvers)


def design_rbar_approach(range_initial_km: float,
                         range_final_km: float,
                         n: float,
                         n_hops: int,
                         hop_duration_orbits: float = 0.5
                         ) -> ManeuverPlan:
    """Design a multi-hop R-bar approach (radial).

    R-bar approach provides favorable lighting geometry (sun behind chaser)
    but is not passively safe — a missed maneuver leads to along-track drift
    that could violate keep-out zones.

    Each hop transfers between radial stations (x-axis) using two-impulse
    HCW transfers.

    Args:
        range_initial_km: Starting radial offset [km] (positive = below target).
        range_final_km: Final radial offset [km].
        n: Target mean motion [rad/s].
        n_hops: Number of hops.
        hop_duration_orbits: Duration of each hop in orbital periods.

    Returns:
        ManeuverPlan with the approach maneuver sequence.
    """
    period_s = 2.0 * np.pi / n
    dt = hop_duration_orbits * period_s

    # Waypoint radial distances (equally spaced)
    x_stations = np.linspace(range_initial_km, range_final_km, n_hops + 1)

    maneuvers = []
    cumulative_time = 0.0

    for i in range(n_hops):
        # Station on R-bar: (x, 0, 0) with zero relative velocity
        r0 = np.array([x_stations[i], 0.0, 0.0])
        rf = np.array([x_stations[i + 1], 0.0, 0.0])
        v0 = np.zeros(3)

        state0 = np.concatenate([r0, v0])
        statef = np.concatenate([rf, np.zeros(3)])

        dv1, dv2 = hcw_two_impulse_transfer(state0, statef, n, dt)

        maneuvers.append(Maneuver(
            epoch_mjd_tt=cumulative_time,
            dv_vector_eci=dv1,
            maneuver_type=ManeuverType.IMPULSIVE,
        ))
        maneuvers.append(Maneuver(
            epoch_mjd_tt=cumulative_time + dt,
            dv_vector_eci=dv2,
            maneuver_type=ManeuverType.IMPULSIVE,
        ))

        cumulative_time += dt

    return ManeuverPlan(maneuvers=maneuvers)


def design_circumnavigation(center_offset_km: np.ndarray,
                            ellipse_axes_km: np.ndarray,
                            n: float) -> ManeuverPlan:
    """Design a relative circumnavigation trajectory for multi-aspect imaging.

    Natural HCW relative motion forms 2:1 ellipses (in-track : radial).
    This designs a single maneuver to enter a circumnavigation orbit
    centered at center_offset with the specified ellipse dimensions.

    The natural HCW ellipse has the property:
        y_amplitude = 2 * x_amplitude
    so the user specifies the radial (x) semi-axis and the in-track (y)
    semi-axis is constrained to be twice that.

    The entry maneuver is applied at a point on the desired ellipse
    to transition from a stationary hold at center_offset into the
    circumnavigation orbit.

    Args:
        center_offset_km: LVLH offset of ellipse center [km], shape (3,).
            Typically [0, y_offset, 0] for a V-bar centered orbit.
        ellipse_axes_km: Desired [radial_semi, in_track_semi] axes [km], shape (2,).
            Note: in-track should be ~2x radial for a natural orbit.
        n: Target mean motion [rad/s].

    Returns:
        ManeuverPlan with a single entry maneuver (at epoch 0).
    """
    x_amp = ellipse_axes_km[0]  # radial semi-axis
    # In-track semi-axis is 2*x_amp for natural HCW motion
    # If user requests different, we use x_amp to set the natural orbit

    # The HCW free-motion solution for a centered circumnavigation:
    #   x(t)  = x_amp * sin(n*t + phi)
    #   y(t)  = y_center - 2*x_amp * cos(n*t + phi)
    #   dx/dt = x_amp * n * cos(n*t + phi)
    #   dy/dt = 2*x_amp * n * sin(n*t + phi)
    #
    # At t=0, phi=0: x=0, y = y_center - 2*x_amp, dx = x_amp*n, dy = 0
    # This is the bottom of the ellipse (closest in-track approach).

    # The maneuver enters the circumnavigation from a hold at center_offset.
    # At the entry point (bottom of ellipse):
    #   r_entry = [0, center_offset[1] - 2*x_amp, center_offset[2]]
    #   v_entry = [x_amp * n, 0, 0]
    #
    # If chaser is currently at rest at center_offset, the maneuver is:
    #   dv = v_entry - v_hold
    # But a hold at center_offset has v=0 only if there's no along-track drift.
    # For a hold at [0, y0, 0], the HCW equilibrium requires:
    #   dx = -2*n*x (for radial), or simply v=0 at [0, y0, 0] (on V-bar).
    #
    # We'll define the maneuver as: from rest at center_offset, apply dv
    # to enter the circumnavigation orbit at the nearest point on the ellipse.
    #
    # At the entry point (x=0, y = y_center - 2*x_amp):
    #   Required velocity: [x_amp*n, 0, 0]
    #   From rest: dv = [x_amp*n, 0, 0]

    dv = np.array([x_amp * n, 0.0, 0.0])

    maneuvers = [Maneuver(
        epoch_mjd_tt=0.0,
        dv_vector_eci=dv,  # in LVLH; caller converts to ECI
        maneuver_type=ManeuverType.IMPULSIVE,
    )]

    return ManeuverPlan(maneuvers=maneuvers)
