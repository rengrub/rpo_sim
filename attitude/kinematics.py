"""
Attitude kinematics for sensor pointing.

Provides:
    - Target-pointing quaternion computation
    - Slew profile planning (trapezoidal angular rate)
    - Pointing performance model (bias, jitter, stability)
    - Attitude timeline generation over a trajectory

This is a KINEMATIC model — no Euler equation, no reaction wheel dynamics.
ADCS performance is captured parametrically via ADCSParams.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from ..core.types import ADCSParams, AttitudeState, PropagationResult
from .quaternion import (
    q_normalize, q_from_axis_angle, q_to_dcm, dcm_to_q,
    q_slerp, q_angle_between, q_multiply, q_conjugate
)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class PointingPerformance:
    """Pointing performance metrics for a given integration time.

    Attributes:
        bias_rad: Systematic pointing error [rad, 1 sigma].
        jitter_rad: High-frequency pointing oscillation [rad, 1 sigma].
        stability_rad: Total pointing motion during integration [rad, 1 sigma].
        effective_resolution_factor: Resolution degradation factor (>= 1.0).
            Multiply diffraction-limited resolution by this to get
            effective resolution including pointing effects.
    """
    bias_rad: float
    jitter_rad: float
    stability_rad: float
    effective_resolution_factor: float


@dataclass
class SlewProfile:
    """Planned slew maneuver between two orientations.

    Uses a trapezoidal angular velocity profile: constant acceleration
    to max rate, coast at max rate, constant deceleration to zero.
    For small slews that never reach max rate, the profile is triangular.

    Attributes:
        q_start: Starting quaternion, shape (4,).
        q_end: Ending quaternion, shape (4,).
        slew_angle_rad: Total rotation angle [rad].
        accel_time_s: Duration of acceleration phase [s].
        coast_time_s: Duration of coast (max rate) phase [s].
        decel_time_s: Duration of deceleration phase [s].
        total_slew_time_s: Slew duration excluding settle [s].
        settle_time_s: Post-slew settle time [s].
        total_time_s: Slew + settle [s].
        peak_rate_rad_s: Peak angular rate achieved [rad/s].
    """
    q_start: np.ndarray
    q_end: np.ndarray
    slew_angle_rad: float
    accel_time_s: float
    coast_time_s: float
    decel_time_s: float
    total_slew_time_s: float
    settle_time_s: float
    total_time_s: float
    peak_rate_rad_s: float

    def quaternion_at(self, t: float) -> np.ndarray:
        """Interpolate quaternion along the slew path at time t.

        Uses SLERP parameterized by the integrated trapezoidal profile.

        Args:
            t: Time since slew start [seconds]. Clamped to valid range.

        Returns:
            Quaternion at time t, shape (4,).
        """
        if t <= 0.0:
            return self.q_start.copy()
        if t >= self.total_slew_time_s:
            return self.q_end.copy()

        frac = self._angle_fraction(t)
        return q_slerp(self.q_start, self.q_end, frac)

    def angular_rate_at(self, t: float) -> float:
        """Angular rate magnitude at time t [rad/s].

        Args:
            t: Time since slew start [seconds].

        Returns:
            Angular rate [rad/s].
        """
        if t < 0.0 or t > self.total_slew_time_s:
            return 0.0

        t_a = self.accel_time_s
        t_c = self.coast_time_s
        omega_peak = self.peak_rate_rad_s

        if t <= t_a:
            return omega_peak * (t / t_a) if t_a > 0 else 0.0
        elif t <= t_a + t_c:
            return omega_peak
        else:
            t_into_decel = t - t_a - t_c
            t_d = self.decel_time_s
            return omega_peak * (1.0 - t_into_decel / t_d) if t_d > 0 else 0.0

    def is_settled(self, t: float) -> bool:
        """Whether the slew is complete and settled at time t.

        Args:
            t: Time since slew start [seconds].

        Returns:
            True if slew + settle is complete.
        """
        return t >= self.total_time_s

    def _angle_fraction(self, t: float) -> float:
        """Fraction of total slew angle traversed at time t.

        Integrates the trapezoidal rate profile analytically.

        Args:
            t: Time since slew start [seconds].

        Returns:
            Fraction in [0, 1].
        """
        theta = self.slew_angle_rad
        if theta < 1e-15:
            return 1.0

        t_a = self.accel_time_s
        t_c = self.coast_time_s
        t_d = self.decel_time_s
        omega_peak = self.peak_rate_rad_s

        angle = 0.0

        # Acceleration phase: angle = 0.5 * alpha * t^2 = 0.5 * (omega/t_a) * t^2
        if t <= t_a:
            if t_a > 0:
                angle = 0.5 * (omega_peak / t_a) * t * t
            return np.clip(angle / theta, 0.0, 1.0)

        angle += 0.5 * omega_peak * t_a  # Full accel phase

        t_rem = t - t_a
        if t_rem <= t_c:
            angle += omega_peak * t_rem
            return np.clip(angle / theta, 0.0, 1.0)

        angle += omega_peak * t_c  # Full coast phase

        t_rem -= t_c
        if t_rem <= t_d and t_d > 0:
            alpha_d = omega_peak / t_d
            angle += omega_peak * t_rem - 0.5 * alpha_d * t_rem * t_rem
            return np.clip(angle / theta, 0.0, 1.0)

        return 1.0


@dataclass
class AttitudeTimeline:
    """Attitude history over a trajectory.

    Attributes:
        epochs: Time tags [MJD TT], shape (N,).
        quaternions: ECI-to-body quaternions, shape (N, 4).
        angular_rates: Angular rate magnitudes [rad/s], shape (N,).
        pointing_errors: Pointing performance at each epoch.
        modes: Attitude mode at each epoch ('tracking', 'slewing', 'settled').
    """
    epochs: np.ndarray
    quaternions: np.ndarray
    angular_rates: np.ndarray
    pointing_errors: list[PointingPerformance]
    modes: list[str]


# ---------------------------------------------------------------------------
# Attitude Profile Manager
# ---------------------------------------------------------------------------

class AttitudeProfile:
    """Manages attitude computation, slew planning, and performance assessment.

    Attributes:
        adcs: ADCS performance parameters.
    """

    def __init__(self, adcs: ADCSParams):
        """Initialize with ADCS parameters.

        Args:
            adcs: ADCS performance parameter set.
        """
        self.adcs = adcs

    def compute_target_pointing_quaternion(
            self,
            r_chaser_eci: np.ndarray,
            v_chaser_eci: np.ndarray,
            r_target_eci: np.ndarray,
            boresight_body: np.ndarray = None,
            secondary_body: np.ndarray = None,
            secondary_eci: np.ndarray = None
    ) -> np.ndarray:
        """Compute the ECI-to-body quaternion that points the sensor at the target.

        Points the body-frame boresight axis along the chaser-to-target
        line of sight. The roll about the boresight is constrained by
        aligning a secondary body axis as close as possible to a
        secondary ECI direction (defaults to orbit normal).

        Args:
            r_chaser_eci: Chaser ECI position [km], shape (3,).
            v_chaser_eci: Chaser ECI velocity [km/s], shape (3,).
            r_target_eci: Target ECI position [km], shape (3,).
            boresight_body: Sensor boresight in body frame, shape (3,).
                Defaults to +Z body axis [0, 0, 1].
            secondary_body: Secondary body axis for roll constraint, shape (3,).
                Defaults to +Y body axis [0, 1, 0].
            secondary_eci: Secondary reference direction in ECI, shape (3,).
                Defaults to orbit normal (h-hat).

        Returns:
            Quaternion (ECI-to-body), shape (4,).
        """
        if boresight_body is None:
            boresight_body = np.array([0., 0., 1.])
        if secondary_body is None:
            secondary_body = np.array([0., 1., 0.])

        # Desired boresight direction in ECI
        los = r_target_eci - r_chaser_eci
        los_mag = np.linalg.norm(los)
        if los_mag < 1e-10:
            return np.array([0., 0., 0., 1.])
        boresight_eci = los / los_mag

        # Default secondary: orbit normal
        if secondary_eci is None:
            h = np.cross(r_chaser_eci, v_chaser_eci)
            h_mag = np.linalg.norm(h)
            secondary_eci = h / h_mag if h_mag > 1e-10 else np.array([0., 0., 1.])

        # Build desired ECI triad from boresight + secondary constraint.
        # z_eci = boresight direction
        z_eci = boresight_eci

        # y_eci = secondary projected perpendicular to boresight
        y_eci = secondary_eci - np.dot(secondary_eci, z_eci) * z_eci
        y_mag = np.linalg.norm(y_eci)
        if y_mag < 1e-10:
            perp_seed = np.array([1., 0., 0.]) if abs(z_eci[0]) < 0.9 else np.array([0., 1., 0.])
            y_eci = np.cross(z_eci, perp_seed)
            y_mag = np.linalg.norm(y_eci)
        y_eci = y_eci / y_mag

        x_eci = np.cross(y_eci, z_eci)
        x_eci = x_eci / np.linalg.norm(x_eci)

        # R_default_b2e: body-to-ECI assuming boresight=+Z, secondary=+Y
        R_default_b2e = np.column_stack([x_eci, y_eci, z_eci])

        # Handle arbitrary boresight/secondary body axes
        third_body = np.cross(secondary_body, boresight_body)
        t_mag = np.linalg.norm(third_body)
        if t_mag < 1e-10:
            R_eci_to_body = R_default_b2e.T
            return dcm_to_q(R_eci_to_body)

        third_body = third_body / t_mag
        secondary_body_orth = np.cross(boresight_body, third_body)
        secondary_body_orth = secondary_body_orth / np.linalg.norm(secondary_body_orth)

        # R_axes maps default [x=third, y=sec, z=bore] to body axes
        R_axes = np.column_stack([third_body, secondary_body_orth, boresight_body])

        R_body_to_eci = R_default_b2e @ R_axes.T
        R_eci_to_body = R_body_to_eci.T

        return dcm_to_q(R_eci_to_body)

    def plan_slew(self, q_start: np.ndarray, q_end: np.ndarray) -> SlewProfile:
        """Plan a slew between two orientations using trapezoidal rate profile.

        For large slews: accelerate → coast at max rate → decelerate.
        For small slews: triangular profile (never reaches max rate).

        Args:
            q_start: Starting quaternion (ECI-to-body), shape (4,).
            q_end: Target quaternion (ECI-to-body), shape (4,).

        Returns:
            SlewProfile with timing and interpolation capability.
        """
        slew_angle = q_angle_between(q_start, q_end)
        omega_max = self.adcs.max_slew_rate
        alpha = self.adcs.max_slew_accel
        settle = self.adcs.settle_time_s

        if slew_angle < 1e-10:
            return SlewProfile(
                q_start=q_start.copy(), q_end=q_end.copy(),
                slew_angle_rad=0.0,
                accel_time_s=0.0, coast_time_s=0.0, decel_time_s=0.0,
                total_slew_time_s=0.0, settle_time_s=0.0, total_time_s=0.0,
                peak_rate_rad_s=0.0
            )

        if alpha <= 0:
            t_slew = slew_angle / omega_max if omega_max > 0 else 0.0
            return SlewProfile(
                q_start=q_start.copy(), q_end=q_end.copy(),
                slew_angle_rad=slew_angle,
                accel_time_s=0.0, coast_time_s=t_slew, decel_time_s=0.0,
                total_slew_time_s=t_slew, settle_time_s=settle,
                total_time_s=t_slew + settle,
                peak_rate_rad_s=omega_max
            )

        # Angle consumed by symmetric accel + decel to/from max rate
        t_ramp = omega_max / alpha
        angle_ramps = alpha * t_ramp**2  # = omega_max^2 / alpha

        if slew_angle <= angle_ramps:
            # Triangular: theta = alpha * t_a^2, t_a = sqrt(theta / alpha)
            t_a = np.sqrt(slew_angle / alpha)
            peak_rate = alpha * t_a
            return SlewProfile(
                q_start=q_start.copy(), q_end=q_end.copy(),
                slew_angle_rad=slew_angle,
                accel_time_s=t_a, coast_time_s=0.0, decel_time_s=t_a,
                total_slew_time_s=2.0 * t_a, settle_time_s=settle,
                total_time_s=2.0 * t_a + settle,
                peak_rate_rad_s=peak_rate
            )
        else:
            # Trapezoidal
            t_coast = (slew_angle - angle_ramps) / omega_max
            t_total = 2.0 * t_ramp + t_coast
            return SlewProfile(
                q_start=q_start.copy(), q_end=q_end.copy(),
                slew_angle_rad=slew_angle,
                accel_time_s=t_ramp, coast_time_s=t_coast, decel_time_s=t_ramp,
                total_slew_time_s=t_total, settle_time_s=settle,
                total_time_s=t_total + settle,
                peak_rate_rad_s=omega_max
            )

    def pointing_error_model(self, integration_time_s: float) -> PointingPerformance:
        """Compute pointing performance for a given detector integration time.

        Pointing stability during integration combines jitter and drift:
            sigma_stability = sqrt(sigma_jitter^2 + (drift_rate * dt)^2)

        Args:
            integration_time_s: Detector integration time [seconds].

        Returns:
            PointingPerformance with bias, jitter, and stability metrics.
        """
        bias = self.adcs.bias_1sigma_rad
        jitter = self.adcs.jitter_1sigma_rad
        drift = self.adcs.drift_rate_rad_per_s * integration_time_s
        stability = np.sqrt(jitter**2 + drift**2)

        return PointingPerformance(
            bias_rad=bias,
            jitter_rad=jitter,
            stability_rad=stability,
            effective_resolution_factor=1.0  # Caller combines with optics
        )

    def generate_timeline(
            self,
            chaser_trajectory: PropagationResult,
            target_trajectory: PropagationResult,
            boresight_body: np.ndarray = None,
            integration_time_s: float = 0.1
    ) -> AttitudeTimeline:
        """Generate an attitude timeline for target tracking over a trajectory.

        At each epoch, computes the target-pointing quaternion and the
        angular rate required to maintain track. Classifies each epoch
        as 'settled', 'tracking', or 'slewing' based on rate vs ADCS limits.

        Args:
            chaser_trajectory: Chaser propagation result.
            target_trajectory: Target propagation result (same epoch grid).
            boresight_body: Sensor boresight in body frame, shape (3,).
            integration_time_s: Nominal integration time for perf calc [s].

        Returns:
            AttitudeTimeline with quaternion history, rates, and performance.
        """
        if boresight_body is None:
            boresight_body = np.array([0., 0., 1.])

        n = len(chaser_trajectory.epochs)
        quaternions = np.zeros((n, 4))
        angular_rates = np.zeros(n)
        performances = []
        modes = []

        perf = self.pointing_error_model(integration_time_s)

        for i in range(n):
            r_c = chaser_trajectory.states[i, 0:3]
            v_c = chaser_trajectory.states[i, 3:6]
            r_t = target_trajectory.states[i, 0:3]

            quaternions[i] = self.compute_target_pointing_quaternion(
                r_c, v_c, r_t, boresight_body=boresight_body
            )

            if i > 0:
                dt = (chaser_trajectory.epochs[i] -
                      chaser_trajectory.epochs[i - 1]) * 86400.0
                if dt > 0:
                    angular_rates[i] = q_angle_between(
                        quaternions[i - 1], quaternions[i]
                    ) / dt

            if angular_rates[i] > self.adcs.max_slew_rate * 0.9:
                modes.append("slewing")
            elif angular_rates[i] > self.adcs.max_slew_rate * 0.01:
                modes.append("tracking")
            else:
                modes.append("settled")

            performances.append(perf)

        return AttitudeTimeline(
            epochs=chaser_trajectory.epochs.copy(),
            quaternions=quaternions,
            angular_rates=angular_rates,
            pointing_errors=performances,
            modes=modes
        )
