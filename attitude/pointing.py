"""
Sensor pointing geometry.

Connects the attitude subsystem to the navigation (covariance) subsystem:
    - Projects relative position covariance into angular bearing uncertainty
    - Evaluates whether the target falls within the sensor FOV
    - Computes acquisition probability given navigation uncertainty

Also provides geometric utilities for line-of-sight analysis.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional

from .quaternion import q_to_dcm, q_rotate_vector
from ..core.types import OpticalPayload


@dataclass
class BearingUncertainty:
    """Angular uncertainty in target bearing as seen from the chaser.

    Derived from the relative position covariance projected into the
    plane perpendicular to the line of sight.

    Attributes:
        sigma_cross1_rad: 1-sigma angular uncertainty along first
            perpendicular axis [rad].
        sigma_cross2_rad: 1-sigma angular uncertainty along second
            perpendicular axis [rad].
        sigma_total_rad: RSS total angular uncertainty [rad].
        range_km: Chaser-to-target range [km].
        in_fov_probability: Approximate probability that the target is
            within the sensor FOV given the uncertainty.
    """
    sigma_cross1_rad: float
    sigma_cross2_rad: float
    sigma_total_rad: float
    range_km: float
    in_fov_probability: float


class SensorPointing:
    """Sensor pointing geometry and FOV analysis.

    Evaluates whether a target is within the sensor's field of view,
    accounting for navigation uncertainty (via covariance) and
    pointing errors.

    Attributes:
        boresight_body: Sensor boresight unit vector in body frame, shape (3,).
        fov_half_angle_rad: Half-angle of the sensor FOV [rad].
    """

    def __init__(self, optics: OpticalPayload):
        """Initialize from optical payload definition.

        Args:
            optics: Optical system parameters (provides boresight and FOV).
        """
        self.boresight_body = optics.boresight_body / np.linalg.norm(optics.boresight_body)
        self.fov_half_angle_rad = optics.fov_half_angle_rad
        self.optics = optics

    def boresight_eci(self, q_eci_to_body: np.ndarray) -> np.ndarray:
        """Compute sensor boresight direction in ECI.

        Args:
            q_eci_to_body: ECI-to-body quaternion, shape (4,).

        Returns:
            Boresight unit vector in ECI, shape (3,).
        """
        # Body-to-ECI is the inverse (conjugate) of ECI-to-body
        R_eci_to_body = q_to_dcm(q_eci_to_body)
        R_body_to_eci = R_eci_to_body.T
        return R_body_to_eci @ self.boresight_body

    def target_angular_offset(self, q_eci_to_body: np.ndarray,
                               r_chaser_eci: np.ndarray,
                               r_target_eci: np.ndarray) -> float:
        """Angular offset between boresight and target direction.

        Args:
            q_eci_to_body: Current attitude quaternion, shape (4,).
            r_chaser_eci: Chaser ECI position [km], shape (3,).
            r_target_eci: Target ECI position [km], shape (3,).

        Returns:
            Angular offset [rad] between boresight and line of sight to target.
        """
        los = r_target_eci - r_chaser_eci
        los_mag = np.linalg.norm(los)
        if los_mag < 1e-10:
            return 0.0
        los_hat = los / los_mag

        bore_eci = self.boresight_eci(q_eci_to_body)

        cos_angle = np.clip(np.dot(bore_eci, los_hat), -1.0, 1.0)
        return np.arccos(cos_angle)

    def target_in_fov(self, q_eci_to_body: np.ndarray,
                      r_chaser_eci: np.ndarray,
                      r_target_eci: np.ndarray) -> bool:
        """Check whether the target is within the sensor FOV.

        Args:
            q_eci_to_body: Current attitude quaternion, shape (4,).
            r_chaser_eci: Chaser ECI position [km], shape (3,).
            r_target_eci: Target ECI position [km], shape (3,).

        Returns:
            True if target angular offset < FOV half-angle.
        """
        offset = self.target_angular_offset(q_eci_to_body, r_chaser_eci, r_target_eci)
        return offset < self.fov_half_angle_rad

    def bearing_uncertainty(self,
                            r_chaser_eci: np.ndarray,
                            r_target_eci: np.ndarray,
                            P_rel_eci: np.ndarray
                            ) -> BearingUncertainty:
        """Compute angular bearing uncertainty from relative position covariance.

        Projects the 3x3 relative position covariance into the plane
        perpendicular to the line of sight, then converts to angular
        uncertainty at the given range.

        Args:
            r_chaser_eci: Chaser ECI position [km], shape (3,).
            r_target_eci: Target ECI position [km], shape (3,).
            P_rel_eci: Relative position covariance in ECI [km^2], shape (3,3).

        Returns:
            BearingUncertainty with angular uncertainties and FOV probability.
        """
        los = r_target_eci - r_chaser_eci
        rng = np.linalg.norm(los)

        if rng < 1e-10:
            return BearingUncertainty(
                sigma_cross1_rad=0.0,
                sigma_cross2_rad=0.0,
                sigma_total_rad=0.0,
                range_km=0.0,
                in_fov_probability=1.0,
            )

        los_hat = los / rng

        # Projection matrix onto plane perpendicular to LOS
        P_perp = np.eye(3) - np.outer(los_hat, los_hat)

        # Project covariance into the perpendicular plane
        P_cross = P_perp @ P_rel_eci @ P_perp.T

        # The projected covariance is rank-2 (in the perpendicular plane).
        # Extract the 2x2 covariance by eigendecomposition.
        eigvals, _ = np.linalg.eigh(P_cross)

        # Sort descending; the smallest eigenvalue should be ~0 (along LOS)
        eigvals = np.sort(eigvals)[::-1]

        # The two meaningful eigenvalues are the cross-LOS position variances
        sigma_pos1 = np.sqrt(max(eigvals[0], 0.0))  # km
        sigma_pos2 = np.sqrt(max(eigvals[1], 0.0))  # km

        # Convert to angular uncertainty: sigma_angle = sigma_pos / range
        sigma1 = sigma_pos1 / rng  # rad
        sigma2 = sigma_pos2 / rng  # rad
        sigma_total = np.sqrt(sigma1**2 + sigma2**2)

        # Approximate probability that target is within FOV
        # Model as 2D Gaussian; P(within circle of radius theta_fov)
        # For a circular FOV with half-angle theta_fov and 2D Gaussian
        # with sigmas sigma1, sigma2:
        #   P ≈ 1 - exp(-theta_fov^2 / (2 * sigma_avg^2))
        # where sigma_avg = sqrt(sigma1 * sigma2) is geometric mean.
        # This is an approximation valid when the Gaussian is roughly circular.
        sigma_avg = np.sqrt(sigma1 * sigma2) if (sigma1 > 0 and sigma2 > 0) else 0.0

        if sigma_avg > 1e-15:
            theta_fov = self.fov_half_angle_rad
            p_in_fov = 1.0 - np.exp(-0.5 * (theta_fov / sigma_avg)**2)
        else:
            # No uncertainty — target is exactly known
            p_in_fov = 1.0

        return BearingUncertainty(
            sigma_cross1_rad=sigma1,
            sigma_cross2_rad=sigma2,
            sigma_total_rad=sigma_total,
            range_km=rng,
            in_fov_probability=p_in_fov,
        )

    def combined_pointing_miss(self,
                               bearing_uncertainty: BearingUncertainty,
                               pointing_performance: object
                               ) -> float:
        """Total pointing miss combining navigation and ADCS errors.

        Even with perfect attitude control, the navigation uncertainty means
        the boresight command points to the estimated (not true) target position.
        The total miss is the RSS of bearing uncertainty and pointing error.

        sigma_total = sqrt(sigma_bearing^2 + sigma_pointing^2)

        Args:
            bearing_uncertainty: From covariance projection.
            pointing_performance: PointingPerformance from attitude model.

        Returns:
            Total pointing miss [rad, 1 sigma].
        """
        sigma_bearing = bearing_uncertainty.sigma_total_rad
        sigma_pointing = pointing_performance.stability_rad
        return np.sqrt(sigma_bearing**2 + sigma_pointing**2)
