"""
Maneuver execution error models.

Models the uncertainty introduced by imperfect maneuver execution:
    - Magnitude error: proportional to commanded delta-V
    - Pointing error: angular deviation from commanded thrust direction
"""

from __future__ import annotations

import numpy as np


class ManeuverExecutionError:
    """Maneuver execution error model for covariance inflation.

    Decomposes errors into thrust-frame components:
        - Along thrust direction: magnitude error
        - Perpendicular to thrust: pointing error

    Attributes:
        magnitude_error_1sigma: Fractional magnitude error (1 sigma).
        pointing_error_1sigma_rad: Pointing error [rad] (1 sigma).
    """

    def __init__(self, magnitude_error_1sigma: float,
                 pointing_error_1sigma_rad: float):
        """Initialize execution error model.

        Args:
            magnitude_error_1sigma: Fractional delta-V magnitude error, 1 sigma.
                E.g., 0.01 = 1% magnitude uncertainty.
            pointing_error_1sigma_rad: Thrust pointing error [rad], 1 sigma.
                E.g., 0.005 rad ~ 0.3 degrees.
        """
        self.magnitude_error_1sigma = magnitude_error_1sigma
        self.pointing_error_1sigma_rad = pointing_error_1sigma_rad

    def dv_covariance(self, dv_vec: np.ndarray) -> np.ndarray:
        """Compute delta-V execution error covariance in ECI.

        Constructs the 3x3 covariance in the thrust frame, then rotates to ECI.

        Args:
            dv_vec: Commanded delta-V vector in ECI [km/s], shape (3,).

        Returns:
            P_dv: 3x3 delta-V error covariance in ECI [km^2/s^2].
        """
        dv_mag = np.linalg.norm(dv_vec)
        if dv_mag < 1e-15:
            return np.zeros((3, 3))

        # Thrust direction and perpendicular basis
        u_thrust = dv_vec / dv_mag

        # Find perpendicular vectors
        if abs(u_thrust[0]) < 0.9:
            u_perp1 = np.cross(u_thrust, np.array([1., 0., 0.]))
        else:
            u_perp1 = np.cross(u_thrust, np.array([0., 1., 0.]))
        u_perp1 /= np.linalg.norm(u_perp1)
        u_perp2 = np.cross(u_thrust, u_perp1)

        # Rotation matrix: thrust frame to ECI
        R_tf_to_eci = np.column_stack([u_thrust, u_perp1, u_perp2])

        # Variances in thrust frame
        sigma_mag = self.magnitude_error_1sigma * dv_mag
        sigma_cross = self.pointing_error_1sigma_rad * dv_mag

        P_tf = np.diag([sigma_mag**2, sigma_cross**2, sigma_cross**2])

        # Rotate to ECI
        P_dv = R_tf_to_eci @ P_tf @ R_tf_to_eci.T

        return P_dv

    def inflate_state_covariance(self, P_pre: np.ndarray,
                                  dv_vec: np.ndarray) -> np.ndarray:
        """Inflate state covariance with maneuver execution error.

        P_post = P_pre + B * P_dv * B^T

        where B maps velocity perturbations into the state space:
            B = [[0_3x3], [I_3x3]]  (impulsive maneuver)

        Args:
            P_pre: Pre-maneuver 6x6 state covariance.
            dv_vec: Commanded delta-V vector [km/s], shape (3,).

        Returns:
            P_post: Post-maneuver 6x6 state covariance.
        """
        P_dv = self.dv_covariance(dv_vec)

        P_post = P_pre.copy()
        # Add dv covariance into the velocity partition (lower-right 3x3)
        P_post[3:6, 3:6] += P_dv

        return P_post
