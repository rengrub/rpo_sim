"""
Covariance propagation via the State Transition Matrix.

Propagates state uncertainty algebraically (not via differential equation
integration) for computational efficiency and numerical stability:

    P(t) = Phi(t, t0) * P(t0) * Phi(t, t0)^T + Q(t, t0)

At maneuver epochs, the covariance is inflated by maneuver execution errors.
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from ..core.types import (
    CovarianceState, CovarianceHistory, PropagationResult,
    FiniteBurnResult
)
from ..core.config import CovarianceConfig
from .maneuver_errors import ManeuverExecutionError
from .process_noise import constant_acceleration_process_noise


class CovariancePropagator:
    """Propagates state covariance along a trajectory using precomputed STMs.

    This is NOT a Kalman filter. It propagates uncertainty forward in time
    for mission design assessment, without incorporating measurements.

    Attributes:
        config: Covariance configuration (process noise, maneuver errors).
    """

    def __init__(self, config: CovarianceConfig):
        """Initialize the covariance propagator.

        Args:
            config: Covariance propagation configuration.
        """
        self.config = config
        self.maneuver_error = ManeuverExecutionError(
            magnitude_error_1sigma=config.maneuver_mag_error_1sigma,
            pointing_error_1sigma_rad=config.maneuver_point_error_1sigma_rad
        )

    def propagate_step(self,
                       P0: np.ndarray,
                       stm: np.ndarray,
                       dt_s: float) -> np.ndarray:
        """Propagate covariance over a single time step.

        Args:
            P0: Initial 6x6 covariance [km, km/s].
            stm: 6x6 STM mapping from t0 to t.
            dt_s: Time step duration [seconds].

        Returns:
            P: Propagated 6x6 covariance.
        """
        Q = constant_acceleration_process_noise(self.config.process_noise_q, dt_s)
        P = stm @ P0 @ stm.T + Q
        # Enforce symmetry (numerical hygiene)
        P = 0.5 * (P + P.T)
        return P

    def propagate_along_trajectory(self,
                                   P0: np.ndarray,
                                   prop_result: PropagationResult,
                                   maneuver_epochs: Optional[list[float]] = None,
                                   maneuver_dvs: Optional[list[np.ndarray]] = None
                                   ) -> CovarianceHistory:
        """Propagate covariance along a precomputed trajectory.

        At each output epoch, uses the STM to map the covariance forward.
        At maneuver epochs, inflates the covariance with execution errors.

        Args:
            P0: Initial 6x6 covariance at the first trajectory epoch.
            prop_result: Precomputed trajectory with STMs at each output epoch.
            maneuver_epochs: List of maneuver epochs [MJD TT] where
                covariance should be inflated.
            maneuver_dvs: Corresponding delta-V vectors [km/s] for error computation.

        Returns:
            CovarianceHistory with time-tagged covariance matrices.
        """
        epochs = prop_result.epochs
        n_epochs = len(epochs)

        covariances = [P0.copy()]
        P_current = P0.copy()

        if maneuver_epochs is None:
            maneuver_epochs = []
        if maneuver_dvs is None:
            maneuver_dvs = []

        maneuver_set = set()
        maneuver_map = {}
        for ep, dv in zip(maneuver_epochs, maneuver_dvs):
            maneuver_set.add(ep)
            maneuver_map[ep] = dv

        for i in range(1, n_epochs):
            dt = (epochs[i] - epochs[i-1]) * 86400.0  # seconds

            # STM for this step: Phi(t_i, t_{i-1})
            # The stored STM is Phi(t_i, t_0), so the step STM is:
            # Phi(t_i, t_{i-1}) = Phi(t_i, t_0) * Phi(t_{i-1}, t_0)^{-1}
            # For numerical stability, prefer sequential composition.
            #
            # If STMs were reset at maneuver boundaries (as recommended),
            # this is handled by the propagator already.
            stm_i = prop_result.stms[i]
            stm_prev = prop_result.stms[i-1]

            # Compute step STM
            try:
                stm_step = stm_i @ np.linalg.inv(stm_prev)
            except np.linalg.LinAlgError:
                # Fallback: use full STM from origin (less ideal)
                stm_step = stm_i

            P_current = self.propagate_step(P_current, stm_step, dt)

            # Check for maneuver at this epoch
            epoch_i = epochs[i]
            if epoch_i in maneuver_set:
                dv = maneuver_map[epoch_i]
                P_current = self.maneuver_error.inflate_state_covariance(
                    P_current, dv
                )

            covariances.append(P_current.copy())

        return CovarianceHistory(
            epochs=epochs.copy(),
            covariances=covariances
        )

    @staticmethod
    def relative_covariance(P_chaser: np.ndarray,
                            P_target: np.ndarray,
                            P_cross: Optional[np.ndarray] = None
                            ) -> np.ndarray:
        """Compute relative state covariance.

        Args:
            P_chaser: Chaser 6x6 covariance.
            P_target: Target 6x6 covariance.
            P_cross: Optional 6x6 cross-correlation. If None, assumed uncorrelated.

        Returns:
            P_rel: 6x6 relative state covariance.
        """
        if P_cross is None:
            return P_chaser + P_target
        else:
            return P_chaser + P_target - P_cross - P_cross.T
