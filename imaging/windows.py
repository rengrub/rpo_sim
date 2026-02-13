"""
Observation window computation.

Evaluates ALL imaging constraints simultaneously at each epoch to identify
contiguous time windows where resolved imaging is feasible. This is the
top-level imaging analysis module that consumes outputs from optics,
radiometry, geometry, attitude, and navigation modules.

Constraints evaluated at each epoch:
    1. Range within bounds
    2. Resolution sufficient (accounting for pointing stability)
    3. Phase angle within acceptable bounds
    4. SNR above threshold
    5. Target sunlit (not in eclipse)
    6. Pointing constraints satisfied (sun exclusion, etc.)
    7. Target within FOV given navigation uncertainty
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from ..core.types import (
    PropagationResult, CovarianceHistory, ImagingMetrics,
    ObservationWindow, ObservationSchedule, OpticalPayload,
    TargetModel
)
from .optics import OpticalSystem
from .radiometry import RadiometricModel
from .geometry import (
    phase_angle, is_target_sunlit_conical, compute_geometry_at_epoch
)
from ..attitude.kinematics import AttitudeTimeline, PointingPerformance
from ..attitude.pointing import SensorPointing
from ..attitude.constraints import PointingConstraintSet, ConstraintResult


@dataclass
class ImagingConstraints:
    """Thresholds for imaging feasibility.

    Attributes:
        min_resolution_elements: Minimum required resolution elements
            across the target for useful resolved imaging.
        min_snr: Minimum signal-to-noise ratio.
        max_range_km: Maximum allowed range [km].
        min_range_km: Minimum safe range [km] (proximity safety).
        phase_angle_min_rad: Minimum acceptable phase angle [rad].
        phase_angle_max_rad: Maximum acceptable phase angle [rad].
        min_fov_probability: Minimum probability that target is in FOV.
        integration_time_s: Detector integration time [s].
    """
    min_resolution_elements: float = 3.0
    min_snr: float = 5.0
    max_range_km: float = 100.0
    min_range_km: float = 0.05   # 50 meters
    phase_angle_min_rad: float = np.deg2rad(20.0)
    phase_angle_max_rad: float = np.deg2rad(120.0)
    min_fov_probability: float = 0.95
    integration_time_s: float = 0.1


class ObservationWindowCalculator:
    """Computes observation windows by evaluating all constraints.

    Ties together the optics, radiometry, geometry, attitude, and
    navigation subsystems into a single feasibility assessment.

    Attributes:
        optical_system: Resolution and GSD calculator.
        radiometric_model: SNR calculator.
        sensor_pointing: FOV and bearing analysis.
        pointing_constraints: Attitude constraint set (optional).
    """

    def __init__(self,
                 optics: OpticalPayload,
                 pointing_constraints: Optional[PointingConstraintSet] = None):
        """Initialize the observation window calculator.

        Args:
            optics: Sensor specification.
            pointing_constraints: Attitude constraints (sun exclusion, etc.).
                If None, pointing is assumed always feasible.
        """
        self.optical_system = OpticalSystem(optics)
        self.radiometric_model = RadiometricModel(optics)
        self.sensor_pointing = SensorPointing(optics)
        self.pointing_constraints = pointing_constraints

    def evaluate_epoch(self,
                       r_chaser_eci: np.ndarray,
                       r_target_eci: np.ndarray,
                       r_sun_eci: np.ndarray,
                       epoch_mjd_tt: float,
                       target: TargetModel,
                       pointing_perf: PointingPerformance,
                       constraints: ImagingConstraints,
                       q_eci_to_body: Optional[np.ndarray] = None,
                       P_rel_pos: Optional[np.ndarray] = None
                       ) -> ImagingMetrics:
        """Evaluate all imaging metrics at a single epoch.

        Args:
            r_chaser_eci: Chaser ECI position [km], shape (3,).
            r_target_eci: Target ECI position [km], shape (3,).
            r_sun_eci: Sun ECI position [km], shape (3,).
            epoch_mjd_tt: Evaluation epoch.
            target: Target physical model.
            pointing_perf: Pointing performance for this epoch.
            constraints: Imaging constraint thresholds.
            q_eci_to_body: Attitude quaternion (optional; needed for
                pointing constraint and FOV evaluation).
            P_rel_pos: Relative position covariance [km^2], shape (3,3).
                Optional; if None, target is assumed perfectly known.

        Returns:
            ImagingMetrics with all computed quantities.
        """
        # --- Geometry ---
        geom = compute_geometry_at_epoch(r_chaser_eci, r_target_eci, r_sun_eci)
        range_km = geom['range_km']
        phase_rad = geom['phase_angle_rad']
        sunlit = geom['target_sunlit']
        helio_au = geom['helio_distance_au']

        # --- Resolution ---
        stability = pointing_perf.stability_rad
        gsd = self.optical_system.effective_gsd_m(range_km, stability)
        res_elements = self.optical_system.effective_resolution_elements(
            range_km, target.characteristic_length_m, stability
        )

        # --- Radiometry ---
        n_pixels = self.optical_system.pixels_on_target(
            range_km, target.characteristic_length_m
        )
        # Only compute SNR if target is sunlit and phase angle is reasonable
        if sunlit and 0 < phase_rad < np.pi:
            target_area = target.characteristic_length_m ** 2  # crude projected area
            snr_val = self.radiometric_model.snr(
                albedo=target.albedo,
                phase_angle_rad=phase_rad,
                target_area_m2=target_area,
                range_km=range_km,
                integration_time_s=constraints.integration_time_s,
                n_pixels_target=max(n_pixels, 1.0),
                solar_distance_au=helio_au
            )
        else:
            snr_val = 0.0

        # --- Pointing constraints ---
        pointing_ok = True
        if self.pointing_constraints is not None and q_eci_to_body is not None:
            cr = self.pointing_constraints.evaluate(
                q_eci_to_body, r_chaser_eci, r_sun_eci
            )
            pointing_ok = cr.all_satisfied

        # --- FOV / bearing uncertainty ---
        in_fov = True
        if P_rel_pos is not None:
            bu = self.sensor_pointing.bearing_uncertainty(
                r_chaser_eci, r_target_eci, P_rel_pos
            )
            in_fov = bu.in_fov_probability >= constraints.min_fov_probability
        elif q_eci_to_body is not None:
            in_fov = self.sensor_pointing.target_in_fov(
                q_eci_to_body, r_chaser_eci, r_target_eci
            )

        return ImagingMetrics(
            epoch_mjd_tt=epoch_mjd_tt,
            range_km=range_km,
            phase_angle_rad=phase_rad,
            gsd_m=gsd,
            resolution_elements=res_elements,
            snr=snr_val,
            target_sunlit=sunlit,
            pointing_feasible=pointing_ok,
            target_in_fov=in_fov
        )

    def compute_windows(self,
                        chaser_trajectory: PropagationResult,
                        target_trajectory: PropagationResult,
                        sun_positions: np.ndarray,
                        target: TargetModel,
                        constraints: ImagingConstraints,
                        attitude_timeline: Optional[AttitudeTimeline] = None,
                        covariance_history: Optional[CovarianceHistory] = None,
                        pointing_performance: Optional[PointingPerformance] = None
                        ) -> ObservationSchedule:
        """Compute observation windows over a full trajectory.

        Evaluates all imaging constraints at each epoch and identifies
        contiguous windows where all constraints are simultaneously met.

        Args:
            chaser_trajectory: Chaser propagation result.
            target_trajectory: Target propagation result (same epoch grid).
            sun_positions: Sun ECI positions [km], shape (N, 3).
            target: Target physical model.
            constraints: Imaging feasibility thresholds.
            attitude_timeline: Attitude history (optional). If provided,
                used for pointing constraints and FOV checks.
            covariance_history: Covariance history (optional). If provided,
                used for bearing uncertainty / FOV probability.
            pointing_performance: Default pointing performance to use at
                all epochs. Overridden by attitude_timeline if provided.

        Returns:
            ObservationSchedule with windows and per-epoch metrics.
        """
        n_epochs = len(chaser_trajectory.epochs)

        # Default pointing performance if not provided
        if pointing_performance is None:
            from ..core.types import ADCSParams
            from ..attitude.kinematics import AttitudeProfile
            default_adcs = ADCSParams()
            pointing_performance = AttitudeProfile(default_adcs).pointing_error_model(
                constraints.integration_time_s
            )

        metrics_list: list[ImagingMetrics] = []

        for i in range(n_epochs):
            epoch = chaser_trajectory.epochs[i]
            r_c = chaser_trajectory.states[i, 0:3]
            r_t = target_trajectory.states[i, 0:3]
            r_sun = sun_positions[i]

            # Attitude
            q = None
            if attitude_timeline is not None:
                q = attitude_timeline.quaternions[i]

            # Pointing performance (per-epoch if available)
            perf = pointing_performance
            if (attitude_timeline is not None and
                    attitude_timeline.pointing_errors is not None):
                perf = attitude_timeline.pointing_errors[i]

            # Relative position covariance (position partition only)
            P_rel = None
            if covariance_history is not None:
                P_full = covariance_history.covariances[i]
                P_rel = P_full[0:3, 0:3]

            metrics = self.evaluate_epoch(
                r_c, r_t, r_sun, epoch, target, perf, constraints,
                q_eci_to_body=q, P_rel_pos=P_rel
            )
            metrics_list.append(metrics)

        # --- Identify valid epochs ---
        valid = np.array([self._epoch_passes(m, constraints)
                          for m in metrics_list])

        # --- Extract contiguous windows ---
        windows = self._extract_windows(
            chaser_trajectory.epochs, metrics_list, valid
        )

        return ObservationSchedule(
            windows=windows,
            metrics_timeline=metrics_list
        )

    def _epoch_passes(self, m: ImagingMetrics,
                      c: ImagingConstraints) -> bool:
        """Check if a single epoch passes all constraints.

        Args:
            m: Imaging metrics at the epoch.
            c: Constraint thresholds.

        Returns:
            True if all constraints are met.
        """
        if not m.target_sunlit:
            return False
        if not m.pointing_feasible:
            return False
        if not m.target_in_fov:
            return False
        if m.range_km > c.max_range_km or m.range_km < c.min_range_km:
            return False
        if m.resolution_elements < c.min_resolution_elements:
            return False
        if m.snr < c.min_snr:
            return False
        if m.phase_angle_rad < c.phase_angle_min_rad:
            return False
        if m.phase_angle_rad > c.phase_angle_max_rad:
            return False
        return True

    def _extract_windows(self,
                         epochs: np.ndarray,
                         metrics: list[ImagingMetrics],
                         valid: np.ndarray
                         ) -> list[ObservationWindow]:
        """Extract contiguous observation windows from boolean timeline.

        Args:
            epochs: Epoch array [MJD TT].
            metrics: Per-epoch imaging metrics.
            valid: Boolean array — True where all constraints met.

        Returns:
            List of ObservationWindow objects.
        """
        windows = []
        n = len(epochs)
        if n == 0:
            return windows

        in_window = False
        win_start = 0

        for i in range(n):
            if valid[i] and not in_window:
                # Window opens
                win_start = i
                in_window = True
            elif not valid[i] and in_window:
                # Window closes
                windows.append(self._build_window(
                    epochs, metrics, win_start, i - 1
                ))
                in_window = False

        # Close final window if still open
        if in_window:
            windows.append(self._build_window(
                epochs, metrics, win_start, n - 1
            ))

        return windows

    def _build_window(self, epochs, metrics, i_start, i_end):
        """Build an ObservationWindow from a contiguous index range.

        Args:
            epochs: Full epoch array.
            metrics: Full metrics list.
            i_start, i_end: Inclusive index range.

        Returns:
            ObservationWindow.
        """
        window_metrics = metrics[i_start:i_end + 1]

        peak_res = max(m.resolution_elements for m in window_metrics)
        snr_vals = [m.snr for m in window_metrics if m.snr > 0]
        mean_snr = np.mean(snr_vals) if snr_vals else 0.0

        # Identify which constraint is tightest at the boundaries
        limiting = ""
        if i_start > 0:
            m_prev = metrics[i_start - 1]
            limiting = self._identify_limiting(m_prev)
        if i_end < len(metrics) - 1:
            m_next = metrics[i_end + 1]
            lim_end = self._identify_limiting(m_next)
            if limiting:
                limiting += f" | {lim_end}"
            else:
                limiting = lim_end

        return ObservationWindow(
            start_mjd_tt=epochs[i_start],
            end_mjd_tt=epochs[i_end],
            peak_resolution_elements=peak_res,
            mean_snr=mean_snr,
            limiting_constraint=limiting
        )

    @staticmethod
    def _identify_limiting(m: ImagingMetrics) -> str:
        """Identify which constraint is most likely the limiting factor.

        Heuristic: check which constraint is violated.

        Args:
            m: Metrics at a failed epoch.

        Returns:
            Name of the likely limiting constraint.
        """
        if not m.target_sunlit:
            return "eclipse"
        if not m.pointing_feasible:
            return "pointing"
        if not m.target_in_fov:
            return "fov"
        if m.snr <= 0:
            return "snr"
        if m.resolution_elements <= 0:
            return "resolution"
        # Could be range or phase angle, but we don't have thresholds here
        return "range_or_phase"
