"""
Numerical orbit propagator.

Wraps scipy.integrate.solve_ivp (DOP853) with:
    - Augmented state integration (state + mass + STM)
    - Event detection (shadow boundaries, maneuver epochs)
    - Maneuver handling (impulsive: stop/apply/restart; finite: EOM switch)
    - Dense output for interpolation
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from typing import Optional, Callable

from ..core.types import (
    AugmentedState, SpacecraftState, PropagationResult,
    Maneuver, ManeuverType, FiniteBurnProfile
)
from ..core.config import SimConfig
from ..core.constants import SECONDS_PER_DAY
from .eom import eom_full, make_thrust_func


class Propagator:
    """Numerical orbit propagator for ECI trajectories with STM.

    Integrates the 43-element augmented state vector (position, velocity,
    mass, and 6x6 STM) using an adaptive 8th-order Dormand-Prince method.

    Attributes:
        config: Simulation configuration.
    """

    def __init__(self, config: SimConfig):
        """Initialize the propagator.

        Args:
            config: Simulation configuration with integrator and force model settings.
        """
        self.config = config

    def propagate(self,
                  state: AugmentedState,
                  duration_s: float,
                  cr: float,
                  area_over_mass: float,
                  maneuvers: Optional[list[Maneuver]] = None,
                  output_step_s: Optional[float] = None,
                  events: Optional[list[Callable]] = None
                  ) -> PropagationResult:
        """Propagate an augmented state forward in time.

        Handles impulsive maneuvers by stopping integration, applying the
        delta-V, resetting the STM, and restarting. Finite burns are
        handled by switching the EOM to include thrust.

        Args:
            state: Initial augmented state (position, velocity, mass, STM).
            duration_s: Total propagation duration [seconds].
            cr: SRP reflectivity coefficient.
            area_over_mass: Area-to-mass ratio [m^2/kg].
            maneuvers: Optional list of maneuvers to apply during propagation.
            output_step_s: Output cadence [seconds]. Defaults to config value.
            events: Optional list of event functions for solve_ivp.

        Returns:
            PropagationResult with time-tagged states, masses, and STMs.
        """
        if output_step_s is None:
            output_step_s = self.config.output_step_s

        epoch_ref = state.sc_state.epoch_mjd_tt
        t_start = 0.0
        t_end = duration_s

        # Sort maneuvers by epoch
        if maneuvers:
            maneuvers = sorted(maneuvers, key=lambda m: m.epoch_mjd_tt)

        # Build list of integration segments separated by impulsive maneuvers
        segments = self._build_segments(t_start, t_end, maneuvers, epoch_ref)

        # Accumulate results across segments
        all_epochs = []
        all_states = []
        all_masses = []
        all_stms = []
        accumulated_stm = np.eye(6)

        current_y = state.to_flat_vector()
        current_t = t_start

        for segment in segments:
            seg_t_start = segment['t_start']
            seg_t_end = segment['t_end']
            seg_thrust = segment.get('thrust_func', None)

            if seg_t_end <= seg_t_start:
                continue

            # Output times for this segment
            t_eval = np.arange(seg_t_start, seg_t_end, output_step_s)
            if len(t_eval) == 0 or t_eval[-1] < seg_t_end:
                t_eval = np.append(t_eval, seg_t_end)

            # Integrate
            result = self._integrate_segment(
                current_y, seg_t_start, seg_t_end,
                cr, area_over_mass, epoch_ref,
                t_eval=t_eval,
                thrust_func=seg_thrust,
                events=events
            )

            # Store results
            for j in range(len(result.t)):
                t_j = result.t[j]
                y_j = result.y[:, j]
                epoch_j = epoch_ref + t_j / SECONDS_PER_DAY

                all_epochs.append(epoch_j)
                all_states.append(y_j[0:6])
                all_masses.append(y_j[6])

                seg_stm = y_j[7:43].reshape(6, 6)
                # Compose with accumulated STM from prior segments
                all_stms.append(seg_stm @ accumulated_stm)

            # Update for next segment
            current_y = result.y[:, -1].copy()
            accumulated_stm = current_y[7:43].reshape(6, 6) @ accumulated_stm

            # Apply impulsive maneuver if one starts this segment boundary
            if 'maneuver' in segment and segment['maneuver'] is not None:
                man = segment['maneuver']
                if man.maneuver_type == ManeuverType.IMPULSIVE:
                    current_y[3:6] += man.dv_vector_eci
                    # Reset STM to identity for next segment
                    current_y[7:43] = np.eye(6).flatten()

        return PropagationResult(
            epochs=np.array(all_epochs),
            states=np.array(all_states),
            masses=np.array(all_masses),
            stms=all_stms,
            dense_output=None  # Could store from last segment if needed
        )

    def propagate_segment(self,
                          y0: np.ndarray,
                          t_start: float,
                          t_end: float,
                          cr: float,
                          area_over_mass: float,
                          epoch_ref: float,
                          thrust_func=None
                          ) -> object:
        """Low-level single-segment propagation.

        Args:
            y0: Initial augmented state vector, shape (43,).
            t_start: Segment start time [seconds since epoch_ref].
            t_end: Segment end time [seconds since epoch_ref].
            cr: SRP reflectivity coefficient.
            area_over_mass: Area-to-mass ratio [m^2/kg].
            epoch_ref: Reference epoch [MJD TT].
            thrust_func: Optional thrust function for finite burns.

        Returns:
            scipy OdeResult object.
        """
        return self._integrate_segment(
            y0, t_start, t_end,
            cr, area_over_mass, epoch_ref,
            thrust_func=thrust_func
        )

    def _integrate_segment(self, y0, t_start, t_end,
                           cr, area_over_mass, epoch_ref,
                           t_eval=None, thrust_func=None,
                           events=None):
        """Core integration call wrapping scipy.integrate.solve_ivp.

        Args:
            y0: Initial state vector (43,).
            t_start, t_end: Time span [seconds since epoch_ref].
            cr, area_over_mass: Spacecraft SRP parameters.
            epoch_ref: Reference epoch [MJD TT].
            t_eval: Specific output times.
            thrust_func: Thrust callable or None.
            events: Event functions.

        Returns:
            scipy OdeResult.
        """
        cfg = self.config

        def rhs(t, y):
            return eom_full(
                t, y, cfg, cr, area_over_mass, epoch_ref,
                thrust_func=thrust_func
            )

        result = solve_ivp(
            rhs,
            t_span=(t_start, t_end),
            y0=y0,
            method=cfg.integrator.method,
            rtol=cfg.integrator.rtol,
            atol=cfg.integrator.atol,
            max_step=cfg.integrator.max_step_s,
            dense_output=cfg.integrator.dense_output,
            t_eval=t_eval,
            events=events
        )

        if not result.success:
            raise RuntimeError(
                f"Integration failed: {result.message} "
                f"(t_start={t_start:.1f}, t_end={t_end:.1f})"
            )

        return result

    def _build_segments(self, t_start, t_end, maneuvers, epoch_ref):
        """Build integration segments separated by maneuver events.

        Each segment is a dict with t_start, t_end, and optional maneuver
        to apply at the segment boundary.

        Args:
            t_start, t_end: Overall propagation time span [seconds].
            maneuvers: Sorted list of Maneuver objects.
            epoch_ref: Reference epoch [MJD TT].

        Returns:
            List of segment dicts.
        """
        segments = []
        current_t = t_start

        if maneuvers is None:
            maneuvers = []

        for man in maneuvers:
            t_man = (man.epoch_mjd_tt - epoch_ref) * SECONDS_PER_DAY

            if t_man < t_start or t_man > t_end:
                continue

            if man.maneuver_type == ManeuverType.IMPULSIVE:
                # Coast to maneuver epoch
                segments.append({
                    't_start': current_t,
                    't_end': t_man,
                    'maneuver': man
                })
                current_t = t_man

            elif man.maneuver_type in (ManeuverType.FINITE_BURN_FIXED,
                                       ManeuverType.FINITE_BURN_STEERED):
                # Coast to burn start
                burn = man.burn_profile
                t_burn_start = (burn.t_start_mjd_tt - epoch_ref) * SECONDS_PER_DAY
                t_burn_end = t_burn_start + burn.duration_s

                segments.append({
                    't_start': current_t,
                    't_end': t_burn_start,
                })

                # Burn segment
                thrust_func = make_thrust_func(burn, epoch_ref)
                segments.append({
                    't_start': t_burn_start,
                    't_end': t_burn_end,
                    'thrust_func': thrust_func,
                })

                current_t = t_burn_end

        # Final coast to end
        segments.append({
            't_start': current_t,
            't_end': t_end,
        })

        return segments
