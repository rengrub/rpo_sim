"""
Differential correction via shooting methods.

Bridges HCW initial guesses to high-fidelity ECI solutions.
Uses the STM (integrated alongside the state) to compute the
Jacobian for Newton-Raphson iteration.
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from ..core.types import (
    AugmentedState, SpacecraftState, PropagationResult,
    CorrectionResult, Maneuver, ManeuverType
)
from ..core.config import SimConfig
from ..core.constants import SECONDS_PER_DAY
from ..core.frames import eci_to_lvlh, relative_state_eci_to_lvlh
from ..astrodynamics.propagator import Propagator


class SingleShooter:
    """Single-leg differential corrector.

    Solves for the impulsive delta-V at a maneuver epoch such that the
    chaser arrives at a desired relative position at a target epoch,
    under the full nonlinear ECI force model.

    The 3x3 system is:
        Free variables:  dv = [dvx, dvy, dvz]_ECI at t_man
        Constraints:     c = dr_LVLH(t_target) - dr_desired = 0

    Jacobian:  dc/ddv = R_ECI_to_LVLH(t_target) * Phi_rv(t_target, t_man)
    where Phi_rv is the upper-right 3x3 block of the STM.

    Attributes:
        propagator: Numerical propagator instance.
        config: Simulation configuration.
    """

    def __init__(self, propagator: Propagator, config: SimConfig):
        """Initialize the single-leg corrector.

        Args:
            propagator: Configured Propagator instance.
            config: Simulation configuration.
        """
        self.propagator = propagator
        self.config = config

    def correct_leg(self,
                    chaser_state: SpacecraftState,
                    dv_guess_eci: np.ndarray,
                    target_rel_pos_lvlh: np.ndarray,
                    t_arrival_mjd_tt: float,
                    target_state_at_arrival: np.ndarray,
                    cr: float,
                    area_over_mass: float,
                    tol_km: float = 1e-3,
                    max_iter: int = 20
                    ) -> CorrectionResult:
        """Correct a single trajectory leg via Newton-Raphson shooting.

        Args:
            chaser_state: Chaser state at maneuver epoch (pre-maneuver).
            dv_guess_eci: Initial guess delta-V in ECI [km/s], shape (3,).
            target_rel_pos_lvlh: Desired relative position at arrival
                in target LVLH frame [km], shape (3,).
            t_arrival_mjd_tt: Arrival epoch [MJD TT].
            target_state_at_arrival: Target ECI state [r,v] at arrival, shape (6,).
            cr: SRP reflectivity coefficient for chaser.
            area_over_mass: Chaser area-to-mass ratio [m^2/kg].
            tol_km: Position convergence tolerance [km].
            max_iter: Maximum Newton iterations.

        Returns:
            CorrectionResult with converged delta-V and trajectory.
        """
        dv = dv_guess_eci.copy()
        duration_s = (t_arrival_mjd_tt - chaser_state.epoch_mjd_tt) * SECONDS_PER_DAY

        r_tgt = target_state_at_arrival[0:3]
        v_tgt = target_state_at_arrival[3:6]

        for iteration in range(max_iter):
            # Apply current dv guess
            aug = AugmentedState(
                sc_state=SpacecraftState(
                    epoch_mjd_tt=chaser_state.epoch_mjd_tt,
                    position=chaser_state.position.copy(),
                    velocity=chaser_state.velocity + dv,
                    mass=chaser_state.mass
                ),
                stm=np.eye(6)
            )

            # Propagate to arrival
            result = self.propagator.propagate(
                aug, duration_s, cr, area_over_mass
            )

            # Extract terminal state
            r_chaser_f = result.states[-1, 0:3]
            v_chaser_f = result.states[-1, 3:6]
            stm_f = result.stms[-1]

            # Compute relative position at arrival in LVLH
            rel = relative_state_eci_to_lvlh(
                r_chaser_f, v_chaser_f, r_tgt, v_tgt, t_arrival_mjd_tt
            )

            # Constraint violation
            constraint = rel.position_lvlh - target_rel_pos_lvlh
            error_km = np.linalg.norm(constraint)

            if error_km < tol_km:
                return CorrectionResult(
                    converged=True,
                    iterations=iteration + 1,
                    final_error_km=error_km,
                    dv_corrected_eci=dv.copy(),
                    trajectory=result
                )

            # Jacobian: dc/ddv = R_LVLH * Phi_rv
            R_lvlh, _ = eci_to_lvlh(r_tgt, v_tgt)
            phi_rv = stm_f[0:3, 3:6]  # Upper-right 3x3 of STM
            jacobian = R_lvlh @ phi_rv

            # Newton update
            dv_update = np.linalg.solve(jacobian, -constraint)
            dv += dv_update

        # Did not converge
        return CorrectionResult(
            converged=False,
            iterations=max_iter,
            final_error_km=error_km,
            dv_corrected_eci=dv.copy(),
            trajectory=result
        )


class MultiLegCorrector:
    """Sequential multi-leg differential corrector.

    Corrects each leg in sequence: the converged terminal state
    of leg N becomes the initial state of leg N+1.

    Attributes:
        shooter: Single-leg corrector.
    """

    def __init__(self, propagator: Propagator, config: SimConfig):
        """Initialize the multi-leg corrector.

        Args:
            propagator: Configured Propagator instance.
            config: Simulation configuration.
        """
        self.shooter = SingleShooter(propagator, config)
        self.config = config

    def correct_sequence(self,
                         initial_chaser_state: SpacecraftState,
                         legs: list[dict],
                         target_trajectory: PropagationResult,
                         cr: float,
                         area_over_mass: float,
                         tol_km: float = 1e-3
                         ) -> list[CorrectionResult]:
        """Correct a multi-leg approach trajectory.

        Each leg dict contains:
            'dv_guess_eci': Initial guess delta-V [km/s], shape (3,).
            'target_rel_pos_lvlh': Desired relative position [km], shape (3,).
            't_arrival_mjd_tt': Arrival epoch.

        Args:
            initial_chaser_state: Chaser state at first maneuver epoch.
            legs: List of leg definition dicts.
            target_trajectory: Pre-propagated target trajectory for
                state interpolation at leg boundaries.
            cr: Chaser SRP reflectivity coefficient.
            area_over_mass: Chaser area-to-mass ratio [m^2/kg].
            tol_km: Convergence tolerance per leg [km].

        Returns:
            List of CorrectionResult, one per leg.
        """
        results = []
        current_state = initial_chaser_state

        for leg in legs:
            t_arrival = leg['t_arrival_mjd_tt']

            # Get target state at arrival via interpolation
            target_state = target_trajectory.state_at(t_arrival)

            result = self.shooter.correct_leg(
                chaser_state=current_state,
                dv_guess_eci=leg['dv_guess_eci'],
                target_rel_pos_lvlh=leg['target_rel_pos_lvlh'],
                t_arrival_mjd_tt=t_arrival,
                target_state_at_arrival=target_state,
                cr=cr,
                area_over_mass=area_over_mass,
                tol_km=tol_km
            )
            results.append(result)

            if not result.converged:
                break  # Stop if a leg fails to converge

            # Update chaser state for next leg
            final_state = result.trajectory.states[-1]
            final_mass = result.trajectory.masses[-1]
            current_state = SpacecraftState(
                epoch_mjd_tt=t_arrival,
                position=final_state[0:3].copy(),
                velocity=final_state[3:6].copy(),
                mass=final_mass
            )

        return results
