"""
Pointing constraint evaluation.

Models physical constraints on spacecraft attitude that limit when
imaging is feasible:
    - Sun exclusion: sensor cannot point within N degrees of the sun
    - Earth exclusion: sensor cannot point within N degrees of Earth limb
    - Body-axis sun constraint: e.g., solar array must maintain sun angle
    - Gimbal limits: for gimbaled sensors, maximum articulation angle

Each constraint is evaluated as a scalar margin (positive = satisfied,
negative = violated) enabling both boolean feasibility checks and
continuous constraint tracking.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from .quaternion import q_to_dcm


@dataclass
class ConstraintResult:
    """Result of evaluating all pointing constraints at a single epoch.

    Attributes:
        all_satisfied: True if all constraints have positive margin.
        margins_rad: Dict mapping constraint name to margin [rad].
            Positive = satisfied by that margin. Negative = violated.
        limiting_constraint: Name of the constraint with smallest margin.
    """
    all_satisfied: bool
    margins_rad: dict[str, float]
    limiting_constraint: str


@dataclass
class FeasibilityTimeline:
    """Constraint feasibility evaluated over a trajectory.

    Attributes:
        epochs: Evaluation epochs [MJD TT], shape (N,).
        feasible: Boolean array, shape (N,). True where all constraints met.
        margins: Dict of constraint name -> margin array shape (N,).
        limiting: List of limiting constraint name at each epoch.
    """
    epochs: np.ndarray
    feasible: np.ndarray
    margins: dict[str, np.ndarray]
    limiting: list[str]


@dataclass
class _ExclusionCone:
    """Internal representation of an exclusion zone constraint.

    Attributes:
        name: Constraint identifier.
        body_axis: Body-frame axis to evaluate (e.g., sensor boresight), shape (3,).
        min_angle_rad: Minimum allowed angle between body_axis and the
            exclusion direction [rad]. The constraint is satisfied when
            the angle exceeds this value.
        direction_type: 'sun', 'earth', or 'custom'.
    """
    name: str
    body_axis: np.ndarray
    min_angle_rad: float
    direction_type: str


@dataclass
class _BodyAxisSunConstraint:
    """Body axis must maintain a sun angle within [min, max] bounds.

    Used for solar array pointing: the array normal must stay within
    a cone around the sun direction to maintain power.

    Attributes:
        name: Constraint identifier.
        body_axis: Body-frame axis (e.g., solar array normal), shape (3,).
        min_angle_rad: Minimum sun angle [rad].
        max_angle_rad: Maximum sun angle [rad].
    """
    name: str
    body_axis: np.ndarray
    min_angle_rad: float
    max_angle_rad: float


class PointingConstraintSet:
    """Collection of pointing constraints evaluated against spacecraft attitude.

    Build a constraint set by adding individual constraints, then evaluate
    against an attitude and environment geometry.

    Example:
        constraints = PointingConstraintSet()
        constraints.add_sun_exclusion(
            min_angle_rad=np.deg2rad(30),
            body_axis=np.array([0, 0, 1])  # sensor boresight
        )
        constraints.add_body_axis_sun_constraint(
            body_axis=np.array([0, 1, 0]),  # solar array normal
            min_angle_rad=np.deg2rad(0),
            max_angle_rad=np.deg2rad(30)
        )
        result = constraints.evaluate(q, r_sat, r_sun)
    """

    def __init__(self):
        """Initialize an empty constraint set."""
        self._exclusion_cones: list[_ExclusionCone] = []
        self._body_sun_constraints: list[_BodyAxisSunConstraint] = []

    def add_sun_exclusion(self, min_angle_rad: float,
                          body_axis: np.ndarray,
                          name: Optional[str] = None):
        """Add a sun exclusion constraint.

        The angle between body_axis and the sun direction (in body frame)
        must exceed min_angle_rad.

        Args:
            min_angle_rad: Minimum allowed angle to sun [rad].
            body_axis: Body-frame axis to protect (e.g., sensor boresight).
            name: Optional constraint name.
        """
        if name is None:
            name = f"sun_exclusion_{len(self._exclusion_cones)}"
        self._exclusion_cones.append(_ExclusionCone(
            name=name,
            body_axis=body_axis / np.linalg.norm(body_axis),
            min_angle_rad=min_angle_rad,
            direction_type='sun'
        ))

    def add_earth_exclusion(self, min_angle_rad: float,
                            body_axis: np.ndarray,
                            name: Optional[str] = None):
        """Add an Earth exclusion constraint.

        The angle between body_axis and the nadir direction (toward Earth
        center) must exceed min_angle_rad. Useful for preventing stray
        light from Earth's limb.

        Args:
            min_angle_rad: Minimum allowed angle to nadir [rad].
            body_axis: Body-frame axis to protect.
            name: Optional constraint name.
        """
        if name is None:
            name = f"earth_exclusion_{len(self._exclusion_cones)}"
        self._exclusion_cones.append(_ExclusionCone(
            name=name,
            body_axis=body_axis / np.linalg.norm(body_axis),
            min_angle_rad=min_angle_rad,
            direction_type='earth'
        ))

    def add_body_axis_sun_constraint(self,
                                      body_axis: np.ndarray,
                                      min_angle_rad: float,
                                      max_angle_rad: float,
                                      name: Optional[str] = None):
        """Add a body-axis-to-sun angle band constraint.

        The angle between body_axis and the sun direction must be within
        [min_angle_rad, max_angle_rad]. Used for solar array keep-alive.

        Args:
            body_axis: Body-frame axis (e.g., solar array normal).
            min_angle_rad: Minimum sun angle [rad].
            max_angle_rad: Maximum sun angle [rad].
            name: Optional constraint name.
        """
        if name is None:
            name = f"body_sun_{len(self._body_sun_constraints)}"
        self._body_sun_constraints.append(_BodyAxisSunConstraint(
            name=name,
            body_axis=body_axis / np.linalg.norm(body_axis),
            min_angle_rad=min_angle_rad,
            max_angle_rad=max_angle_rad
        ))

    def evaluate(self, q_eci_to_body: np.ndarray,
                 r_sat_eci: np.ndarray,
                 r_sun_eci: np.ndarray) -> ConstraintResult:
        """Evaluate all constraints at a single epoch.

        Args:
            q_eci_to_body: ECI-to-body quaternion, shape (4,).
            r_sat_eci: Spacecraft ECI position [km], shape (3,).
            r_sun_eci: Sun ECI position [km], shape (3,).

        Returns:
            ConstraintResult with per-constraint margins.
        """
        R_eci_to_body = q_to_dcm(q_eci_to_body)

        # Precompute directions in body frame
        # Sun direction in body frame
        sun_eci = r_sun_eci - r_sat_eci
        sun_eci_hat = sun_eci / np.linalg.norm(sun_eci)
        sun_body = R_eci_to_body @ sun_eci_hat

        # Earth (nadir) direction in body frame
        nadir_eci = -r_sat_eci
        nadir_mag = np.linalg.norm(nadir_eci)
        if nadir_mag > 1e-10:
            nadir_body = R_eci_to_body @ (nadir_eci / nadir_mag)
        else:
            nadir_body = np.array([0., 0., -1.])

        margins = {}
        all_ok = True

        # --- Exclusion cones ---
        for cone in self._exclusion_cones:
            if cone.direction_type == 'sun':
                direction_body = sun_body
            elif cone.direction_type == 'earth':
                direction_body = nadir_body
            else:
                continue

            cos_angle = np.clip(np.dot(cone.body_axis, direction_body), -1.0, 1.0)
            angle = np.arccos(cos_angle)

            margin = angle - cone.min_angle_rad
            margins[cone.name] = margin

            if margin < 0:
                all_ok = False

        # --- Body-axis sun band constraints ---
        for bsc in self._body_sun_constraints:
            cos_angle = np.clip(np.dot(bsc.body_axis, sun_body), -1.0, 1.0)
            angle = np.arccos(cos_angle)

            # Margin is distance to nearest bound (positive = inside band)
            margin_low = angle - bsc.min_angle_rad
            margin_high = bsc.max_angle_rad - angle
            margin = min(margin_low, margin_high)
            margins[bsc.name] = margin

            if margin < 0:
                all_ok = False

        # Find limiting constraint
        if margins:
            limiting = min(margins, key=margins.get)
        else:
            limiting = "none"

        return ConstraintResult(
            all_satisfied=all_ok,
            margins_rad=margins,
            limiting_constraint=limiting
        )

    def evaluate_over_trajectory(
            self,
            quaternions: np.ndarray,
            r_sat_eci_history: np.ndarray,
            r_sun_eci_history: np.ndarray,
            epochs: np.ndarray
    ) -> FeasibilityTimeline:
        """Evaluate all constraints at every epoch along a trajectory.

        Args:
            quaternions: Attitude quaternions, shape (N, 4).
            r_sat_eci_history: Spacecraft ECI positions, shape (N, 3).
            r_sun_eci_history: Sun ECI positions, shape (N, 3).
            epochs: Epoch array [MJD TT], shape (N,).

        Returns:
            FeasibilityTimeline with boolean and margin histories.
        """
        n = len(epochs)
        feasible = np.zeros(n, dtype=bool)
        margin_arrays: dict[str, list[float]] = {}
        limiting_list: list[str] = []

        # Initialize margin arrays from constraint names
        all_names = ([c.name for c in self._exclusion_cones]
                     + [c.name for c in self._body_sun_constraints])
        for name in all_names:
            margin_arrays[name] = []

        for i in range(n):
            result = self.evaluate(
                quaternions[i], r_sat_eci_history[i], r_sun_eci_history[i]
            )
            feasible[i] = result.all_satisfied
            limiting_list.append(result.limiting_constraint)

            for name in all_names:
                margin_arrays[name].append(result.margins_rad.get(name, 0.0))

        # Convert to numpy arrays
        margin_np = {name: np.array(vals) for name, vals in margin_arrays.items()}

        return FeasibilityTimeline(
            epochs=epochs.copy(),
            feasible=feasible,
            margins=margin_np,
            limiting=limiting_list,
        )
