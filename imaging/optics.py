"""
Optical system performance model.

Computes spatial resolution, diffraction limits, and effective point spread
function (PSF) characteristics for resolved imaging of an RSO.

The optical system parameters come from ``OpticalPayload`` in core/types.py.
This module wraps those parameters with range-dependent performance methods.

All internal units: SI (meters, radians, seconds).
Range inputs accept km and convert internally.
"""

from __future__ import annotations

import numpy as np
from ..core.types import OpticalPayload


class OpticalSystem:
    """Range-dependent optical performance calculator.

    Wraps an ``OpticalPayload`` specification and adds methods that evaluate
    imaging performance as a function of target range, pointing stability,
    and target geometry.

    Attributes:
        optics: Underlying payload specification.
    """

    def __init__(self, optics: OpticalPayload):
        """Initialize from an optical payload definition.

        Args:
            optics: Sensor hardware parameters.
        """
        self.optics = optics

    # ------------------------------------------------------------------
    # Angular resolution properties
    # ------------------------------------------------------------------

    @property
    def ifov_rad(self) -> float:
        """Instantaneous field of view per pixel [rad]."""
        return self.optics.ifov_rad

    @property
    def diffraction_limit_rad(self) -> float:
        """Rayleigh diffraction-limited angular resolution [rad]."""
        return self.optics.diffraction_limit_rad

    @property
    def angular_resolution_rad(self) -> float:
        """System angular resolution [rad].

        The larger of the pixel IFOV and the diffraction limit determines
        whether the system is detector-limited or optics-limited.
        """
        return max(self.ifov_rad, self.diffraction_limit_rad)

    @property
    def fov_half_angle_rad(self) -> float:
        """Detector FOV half-angle [rad]."""
        return self.optics.fov_half_angle_rad

    @property
    def collecting_area_m2(self) -> float:
        """Primary aperture collecting area [m^2]."""
        return np.pi * (self.optics.aperture_diameter_m / 2.0) ** 2

    # ------------------------------------------------------------------
    # Range-dependent metrics
    # ------------------------------------------------------------------

    def gsd_m(self, range_km: float) -> float:
        """Ground Sample Distance (linear resolution on target).

        The smallest resolvable feature size, limited by the worse of
        pixel sampling and diffraction.

        Args:
            range_km: Chaser-to-target range [km].

        Returns:
            GSD [meters].
        """
        range_m = range_km * 1000.0
        return range_m * self.angular_resolution_rad

    def resolution_elements_on_target(self, range_km: float,
                                       target_size_m: float) -> float:
        """Number of resolution elements spanning the target.

        This is the key metric for resolved imaging: a target must span
        at least ~3-5 resolution elements for meaningful feature
        discrimination.

        Args:
            range_km: Chaser-to-target range [km].
            target_size_m: Target characteristic dimension [meters].

        Returns:
            Number of resolution elements across the target.
        """
        gsd = self.gsd_m(range_km)
        if gsd <= 0:
            return 0.0
        return target_size_m / gsd

    def effective_angular_resolution_rad(self,
                                          pointing_stability_rad: float
                                          ) -> float:
        """Effective angular resolution including pointing jitter.

        The PSF broadens due to pointing motion during integration.
        Modeled as RSS of diffraction PSF and pointing stability:
            theta_eff = sqrt(theta_diff^2 + sigma_point^2)

        Args:
            pointing_stability_rad: 1-sigma pointing stability during
                integration [rad].

        Returns:
            Effective angular resolution [rad].
        """
        theta_sys = self.angular_resolution_rad
        return np.sqrt(theta_sys ** 2 + pointing_stability_rad ** 2)

    def effective_gsd_m(self, range_km: float,
                        pointing_stability_rad: float) -> float:
        """Effective GSD including pointing degradation.

        Args:
            range_km: Range to target [km].
            pointing_stability_rad: Pointing stability [rad, 1-sigma].

        Returns:
            Effective GSD [meters].
        """
        range_m = range_km * 1000.0
        theta_eff = self.effective_angular_resolution_rad(pointing_stability_rad)
        return range_m * theta_eff

    def effective_resolution_elements(self, range_km: float,
                                       target_size_m: float,
                                       pointing_stability_rad: float
                                       ) -> float:
        """Resolution elements on target accounting for pointing.

        Args:
            range_km: Range [km].
            target_size_m: Target size [m].
            pointing_stability_rad: Pointing stability [rad, 1-sigma].

        Returns:
            Effective resolution elements across target.
        """
        gsd = self.effective_gsd_m(range_km, pointing_stability_rad)
        if gsd <= 0:
            return 0.0
        return target_size_m / gsd

    def target_angular_size_rad(self, range_km: float,
                                 target_size_m: float) -> float:
        """Angular subtense of the target as seen from the chaser.

        Args:
            range_km: Range [km].
            target_size_m: Target characteristic dimension [m].

        Returns:
            Target angular size [rad].
        """
        range_m = range_km * 1000.0
        if range_m <= 0:
            return np.pi
        return target_size_m / range_m

    def pixels_on_target(self, range_km: float,
                          target_size_m: float) -> float:
        """Number of detector pixels spanning the target.

        Independent of diffraction — purely geometric projection onto
        the focal plane.

        Args:
            range_km: Range [km].
            target_size_m: Target size [m].

        Returns:
            Number of pixels across the target.
        """
        angular_size = self.target_angular_size_rad(range_km, target_size_m)
        if self.ifov_rad <= 0:
            return 0.0
        return angular_size / self.ifov_rad

    def max_imaging_range_km(self, target_size_m: float,
                              min_resolution_elements: float) -> float:
        """Maximum range at which target meets resolution requirement.

        Args:
            target_size_m: Target characteristic dimension [m].
            min_resolution_elements: Minimum required resolution elements.

        Returns:
            Maximum range [km].
        """
        if min_resolution_elements <= 0 or self.angular_resolution_rad <= 0:
            return np.inf
        # resolution_elements = target_size / (range * theta)
        # range_max = target_size / (min_res_el * theta)
        range_m = target_size_m / (min_resolution_elements *
                                    self.angular_resolution_rad)
        return range_m / 1000.0
