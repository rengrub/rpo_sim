"""
Radiometric model for signal-to-noise ratio estimation.

Computes the expected signal and noise for imaging an RSO illuminated
by reflected sunlight, using a simplified Lambertian + phase-function
BRDF model.

Signal chain:
    Sun → target (reflected) → sensor aperture → focal plane → detector

Noise sources:
    - Shot noise (Poisson statistics on signal photoelectrons)
    - Detector read noise
    - Detector dark current
    - Background (zodiacal / stray light — optional)

References:
    Holst, "Electro-Optical Imaging System Performance", Ch. 4-6
    Shell, "Optimizing Orbital Debris Monitoring with Optical Telescopes"
"""

from __future__ import annotations

import numpy as np
from ..core.types import OpticalPayload
from ..core.constants import SOLAR_FLUX_1AU, AU_KM


# Planck constant * speed of light for photon energy: E = hc/lambda
_HC = 6.62607015e-34 * 2.99792458e8  # J·m


class RadiometricModel:
    """SNR calculator for reflected-sunlight RSO imaging.

    Attributes:
        optics: Optical payload parameters.
    """

    def __init__(self, optics: OpticalPayload):
        """Initialize with optical payload specification.

        Args:
            optics: Sensor parameters (aperture, QE, noise, etc.).
        """
        self.optics = optics

    def target_irradiance_at_sensor(self,
                                     albedo: float,
                                     phase_angle_rad: float,
                                     target_area_m2: float,
                                     range_km: float,
                                     solar_distance_au: float = 1.0
                                     ) -> float:
        """Spectral irradiance at the sensor from reflected sunlight.

        Uses a Lambertian BRDF with a diffuse phase function:
            phi(alpha) = (2/3pi) * [(pi - alpha) * cos(alpha) + sin(alpha)]
        where alpha is the phase angle. This is the Lommel-Seeliger /
        Lambertian sphere phase law.

        Args:
            albedo: Target geometric albedo [0-1].
            phase_angle_rad: Sun-target-chaser phase angle [rad].
            target_area_m2: Target projected cross-section [m^2].
            range_km: Chaser-to-target range [km].
            solar_distance_au: Target distance from Sun [AU].

        Returns:
            Spectral irradiance at sensor aperture [W/m^2].
        """
        range_m = range_km * 1000.0
        if range_m <= 0:
            return 0.0

        # Solar flux at target
        solar_flux = SOLAR_FLUX_1AU / solar_distance_au ** 2  # W/m^2

        # Lambertian phase function (diffuse sphere)
        alpha = np.clip(phase_angle_rad, 0.0, np.pi)
        phase_func = self._lambertian_sphere_phase(alpha)

        # Reflected power from target toward observer
        # P_reflected = albedo * solar_flux * target_area * phase_func
        # Irradiance at sensor = P_reflected / (pi * range^2)
        # The 1/pi comes from the Lambertian BRDF normalization
        # combined with the solid angle geometry.
        irradiance = (albedo * solar_flux * target_area_m2 * phase_func /
                      (np.pi * range_m ** 2))

        return irradiance

    def signal_electrons(self,
                          albedo: float,
                          phase_angle_rad: float,
                          target_area_m2: float,
                          range_km: float,
                          integration_time_s: float,
                          solar_distance_au: float = 1.0
                          ) -> float:
        """Signal photoelectrons collected during integration.

        Args:
            albedo: Target geometric albedo.
            phase_angle_rad: Sun-target-chaser phase angle [rad].
            target_area_m2: Target projected area [m^2].
            range_km: Range [km].
            integration_time_s: Detector integration time [seconds].
            solar_distance_au: Target heliocentric distance [AU].

        Returns:
            Total signal photoelectrons (summed over target pixels).
        """
        irradiance = self.target_irradiance_at_sensor(
            albedo, phase_angle_rad, target_area_m2, range_km,
            solar_distance_au
        )

        # Photon flux at detector: irradiance * aperture_area * throughput
        aperture_area = np.pi * (self.optics.aperture_diameter_m / 2.0) ** 2
        power_at_detector = irradiance * aperture_area * self.optics.optical_throughput

        # Convert power to photon rate: N_photon = P * lambda / hc
        photon_rate = power_at_detector * self.optics.wavelength_m / _HC

        # Photoelectrons = photon_rate * QE * integration_time
        signal = photon_rate * self.optics.quantum_efficiency * integration_time_s

        return max(signal, 0.0)

    def noise_electrons(self,
                         signal_e: float,
                         integration_time_s: float,
                         n_pixels_target: float,
                         background_rate_e_per_s_per_pix: float = 0.0
                         ) -> float:
        """Total noise in photoelectrons (1-sigma).

        Noise model:
            sigma = sqrt(S_signal + N_pix * (dark*t + read^2 + bg*t))

        Args:
            signal_e: Signal photoelectrons (total over target).
            integration_time_s: Integration time [s].
            n_pixels_target: Number of pixels subtended by the target.
            background_rate_e_per_s_per_pix: Sky background rate
                [electrons/s/pixel]. Defaults to 0 (dark sky).

        Returns:
            Total noise [electrons, 1-sigma].
        """
        n_pix = max(n_pixels_target, 1.0)

        shot_noise_var = signal_e  # Poisson
        dark_var = n_pix * self.optics.dark_current_e_per_s * integration_time_s
        read_var = n_pix * self.optics.read_noise_e ** 2
        bg_var = n_pix * background_rate_e_per_s_per_pix * integration_time_s

        total_var = shot_noise_var + dark_var + read_var + bg_var
        return np.sqrt(max(total_var, 1e-30))

    def snr(self,
            albedo: float,
            phase_angle_rad: float,
            target_area_m2: float,
            range_km: float,
            integration_time_s: float,
            n_pixels_target: float,
            solar_distance_au: float = 1.0,
            background_rate: float = 0.0
            ) -> float:
        """Signal-to-noise ratio for a single integration.

        Args:
            albedo: Target geometric albedo.
            phase_angle_rad: Phase angle [rad].
            target_area_m2: Target projected area [m^2].
            range_km: Range [km].
            integration_time_s: Integration time [s].
            n_pixels_target: Pixels subtended by target.
            solar_distance_au: Heliocentric distance [AU].
            background_rate: Background [e/s/pixel].

        Returns:
            SNR (dimensionless).
        """
        sig = self.signal_electrons(
            albedo, phase_angle_rad, target_area_m2, range_km,
            integration_time_s, solar_distance_au
        )
        noise = self.noise_electrons(
            sig, integration_time_s, n_pixels_target, background_rate
        )
        if noise <= 0:
            return 0.0
        return sig / noise

    def minimum_integration_time(self,
                                  snr_threshold: float,
                                  albedo: float,
                                  phase_angle_rad: float,
                                  target_area_m2: float,
                                  range_km: float,
                                  n_pixels_target: float,
                                  solar_distance_au: float = 1.0,
                                  background_rate: float = 0.0,
                                  max_time_s: float = 60.0
                                  ) -> float:
        """Find minimum integration time to achieve a desired SNR.

        Uses a bisection search since the SNR equation (with read noise)
        is not analytically invertible.

        Args:
            snr_threshold: Desired minimum SNR.
            albedo: Target albedo.
            phase_angle_rad: Phase angle [rad].
            target_area_m2: Target area [m^2].
            range_km: Range [km].
            n_pixels_target: Pixels on target.
            solar_distance_au: Heliocentric distance [AU].
            background_rate: Background rate [e/s/pixel].
            max_time_s: Maximum allowed integration time [s].

        Returns:
            Minimum integration time [s], or np.inf if threshold
            cannot be achieved within max_time_s.
        """
        # Quick check: can we meet threshold at max time?
        snr_max = self.snr(albedo, phase_angle_rad, target_area_m2,
                           range_km, max_time_s, n_pixels_target,
                           solar_distance_au, background_rate)
        if snr_max < snr_threshold:
            return np.inf

        # Bisection
        t_lo = 1e-6
        t_hi = max_time_s

        for _ in range(60):  # sufficient for double-precision convergence
            t_mid = 0.5 * (t_lo + t_hi)
            snr_mid = self.snr(albedo, phase_angle_rad, target_area_m2,
                               range_km, t_mid, n_pixels_target,
                               solar_distance_au, background_rate)
            if snr_mid < snr_threshold:
                t_lo = t_mid
            else:
                t_hi = t_mid

            if (t_hi - t_lo) < 1e-9:
                break

        return t_hi

    @staticmethod
    def _lambertian_sphere_phase(alpha: float) -> float:
        """Lambertian sphere phase function.

        Normalized so that phi(0) = 1 (full illumination at zero phase angle).

            phi(alpha) = (1/pi) * [(pi - alpha) * cos(alpha) + sin(alpha)]

        This is the exact phase integral for a Lambertian sphere.

        Args:
            alpha: Phase angle [rad], in [0, pi].

        Returns:
            Phase function value in [0, 1].
        """
        return (1.0 / np.pi) * ((np.pi - alpha) * np.cos(alpha) +
                                 np.sin(alpha))
