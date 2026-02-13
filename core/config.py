"""
Simulation configuration.

Central configuration object with toggleable force models and integrator settings.
"""

from dataclasses import dataclass, field
from .types import ShadowModel


@dataclass
class ForceModelConfig:
    """Toggleable force model configuration.

    All perturbation models can be independently enabled/disabled for
    sensitivity analysis and computational cost management.
    """
    # Gravitational perturbations
    enable_j2: bool = True
    enable_j3: bool = True
    enable_j4: bool = True
    enable_j22_tesseral: bool = True

    # Non-gravitational perturbations
    enable_srp: bool = True
    shadow_model: ShadowModel = ShadowModel.CONICAL

    # Third-body perturbations
    enable_solar_gravity: bool = False
    enable_lunar_gravity: bool = False

    def describe(self) -> str:
        """Human-readable description of active force models."""
        models = ["Two-body"]
        if self.enable_j2: models.append("J2")
        if self.enable_j3: models.append("J3")
        if self.enable_j4: models.append("J4")
        if self.enable_j22_tesseral: models.append("J22 tesseral")
        if self.enable_srp: models.append(f"SRP (shadow: {self.shadow_model.name})")
        if self.enable_solar_gravity: models.append("Solar gravity")
        if self.enable_lunar_gravity: models.append("Lunar gravity")
        return " + ".join(models)


@dataclass
class IntegratorConfig:
    """Numerical integrator configuration.

    Uses scipy's DOP853 (8th-order Dormand-Prince) by default.
    Tight tolerances are required for accurate STM integration.
    """
    method: str = "DOP853"
    rtol: float = 1e-12
    atol: float = 1e-12
    max_step_s: float = 300.0       # Maximum step size [seconds]
    dense_output: bool = True       # Enable dense output for interpolation


@dataclass
class CovarianceConfig:
    """Covariance propagation configuration.

    Attributes:
        process_noise_q: Acceleration noise spectral density [km²/s⁵].
            Typical values for GEO: 1e-15 to 1e-12.
        maneuver_mag_error_1sigma: ΔV magnitude error, fractional, 1σ.
        maneuver_point_error_1sigma_rad: Thrust pointing error [rad], 1σ.
    """
    process_noise_q: float = 1e-14          # km²/s⁵
    maneuver_mag_error_1sigma: float = 0.01 # 1% 1σ
    maneuver_point_error_1sigma_rad: float = 0.005  # ~0.3 deg 1σ


@dataclass
class SimConfig:
    """Top-level simulation configuration."""
    force_model: ForceModelConfig = field(default_factory=ForceModelConfig)
    integrator: IntegratorConfig = field(default_factory=IntegratorConfig)
    covariance: CovarianceConfig = field(default_factory=CovarianceConfig)
    output_step_s: float = 60.0     # Output cadence [seconds]
