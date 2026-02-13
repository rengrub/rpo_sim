"""
Foundational data types for the RPO simulation.

All state and configuration data flows through these well-defined dataclasses.
Convention:
    - Distances: km
    - Time: seconds (integration), MJD TT (epochs)
    - Velocity: km/s
    - Mass: kg
    - Angles: radians
    - Thrust: Newtons
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Callable


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class FrameType(Enum):
    """Reference frame identifiers."""
    ECI_J2000 = auto()
    ECEF_ITRF = auto()
    LVLH = auto()       # Target-centered local-vertical-local-horizontal
    RSW = auto()        # Radial / Along-track / Cross-track (alias for LVLH)
    BODY = auto()       # Spacecraft body frame


class ManeuverType(Enum):
    """Maneuver fidelity type."""
    IMPULSIVE = auto()
    FINITE_BURN_FIXED = auto()      # Fixed inertial thrust direction
    FINITE_BURN_STEERED = auto()    # Velocity-to-be-gained steering


class ShadowModel(Enum):
    """Eclipse shadow model type."""
    NONE = auto()
    CYLINDRICAL = auto()
    CONICAL = auto()


# ---------------------------------------------------------------------------
# Spacecraft and Target State
# ---------------------------------------------------------------------------

@dataclass
class SpacecraftState:
    """ECI state of a spacecraft at a given epoch.

    Attributes:
        epoch_mjd_tt: Modified Julian Date in Terrestrial Time.
        position: ECI position vector [km], shape (3,).
        velocity: ECI velocity vector [km/s], shape (3,).
        mass: Spacecraft wet mass [kg].
    """
    epoch_mjd_tt: float
    position: np.ndarray        # (3,) km
    velocity: np.ndarray        # (3,) km/s
    mass: float                 # kg

    @property
    def state_vector(self) -> np.ndarray:
        """Combined [r, v] state vector, shape (6,)."""
        return np.concatenate([self.position, self.velocity])

    @state_vector.setter
    def state_vector(self, rv: np.ndarray):
        self.position = rv[:3].copy()
        self.velocity = rv[3:6].copy()


@dataclass
class AugmentedState:
    """Spacecraft state augmented with the 6x6 State Transition Matrix.

    The STM maps perturbations at the reference epoch to the current epoch.
    Total integrated vector length: 6 (state) + 1 (mass) + 36 (STM) = 43.

    Attributes:
        sc_state: The spacecraft translational state and mass.
        stm: 6x6 State Transition Matrix, shape (6,6).
    """
    sc_state: SpacecraftState
    stm: np.ndarray = field(default_factory=lambda: np.eye(6))  # (6,6)

    def to_flat_vector(self) -> np.ndarray:
        """Pack into the 43-element integration vector.

        Layout: [x,y,z, vx,vy,vz, mass, Φ_11..Φ_16, Φ_21..Φ_66]
        """
        return np.concatenate([
            self.sc_state.state_vector,     # 6
            [self.sc_state.mass],           # 1
            self.stm.flatten()              # 36
        ])

    @classmethod
    def from_flat_vector(cls, y: np.ndarray, epoch_mjd_tt: float) -> AugmentedState:
        """Unpack from the 43-element integration vector."""
        sc = SpacecraftState(
            epoch_mjd_tt=epoch_mjd_tt,
            position=y[0:3].copy(),
            velocity=y[3:6].copy(),
            mass=y[6]
        )
        stm = y[7:43].reshape(6, 6).copy()
        return cls(sc_state=sc, stm=stm)


@dataclass
class AttitudeState:
    """Spacecraft attitude at a given epoch.

    Convention: quaternion q = [q1, q2, q3, q4] with q4 as scalar.
    Represents rotation from ECI to body frame.

    Attributes:
        epoch_mjd_tt: Epoch.
        quaternion: ECI-to-body quaternion, shape (4,).
        angular_velocity_body: Angular velocity in body frame [rad/s], shape (3,).
    """
    epoch_mjd_tt: float
    quaternion: np.ndarray          # (4,) [q1,q2,q3, q4_scalar]
    angular_velocity_body: np.ndarray = field(
        default_factory=lambda: np.zeros(3)
    )  # (3,) rad/s


@dataclass
class FullState:
    """Complete spacecraft state: translational + STM + attitude."""
    augmented: AugmentedState
    attitude: AttitudeState


@dataclass
class RelativeState:
    """Relative state in the target's LVLH frame.

    Attributes:
        epoch_mjd_tt: Epoch.
        position_lvlh: Relative position [km] in LVLH (R, I, C), shape (3,).
        velocity_lvlh: Relative velocity [km/s] in LVLH, shape (3,).
    """
    epoch_mjd_tt: float
    position_lvlh: np.ndarray      # (3,) km   [radial, in-track, cross-track]
    velocity_lvlh: np.ndarray      # (3,) km/s


# ---------------------------------------------------------------------------
# Orbital Elements
# ---------------------------------------------------------------------------

@dataclass
class OrbitalElements:
    """Classical Keplerian orbital elements.

    Attributes:
        a: Semi-major axis [km].
        e: Eccentricity.
        i: Inclination [rad].
        raan: Right ascension of ascending node [rad].
        aop: Argument of perigee [rad].
        ta: True anomaly [rad].
    """
    a: float
    e: float
    i: float
    raan: float
    aop: float
    ta: float


# ---------------------------------------------------------------------------
# Maneuver Definitions
# ---------------------------------------------------------------------------

@dataclass
class Maneuver:
    """A single maneuver specification.

    For impulsive: dv_vector is the instantaneous ΔV.
    For finite burn: dv_vector is the initial guess / nominal direction,
    and burn_profile contains the detailed thrust profile.

    Attributes:
        epoch_mjd_tt: Maneuver epoch (center for finite burns).
        dv_vector_eci: ΔV vector in ECI [km/s], shape (3,).
        maneuver_type: IMPULSIVE or FINITE_BURN_*.
        burn_profile: Detailed finite-burn parameters (if applicable).
        frame: Frame in which dv_vector is expressed.
    """
    epoch_mjd_tt: float
    dv_vector_eci: np.ndarray       # (3,) km/s
    maneuver_type: ManeuverType = ManeuverType.IMPULSIVE
    burn_profile: Optional[FiniteBurnProfile] = None
    frame: FrameType = FrameType.ECI_J2000

    @property
    def dv_magnitude(self) -> float:
        return float(np.linalg.norm(self.dv_vector_eci))


@dataclass
class FiniteBurnProfile:
    """Finite-duration thrust profile.

    Attributes:
        thrust_n: Thrust magnitude [Newtons].
        isp_s: Specific impulse [seconds].
        t_start_mjd_tt: Burn start epoch.
        duration_s: Burn duration [seconds].
        direction_eci: Thrust direction unit vector in ECI, shape (3,).
            For steered burns, this is the initial direction; actual
            direction is computed by the steering law at each step.
        steering_law: Optional callable(t, state) → unit vector for steered burns.
    """
    thrust_n: float
    isp_s: float
    t_start_mjd_tt: float
    duration_s: float
    direction_eci: np.ndarray       # (3,) unit vector
    steering_law: Optional[Callable] = None

    @property
    def t_end_mjd_tt(self) -> float:
        return self.t_start_mjd_tt + self.duration_s / 86400.0

    @property
    def mass_flow_rate(self) -> float:
        """Propellant mass flow rate [kg/s]."""
        from .constants import G0
        return self.thrust_n / (self.isp_s * G0)

    @classmethod
    def from_impulsive(cls, dv_vec_eci: np.ndarray, mass_kg: float,
                       thrust_n: float, isp_s: float,
                       t_center_mjd_tt: float) -> FiniteBurnProfile:
        """Create a finite burn profile equivalent to an impulsive ΔV.

        Centers the burn on the impulsive maneuver epoch.
        """
        from .constants import G0
        dv_mag = np.linalg.norm(dv_vec_eci)
        # Rocket equation: Δv = Isp*g0 * ln(m0/mf) → duration
        ve = isp_s * G0  # exhaust velocity m/s
        # mass ratio
        mass_ratio = np.exp(dv_mag * 1000.0 / ve)  # dv in m/s
        mass_propellant = mass_kg * (1.0 - 1.0 / mass_ratio)
        mdot = thrust_n / ve  # kg/s
        duration = mass_propellant / mdot  # seconds

        direction = dv_vec_eci / dv_mag if dv_mag > 0 else np.array([1., 0., 0.])
        t_start = t_center_mjd_tt - (duration / 2.0) / 86400.0

        return cls(
            thrust_n=thrust_n,
            isp_s=isp_s,
            t_start_mjd_tt=t_start,
            duration_s=duration,
            direction_eci=direction
        )


@dataclass
class ManeuverPlan:
    """Ordered sequence of maneuvers defining an approach trajectory."""
    maneuvers: list[Maneuver] = field(default_factory=list)

    @property
    def total_dv(self) -> float:
        return sum(m.dv_magnitude for m in self.maneuvers)

    @property
    def n_maneuvers(self) -> int:
        return len(self.maneuvers)


# ---------------------------------------------------------------------------
# Covariance
# ---------------------------------------------------------------------------

@dataclass
class CovarianceState:
    """State covariance at a given epoch.

    Attributes:
        epoch_mjd_tt: Epoch.
        covariance: 6x6 position-velocity covariance [km, km/s], shape (6,6).
        frame: Frame in which covariance is expressed.
    """
    epoch_mjd_tt: float
    covariance: np.ndarray          # (6,6)
    frame: FrameType = FrameType.ECI_J2000

    @property
    def position_sigmas(self) -> np.ndarray:
        """1σ position uncertainties, shape (3,)."""
        return np.sqrt(np.diag(self.covariance[:3, :3]))

    @property
    def velocity_sigmas(self) -> np.ndarray:
        """1σ velocity uncertainties, shape (3,)."""
        return np.sqrt(np.diag(self.covariance[3:6, 3:6]))


@dataclass
class CovarianceHistory:
    """Time history of covariance along a trajectory.

    Attributes:
        epochs: Array of epochs [MJD TT], shape (N,).
        covariances: List of N 6x6 covariance matrices.
    """
    epochs: np.ndarray              # (N,)
    covariances: list[np.ndarray]   # list of (6,6)

    @property
    def position_sigmas(self) -> np.ndarray:
        """1σ position uncertainties over time, shape (N, 3)."""
        return np.array([np.sqrt(np.diag(P[:3, :3])) for P in self.covariances])

    @property
    def velocity_sigmas(self) -> np.ndarray:
        """1σ velocity uncertainties over time, shape (N, 3)."""
        return np.array([np.sqrt(np.diag(P[3:6, 3:6])) for P in self.covariances])


# ---------------------------------------------------------------------------
# Spacecraft and Target Models
# ---------------------------------------------------------------------------

@dataclass
class ThrusterModel:
    """Thruster performance parameters.

    Attributes:
        thrust_n: Nominal thrust [Newtons].
        isp_s: Specific impulse [seconds].
        min_on_time_s: Minimum impulse bit duration [seconds].
    """
    thrust_n: float
    isp_s: float
    min_on_time_s: float = 0.0


@dataclass
class OpticalPayload:
    """Imaging sensor parameters.

    Attributes:
        aperture_diameter_m: Primary aperture diameter [meters].
        focal_length_m: Effective focal length [meters].
        pixel_pitch_m: Detector pixel pitch [meters].
        wavelength_m: Center wavelength [meters].
        quantum_efficiency: Detector QE [0-1].
        read_noise_e: Read noise [electrons rms].
        dark_current_e_per_s: Dark current [electrons/s/pixel].
        optical_throughput: End-to-end optical throughput [0-1].
        detector_shape: (rows, cols) pixel count.
        boresight_body: Sensor boresight unit vector in body frame, shape (3,).
    """
    aperture_diameter_m: float
    focal_length_m: float
    pixel_pitch_m: float
    wavelength_m: float = 550e-9
    quantum_efficiency: float = 0.7
    read_noise_e: float = 5.0
    dark_current_e_per_s: float = 0.1
    optical_throughput: float = 0.8
    detector_shape: tuple[int, int] = (1024, 1024)
    boresight_body: np.ndarray = field(
        default_factory=lambda: np.array([0., 0., 1.])
    )

    @property
    def ifov_rad(self) -> float:
        """Instantaneous field of view per pixel [rad]."""
        return self.pixel_pitch_m / self.focal_length_m

    @property
    def diffraction_limit_rad(self) -> float:
        """Rayleigh diffraction limit [rad]."""
        return 1.22 * self.wavelength_m / self.aperture_diameter_m

    @property
    def fov_half_angle_rad(self) -> float:
        """Half-angle of the full detector FOV [rad]."""
        max_dim = max(self.detector_shape)
        return 0.5 * max_dim * self.ifov_rad


@dataclass
class ADCSParams:
    """Attitude determination and control system performance parameters.

    Attributes:
        max_slew_rate: Maximum slew rate [rad/s].
        max_slew_accel: Maximum slew acceleration [rad/s²].
        settle_time_s: Post-slew settle time [seconds].
        bias_1sigma_rad: Pointing bias [rad, 1σ].
        jitter_1sigma_rad: Pointing jitter [rad, 1σ].
        drift_rate_rad_per_s: Pointing drift rate [rad/s].
    """
    max_slew_rate: float = np.deg2rad(1.0)      # 1 deg/s
    max_slew_accel: float = np.deg2rad(0.1)     # 0.1 deg/s²
    settle_time_s: float = 5.0
    bias_1sigma_rad: float = np.deg2rad(0.01)   # 36 arcsec
    jitter_1sigma_rad: float = np.deg2rad(0.001) # 3.6 arcsec
    drift_rate_rad_per_s: float = 1e-6


@dataclass
class SpacecraftModel:
    """Complete chaser spacecraft model.

    Attributes:
        dry_mass_kg: Dry mass [kg].
        fuel_mass_kg: Propellant mass [kg].
        cr: SRP reflectivity coefficient [dimensionless].
        area_m2: SRP cross-sectional area [m²].
        thruster: Thruster performance model.
        optics: Imaging payload model.
        adcs: ADCS performance model.
    """
    dry_mass_kg: float
    fuel_mass_kg: float
    cr: float = 1.5
    area_m2: float = 10.0
    thruster: ThrusterModel = field(default_factory=lambda: ThrusterModel(22.0, 230.0))
    optics: OpticalPayload = field(default_factory=OpticalPayload)
    adcs: ADCSParams = field(default_factory=ADCSParams)

    @property
    def wet_mass_kg(self) -> float:
        return self.dry_mass_kg + self.fuel_mass_kg

    @property
    def area_over_mass(self) -> float:
        """Area-to-mass ratio [m²/kg]."""
        return self.area_m2 / self.wet_mass_kg


@dataclass
class TargetModel:
    """Target RSO model.

    Attributes:
        characteristic_length_m: Representative dimension [meters].
        albedo: Visual geometric albedo [0-1].
        cr: SRP reflectivity coefficient.
        area_m2: SRP cross-sectional area [m²].
        mass_kg: Estimated mass [kg].
    """
    characteristic_length_m: float = 5.0
    albedo: float = 0.2
    cr: float = 1.5
    area_m2: float = 20.0
    mass_kg: float = 2000.0

    @property
    def area_over_mass(self) -> float:
        return self.area_m2 / self.mass_kg


# ---------------------------------------------------------------------------
# Propagation Results
# ---------------------------------------------------------------------------

@dataclass
class PropagationResult:
    """Output of a numerical propagation.

    Attributes:
        epochs: Time history of epochs [MJD TT], shape (N,).
        states: State vectors [x,y,z,vx,vy,vz] over time, shape (N, 6).
        masses: Mass history [kg], shape (N,).
        stms: STM at each output epoch, list of (6,6) arrays.
        dense_output: scipy dense output object for interpolation (if available).
    """
    epochs: np.ndarray              # (N,)
    states: np.ndarray              # (N, 6)
    masses: np.ndarray              # (N,)
    stms: list[np.ndarray]          # list of (6,6)
    dense_output: Optional[object] = None   # scipy OdeSolution

    @property
    def positions(self) -> np.ndarray:
        """Position history, shape (N, 3)."""
        return self.states[:, :3]

    @property
    def velocities(self) -> np.ndarray:
        """Velocity history, shape (N, 3)."""
        return self.states[:, 3:6]

    def state_at(self, epoch_mjd_tt: float) -> np.ndarray:
        """Interpolate state at an arbitrary epoch.

        Uses dense output if available, otherwise linear interpolation.
        """
        raise NotImplementedError

    def stm_at(self, epoch_mjd_tt: float) -> np.ndarray:
        """Interpolate STM at an arbitrary epoch.

        For covariance propagation, prefer using the STM between
        discrete output epochs rather than interpolating.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Correction and Burn Results
# ---------------------------------------------------------------------------

@dataclass
class CorrectionResult:
    """Result of a differential correction iteration.

    Attributes:
        converged: Whether the corrector converged.
        iterations: Number of iterations taken.
        final_error_km: Final constraint violation magnitude [km].
        dv_corrected_eci: Corrected ΔV vector in ECI [km/s].
        trajectory: Propagation result for the corrected leg.
    """
    converged: bool
    iterations: int
    final_error_km: float
    dv_corrected_eci: np.ndarray
    trajectory: PropagationResult


@dataclass
class FiniteBurnResult:
    """Result of finite-burn conversion and re-correction.

    Attributes:
        burn_profile: The finite burn specification.
        trajectory: Propagation result with the finite burn.
        dv_actual_km_s: Integrated ΔV from the thrust profile [km/s].
        mass_consumed_kg: Propellant consumed [kg].
        gravity_loss_km_s: ΔV penalty vs impulsive [km/s].
        converged: Whether re-correction converged.
    """
    burn_profile: FiniteBurnProfile
    trajectory: PropagationResult
    dv_actual_km_s: float
    mass_consumed_kg: float
    gravity_loss_km_s: float
    converged: bool


# ---------------------------------------------------------------------------
# Imaging and Observation
# ---------------------------------------------------------------------------

@dataclass
class ImagingMetrics:
    """Imaging quality metrics at a single epoch.

    Attributes:
        epoch_mjd_tt: Evaluation epoch.
        range_km: Chaser-to-target range [km].
        phase_angle_rad: Sun-target-chaser phase angle [rad].
        gsd_m: Ground sample distance on target [m].
        resolution_elements: Number of resolution elements across target.
        snr: Signal-to-noise ratio.
        target_sunlit: Whether target is in sunlight.
        pointing_feasible: Whether pointing constraints are satisfied.
        target_in_fov: Whether target is within sensor FOV given uncertainty.
    """
    epoch_mjd_tt: float
    range_km: float
    phase_angle_rad: float
    gsd_m: float
    resolution_elements: float
    snr: float
    target_sunlit: bool
    pointing_feasible: bool
    target_in_fov: bool

    @property
    def is_valid_observation(self) -> bool:
        """All imaging constraints simultaneously satisfied."""
        return (self.target_sunlit and
                self.pointing_feasible and
                self.target_in_fov and
                self.snr > 0 and
                self.resolution_elements > 0)


@dataclass
class ObservationWindow:
    """A contiguous time window where imaging is feasible.

    Attributes:
        start_mjd_tt: Window start epoch.
        end_mjd_tt: Window end epoch.
        peak_resolution_elements: Best resolution achieved in window.
        mean_snr: Average SNR during window.
        limiting_constraint: Which constraint closes the window at each boundary.
    """
    start_mjd_tt: float
    end_mjd_tt: float
    peak_resolution_elements: float
    mean_snr: float
    limiting_constraint: str = ""

    @property
    def duration_s(self) -> float:
        return (self.end_mjd_tt - self.start_mjd_tt) * 86400.0


@dataclass
class ObservationSchedule:
    """Complete set of observation opportunities.

    Attributes:
        windows: List of valid observation windows.
        metrics_timeline: Full timeline of imaging metrics at each evaluation epoch.
    """
    windows: list[ObservationWindow]
    metrics_timeline: list[ImagingMetrics]


# ---------------------------------------------------------------------------
# Mission-Level Containers
# ---------------------------------------------------------------------------

@dataclass
class Waypoint:
    """Desired relative state at a specific epoch.

    Attributes:
        epoch_mjd_tt: Target arrival epoch.
        position_lvlh_km: Desired relative position in LVLH [km], shape (3,).
        velocity_lvlh_km_s: Desired relative velocity in LVLH [km/s] (optional).
        dwell_duration_s: Time to hold at this waypoint [seconds].
    """
    epoch_mjd_tt: float
    position_lvlh_km: np.ndarray
    velocity_lvlh_km_s: Optional[np.ndarray] = None
    dwell_duration_s: float = 0.0


@dataclass
class PerformanceSummary:
    """Top-level mission performance metrics.

    Attributes:
        total_dv_km_s: Total ΔV budget [km/s].
        total_propellant_kg: Total propellant consumed [kg].
        total_observation_time_s: Cumulative observation window time [s].
        n_observation_windows: Number of distinct observation windows.
        max_resolution_elements: Best achievable resolution on target.
        max_position_uncertainty_km: Worst-case 3σ position uncertainty [km].
        mission_duration_days: Total mission elapsed time [days].
        gravity_loss_total_km_s: Total finite-burn gravity losses [km/s].
    """
    total_dv_km_s: float = 0.0
    total_propellant_kg: float = 0.0
    total_observation_time_s: float = 0.0
    n_observation_windows: int = 0
    max_resolution_elements: float = 0.0
    max_position_uncertainty_km: float = 0.0
    mission_duration_days: float = 0.0
    gravity_loss_total_km_s: float = 0.0


@dataclass
class MissionResult:
    """Complete simulation output.

    This is the top-level container returned by the SimulationExecutive.
    """
    chaser_trajectory: PropagationResult
    target_trajectory: PropagationResult
    maneuvers: list[FiniteBurnResult]
    covariance_history: CovarianceHistory
    attitude_timeline: Optional[object] = None   # AttitudeTimeline (defined in attitude module)
    observation_schedule: Optional[ObservationSchedule] = None
    performance: PerformanceSummary = field(default_factory=PerformanceSummary)
