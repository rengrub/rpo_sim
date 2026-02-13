# RPO Simulation Infrastructure — Architecture Design

## 1. Overview

A blended medium-high fidelity simulation for GEO Rendezvous and Proximity Operations (RPO)
missions whose objective is resolved imaging of a target Resident Space Object (RSO).

### Design Philosophy

- **HCW as topology oracle**: Hill-Clohessy-Wiltshire equations provide relative motion
  structure and initial guesses. All performance-quality propagation is numerical ECI.
- **Differential correction bridges fidelity levels**: HCW guesses are corrected in the
  high-fidelity force model via shooting methods.
- **Finite burns derived from impulsive solutions**: Impulsive maneuvers are converted to
  realistic thrust profiles and re-corrected.
- **Covariance propagation (not filtering)**: State uncertainty is propagated via the STM
  for mission design assessment, not sequential estimation.
- **Kinematic attitude model**: Sensor pointing is modeled via quaternion kinematics with
  parametric ADCS performance, not full rotational dynamics.

### Key Capabilities

1. Numerical ECI propagation with toggleable force models (J2, J3, J4, tesseral J22, SRP cannonball, luni-solar)
2. STM integration alongside the state for differential correction and covariance propagation
3. HCW relative motion for trajectory design initial guesses
4. Single-shooting and multi-leg differential correction
5. Impulsive-to-finite-burn conversion with re-correction
6. Linear covariance propagation with maneuver execution error modeling
7. Quaternion-based attitude kinematics with slew profiling
8. Imaging performance assessment (resolution, SNR, phase angle, pointing constraints)
9. Observation window computation

---

## 2. Package Structure

```
rpo_sim/
├── __init__.py
├── ARCHITECTURE.md
│
├── core/                       # Foundational data types and configuration
│   ├── __init__.py
│   ├── types.py                # Dataclasses: SpacecraftState, OrbitalElements, etc.
│   ├── constants.py            # Physical and mathematical constants
│   ├── config.py               # Simulation configuration / force model toggles
│   └── frames.py               # Reference frame definitions and transformations
│
├── astrodynamics/              # Orbital mechanics and propagation
│   ├── __init__.py
│   ├── gravity.py              # Gravitational acceleration models (J2-J4, tesserals)
│   ├── srp.py                  # Solar radiation pressure (cannonball)
│   ├── thirdbody.py            # Luni-solar point-mass gravity
│   ├── eom.py                  # Assembled equations of motion + STM variational eqs
│   ├── propagator.py           # Numerical integrator wrapper (RK7(8) adaptive)
│   ├── hcw.py                  # Hill-Clohessy-Wiltshire relative motion
│   ├── ephemeris.py            # Sun/Moon position (analytical)
│   └── shadow.py               # Eclipse/shadow modeling (cylindrical + conical)
│
├── maneuvers/                  # Maneuver design and correction
│   ├── __init__.py
│   ├── impulsive.py            # Impulsive maneuver representation and application
│   ├── finite_burn.py          # Finite burn modeling (chemical + EP profiles)
│   ├── differential_correction.py  # Single-leg and multi-leg shooting
│   └── burn_conversion.py     # Impulsive → finite burn conversion + re-correction
│
├── navigation/                 # State uncertainty and covariance
│   ├── __init__.py
│   ├── covariance.py           # Covariance propagation via STM
│   ├── maneuver_errors.py      # Maneuver execution error models
│   └── process_noise.py        # Process noise modeling
│
├── attitude/                   # Attitude kinematics and pointing
│   ├── __init__.py
│   ├── quaternion.py           # Quaternion operations (composition, SLERP, conversion)
│   ├── kinematics.py           # Attitude propagation and slew profiling
│   ├── pointing.py             # Sensor pointing geometry and constraints
│   └── constraints.py          # Keep-out zones, sun exclusion, gimbal limits
│
├── imaging/                    # Imaging performance assessment
│   ├── __init__.py
│   ├── optics.py               # Resolution, diffraction, FOV calculations
│   ├── radiometry.py           # SNR estimation
│   ├── geometry.py             # Phase angle, lighting, target angular size
│   └── windows.py              # Observation window computation
│
├── mission/                    # Mission-level orchestration
│   ├── __init__.py
│   ├── scenario.py             # Scenario definition and parameter management
│   ├── executive.py            # Top-level simulation workflow
│   ├── trajectory_design.py    # HCW-based approach trajectory design
│   └── performance.py          # Performance metric computation and reporting
│
└── utils/                      # Shared utilities
    ├── __init__.py
    ├── math_utils.py           # Cross-product matrix, rotation utilities
    ├── time_utils.py           # Time system conversions (UTC, TT, TDB, MJD)
    └── validation.py           # State vector and covariance validation checks
```

---

## 3. Module Specifications

### 3.1 `core/types.py` — Foundational Data Types

All state and configuration data flows through well-defined dataclasses.

```
SpacecraftState         6-element ECI pos/vel + mass + epoch
AugmentedState          SpacecraftState + 6×6 STM (42+1 elements)
AttitudeState           Quaternion + angular velocity
FullState               AugmentedState + AttitudeState
OrbitalElements         Classical Keplerian elements (a, e, i, Ω, ω, ν)
RelativeState           LVLH relative position and velocity
ManeuverPlan            Sequence of Maneuver objects with epochs and ΔVs
Maneuver                Single impulsive or finite burn specification
FiniteBurn              Thrust magnitude, Isp, direction profile, start/end times
CovarianceState         6×6 covariance matrix + epoch + frame identifier
SpacecraftModel         Physical properties (mass, Cr, A/m, thruster specs, optics)
TargetModel             Physical properties (size, shape, albedo, Cr, A/m)
PropagationResult       Time history of states, STMs, and derived quantities
```

### 3.2 `core/config.py` — Simulation Configuration

A single configuration object controls all toggleable options.

```
ForceModelConfig:
    enable_j2: bool = True
    enable_j3: bool = True
    enable_j4: bool = True
    enable_j22_tesseral: bool = True
    enable_srp: bool = True
    enable_solar_gravity: bool = False
    enable_lunar_gravity: bool = False
    shadow_model: "cylindrical" | "conical" | "none"

IntegratorConfig:
    method: "DOP853"              # scipy's 8th-order Dormand-Prince
    rtol: float = 1e-12
    atol: float = 1e-12
    max_step: float = 300.0       # seconds
    dense_output: bool = True

CovarianceConfig:
    process_noise_spectral_density: float   # m²/s⁵
    maneuver_magnitude_error_3sigma: float  # fractional
    maneuver_pointing_error_3sigma: float   # radians

SimConfig:
    force_model: ForceModelConfig
    integrator: IntegratorConfig
    covariance: CovarianceConfig
    output_step: float = 60.0     # seconds
```

### 3.3 `core/frames.py` — Reference Frame Transformations

Critical infrastructure. Every transformation function returns both the rotation
matrix and its time derivative.

```
eci_to_ecef(epoch) → (R_3x3, dR_3x3)
    - Uses Earth rotation angle + precession/nutation (IAU 2006 simplified)
    - Needs UT1-UTC correction (parameterized or from table)

eci_to_lvlh(r_target, v_target) → (R_3x3, dR_3x3)
    - R-bar: r_hat (radial outward)
    - V-bar: completes triad (approximately along-track for near-circular)
    - H-bar: h_hat = r × v / |r × v| (orbit normal)
    - dR computed from angular velocity ω = h / r²

eci_to_rsw(r, v) → (R_3x3, dR_3x3)
    - Alias for LVLH with explicit RSW naming

relative_state(r_chaser, v_chaser, r_target, v_target) → RelativeState
    - Computes Δr, Δv in target LVLH frame
    - Correctly accounts for frame rotation in velocity transformation
```

### 3.4 `astrodynamics/gravity.py` — Gravitational Accelerations

Each model returns acceleration vector and its Jacobian (∂a/∂r, 3×3) for STM integration.

```
two_body(r, mu) → (a_3, da_dr_3x3)

zonal_j2(r, mu, R_earth, J2) → (a_3, da_dr_3x3)
zonal_j3(r, mu, R_earth, J3) → (a_3, da_dr_3x3)
zonal_j4(r, mu, R_earth, J4) → (a_3, da_dr_3x3)

tesseral_j22(r, mu, R_earth, C22, S22, gmst) → (a_3, da_dr_3x3)
    - Requires GMST to rotate the tesseral field with the Earth
    - Returns acceleration in ECI

combined_gravity(r, epoch, config) → (a_3, da_dr_3x3)
    - Assembles all enabled gravity terms
    - Single interface for the EOM module
```

### 3.5 `astrodynamics/srp.py` — Solar Radiation Pressure

```
cannonball_srp(r_sat, r_sun, Cr, area_over_mass, shadow_factor) → (a_3, da_dr_3x3)
    - shadow_factor ∈ [0, 1]: 0 = full shadow, 1 = full sunlight
    - Jacobian includes partials of 1/r² sun-satellite term w.r.t. satellite position

shadow_function(r_sat, r_sun, R_earth, model="conical") → float
    - Returns shadow factor (0 to 1)
    - Cylindrical: binary 0 or 1
    - Conical: smooth penumbra transition
```

### 3.6 `astrodynamics/eom.py` — Equations of Motion

The central module that assembles the full state derivative vector.

```
eom_full(t, y, config, spacecraft_model, epoch_ref) → dy_dt
    - y is the augmented state: [x,y,z,vx,vy,vz, m, Φ_36_elements]
    - Total: 43 elements (6 state + 1 mass + 36 STM)
    - During coast: dm/dt = 0, thrust = 0
    - During burn: dm/dt = -T/(Isp*g0), thrust acceleration included
    - STM derivative: dΦ/dt = A(t) · Φ
    - A(t) assembled from Jacobians of all active force models:
        A = [[0₃, I₃], [∂a/∂r, ∂a/∂v]]
      where ∂a/∂r sums contributions from gravity, SRP, third-body
      and ∂a/∂v is typically zero (no velocity-dependent forces at GEO)

    - epoch_ref: reference epoch (MJD TT) so that integration variable t
      (seconds since epoch_ref) can be converted to absolute epoch for
      ephemeris lookups and Earth rotation
```

### 3.7 `astrodynamics/propagator.py` — Numerical Integrator

```
class Propagator:
    def __init__(self, config: SimConfig)

    def propagate(self, state: AugmentedState, t_span: (float, float),
                  events: list[EventFunction] = None,
                  maneuvers: list[Maneuver] = None) → PropagationResult
        - Wraps scipy.integrate.solve_ivp with DOP853
        - Handles maneuver events: stops integration at maneuver epoch,
          applies ΔV (impulsive) or switches to burn mode (finite),
          restarts integration
        - Handles shadow boundary events for clean SRP transitions
        - Returns dense output for interpolation

    def propagate_segment(self, y0, t0, tf, burn: FiniteBurn = None) → result
        - Low-level single-segment propagation
        - If burn is provided, includes thrust in EOM for [t_start, t_end] of burn
```

### 3.8 `astrodynamics/hcw.py` — Hill-Clohessy-Wiltshire

```
hcw_state_transition(n, dt) → Φ_hcw_6x6
    - Analytical CW state transition matrix
    - n = mean motion of target orbit

hcw_free_motion(rel_state_0, n, t_array) → rel_state_history
    - Propagate relative state analytically

hcw_two_impulse_transfer(rel_state_0, rel_state_f, n, dt) → (dv1, dv2)
    - Lambert-like: find impulsive ΔVs to go from initial to final
      relative state in time dt

hcw_coelliptic_approach(range_initial, range_final, n, num_hops) → ManeuverPlan
    - Design a multi-hop V-bar or R-bar approach sequence

hcw_circumnavigation(center_offset, ellipse_size, n) → ManeuverPlan
    - Design a forced or natural fly-around trajectory
```

### 3.9 `maneuvers/differential_correction.py` — Shooting Method

```
class SingleShooter:
    def __init__(self, propagator: Propagator, config: SimConfig)

    def correct_leg(self, state_0: AugmentedState,
                    dv_guess: np.ndarray,       # 3-vector ECI
                    t_maneuver: float,
                    target_rel_pos: np.ndarray,  # 3-vector LVLH desired
                    t_target: float,
                    target_state_prop: PropagationResult,
                    tol: float = 1e-6,          # km convergence
                    max_iter: int = 20) → CorrectionResult
        - Solves 3×3 system: find ΔV at t_maneuver such that relative
          position at t_target matches target_rel_pos
        - Uses STM upper-right block Φ_rv for Jacobian
        - Returns: converged ΔV, converged trajectory, iteration history

class MultiLegCorrector:
    def __init__(self, propagator: Propagator, config: SimConfig)

    def correct_sequence(self, legs: list[LegDefinition]) → list[CorrectionResult]
        - Sequential single-shooting through multiple legs
        - Each leg's terminal state becomes next leg's initial state

CorrectionResult:
    converged: bool
    iterations: int
    final_constraint_error: float
    dv_corrected: np.ndarray
    trajectory: PropagationResult
    stm_final: np.ndarray
```

### 3.10 `maneuvers/finite_burn.py` — Finite Burn Modeling

```
class FiniteBurnProfile:
    thrust: float               # Newtons
    isp: float                  # seconds
    direction: np.ndarray       # unit vector ECI (or callable for steered burns)
    t_start: float
    duration: float             # computed from impulsive ΔV and thrust/mass

    @classmethod
    def from_impulsive(cls, dv_vec: np.ndarray, mass: float,
                       thrust: float, isp: float,
                       t_center: float) → FiniteBurnProfile
        - Computes burn duration from rocket equation
        - Centers burn on the impulsive epoch (t_start = t_center - duration/2)
        - Sets direction to dv_vec / |dv_vec|

    def thrust_acceleration(self, t: float, mass: float) → np.ndarray
        - Returns thrust acceleration vector at time t during the burn
        - Zero outside [t_start, t_start + duration]

    def mass_flow_rate(self) → float
        - Returns -T / (Isp * g0)

class SteeredBurn(FiniteBurnProfile):
    - Overrides direction with velocity-to-be-gained steering:
      at each integration step, compute remaining ΔV needed and
      steer thrust along that direction
```

### 3.11 `maneuvers/burn_conversion.py` — Impulsive to Finite Conversion

```
class BurnConverter:
    def __init__(self, propagator, corrector, config)

    def convert_and_correct(self,
                            impulsive_solution: CorrectionResult,
                            spacecraft: SpacecraftModel,
                            target_trajectory: PropagationResult,
                            target_rel_pos: np.ndarray,
                            t_target: float) → FiniteBurnResult
        - Step 1: Create FiniteBurnProfile from impulsive ΔV
        - Step 2: Propagate with finite burn in EOM
        - Step 3: Evaluate constraint violation at target epoch
        - Step 4: Re-correct using finite-burn differential correction
          (free variables: burn start time, direction angles)
        - Returns: converged finite-burn trajectory + performance metrics

FiniteBurnResult:
    burn_profile: FiniteBurnProfile
    trajectory: PropagationResult
    dv_actual: float            # integrated ΔV (from thrust profile)
    mass_consumed: float
    gravity_loss: float         # dv_actual - dv_impulsive
    converged: bool
```

### 3.12 `navigation/covariance.py` — Covariance Propagation

```
class CovariancePropagator:
    def __init__(self, config: CovarianceConfig)

    def propagate(self, P0: np.ndarray,         # 6×6
                  stm: np.ndarray,              # 6×6 from propagation
                  dt: float) → np.ndarray       # 6×6
        - P(t) = Φ · P0 · Φᵀ + Q(dt)
        - Q from process noise model

    def propagate_along_trajectory(self,
                                   P0: np.ndarray,
                                   prop_result: PropagationResult,
                                   maneuver_errors: list[ManeuverError] = None
                                   ) → CovarianceHistory
        - Walks along the trajectory, propagating covariance via STM
          at each output epoch
        - At maneuver epochs, applies maneuver execution error inflation
        - Returns time-tagged covariance history

    def relative_covariance(self, P_chaser, P_target,
                            cross_correlation=None) → np.ndarray
        - P_rel = P_chaser + P_target - (cross terms if correlated)

CovarianceHistory:
    epochs: np.ndarray
    covariances: list[np.ndarray]   # list of 6×6 matrices
    position_sigmas: np.ndarray     # Nx3 (1σ in R, I, C or ECI)
    velocity_sigmas: np.ndarray     # Nx3
```

### 3.13 `navigation/maneuver_errors.py`

```
class ManeuverExecutionError:
    magnitude_error_1sigma: float   # fractional (e.g. 0.01 = 1%)
    pointing_error_1sigma: float    # radians

    def covariance(self, dv_vec: np.ndarray) → np.ndarray   # 3×3
        - Constructs ΔV error covariance in ECI from magnitude
          and pointing errors applied in the thrust frame
        - Transforms to ECI via thrust-frame-to-ECI rotation

    def inflate_state_covariance(self, P_pre: np.ndarray,
                                  dv_vec: np.ndarray) → np.ndarray
        - P_post = P_pre + B · P_dv · Bᵀ
        - B maps velocity errors into state (identity in velocity partition)
```

### 3.14 `attitude/quaternion.py` — Quaternion Operations

Convention: q = [q1, q2, q3, q4] where q4 is the scalar component.

```
q_multiply(q1, q2) → q
q_conjugate(q) → q
q_normalize(q) → q
q_to_dcm(q) → R_3x3
dcm_to_q(R) → q
q_rotate_vector(q, v) → v_rotated
q_from_axis_angle(axis, angle) → q
q_to_axis_angle(q) → (axis, angle)
q_slerp(q1, q2, t) → q          # t ∈ [0, 1]
q_angle_between(q1, q2) → float  # radians
euler_to_q(roll, pitch, yaw, sequence='321') → q
q_to_euler(q, sequence='321') → (roll, pitch, yaw)
```

### 3.15 `attitude/kinematics.py` — Attitude Propagation & Slew

```
class AttitudeProfile:
    def __init__(self, adcs_params: ADCSParams)

    def compute_target_pointing_quaternion(self,
            r_chaser, v_chaser, r_target,
            roll_about_boresight=0.0) → np.ndarray
        - Boresight along chaser→target direction
        - Secondary axis constraint (e.g., solar array toward sun)
        - Returns quaternion ECI→body

    def plan_slew(self, q_start, q_end) → SlewProfile
        - Computes SLERP path with trapezoidal rate profile
        - Returns time history of quaternion and angular velocity

    def pointing_error_model(self, dt_integration) → PointingPerformance
        - Returns bias, jitter, stability over integration time

ADCSParams:
    max_slew_rate: float        # rad/s
    max_slew_accel: float       # rad/s²
    settle_time: float          # seconds
    bias_1sigma: float          # radians
    jitter_1sigma: float        # radians
    drift_rate: float           # rad/s

SlewProfile:
    duration: float
    settle_time: float
    total_time: float           # duration + settle_time
    quaternion_profile: callable  # q(t)
    angular_velocity_profile: callable  # ω(t)

PointingPerformance:
    bias: float
    jitter: float
    stability: float            # over integration time
    effective_resolution_degradation: float  # multiplicative factor ≥ 1
```

### 3.16 `attitude/pointing.py` — Sensor Pointing Geometry

```
class SensorPointing:
    def __init__(self, boresight_body: np.ndarray,  # unit vec in body frame
                 fov_half_angle: float)              # radians

    def boresight_eci(self, q_body_to_eci) → np.ndarray
        - Rotate boresight to ECI

    def target_in_fov(self, q, r_chaser, r_target) → bool
        - Is the target within the FOV cone?

    def target_bearing_error(self, q, r_chaser, r_target,
                              P_rel: np.ndarray) → BearingUncertainty
        - Project relative position covariance into angular uncertainty
          perpendicular to line of sight
        - Compare to FOV for acquisition probability

BearingUncertainty:
    sigma_cross1: float         # radians, 1σ
    sigma_cross2: float         # radians, 1σ
    in_fov_probability: float   # P(target within FOV)
```

### 3.17 `attitude/constraints.py` — Pointing Constraints

```
class PointingConstraintSet:
    def __init__(self)

    def add_sun_exclusion(self, min_angle: float, boresight_body: np.ndarray)
    def add_earth_exclusion(self, min_angle: float, boresight_body: np.ndarray)
    def add_gimbal_limit(self, axis_body: np.ndarray, max_angle: float)
    def add_body_axis_sun_constraint(self, axis_body, min_angle, max_angle)

    def evaluate(self, q, r_sat, r_sun, r_earth_center=None) → ConstraintResult
        - Checks all constraints at a given attitude
        - Returns: all_satisfied, per-constraint margins

    def feasibility_over_trajectory(self, q_profile, trajectory,
                                     sun_positions) → FeasibilityTimeline
        - Evaluates constraints at every output epoch
        - Returns boolean timeline + margin history
```

### 3.18 `imaging/optics.py` — Optical System Model

```
class OpticalSystem:
    aperture_diameter: float    # meters
    focal_length: float         # meters
    pixel_pitch: float          # meters
    wavelength: float           # meters (center)
    quantum_efficiency: float
    read_noise: float           # electrons
    dark_current: float         # electrons/s
    optical_throughput: float   # dimensionless
    fov_pixels: (int, int)      # detector array size

    @property
    def ifov(self) → float
        - pixel_pitch / focal_length (radians)

    @property
    def diffraction_limit(self) → float
        - 1.22 * wavelength / aperture_diameter (radians)

    def ground_sample_distance(self, range_m: float) → float
        - range * max(ifov, diffraction_limit)

    def resolution_elements_on_target(self, range_m, target_size_m) → float
        - target_size / gsd

    def effective_resolution(self, pointing_stability: float) → float
        - sqrt(diffraction_limit² + pointing_stability²)
```

### 3.19 `imaging/radiometry.py` — SNR Estimation

```
class RadiometricModel:
    def __init__(self, optics: OpticalSystem)

    def signal_electrons(self, solar_irradiance, albedo, phase_angle,
                          target_area, range_m, integration_time) → float
        - Reflected flux from target (Lambertian BRDF with phase function)
        - Collected by aperture, focused on detector
        - Converted to photoelectrons via QE

    def noise_electrons(self, signal, integration_time,
                         n_pixels_target) → float
        - Shot noise + read noise + dark current

    def snr(self, ...) → float
        - signal / noise

    def minimum_integration_time(self, snr_threshold, ...) → float
        - Solve for integration time to achieve desired SNR
```

### 3.20 `imaging/windows.py` — Observation Windows

```
class ObservationWindowCalculator:
    def __init__(self, optics, radiometric_model, pointing_constraints,
                 attitude_profile, covariance_propagator)

    def compute_windows(self,
                        chaser_trajectory: PropagationResult,
                        target_trajectory: PropagationResult,
                        sun_positions: np.ndarray,
                        covariance_history: CovarianceHistory,
                        min_resolution_elements: float,
                        min_snr: float,
                        max_range: float,
                        phase_angle_bounds: (float, float)
                        ) → ObservationSchedule
        - At each epoch, evaluate ALL constraints simultaneously:
          1. Range within bounds
          2. Resolution sufficient (accounting for pointing stability)
          3. Phase angle within bounds (target illuminated favorably)
          4. SNR above threshold
          5. Target not in eclipse
          6. Pointing constraints satisfied (sun exclusion, etc.)
          7. Target within FOV given bearing uncertainty
        - Returns windows where all constraints are simultaneously met

ObservationSchedule:
    windows: list[TimeWindow]   # (start, end) pairs
    quality_timeline: dict      # per-epoch metrics for each constraint
    limiting_constraint: list   # which constraint is most restrictive at each epoch
```

### 3.21 `mission/scenario.py` — Scenario Definition

```
@dataclass
class Scenario:
    name: str
    epoch: float                        # MJD TT

    target_state_eci: np.ndarray        # 6-element
    target_model: TargetModel

    chaser_state_eci: np.ndarray        # 6-element
    chaser_model: SpacecraftModel

    initial_covariance_chaser: np.ndarray   # 6×6
    initial_covariance_target: np.ndarray   # 6×6

    approach_waypoints: list[Waypoint]  # LVLH relative positions + epochs
    observation_stations: list[ObsStation]  # where to dwell for imaging

    config: SimConfig
```

### 3.22 `mission/executive.py` — Simulation Workflow

```
class SimulationExecutive:
    def __init__(self, scenario: Scenario)

    def run(self) → MissionResult:
        1. Propagate target trajectory over full mission duration
        2. Generate HCW initial guess for approach maneuver sequence
        3. Convert HCW solution to ECI (impulsive ΔVs at maneuver epochs)
        4. For each leg:
           a. Differential correction → converged impulsive trajectory
           b. Convert to finite burn → FiniteBurnProfile
           c. Re-correct with finite burn → converged finite-burn trajectory
        5. Assemble full chaser trajectory
        6. Initialize covariance, propagate along trajectory
           - Inflate at each maneuver epoch with execution errors
        7. Compute attitude timeline:
           - Desired pointing quaternion at each epoch
           - Slew profiles between attitude modes
           - Pointing performance metrics
        8. Evaluate imaging windows:
           - Intersect all constraint timelines
        9. Compile performance metrics:
           - Total ΔV and propellant mass (from finite burn integration)
           - ΔV dispersion statistics (from maneuver covariance)
           - Imaging opportunity count, duration, quality
           - Timeline: maneuver schedule, slew schedule, observation schedule
           - Safety: closest approach distance, collision probability bounds

MissionResult:
    chaser_trajectory: PropagationResult
    target_trajectory: PropagationResult
    maneuvers: list[FiniteBurnResult]
    covariance_history: CovarianceHistory
    attitude_timeline: AttitudeTimeline
    observation_schedule: ObservationSchedule
    performance_summary: PerformanceSummary
```

---

## 4. Data Flow Diagram

```
                    ┌─────────────┐
                    │  Scenario    │
                    │  Definition  │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │   HCW       │
                    │   Design    │──── ManeuverPlan (impulsive, LVLH)
                    └──────┬──────┘
                           │
                    ┌──────▼──────────┐
                    │  LVLH→ECI       │
                    │  Conversion     │──── ManeuverPlan (impulsive, ECI)
                    └──────┬──────────┘
                           │
              ┌────────────▼────────────┐
              │  Differential Corrector │
              │  (per leg, impulsive)   │──── Converged impulsive trajectories
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │  Finite Burn Converter  │
              │  + Re-correction        │──── Converged finite-burn trajectories
              └────────────┬────────────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
  ┌──────▼──────┐  ┌──────▼──────┐  ┌──────▼──────┐
  │ Covariance  │  │  Attitude   │  │   Imaging   │
  │ Propagation │  │  Timeline   │  │  Geometry   │
  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘
         │                │                 │
         └────────────────┼─────────────────┘
                          │
                   ┌──────▼──────┐
                   │ Observation │
                   │   Windows   │
                   └──────┬──────┘
                          │
                   ┌──────▼──────┐
                   │ Performance │
                   │   Metrics   │
                   └─────────────┘
```

---

## 5. Key Design Decisions

### 5.1 STM Integration Strategy
- Integrate the 6×6 STM as 36 additional elements alongside the 6 state + 1 mass = 43 total
- Use variational equations (analytically derived Jacobians) not finite differencing
- Reset STM to identity at each maneuver epoch to avoid numerical conditioning issues
  over long arcs — compose STMs across segments: Φ(t2,t0) = Φ(t2,t1) · Φ(t1,t0)

### 5.2 Covariance Propagation Strategy
- Do NOT integrate the covariance differential equation (Lyapunov equation)
- Propagate algebraically: P(t) = Φ(t,t0) P(t0) Φ(t,t0)ᵀ + Q
- Compute at output epochs from stored STMs
- This is cheaper and avoids the numerical stiffness of the Lyapunov equation

### 5.3 Attitude Modeling Scope
- Kinematics only — no Euler equation, no reaction wheel dynamics
- Parametric performance model captures ADCS capability
- Quaternion propagation for pointing timeline and slew planning
- Pointing error modeled statistically, not dynamically

### 5.4 Event Handling in Integration
- Shadow entry/exit: event detection (zero-crossing of shadow function)
- Maneuver start/end: event detection on time
- Integrator restarts at each event to maintain accuracy across discontinuities

### 5.5 Units Convention
- Distance: kilometers
- Time: seconds (integration variable), MJD TT (epochs)
- Velocity: km/s
- Mass: kg
- Angles: radians (internal), degrees for user-facing I/O
- Thrust: Newtons (converted to km/s² internally by dividing by mass×1000)

---

## 6. Dependencies

- **numpy**: Array operations, linear algebra
- **scipy**: ODE integration (solve_ivp with DOP853), optimization (for finite-burn correction)
- **Standard library**: dataclasses, enum, typing, math

No external astrodynamics libraries. The simulation is self-contained.
```
