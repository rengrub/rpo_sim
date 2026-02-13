"""
Physical and mathematical constants.

Sources:
    - EGM2008 for gravity field coefficients
    - IAU 2012 for astronomical constants
    - IERS conventions for Earth parameters
"""

import numpy as np

# ---------------------------------------------------------------------------
# Mathematical constants
# ---------------------------------------------------------------------------
TWO_PI = 2.0 * np.pi
DEG2RAD = np.pi / 180.0
RAD2DEG = 180.0 / np.pi
ARCSEC2RAD = DEG2RAD / 3600.0

# ---------------------------------------------------------------------------
# Time constants
# ---------------------------------------------------------------------------
MJD_J2000 = 51544.5                     # MJD of J2000.0 epoch (2000-01-01 12:00 TT)
TT_TAI_OFFSET = 32.184                  # TT - TAI [seconds], fixed
SECONDS_PER_DAY = 86400.0
DAYS_PER_CENTURY = 36525.0

# ---------------------------------------------------------------------------
# Earth parameters
# ---------------------------------------------------------------------------
MU_EARTH = 398600.4418                  # Gravitational parameter [km³/s²]
R_EARTH = 6378.137                      # Equatorial radius [km]
OMEGA_EARTH = 7.2921150e-5              # Earth rotation rate [rad/s]

# Zonal harmonics (unnormalized, from EGM2008)
J2 = 1.08262668355e-3
J3 = -2.53265648533e-6
J4 = -1.61962159137e-6

# Tesseral harmonics (unnormalized, from EGM2008)
# C22 and S22 drive the GEO longitude-dependent perturbation
C22 = 1.57446037456e-6
S22 = -9.03803806639e-7

# ---------------------------------------------------------------------------
# Solar parameters
# ---------------------------------------------------------------------------
MU_SUN = 1.32712440018e11              # Sun gravitational parameter [km³/s²]
AU_KM = 149597870.7                     # Astronomical unit [km]
SOLAR_FLUX_1AU = 1361.0                 # Total solar irradiance at 1 AU [W/m²]
C_LIGHT = 299792.458                    # Speed of light [km/s]
SOLAR_PRESSURE_1AU = SOLAR_FLUX_1AU / (C_LIGHT * 1e3)  # N/m² at 1 AU

# ---------------------------------------------------------------------------
# Lunar parameters
# ---------------------------------------------------------------------------
MU_MOON = 4902.800066                   # Moon gravitational parameter [km³/s²]

# ---------------------------------------------------------------------------
# General
# ---------------------------------------------------------------------------
G0 = 9.80665                            # Standard gravitational acceleration [m/s²]

# ---------------------------------------------------------------------------
# GEO reference values
# ---------------------------------------------------------------------------
R_GEO = 42164.0                         # GEO radius [km]
N_GEO = np.sqrt(MU_EARTH / R_GEO**3)   # GEO mean motion [rad/s]
PERIOD_GEO = TWO_PI / N_GEO             # GEO orbital period [s] ≈ 86164 s
