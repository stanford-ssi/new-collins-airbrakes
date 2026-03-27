"""
Configuration file for airbrake simulation parameters.
All units in metric (SI) unless otherwise specified.
"""

import numpy as np

# =============================================================================
# Physical Constants
# =============================================================================
GRAVITY = 9.80665  # m/s^2
GAS_CONSTANT = 287.05  # J/(kg·K) for dry air
GAMMA = 1.4  # Ratio of specific heats for air

# =============================================================================
# Atmospheric Model Constants (ISA - International Standard Atmosphere)
# =============================================================================
SEA_LEVEL_PRESSURE = 101325.0  # Pa
SEA_LEVEL_TEMPERATURE = 288.15  # K (15°C)
SEA_LEVEL_DENSITY = 1.225  # kg/m^3
TEMPERATURE_LAPSE_RATE = 0.0065  # K/m (in troposphere)

# Launch Site Altitude (altitude above sea level where simulation starts)
LAUNCH_ALTITUDE_FT = 2070  # feet
LAUNCH_ALTITUDE_M = LAUNCH_ALTITUDE_FT * 0.3048  # meters (~630.9 m)

# =============================================================================
# Rocket Body Parameters
# =============================================================================
ROCKET_DIAMETER_MM = 156.72  # mm
ROCKET_DIAMETER_M = ROCKET_DIAMETER_MM / 1000.0  # m
ROCKET_RADIUS_M = ROCKET_DIAMETER_M / 2.0  # m
ROCKET_REFERENCE_AREA_M2 = np.pi * ROCKET_RADIUS_M ** 2  # m^2 (circular cross-section)

# =============================================================================
# Target Apogee
# =============================================================================
TARGET_APOGEE_FT = 30000  # feet
TARGET_APOGEE_M = TARGET_APOGEE_FT * 0.3048  # meters (~9144 m)

# =============================================================================
# Airbrake Parameters
# =============================================================================
AIRBRAKE_CD = 1.28  # Drag coefficient for flat disc
AIRBRAKE_MAX_AREA_MM2 = 10000  # mm^2
AIRBRAKE_MAX_AREA_M2 = AIRBRAKE_MAX_AREA_MM2 * 1e-6  # m^2
AIRBRAKE_MAX_ANGLE_DEG = 95  # degrees
AIRBRAKE_SLEW_RATE_DEG_S = 100  # degrees per second

# Airbrake area is proportional to angle
# Area = (angle / max_angle) * max_area

# =============================================================================
# Mach Number Constraints
# =============================================================================
MACH_DEPLOY_LIMIT = 1.0  # Airbrakes may not deploy above this Mach number

# =============================================================================
# Sensor Noise Parameters (default values, can be overridden in Monte Carlo)
# =============================================================================
PRESSURE_NOISE_STD = 50.0  # Pa (typical barometer noise)
TEMPERATURE_NOISE_STD = 0.5  # K
ACCEL_Z_NOISE_STD = 0.5  # m/s^2 (accelerometer noise)

# =============================================================================
# EKF Parameters
# =============================================================================
EKF_PROCESS_NOISE_ALTITUDE = 1.0  # m^2
EKF_PROCESS_NOISE_VELOCITY = 0.5  # (m/s)^2
EKF_MEASUREMENT_NOISE_ALTITUDE = 10.0  # m^2
EKF_MEASUREMENT_NOISE_ACCEL = 1.0  # (m/s^2)^2

# =============================================================================
# Simulation Parameters
# =============================================================================
SIMULATION_DT = 0.001  # Time step in seconds (1 ms)
CONTROL_DT = 0.01  # Control loop time step (10 ms, 100 Hz)

# =============================================================================
# Lookup Table Resolution
# =============================================================================
ALTITUDE_LUT_RESOLUTION = 100  # meters between lookup table entries
ALTITUDE_LUT_MAX = 15000  # meters (covers 0-50k ft range, sufficient for high thrust cases)

# Control system Cd lookup table coarseness (number of Mach points)
# Set to None to use full resolution from data file
# Lower values = coarser table, faster but less accurate
CONTROL_CD_TABLE_RESOLUTION = 50  # Number of Mach points for control system

# =============================================================================
# Monte Carlo Parameters
# =============================================================================
DEFAULT_MONTE_CARLO_RUNS = 100

# =============================================================================
# Conversion Factors
# =============================================================================
METERS_TO_FEET = 3.28084
FEET_TO_METERS = 0.3048
