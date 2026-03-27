"""
Atmospheric models valid for 0-30,000 ft (0-9144 m).
Implements ISA (International Standard Atmosphere) equations for the troposphere.
"""

import numpy as np
from config import (
    SEA_LEVEL_PRESSURE,
    SEA_LEVEL_TEMPERATURE,
    SEA_LEVEL_DENSITY,
    TEMPERATURE_LAPSE_RATE,
    GRAVITY,
    GAS_CONSTANT,
    GAMMA,
    LAUNCH_ALTITUDE_M,
)


def temperature_from_altitude(altitude_m: float) -> float:
    """
    Calculate temperature at a given altitude using ISA model.
    Valid for troposphere (0-11,000 m).
    
    Args:
        altitude_m: Altitude AGL (above ground level) in meters
        
    Returns:
        Temperature in Kelvin
    """
    # Convert AGL to MSL (mean sea level) altitude
    altitude_msl = altitude_m + LAUNCH_ALTITUDE_M
    return SEA_LEVEL_TEMPERATURE - TEMPERATURE_LAPSE_RATE * altitude_msl


def pressure_from_altitude(altitude_m: float) -> float:
    """
    Calculate pressure at a given altitude using ISA barometric formula.
    Valid for troposphere (0-11,000 m).
    
    Args:
        altitude_m: Altitude AGL (above ground level) in meters
        
    Returns:
        Pressure in Pascals
    """
    # Convert AGL to MSL altitude for calculation
    altitude_msl = altitude_m + LAUNCH_ALTITUDE_M
    T = SEA_LEVEL_TEMPERATURE - TEMPERATURE_LAPSE_RATE * altitude_msl
    exponent = GRAVITY / (GAS_CONSTANT * TEMPERATURE_LAPSE_RATE)
    return SEA_LEVEL_PRESSURE * (T / SEA_LEVEL_TEMPERATURE) ** exponent


def density_from_altitude(altitude_m: float) -> float:
    """
    Calculate air density at a given altitude using ideal gas law.
    
    Args:
        altitude_m: Altitude in meters
        
    Returns:
        Air density in kg/m^3
    """
    P = pressure_from_altitude(altitude_m)
    T = temperature_from_altitude(altitude_m)
    return P / (GAS_CONSTANT * T)


def altitude_from_pressure(pressure_pa: float) -> float:
    """
    Calculate altitude from pressure using inverted barometric formula.
    
    Args:
        pressure_pa: Pressure in Pascals
        
    Returns:
        Altitude in meters
    """
    exponent = GAS_CONSTANT * TEMPERATURE_LAPSE_RATE / GRAVITY
    T_ratio = (pressure_pa / SEA_LEVEL_PRESSURE) ** exponent
    T = SEA_LEVEL_TEMPERATURE * T_ratio
    return (SEA_LEVEL_TEMPERATURE - T) / TEMPERATURE_LAPSE_RATE


def altitude_from_pressure_and_temperature(pressure_pa: float, temperature_k: float) -> float:
    """
    Calculate altitude using both pressure and temperature measurements.
    Uses hypsometric equation for improved accuracy.
    
    Args:
        pressure_pa: Pressure in Pascals
        temperature_k: Temperature in Kelvin
        
    Returns:
        Altitude in meters
    """
    return (temperature_k / TEMPERATURE_LAPSE_RATE) * (
        1 - (pressure_pa / SEA_LEVEL_PRESSURE) ** (GAS_CONSTANT * TEMPERATURE_LAPSE_RATE / GRAVITY)
    )


def speed_of_sound(altitude_m: float) -> float:
    """
    Calculate speed of sound at a given altitude.
    
    Args:
        altitude_m: Altitude in meters
        
    Returns:
        Speed of sound in m/s
    """
    T = temperature_from_altitude(altitude_m)
    return np.sqrt(GAMMA * GAS_CONSTANT * T)


def mach_number(velocity_m_s: float, altitude_m: float) -> float:
    """
    Calculate Mach number from velocity and altitude.
    
    Args:
        velocity_m_s: Velocity in m/s
        altitude_m: Altitude in meters
        
    Returns:
        Mach number (dimensionless)
    """
    a = speed_of_sound(altitude_m)
    return abs(velocity_m_s) / a


def velocity_from_mach(mach: float, altitude_m: float) -> float:
    """
    Calculate velocity from Mach number and altitude.
    
    Args:
        mach: Mach number
        altitude_m: Altitude in meters
        
    Returns:
        Velocity in m/s
    """
    a = speed_of_sound(altitude_m)
    return mach * a


def dynamic_pressure(velocity_m_s: float, altitude_m: float) -> float:
    """
    Calculate dynamic pressure (q = 0.5 * rho * v^2).
    
    Args:
        velocity_m_s: Velocity in m/s
        altitude_m: Altitude in meters
        
    Returns:
        Dynamic pressure in Pascals
    """
    rho = density_from_altitude(altitude_m)
    return 0.5 * rho * velocity_m_s ** 2
