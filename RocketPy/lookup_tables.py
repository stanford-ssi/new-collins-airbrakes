"""
Lookup tables for control system.
Pre-computed tables to avoid expensive calculations in control loop.
Uses 1D tables with linear interpolation where possible.
"""

import numpy as np
from config import (
    ALTITUDE_LUT_RESOLUTION,
    ALTITUDE_LUT_MAX,
    SEA_LEVEL_PRESSURE,
    SEA_LEVEL_TEMPERATURE,
    TEMPERATURE_LAPSE_RATE,
    GRAVITY,
    GAS_CONSTANT,
    GAMMA,
    LAUNCH_ALTITUDE_M,
)


class AtmosphereLookupTable:
    """
    1D lookup table for atmospheric properties indexed by altitude.
    Control system uses pressure as input, so we create an inverse table
    mapping pressure -> altitude.
    
    Since pressure is monotonically decreasing with altitude, we can use
    a 1D lookup table with pressure as the index variable.
    """
    
    def __init__(self, resolution_m: float = ALTITUDE_LUT_RESOLUTION, 
                 max_altitude_m: float = ALTITUDE_LUT_MAX):
        """
        Initialize lookup tables.
        
        Args:
            resolution_m: Altitude resolution in meters
            max_altitude_m: Maximum altitude in meters
        """
        self.resolution = resolution_m
        self.max_altitude = max_altitude_m
        
        self.altitudes = np.arange(0, max_altitude_m + resolution_m, resolution_m)
        self.n_entries = len(self.altitudes)
        
        self.temperatures = np.zeros(self.n_entries)
        self.pressures = np.zeros(self.n_entries)
        self.densities = np.zeros(self.n_entries)
        self.speeds_of_sound = np.zeros(self.n_entries)
        
        self._build_tables()
        
        self.pressure_to_altitude_pressures = self.pressures[::-1]
        self.pressure_to_altitude_altitudes = self.altitudes[::-1]
        
    def _build_tables(self):
        """Build all lookup tables accounting for launch altitude."""
        for i, alt_agl in enumerate(self.altitudes):
            # Convert AGL (above ground level) to MSL (mean sea level)
            alt_msl = alt_agl + LAUNCH_ALTITUDE_M
            
            self.temperatures[i] = SEA_LEVEL_TEMPERATURE - TEMPERATURE_LAPSE_RATE * alt_msl
            
            exponent = GRAVITY / (GAS_CONSTANT * TEMPERATURE_LAPSE_RATE)
            self.pressures[i] = SEA_LEVEL_PRESSURE * (
                self.temperatures[i] / SEA_LEVEL_TEMPERATURE
            ) ** exponent
            
            self.densities[i] = self.pressures[i] / (GAS_CONSTANT * self.temperatures[i])
            
            self.speeds_of_sound[i] = np.sqrt(GAMMA * GAS_CONSTANT * self.temperatures[i])
    
    def get_altitude_from_pressure(self, pressure_pa: float) -> float:
        """
        Get altitude from pressure using lookup table with linear interpolation.
        
        Args:
            pressure_pa: Pressure in Pascals
            
        Returns:
            Altitude in meters
        """
        pressure_pa = np.clip(pressure_pa, self.pressures[-1], self.pressures[0])
        return np.interp(pressure_pa, self.pressure_to_altitude_pressures, 
                        self.pressure_to_altitude_altitudes)
    
    def get_temperature_from_altitude(self, altitude_m: float) -> float:
        """
        Get temperature from altitude using lookup table.
        
        Args:
            altitude_m: Altitude in meters
            
        Returns:
            Temperature in Kelvin
        """
        altitude_m = np.clip(altitude_m, 0, self.max_altitude)
        return np.interp(altitude_m, self.altitudes, self.temperatures)
    
    def get_density_from_altitude(self, altitude_m: float) -> float:
        """
        Get air density from altitude using lookup table.
        
        Args:
            altitude_m: Altitude in meters
            
        Returns:
            Density in kg/m^3
        """
        altitude_m = np.clip(altitude_m, 0, self.max_altitude)
        return np.interp(altitude_m, self.altitudes, self.densities)
    
    def get_speed_of_sound_from_altitude(self, altitude_m: float) -> float:
        """
        Get speed of sound from altitude using lookup table.
        
        Args:
            altitude_m: Altitude in meters
            
        Returns:
            Speed of sound in m/s
        """
        altitude_m = np.clip(altitude_m, 0, self.max_altitude)
        return np.interp(altitude_m, self.altitudes, self.speeds_of_sound)
    
    def get_mach_from_velocity_altitude(self, velocity_m_s: float, altitude_m: float) -> float:
        """
        Get Mach number from velocity and altitude.
        Uses 1D lookup for speed of sound, then computes Mach.
        
        Args:
            velocity_m_s: Velocity in m/s
            altitude_m: Altitude in meters
            
        Returns:
            Mach number
        """
        a = self.get_speed_of_sound_from_altitude(altitude_m)
        return abs(velocity_m_s) / a


class AltitudePressureTable:
    """
    Simplified 1D lookup table specifically for altitude from pressure.
    Optimized for control system use where pressure is the primary measurement.
    """
    
    def __init__(self, num_entries: int = 200, max_altitude_m: float = ALTITUDE_LUT_MAX):
        """
        Initialize pressure->altitude lookup table.
        
        Args:
            num_entries: Number of entries in the table
            max_altitude_m: Maximum altitude to support (meters AGL)
        """
        self.num_entries = num_entries
        
        # Calculate min pressure from max altitude (AGL -> MSL)
        alt_msl = max_altitude_m + LAUNCH_ALTITUDE_M
        T_at_max = SEA_LEVEL_TEMPERATURE - TEMPERATURE_LAPSE_RATE * alt_msl
        exponent = GRAVITY / (GAS_CONSTANT * TEMPERATURE_LAPSE_RATE)
        min_pressure = SEA_LEVEL_PRESSURE * (T_at_max / SEA_LEVEL_TEMPERATURE) ** exponent
        max_pressure = SEA_LEVEL_PRESSURE
        
        self.pressures = np.linspace(max_pressure, min_pressure, num_entries)
        self.altitudes = np.zeros(num_entries)
        
        for i, p in enumerate(self.pressures):
            exponent = GAS_CONSTANT * TEMPERATURE_LAPSE_RATE / GRAVITY
            T_ratio = (p / SEA_LEVEL_PRESSURE) ** exponent
            T = SEA_LEVEL_TEMPERATURE * T_ratio
            # Calculate MSL altitude, then convert to AGL
            alt_msl = (SEA_LEVEL_TEMPERATURE - T) / TEMPERATURE_LAPSE_RATE
            self.altitudes[i] = alt_msl - LAUNCH_ALTITUDE_M
    
    def lookup(self, pressure_pa: float) -> float:
        """
        Get altitude from pressure.
        
        Args:
            pressure_pa: Pressure in Pascals
            
        Returns:
            Altitude in meters
        """
        pressure_pa = np.clip(pressure_pa, self.pressures[-1], self.pressures[0])
        return np.interp(pressure_pa, self.pressures[::-1], self.altitudes[::-1])


_default_atmosphere_lut = None
_default_pressure_lut = None


def reset_lookup_tables():
    """Reset cached lookup tables (useful after config changes)."""
    global _default_atmosphere_lut, _default_pressure_lut
    _default_atmosphere_lut = None
    _default_pressure_lut = None


def get_atmosphere_lut() -> AtmosphereLookupTable:
    """Get or create the default atmosphere lookup table."""
    global _default_atmosphere_lut
    if _default_atmosphere_lut is None:
        _default_atmosphere_lut = AtmosphereLookupTable()
    return _default_atmosphere_lut


def get_pressure_lut() -> AltitudePressureTable:
    """Get or create the default pressure lookup table."""
    global _default_pressure_lut
    if _default_pressure_lut is None:
        _default_pressure_lut = AltitudePressureTable()
    return _default_pressure_lut
