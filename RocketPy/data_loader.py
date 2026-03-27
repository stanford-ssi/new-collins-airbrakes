"""
Data loader for rocket parameters from CSV tables.
Loads thrust curve, mass curve, and Cd vs Mach data.
"""

import numpy as np
import os
from typing import Tuple, Optional

TABLES_DIR = os.path.join(os.path.dirname(__file__), 'tables')

THRUST_FILE = os.path.join(TABLES_DIR, 'V2 Thrust.csv')
MASS_FILE = os.path.join(TABLES_DIR, 'V2 Mass Change.csv')
CD_MACH_FILE = os.path.join(TABLES_DIR, 'V2 CD Mach Number.csv')


def load_thrust_curve() -> Tuple[np.ndarray, np.ndarray]:
    """
    Load motor thrust curve from CSV.
    
    Returns:
        Tuple of (times_s, thrusts_N)
    """
    times = []
    thrusts = []
    
    with open(THRUST_FILE, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            if len(parts) >= 2:
                try:
                    t = float(parts[0])
                    thrust = float(parts[1])
                    times.append(t)
                    thrusts.append(thrust)
                except ValueError:
                    continue
    
    times = np.array(times)
    thrusts = np.array(thrusts)
    
    sort_idx = np.argsort(times)
    times = times[sort_idx]
    thrusts = thrusts[sort_idx]
    
    _, unique_idx = np.unique(times, return_index=True)
    times = times[unique_idx]
    thrusts = thrusts[unique_idx]
    
    return times, thrusts


def load_mass_curve() -> Tuple[np.ndarray, np.ndarray]:
    """
    Load mass vs time curve from CSV.
    Mass in file is in grams, returned in kg.
    
    Returns:
        Tuple of (times_s, masses_kg)
    """
    times = []
    masses = []
    
    with open(MASS_FILE, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            if len(parts) >= 2:
                try:
                    t = float(parts[0])
                    mass_g = float(parts[1])
                    times.append(t)
                    masses.append(mass_g / 1000.0)  # Convert g to kg
                except ValueError:
                    continue
    
    times = np.array(times)
    masses = np.array(masses)
    
    sort_idx = np.argsort(times)
    times = times[sort_idx]
    masses = masses[sort_idx]
    
    _, unique_idx = np.unique(times, return_index=True)
    times = times[unique_idx]
    masses = masses[unique_idx]
    
    return times, masses


def load_cd_vs_mach(max_cd: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load drag coefficient vs Mach number from CSV.
    Filters out unreasonable Cd values.
    Note: CSV format is Cd,Mach (Cd first column).
    
    Args:
        max_cd: Maximum reasonable Cd value (filter threshold)
    
    Returns:
        Tuple of (mach_numbers, cd_values) sorted by Mach
    """
    machs = []
    cds = []
    
    with open(CD_MACH_FILE, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            if len(parts) >= 2:
                try:
                    cd = float(parts[0])
                    mach = float(parts[1])
                    if cd <= max_cd and cd > 0 and mach >= 0:
                        machs.append(mach)
                        cds.append(cd)
                except ValueError:
                    continue
    
    machs = np.array(machs)
    cds = np.array(cds)
    
    sort_idx = np.argsort(machs)
    machs = machs[sort_idx]
    cds = cds[sort_idx]
    
    return machs, cds


def get_cd_at_mach(mach: float, mach_array: np.ndarray = None, 
                   cd_array: np.ndarray = None) -> float:
    """
    Get Cd at a specific Mach number via linear interpolation.
    
    Args:
        mach: Mach number
        mach_array: Array of Mach numbers (if None, loads from file)
        cd_array: Array of Cd values (if None, loads from file)
        
    Returns:
        Interpolated Cd value
    """
    if mach_array is None or cd_array is None:
        mach_array, cd_array = load_cd_vs_mach()
    
    return np.interp(mach, mach_array, cd_array)


def get_thrust_at_time(time: float, time_array: np.ndarray = None,
                       thrust_array: np.ndarray = None) -> float:
    """
    Get thrust at a specific time via linear interpolation.
    
    Args:
        time: Time in seconds
        time_array: Array of times (if None, loads from file)
        thrust_array: Array of thrusts (if None, loads from file)
        
    Returns:
        Interpolated thrust value in N
    """
    if time_array is None or thrust_array is None:
        time_array, thrust_array = load_thrust_curve()
    
    if time < time_array[0] or time > time_array[-1]:
        return 0.0
    
    return np.interp(time, time_array, thrust_array)


def get_mass_at_time(time: float, time_array: np.ndarray = None,
                     mass_array: np.ndarray = None) -> float:
    """
    Get mass at a specific time via linear interpolation.
    
    Args:
        time: Time in seconds
        time_array: Array of times (if None, loads from file)
        mass_array: Array of masses in kg (if None, loads from file)
        
    Returns:
        Interpolated mass value in kg
    """
    if time_array is None or mass_array is None:
        time_array, mass_array = load_mass_curve()
    
    if time <= time_array[0]:
        return mass_array[0]
    if time >= time_array[-1]:
        return mass_array[-1]
    
    return np.interp(time, time_array, mass_array)


def downsample_table(x: np.ndarray, y: np.ndarray, 
                     num_points: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Downsample a table to a specified number of points.
    Uses linear interpolation at evenly spaced x values.
    
    Args:
        x: Original x values
        y: Original y values  
        num_points: Target number of points
        
    Returns:
        Tuple of (downsampled_x, downsampled_y)
    """
    x_new = np.linspace(x.min(), x.max(), num_points)
    y_new = np.interp(x_new, x, y)
    return x_new, y_new


class RocketDataTables:
    """
    Container for all rocket data tables with interpolation.
    Simulation uses full resolution, control system can use coarser tables.
    """
    
    def __init__(self, control_table_resolution: int = None):
        """
        Initialize rocket data tables.
        
        Args:
            control_table_resolution: Number of points for control system 
                                      lookup tables. If None, uses full resolution.
        """
        self.thrust_times, self.thrust_values = load_thrust_curve()
        self.mass_times, self.mass_values = load_mass_curve()
        self.cd_machs, self.cd_values = load_cd_vs_mach()
        
        self.initial_mass = self.mass_values[0]
        self.dry_mass = self.mass_values[-1]
        
        thrust_nonzero = np.where(self.thrust_values > 10)[0]
        if len(thrust_nonzero) > 0:
            self.burnout_time = self.thrust_times[thrust_nonzero[-1]]
        else:
            self.burnout_time = 0.0
        
        if control_table_resolution is not None:
            self.ctrl_cd_machs, self.ctrl_cd_values = downsample_table(
                self.cd_machs, self.cd_values, control_table_resolution
            )
        else:
            self.ctrl_cd_machs = self.cd_machs
            self.ctrl_cd_values = self.cd_values
            
    def get_thrust(self, time: float) -> float:
        """Get thrust at time (full resolution)."""
        return get_thrust_at_time(time, self.thrust_times, self.thrust_values)
    
    def get_mass(self, time: float) -> float:
        """Get mass at time (full resolution)."""
        return get_mass_at_time(time, self.mass_times, self.mass_values)
    
    def get_cd(self, mach: float) -> float:
        """Get Cd at Mach (full resolution for simulation)."""
        return get_cd_at_mach(mach, self.cd_machs, self.cd_values)
    
    def get_cd_control(self, mach: float) -> float:
        """Get Cd at Mach (coarser resolution for control system)."""
        return get_cd_at_mach(mach, self.ctrl_cd_machs, self.ctrl_cd_values)
    
    def get_max_mach(self) -> float:
        """Get maximum Mach number in Cd table."""
        return self.cd_machs.max()


_default_tables = None


def get_rocket_tables(control_resolution: int = None) -> RocketDataTables:
    """
    Get or create the default rocket data tables.
    
    Args:
        control_resolution: Resolution for control system tables
        
    Returns:
        RocketDataTables instance
    """
    global _default_tables
    if _default_tables is None:
        _default_tables = RocketDataTables(control_table_resolution=control_resolution)
    return _default_tables


def reset_rocket_tables():
    """Reset cached tables (useful for testing)."""
    global _default_tables
    _default_tables = None
