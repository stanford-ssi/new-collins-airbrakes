"""
Rocket model for simulation environment.
Handles mass, thrust, and body drag properties.
Loads data from tables/ directory by default.
"""

import numpy as np
from typing import Callable, Optional, List, Tuple

from config import ROCKET_REFERENCE_AREA_M2


class ThrustCurve:
    """
    Thrust curve model loaded from data or defined programmatically.
    """
    
    def __init__(self, times: np.ndarray = None, thrusts: np.ndarray = None):
        """
        Initialize thrust curve.
        
        Args:
            times: Array of time points in seconds
            thrusts: Array of thrust values in Newtons
        """
        if times is None or thrusts is None:
            from data_loader import load_thrust_curve
            self.times, self.thrusts = load_thrust_curve()
        else:
            self.times = np.array(times)
            self.thrusts = np.array(thrusts)
            
        self.burnout_time = self.times[np.where(self.thrusts > 0)[0][-1]] if np.any(self.thrusts > 0) else 0.0
        
    @classmethod
    def from_file(cls, filepath: str) -> 'ThrustCurve':
        """
        Load thrust curve from CSV file.
        Expected format: time(s), thrust(N)
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            ThrustCurve instance
        """
        data = np.loadtxt(filepath, delimiter=',', skiprows=1)
        return cls(times=data[:, 0], thrusts=data[:, 1])
    
    @classmethod
    def from_arrays(cls, times: List[float], thrusts: List[float]) -> 'ThrustCurve':
        """
        Create thrust curve from lists.
        
        Args:
            times: List of time points
            thrusts: List of thrust values
            
        Returns:
            ThrustCurve instance
        """
        return cls(times=np.array(times), thrusts=np.array(thrusts))
    
    def get_thrust(self, time: float) -> float:
        """
        Get thrust at given time.
        
        Args:
            time: Time in seconds
            
        Returns:
            Thrust in Newtons
        """
        if time < self.times[0] or time > self.times[-1]:
            return 0.0
        return np.interp(time, self.times, self.thrusts)
    
    def get_burnout_time(self) -> float:
        """Get motor burnout time."""
        return self.burnout_time
    
    def get_total_impulse(self) -> float:
        """Calculate total impulse (integral of thrust)."""
        return np.trapz(self.thrusts, self.times)


class MassModel:
    """
    Mass model with time-varying mass.
    Loads from table by default for accurate propellant consumption.
    """
    
    def __init__(self, times: np.ndarray = None, masses: np.ndarray = None):
        """
        Initialize mass model.
        
        Args:
            times: Array of time points in seconds
            masses: Array of mass values in kg
        """
        if times is None or masses is None:
            from data_loader import load_mass_curve
            self.times, self.masses = load_mass_curve()
        else:
            self.times = np.array(times)
            self.masses = np.array(masses)
            
        self.initial_mass = self.masses[0]
        self.dry_mass = self.masses[-1]
            
    def get_mass(self, time: float) -> float:
        """
        Get mass at given time via linear interpolation.
        
        Args:
            time: Time in seconds
            
        Returns:
            Mass in kg
        """
        if time <= self.times[0]:
            return self.masses[0]
        if time >= self.times[-1]:
            return self.masses[-1]
        return np.interp(time, self.times, self.masses)
    
    def get_initial_mass(self) -> float:
        """Get initial (wet) mass."""
        return self.initial_mass
    
    def get_dry_mass(self) -> float:
        """Get dry mass."""
        return self.dry_mass


class DragModel:
    """
    Body drag model with Mach-varying Cd from lookup table.
    """
    
    def __init__(
        self,
        reference_area: float = None,
        mach_array: np.ndarray = None,
        cd_array: np.ndarray = None,
    ):
        """
        Initialize drag model.
        
        Args:
            reference_area: Reference area in m^2 (default from config)
            mach_array: Array of Mach numbers for Cd lookup
            cd_array: Array of Cd values corresponding to Mach numbers
        """
        self.reference_area = reference_area if reference_area else ROCKET_REFERENCE_AREA_M2
        
        if mach_array is None or cd_array is None:
            from data_loader import load_cd_vs_mach
            self.mach_array, self.cd_array = load_cd_vs_mach()
        else:
            self.mach_array = np.array(mach_array)
            self.cd_array = np.array(cd_array)
            
    def get_cd(self, mach: float = 0.0, time: float = 0.0) -> float:
        """
        Get drag coefficient at given Mach via linear interpolation.
        
        Args:
            mach: Mach number
            time: Time in seconds (unused, kept for API compatibility)
            
        Returns:
            Drag coefficient
        """
        return np.interp(mach, self.mach_array, self.cd_array)
    
    def get_drag_force(self, velocity: float, density: float, mach: float = 0.0, time: float = 0.0) -> float:
        """
        Calculate drag force.
        
        Args:
            velocity: Velocity in m/s
            density: Air density in kg/m^3
            mach: Mach number
            time: Time in seconds
            
        Returns:
            Drag force in Newtons
        """
        cd = self.get_cd(mach, time)
        return 0.5 * density * velocity**2 * cd * self.reference_area


class RocketModel:
    """
    Complete rocket model combining thrust, mass, and drag.
    """
    
    def __init__(
        self,
        thrust_curve: ThrustCurve = None,
        mass_model: MassModel = None,
        drag_model: DragModel = None,
    ):
        """
        Initialize rocket model.
        
        Args:
            thrust_curve: ThrustCurve instance
            mass_model: MassModel instance
            drag_model: DragModel instance
        """
        self.thrust = thrust_curve if thrust_curve else ThrustCurve()
        self.mass = mass_model if mass_model else MassModel()
        self.drag = drag_model if drag_model else DragModel()
        
    def get_thrust(self, time: float) -> float:
        """Get thrust at time."""
        return self.thrust.get_thrust(time)
    
    def get_mass(self, time: float) -> float:
        """Get mass at time."""
        return self.mass.get_mass(time)
    
    def get_body_cd(self, mach: float, time: float) -> float:
        """Get body Cd at conditions."""
        return self.drag.get_cd(mach, time)
    
    def get_body_drag(self, velocity: float, density: float, mach: float, time: float) -> float:
        """Get body drag force."""
        return self.drag.get_drag_force(velocity, density, mach, time)
    
    def get_reference_area(self) -> float:
        """Get body reference area."""
        return self.drag.reference_area
    
    def get_burnout_time(self) -> float:
        """Get motor burnout time."""
        return self.thrust.get_burnout_time()
    
    @classmethod
    def from_tables(cls, reference_area: float = None) -> 'RocketModel':
        """
        Create rocket model from data tables (default behavior).
        Loads thrust, mass, and Cd data from tables/ directory.
        
        Args:
            reference_area: Reference area in m^2 (default from config)
            
        Returns:
            RocketModel instance
        """
        return cls(
            thrust_curve=ThrustCurve(),
            mass_model=MassModel(),
            drag_model=DragModel(reference_area=reference_area),
        )
    
    @classmethod
    def from_arrays(
        cls,
        thrust_times: List[float],
        thrust_values: List[float],
        mass_times: List[float],
        mass_values: List[float],
        cd_machs: List[float],
        cd_values: List[float],
        reference_area: float = None,
    ) -> 'RocketModel':
        """
        Create rocket model from explicit arrays.
        
        Args:
            thrust_times: Thrust curve time points
            thrust_values: Thrust curve values in N
            mass_times: Mass curve time points
            mass_values: Mass curve values in kg
            cd_machs: Mach numbers for Cd table
            cd_values: Cd values
            reference_area: Reference area in m^2
            
        Returns:
            RocketModel instance
        """
        thrust = ThrustCurve(
            times=np.array(thrust_times),
            thrusts=np.array(thrust_values),
        )
        mass = MassModel(
            times=np.array(mass_times),
            masses=np.array(mass_values),
        )
        drag = DragModel(
            reference_area=reference_area,
            mach_array=np.array(cd_machs),
            cd_array=np.array(cd_values),
        )
        return cls(thrust, mass, drag)
