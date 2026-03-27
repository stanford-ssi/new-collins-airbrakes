"""
Airbrake physics model.
Models flat disc airbrakes with variable deployment area and slew rate limits.
"""

import numpy as np
from config import (
    AIRBRAKE_CD,
    AIRBRAKE_MAX_AREA_M2,
    AIRBRAKE_MAX_ANGLE_DEG,
    AIRBRAKE_SLEW_RATE_DEG_S,
    MACH_DEPLOY_LIMIT,
)


class Airbrake:
    """
    Airbrake model with deployment angle, area, and slew rate limiting.
    
    Area is proportional to angle:
        area = (angle / max_angle) * max_area
    """
    
    def __init__(
        self,
        cd: float = AIRBRAKE_CD,
        max_area_m2: float = AIRBRAKE_MAX_AREA_M2,
        max_angle_deg: float = AIRBRAKE_MAX_ANGLE_DEG,
        slew_rate_deg_s: float = AIRBRAKE_SLEW_RATE_DEG_S,
        mach_deploy_limit: float = MACH_DEPLOY_LIMIT,
    ):
        """
        Initialize airbrake model.
        
        Args:
            cd: Drag coefficient (1.28 for flat disc)
            max_area_m2: Maximum deployed area in m^2
            max_angle_deg: Maximum deployment angle in degrees
            slew_rate_deg_s: Maximum slew rate in degrees/second
            mach_deploy_limit: Maximum Mach number for deployment
        """
        self.cd = cd
        self.max_area_m2 = max_area_m2
        self.max_angle_deg = max_angle_deg
        self.slew_rate_deg_s = slew_rate_deg_s
        self.mach_deploy_limit = mach_deploy_limit
        
        self.current_angle_deg = 0.0
        self.commanded_angle_deg = 0.0
        
    def reset(self):
        """Reset airbrake to retracted state."""
        self.current_angle_deg = 0.0
        self.commanded_angle_deg = 0.0
        
    def set_commanded_angle(self, angle_deg: float):
        """
        Set the commanded deployment angle.
        
        Args:
            angle_deg: Commanded angle in degrees (0 to max_angle_deg)
        """
        self.commanded_angle_deg = np.clip(angle_deg, 0.0, self.max_angle_deg)
        
    def update(self, dt: float, current_mach: float) -> float:
        """
        Update airbrake position with slew rate limiting and Mach constraints.
        
        Args:
            dt: Time step in seconds
            current_mach: Current Mach number
            
        Returns:
            Current deployment angle in degrees
        """
        if current_mach > self.mach_deploy_limit:
            target_angle = 0.0
        else:
            target_angle = self.commanded_angle_deg
            
        angle_error = target_angle - self.current_angle_deg
        max_angle_change = self.slew_rate_deg_s * dt
        
        if abs(angle_error) <= max_angle_change:
            self.current_angle_deg = target_angle
        else:
            self.current_angle_deg += np.sign(angle_error) * max_angle_change
            
        self.current_angle_deg = np.clip(self.current_angle_deg, 0.0, self.max_angle_deg)
        
        return self.current_angle_deg
    
    def get_area(self) -> float:
        """
        Get current deployed area.
        
        Returns:
            Current area in m^2
        """
        return (self.current_angle_deg / self.max_angle_deg) * self.max_area_m2
    
    def get_area_at_angle(self, angle_deg: float) -> float:
        """
        Get area at a specific angle.
        
        Args:
            angle_deg: Deployment angle in degrees
            
        Returns:
            Area in m^2
        """
        angle_deg = np.clip(angle_deg, 0.0, self.max_angle_deg)
        return (angle_deg / self.max_angle_deg) * self.max_area_m2
    
    def get_drag_force(self, velocity_m_s: float, air_density: float) -> float:
        """
        Calculate drag force from airbrakes.
        
        F_drag = 0.5 * rho * v^2 * Cd * A
        
        Args:
            velocity_m_s: Velocity in m/s
            air_density: Air density in kg/m^3
            
        Returns:
            Drag force in Newtons
        """
        area = self.get_area()
        dynamic_pressure = 0.5 * air_density * velocity_m_s ** 2
        return dynamic_pressure * self.cd * area
    
    def get_drag_force_at_angle(
        self, angle_deg: float, velocity_m_s: float, air_density: float
    ) -> float:
        """
        Calculate drag force at a specific deployment angle.
        
        Args:
            angle_deg: Deployment angle in degrees
            velocity_m_s: Velocity in m/s
            air_density: Air density in kg/m^3
            
        Returns:
            Drag force in Newtons
        """
        area = self.get_area_at_angle(angle_deg)
        dynamic_pressure = 0.5 * air_density * velocity_m_s ** 2
        return dynamic_pressure * self.cd * area
    
    def get_max_drag_force(self, velocity_m_s: float, air_density: float) -> float:
        """
        Calculate maximum possible drag force (fully deployed).
        
        Args:
            velocity_m_s: Velocity in m/s
            air_density: Air density in kg/m^3
            
        Returns:
            Maximum drag force in Newtons
        """
        return self.get_drag_force_at_angle(self.max_angle_deg, velocity_m_s, air_density)
    
    def get_min_drag_force(self, velocity_m_s: float, air_density: float) -> float:
        """
        Calculate minimum drag force (fully retracted).
        
        Args:
            velocity_m_s: Velocity in m/s
            air_density: Air density in kg/m^3
            
        Returns:
            Minimum drag force in Newtons (should be 0)
        """
        return self.get_drag_force_at_angle(0.0, velocity_m_s, air_density)
    
    @property
    def deployment_fraction(self) -> float:
        """Get deployment as fraction of maximum (0 to 1)."""
        return self.current_angle_deg / self.max_angle_deg
    
    @property
    def is_deployed(self) -> bool:
        """Check if airbrakes are deployed at all."""
        return self.current_angle_deg > 0.01
