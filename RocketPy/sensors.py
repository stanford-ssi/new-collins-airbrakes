"""
Sensor models with Gaussian noise for simulation.
Provides noisy measurements to the control system.
"""

import numpy as np
from config import (
    PRESSURE_NOISE_STD,
    TEMPERATURE_NOISE_STD,
    ACCEL_Z_NOISE_STD,
)


class SensorModel:
    """
    Sensor model that adds Gaussian noise to true values.
    """
    
    def __init__(
        self,
        pressure_noise_std: float = PRESSURE_NOISE_STD,
        temperature_noise_std: float = TEMPERATURE_NOISE_STD,
        accel_noise_std: float = ACCEL_Z_NOISE_STD,
        pressure_offset: float = 0.0,
        temperature_offset: float = 0.0,
        accel_offset: float = 0.0,
        seed: int = None,
    ):
        """
        Initialize sensor model.
        
        Args:
            pressure_noise_std: Pressure noise standard deviation (Pa)
            temperature_noise_std: Temperature noise standard deviation (K)
            accel_noise_std: Acceleration noise standard deviation (m/s^2)
            pressure_offset: Pressure bias/offset (Pa)
            temperature_offset: Temperature bias/offset (K)
            accel_offset: Acceleration bias/offset (m/s^2)
            seed: Random seed for reproducibility
        """
        self.pressure_noise_std = pressure_noise_std
        self.temperature_noise_std = temperature_noise_std
        self.accel_noise_std = accel_noise_std
        self.pressure_offset = pressure_offset
        self.temperature_offset = temperature_offset
        self.accel_offset = accel_offset
        
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random.RandomState()
            
    def set_seed(self, seed: int):
        """Set random seed."""
        self.rng = np.random.RandomState(seed)
        
    def set_noise_levels(
        self,
        pressure_std: float = None,
        temperature_std: float = None,
        accel_std: float = None,
    ):
        """
        Update noise levels.
        
        Args:
            pressure_std: New pressure noise std
            temperature_std: New temperature noise std
            accel_std: New acceleration noise std
        """
        if pressure_std is not None:
            self.pressure_noise_std = pressure_std
        if temperature_std is not None:
            self.temperature_noise_std = temperature_std
        if accel_std is not None:
            self.accel_noise_std = accel_std
            
    def measure_pressure(self, true_pressure: float) -> float:
        """
        Get noisy pressure measurement.
        
        Args:
            true_pressure: True pressure in Pa
            
        Returns:
            Noisy pressure measurement in Pa
        """
        noise = self.rng.normal(0, self.pressure_noise_std)
        return true_pressure + self.pressure_offset + noise
    
    def measure_temperature(self, true_temperature: float) -> float:
        """
        Get noisy temperature measurement.
        
        Args:
            true_temperature: True temperature in K
            
        Returns:
            Noisy temperature measurement in K
        """
        noise = self.rng.normal(0, self.temperature_noise_std)
        return true_temperature + self.temperature_offset + noise
    
    def measure_acceleration(self, true_acceleration: float) -> float:
        """
        Get noisy acceleration measurement.
        
        Args:
            true_acceleration: True acceleration in m/s^2
            
        Returns:
            Noisy acceleration measurement in m/s^2
        """
        noise = self.rng.normal(0, self.accel_noise_std)
        return true_acceleration + self.accel_offset + noise
    
    def get_measurements(
        self, true_pressure: float, true_temperature: float, true_acceleration: float
    ) -> tuple:
        """
        Get all noisy measurements at once.
        
        Args:
            true_pressure: True pressure in Pa
            true_temperature: True temperature in K
            true_acceleration: True acceleration in m/s^2
            
        Returns:
            Tuple of (noisy_pressure, noisy_temperature, noisy_acceleration)
        """
        return (
            self.measure_pressure(true_pressure),
            self.measure_temperature(true_temperature),
            self.measure_acceleration(true_acceleration),
        )


class SensorModelVariations:
    """
    Factory for creating sensor models with various noise configurations.
    Used for Monte Carlo simulations.
    """
    
    @staticmethod
    def nominal() -> SensorModel:
        """Create nominal sensor model."""
        return SensorModel()
    
    @staticmethod
    def low_noise() -> SensorModel:
        """Create low-noise sensor model."""
        return SensorModel(
            pressure_noise_std=PRESSURE_NOISE_STD * 0.5,
            temperature_noise_std=TEMPERATURE_NOISE_STD * 0.5,
            accel_noise_std=ACCEL_Z_NOISE_STD * 0.5,
        )
    
    @staticmethod
    def high_noise() -> SensorModel:
        """Create high-noise sensor model."""
        return SensorModel(
            pressure_noise_std=PRESSURE_NOISE_STD * 2.0,
            temperature_noise_std=TEMPERATURE_NOISE_STD * 2.0,
            accel_noise_std=ACCEL_Z_NOISE_STD * 2.0,
        )
    
    @staticmethod
    def custom(
        pressure_mult: float = 1.0,
        temperature_mult: float = 1.0,
        accel_mult: float = 1.0,
        pressure_noise_std: float = None,
        temperature_noise_std: float = None,
        accel_noise_std: float = None,
        pressure_offset: float = 0.0,
        temperature_offset: float = 0.0,
        accel_offset: float = 0.0,
        seed: int = None,
    ) -> SensorModel:
        """
        Create sensor model with custom noise parameters.
        
        Args:
            pressure_mult: Multiplier for default pressure noise std
            temperature_mult: Multiplier for default temperature noise std
            accel_mult: Multiplier for default acceleration noise std
            pressure_noise_std: Direct pressure noise std (overrides mult)
            temperature_noise_std: Direct temperature noise std (overrides mult)
            accel_noise_std: Direct acceleration noise std (overrides mult)
            pressure_offset: Pressure bias/offset (Pa)
            temperature_offset: Temperature bias/offset (K)
            accel_offset: Acceleration bias/offset (m/s^2)
            seed: Random seed
            
        Returns:
            SensorModel instance
        """
        p_std = pressure_noise_std if pressure_noise_std is not None else PRESSURE_NOISE_STD * pressure_mult
        t_std = temperature_noise_std if temperature_noise_std is not None else TEMPERATURE_NOISE_STD * temperature_mult
        a_std = accel_noise_std if accel_noise_std is not None else ACCEL_Z_NOISE_STD * accel_mult
        
        return SensorModel(
            pressure_noise_std=p_std,
            temperature_noise_std=t_std,
            accel_noise_std=a_std,
            pressure_offset=pressure_offset,
            temperature_offset=temperature_offset,
            accel_offset=accel_offset,
            seed=seed,
        )
