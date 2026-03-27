"""
Extended Kalman Filter for state estimation.
Fuses altitude (from pressure), and acceleration measurements.
"""

import numpy as np
from config import (
    EKF_PROCESS_NOISE_ALTITUDE,
    EKF_PROCESS_NOISE_VELOCITY,
    EKF_MEASUREMENT_NOISE_ALTITUDE,
    EKF_MEASUREMENT_NOISE_ACCEL,
    GRAVITY,
)


class AltitudeVelocityEKF:
    """
    Extended Kalman Filter for altitude and velocity estimation.
    
    State vector: [altitude, velocity]
    Measurements: altitude (from pressure), acceleration
    
    State dynamics:
        altitude_dot = velocity
        velocity_dot = acceleration - gravity
    """
    
    def __init__(
        self,
        initial_altitude: float = 0.0,
        initial_velocity: float = 0.0,
        process_noise_alt: float = EKF_PROCESS_NOISE_ALTITUDE,
        process_noise_vel: float = EKF_PROCESS_NOISE_VELOCITY,
        measurement_noise_alt: float = EKF_MEASUREMENT_NOISE_ALTITUDE,
        measurement_noise_accel: float = EKF_MEASUREMENT_NOISE_ACCEL,
    ):
        """
        Initialize EKF.
        
        Args:
            initial_altitude: Initial altitude estimate in meters
            initial_velocity: Initial velocity estimate in m/s
            process_noise_alt: Process noise variance for altitude
            process_noise_vel: Process noise variance for velocity
            measurement_noise_alt: Measurement noise variance for altitude
            measurement_noise_accel: Measurement noise variance for acceleration
        """
        self.x = np.array([initial_altitude, initial_velocity])
        
        self.P = np.array([
            [100.0, 0.0],
            [0.0, 10.0]
        ])
        
        self.Q = np.array([
            [process_noise_alt, 0.0],
            [0.0, process_noise_vel]
        ])
        
        self.R_alt = measurement_noise_alt
        self.R_accel = measurement_noise_accel
        
    def reset(self, altitude: float = 0.0, velocity: float = 0.0):
        """Reset filter state."""
        self.x = np.array([altitude, velocity])
        self.P = np.array([
            [100.0, 0.0],
            [0.0, 10.0]
        ])
        
    def predict(self, dt: float, acceleration: float = 0.0):
        """
        Prediction step using acceleration input.
        
        Args:
            dt: Time step in seconds
            acceleration: Measured acceleration in m/s^2 (body frame, positive up)
        """
        F = np.array([
            [1.0, dt],
            [0.0, 1.0]
        ])
        
        B = np.array([
            [0.5 * dt * dt],
            [dt]
        ])
        
        net_accel = acceleration - GRAVITY
        
        self.x = F @ self.x + B.flatten() * net_accel
        
        self.P = F @ self.P @ F.T + self.Q * dt
        
    def update_altitude(self, measured_altitude: float):
        """
        Update step with altitude measurement.
        
        Args:
            measured_altitude: Measured altitude in meters (from pressure sensor)
        """
        H = np.array([[1.0, 0.0]])
        
        y = measured_altitude - H @ self.x
        
        S = H @ self.P @ H.T + self.R_alt
        
        K = self.P @ H.T / S
        
        self.x = self.x + K.flatten() * y
        
        self.P = (np.eye(2) - np.outer(K, H)) @ self.P
        
    def update_acceleration(self, measured_acceleration: float, dt: float):
        """
        Update step using acceleration measurement to refine velocity estimate.
        This is done implicitly through the predict step, but we can add
        additional correction if we have redundant measurements.
        
        Args:
            measured_acceleration: Measured acceleration in m/s^2
            dt: Time step for integration
        """
        pass
        
    def get_state(self) -> tuple:
        """
        Get current state estimate.
        
        Returns:
            Tuple of (altitude, velocity)
        """
        return self.x[0], self.x[1]
    
    def get_altitude(self) -> float:
        """Get estimated altitude in meters."""
        return self.x[0]
    
    def get_velocity(self) -> float:
        """Get estimated velocity in m/s."""
        return self.x[1]
    
    def get_covariance(self) -> np.ndarray:
        """Get state covariance matrix."""
        return self.P.copy()
    
    def get_altitude_uncertainty(self) -> float:
        """Get altitude estimate uncertainty (1-sigma) in meters."""
        return np.sqrt(self.P[0, 0])
    
    def get_velocity_uncertainty(self) -> float:
        """Get velocity estimate uncertainty (1-sigma) in m/s."""
        return np.sqrt(self.P[1, 1])


class FullStateEKF:
    """
    Full state EKF that estimates altitude, velocity, and acceleration bias.
    
    State vector: [altitude, velocity, accel_bias]
    """
    
    def __init__(
        self,
        initial_altitude: float = 0.0,
        initial_velocity: float = 0.0,
        initial_accel_bias: float = 0.0,
    ):
        """
        Initialize full state EKF.
        
        Args:
            initial_altitude: Initial altitude in meters
            initial_velocity: Initial velocity in m/s
            initial_accel_bias: Initial accelerometer bias in m/s^2
        """
        self.x = np.array([initial_altitude, initial_velocity, initial_accel_bias])
        
        self.P = np.diag([100.0, 10.0, 1.0])
        
        self.Q = np.diag([1.0, 0.5, 0.01])
        
        self.R_alt = 10.0
        self.R_accel = 1.0
        
    def reset(self, altitude: float = 0.0, velocity: float = 0.0, accel_bias: float = 0.0):
        """Reset filter state."""
        self.x = np.array([altitude, velocity, accel_bias])
        self.P = np.diag([100.0, 10.0, 1.0])
        
    def predict(self, dt: float, acceleration: float):
        """
        Prediction step.
        
        Args:
            dt: Time step in seconds
            acceleration: Raw measured acceleration in m/s^2
        """
        alt, vel, bias = self.x
        
        corrected_accel = acceleration - bias - GRAVITY
        
        new_alt = alt + vel * dt + 0.5 * corrected_accel * dt * dt
        new_vel = vel + corrected_accel * dt
        new_bias = bias  # Bias modeled as random walk
        
        self.x = np.array([new_alt, new_vel, new_bias])
        
        F = np.array([
            [1.0, dt, -0.5 * dt * dt],
            [0.0, 1.0, -dt],
            [0.0, 0.0, 1.0]
        ])
        
        self.P = F @ self.P @ F.T + self.Q * dt
        
    def update_altitude(self, measured_altitude: float):
        """
        Update with altitude measurement.
        
        Args:
            measured_altitude: Measured altitude in meters
        """
        H = np.array([[1.0, 0.0, 0.0]])
        
        y = measured_altitude - self.x[0]
        S = H @ self.P @ H.T + self.R_alt
        K = (self.P @ H.T) / S
        
        self.x = self.x + K.flatten() * y
        self.P = (np.eye(3) - np.outer(K, H)) @ self.P
        
    def get_state(self) -> tuple:
        """Get (altitude, velocity, accel_bias)."""
        return tuple(self.x)
    
    def get_altitude(self) -> float:
        """Get estimated altitude."""
        return self.x[0]
    
    def get_velocity(self) -> float:
        """Get estimated velocity."""
        return self.x[1]
    
    def get_accel_bias(self) -> float:
        """Get estimated accelerometer bias."""
        return self.x[2]
