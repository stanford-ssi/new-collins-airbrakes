"""
Airbrake Control System - SEPARATE FROM SIMULATION ENVIRONMENT

This module contains the control system that interfaces with the rocket
through helper functions. It receives sensor data (with noise) and
outputs airbrake commands.

The control system uses:
- Lookup tables derived from truth model
- EKF for state estimation  
- Feed-forward apogee prediction with internal model
- Binary search to find optimal Cd for target apogee
- State machine: Boost -> Transonic Lockout -> Active Coast -> Recovery
"""

import numpy as np
from lookup_tables import get_pressure_lut
from ekf import AltitudeVelocityEKF
from truth_model import get_truth_model, ControlModel, ControlLUTConfig
from debug import get_debugger
from config import (
    TARGET_APOGEE_M,
    AIRBRAKE_MAX_ANGLE_DEG,
    MACH_DEPLOY_LIMIT,
    GRAVITY,
    CONTROL_DT,
    AIRBRAKE_CD,
    AIRBRAKE_MAX_AREA_M2,
    ROCKET_REFERENCE_AREA_M2,
)


class SensorInterface:
    """
    Interface for receiving sensor data from the simulation environment.
    All data comes through these helper functions which add appropriate noise.
    """
    
    def __init__(self):
        self._pressure_pa = 101325.0
        self._temperature_k = 288.15
        self._accel_z_m_s2 = 0.0
        self._time_s = 0.0
        
    def update(self, pressure_pa: float, temperature_k: float, accel_z_m_s2: float, time_s: float):
        """
        Update sensor readings (called by simulation environment).
        
        Args:
            pressure_pa: Barometric pressure in Pascals (with noise)
            temperature_k: Temperature in Kelvin (with noise)
            accel_z_m_s2: Z-axis acceleration in m/s^2 (with noise)
            time_s: Current simulation time
        """
        self._pressure_pa = pressure_pa
        self._temperature_k = temperature_k
        self._accel_z_m_s2 = accel_z_m_s2
        self._time_s = time_s
        
    def get_pressure(self) -> float:
        """Get current pressure reading in Pascals."""
        return self._pressure_pa
    
    def get_temperature(self) -> float:
        """Get current temperature reading in Kelvin."""
        return self._temperature_k
    
    def get_acceleration_z(self) -> float:
        """Get current Z-axis acceleration in m/s^2."""
        return self._accel_z_m_s2
    
    def get_time(self) -> float:
        """Get current time in seconds."""
        return self._time_s


class AirbrakeController:
    """
    Main airbrake control system using feed-forward apogee prediction.
    
    Control approach:
    1. Use internal model to predict apogee with different Cd settings
    2. Binary search to find Cd that achieves target apogee
    3. Apply slew rate limiting to Cd command
    4. State machine handles different flight phases
    """
    
    def __init__(
        self,
        target_apogee_m: float = TARGET_APOGEE_M,
        control_dt: float = CONTROL_DT,
        cd_error_mach_low: float = 0.0,
        cd_error_mach_high: float = 0.0,
        lut_config: ControlLUTConfig = None,
    ):
        """
        Initialize controller.
        
        Args:
            target_apogee_m: Target apogee altitude in meters
            control_dt: Control loop time step in seconds
            cd_error_mach_low: Cd error at Mach 0 (fractional, e.g., 0.1 = 10% high)
            cd_error_mach_high: Cd error at Mach 1.5 (fractional)
            lut_config: Configuration for control system lookup tables
        """
        self.target_apogee = target_apogee_m
        self.control_dt = control_dt
        
        self.cd_error_mach_low = cd_error_mach_low
        self.cd_error_mach_high = cd_error_mach_high
        
        self.sensors = SensorInterface()
        
        # Build control model from truth model with configurable LUT resolution
        self.lut_config = lut_config or ControlLUTConfig()
        self.control_model = ControlModel(get_truth_model(), self.lut_config)
        
        # Pressure LUT for altitude estimation from barometer
        self.pressure_lut = get_pressure_lut()
        
        self.ekf = AltitudeVelocityEKF()
        
        # --- Control Parameters ---
        self.burn_window = self.control_model.burnout_time  # Use actual burnout time
        self.max_iter = 6  # Binary search iterations (reduced for speed)
        self.alt_tol = 10.0  # Altitude tolerance for convergence (meters)
        
        # Cd limits (Cd_add is additional Cd from airbrake)
        self.A_body = ROCKET_REFERENCE_AREA_M2
        self.A_deploy_max = AIRBRAKE_MAX_AREA_M2
        self.Cd_panel = AIRBRAKE_CD
        self.Cd_air_max = self.Cd_panel * (self.A_deploy_max / self.A_body)
        self.Cd_air_min = 0.0
        
        # Slew rate limiting (Cd units per second)
        self.max_rate = 1.5 * 1000  # Matches MATLAB
        
        # Pre-computed constants for speed
        self._A_over_m = self.A_body / self.control_model.dry_mass
        
        # --- State Variables ---
        self.Cd_add_prev = 0.0  # Previous Cd command
        self.commanded_angle = 0.0
        self.is_enabled = True
        self.motor_burnout_detected = False
        self.burnout_time = None
        self.apogee_time = None  # Time when apogee was reached
        self._prev_vel = 0.0
        
        self._last_update_time = 0.0
        self._prev_accel = 0.0
        
        self.telemetry = self._create_empty_telemetry()
        
    def _create_empty_telemetry(self) -> dict:
        """Create empty telemetry dict."""
        return {
            'time': [],
            'altitude_est': [],
            'velocity_est': [],
            'altitude_meas': [],
            'mach_est': [],
            'commanded_angle': [],
            'predicted_apogee': [],
            'apogee_clean': [],
            'apogee_full_brake': [],
            'cd_add_cmd': [],
            # Internal model variables for comparison
            'ctrl_density': [],
            'ctrl_cd_body': [],
            'ctrl_cd_total': [],
            'ctrl_pressure': [],
            'ctrl_temperature': [],
            'ctrl_accel': [],
        }
        
    def reset(self):
        """Reset controller state."""
        self.ekf.reset()
        self.Cd_add_prev = 0.0
        self.commanded_angle = 0.0
        self.motor_burnout_detected = False
        self.burnout_time = None
        self.apogee_time = None
        self._prev_vel = 0.0
        self._last_update_time = 0.0
        self._prev_accel = 0.0
        self.telemetry = self._create_empty_telemetry()
        
    def update_sensors(self, pressure_pa: float, temperature_k: float, 
                       accel_z_m_s2: float, time_s: float):
        """
        Update sensor readings from simulation environment.
        
        Args:
            pressure_pa: Barometric pressure with noise
            temperature_k: Temperature with noise
            accel_z_m_s2: Z-acceleration with noise
            time_s: Current time
        """
        self.sensors.update(pressure_pa, temperature_k, accel_z_m_s2, time_s)
        
    def _get_cd_error_at_mach(self, mach: float) -> float:
        """
        Get Cd error factor at given Mach number.
        Linearly interpolates between Mach 0 and Mach 1.5.
        
        Args:
            mach: Current Mach number
            
        Returns:
            Cd error factor (1.0 = no error, 1.1 = 10% high)
        """
        mach_clamped = np.clip(mach, 0.0, 1.5)
        t = mach_clamped / 1.5
        error = self.cd_error_mach_low + t * (self.cd_error_mach_high - self.cd_error_mach_low)
        return 1.0 + error
    
    def _get_body_cd(self, mach: float) -> float:
        """
        Get body Cd at given Mach from control system's lookup table.
        
        Args:
            mach: Mach number
            
        Returns:
            Body drag coefficient
        """
        return self.control_model.get_cd(mach)
    
    def _get_dry_mass(self) -> float:
        """Get rocket dry mass from control model."""
        return self.control_model.dry_mass
        
    def _estimate_state(self) -> tuple:
        """
        Estimate current state using EKF and sensor data.
        
        Returns:
            Tuple of (altitude_m, velocity_m_s, mach)
        """
        pressure = self.sensors.get_pressure()
        accel = self.sensors.get_acceleration_z()
        time = self.sensors.get_time()
        
        dt = time - self._last_update_time
        if dt <= 0:
            dt = self.control_dt
        self._last_update_time = time
        
        measured_altitude = self.pressure_lut.lookup(pressure)
        
        self.ekf.predict(dt, accel)
        self.ekf.update_altitude(measured_altitude)
        
        altitude, velocity = self.ekf.get_state()
        
        speed_of_sound = self.control_model.get_speed_of_sound(altitude)
        mach = abs(velocity) / speed_of_sound if speed_of_sound > 0 else abs(velocity) / 340.0
        
        return altitude, velocity, mach, measured_altitude
    
    def _simulate_coast(self, h0: float, v0: float, Cd_add_target: float) -> tuple:
        """
        Simulate coast trajectory to apogee with given Cd_add target.
        Uses optimized Euler integration with Cd updated every few steps.
        
        Args:
            h0: Initial altitude (m)
            v0: Initial velocity (m/s)
            Cd_add_target: Target additional Cd from airbrake
            
        Returns:
            Tuple of (apogee_altitude, time_to_apogee)
        """
        if v0 <= 0:
            return max(0.0, h0), 0.0
            
        h = h0
        v = v0
        t = 0.0
        dt = 0.2  # Larger time step for speed
        
        # Pre-compute constants (use dry mass - motor is burnt out)
        A_over_m = self._A_over_m
        Cd_add = min(max(Cd_add_target, self.Cd_air_min), self.Cd_air_max)
        
        # Initial Cd computation
        speed_of_sound = self.control_model.get_speed_of_sound(h)
        mach = v / speed_of_sound if speed_of_sound > 0 else v / 340.0
        Cd_body = self._get_body_cd(mach)
        Cd_total = Cd_body + Cd_add
        
        cd_update_interval = 5  # Update Cd every N steps for efficiency
        
        for i in range(150):  # Max ~30 seconds
            # Update Cd periodically as Mach changes
            if i % cd_update_interval == 0:
                speed_of_sound = self.control_model.get_speed_of_sound(h)
                mach = v / speed_of_sound if speed_of_sound > 0 else v / 340.0
                Cd_body = self._get_body_cd(mach)
                Cd_total = Cd_body + Cd_add
            
            # Use control model's density lookup table
            rho = self.control_model.get_density(h)
            
            # Euler integration
            drag_accel = -0.5 * rho * Cd_total * A_over_m * v * v
            a = drag_accel - GRAVITY
            
            v_new = v + a * dt
            h_new = h + v * dt + 0.5 * a * dt * dt
            
            # Check for apogee
            if v_new <= 0:
                if a != 0:
                    t_zero = -v / a
                    t_zero = max(0.0, min(dt, t_zero))
                else:
                    t_zero = 0.0
                apogee = h + v * t_zero + 0.5 * a * t_zero * t_zero
                return max(0.0, apogee), t + t_zero
            
            v = v_new
            h = h_new
            t += dt
            
            if h <= 0:
                return 0.0, t
                
        return max(0.0, h), t
    
    def compute_command(
        self, body_cd: float = None, body_area: float = None, mass: float = None
    ) -> float:
        """
        Main control loop using feed-forward apogee prediction.
        
        State machine:
        1. Boost phase: Airbrakes retracted
        2. Transonic lockout (Mach > 1): Airbrakes retracted
        3. Descent/past target: Full brakes
        4. Active coast: Binary search for optimal Cd
        5. Recovery (post-apogee): Retract brakes
        
        Returns:
            Commanded airbrake angle in degrees
        """
        if not self.is_enabled:
            self.commanded_angle = 0.0
            return self.commanded_angle
            
        altitude, velocity, mach, measured_altitude = self._estimate_state()
        time = self.sensors.get_time()
        
        # Compute dt for slew rate limiting
        dt_cmd = time - self._last_update_time
        if dt_cmd <= 0:
            dt_cmd = self.control_dt
        elif dt_cmd > 0.1:
            dt_cmd = 0.1
        self._last_update_time = time
        
        # Update Cd_air_max based on current Mach (panel Cd may vary)
        # For now, use constant Cd_panel
        Cd_air_max = self.Cd_air_max
        
        # Clamp previous Cd
        self.Cd_add_prev = np.clip(self.Cd_add_prev, self.Cd_air_min, Cd_air_max)
        
        # --- Apogee Detection ---
        motor_burnt_out = time >= self.burn_window
        if self.apogee_time is None and motor_burnt_out:
            if self._prev_vel > 0 and velocity <= 0:
                self.apogee_time = time
        self._prev_vel = velocity
        
        # Default values
        Cd_add_target = self.Cd_add_prev
        apogee_est = altitude
        apogee_clean = altitude
        apogee_full_brake = altitude
        
        # --- Always compute influence bounds during coast ---
        # This shows the theoretical max altitude shed at each point
        in_coast_phase = (time >= self.burn_window and velocity > 0 and 
                         (self.apogee_time is None or time < self.apogee_time + 2))
        
        if in_coast_phase:
            apogee_clean, _ = self._simulate_coast(altitude, velocity, 0.0)
            apogee_full_brake, _ = self._simulate_coast(altitude, velocity, Cd_air_max)
        
        # --- State Machine ---
        control_state = "UNKNOWN"
        
        # Case 1: Recovery Phase (post-apogee + 2 seconds)
        if self.apogee_time is not None and time >= self.apogee_time + 2:
            Cd_add_target = 0.0
            apogee_est = altitude
            control_state = "RECOVERY"
            
        # Case 2: Boost Phase
        elif time < self.burn_window:
            Cd_add_target = 0.0
            control_state = "BOOST"
            
        # Case 3: Transonic/Supersonic lockout
        elif mach > MACH_DEPLOY_LIMIT:
            Cd_add_target = 0.0
            apogee_est = apogee_clean  # Use already-computed value
            control_state = "SUPERSONIC_LOCKOUT"
            
        # Case 4: Descent or already past target
        elif velocity <= 0 or altitude >= self.target_apogee:
            Cd_add_target = Cd_air_max
            apogee_est = apogee_full_brake  # Use already-computed value
            control_state = "DESCENT" if velocity <= 0 else "PAST_TARGET"
            
        # Case 5: Active Coast Control
        else:
            # apogee_clean and apogee_full_brake already computed above
            
            if apogee_clean <= self.target_apogee + self.alt_tol:
                # Already undershooting - no brakes needed
                Cd_add_target = 0.0
                apogee_est = apogee_clean
                control_state = "COAST_UNDERSHOOT"
                
            elif Cd_air_max <= 0:
                # No airbrake capability
                Cd_add_target = 0.0
                apogee_est = apogee_clean
                control_state = "COAST_NO_BRAKES"
                
            else:
                if apogee_full_brake > self.target_apogee:
                    # Even full brakes overshoot - use max
                    Cd_add_target = Cd_air_max
                    apogee_est = apogee_full_brake
                    control_state = "COAST_MAX_BRAKES"
                    
                else:
                    # Binary search for optimal Cd
                    low_cd = 0.0
                    high_cd = Cd_air_max
                    Cd_add_target = self.Cd_add_prev
                    control_state = "COAST_MODULATING"
                    
                    for _ in range(self.max_iter):
                        mid_cd = 0.5 * (low_cd + high_cd)
                        mid_apogee, _ = self._simulate_coast(altitude, velocity, mid_cd)
                        
                        if abs(mid_apogee - self.target_apogee) <= self.alt_tol:
                            Cd_add_target = mid_cd
                            apogee_est = mid_apogee
                            break
                            
                        if mid_apogee > self.target_apogee:
                            low_cd = mid_cd
                        else:
                            high_cd = mid_cd
                            
                        Cd_add_target = 0.5 * (low_cd + high_cd)
                        apogee_est, _ = self._simulate_coast(altitude, velocity, Cd_add_target)
        
        # --- Slew Rate Limiting ---
        Cd_add_target = np.clip(Cd_add_target, self.Cd_air_min, Cd_air_max)
        max_step = self.max_rate * dt_cmd
        delta = Cd_add_target - self.Cd_add_prev
        delta = np.clip(delta, -max_step, max_step)
        Cd_add_cmd = self.Cd_add_prev + delta
        Cd_add_cmd = np.clip(Cd_add_cmd, self.Cd_air_min, Cd_air_max)
        self.Cd_add_prev = Cd_add_cmd
        
        # --- Convert Cd command to deployment angle ---
        # A_deploy = Cd_add * (A_body / Cd_panel)
        A_deploy = Cd_add_cmd * (self.A_body / self.Cd_panel)
        A_deploy = np.clip(A_deploy, 0.0, self.A_deploy_max)
        
        # Convert area to angle (linear relationship)
        self.commanded_angle = (A_deploy / self.A_deploy_max) * AIRBRAKE_MAX_ANGLE_DEG
        
        # --- Compute internal model variables for telemetry ---
        # These show what the controller's internal model computes
        ctrl_density = self.control_model.get_density(altitude)
        ctrl_cd_body = self._get_body_cd(mach) * self._get_cd_error_at_mach(mach)
        ctrl_cd_total = ctrl_cd_body + Cd_add_cmd
        ctrl_pressure = self.sensors.get_pressure()
        ctrl_temperature = self.sensors.get_temperature()
        
        # Estimated acceleration from controller's model
        drag_accel = -0.5 * ctrl_density * ctrl_cd_total * self._A_over_m * velocity * abs(velocity)
        ctrl_accel = drag_accel - GRAVITY
        
        # --- Debug Recording ---
        debugger = get_debugger()
        debugger.record_control_state(
            time=time,
            velocity_est=velocity,
            altitude_est=altitude,
            acceleration=ctrl_accel,
            mach=mach,
            cd_total=ctrl_cd_total,
            apogee_pred=apogee_est,
            airbrake_angle=self.commanded_angle,
            control_state=control_state,
        )
        
        # --- Record Telemetry ---
        self.telemetry['time'].append(time)
        self.telemetry['altitude_est'].append(altitude)
        self.telemetry['velocity_est'].append(velocity)
        self.telemetry['altitude_meas'].append(measured_altitude)
        self.telemetry['mach_est'].append(mach)
        self.telemetry['commanded_angle'].append(self.commanded_angle)
        self.telemetry['predicted_apogee'].append(apogee_est)
        self.telemetry['apogee_clean'].append(apogee_clean)
        self.telemetry['apogee_full_brake'].append(apogee_full_brake)
        self.telemetry['cd_add_cmd'].append(Cd_add_cmd)
        self.telemetry['ctrl_density'].append(ctrl_density)
        self.telemetry['ctrl_cd_body'].append(ctrl_cd_body)
        self.telemetry['ctrl_cd_total'].append(ctrl_cd_total)
        self.telemetry['ctrl_pressure'].append(ctrl_pressure)
        self.telemetry['ctrl_temperature'].append(ctrl_temperature)
        self.telemetry['ctrl_accel'].append(ctrl_accel)
        
        return self.commanded_angle
    
    def disable(self):
        """Disable the controller."""
        self.is_enabled = False
        self.commanded_angle = 0.0
        
    def enable(self):
        """Enable the controller."""
        self.is_enabled = True
        
    def get_telemetry(self) -> dict:
        """Get recorded telemetry data."""
        return self.telemetry
