"""
Main simulation environment.
Integrates rocket dynamics from T0 to apogee.
"""

import numpy as np
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

from config import (
    SIMULATION_DT,
    CONTROL_DT,
    GRAVITY,
    METERS_TO_FEET,
)
from truth_model import (
    get_truth_model,
    TruthModel,
    TruthModelConfig,
    SimulationModel,
    ControlModel,
    ControlLUTConfig,
)
from rocket_model import RocketModel
from airbrake import Airbrake
from sensors import SensorModel
from control_system import AirbrakeController
from debug import get_debugger


@dataclass
class SimulationState:
    """Current simulation state."""
    time: float = 0.0
    altitude: float = 0.0
    velocity: float = 0.0
    acceleration: float = 0.0
    mach: float = 0.0
    mass: float = 25.0
    pressure: float = 101325.0
    temperature: float = 288.15
    density: float = 1.225
    thrust: float = 0.0
    body_drag: float = 0.0
    airbrake_drag: float = 0.0
    airbrake_angle: float = 0.0
    
    
@dataclass
class SimulationResults:
    """Complete simulation results."""
    # Use lists during simulation for O(1) append, convert to arrays at end
    _time: list = field(default_factory=list)
    _altitude: list = field(default_factory=list)
    _velocity: list = field(default_factory=list)
    _acceleration: list = field(default_factory=list)
    _mach: list = field(default_factory=list)
    _mass: list = field(default_factory=list)
    _thrust: list = field(default_factory=list)
    _body_drag: list = field(default_factory=list)
    _airbrake_drag: list = field(default_factory=list)
    _airbrake_angle: list = field(default_factory=list)
    _pressure: list = field(default_factory=list)
    _temperature: list = field(default_factory=list)
    
    # Sensor noise tracking (true vs noisy readings)
    _sensor_time: list = field(default_factory=list)
    _sensor_pressure_true: list = field(default_factory=list)
    _sensor_pressure_noisy: list = field(default_factory=list)
    _sensor_temp_true: list = field(default_factory=list)
    _sensor_temp_noisy: list = field(default_factory=list)
    _sensor_accel_true: list = field(default_factory=list)
    _sensor_accel_noisy: list = field(default_factory=list)
    
    # These become arrays after finalize()
    time: np.ndarray = field(default_factory=lambda: np.array([]))
    altitude: np.ndarray = field(default_factory=lambda: np.array([]))
    velocity: np.ndarray = field(default_factory=lambda: np.array([]))
    acceleration: np.ndarray = field(default_factory=lambda: np.array([]))
    mach: np.ndarray = field(default_factory=lambda: np.array([]))
    mass: np.ndarray = field(default_factory=lambda: np.array([]))
    thrust: np.ndarray = field(default_factory=lambda: np.array([]))
    body_drag: np.ndarray = field(default_factory=lambda: np.array([]))
    airbrake_drag: np.ndarray = field(default_factory=lambda: np.array([]))
    airbrake_angle: np.ndarray = field(default_factory=lambda: np.array([]))
    pressure: np.ndarray = field(default_factory=lambda: np.array([]))
    temperature: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Sensor noise arrays
    sensor_time: np.ndarray = field(default_factory=lambda: np.array([]))
    sensor_pressure_true: np.ndarray = field(default_factory=lambda: np.array([]))
    sensor_pressure_noisy: np.ndarray = field(default_factory=lambda: np.array([]))
    sensor_temp_true: np.ndarray = field(default_factory=lambda: np.array([]))
    sensor_temp_noisy: np.ndarray = field(default_factory=lambda: np.array([]))
    sensor_accel_true: np.ndarray = field(default_factory=lambda: np.array([]))
    sensor_accel_noisy: np.ndarray = field(default_factory=lambda: np.array([]))
    
    apogee_m: float = 0.0
    apogee_ft: float = 0.0
    apogee_time: float = 0.0
    burnout_altitude: float = 0.0
    burnout_velocity: float = 0.0
    burnout_time: float = 0.0
    max_velocity: float = 0.0
    max_mach: float = 0.0
    max_acceleration: float = 0.0
    
    controller_telemetry: Dict[str, Any] = field(default_factory=dict)
    
    # Truth-based apogee predictions (computed using sim model, not control model)
    _truth_pred_time: list = field(default_factory=list)
    _truth_apogee_current: list = field(default_factory=list)  # With current airbrake setting
    _truth_apogee_retracted: list = field(default_factory=list)  # Airbrakes fully retracted
    _truth_apogee_extended: list = field(default_factory=list)  # Airbrakes fully extended
    
    truth_pred_time: np.ndarray = field(default_factory=lambda: np.array([]))
    truth_apogee_current: np.ndarray = field(default_factory=lambda: np.array([]))
    truth_apogee_retracted: np.ndarray = field(default_factory=lambda: np.array([]))
    truth_apogee_extended: np.ndarray = field(default_factory=lambda: np.array([]))
    
    def finalize(self):
        """Convert lists to numpy arrays. Call once after simulation."""
        self.time = np.array(self._time)
        self.altitude = np.array(self._altitude)
        self.velocity = np.array(self._velocity)
        self.acceleration = np.array(self._acceleration)
        self.mach = np.array(self._mach)
        self.mass = np.array(self._mass)
        self.thrust = np.array(self._thrust)
        self.body_drag = np.array(self._body_drag)
        self.airbrake_drag = np.array(self._airbrake_drag)
        self.airbrake_angle = np.array(self._airbrake_angle)
        self.pressure = np.array(self._pressure)
        self.temperature = np.array(self._temperature)
        
        # Sensor noise data
        self.sensor_time = np.array(self._sensor_time)
        self.sensor_pressure_true = np.array(self._sensor_pressure_true)
        self.sensor_pressure_noisy = np.array(self._sensor_pressure_noisy)
        self.sensor_temp_true = np.array(self._sensor_temp_true)
        self.sensor_temp_noisy = np.array(self._sensor_temp_noisy)
        self.sensor_accel_true = np.array(self._sensor_accel_true)
        self.sensor_accel_noisy = np.array(self._sensor_accel_noisy)
        
        # Truth-based predictions
        self.truth_pred_time = np.array(self._truth_pred_time)
        self.truth_apogee_current = np.array(self._truth_apogee_current)
        self.truth_apogee_retracted = np.array(self._truth_apogee_retracted)
        self.truth_apogee_extended = np.array(self._truth_apogee_extended)
    
    def compute_statistics(self):
        """Compute derived statistics."""
        if len(self.altitude) > 0:
            apogee_idx = np.argmax(self.altitude)
            self.apogee_m = self.altitude[apogee_idx]
            self.apogee_ft = self.apogee_m * METERS_TO_FEET
            self.apogee_time = self.time[apogee_idx]
            
        if len(self.velocity) > 0:
            self.max_velocity = np.max(self.velocity)
            
        if len(self.mach) > 0:
            self.max_mach = np.max(self.mach)
            
        if len(self.acceleration) > 0:
            self.max_acceleration = np.max(self.acceleration)


class SimulationEnvironment:
    """
    Main simulation environment.
    Runs physics simulation from T0 to apogee.
    
    Uses SimulationModel derived from TruthModel for physics calculations.
    The simulation can apply scaling to truth model values.
    """
    
    def __init__(
        self,
        sim_model: SimulationModel = None,
        rocket: RocketModel = None,
        airbrake: Airbrake = None,
        controller: AirbrakeController = None,
        sensors: SensorModel = None,
        dt: float = SIMULATION_DT,
        control_dt: float = CONTROL_DT,
        enable_airbrakes: bool = True,
    ):
        """
        Initialize simulation environment.
        
        Args:
            sim_model: SimulationModel with scaling config (from truth model)
            rocket: RocketModel instance
            airbrake: Airbrake instance
            controller: AirbrakeController instance
            sensors: SensorModel instance
            dt: Simulation time step
            control_dt: Control loop time step
            enable_airbrakes: Whether airbrakes are enabled
        """
        # Create simulation model from truth if not provided
        if sim_model is None:
            sim_model = SimulationModel(get_truth_model())
        self.sim_model = sim_model
        
        self.rocket = rocket if rocket else RocketModel()
        self.airbrake = airbrake if airbrake else Airbrake()
        self.controller = controller if controller else AirbrakeController()
        self.sensors = sensors if sensors else SensorModel()
        
        self.dt = dt
        self.control_dt = control_dt
        self.enable_airbrakes = enable_airbrakes
        
        # Control latency (delay buffer)
        self._control_latency_s = self.sim_model.config.control_latency_ms / 1000.0
        self._control_cmd_buffer = []  # List of (time, command) tuples
        
        self.state = SimulationState()
        self.results = SimulationResults()
        
        self._control_accumulator = 0.0
        self._burnout_recorded = False
        self._has_lifted_off = False
        
    def reset(self):
        """Reset simulation to initial conditions."""
        self.state = SimulationState()
        self.state.mass = self.sim_model.get_mass(0.0)
        self.state.pressure = self.sim_model.get_pressure(0.0)
        self.state.temperature = self.sim_model.get_temperature(0.0)
        self.state.density = self.sim_model.get_density(0.0)
        
        self.results = SimulationResults()
        self.airbrake.reset()
        self.controller.reset()
        
        self._control_accumulator = 0.0
        self._burnout_recorded = False
        self._has_lifted_off = False
        self._control_cmd_buffer = []
        
    def _compute_forces(self) -> tuple:
        """
        Compute all forces acting on the rocket.
        Uses sim_model which applies any configured scaling to truth values.
        
        Returns:
            Tuple of (thrust, body_drag, airbrake_drag)
        """
        # Thrust from sim_model (can be scaled)
        thrust = self.sim_model.get_thrust(self.state.time)
        
        # Body Cd from sim_model (scaled by cd_scale)
        body_cd = self.sim_model.get_cd(self.state.mach)
        
        # Density from sim_model (scaled by density_scale)  
        density = self.sim_model.get_density(self.state.altitude)
        
        body_drag = 0.5 * density * self.state.velocity**2 * \
                    body_cd * self.rocket.get_reference_area()
        
        if self.enable_airbrakes and self.airbrake.is_deployed:
            # Airbrake Cd from sim model (uses CSV if provided, otherwise scaled baseline)
            airbrake_cd = self.sim_model.get_airbrake_cd(self.state.mach)
            airbrake_drag = 0.5 * density * self.state.velocity**2 * \
                           airbrake_cd * self.airbrake.get_area()
        else:
            airbrake_drag = 0.0
            
        return thrust, body_drag, airbrake_drag
    
    def _step_physics(self):
        """Advance physics by one time step."""
        thrust, body_drag, airbrake_drag = self._compute_forces()
        
        self.state.thrust = thrust
        self.state.body_drag = body_drag
        self.state.airbrake_drag = airbrake_drag
        
        total_drag = body_drag + airbrake_drag
        if self.state.velocity < 0:
            total_drag = -total_drag  # Drag opposes motion
            
        net_force = thrust - total_drag - self.state.mass * GRAVITY
        self.state.acceleration = net_force / self.state.mass
        
        self.state.velocity += self.state.acceleration * self.dt
        self.state.altitude += self.state.velocity * self.dt
        
        self.state.altitude = max(0.0, self.state.altitude)
        
        self.state.time += self.dt
        self.state.mass = self.sim_model.get_mass(self.state.time)
        
        # Update atmospheric state from sim_model
        self.state.pressure = self.sim_model.get_pressure(self.state.altitude)
        self.state.temperature = self.sim_model.get_temperature(self.state.altitude)
        self.state.density = self.sim_model.get_density(self.state.altitude)
        
        # Compute Mach number
        speed_of_sound = self.sim_model.get_speed_of_sound(self.state.altitude)
        self.state.mach = abs(self.state.velocity) / speed_of_sound if speed_of_sound > 0 else 0.0
        
    def _step_control(self):
        """Run control system update with optional latency."""
        if not self.enable_airbrakes:
            return
            
        true_accel = self.state.acceleration + GRAVITY  # Accelerometer measures specific force
        noisy_pressure, noisy_temp, noisy_accel = self.sensors.get_measurements(
            self.state.pressure,
            self.state.temperature,
            true_accel,
        )
        
        # Record sensor noise data
        self.results._sensor_time.append(self.state.time)
        self.results._sensor_pressure_true.append(self.state.pressure)
        self.results._sensor_pressure_noisy.append(noisy_pressure)
        self.results._sensor_temp_true.append(self.state.temperature)
        self.results._sensor_temp_noisy.append(noisy_temp)
        self.results._sensor_accel_true.append(true_accel)
        self.results._sensor_accel_noisy.append(noisy_accel)
        
        self.controller.update_sensors(
            noisy_pressure,
            noisy_temp,
            noisy_accel,
            self.state.time,
        )
        
        commanded_angle = self.controller.compute_command(
            body_cd=self.rocket.get_body_cd(self.state.mach, self.state.time),
            body_area=self.rocket.get_reference_area(),
            mass=self.state.mass,
        )
        
        # Apply control latency if configured
        if self._control_latency_s > 0:
            # Add command to delay buffer
            self._control_cmd_buffer.append((self.state.time, commanded_angle))
            
            # Find command that should be applied now (issued latency_s ago)
            target_time = self.state.time - self._control_latency_s
            delayed_angle = 0.0  # Default to retracted if no command yet
            
            for cmd_time, cmd_angle in self._control_cmd_buffer:
                if cmd_time <= target_time:
                    delayed_angle = cmd_angle
                else:
                    break
            
            # Clean up old commands from buffer
            self._control_cmd_buffer = [
                (t, a) for t, a in self._control_cmd_buffer 
                if t > target_time - 0.1  # Keep small buffer for safety
            ]
            
            self.airbrake.set_commanded_angle(delayed_angle)
        else:
            self.airbrake.set_commanded_angle(commanded_angle)
        
        # Record truth-based apogee predictions (only after burnout, ascending)
        if self._burnout_recorded and self.state.velocity > 0:
            self._record_truth_predictions()
    
    def _simulate_coast_truth(self, airbrake_cd_add: float) -> float:
        """
        Simulate coast to apogee using the simulation model (truth-based).
        
        Args:
            airbrake_cd_add: Additional Cd from airbrake deployment
            
        Returns:
            Predicted apogee altitude in meters
        """
        h = self.state.altitude
        v = self.state.velocity
        
        if v <= 0:
            return h
        
        dt = 0.2  # Large step for speed
        A_body = self.rocket.get_reference_area()
        # Use dry mass after burnout (motor propellant consumed)
        mass = self.sim_model.dry_mass
        
        # Pre-compute A/m ratio
        A_over_m = A_body / mass
        
        # Update Cd every N steps for efficiency (matches control system)
        cd_update_interval = 5
        
        # Initial Cd computation
        speed_of_sound = self.sim_model.get_speed_of_sound(h)
        mach = v / speed_of_sound if speed_of_sound > 0 else v / 340.0
        body_cd = self.sim_model.get_cd(mach)
        # Use actual airbrake Cd from sim model (CSV if provided)
        # Scale by deployment fraction (airbrake_cd_add is normalized 0-1 range of max Cd_add)
        from config import AIRBRAKE_CD, AIRBRAKE_MAX_AREA_M2, ROCKET_REFERENCE_AREA_M2
        max_cd_add = AIRBRAKE_CD * (AIRBRAKE_MAX_AREA_M2 / ROCKET_REFERENCE_AREA_M2)
        deploy_fraction = airbrake_cd_add / max_cd_add if max_cd_add > 0 else 0
        airbrake_cd = deploy_fraction * self.sim_model.get_airbrake_cd(mach) * (AIRBRAKE_MAX_AREA_M2 / ROCKET_REFERENCE_AREA_M2)
        total_cd = body_cd + airbrake_cd
        
        for i in range(150):  # Max ~30 seconds
            # Update Cd periodically as Mach changes
            if i % cd_update_interval == 0:
                speed_of_sound = self.sim_model.get_speed_of_sound(h)
                mach = v / speed_of_sound if speed_of_sound > 0 else v / 340.0
                body_cd = self.sim_model.get_cd(mach)
                airbrake_cd = deploy_fraction * self.sim_model.get_airbrake_cd(mach) * (AIRBRAKE_MAX_AREA_M2 / ROCKET_REFERENCE_AREA_M2)
                total_cd = body_cd + airbrake_cd
            
            # Use simulation model for density
            rho = self.sim_model.get_density(h)
            
            # Euler integration
            drag_accel = -0.5 * rho * total_cd * A_over_m * v * v
            a = drag_accel - GRAVITY
            
            v_new = v + a * dt
            h_new = h + v * dt + 0.5 * a * dt * dt
            
            if v_new <= 0:
                # Interpolate to apogee
                if a != 0:
                    t_zero = min(dt, max(0, -v / a))
                else:
                    t_zero = 0
                return h + v * t_zero + 0.5 * a * t_zero * t_zero
            
            v = v_new
            h = h_new
            
            if h <= 0:
                return 0.0
        
        return h
    
    def _record_truth_predictions(self):
        """Record truth-based apogee predictions at current state."""
        # Cd_add values must be referenced to body area (matches control system convention)
        A_body = self.rocket.get_reference_area()
        
        # Current airbrake Cd contribution (referenced to body area)
        current_cd_add = self.airbrake.cd * (self.airbrake.get_area() / A_body)
        
        # Max airbrake Cd (fully extended, referenced to body area)
        max_cd_add = self.airbrake.cd * (self.airbrake.max_area_m2 / A_body)
        
        # Compute predictions
        apogee_current = self._simulate_coast_truth(current_cd_add)
        apogee_retracted = self._simulate_coast_truth(0.0)
        apogee_extended = self._simulate_coast_truth(max_cd_add)
        
        self.results._truth_pred_time.append(self.state.time)
        self.results._truth_apogee_current.append(apogee_current)
        self.results._truth_apogee_retracted.append(apogee_retracted)
        self.results._truth_apogee_extended.append(apogee_extended)
        
    def _record_state(self):
        """Record current state to results (O(1) list append)."""
        self.results._time.append(self.state.time)
        self.results._altitude.append(self.state.altitude)
        self.results._velocity.append(self.state.velocity)
        self.results._acceleration.append(self.state.acceleration)
        self.results._mach.append(self.state.mach)
        self.results._mass.append(self.state.mass)
        self.results._thrust.append(self.state.thrust)
        self.results._body_drag.append(self.state.body_drag)
        self.results._airbrake_drag.append(self.state.airbrake_drag)
        self.results._airbrake_angle.append(self.state.airbrake_angle)
        self.results._pressure.append(self.state.pressure)
        self.results._temperature.append(self.state.temperature)
        
        if not self._burnout_recorded and self.state.thrust == 0 and self.state.time > 0.1:
            self._burnout_recorded = True
            self.results.burnout_time = self.state.time
            self.results.burnout_altitude = self.state.altitude
            self.results.burnout_velocity = self.state.velocity
            
    def run(self) -> SimulationResults:
        """
        Run simulation from T0 to apogee.
        
        Returns:
            SimulationResults with complete trajectory data
        """
        self.reset()
        
        max_time = 300.0  # Safety limit
        
        while self.state.time < max_time:
            self._step_physics()
            
            self.airbrake.update(self.dt, self.state.mach)
            self.state.airbrake_angle = self.airbrake.current_angle_deg
            
            self._control_accumulator += self.dt
            if self._control_accumulator >= self.control_dt:
                self._step_control()
                self._control_accumulator = 0.0
                
            self._record_state()
            
            # Track if rocket has lifted off (positive velocity achieved)
            if self.state.velocity > 0:
                self._has_lifted_off = True
            
            # Only check for apogee after rocket has lifted off and is descending
            if self._has_lifted_off and self.state.velocity < 0:
                break
                
        self.results.controller_telemetry = self.controller.get_telemetry()
        self.results.finalize()  # Convert lists to arrays
        self.results.compute_statistics()
        
        # Print debug summary
        debugger = get_debugger()
        debugger.print_summary()
        
        return self.results


def create_simulation(
    target_apogee_m: float = None,
    enable_airbrakes: bool = True,
    # Airframe Cd scaling (Mach-dependent)
    cd_scale_mach0: float = 1.0,
    cd_scale_mach2: float = 1.0,
    # Airbrake Cd scaling (Mach-dependent)
    airbrake_cd_scale_mach0: float = 1.0,
    airbrake_cd_scale_mach2: float = 1.0,
    # Motor performance
    thrust_scale: float = 1.0,
    # Launch site conditions (offsets from truth)
    launch_altitude_offset_m: float = 0.0,
    launch_temp_offset_k: float = 0.0,
    # Airbrake parameters
    airbrake_slew_rate_deg_s: float = 180.0,
    airbrake_max_area_m2: float = 0.006,
    # Custom Cd-Mach CSV files (overrides scaling if provided)
    airframe_cd_csv: str = None,
    airbrake_cd_csv: str = None,
    # Control system latency
    control_latency_ms: float = 0.0,
    # Control system LUT resolution
    control_cd_resolution: int = 50,
    control_density_resolution: int = 100,
    control_mass_resolution: int = 20,
    # Sensor noise parameters (offset = bias, std = random noise)
    pressure_noise_offset_pa: float = 0.0,
    pressure_noise_std_pa: float = None,
    temperature_noise_offset_k: float = 0.0,
    temperature_noise_std_k: float = None,
    accel_noise_offset_mss: float = 0.0,
    accel_noise_std_mss: float = None,
    seed: int = None,
) -> SimulationEnvironment:
    """
    Factory function to create a simulation environment.
    
    Architecture:
    - Truth Model: Source of truth for Cd, atmosphere, rocket data
    - Simulation Model: Applies Mach-dependent scaling and launch site conditions
    - Control Model: Sparse LUTs from truth (configurable resolution)
    
    Args:
        target_apogee_m: Target apogee in meters (default from config)
        enable_airbrakes: Whether to enable airbrakes
        
        cd_scale_mach0: Airframe Cd multiplier at Mach 0 (1.0 = truth)
        cd_scale_mach2: Airframe Cd multiplier at Mach 2+ (linear interp between)
        airbrake_cd_scale_mach0: Airbrake Cd multiplier at Mach 0
        airbrake_cd_scale_mach2: Airbrake Cd multiplier at Mach 2+
        thrust_scale: Motor thrust multiplier (1.0 = nominal)
        launch_altitude_offset_m: Altitude offset from truth model (positive = higher)
        launch_temp_offset_k: Temperature offset from ISA in Kelvin
        airbrake_slew_rate_deg_s: Airbrake slew rate in degrees/second
        airbrake_max_area_m2: Maximum airbrake area in m^2
        
        control_cd_resolution: Control system Cd LUT resolution
        control_density_resolution: Control system density LUT resolution
        control_mass_resolution: Control system mass LUT resolution
        
        pressure_noise_offset_pa: Pressure sensor bias/offset (Pa)
        pressure_noise_std_pa: Pressure sensor noise std dev (Pa), None = default
        temperature_noise_offset_k: Temperature sensor bias/offset (K)
        temperature_noise_std_k: Temperature sensor noise std dev (K), None = default
        accel_noise_offset_mss: Accelerometer bias/offset (m/s^2)
        accel_noise_std_mss: Accelerometer noise std dev (m/s^2), None = default
        seed: Random seed for sensor noise
        
    Returns:
        Configured SimulationEnvironment
    """
    from config import TARGET_APOGEE_M
    from sensors import SensorModelVariations
    from truth_model import SimulationConfig
    
    # Get truth model (source of truth)
    truth = get_truth_model()
    
    # Create simulation model with full configuration
    sim_config = SimulationConfig(
        cd_scale_mach0=cd_scale_mach0,
        cd_scale_mach2=cd_scale_mach2,
        airbrake_cd_scale_mach0=airbrake_cd_scale_mach0,
        airbrake_cd_scale_mach2=airbrake_cd_scale_mach2,
        thrust_scale=thrust_scale,
        launch_altitude_offset_m=launch_altitude_offset_m,
        launch_temp_offset_k=launch_temp_offset_k,
        airbrake_slew_rate_deg_s=airbrake_slew_rate_deg_s,
        airbrake_max_area_m2=airbrake_max_area_m2,
        airframe_cd_csv=airframe_cd_csv,
        airbrake_cd_csv=airbrake_cd_csv,
        control_latency_ms=control_latency_ms,
    )
    sim_model = SimulationModel(truth, sim_config)
    
    # Create control model with LUT configuration
    lut_config = ControlLUTConfig(
        cd_resolution=control_cd_resolution,
        density_resolution=control_density_resolution,
        mass_resolution=control_mass_resolution,
    )
    
    # Create controller using control model
    target = target_apogee_m if target_apogee_m else TARGET_APOGEE_M
    controller = AirbrakeController(
        target_apogee_m=target,
        lut_config=lut_config,
    )
    
    # Store configs for dashboard display
    controller.sim_config = sim_config
    controller.lut_config = lut_config
    
    rocket = RocketModel.from_tables()
    
    # Create airbrake with configured parameters
    airbrake = Airbrake(
        slew_rate_deg_s=sim_config.airbrake_slew_rate_deg_s,
        max_area_m2=sim_config.airbrake_max_area_m2,
    )
    
    sensors = SensorModelVariations.custom(
        pressure_noise_std=pressure_noise_std_pa,
        temperature_noise_std=temperature_noise_std_k,
        accel_noise_std=accel_noise_std_mss,
        pressure_offset=pressure_noise_offset_pa,
        temperature_offset=temperature_noise_offset_k,
        accel_offset=accel_noise_offset_mss,
        seed=seed,
    )
    
    sim = SimulationEnvironment(
        sim_model=sim_model,
        rocket=rocket,
        airbrake=airbrake,
        controller=controller,
        sensors=sensors,
        enable_airbrakes=enable_airbrakes,
    )
    
    # Store configs on sim for dashboard
    sim.sim_config = sim_config
    sim.lut_config = lut_config
    
    return sim
