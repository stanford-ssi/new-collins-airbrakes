"""
Debug and diagnostic framework for simulation troubleshooting.

Provides:
- Configurable logging levels
- Anomaly detection for sudden state changes
- State transition tracking
- Summary reports at end of simulation
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Callable
from enum import Enum


class DebugLevel(Enum):
    OFF = 0
    SUMMARY = 1      # Only print summary at end
    ANOMALIES = 2    # Print when anomalies detected
    TRANSITIONS = 3  # Print state transitions
    VERBOSE = 4      # Print every step


@dataclass
class Anomaly:
    """Record of a detected anomaly."""
    time: float
    category: str
    message: str
    values: dict
    
    def __str__(self):
        vals = ", ".join(f"{k}={v:.4g}" for k, v in self.values.items())
        return f"[t={self.time:.3f}s] {self.category}: {self.message} ({vals})"


@dataclass
class StateTransition:
    """Record of a state transition."""
    time: float
    from_state: str
    to_state: str
    reason: str
    
    def __str__(self):
        return f"[t={self.time:.3f}s] {self.from_state} -> {self.to_state}: {self.reason}"


@dataclass
class DebugConfig:
    """Configuration for debug output."""
    level: DebugLevel = DebugLevel.ANOMALIES
    
    # Anomaly detection thresholds
    velocity_change_threshold: float = 50.0      # m/s per control step
    acceleration_change_threshold: float = 100.0  # m/s² per control step
    cd_change_threshold: float = 0.5             # Cd units per control step
    apogee_change_threshold: float = 500.0       # meters per control step
    
    # Print settings
    print_every_n_steps: int = 50  # For verbose mode
    

class SimulationDebugger:
    """
    Debug helper that tracks simulation state and detects anomalies.
    
    Usage:
        debugger = SimulationDebugger(DebugConfig(level=DebugLevel.ANOMALIES))
        
        # In simulation loop:
        debugger.record_state(time, velocity, altitude, ...)
        debugger.check_anomalies()
        
        # At end:
        debugger.print_summary()
    """
    
    def __init__(self, config: DebugConfig = None):
        self.config = config or DebugConfig()
        self.anomalies: List[Anomaly] = []
        self.transitions: List[StateTransition] = []
        
        # State history (last N values for change detection)
        self._history_size = 10
        self._time_history: List[float] = []
        self._velocity_history: List[float] = []
        self._altitude_history: List[float] = []
        self._acceleration_history: List[float] = []
        self._mach_history: List[float] = []
        self._cd_total_history: List[float] = []
        self._apogee_pred_history: List[float] = []
        self._airbrake_angle_history: List[float] = []
        
        # Control state tracking
        self._last_control_state: str = "INIT"
        self._step_count = 0
        
        # Statistics
        self._max_velocity = 0.0
        self._max_altitude = 0.0
        self._max_mach = 0.0
        self._min_apogee_pred = float('inf')
        self._max_apogee_pred = 0.0
        
    def reset(self):
        """Reset debugger state."""
        self.anomalies.clear()
        self.transitions.clear()
        self._time_history.clear()
        self._velocity_history.clear()
        self._altitude_history.clear()
        self._acceleration_history.clear()
        self._mach_history.clear()
        self._cd_total_history.clear()
        self._apogee_pred_history.clear()
        self._airbrake_angle_history.clear()
        self._last_control_state = "INIT"
        self._step_count = 0
        self._max_velocity = 0.0
        self._max_altitude = 0.0
        self._max_mach = 0.0
        self._min_apogee_pred = float('inf')
        self._max_apogee_pred = 0.0
        
    def record_control_state(
        self,
        time: float,
        velocity_est: float,
        altitude_est: float,
        acceleration: float,
        mach: float,
        cd_total: float,
        apogee_pred: float,
        airbrake_angle: float,
        control_state: str,
    ):
        """
        Record control system state for anomaly detection.
        
        Args:
            time: Current time (s)
            velocity_est: EKF velocity estimate (m/s)
            altitude_est: EKF altitude estimate (m)
            acceleration: Measured/estimated acceleration (m/s²)
            mach: Estimated Mach number
            cd_total: Total Cd being used
            apogee_pred: Predicted apogee (m)
            airbrake_angle: Commanded airbrake angle (deg)
            control_state: Current control state name
        """
        self._step_count += 1
        
        # Update statistics
        self._max_velocity = max(self._max_velocity, velocity_est)
        self._max_altitude = max(self._max_altitude, altitude_est)
        self._max_mach = max(self._max_mach, mach)
        if apogee_pred > 0:
            self._min_apogee_pred = min(self._min_apogee_pred, apogee_pred)
            self._max_apogee_pred = max(self._max_apogee_pred, apogee_pred)
        
        # Check for state transitions
        if control_state != self._last_control_state:
            self._record_transition(time, self._last_control_state, control_state)
            self._last_control_state = control_state
        
        # Check for anomalies (need at least 2 data points)
        if len(self._velocity_history) > 0:
            self._check_anomalies(
                time, velocity_est, altitude_est, acceleration,
                mach, cd_total, apogee_pred, airbrake_angle
            )
        
        # Update history
        self._add_to_history(self._time_history, time)
        self._add_to_history(self._velocity_history, velocity_est)
        self._add_to_history(self._altitude_history, altitude_est)
        self._add_to_history(self._acceleration_history, acceleration)
        self._add_to_history(self._mach_history, mach)
        self._add_to_history(self._cd_total_history, cd_total)
        self._add_to_history(self._apogee_pred_history, apogee_pred)
        self._add_to_history(self._airbrake_angle_history, airbrake_angle)
        
        # Verbose output
        if self.config.level == DebugLevel.VERBOSE:
            if self._step_count % self.config.print_every_n_steps == 0:
                self._print_state(time, velocity_est, altitude_est, mach, 
                                 apogee_pred, airbrake_angle, control_state)
    
    def record_simulation_state(
        self,
        time: float,
        velocity_true: float,
        altitude_true: float,
        acceleration_true: float,
        mach_true: float,
        airbrake_drag: float,
        body_drag: float,
    ):
        """
        Record simulation (truth) state for comparison.
        
        This can be used to detect divergence between control estimates and reality.
        """
        # For now, just track for summary
        pass
        
    def _add_to_history(self, history: list, value: float):
        """Add value to history, maintaining max size."""
        history.append(value)
        if len(history) > self._history_size:
            history.pop(0)
            
    def _check_anomalies(
        self,
        time: float,
        velocity: float,
        altitude: float,
        acceleration: float,
        mach: float,
        cd_total: float,
        apogee_pred: float,
        airbrake_angle: float,
    ):
        """Check for anomalous state changes."""
        prev_vel = self._velocity_history[-1]
        prev_acc = self._acceleration_history[-1]
        prev_cd = self._cd_total_history[-1]
        prev_apogee = self._apogee_pred_history[-1]
        prev_time = self._time_history[-1]
        
        dt = time - prev_time
        if dt <= 0:
            return
            
        # Velocity change anomaly
        vel_change = abs(velocity - prev_vel)
        if vel_change > self.config.velocity_change_threshold:
            self._add_anomaly(
                time, "VELOCITY_JUMP",
                f"Velocity changed by {vel_change:.1f} m/s in {dt:.3f}s",
                {"prev_vel": prev_vel, "curr_vel": velocity, "dt": dt,
                 "acceleration": acceleration}
            )
            
        # Acceleration change anomaly  
        acc_change = abs(acceleration - prev_acc)
        if acc_change > self.config.acceleration_change_threshold:
            self._add_anomaly(
                time, "ACCELERATION_JUMP",
                f"Acceleration changed by {acc_change:.1f} m/s² in {dt:.3f}s",
                {"prev_acc": prev_acc, "curr_acc": acceleration, "dt": dt,
                 "mach": mach, "cd_total": cd_total}
            )
            
        # Cd change anomaly
        cd_change = abs(cd_total - prev_cd)
        if cd_change > self.config.cd_change_threshold:
            self._add_anomaly(
                time, "CD_JUMP",
                f"Cd changed by {cd_change:.3f} in {dt:.3f}s",
                {"prev_cd": prev_cd, "curr_cd": cd_total, "mach": mach}
            )
            
        # Apogee prediction change anomaly
        if prev_apogee > 0 and apogee_pred > 0:
            apogee_change = abs(apogee_pred - prev_apogee)
            if apogee_change > self.config.apogee_change_threshold:
                self._add_anomaly(
                    time, "APOGEE_PRED_JUMP",
                    f"Apogee prediction changed by {apogee_change:.0f}m in {dt:.3f}s",
                    {"prev_apogee": prev_apogee, "curr_apogee": apogee_pred,
                     "velocity": velocity, "altitude": altitude}
                )
                
        # Velocity going negative while still high altitude
        if velocity <= 0 and prev_vel > 10 and altitude > 1000:
            self._add_anomaly(
                time, "UNEXPECTED_APOGEE",
                f"Velocity went to {velocity:.1f} from {prev_vel:.1f} at altitude {altitude:.0f}m",
                {"prev_vel": prev_vel, "curr_vel": velocity, "altitude": altitude,
                 "acceleration": acceleration}
            )
    
    def _add_anomaly(self, time: float, category: str, message: str, values: dict):
        """Add anomaly and optionally print it."""
        anomaly = Anomaly(time, category, message, values)
        self.anomalies.append(anomaly)
        
        if self.config.level.value >= DebugLevel.ANOMALIES.value:
            print(f"⚠️  ANOMALY: {anomaly}")
            
    def _record_transition(self, time: float, from_state: str, to_state: str):
        """Record a state transition."""
        # Determine reason based on state names
        reason = f"{from_state} -> {to_state}"
        
        transition = StateTransition(time, from_state, to_state, reason)
        self.transitions.append(transition)
        
        if self.config.level.value >= DebugLevel.TRANSITIONS.value:
            print(f"🔄 TRANSITION: {transition}")
            
    def _print_state(
        self,
        time: float,
        velocity: float,
        altitude: float,
        mach: float,
        apogee_pred: float,
        airbrake_angle: float,
        control_state: str,
    ):
        """Print current state (verbose mode)."""
        print(f"[t={time:.2f}s] v={velocity:.1f}m/s alt={altitude:.0f}m "
              f"M={mach:.2f} apogee={apogee_pred:.0f}m "
              f"brake={airbrake_angle:.1f}° state={control_state}")
              
    def print_summary(self):
        """Print summary of simulation debug info."""
        if self.config.level == DebugLevel.OFF:
            return
            
        print("\n" + "="*60)
        print("SIMULATION DEBUG SUMMARY")
        print("="*60)
        
        print(f"\n📊 Statistics:")
        print(f"   Max velocity: {self._max_velocity:.1f} m/s")
        print(f"   Max altitude: {self._max_altitude:.0f} m")
        print(f"   Max Mach: {self._max_mach:.2f}")
        if self._max_apogee_pred > 0:
            print(f"   Apogee predictions: {self._min_apogee_pred:.0f} - {self._max_apogee_pred:.0f} m")
        
        print(f"\n🔄 State Transitions ({len(self.transitions)}):")
        for t in self.transitions[:20]:  # Limit output
            print(f"   {t}")
        if len(self.transitions) > 20:
            print(f"   ... and {len(self.transitions) - 20} more")
            
        print(f"\n⚠️  Anomalies ({len(self.anomalies)}):")
        if self.anomalies:
            # Group by category
            by_category = {}
            for a in self.anomalies:
                by_category.setdefault(a.category, []).append(a)
                
            for cat, anomalies in by_category.items():
                print(f"\n   {cat} ({len(anomalies)} occurrences):")
                for a in anomalies[:5]:  # Show first 5 of each type
                    print(f"      {a}")
                if len(anomalies) > 5:
                    print(f"      ... and {len(anomalies) - 5} more")
        else:
            print("   None detected")
            
        print("\n" + "="*60)
        
    def get_anomaly_summary(self) -> str:
        """Get a short summary string of anomalies."""
        if not self.anomalies:
            return "No anomalies detected"
        
        by_category = {}
        for a in self.anomalies:
            by_category.setdefault(a.category, []).append(a)
            
        parts = [f"{cat}: {len(lst)}" for cat, lst in by_category.items()]
        return ", ".join(parts)


# Global debugger instance (can be configured at startup)
_debugger: Optional[SimulationDebugger] = None


def get_debugger() -> SimulationDebugger:
    """Get or create the global debugger instance."""
    global _debugger
    if _debugger is None:
        _debugger = SimulationDebugger()
    return _debugger


def set_debugger(debugger: SimulationDebugger):
    """Set the global debugger instance."""
    global _debugger
    _debugger = debugger


def configure_debug(level: DebugLevel = DebugLevel.ANOMALIES, **kwargs):
    """Configure debugging with specified level and options."""
    config = DebugConfig(level=level, **kwargs)
    set_debugger(SimulationDebugger(config))
    return get_debugger()
