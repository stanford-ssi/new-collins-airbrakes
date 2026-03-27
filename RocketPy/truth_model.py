"""
Truth Model - Source of truth for rocket and atmospheric properties.

This module contains the "best guess" models of real dynamics that both
the simulation environment and control system derive from.

- Simulation: Can scale truth model data by a percentage
- Control: Builds sparse lookup tables from truth model
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional
import os

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


TABLES_DIR = os.path.join(os.path.dirname(__file__), 'tables')
THRUST_FILE = os.path.join(TABLES_DIR, 'V2 Thrust.csv')
MASS_FILE = os.path.join(TABLES_DIR, 'V2 Mass Change.csv')
CD_MACH_FILE = os.path.join(TABLES_DIR, 'V2 CD Mach Number.csv')


@dataclass
class SimulationConfig:
    """Configuration for simulation environment model.
    
    Cd and airbrake Cd scaling are Mach-dependent:
    - At Mach 0: use *_scale_mach0
    - At Mach 2+: use *_scale_mach2
    - Between: linear interpolation
    
    Alternatively, provide CSV files for custom Cd-Mach curves.
    """
    # Airframe Cd scaling (Mach-dependent)
    cd_scale_mach0: float = 1.0  # Cd multiplier at Mach 0
    cd_scale_mach2: float = 1.0  # Cd multiplier at Mach 2+
    
    # Airbrake Cd scaling (Mach-dependent)
    airbrake_cd_scale_mach0: float = 1.0  # Airbrake Cd multiplier at Mach 0
    airbrake_cd_scale_mach2: float = 1.0  # Airbrake Cd multiplier at Mach 2+
    
    # Motor performance
    thrust_scale: float = 1.0  # Motor thrust multiplier (1.0 = nominal)
    
    # Launch site conditions (offsets from truth model)
    launch_altitude_offset_m: float = 0.0  # Altitude offset from truth (positive = higher)
    launch_temp_offset_k: float = 0.0  # Temperature offset from ISA (positive = hotter)
    
    # Airbrake parameters
    airbrake_slew_rate_deg_s: float = 180.0  # Airbrake slew rate in deg/s
    airbrake_max_area_m2: float = 0.006  # Max airbrake area in m^2
    
    # Custom Cd-Mach CSV files (overrides scaling if provided)
    airframe_cd_csv: str = None  # Path to airframe Cd-Mach CSV (columns: Mach, Cd)
    airbrake_cd_csv: str = None  # Path to airbrake Cd-Mach CSV (columns: Mach, Cd)
    
    # Control system latency
    control_latency_ms: float = 0.0  # Delay between sensor reading and control output (ms)
    
    def get_cd_scale(self, mach: float) -> float:
        """Get Cd scale factor at given Mach (linear interpolation)."""
        if mach <= 0:
            return self.cd_scale_mach0
        elif mach >= 2.0:
            return self.cd_scale_mach2
        else:
            t = mach / 2.0
            return self.cd_scale_mach0 + t * (self.cd_scale_mach2 - self.cd_scale_mach0)
    
    def get_airbrake_cd_scale(self, mach: float) -> float:
        """Get airbrake Cd scale factor at given Mach (linear interpolation)."""
        if mach <= 0:
            return self.airbrake_cd_scale_mach0
        elif mach >= 2.0:
            return self.airbrake_cd_scale_mach2
        else:
            t = mach / 2.0
            return self.airbrake_cd_scale_mach0 + t * (self.airbrake_cd_scale_mach2 - self.airbrake_cd_scale_mach0)
    
    def __repr__(self):
        return (f"SimulationConfig(\n"
                f"  cd_scale: M0={self.cd_scale_mach0:.2f}, M2={self.cd_scale_mach2:.2f}\n"
                f"  airbrake_cd_scale: M0={self.airbrake_cd_scale_mach0:.2f}, M2={self.airbrake_cd_scale_mach2:.2f}\n"
                f"  thrust_scale={self.thrust_scale:.2f}\n"
                f"  launch: alt_offset={self.launch_altitude_offset_m:+.0f}m, temp_offset={self.launch_temp_offset_k:+.1f}K\n"
                f"  airbrake: slew={self.airbrake_slew_rate_deg_s:.0f} deg/s, area={self.airbrake_max_area_m2*10000:.1f} cm²)")


# Backwards compatibility alias
TruthModelConfig = SimulationConfig


@dataclass 
class ControlLUTConfig:
    """Configuration for control system lookup tables."""
    cd_resolution: int = 50  # Points in Cd vs Mach table
    density_resolution: int = 100  # Points in density vs altitude table
    mass_resolution: int = 20  # Points in mass vs time table
    
    def __repr__(self):
        return (f"ControlLUTConfig(cd={self.cd_resolution}, "
                f"density={self.density_resolution}, mass={self.mass_resolution})")


class TruthModel:
    """
    Source of truth for rocket and atmospheric properties.
    
    This class loads and stores the "best guess" physical models.
    Both simulation and control systems derive their data from here.
    """
    
    def __init__(self):
        """Initialize truth model by loading base data."""
        self._load_rocket_data()
        self._compute_derived_values()
        
    def _load_rocket_data(self):
        """Load rocket data from CSV files."""
        # Thrust curve
        self.thrust_times, self.thrust_values = self._load_csv_pair(
            THRUST_FILE, col_order=(0, 1)
        )
        
        # Mass curve (convert g to kg)
        times, masses_g = self._load_csv_pair(MASS_FILE, col_order=(0, 1))
        self.mass_times = times
        self.mass_values = masses_g / 1000.0
        
        # Cd vs Mach (note: CSV is Cd,Mach order)
        # Filter out unrealistic Cd values (max_cd=1.0)
        cds, machs = self._load_csv_pair(CD_MACH_FILE, col_order=(0, 1), max_cd=1.0)
        # Sort by Mach
        sort_idx = np.argsort(machs)
        self.cd_machs = machs[sort_idx]
        self.cd_values = cds[sort_idx]
        
    def _load_csv_pair(self, filepath: str, col_order: Tuple[int, int],
                       max_val: float = None, max_cd: float = None) -> Tuple[np.ndarray, np.ndarray]:
        """Load two columns from CSV file.
        
        Args:
            filepath: Path to CSV file
            col_order: Tuple of column indices to read
            max_val: Filter out rows where either value exceeds this
            max_cd: Filter out rows where first column (Cd) exceeds this
        """
        col1, col2 = [], []
        
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split(',')
                if len(parts) >= 2:
                    try:
                        v1 = float(parts[col_order[0]])
                        v2 = float(parts[col_order[1]])
                        # Filter by max_val (both columns)
                        if max_val is not None and (v1 > max_val or v2 > max_val):
                            continue
                        # Filter by max_cd (first column only, for Cd tables)
                        if max_cd is not None and v1 > max_cd:
                            continue
                        col1.append(v1)
                        col2.append(v2)
                    except ValueError:
                        continue
        
        arr1 = np.array(col1)
        arr2 = np.array(col2)
        
        # Sort by first column and remove duplicates
        sort_idx = np.argsort(arr1)
        arr1 = arr1[sort_idx]
        arr2 = arr2[sort_idx]
        _, unique_idx = np.unique(arr1, return_index=True)
        
        return arr1[unique_idx], arr2[unique_idx]
    
    def _compute_derived_values(self):
        """Compute derived values from loaded data."""
        self.initial_mass = self.mass_values[0]
        self.dry_mass = self.mass_values[-1]
        
        # Find burnout time
        thrust_nonzero = np.where(self.thrust_values > 10)[0]
        if len(thrust_nonzero) > 0:
            self.burnout_time = self.thrust_times[thrust_nonzero[-1]]
        else:
            self.burnout_time = 0.0
            
    # --- Truth Model Query Methods ---
    
    def get_cd(self, mach: float) -> float:
        """Get drag coefficient at Mach number (truth value)."""
        return np.interp(mach, self.cd_machs, self.cd_values)
    
    def get_thrust(self, time: float) -> float:
        """Get thrust at time (truth value)."""
        if time < self.thrust_times[0] or time > self.thrust_times[-1]:
            return 0.0
        return np.interp(time, self.thrust_times, self.thrust_values)
    
    def get_mass(self, time: float) -> float:
        """Get mass at time (truth value)."""
        if time <= self.mass_times[0]:
            return self.mass_values[0]
        if time >= self.mass_times[-1]:
            return self.mass_values[-1]
        return np.interp(time, self.mass_times, self.mass_values)
    
    def get_density(self, altitude_m: float) -> float:
        """Get air density at altitude using ISA model (truth value)."""
        alt_msl = altitude_m + LAUNCH_ALTITUDE_M
        T = SEA_LEVEL_TEMPERATURE - TEMPERATURE_LAPSE_RATE * alt_msl
        exponent = GRAVITY / (GAS_CONSTANT * TEMPERATURE_LAPSE_RATE)
        P = SEA_LEVEL_PRESSURE * (T / SEA_LEVEL_TEMPERATURE) ** exponent
        return P / (GAS_CONSTANT * T)
    
    def get_temperature(self, altitude_m: float) -> float:
        """Get temperature at altitude using ISA model (truth value)."""
        alt_msl = altitude_m + LAUNCH_ALTITUDE_M
        return SEA_LEVEL_TEMPERATURE - TEMPERATURE_LAPSE_RATE * alt_msl
    
    def get_pressure(self, altitude_m: float) -> float:
        """Get pressure at altitude using ISA model (truth value)."""
        alt_msl = altitude_m + LAUNCH_ALTITUDE_M
        T = SEA_LEVEL_TEMPERATURE - TEMPERATURE_LAPSE_RATE * alt_msl
        exponent = GRAVITY / (GAS_CONSTANT * TEMPERATURE_LAPSE_RATE)
        return SEA_LEVEL_PRESSURE * (T / SEA_LEVEL_TEMPERATURE) ** exponent
    
    def get_speed_of_sound(self, altitude_m: float) -> float:
        """Get speed of sound at altitude (truth value)."""
        T = self.get_temperature(altitude_m)
        return np.sqrt(GAMMA * GAS_CONSTANT * T)


class SimulationModel:
    """
    Simulation's view of the physical model.
    
    Applies configurable scaling to truth model data to simulate
    real-world uncertainties and model errors.
    """
    
    def __init__(self, truth: TruthModel, config: SimulationConfig = None):
        """
        Initialize simulation model.
        
        Args:
            truth: Truth model to derive from
            config: Simulation configuration
        """
        self.truth = truth
        self.config = config or SimulationConfig()
        
        # Cache launch site conditions (offset from truth)
        self._launch_alt = LAUNCH_ALTITUDE_M + self.config.launch_altitude_offset_m
        self._temp_offset = self.config.launch_temp_offset_k
        
        # Load custom Cd-Mach curves if provided
        self._custom_airframe_cd = None
        self._custom_airbrake_cd = None
        
        if self.config.airframe_cd_csv:
            self._custom_airframe_cd = self._load_cd_csv(self.config.airframe_cd_csv)
            
        if self.config.airbrake_cd_csv:
            self._custom_airbrake_cd = self._load_cd_csv(self.config.airbrake_cd_csv)
    
    def _load_cd_csv(self, filepath: str) -> tuple:
        """Load Cd-Mach data from CSV file.
        
        Expected format: Mach,Cd (with optional header)
        
        Returns:
            Tuple of (mach_array, cd_array) sorted by Mach
        """
        machs, cds = [], []
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split(',')
                if len(parts) >= 2:
                    try:
                        mach = float(parts[0])
                        cd = float(parts[1])
                        machs.append(mach)
                        cds.append(cd)
                    except ValueError:
                        continue  # Skip header or invalid lines
        
        machs = np.array(machs)
        cds = np.array(cds)
        sort_idx = np.argsort(machs)
        return machs[sort_idx], cds[sort_idx]
        
    def get_cd(self, mach: float) -> float:
        """Get Cd at Mach number.
        
        Uses custom CSV if provided, otherwise scales truth model.
        """
        if self._custom_airframe_cd is not None:
            machs, cds = self._custom_airframe_cd
            return np.interp(mach, machs, cds)
        return self.truth.get_cd(mach) * self.config.get_cd_scale(mach)
    
    def get_airbrake_cd(self, mach: float) -> float:
        """Get airbrake Cd at Mach number.
        
        Uses custom CSV if provided, otherwise returns baseline (truth uses 1.0).
        """
        if self._custom_airbrake_cd is not None:
            machs, cds = self._custom_airbrake_cd
            return np.interp(mach, machs, cds)
        # Default: no custom airbrake Cd curve, return a nominal value
        # This is just the scale factor (actual Cd_add is computed elsewhere)
        return 1.0 * self.config.get_airbrake_cd_scale(mach)
    
    def get_airbrake_cd_scale(self, mach: float) -> float:
        """Get airbrake Cd scale at Mach number."""
        return self.config.get_airbrake_cd_scale(mach)
    
    def get_thrust(self, time: float) -> float:
        """Get scaled thrust at time."""
        return self.truth.get_thrust(time) * self.config.thrust_scale
    
    def get_mass(self, time: float) -> float:
        """Get mass at time (not scaled)."""
        return self.truth.get_mass(time)
    
    def get_density(self, altitude_agl: float) -> float:
        """Get density at altitude AGL using configured launch site."""
        T = self.get_temperature(altitude_agl)
        P = self.get_pressure(altitude_agl)
        return P / (GAS_CONSTANT * T)
    
    def get_temperature(self, altitude_agl: float) -> float:
        """Get temperature at altitude AGL with launch site offset."""
        alt_msl = altitude_agl + self._launch_alt
        T_isa = SEA_LEVEL_TEMPERATURE - TEMPERATURE_LAPSE_RATE * alt_msl
        return T_isa + self._temp_offset
    
    def get_pressure(self, altitude_agl: float) -> float:
        """Get pressure at altitude AGL using configured launch site."""
        alt_msl = altitude_agl + self._launch_alt
        # Use ISA temperature (not offset) for pressure calculation
        T_isa = SEA_LEVEL_TEMPERATURE - TEMPERATURE_LAPSE_RATE * alt_msl
        exponent = GRAVITY / (GAS_CONSTANT * TEMPERATURE_LAPSE_RATE)
        return SEA_LEVEL_PRESSURE * (T_isa / SEA_LEVEL_TEMPERATURE) ** exponent
    
    def get_speed_of_sound(self, altitude_agl: float) -> float:
        """Get speed of sound at altitude."""
        T = self.get_temperature(altitude_agl)
        return np.sqrt(GAMMA * GAS_CONSTANT * T)
    
    @property
    def launch_altitude_m(self) -> float:
        return self._launch_alt
    
    @property
    def dry_mass(self) -> float:
        return self.truth.dry_mass
    
    @property
    def initial_mass(self) -> float:
        return self.truth.initial_mass
    
    @property
    def burnout_time(self) -> float:
        return self.truth.burnout_time


class ControlModel:
    """
    Control system's view of the physical model.
    
    Builds sparse lookup tables from truth model for efficient
    real-time computation. Resolution is configurable.
    """
    
    def __init__(self, truth: TruthModel, config: ControlLUTConfig = None):
        """
        Initialize control model with lookup tables.
        
        Args:
            truth: Truth model to derive from
            config: LUT configuration
        """
        self.truth = truth
        self.config = config or ControlLUTConfig()
        
        self._build_cd_lut()
        self._build_density_lut()
        
    def _build_cd_lut(self):
        """Build Cd vs Mach lookup table."""
        mach_min = self.truth.cd_machs.min()
        mach_max = self.truth.cd_machs.max()
        
        self.cd_machs = np.linspace(mach_min, mach_max, self.config.cd_resolution)
        self.cd_values = np.array([self.truth.get_cd(m) for m in self.cd_machs])
        
    def _build_density_lut(self):
        """Build density vs altitude lookup table."""
        self.density_altitudes = np.linspace(0, 12000, self.config.density_resolution)
        self.density_values = np.array([
            self.truth.get_density(alt) for alt in self.density_altitudes
        ])
        self.temperature_values = np.array([
            self.truth.get_temperature(alt) for alt in self.density_altitudes
        ])
        self.speed_of_sound_values = np.array([
            self.truth.get_speed_of_sound(alt) for alt in self.density_altitudes
        ])
        
    def get_cd(self, mach: float) -> float:
        """Get Cd at Mach from lookup table."""
        return np.interp(mach, self.cd_machs, self.cd_values)
    
    def get_density(self, altitude_m: float) -> float:
        """Get density at altitude from lookup table."""
        alt_clamped = np.clip(altitude_m, 0, self.density_altitudes[-1])
        return np.interp(alt_clamped, self.density_altitudes, self.density_values)
    
    def get_temperature(self, altitude_m: float) -> float:
        """Get temperature at altitude from lookup table."""
        alt_clamped = np.clip(altitude_m, 0, self.density_altitudes[-1])
        return np.interp(alt_clamped, self.density_altitudes, self.temperature_values)
    
    def get_speed_of_sound(self, altitude_m: float) -> float:
        """Get speed of sound at altitude from lookup table."""
        alt_clamped = np.clip(altitude_m, 0, self.density_altitudes[-1])
        return np.interp(alt_clamped, self.density_altitudes, self.speed_of_sound_values)
    
    def get_mass(self, time: float) -> float:
        """Get mass at time (from truth, not LUT)."""
        return self.truth.get_mass(time)
    
    @property
    def dry_mass(self) -> float:
        return self.truth.dry_mass
    
    @property
    def burnout_time(self) -> float:
        return self.truth.burnout_time


# Global truth model instance
_truth_model = None


def get_truth_model() -> TruthModel:
    """Get or create the global truth model instance."""
    global _truth_model
    if _truth_model is None:
        _truth_model = TruthModel()
    return _truth_model


def reset_truth_model():
    """Reset the global truth model (for testing)."""
    global _truth_model
    _truth_model = None
