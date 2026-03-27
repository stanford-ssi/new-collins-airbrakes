# Airbrake Simulation System

A comprehensive simulation system for high-power rocket airbrakes, targeting 30,000 ft apogee. The system includes a simulation environment, separate control system, EKF-based state estimation, and Monte Carlo analysis capabilities.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Data Tables](#data-tables)
- [System Architecture](#system-architecture)
- [Configuration](#configuration)
- [Running Simulations](#running-simulations)
- [Monte Carlo Analysis](#monte-carlo-analysis)
- [Visualizations](#visualizations)
- [Control System](#control-system)
- [API Reference](#api-reference)

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Run a basic simulation
python main.py
```

### Requirements
- Python 3.8+
- NumPy >= 1.21.0
- SciPy >= 1.7.0
- Matplotlib >= 3.4.0

## Quick Start

```python
from simulation import create_simulation
from visualization import plot_trajectory, create_summary_report

# Create and run simulation (loads data from tables/ directory)
sim = create_simulation(enable_airbrakes=True)
results = sim.run()

# Display results
print(f"Apogee: {results.apogee_ft:.0f} ft")
plot_trajectory(results)
create_summary_report(results)
```

## Data Tables

The simulation loads rocket data from CSV files in the `tables/` directory:

| File | Format | Description |
|------|--------|-------------|
| `Motor Thrust.csv` | `Time (s), Thrust (N)` | Motor thrust curve |
| `Mass Change.csv` | `Time (s), Mass (g)` | Time-varying mass (in grams) |
| `CD Mach Number.csv` | `Cd, Mach` | Drag coefficient vs Mach number |

### Updating Tables

To use different rocket data, simply replace the CSV files in the `tables/` directory. The simulation will automatically load and linearly interpolate between data points.

**Important notes:**
- Mass values in `Mass Change.csv` are in **grams** (converted to kg internally)
- `CD Mach Number.csv` has Cd in the **first column**, Mach in the **second**
- Invalid Cd values (>1.0) are automatically filtered out

### Rocket Parameters

The rocket reference diameter is set in `config.py`:
- **Diameter**: 156.72 mm
- **Reference Area**: π × (78.36mm)² = 19,290 mm² (0.01929 m²)

## System Architecture

The system is split into two main components:

### 1. Simulation Environment (`simulation.py`)
- Physics simulation from T0 to apogee
- True atmospheric model calculations
- Force integration (thrust, drag, gravity)
- Sensor noise injection

### 2. Control System (`control_system.py`)
- **Completely separate from simulation**
- Receives noisy sensor data through helper functions
- Uses lookup tables (not direct calculations)
- EKF for state estimation
- Predictive control algorithm

```
┌─────────────────────────────────────────────────────────────┐
│                    SIMULATION ENVIRONMENT                    │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │ Rocket Model│  │ Atmospheric  │  │ Airbrake Physics  │  │
│  │ (thrust,    │  │ Model        │  │ (slew rate,       │  │
│  │  mass, drag)│  │ (true values)│  │  area, drag)      │  │
│  └─────────────┘  └──────────────┘  └───────────────────┘  │
│                           │                                  │
│                    ┌──────▼──────┐                          │
│                    │   SENSORS   │ ← Gaussian noise added   │
│                    │ (P, T, accel)│                         │
│                    └──────┬──────┘                          │
└───────────────────────────┼─────────────────────────────────┘
                            │ Noisy measurements
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      CONTROL SYSTEM                          │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │ Lookup      │  │     EKF      │  │ Predictive        │  │
│  │ Tables      │  │ (altitude,   │  │ Controller        │  │
│  │ (atm model) │  │  velocity)   │  │ (apogee targeting)│  │
│  └─────────────┘  └──────────────┘  └───────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Configuration

### Simulation Parameters

The `create_simulation()` function accepts these optional parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `target_apogee_m` | Target apogee in meters | 9144 (30,000 ft) |
| `enable_airbrakes` | Enable airbrake control | `True` |
| `sensor_noise_mult` | Sensor noise multiplier | 1.0 |
| `control_cd_resolution` | Cd table points for control system | 50 |
| `cd_error_body` | (low, high) Cd error at Mach 0 and 1.5 | (0, 0) |
| `cd_error_airbrake` | (low, high) airbrake Cd error | (0, 0) |

### Control System Cd Table Coarseness

The control system uses a separate, coarser Cd lookup table for efficiency. You can configure the resolution:

```python
# Use full resolution (all ~800 data points)
sim = create_simulation(control_cd_resolution=None)

# Use coarser table (faster, less accurate)
sim = create_simulation(control_cd_resolution=20)

# Default: 50 points
sim = create_simulation()
```

From command line:
```bash
python main.py --control-resolution 20
```

### Sensor Noise

Default sensor noise parameters (1σ values):

| Sensor | Default Noise | Units |
|--------|---------------|-------|
| Barometric Pressure | 50 | Pa |
| Temperature | 0.5 | K |
| Z-Acceleration | 0.5 | m/s² |

Adjust via multiplier:

```python
sim = create_simulation(
    sensor_noise_mult=1.5,  # 50% higher noise
    ...
)
```

Or configure in `config.py`:

```python
PRESSURE_NOISE_STD = 50.0    # Pa
TEMPERATURE_NOISE_STD = 0.5  # K
ACCEL_Z_NOISE_STD = 0.5      # m/s²
```

## Running Simulations

### Command Line Interface

```bash
# Basic simulation with airbrakes (loads data from tables/)
python main.py

# Simulation without airbrakes
python main.py --no-airbrakes

# Compare with/without airbrakes
python main.py --compare

# Set target apogee (in meters)
python main.py --target 9144

# Adjust control system Cd table resolution
python main.py --control-resolution 20

# Save plots to files
python main.py --save-plots --output-prefix my_rocket

# Suppress plot display
python main.py --no-plots --save-plots
```

### Python API

```python
from simulation import create_simulation

# Basic simulation using data tables
sim = create_simulation()
results = sim.run()
print(f"Apogee: {results.apogee_ft:.0f} ft")

# With custom target
sim = create_simulation(target_apogee_m=8500)

# Coarser control system tables (faster)
sim = create_simulation(control_cd_resolution=20)
```

## Monte Carlo Analysis

Monte Carlo simulations vary parameters to assess system robustness.

### Parameters Varied

| Parameter | Description | How It's Varied |
|-----------|-------------|-----------------|
| Body Cd | Body drag coefficient | % error at Mach 0 and 1.5, linearly interpolated |
| Airbrake Cd | Airbrake drag coefficient | Same as body Cd |
| Sensor Noise | Measurement noise levels | Multiplier on base noise |

### Running Monte Carlo

```bash
# 100 runs with default settings
python main.py --monte-carlo --runs 100

# Custom uncertainty parameters
python main.py --monte-carlo --runs 500 --cd-std 0.10 --noise-std 0.3 --seed 123
```

### Python API

```python
from monte_carlo import run_monte_carlo

# Run Monte Carlo (loads data from tables/)
results = run_monte_carlo(
    num_runs=100,
    cd_body_error_std=0.05,      # 5% Cd uncertainty
    cd_airbrake_error_std=0.05,  # 5% airbrake Cd uncertainty
    sensor_noise_std=0.2,        # 20% noise variation
    seed=42,
)

print(f"Mean apogee: {results.apogee_mean_ft:.0f} ft")
print(f"Std dev: {results.apogee_std_ft:.0f} ft")
print(f"Success rate (±500 ft): {results.get_success_rate(30000, 500)*100:.1f}%")
```

### Cd Error Model

The Cd error is modeled as a percentage difference that varies linearly with Mach number:

```
Cd_actual = Cd_model * (1 + error)

where:
  error = error_mach0 + (mach / 1.5) * (error_mach1.5 - error_mach0)
```

Example: If `error_mach0 = -0.10` and `error_mach1.5 = +0.10`:
- At Mach 0: Actual Cd is 10% lower than model
- At Mach 0.75: Actual Cd equals model
- At Mach 1.5: Actual Cd is 10% higher than model

## Visualizations

The system provides several visualization functions:

### Basic Trajectory
```python
from visualization import plot_trajectory
plot_trajectory(results, title="My Rocket", save_path="trajectory.png")
```

### Airbrake Performance (burnout to apogee)
```python
from visualization import plot_airbrake_performance
plot_airbrake_performance(results)
```
Shows:
- Deployment angle over time
- Mach number
- Body drag vs airbrake drag
- Airbrake influence (delta between full deploy/retract)

### Controller Telemetry
```python
from visualization import plot_controller_telemetry
plot_controller_telemetry(results)
```
Shows:
- Altitude estimation (estimated vs measured)
- Velocity estimation
- Mach estimation
- Commanded angle
- Predicted apogee
- Airbrake influence

### Monte Carlo Results
```python
from visualization import plot_monte_carlo_results
plot_monte_carlo_results(mc_results)
```
Shows:
- Apogee distribution histogram
- Sample trajectories
- Error scatter plot
- Statistics summary

### Summary Report
```python
from visualization import create_summary_report
create_summary_report(results, save_path="report.png")
```
Single-page comprehensive overview.

### Comparison Plot
```python
from visualization import plot_comparison
plot_comparison(
    [results_with_airbrakes, results_without_airbrakes],
    ["With Airbrakes", "Without Airbrakes"]
)
```

## Control System

The control system (`control_system.py`) is completely separate from the simulation environment.

### Sensor Inputs

The control system receives three measurements:
1. **Barometric Pressure** (Pa) - with Gaussian noise
2. **Temperature** (K) - with Gaussian noise  
3. **Z-Acceleration** (m/s²) - with Gaussian noise (specific force, includes gravity)

### State Estimation (EKF)

An Extended Kalman Filter fuses sensor data:
- **State**: [altitude, velocity]
- **Prediction**: Uses acceleration measurement
- **Update**: Uses altitude derived from pressure

### Lookup Tables

The control system uses pre-computed lookup tables instead of direct atmospheric calculations:
- Pressure → Altitude (1D, pressure is monotonic)
- Altitude → Temperature
- Altitude → Density
- Altitude → Speed of Sound

### Control Algorithm

1. Detect motor burnout (acceleration drops below 1g)
2. Wait for Mach < 1.0 before deploying
3. Predict apogee using numerical integration
4. Adjust airbrake angle to minimize apogee error

### Airbrake Constraints

| Parameter | Value |
|-----------|-------|
| Max Area | 10,000 mm² |
| Max Angle | 95° |
| Slew Rate | 100 °/s |
| Cd | 1.28 (flat disc) |
| Deploy Limit | Mach < 1.0 |

Area is proportional to angle:
```
area = (angle / 95°) * 10,000 mm²
```

## API Reference

### `create_simulation()`

Factory function to create a simulation environment.

```python
create_simulation(
    thrust_times: list = None,      # Time points (s)
    thrust_values: list = None,     # Thrust values (N)
    thrust_file: str = None,        # Path to thrust CSV
    dry_mass: float = 20.0,         # Dry mass (kg)
    propellant_mass: float = 5.0,   # Propellant mass (kg)
    reference_area: float = 0.01,   # Body reference area (m²)
    base_cd: float = 0.5,           # Base drag coefficient
    cd_vs_mach: list = None,        # [(mach, cd), ...] tuples
    target_apogee_m: float = None,  # Target apogee (m), default 9144
    enable_airbrakes: bool = True,  # Enable airbrake control
    sensor_noise_mult: float = 1.0, # Sensor noise multiplier
    cd_error_body: tuple = (0, 0),  # (mach0_err, mach1.5_err)
    cd_error_airbrake: tuple = (0, 0),
    seed: int = None,               # Random seed
) -> SimulationEnvironment
```

### `SimulationResults`

Results from a simulation run.

```python
results.time           # Time array (s)
results.altitude       # Altitude array (m)
results.velocity       # Velocity array (m/s)
results.acceleration   # Acceleration array (m/s²)
results.mach           # Mach number array
results.thrust         # Thrust array (N)
results.body_drag      # Body drag array (N)
results.airbrake_drag  # Airbrake drag array (N)
results.airbrake_angle # Deployment angle array (deg)

results.apogee_m       # Apogee altitude (m)
results.apogee_ft      # Apogee altitude (ft)
results.apogee_time    # Time of apogee (s)
results.max_velocity   # Maximum velocity (m/s)
results.max_mach       # Maximum Mach number
results.burnout_time   # Motor burnout time (s)

results.controller_telemetry  # Dict of controller data
```

### `run_monte_carlo()`

Run Monte Carlo simulation.

```python
run_monte_carlo(
    # ... same as create_simulation() ...
    num_runs: int = 100,
    cd_body_error_std: float = 0.05,
    cd_airbrake_error_std: float = 0.05,
    sensor_noise_std: float = 0.2,
    seed: int = 42,
    progress_callback: callable = None,
) -> MonteCarloResults
```

### `MonteCarloResults`

```python
mc_results.num_runs          # Number of runs
mc_results.apogees_ft        # Array of apogees (ft)
mc_results.apogee_mean_ft    # Mean apogee (ft)
mc_results.apogee_std_ft     # Std dev (ft)
mc_results.apogee_min_ft     # Minimum apogee (ft)
mc_results.apogee_max_ft     # Maximum apogee (ft)

mc_results.get_percentile(p) # Get p-th percentile (ft)
mc_results.get_success_rate(target, tolerance)  # Fraction within tolerance

mc_results.individual_results  # List of SimulationResults
mc_results.run_parameters      # List of parameter dicts
```

## Units

All internal calculations are done in **SI units (metric)**:
- Distance: meters (m)
- Mass: kilograms (kg)
- Time: seconds (s)
- Force: Newtons (N)
- Pressure: Pascals (Pa)
- Temperature: Kelvin (K)
- Area: square meters (m²)

**Apogee is reported in feet** as specified.

## File Structure

```
RocketPy/
├── main.py              # Entry point, CLI
├── config.py            # Configuration constants
├── simulation.py        # Simulation environment
├── control_system.py    # Control system (SEPARATE)
├── rocket_model.py      # Rocket model (thrust, mass, drag)
├── airbrake.py          # Airbrake physics
├── atmosphere.py        # Atmospheric models
├── lookup_tables.py     # Pre-computed lookup tables
├── sensors.py           # Sensor noise models
├── ekf.py               # Extended Kalman Filter
├── monte_carlo.py       # Monte Carlo framework
├── visualization.py     # Plotting functions
├── requirements.txt     # Dependencies
└── README.md            # This file
```

## License

MIT License - Stanford Student Space Initiative
