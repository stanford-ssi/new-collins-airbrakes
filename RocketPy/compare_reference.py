"""Compare simulation results to V3 reference data."""
import numpy as np
from simulation import create_simulation
from data_loader import reset_rocket_tables
from config import METERS_TO_FEET

# Reset cached tables to pick up V2 files
reset_rocket_tables()

# Run simulation without airbrakes
sim = create_simulation(enable_airbrakes=False)
results = sim.run()

# Load V3 reference
ref_data = []
with open('reference/V3 Data Export.csv', 'r') as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split(',')
        if len(parts) >= 4:
            try:
                ref_data.append({
                    'time': float(parts[0]),
                    'alt_ft': float(parts[1]),
                    'vel': float(parts[2]),
                    'accel': float(parts[3]),
                })
            except:
                continue

print("="*70)
print("TRAJECTORY COMPARISON: Simulation vs V3 Reference")
print("="*70)

# Find reference apogee
max_alt_ref = max(r['alt_ft'] for r in ref_data)

print("\n## KEY METRICS")
print(f"Simulation Apogee: {results.apogee_ft:.0f} ft")
print(f"Reference Apogee:  {max_alt_ref:.0f} ft")
print(f"Difference:        {results.apogee_ft - max_alt_ref:+.0f} ft ({(results.apogee_ft/max_alt_ref - 1)*100:+.2f}%)")

# Find reference max velocity
max_vel_ref = max(r['vel'] for r in ref_data)
print(f"\nSimulation Max Velocity: {results.max_velocity:.1f} m/s")
print(f"Reference Max Velocity:  {max_vel_ref:.1f} m/s")
print(f"Difference:              {results.max_velocity - max_vel_ref:+.1f} m/s ({(results.max_velocity/max_vel_ref - 1)*100:+.2f}%)")

# Compare at specific time points
print("\n## TRAJECTORY COMPARISON AT KEY TIMES")
print(f"{'Time':>8} {'Sim Alt(ft)':>12} {'Ref Alt(ft)':>12} {'Diff(ft)':>10} {'Sim Vel':>10} {'Ref Vel':>10} {'Diff':>8}")
print("-"*76)

# Sample at key times from V3 reference
key_times = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]

for t in key_times:
    # Find closest simulation point
    sim_idx = np.argmin(np.abs(np.array(results.time) - t))
    sim_alt = results.altitude[sim_idx] * METERS_TO_FEET
    sim_vel = results.velocity[sim_idx]
    
    # Find closest reference point
    ref_idx = min(range(len(ref_data)), key=lambda i: abs(ref_data[i]['time'] - t))
    ref_alt = ref_data[ref_idx]['alt_ft']
    ref_vel = ref_data[ref_idx]['vel']
    
    alt_diff = sim_alt - ref_alt
    vel_diff = sim_vel - ref_vel
    print(f"{t:8.1f} {sim_alt:12.0f} {ref_alt:12.0f} {alt_diff:+10.0f} {sim_vel:10.1f} {ref_vel:10.1f} {vel_diff:+8.1f}")

print("="*70)
