"""Diagnose model mismatch between controller and simulation."""
import numpy as np
from lookup_tables import get_atmosphere_lut, reset_lookup_tables

# Reset lookup tables to pick up launch altitude fix
reset_lookup_tables()
from atmosphere import density_from_altitude, temperature_from_altitude, speed_of_sound
from config import ROCKET_REFERENCE_AREA_M2, GRAVITY, LAUNCH_ALTITUDE_M
from data_loader import RocketDataTables, reset_rocket_tables

reset_rocket_tables()

# Get controller's lookup table
lut = get_atmosphere_lut()
tables = RocketDataTables()

print("="*70)
print("MODEL MISMATCH DIAGNOSIS")
print("="*70)

# 1. Compare atmospheric density
print("\n## 1. ATMOSPHERIC DENSITY COMPARISON")
print("Controller uses lookup tables, simulation uses direct ISA calculation")
print(f"Launch altitude: {LAUNCH_ALTITUDE_M:.1f} m")
print(f"{'Alt AGL (m)':>12} {'Sim rho':>12} {'Ctrl rho':>12} {'Diff%':>10}")
print("-"*50)

for alt in [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000]:
    sim_rho = density_from_altitude(alt)
    ctrl_rho = lut.get_density_from_altitude(alt)
    diff = (ctrl_rho / sim_rho - 1) * 100
    print(f"{alt:12.0f} {sim_rho:12.6f} {ctrl_rho:12.6f} {diff:+10.2f}%")

# 2. Compare speed of sound / Mach calculation
print("\n## 2. SPEED OF SOUND / MACH COMPARISON")
print(f"{'Alt (m)':>10} {'Vel (m/s)':>12} {'Sim Mach':>10} {'Ctrl Mach':>10} {'Diff%':>10}")
print("-"*55)

for alt in [0, 3000, 6000, 9000]:
    for vel in [100, 300, 500]:
        sim_sos = speed_of_sound(alt)
        sim_mach = vel / sim_sos
        ctrl_mach = lut.get_mach_from_velocity_altitude(vel, alt)
        diff = (ctrl_mach / sim_mach - 1) * 100
        print(f"{alt:10.0f} {vel:12.0f} {sim_mach:10.3f} {ctrl_mach:10.3f} {diff:+10.2f}%")

# 3. Compare Cd values
print("\n## 3. CD VALUES COMPARISON")
print("Both should use same data tables")
machs_test = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.2, 1.5, 1.8]
print(f"{'Mach':>8} {'Sim Cd':>10} {'Ctrl Cd':>10} {'Diff%':>10}")
print("-"*40)

for mach in machs_test:
    sim_cd = tables.get_cd(mach)
    ctrl_cd = tables.get_cd_control(mach)
    diff = (ctrl_cd / sim_cd - 1) * 100 if sim_cd > 0 else 0
    print(f"{mach:8.2f} {sim_cd:10.4f} {ctrl_cd:10.4f} {diff:+10.2f}%")

# 4. Check mass value
print("\n## 4. MASS VALUES")
print(f"Dry mass from tables: {tables.dry_mass:.3f} kg")
print(f"Initial mass: {tables.initial_mass:.3f} kg")
print(f"Burnout time: {tables.burnout_time:.3f} s")

# 5. Reference area
print("\n## 5. REFERENCE AREA")
print(f"ROCKET_REFERENCE_AREA_M2 = {ROCKET_REFERENCE_AREA_M2:.6f} m^2")

# 6. Test coast prediction accuracy
print("\n## 6. COAST PREDICTION TEST")
print("Simulating coast from burnout conditions...")

# Typical burnout conditions
h0 = 1600  # ~5200 ft
v0 = 630   # m/s

from control_system import AirbrakeController

ctrl = AirbrakeController(target_apogee_m=9144)

# Predict with no brakes
apogee_clean, t_clean = ctrl._simulate_coast(h0, v0, 0.0)
apogee_full, t_full = ctrl._simulate_coast(h0, v0, ctrl.Cd_air_max)

print(f"From h={h0}m, v={v0}m/s:")
print(f"  No brakes:   apogee = {apogee_clean:.0f} m ({apogee_clean*3.28084:.0f} ft), t = {t_clean:.1f}s")
print(f"  Full brakes: apogee = {apogee_full:.0f} m ({apogee_full*3.28084:.0f} ft), t = {t_full:.1f}s")
print(f"  Cd_air_max = {ctrl.Cd_air_max:.4f}")

# 7. Run actual simulation and compare
print("\n## 7. ACTUAL VS PREDICTED APOGEE")
from simulation import create_simulation

# Run without airbrakes
sim = create_simulation(enable_airbrakes=False)
results = sim.run()

print(f"Actual simulation (no brakes): {results.apogee_ft:.0f} ft ({results.apogee_ft/3.28084:.0f} m)")

# Get burnout state
burnout_idx = np.argmin(np.abs(np.array(results.time) - results.burnout_time))
h_burnout = results.altitude[burnout_idx]
v_burnout = results.velocity[burnout_idx]

print(f"At burnout (t={results.burnout_time:.2f}s):")
print(f"  Altitude: {h_burnout:.1f} m ({h_burnout*3.28084:.0f} ft)")
print(f"  Velocity: {v_burnout:.1f} m/s")

# Controller's prediction from burnout
pred_apogee, pred_t = ctrl._simulate_coast(h_burnout, v_burnout, 0.0)
print(f"\nController prediction (from burnout, no brakes):")
print(f"  Predicted apogee: {pred_apogee:.0f} m ({pred_apogee*3.28084:.0f} ft)")
print(f"  Actual apogee:    {results.apogee_ft/3.28084:.0f} m ({results.apogee_ft:.0f} ft)")
print(f"  Error: {(pred_apogee - results.apogee_ft/3.28084):.0f} m ({(pred_apogee*3.28084 - results.apogee_ft):.0f} ft)")

print("\n" + "="*70)
print("DIAGNOSIS SUMMARY")
print("="*70)
