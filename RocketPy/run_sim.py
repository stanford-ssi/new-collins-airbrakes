from simulation import create_simulation
from debug import configure_debug, DebugLevel

# Enable debug output to diagnose issues
configure_debug(level=DebugLevel.ANOMALIES)

sim = create_simulation(
    target_apogee_m=9144,
    
    # Airframe Cd scaling (Mach-dependent, linear interpolation)
    # cd_scale_mach0=0.9,       # Cd multiplier at Mach 0
    # cd_scale_mach2=1.1,       # Cd multiplier at Mach 2+
    
    # Airbrake Cd scaling (Mach-dependent)
    # airbrake_cd_scale_mach0=1.1,
    # airbrake_cd_scale_mach2=1.1,
    
    # Motor performance
    thrust_scale=1.1,
    
    # Launch site conditions (offsets from truth)
    launch_altitude_offset_m=0,  # Offset from truth (default ~630m MSL)
    launch_temp_offset_k=0.0,  # +5K hotter than ISA
    
    # Airbrake slew rate
    airbrake_slew_rate_deg_s=200,
    
    # Control system LUT resolutions
    control_cd_resolution=10,
    control_density_resolution=10,
    control_mass_resolution=20, #Not used

    airbrake_max_area_m2=0.01,  # default: 0.006 = 60 cm²

    pressure_noise_offset_pa=0.0,      # Pressure bias (Pa)
    pressure_noise_std_pa=300.0,       # Pressure std dev (Pa), None = default
    temperature_noise_offset_k=0.0,    # Temperature bias (K)
    temperature_noise_std_k=5,       # Temperature std dev (K), None = default
    accel_noise_offset_mss=0.0,        # Accelerometer bias (m/s²)
    accel_noise_std_mss=5.0,           # Accelerometer std dev (m/s²), None = default

    #airframe_cd_csv="path/to/airframe_cd.csv",  # Format: Mach,Cd
    airbrake_cd_csv="cd mach relations/flat_disc_cd.csv" , # Format: Mach,Cd
    airframe_cd_csv="cd mach relations/V2_CD_Mach_modified.csv",

    control_latency_ms=0

)

# Run the simulation
results = sim.run()

# Print results
print(f"Apogee: {results.apogee_ft:.0f} ft (target: 30000 ft)")
print(f"Error: {results.apogee_ft - 30000:+.0f} ft")
print(f"Max Mach: {results.max_mach:.2f}")

# Optional: Launch dashboard
from dashboard import run_dashboard
run_dashboard(results, sim.sim_config, sim.lut_config)