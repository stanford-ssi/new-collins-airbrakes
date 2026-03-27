#!/usr/bin/env python3
"""
Example script demonstrating how to use the airbrake simulation system.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation import create_simulation
from monte_carlo import run_monte_carlo
from visualization import (
    plot_trajectory,
    plot_airbrake_performance,
    plot_controller_telemetry,
    plot_monte_carlo_results,
    plot_comparison,
    create_summary_report,
)
from config import TARGET_APOGEE_M, METERS_TO_FEET


def example_single_simulation():
    """Run a single simulation with airbrakes enabled."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Single Simulation with Airbrakes")
    print("="*60)
    
    sim = create_simulation(
        thrust_times=[0.0, 0.1, 3.0, 3.5],
        thrust_values=[0.0, 5000.0, 5000.0, 0.0],
        dry_mass=20.0,
        propellant_mass=5.0,
        reference_area=0.01,
        base_cd=0.5,
        target_apogee_m=TARGET_APOGEE_M,
        enable_airbrakes=True,
    )
    
    results = sim.run()
    
    print(f"\nResults:")
    print(f"  Apogee: {results.apogee_ft:.0f} ft ({results.apogee_m:.1f} m)")
    print(f"  Target: {TARGET_APOGEE_M * METERS_TO_FEET:.0f} ft")
    print(f"  Error:  {results.apogee_ft - TARGET_APOGEE_M * METERS_TO_FEET:+.0f} ft")
    print(f"  Max Mach: {results.max_mach:.2f}")
    
    plot_trajectory(results, title="Single Simulation", show=True)
    create_summary_report(results, show=True)
    
    return results


def example_comparison():
    """Compare simulation with and without airbrakes."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Comparison (With vs Without Airbrakes)")
    print("="*60)
    
    sim_with = create_simulation(
        thrust_times=[0.0, 0.1, 3.0, 3.5],
        thrust_values=[0.0, 5000.0, 5000.0, 0.0],
        dry_mass=20.0,
        propellant_mass=5.0,
        reference_area=0.01,
        base_cd=0.5,
        enable_airbrakes=True,
    )
    results_with = sim_with.run()
    
    sim_without = create_simulation(
        thrust_times=[0.0, 0.1, 3.0, 3.5],
        thrust_values=[0.0, 5000.0, 5000.0, 0.0],
        dry_mass=20.0,
        propellant_mass=5.0,
        reference_area=0.01,
        base_cd=0.5,
        enable_airbrakes=False,
    )
    results_without = sim_without.run()
    
    print(f"\nResults:")
    print(f"  With Airbrakes:    {results_with.apogee_ft:.0f} ft")
    print(f"  Without Airbrakes: {results_without.apogee_ft:.0f} ft")
    print(f"  Difference:        {results_without.apogee_ft - results_with.apogee_ft:.0f} ft")
    
    plot_comparison(
        [results_with, results_without],
        ["With Airbrakes", "Without Airbrakes"],
        title="Airbrake Comparison",
        show=True,
    )
    
    return results_with, results_without


def example_mach_varying_cd():
    """Simulation with Mach-varying drag coefficient."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Mach-Varying Drag Coefficient")
    print("="*60)
    
    cd_vs_mach = [
        (0.0, 0.45),
        (0.8, 0.55),
        (1.0, 0.80),
        (1.2, 0.70),
        (1.5, 0.55),
    ]
    
    sim = create_simulation(
        thrust_times=[0.0, 0.1, 3.0, 3.5],
        thrust_values=[0.0, 5000.0, 5000.0, 0.0],
        dry_mass=20.0,
        propellant_mass=5.0,
        reference_area=0.01,
        cd_vs_mach=cd_vs_mach,
        enable_airbrakes=True,
    )
    
    results = sim.run()
    
    print(f"\nResults with Mach-varying Cd:")
    print(f"  Apogee: {results.apogee_ft:.0f} ft")
    print(f"  Max Mach: {results.max_mach:.2f}")
    
    plot_airbrake_performance(results, title="Mach-Varying Cd Simulation", show=True)
    
    return results


def example_monte_carlo():
    """Run Monte Carlo simulation."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Monte Carlo Simulation (50 runs)")
    print("="*60)
    
    def progress(i, total):
        print(f"  Run {i+1}/{total}...", end='\r')
    
    mc_results = run_monte_carlo(
        thrust_times=[0.0, 0.1, 3.0, 3.5],
        thrust_values=[0.0, 5000.0, 5000.0, 0.0],
        dry_mass=20.0,
        propellant_mass=5.0,
        reference_area=0.01,
        base_cd=0.5,
        enable_airbrakes=True,
        num_runs=50,
        cd_body_error_std=0.05,
        cd_airbrake_error_std=0.05,
        sensor_noise_std=0.2,
        seed=42,
        progress_callback=progress,
    )
    
    print(f"\n\nMonte Carlo Results:")
    print(f"  Mean Apogee: {mc_results.apogee_mean_ft:.0f} ft")
    print(f"  Std Dev:     {mc_results.apogee_std_ft:.0f} ft")
    print(f"  Min:         {mc_results.apogee_min_ft:.0f} ft")
    print(f"  Max:         {mc_results.apogee_max_ft:.0f} ft")
    
    target = TARGET_APOGEE_M * METERS_TO_FEET
    print(f"\n  Success Rate (±500 ft):  {mc_results.get_success_rate(target, 500)*100:.1f}%")
    print(f"  Success Rate (±1000 ft): {mc_results.get_success_rate(target, 1000)*100:.1f}%")
    
    plot_monte_carlo_results(mc_results, show=True)
    
    return mc_results


def main():
    """Run all examples."""
    print("\nAirbrake Simulation Examples")
    print("="*60)
    
    example_single_simulation()
    
    example_comparison()
    
    example_mach_varying_cd()
    
    example_monte_carlo()
    
    print("\n" + "="*60)
    print("All examples completed!")
    print("="*60)


if __name__ == '__main__':
    main()
