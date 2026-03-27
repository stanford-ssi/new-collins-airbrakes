#!/usr/bin/env python3
"""
Airbrake Simulation Main Entry Point

This is the main entry point for running airbrake simulations.
Supports single runs, Monte Carlo simulations, and comparison runs.
"""

import argparse
import json
import sys
from typing import Optional

from simulation import create_simulation, SimulationResults
from monte_carlo import run_monte_carlo, MonteCarloSimulation, MonteCarloConfig
from visualization import (
    plot_trajectory,
    plot_airbrake_performance,
    plot_controller_telemetry,
    plot_monte_carlo_results,
    plot_comparison,
    create_summary_report,
)
from config import TARGET_APOGEE_M, METERS_TO_FEET
from dashboard import run_dashboard, show_figures


def load_config_file(filepath: str) -> dict:
    """
    Load simulation configuration from JSON file.
    
    Args:
        filepath: Path to JSON config file
        
    Returns:
        Configuration dictionary
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def run_single_simulation(
    config: dict,
    enable_airbrakes: bool = True,
    show_plots: bool = True,
    save_plots: bool = False,
    output_prefix: str = "sim",
    use_dashboard: bool = False,
) -> SimulationResults:
    """
    Run a single simulation.
    
    Args:
        config: Simulation configuration dictionary
        enable_airbrakes: Whether to enable airbrakes
        show_plots: Whether to display plots
        save_plots: Whether to save plots to files
        output_prefix: Prefix for output files
        
    Returns:
        SimulationResults
    """
    config['enable_airbrakes'] = enable_airbrakes
    
    sim = create_simulation(**config)
    results = sim.run()
    
    print("\n" + "="*60)
    print("SIMULATION RESULTS")
    print("="*60)
    print(f"  Apogee: {results.apogee_ft:.0f} ft ({results.apogee_m:.1f} m)")
    print(f"  Target: {TARGET_APOGEE_M * METERS_TO_FEET:.0f} ft")
    print(f"  Error:  {results.apogee_ft - TARGET_APOGEE_M * METERS_TO_FEET:+.0f} ft")
    print()
    print(f"  Max Velocity: {results.max_velocity:.1f} m/s")
    print(f"  Max Mach:     {results.max_mach:.2f}")
    print(f"  Max Accel:    {results.max_acceleration:.1f} m/s² ({results.max_acceleration/9.81:.1f} g)")
    print()
    print(f"  Burnout Time:     {results.burnout_time:.2f} s")
    print(f"  Burnout Altitude: {results.burnout_altitude * METERS_TO_FEET:.0f} ft")
    print(f"  Apogee Time:      {results.apogee_time:.2f} s")
    print("="*60 + "\n")
    
    if use_dashboard:
        # Pass configs from simulation for dashboard display
        sim_config = getattr(sim, 'sim_config', None)
        lut_config = getattr(sim, 'lut_config', None)
        run_dashboard(results, sim_config, lut_config)
    elif show_plots or save_plots:
        save_path = f"{output_prefix}_trajectory.png" if save_plots else None
        plot_trajectory(results, title="Rocket Trajectory", save_path=save_path, show=show_plots)
        
        if enable_airbrakes:
            save_path = f"{output_prefix}_airbrake.png" if save_plots else None
            plot_airbrake_performance(results, title="Airbrake Performance", 
                                     save_path=save_path, show=show_plots)
            
            save_path = f"{output_prefix}_controller.png" if save_plots else None
            plot_controller_telemetry(results, title="Controller Telemetry",
                                     save_path=save_path, show=show_plots)
        
        save_path = f"{output_prefix}_report.png" if save_plots else None
        create_summary_report(results, save_path=save_path, show=show_plots)
    
    return results


def run_comparison(
    config: dict,
    show_plots: bool = True,
    save_plots: bool = False,
    output_prefix: str = "comparison",
) -> tuple:
    """
    Run simulation with and without airbrakes for comparison.
    
    Args:
        config: Simulation configuration
        show_plots: Whether to display plots
        save_plots: Whether to save plots
        output_prefix: Output file prefix
        
    Returns:
        Tuple of (with_airbrakes_results, without_airbrakes_results)
    """
    print("\nRunning simulation WITH airbrakes...")
    config_with = config.copy()
    config_with['enable_airbrakes'] = True
    sim_with = create_simulation(**config_with)
    results_with = sim_with.run()
    
    print("Running simulation WITHOUT airbrakes...")
    config_without = config.copy()
    config_without['enable_airbrakes'] = False
    sim_without = create_simulation(**config_without)
    results_without = sim_without.run()
    
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    print(f"  With Airbrakes:    {results_with.apogee_ft:.0f} ft")
    print(f"  Without Airbrakes: {results_without.apogee_ft:.0f} ft")
    print(f"  Difference:        {results_without.apogee_ft - results_with.apogee_ft:.0f} ft")
    print(f"  Target:            {TARGET_APOGEE_M * METERS_TO_FEET:.0f} ft")
    print("="*60 + "\n")
    
    if show_plots or save_plots:
        save_path = f"{output_prefix}.png" if save_plots else None
        plot_comparison(
            [results_with, results_without],
            ["With Airbrakes", "Without Airbrakes"],
            title="Airbrake Comparison",
            save_path=save_path,
            show=show_plots,
        )
    
    return results_with, results_without


def run_monte_carlo_simulation(
    config: dict,
    num_runs: int = 100,
    cd_body_std: float = 0.05,
    cd_airbrake_std: float = 0.05,
    sensor_noise_std: float = 0.2,
    seed: int = 42,
    show_plots: bool = True,
    save_plots: bool = False,
    output_prefix: str = "monte_carlo",
):
    """
    Run Monte Carlo simulation.
    
    Args:
        config: Base simulation configuration
        num_runs: Number of Monte Carlo runs
        cd_body_std: Std dev of body Cd error
        cd_airbrake_std: Std dev of airbrake Cd error
        sensor_noise_std: Std dev of sensor noise multiplier
        seed: Random seed
        show_plots: Whether to display plots
        save_plots: Whether to save plots
        output_prefix: Output file prefix
        
    Returns:
        MonteCarloResults
    """
    def progress(i, total):
        if i % 10 == 0 or i == total - 1:
            print(f"  Run {i+1}/{total}...", end='\r')
    
    print(f"\nRunning Monte Carlo simulation with {num_runs} runs...")
    
    results = run_monte_carlo(
        **config,
        num_runs=num_runs,
        cd_body_error_std=cd_body_std,
        cd_airbrake_error_std=cd_airbrake_std,
        sensor_noise_std=sensor_noise_std,
        seed=seed,
        progress_callback=progress,
    )
    
    print("\n" + "="*60)
    print("MONTE CARLO RESULTS")
    print("="*60)
    print(f"  Number of Runs: {results.num_runs}")
    print()
    print(f"  Mean Apogee:   {results.apogee_mean_ft:.0f} ft")
    print(f"  Std Dev:       {results.apogee_std_ft:.0f} ft")
    print(f"  Min Apogee:    {results.apogee_min_ft:.0f} ft")
    print(f"  Max Apogee:    {results.apogee_max_ft:.0f} ft")
    print()
    print(f"  5th Percentile:  {results.get_percentile(5):.0f} ft")
    print(f"  50th Percentile: {results.get_percentile(50):.0f} ft")
    print(f"  95th Percentile: {results.get_percentile(95):.0f} ft")
    print()
    target_ft = TARGET_APOGEE_M * METERS_TO_FEET
    print(f"  Success Rate (±500 ft):  {results.get_success_rate(target_ft, 500)*100:.1f}%")
    print(f"  Success Rate (±1000 ft): {results.get_success_rate(target_ft, 1000)*100:.1f}%")
    print("="*60 + "\n")
    
    if show_plots or save_plots:
        save_path = f"{output_prefix}_results.png" if save_plots else None
        plot_monte_carlo_results(results, save_path=save_path, show=show_plots)
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Airbrake Simulation System - Uses data from tables/ directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Data Tables:
  The simulation loads rocket data from the tables/ directory:
  - Motor Thrust.csv: Time vs Thrust curve
  - Mass Change.csv: Time vs Mass curve  
  - CD Mach Number.csv: Cd vs Mach lookup table

Examples:
  # Run single simulation
  python main.py
  
  # Run simulation without airbrakes
  python main.py --no-airbrakes
  
  # Run Monte Carlo simulation
  python main.py --monte-carlo --runs 100
  
  # Compare with and without airbrakes
  python main.py --compare
  
  # Set target apogee
  python main.py --target 9144
  
  # Save plots to files
  python main.py --save-plots --output-prefix my_sim
  
  # Set control system Cd table coarseness
  python main.py --control-resolution 20
        """
    )
    
    parser.add_argument('--target', type=float, help='Target apogee in meters (default: 9144m / 30000ft)')
    parser.add_argument('--control-resolution', type=int, default=50, 
                        help='Cd table resolution for control system (default: 50 points)')
    
    parser.add_argument('--no-airbrakes', action='store_true', help='Disable airbrakes')
    parser.add_argument('--compare', action='store_true', help='Compare with/without airbrakes')
    
    parser.add_argument('--monte-carlo', action='store_true', help='Run Monte Carlo simulation')
    parser.add_argument('--runs', type=int, default=100, help='Number of Monte Carlo runs')
    parser.add_argument('--cd-std', type=float, default=0.05, help='Cd error std dev')
    parser.add_argument('--noise-std', type=float, default=0.2, help='Sensor noise std dev')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    parser.add_argument('--no-plots', action='store_true', help='Disable plot display')
    parser.add_argument('--save-plots', action='store_true', help='Save plots to files')
    parser.add_argument('--output-prefix', type=str, default='sim', help='Output file prefix')
    parser.add_argument('--dashboard', action='store_true', help='Launch interactive Plotly/Dash dashboard')
    
    args = parser.parse_args()
    
    config = {}
    
    if args.target:
        config['target_apogee_m'] = args.target
        
    config['control_cd_resolution'] = args.control_resolution
        
    show_plots = not args.no_plots
    
    if args.monte_carlo:
        run_monte_carlo_simulation(
            config,
            num_runs=args.runs,
            cd_body_std=args.cd_std,
            cd_airbrake_std=args.cd_std,
            sensor_noise_std=args.noise_std,
            seed=args.seed,
            show_plots=show_plots,
            save_plots=args.save_plots,
            output_prefix=args.output_prefix,
        )
    elif args.compare:
        run_comparison(
            config,
            show_plots=show_plots,
            save_plots=args.save_plots,
            output_prefix=args.output_prefix,
        )
    else:
        run_single_simulation(
            config,
            enable_airbrakes=not args.no_airbrakes,
            show_plots=show_plots,
            save_plots=args.save_plots,
            output_prefix=args.output_prefix,
            use_dashboard=args.dashboard,
        )


if __name__ == '__main__':
    main()
