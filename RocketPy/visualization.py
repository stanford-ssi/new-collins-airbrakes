"""
Visualization module for airbrake simulation.
Provides comprehensive plotting of simulation results.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Optional, List

from simulation import SimulationResults
from monte_carlo import MonteCarloResults
from config import METERS_TO_FEET, TARGET_APOGEE_FT


def plot_trajectory(
    results: SimulationResults,
    title: str = "Rocket Trajectory",
    save_path: str = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot altitude and velocity over time.
    
    Args:
        results: SimulationResults from a simulation run
        title: Plot title
        save_path: Path to save figure (optional)
        show: Whether to display the plot
        
    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    altitude_ft = results.altitude * METERS_TO_FEET
    axes[0].plot(results.time, altitude_ft, 'b-', linewidth=2, label='Altitude')
    axes[0].axhline(y=results.apogee_ft, color='r', linestyle='--', alpha=0.7, 
                    label=f'Apogee: {results.apogee_ft:.0f} ft')
    axes[0].axhline(y=TARGET_APOGEE_FT, color='g', linestyle=':', alpha=0.7,
                    label=f'Target: {TARGET_APOGEE_FT:.0f} ft')
    axes[0].set_ylabel('Altitude (ft)', fontsize=12)
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(title, fontsize=14)
    
    axes[1].plot(results.time, results.velocity, 'r-', linewidth=2, label='Velocity')
    axes[1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    if results.burnout_time > 0:
        axes[1].axvline(x=results.burnout_time, color='orange', linestyle='--', 
                        alpha=0.7, label=f'Burnout: {results.burnout_time:.2f}s')
    axes[1].set_xlabel('Time (s)', fontsize=12)
    axes[1].set_ylabel('Velocity (m/s)', fontsize=12)
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
        
    return fig


def plot_airbrake_performance(
    results: SimulationResults,
    title: str = "Airbrake Performance",
    save_path: str = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot airbrake-specific metrics from burnout to apogee.
    
    Includes:
    - Deployment angle over time
    - Body drag vs airbrake drag
    - Airbrake influence (delta between full deploy and full retract)
    
    Args:
        results: SimulationResults
        title: Plot title
        save_path: Path to save figure
        show: Whether to display
        
    Returns:
        matplotlib Figure
    """
    burnout_idx = np.searchsorted(results.time, results.burnout_time)
    apogee_idx = np.argmax(results.altitude)
    
    time_slice = results.time[burnout_idx:apogee_idx+1]
    
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 2, figure=fig)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(time_slice, results.airbrake_angle[burnout_idx:apogee_idx+1], 
             'b-', linewidth=2)
    ax1.set_ylabel('Deployment Angle (deg)', fontsize=11)
    ax1.set_title('Airbrake Deployment', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 100])
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(time_slice, results.mach[burnout_idx:apogee_idx+1], 
             'g-', linewidth=2)
    ax2.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='Mach 1')
    ax2.set_ylabel('Mach Number', fontsize=11)
    ax2.set_title('Mach Number', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax3 = fig.add_subplot(gs[1, :])
    ax3.plot(time_slice, results.body_drag[burnout_idx:apogee_idx+1], 
             'b-', linewidth=2, label='Body Drag')
    ax3.plot(time_slice, results.airbrake_drag[burnout_idx:apogee_idx+1], 
             'r-', linewidth=2, label='Airbrake Drag')
    ax3.plot(time_slice, 
             results.body_drag[burnout_idx:apogee_idx+1] + results.airbrake_drag[burnout_idx:apogee_idx+1],
             'k--', linewidth=1.5, label='Total Drag', alpha=0.7)
    ax3.set_xlabel('Time (s)', fontsize=11)
    ax3.set_ylabel('Drag Force (N)', fontsize=11)
    ax3.set_title('Drag Forces: Body vs Airbrakes', fontsize=12)
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    ax4 = fig.add_subplot(gs[2, :])
    telemetry = results.controller_telemetry
    if telemetry and len(telemetry.get('time', [])) > 0:
        tel_time = np.array(telemetry['time'])
        
        # Handle both old and new telemetry formats
        if 'drag_influence' in telemetry:
            tel_influence = np.array(telemetry['drag_influence'])
        elif 'apogee_clean' in telemetry and 'apogee_full_brake' in telemetry:
            # Compute influence from apogee predictions
            apogee_clean = np.array(telemetry['apogee_clean'])
            apogee_full_brake = np.array(telemetry['apogee_full_brake'])
            tel_influence = apogee_clean - apogee_full_brake
        else:
            tel_influence = np.zeros_like(tel_time)
        
        mask = (tel_time >= results.burnout_time) & (tel_time <= results.apogee_time)
        ax4.plot(tel_time[mask], tel_influence[mask] * METERS_TO_FEET, 
                 'purple', linewidth=2)
        ax4.set_ylabel('Airbrake Influence (ft)', fontsize=11)
        ax4.set_title('Airbrake Influence (Altitude Delta: Retracted vs Deployed)', fontsize=12)
    else:
        ax4.text(0.5, 0.5, 'No controller telemetry available', 
                 ha='center', va='center', transform=ax4.transAxes)
    ax4.set_xlabel('Time (s)', fontsize=11)
    ax4.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
        
    return fig


def plot_controller_telemetry(
    results: SimulationResults,
    title: str = "Controller Telemetry",
    save_path: str = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot detailed controller telemetry.
    
    Args:
        results: SimulationResults
        title: Plot title
        save_path: Path to save figure
        show: Whether to display
        
    Returns:
        matplotlib Figure
    """
    telemetry = results.controller_telemetry
    
    if not telemetry or len(telemetry.get('time', [])) == 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No controller telemetry available', 
                ha='center', va='center', fontsize=14)
        if show:
            plt.show()
        return fig
        
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    
    time = np.array(telemetry['time'])
    
    axes[0, 0].plot(time, np.array(telemetry['altitude_est']) * METERS_TO_FEET, 
                    'b-', label='Estimated', linewidth=2)
    axes[0, 0].plot(time, np.array(telemetry['altitude_meas']) * METERS_TO_FEET, 
                    'r.', label='Measured', alpha=0.5, markersize=2)
    axes[0, 0].set_ylabel('Altitude (ft)')
    axes[0, 0].set_title('Altitude Estimation')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(time, telemetry['velocity_est'], 'b-', linewidth=2)
    axes[0, 1].set_ylabel('Velocity (m/s)')
    axes[0, 1].set_title('Velocity Estimation')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(time, telemetry['mach_est'], 'g-', linewidth=2)
    axes[1, 0].axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='Mach 1')
    axes[1, 0].set_ylabel('Mach Number')
    axes[1, 0].set_title('Estimated Mach Number')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(time, telemetry['commanded_angle'], 'b-', linewidth=2)
    axes[1, 1].set_ylabel('Angle (deg)')
    axes[1, 1].set_title('Commanded Airbrake Angle')
    axes[1, 1].set_ylim([0, 100])
    axes[1, 1].grid(True, alpha=0.3)
    
    axes[2, 0].plot(time, np.array(telemetry['predicted_apogee']) * METERS_TO_FEET, 
                    'b-', linewidth=2, label='Predicted')
    axes[2, 0].axhline(y=TARGET_APOGEE_FT, color='g', linestyle='--', alpha=0.7,
                       label=f'Target: {TARGET_APOGEE_FT} ft')
    axes[2, 0].set_xlabel('Time (s)')
    axes[2, 0].set_ylabel('Apogee (ft)')
    axes[2, 0].set_title('Predicted Apogee')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    # Handle both old and new telemetry formats
    if 'drag_influence' in telemetry:
        influence = np.array(telemetry['drag_influence'])
    elif 'apogee_clean' in telemetry and 'apogee_full_brake' in telemetry:
        influence = np.array(telemetry['apogee_clean']) - np.array(telemetry['apogee_full_brake'])
    else:
        influence = np.zeros_like(time)
    axes[2, 1].plot(time, influence * METERS_TO_FEET, 'purple', linewidth=2)
    axes[2, 1].set_xlabel('Time (s)')
    axes[2, 1].set_ylabel('Influence (ft)')
    axes[2, 1].set_title('Airbrake Influence')
    axes[2, 1].grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
        
    return fig


def plot_monte_carlo_results(
    mc_results: MonteCarloResults,
    target_apogee_ft: float = TARGET_APOGEE_FT,
    title: str = "Monte Carlo Results",
    save_path: str = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot Monte Carlo simulation results.
    
    Args:
        mc_results: MonteCarloResults
        target_apogee_ft: Target apogee in feet
        title: Plot title
        save_path: Path to save figure
        show: Whether to display
        
    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].hist(mc_results.apogees_ft, bins=30, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(x=target_apogee_ft, color='g', linestyle='--', linewidth=2,
                       label=f'Target: {target_apogee_ft:.0f} ft')
    axes[0, 0].axvline(x=mc_results.apogee_mean_ft, color='r', linestyle='-', linewidth=2,
                       label=f'Mean: {mc_results.apogee_mean_ft:.0f} ft')
    axes[0, 0].axvline(x=mc_results.apogee_mean_ft + mc_results.apogee_std_ft, 
                       color='r', linestyle=':', linewidth=1.5, alpha=0.7)
    axes[0, 0].axvline(x=mc_results.apogee_mean_ft - mc_results.apogee_std_ft, 
                       color='r', linestyle=':', linewidth=1.5, alpha=0.7,
                       label=f'±1σ: {mc_results.apogee_std_ft:.0f} ft')
    axes[0, 0].set_xlabel('Apogee (ft)')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Apogee Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    for i, result in enumerate(mc_results.individual_results[:20]):  # Plot first 20
        alpha = 0.3 if i > 0 else 0.8
        axes[0, 1].plot(result.time, result.altitude * METERS_TO_FEET, 
                        alpha=alpha, linewidth=0.8)
    axes[0, 1].axhline(y=target_apogee_ft, color='g', linestyle='--', linewidth=2,
                       label=f'Target: {target_apogee_ft} ft')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Altitude (ft)')
    axes[0, 1].set_title('Sample Trajectories (first 20 runs)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    run_indices = np.arange(len(mc_results.apogees_ft))
    errors = mc_results.apogees_ft - target_apogee_ft
    colors = ['g' if abs(e) <= 500 else 'orange' if abs(e) <= 1000 else 'r' for e in errors]
    axes[1, 0].scatter(run_indices, errors, c=colors, alpha=0.6, s=20)
    axes[1, 0].axhline(y=0, color='k', linestyle='-', linewidth=1)
    axes[1, 0].axhline(y=500, color='g', linestyle='--', alpha=0.5)
    axes[1, 0].axhline(y=-500, color='g', linestyle='--', alpha=0.5, label='±500 ft')
    axes[1, 0].set_xlabel('Run Index')
    axes[1, 0].set_ylabel('Apogee Error (ft)')
    axes[1, 0].set_title('Apogee Error by Run')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    values = [mc_results.get_percentile(p) for p in percentiles]
    
    stats_text = f"""
    Runs: {mc_results.num_runs}
    
    Mean: {mc_results.apogee_mean_ft:.0f} ft
    Std Dev: {mc_results.apogee_std_ft:.0f} ft
    
    Min: {mc_results.apogee_min_ft:.0f} ft
    Max: {mc_results.apogee_max_ft:.0f} ft
    
    Percentiles:
      5th: {values[0]:.0f} ft
      25th: {values[2]:.0f} ft
      50th: {values[3]:.0f} ft
      75th: {values[4]:.0f} ft
      95th: {values[6]:.0f} ft
    
    Success Rates:
      ±500 ft: {mc_results.get_success_rate(target_apogee_ft, 500)*100:.1f}%
      ±1000 ft: {mc_results.get_success_rate(target_apogee_ft, 1000)*100:.1f}%
    """
    
    axes[1, 1].text(0.1, 0.95, stats_text, transform=axes[1, 1].transAxes,
                    fontsize=11, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[1, 1].axis('off')
    axes[1, 1].set_title('Statistics Summary')
    
    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
        
    return fig


def plot_comparison(
    results_list: List[SimulationResults],
    labels: List[str],
    title: str = "Simulation Comparison",
    save_path: str = None,
    show: bool = True,
) -> plt.Figure:
    """
    Compare multiple simulation results.
    
    Args:
        results_list: List of SimulationResults
        labels: Labels for each result
        title: Plot title
        save_path: Path to save figure
        show: Whether to display
        
    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = plt.cm.tab10(np.linspace(0, 1, len(results_list)))
    
    for results, label, color in zip(results_list, labels, colors):
        altitude_ft = results.altitude * METERS_TO_FEET
        axes[0, 0].plot(results.time, altitude_ft, color=color, linewidth=2, 
                        label=f'{label}: {results.apogee_ft:.0f} ft')
    axes[0, 0].axhline(y=TARGET_APOGEE_FT, color='k', linestyle='--', alpha=0.5,
                       label=f'Target: {TARGET_APOGEE_FT} ft')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Altitude (ft)')
    axes[0, 0].set_title('Altitude Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    for results, label, color in zip(results_list, labels, colors):
        axes[0, 1].plot(results.time, results.velocity, color=color, linewidth=2, label=label)
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Velocity (m/s)')
    axes[0, 1].set_title('Velocity Comparison')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    for results, label, color in zip(results_list, labels, colors):
        axes[1, 0].plot(results.time, results.mach, color=color, linewidth=2, label=label)
    axes[1, 0].axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Mach 1')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Mach Number')
    axes[1, 0].set_title('Mach Number Comparison')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    for results, label, color in zip(results_list, labels, colors):
        axes[1, 1].plot(results.time, results.airbrake_angle, color=color, linewidth=2, label=label)
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Angle (deg)')
    axes[1, 1].set_title('Airbrake Deployment')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim([0, 100])
    
    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
        
    return fig


def create_summary_report(
    results: SimulationResults,
    save_path: str = "simulation_report.png",
    show: bool = True,
) -> plt.Figure:
    """
    Create a comprehensive single-page summary report.
    
    Args:
        results: SimulationResults
        save_path: Path to save figure
        show: Whether to display
        
    Returns:
        matplotlib Figure
    """
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, :2])
    altitude_ft = results.altitude * METERS_TO_FEET
    ax1.plot(results.time, altitude_ft, 'b-', linewidth=2)
    ax1.axhline(y=results.apogee_ft, color='r', linestyle='--', alpha=0.7)
    ax1.axhline(y=TARGET_APOGEE_FT, color='g', linestyle=':', alpha=0.7)
    ax1.fill_between(results.time, 0, altitude_ft, alpha=0.2)
    ax1.set_ylabel('Altitude (ft)')
    ax1.set_title(f'Trajectory - Apogee: {results.apogee_ft:.0f} ft')
    ax1.grid(True, alpha=0.3)
    
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(results.time, results.velocity, 'r-', linewidth=2)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Velocity (m/s)')
    ax2.set_title('Velocity Profile')
    ax2.grid(True, alpha=0.3)
    
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(results.time, results.mach, 'g-', linewidth=2)
    ax3.axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Mach Number')
    ax3.set_title('Mach Profile')
    ax3.grid(True, alpha=0.3)
    
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.plot(results.time, results.airbrake_angle, 'b-', linewidth=2)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Angle (deg)')
    ax4.set_title('Airbrake Deployment')
    ax4.set_ylim([0, 100])
    ax4.grid(True, alpha=0.3)
    
    ax5 = fig.add_subplot(gs[2, :2])
    ax5.plot(results.time, results.body_drag, 'b-', linewidth=2, label='Body')
    ax5.plot(results.time, results.airbrake_drag, 'r-', linewidth=2, label='Airbrakes')
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Drag Force (N)')
    ax5.set_title('Drag Forces')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    ax6 = fig.add_subplot(gs[0, 2])
    stats = [
        f"Apogee: {results.apogee_ft:.0f} ft ({results.apogee_m:.0f} m)",
        f"Target: {TARGET_APOGEE_FT:.0f} ft",
        f"Error: {results.apogee_ft - TARGET_APOGEE_FT:+.0f} ft",
        "",
        f"Max Velocity: {results.max_velocity:.1f} m/s",
        f"Max Mach: {results.max_mach:.2f}",
        f"Max Accel: {results.max_acceleration:.1f} m/s²",
        "",
        f"Burnout Time: {results.burnout_time:.2f} s",
        f"Burnout Alt: {results.burnout_altitude * METERS_TO_FEET:.0f} ft",
        f"Burnout Vel: {results.burnout_velocity:.1f} m/s",
        "",
        f"Apogee Time: {results.apogee_time:.2f} s",
    ]
    ax6.text(0.1, 0.95, '\n'.join(stats), transform=ax6.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    ax6.axis('off')
    ax6.set_title('Summary Statistics')
    
    ax7 = fig.add_subplot(gs[2, 2])
    telemetry = results.controller_telemetry
    has_influence = (telemetry and 
                     (len(telemetry.get('drag_influence', [])) > 0 or
                      len(telemetry.get('apogee_clean', [])) > 0))
    if has_influence:
        time = np.array(telemetry['time'])
        if 'drag_influence' in telemetry:
            influence = np.array(telemetry['drag_influence']) * METERS_TO_FEET
        elif 'apogee_clean' in telemetry and 'apogee_full_brake' in telemetry:
            influence = (np.array(telemetry['apogee_clean']) - 
                        np.array(telemetry['apogee_full_brake'])) * METERS_TO_FEET
        else:
            influence = np.zeros_like(time)
        ax7.plot(time, influence, 'purple', linewidth=2)
        ax7.set_xlabel('Time (s)')
        ax7.set_ylabel('Influence (ft)')
        ax7.set_title('Airbrake Influence')
    else:
        ax7.text(0.5, 0.5, 'No telemetry', ha='center', va='center')
        ax7.set_title('Airbrake Influence')
    ax7.grid(True, alpha=0.3)
    
    fig.suptitle('Airbrake Simulation Summary Report', fontsize=16, y=0.98)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
        
    return fig
