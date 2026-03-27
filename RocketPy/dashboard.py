"""
Interactive Plotly/Dash dashboard for airbrake simulation visualization.
"""

import numpy as np
from dash import Dash, html, dcc, callback, Output, Input, State, no_update
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from simulation import SimulationResults
from config import METERS_TO_FEET, TARGET_APOGEE_FT
from simulation_storage import save_simulation, load_simulation, list_simulations


def create_trajectory_figure(results: SimulationResults) -> go.Figure:
    """Create trajectory plot with altitude and velocity."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=('Altitude', 'Velocity')
    )
    
    altitude_ft = np.array(results.altitude) * METERS_TO_FEET
    
    # Altitude
    fig.add_trace(
        go.Scatter(x=results.time, y=altitude_ft, mode='lines',
                   name='Altitude', line=dict(color='blue', width=2)),
        row=1, col=1
    )
    fig.add_hline(y=results.apogee_ft, line_dash="dash", line_color="red",
                  annotation_text=f"Apogee: {results.apogee_ft:.0f} ft", row=1, col=1)
    fig.add_hline(y=TARGET_APOGEE_FT, line_dash="dot", line_color="green",
                  annotation_text=f"Target: {TARGET_APOGEE_FT:.0f} ft", row=1, col=1)
    
    # Velocity
    fig.add_trace(
        go.Scatter(x=results.time, y=results.velocity, mode='lines',
                   name='Velocity', line=dict(color='red', width=2)),
        row=2, col=1
    )
    fig.add_hline(y=0, line_color="gray", line_width=1, row=2, col=1)
    
    if results.burnout_time > 0:
        fig.add_vline(x=results.burnout_time, line_dash="dash", line_color="orange",
                      annotation_text=f"Burnout: {results.burnout_time:.2f}s")
    
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_yaxes(title_text="Altitude (ft)", row=1, col=1)
    fig.update_yaxes(title_text="Velocity (m/s)", row=2, col=1)
    
    fig.update_layout(height=600, showlegend=True, template="plotly_white")
    return fig


def create_airbrake_figure(results: SimulationResults) -> go.Figure:
    """Create airbrake performance plot."""
    burnout_idx = np.searchsorted(results.time, results.burnout_time)
    apogee_idx = np.argmax(results.altitude)
    
    time_slice = results.time[burnout_idx:apogee_idx+1]
    angle_slice = results.airbrake_angle[burnout_idx:apogee_idx+1]
    mach_slice = results.mach[burnout_idx:apogee_idx+1]
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Airbrake Deployment', 'Mach Number', 
                       'Drag Forces', 'Predicted Apogee'),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    # Deployment angle
    fig.add_trace(
        go.Scatter(x=time_slice, y=angle_slice, mode='lines',
                   name='Deployment', line=dict(color='blue', width=2)),
        row=1, col=1
    )
    
    # Mach number
    fig.add_trace(
        go.Scatter(x=time_slice, y=mach_slice, mode='lines',
                   name='Mach', line=dict(color='purple', width=2)),
        row=1, col=2
    )
    fig.add_hline(y=1.0, line_dash="dash", line_color="red",
                  annotation_text="Mach 1", row=1, col=2)
    
    # Drag forces
    body_drag = np.array(results.body_drag)[burnout_idx:apogee_idx+1]
    airbrake_drag = np.array(results.airbrake_drag)[burnout_idx:apogee_idx+1]
    
    fig.add_trace(
        go.Scatter(x=time_slice, y=body_drag, mode='lines',
                   name='Body Drag', line=dict(color='blue', width=2)),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=time_slice, y=airbrake_drag, mode='lines',
                   name='Airbrake Drag', line=dict(color='orange', width=2)),
        row=2, col=1
    )
    
    # Predicted apogee from telemetry
    telemetry = results.controller_telemetry
    if telemetry and len(telemetry.get('time', [])) > 0:
        tel_time = np.array(telemetry['time'])
        pred_apogee = np.array(telemetry['predicted_apogee']) * METERS_TO_FEET
        
        mask = (tel_time >= results.burnout_time) & (tel_time <= results.apogee_time)
        fig.add_trace(
            go.Scatter(x=tel_time[mask], y=pred_apogee[mask], mode='lines',
                       name='Predicted Apogee', line=dict(color='green', width=2)),
            row=2, col=2
        )
        fig.add_hline(y=TARGET_APOGEE_FT, line_dash="dash", line_color="red",
                      annotation_text=f"Target", row=2, col=2)
    
    fig.update_xaxes(title_text="Time (s)", row=2)
    fig.update_yaxes(title_text="Angle (°)", row=1, col=1)
    fig.update_yaxes(title_text="Mach", row=1, col=2)
    fig.update_yaxes(title_text="Force (N)", row=2, col=1)
    fig.update_yaxes(title_text="Altitude (ft)", row=2, col=2)
    
    fig.update_layout(height=700, showlegend=True, template="plotly_white")
    return fig


def create_control_figure(results: SimulationResults) -> go.Figure:
    """Create control system telemetry plot."""
    telemetry = results.controller_telemetry
    
    if not telemetry or len(telemetry.get('time', [])) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No telemetry data available",
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    time = np.array(telemetry['time'])
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('State Estimation', 'Cd Command', 
                       'Apogee Predictions', 'Airbrake Influence'),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    # State estimation
    alt_est = np.array(telemetry['altitude_est']) * METERS_TO_FEET
    alt_meas = np.array(telemetry['altitude_meas']) * METERS_TO_FEET
    
    fig.add_trace(
        go.Scatter(x=time, y=alt_est, mode='lines',
                   name='Estimated Alt', line=dict(color='blue', width=2)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=time, y=alt_meas, mode='lines',
                   name='Measured Alt', line=dict(color='gray', width=1, dash='dot')),
        row=1, col=1
    )
    
    # Cd command
    if 'cd_add_cmd' in telemetry:
        cd_cmd = np.array(telemetry['cd_add_cmd'])
        fig.add_trace(
            go.Scatter(x=time, y=cd_cmd, mode='lines',
                       name='Cd Command', line=dict(color='orange', width=2)),
            row=1, col=2
        )
    
    # Apogee predictions - filter to post-burnout only
    burnout_time = results.burnout_time if hasattr(results, 'burnout_time') else 0
    
    # Truth-based predictions (from simulation model)
    if len(results.truth_pred_time) > 0:
        truth_time = results.truth_pred_time
        truth_retracted = results.truth_apogee_retracted * METERS_TO_FEET
        truth_extended = results.truth_apogee_extended * METERS_TO_FEET
        truth_current = results.truth_apogee_current * METERS_TO_FEET
        
        # Retracted (no brakes) - upper bound
        fig.add_trace(
            go.Scatter(x=truth_time, y=truth_retracted, mode='lines',
                       name='Truth: Retracted', line=dict(color='blue', width=2)),
            row=2, col=1
        )
        
        # Extended (full brakes) - lower bound  
        fig.add_trace(
            go.Scatter(x=truth_time, y=truth_extended, mode='lines',
                       name='Truth: Extended', line=dict(color='red', width=2)),
            row=2, col=1
        )
        
        # Current prediction based on actual airbrake state
        fig.add_trace(
            go.Scatter(x=truth_time, y=truth_current, mode='lines',
                       name='Truth: Current', line=dict(color='green', width=2)),
            row=2, col=1
        )
    
    # Control system's predicted apogee (overlay)
    # Filter to post-burnout
    post_burnout_mask = time >= burnout_time
    ctrl_time = time[post_burnout_mask]
    ctrl_pred = np.array(telemetry['predicted_apogee'])[post_burnout_mask] * METERS_TO_FEET
    
    if len(ctrl_time) > 0:
        fig.add_trace(
            go.Scatter(x=ctrl_time, y=ctrl_pred, mode='lines',
                       name='Ctrl Predicted', line=dict(color='orange', width=2, dash='dash')),
            row=2, col=1
        )
    
    fig.add_hline(y=TARGET_APOGEE_FT, line_dash="dot", line_color="gray", row=2, col=1)
    
    # Airbrake influence (truth-based)
    if len(results.truth_pred_time) > 0:
        influence = (results.truth_apogee_retracted - results.truth_apogee_extended) * METERS_TO_FEET
        fig.add_trace(
            go.Scatter(x=results.truth_pred_time, y=influence, mode='lines',
                       name='Influence', line=dict(color='purple', width=2)),
            row=2, col=2
        )
    
    fig.update_xaxes(title_text="Time (s)", row=2)
    fig.update_yaxes(title_text="Altitude (ft)", row=1, col=1)
    fig.update_yaxes(title_text="Cd", row=1, col=2)
    fig.update_yaxes(title_text="Apogee (ft)", row=2, col=1)
    fig.update_yaxes(title_text="Influence (ft)", row=2, col=2)
    
    fig.update_layout(height=700, showlegend=True, template="plotly_white")
    return fig


def create_summary_figure(results: SimulationResults, sim_config=None, lut_config=None) -> go.Figure:
    """Create summary metrics display with configuration info."""
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "table"}, {"type": "table"}]],
        subplot_titles=('Flight Results', 'Model Configuration')
    )
    
    error_ft = results.apogee_ft - TARGET_APOGEE_FT
    error_pct = (error_ft / TARGET_APOGEE_FT) * 100
    
    # Flight results table
    metrics = [
        ['Apogee', f'{results.apogee_ft:.0f} ft'],
        ['Target', f'{TARGET_APOGEE_FT:.0f} ft'],
        ['Error', f'{error_ft:+.0f} ft ({error_pct:+.2f}%)'],
        ['Max Velocity', f'{results.max_velocity:.1f} m/s'],
        ['Max Mach', f'{results.max_mach:.2f}'],
        ['Max Accel', f'{results.max_acceleration:.1f} m/s² ({results.max_acceleration/9.81:.1f} g)'],
        ['Burnout Time', f'{results.burnout_time:.2f} s'],
        ['Apogee Time', f'{results.apogee_time:.2f} s'],
    ]
    
    fig.add_trace(go.Table(
        header=dict(values=['Metric', 'Value'],
                    fill_color='lightblue',
                    align='left',
                    font=dict(size=14)),
        cells=dict(values=[[m[0] for m in metrics], [m[1] for m in metrics]],
                   fill_color='white',
                   align='left',
                   font=dict(size=13),
                   height=30)
    ), row=1, col=1)
    
    # Configuration table
    config_rows = [
        ['--- SIMULATION ENV ---', ''],
    ]
    
    if sim_config:
        config_rows.extend([
            ['Cd Scale (M0/M2)', f'{sim_config.cd_scale_mach0:.2f} / {sim_config.cd_scale_mach2:.2f}'],
            ['Airbrake Cd (M0/M2)', f'{sim_config.airbrake_cd_scale_mach0:.2f} / {sim_config.airbrake_cd_scale_mach2:.2f}'],
            ['Thrust Scale', f'{sim_config.thrust_scale:.1%}'],
            ['Launch Alt Offset', f'{sim_config.launch_altitude_offset_m:+.0f} m'],
            ['Temp Offset', f'{sim_config.launch_temp_offset_k:+.1f} K'],
            ['Airbrake Slew', f'{sim_config.airbrake_slew_rate_deg_s:.0f} deg/s'],
            ['Airbrake Area', f'{sim_config.airbrake_max_area_m2*10000:.1f} cm²'],
        ])
    else:
        config_rows.extend([
            ['Cd Scale (M0/M2)', '1.00 / 1.00'],
            ['Airbrake Cd (M0/M2)', '1.00 / 1.00'],
            ['Thrust Scale', '100%'],
            ['Launch Alt Offset', '+0 m'],
            ['Temp Offset', '+0.0 K'],
            ['Airbrake Slew', '180 deg/s'],
            ['Airbrake Area', '60.0 cm²'],
        ])
    
    config_rows.append(['', ''])
    config_rows.append(['--- CONTROL LUTs ---', ''])
    
    if lut_config:
        config_rows.extend([
            ['Cd Resolution', f'{lut_config.cd_resolution} pts'],
            ['Density Resolution', f'{lut_config.density_resolution} pts'],
            ['Mass Resolution', f'{lut_config.mass_resolution} pts'],
        ])
    else:
        config_rows.extend([
            ['Cd Resolution', '50 pts'],
            ['Density Resolution', '100 pts'],
            ['Mass Resolution', '20 pts'],
        ])
    
    fig.add_trace(go.Table(
        header=dict(values=['Parameter', 'Value'],
                    fill_color='lightyellow',
                    align='left',
                    font=dict(size=14)),
        cells=dict(values=[[c[0] for c in config_rows], [c[1] for c in config_rows]],
                   fill_color='white',
                   align='left',
                   font=dict(size=12),
                   height=25)
    ), row=1, col=2)
    
    fig.update_layout(height=450, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def create_model_comparison_figure(results: SimulationResults, sim_config=None, lut_config=None) -> go.Figure:
    """
    Create figure comparing three models:
    - Truth Model (best guess of actual physics)
    - Simulation Model (truth + scaling for real-world uncertainty)
    - Control Model (sparse LUTs from truth)
    """
    from truth_model import get_truth_model, SimulationModel, ControlModel, TruthModelConfig, ControlLUTConfig
    
    telemetry = results.controller_telemetry
    
    if not telemetry or len(telemetry.get('time', [])) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No telemetry data available",
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Get models
    truth = get_truth_model()
    sim_model = SimulationModel(truth, sim_config or TruthModelConfig())
    ctrl_model = ControlModel(truth, lut_config or ControlLUTConfig())
    
    tel_time = np.array(telemetry['time'])
    tel_alt = np.array(telemetry['altitude_est'])
    mach_est = np.array(telemetry['mach_est'])
    sim_time = np.array(results.time)
    
    fig = make_subplots(
        rows=5, cols=2,
        subplot_titles=(
            'Altitude (m)', 'Velocity (m/s)',
            'Air Density (kg/m³)', 'Mach Number',
            'Airframe Cd', 'Airbrake Cd',
            'Acceleration (m/s²)', 'Model Errors vs Truth (%)',
            'Airframe Cd vs Mach', 'Airbrake Cd vs Mach'
        ),
        vertical_spacing=0.07,
        horizontal_spacing=0.12
    )
    
    # --- Altitude Comparison (Row 1, Col 1) ---
    sim_alt_interp = np.interp(tel_time, sim_time, results.altitude)
    ctrl_alt = np.array(telemetry['altitude_est'])
    
    fig.add_trace(
        go.Scatter(x=tel_time, y=sim_alt_interp, mode='lines',
                   name='Sim Alt', line=dict(color='blue', width=2)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=tel_time, y=ctrl_alt, mode='lines',
                   name='Ctrl Alt', line=dict(color='red', width=2, dash='dash')),
        row=1, col=1
    )
    
    # --- Velocity Comparison (Row 1, Col 2) ---
    sim_vel_interp = np.interp(tel_time, sim_time, results.velocity)
    ctrl_vel = np.array(telemetry['velocity_est'])
    
    fig.add_trace(
        go.Scatter(x=tel_time, y=sim_vel_interp, mode='lines',
                   name='Sim Vel', line=dict(color='blue', width=2)),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=tel_time, y=ctrl_vel, mode='lines',
                   name='Ctrl Vel', line=dict(color='red', width=2, dash='dash')),
        row=1, col=2
    )
    
    # --- Air Density (Row 2, Col 1) ---
    # Compute density from all three models
    truth_density = np.array([truth.get_density(alt) for alt in tel_alt])
    sim_density = np.array([sim_model.get_density(alt) for alt in tel_alt])
    ctrl_density = np.array(telemetry['ctrl_density']) if 'ctrl_density' in telemetry else \
                   np.array([ctrl_model.get_density(alt) for alt in tel_alt])
    
    fig.add_trace(
        go.Scatter(x=tel_time, y=truth_density, mode='lines',
                   name='Truth ρ', line=dict(color='green', width=2)),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=tel_time, y=sim_density, mode='lines',
                   name='Sim ρ', line=dict(color='blue', width=2, dash='dash')),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=tel_time, y=ctrl_density, mode='lines',
                   name='Ctrl ρ', line=dict(color='red', width=2, dash='dot')),
        row=2, col=1
    )
    
    # --- Mach Number (Row 2, Col 2) ---
    sim_mach_interp = np.interp(tel_time, sim_time, results.mach)
    
    fig.add_trace(
        go.Scatter(x=tel_time, y=sim_mach_interp, mode='lines',
                   name='Sim Mach', line=dict(color='blue', width=2)),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(x=tel_time, y=mach_est, mode='lines',
                   name='Ctrl Mach', line=dict(color='red', width=2, dash='dash')),
        row=2, col=2
    )
    fig.add_hline(y=1.0, line_color="orange", line_dash="dot", 
                  annotation_text="Mach 1", row=2, col=2)
    
    # --- Airframe Cd (Row 3, Col 1) ---
    # Compute airframe Cd from all three models at estimated Mach
    truth_cd = np.array([truth.get_cd(m) for m in mach_est])
    sim_cd = np.array([sim_model.get_cd(m) for m in mach_est])
    ctrl_cd = np.array(telemetry['ctrl_cd_body']) if 'ctrl_cd_body' in telemetry else \
              np.array([ctrl_model.get_cd(m) for m in mach_est])
    
    fig.add_trace(
        go.Scatter(x=tel_time, y=truth_cd, mode='lines',
                   name='Truth Cd_body', line=dict(color='green', width=2)),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(x=tel_time, y=sim_cd, mode='lines',
                   name='Sim Cd_body', line=dict(color='blue', width=2, dash='dash')),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(x=tel_time, y=ctrl_cd, mode='lines',
                   name='Ctrl Cd_body', line=dict(color='red', width=2, dash='dot')),
        row=3, col=1
    )
    
    # --- Airbrake Cd (Row 3, Col 2) ---
    # Show actual airbrake Cd values from both sim and control
    # Sim: custom CSV or scaled baseline
    # Ctrl: fixed AIRBRAKE_CD (1.28 default)
    from config import AIRBRAKE_CD
    
    sim_ab_cd = np.array([sim_model.get_airbrake_cd(m) for m in mach_est])
    ctrl_ab_cd = np.full_like(mach_est, AIRBRAKE_CD)  # Control uses fixed value
    
    fig.add_trace(
        go.Scatter(x=tel_time, y=sim_ab_cd, mode='lines',
                   name='Sim AB Cd', line=dict(color='blue', width=2)),
        row=3, col=2
    )
    fig.add_trace(
        go.Scatter(x=tel_time, y=ctrl_ab_cd, mode='lines',
                   name='Ctrl AB Cd', line=dict(color='red', width=2, dash='dash')),
        row=3, col=2
    )
    
    # --- Acceleration (Row 4, Col 1) ---
    # Filter to post-burnout only
    if 'ctrl_accel' in telemetry:
        burnout_time = results.burnout_time if hasattr(results, 'burnout_time') else 0
        post_burnout_mask = tel_time >= burnout_time
        
        accel_time = tel_time[post_burnout_mask]
        ctrl_accel = np.array(telemetry['ctrl_accel'])[post_burnout_mask]
        sim_accel_interp = np.interp(tel_time, sim_time, results.acceleration)[post_burnout_mask]
        
        if len(accel_time) > 0:
            fig.add_trace(
                go.Scatter(x=accel_time, y=sim_accel_interp, mode='lines',
                           name='Sim Accel', line=dict(color='blue', width=2)),
                row=4, col=1
            )
            fig.add_trace(
                go.Scatter(x=accel_time, y=ctrl_accel, mode='lines',
                           name='Ctrl Accel', line=dict(color='red', width=2, dash='dash')),
                row=4, col=1
            )
    
    # --- Model Errors vs Truth (Row 4, Col 2) ---
    # Show how much each model deviates from truth
    sim_cd_error = (sim_cd - truth_cd) / truth_cd * 100
    ctrl_cd_error = (ctrl_cd - truth_cd) / truth_cd * 100
    sim_rho_error = (sim_density - truth_density) / truth_density * 100
    ctrl_rho_error = (ctrl_density - truth_density) / truth_density * 100
    
    fig.add_trace(
        go.Scatter(x=tel_time, y=sim_cd_error, mode='lines',
                   name='Sim Cd err', line=dict(color='blue', width=2)),
        row=4, col=2
    )
    fig.add_trace(
        go.Scatter(x=tel_time, y=ctrl_cd_error, mode='lines',
                   name='Ctrl Cd err', line=dict(color='red', width=2, dash='dash')),
        row=4, col=2
    )
    fig.add_trace(
        go.Scatter(x=tel_time, y=sim_rho_error, mode='lines',
                   name='Sim ρ err', line=dict(color='cyan', width=1)),
        row=4, col=2
    )
    fig.add_trace(
        go.Scatter(x=tel_time, y=ctrl_rho_error, mode='lines',
                   name='Ctrl ρ err', line=dict(color='orange', width=1, dash='dash')),
        row=4, col=2
    )
    fig.add_hline(y=0, line_color="gray", line_dash="dash", row=4, col=2)
    
    # --- Airframe Cd vs Mach (Row 5, Col 1) ---
    mach_range = np.linspace(0, 2.0, 100)
    truth_cd_curve = np.array([truth.get_cd(m) for m in mach_range])
    sim_cd_curve = np.array([sim_model.get_cd(m) for m in mach_range])
    ctrl_cd_curve = np.array([ctrl_model.get_cd(m) for m in mach_range])
    
    fig.add_trace(
        go.Scatter(x=mach_range, y=truth_cd_curve, mode='lines',
                   name='Truth Cd(M)', line=dict(color='green', width=2)),
        row=5, col=1
    )
    fig.add_trace(
        go.Scatter(x=mach_range, y=sim_cd_curve, mode='lines',
                   name='Sim Cd(M)', line=dict(color='blue', width=2, dash='dash')),
        row=5, col=1
    )
    fig.add_trace(
        go.Scatter(x=mach_range, y=ctrl_cd_curve, mode='lines',
                   name='Ctrl Cd(M)', line=dict(color='red', width=2, dash='dot')),
        row=5, col=1
    )
    fig.add_vline(x=1.0, line_color="orange", line_dash="dot", row=5, col=1)
    
    # --- Airbrake Cd vs Mach (Row 5, Col 2) ---
    sim_ab_cd_curve = np.array([sim_model.get_airbrake_cd(m) for m in mach_range])
    ctrl_ab_cd_curve = np.full_like(mach_range, AIRBRAKE_CD)  # Control uses fixed value
    
    fig.add_trace(
        go.Scatter(x=mach_range, y=sim_ab_cd_curve, mode='lines',
                   name='Sim AB Cd(M)', line=dict(color='blue', width=2)),
        row=5, col=2
    )
    fig.add_trace(
        go.Scatter(x=mach_range, y=ctrl_ab_cd_curve, mode='lines',
                   name='Ctrl AB Cd(M)', line=dict(color='red', width=2, dash='dash')),
        row=5, col=2
    )
    fig.add_vline(x=1.0, line_color="orange", line_dash="dot", row=5, col=2)
    
    # Update axes labels
    fig.update_xaxes(title_text="Time (s)", row=5, col=1)
    fig.update_xaxes(title_text="Mach", row=5, col=2)
    
    fig.update_yaxes(title_text="Altitude (m)", row=1, col=1)
    fig.update_yaxes(title_text="Velocity (m/s)", row=1, col=2)
    fig.update_yaxes(title_text="Density (kg/m³)", row=2, col=1)
    fig.update_yaxes(title_text="Mach", row=2, col=2)
    fig.update_yaxes(title_text="Cd", row=3, col=1)
    fig.update_yaxes(title_text="Cd", row=3, col=2)
    fig.update_yaxes(title_text="Accel (m/s²)", row=4, col=1)
    fig.update_yaxes(title_text="Error (%)", row=4, col=2)
    fig.update_yaxes(title_text="Cd", row=5, col=1)
    fig.update_yaxes(title_text="AB Cd", row=5, col=2)
    
    fig.update_layout(
        height=1400, 
        showlegend=True, 
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="center", x=0.5),
        margin=dict(l=60, r=40, t=80, b=40)
    )
    return fig


def create_sensor_noise_figure(results: SimulationResults) -> go.Figure:
    """Create sensor noise comparison plot showing true vs noisy readings and state estimates."""
    fig = make_subplots(
        rows=5, cols=2,
        subplot_titles=(
            'Pressure: True vs Noisy', 'Pressure Error',
            'Temperature: True vs Noisy', 'Temperature Error',
            'Acceleration: True vs Noisy', 'Acceleration Error',
            'Altitude: True vs Estimate', 'Altitude Error',
            'Velocity: True vs Estimate', 'Velocity Error'
        ),
        vertical_spacing=0.06,
        horizontal_spacing=0.08
    )
    
    # Check if sensor data is available
    if len(results.sensor_time) == 0:
        fig.add_annotation(
            text="No sensor noise data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        return fig
    
    t = results.sensor_time
    
    # --- Pressure (Row 1) ---
    p_true = results.sensor_pressure_true
    p_noisy = results.sensor_pressure_noisy
    p_error = p_noisy - p_true
    
    fig.add_trace(
        go.Scatter(x=t, y=p_true, mode='lines',
                   name='Pressure (True)', line=dict(color='blue', width=2)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=t, y=p_noisy, mode='lines',
                   name='Pressure (Noisy)', line=dict(color='red', width=1, dash='dot')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=t, y=p_error, mode='lines',
                   name='P Error', line=dict(color='purple', width=1)),
        row=1, col=2
    )
    fig.add_hline(y=0, line_color="gray", line_dash="dash", row=1, col=2)
    
    # --- Temperature (Row 2) ---
    temp_true = results.sensor_temp_true
    temp_noisy = results.sensor_temp_noisy
    temp_error = temp_noisy - temp_true
    
    fig.add_trace(
        go.Scatter(x=t, y=temp_true, mode='lines',
                   name='Temp (True)', line=dict(color='blue', width=2)),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=t, y=temp_noisy, mode='lines',
                   name='Temp (Noisy)', line=dict(color='red', width=1, dash='dot')),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=t, y=temp_error, mode='lines',
                   name='T Error', line=dict(color='purple', width=1)),
        row=2, col=2
    )
    fig.add_hline(y=0, line_color="gray", line_dash="dash", row=2, col=2)
    
    # --- Acceleration (Row 3) ---
    accel_true = results.sensor_accel_true
    accel_noisy = results.sensor_accel_noisy
    accel_error = accel_noisy - accel_true
    
    fig.add_trace(
        go.Scatter(x=t, y=accel_true, mode='lines',
                   name='Accel (True)', line=dict(color='blue', width=2)),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(x=t, y=accel_noisy, mode='lines',
                   name='Accel (Noisy)', line=dict(color='red', width=1, dash='dot')),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(x=t, y=accel_error, mode='lines',
                   name='A Error', line=dict(color='purple', width=1)),
        row=3, col=2
    )
    fig.add_hline(y=0, line_color="gray", line_dash="dash", row=3, col=2)
    
    # --- Altitude Estimate vs True (Row 4) ---
    telemetry = results.controller_telemetry
    if 'altitude_est' in telemetry and 'time' in telemetry:
        tel_time = np.array(telemetry['time'])
        alt_est = np.array(telemetry['altitude_est'])
        # Interpolate true altitude to telemetry timestamps
        alt_true = np.interp(tel_time, results.time, results.altitude)
        alt_error = alt_est - alt_true
        
        fig.add_trace(
            go.Scatter(x=tel_time, y=alt_true, mode='lines',
                       name='Altitude (True)', line=dict(color='blue', width=2)),
            row=4, col=1
        )
        fig.add_trace(
            go.Scatter(x=tel_time, y=alt_est, mode='lines',
                       name='Altitude (Est)', line=dict(color='red', width=1, dash='dot')),
            row=4, col=1
        )
        fig.add_trace(
            go.Scatter(x=tel_time, y=alt_error, mode='lines',
                       name='Alt Error', line=dict(color='purple', width=1)),
            row=4, col=2
        )
        fig.add_hline(y=0, line_color="gray", line_dash="dash", row=4, col=2)
    
    # --- Velocity Estimate vs True (Row 5) ---
    if 'velocity_est' in telemetry and 'time' in telemetry:
        tel_time = np.array(telemetry['time'])
        vel_est = np.array(telemetry['velocity_est'])
        # Interpolate true velocity to telemetry timestamps
        vel_true = np.interp(tel_time, results.time, results.velocity)
        vel_error = vel_est - vel_true
        
        fig.add_trace(
            go.Scatter(x=tel_time, y=vel_true, mode='lines',
                       name='Velocity (True)', line=dict(color='blue', width=2)),
            row=5, col=1
        )
        fig.add_trace(
            go.Scatter(x=tel_time, y=vel_est, mode='lines',
                       name='Velocity (Est)', line=dict(color='red', width=1, dash='dot')),
            row=5, col=1
        )
        fig.add_trace(
            go.Scatter(x=tel_time, y=vel_error, mode='lines',
                       name='Vel Error', line=dict(color='purple', width=1)),
            row=5, col=2
        )
        fig.add_hline(y=0, line_color="gray", line_dash="dash", row=5, col=2)
    
    # Labels
    fig.update_xaxes(title_text="Time (s)", row=5)
    fig.update_yaxes(title_text="Pressure (Pa)", row=1, col=1)
    fig.update_yaxes(title_text="Error (Pa)", row=1, col=2)
    fig.update_yaxes(title_text="Temp (K)", row=2, col=1)
    fig.update_yaxes(title_text="Error (K)", row=2, col=2)
    fig.update_yaxes(title_text="Accel (m/s²)", row=3, col=1)
    fig.update_yaxes(title_text="Error (m/s²)", row=3, col=2)
    fig.update_yaxes(title_text="Altitude (m)", row=4, col=1)
    fig.update_yaxes(title_text="Error (m)", row=4, col=2)
    fig.update_yaxes(title_text="Velocity (m/s)", row=5, col=1)
    fig.update_yaxes(title_text="Error (m/s)", row=5, col=2)
    
    fig.update_layout(
        height=1200, 
        showlegend=True, 
        template="plotly_white",
        title_text="Sensor Noise & State Estimation"
    )
    return fig


def run_dashboard(results: SimulationResults, sim_config=None, lut_config=None, port: int = 8050):
    """
    Launch interactive Dash dashboard.
    
    Args:
        results: SimulationResults to visualize
        sim_config: TruthModelConfig for simulation scaling
        lut_config: ControlLUTConfig for control system LUTs
        port: Port to run dashboard on
    """
    app = Dash(__name__, suppress_callback_exceptions=True)
    
    # Store for current results and configs
    app.current_results = results
    app.sim_config = sim_config
    app.lut_config = lut_config
    
    def get_simulation_options():
        """Get dropdown options for saved simulations."""
        sims = list_simulations()
        options = [{'label': '-- Current Simulation --', 'value': 'current'}]
        for sim in sims:
            label = f"{sim.get('name', sim['id'])} - {sim['apogee_ft']:.0f} ft ({sim['timestamp'][:10]})"
            options.append({'label': label, 'value': sim['id']})
        return options
    
    def build_layout():
        """Build the dashboard layout."""
        current_results = app.current_results
        error_ft = current_results.apogee_ft - TARGET_APOGEE_FT
        
        return html.Div([
            html.H1('Airbrake Simulation Dashboard', 
                    style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '10px'}),
            
            # Simulation selector and save controls
            html.Div([
                html.Div([
                    html.Label('Load Simulation:', style={'fontWeight': 'bold', 'marginRight': '10px'}),
                    dcc.Dropdown(
                        id='simulation-selector',
                        options=get_simulation_options(),
                        value='current',
                        style={'width': '400px', 'display': 'inline-block', 'verticalAlign': 'middle'},
                        clearable=False,
                    ),
                    html.Button('🔄 Refresh List', id='refresh-btn', n_clicks=0,
                                style={'marginLeft': '10px', 'padding': '8px 16px', 'cursor': 'pointer'}),
                ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '10px'}),
                
                html.Div([
                    html.Label('Save Current:', style={'fontWeight': 'bold', 'marginRight': '10px'}),
                    dcc.Input(
                        id='save-name-input',
                        type='text',
                        placeholder='Enter simulation name...',
                        style={'width': '300px', 'padding': '8px', 'marginRight': '10px'},
                    ),
                    html.Button('💾 Save Simulation', id='save-btn', n_clicks=0,
                                style={'padding': '8px 16px', 'cursor': 'pointer', 'backgroundColor': '#27ae60', 'color': 'white', 'border': 'none', 'borderRadius': '4px'}),
                    html.Span(id='save-status', style={'marginLeft': '10px', 'color': '#27ae60'}),
                ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '20px'}),
            ], style={'backgroundColor': '#f8f9fa', 'padding': '15px', 'borderRadius': '8px', 'marginBottom': '20px'}),
            
            # Results header
            html.Div([
                html.Div([
                    html.H3(id='apogee-display', children=f'Apogee: {current_results.apogee_ft:.0f} ft', 
                            style={'color': '#27ae60' if abs(error_ft) < 100 else '#e74c3c'}),
                    html.P(id='error-display', children=f'Error: {error_ft:+.0f} ft from target'),
                ], style={'textAlign': 'center', 'marginBottom': '20px'}),
            ]),
            
            # Graph tabs
            dcc.Tabs([
                dcc.Tab(label='Trajectory', children=[
                    dcc.Graph(id='trajectory-graph', figure=create_trajectory_figure(current_results))
                ]),
                dcc.Tab(label='Airbrake Performance', children=[
                    dcc.Graph(id='airbrake-graph', figure=create_airbrake_figure(current_results))
                ]),
                dcc.Tab(label='Control System', children=[
                    dcc.Graph(id='control-graph', figure=create_control_figure(current_results))
                ]),
                dcc.Tab(label='Model Comparison', children=[
                    html.P("Green=Truth (best guess), Blue=Simulation (scaled), Red=Control (sparse LUTs)",
                           style={'textAlign': 'center', 'color': '#666', 'marginTop': '10px'}),
                    dcc.Graph(id='model-graph', figure=create_model_comparison_figure(current_results, app.sim_config, app.lut_config))
                ]),
                dcc.Tab(label='Summary', children=[
                    dcc.Graph(id='summary-graph', figure=create_summary_figure(current_results, app.sim_config, app.lut_config))
                ]),
                dcc.Tab(label='Sensor Noise', children=[
                    html.P("Blue=True sensor value, Red=Noisy measurement, Purple=Error",
                           style={'textAlign': 'center', 'color': '#666', 'marginTop': '10px'}),
                    dcc.Graph(id='sensor-noise-graph', figure=create_sensor_noise_figure(current_results))
                ]),
            ]),
            
            # Hidden store for triggering updates
            dcc.Store(id='selected-sim-store', data='current'),
        ], style={'fontFamily': 'Arial, sans-serif', 'padding': '20px'})
    
    app.layout = build_layout
    
    @app.callback(
        Output('simulation-selector', 'options'),
        Input('refresh-btn', 'n_clicks'),
        Input('save-status', 'children'),
    )
    def refresh_simulation_list(n_clicks, save_status):
        """Refresh the simulation dropdown options."""
        return get_simulation_options()
    
    @app.callback(
        Output('save-status', 'children'),
        Input('save-btn', 'n_clicks'),
        State('save-name-input', 'value'),
        prevent_initial_call=True,
    )
    def save_current_simulation(n_clicks, name):
        """Save the current simulation."""
        if n_clicks > 0:
            config_dict = None
            if app.sim_config:
                config_dict = {
                    'cd_scale_mach0': app.sim_config.cd_scale_mach0,
                    'cd_scale_mach2': app.sim_config.cd_scale_mach2,
                    'airbrake_cd_scale_mach0': app.sim_config.airbrake_cd_scale_mach0,
                    'airbrake_cd_scale_mach2': app.sim_config.airbrake_cd_scale_mach2,
                    'thrust_scale': app.sim_config.thrust_scale,
                    'launch_altitude_offset_m': app.sim_config.launch_altitude_offset_m,
                    'launch_temp_offset_k': app.sim_config.launch_temp_offset_k,
                    'airbrake_slew_rate_deg_s': app.sim_config.airbrake_slew_rate_deg_s,
                    'airbrake_max_area_m2': app.sim_config.airbrake_max_area_m2,
                }
            sim_id = save_simulation(app.current_results, name=name, config=config_dict)
            return f'✓ Saved as {sim_id}'
        return ''
    
    @app.callback(
        [
            Output('trajectory-graph', 'figure'),
            Output('airbrake-graph', 'figure'),
            Output('control-graph', 'figure'),
            Output('model-graph', 'figure'),
            Output('summary-graph', 'figure'),
            Output('apogee-display', 'children'),
            Output('apogee-display', 'style'),
            Output('error-display', 'children'),
        ],
        Input('simulation-selector', 'value'),
    )
    def load_selected_simulation(sim_id):
        """Load and display the selected simulation."""
        if sim_id == 'current' or sim_id is None:
            results = app.current_results
            s_config = app.sim_config
            l_config = app.lut_config
        else:
            loaded = load_simulation(sim_id)
            if loaded is None:
                return [no_update] * 8
            results = loaded
            s_config = None
            l_config = None
        
        error_ft = results.apogee_ft - TARGET_APOGEE_FT
        apogee_style = {'color': '#27ae60' if abs(error_ft) < 100 else '#e74c3c'}
        
        return (
            create_trajectory_figure(results),
            create_airbrake_figure(results),
            create_control_figure(results),
            create_model_comparison_figure(results, s_config, l_config),
            create_summary_figure(results, s_config, l_config),
            f'Apogee: {results.apogee_ft:.0f} ft',
            apogee_style,
            f'Error: {error_ft:+.0f} ft from target',
        )
    
    print(f"\n📊 Dashboard running at http://localhost:{port}")
    print("   Press Ctrl+C to stop\n")
    
    app.run(debug=False, port=port)


def show_figures(results: SimulationResults):
    """
    Show all figures in browser without running server.
    Opens each figure in a new browser tab.
    """
    import plotly.io as pio
    
    figs = [
        ('Trajectory', create_trajectory_figure(results)),
        ('Airbrake Performance', create_airbrake_figure(results)),
        ('Control System', create_control_figure(results)),
    ]
    
    for name, fig in figs:
        fig.update_layout(title=name)
        fig.show()
