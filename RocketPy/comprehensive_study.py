#!/usr/bin/env python3
"""
Comprehensive Monte Carlo Simulation Study
Stanford Space Initiative - IREC Rocket Airbrake System

Generates all data and figures for the research paper.
Runs ~15,000+ simulations across 16 studies.
"""

import sys, os, time as clock, types, json, multiprocessing
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm
import scipy.stats as stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulation import create_simulation, SimulationEnvironment, SimulationResults
from truth_model import get_truth_model, SimulationModel, SimulationConfig, ControlLUTConfig
from config import (TARGET_APOGEE_M, METERS_TO_FEET, GRAVITY, AIRBRAKE_CD,
                    AIRBRAKE_MAX_AREA_M2, ROCKET_REFERENCE_AREA_M2,
                    PRESSURE_NOISE_STD, TEMPERATURE_NOISE_STD, ACCEL_Z_NOISE_STD,
                    AIRBRAKE_SLEW_RATE_DEG_S, LAUNCH_ALTITUDE_M)
from debug import configure_debug, DebugLevel
from sensors import SensorModel

configure_debug(level=DebugLevel.OFF)

NUM_WORKERS = 8  # Use 8 of 10 cores

FDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'study_output', 'figures')
DDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'study_output', 'data')
os.makedirs(FDIR, exist_ok=True)
os.makedirs(DDIR, exist_ok=True)

TARGET_FT = TARGET_APOGEE_M * METERS_TO_FEET
M2F = METERS_TO_FEET

# Figure style
plt.rcParams.update({
    'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 11,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 8,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.grid': True, 'grid.alpha': 0.3, 'lines.linewidth': 1.5,
})
C = ['#2196F3', '#FF5722', '#4CAF50', '#FFC107', '#9C27B0',
     '#00BCD4', '#FF9800', '#795548', '#607D8B', '#E91E63',
     '#3F51B5', '#8BC34A', '#CDDC39', '#009688', '#F44336']

# Global results storage for paper
ALL_STATS = {}

# =============================================================================
# Helpers
# =============================================================================
def pbar(i, n, label=''):
    if i % max(1, n//50) == 0 or i == n-1:
        pct = (i+1)/n*100
        print(f'\r  {label} [{pct:5.1f}%] {i+1}/{n}', end='', flush=True)
    if i == n-1: print()

def _init_worker():
    """Initialize worker process: suppress debug, reset truth model."""
    import debug
    debug.configure_debug(level=debug.DebugLevel.OFF)
    from truth_model import reset_truth_model
    reset_truth_model()

def _run_single(cfg):
    """Run a single sim (top-level for multiprocessing pickle).
    Uses relaxed timesteps for speed (10ft accuracy vs 1ms baseline)."""
    sim = create_simulation(**cfg)
    sim._record_truth_predictions = lambda: None  # Skip expensive truth preds
    sim.dt = 0.005          # 5ms physics (vs 1ms default)
    sim.control_dt = 0.05   # 50ms control (vs 10ms default)
    r = sim.run()
    return (r.apogee_m, r.apogee_ft, r.max_velocity, r.max_mach,
            r.burnout_altitude, r.burnout_velocity, r.apogee_time)

def _run_wind_single(args):
    """Run a single wind sim (top-level for multiprocessing)."""
    cfg, wind = args
    sim = create_simulation(**cfg)
    sim._record_truth_predictions = lambda: None
    sim.dt = 0.005; sim.control_dt = 0.05
    w = wind
    def make_cf(s, wv):
        def _cf():
            thrust = s.sim_model.get_thrust(s.state.time)
            bcd = s.sim_model.get_cd(s.state.mach)
            dens = s.sim_model.get_density(s.state.altitude)
            vr = s.state.velocity - wv
            bd = 0.5 * dens * vr * abs(vr) * bcd * s.rocket.get_reference_area()
            if s.enable_airbrakes and s.airbrake.is_deployed:
                acd = s.sim_model.get_airbrake_cd(s.state.mach)
                ad = 0.5 * dens * vr * abs(vr) * acd * s.airbrake.get_area()
            else:
                ad = 0.0
            return thrust, abs(bd), abs(ad)
        return _cf
    sim._compute_forces = make_cf(sim, w)
    r = sim.run()
    return (r.apogee_m, r.apogee_ft)

def _run_ekf_single(args):
    """Run single EKF-tuned sim (module-level for pickle)."""
    qi, ri, q, rv, seed = args
    sim = create_simulation(seed=seed)
    sim._record_truth_predictions = lambda: None
    sim.dt = 0.005; sim.control_dt = 0.05
    sim.controller.ekf.Q = np.array([[q, 0], [0, q*0.5]])
    sim.controller.ekf.R_alt = rv
    r = sim.run()
    return (qi, ri, r.apogee_ft)

def run_batch(configs, label=''):
    """Run list of configs in parallel, return summary arrays."""
    n = len(configs)
    # Get 5 representative full trajectories serially
    rep_idx = [0, n//4, n//2, 3*n//4, n-1] if n > 5 else list(range(n))
    reps = {}
    for i in rep_idx:
        sim = create_simulation(**configs[i])
        reps[i] = sim.run()

    # Run all in parallel
    with multiprocessing.Pool(NUM_WORKERS, initializer=_init_worker) as pool:
        results = []
        for j, r in enumerate(pool.imap(_run_single, configs, chunksize=20)):
            results.append(r)
            pbar(j, n, label)

    ap_m = np.array([r[0] for r in results])
    ap_ft = np.array([r[1] for r in results])
    maxv = np.array([r[2] for r in results])
    maxm = np.array([r[3] for r in results])
    bo_alt = np.array([r[4] for r in results])
    bo_vel = np.array([r[5] for r in results])
    ap_t = np.array([r[6] for r in results])
    return dict(apogees_m=ap_m, apogees_ft=ap_ft, max_velocities=maxv, max_machs=maxm,
                burnout_alts=bo_alt, burnout_vels=bo_vel, apogee_times=ap_t, reps=reps)

def run_wind_batch(configs_with_wind, label=''):
    """Run wind simulations in parallel."""
    n = len(configs_with_wind)
    # Representative runs serially (need full trajectories)
    rep_idx = [0, n//4, n//2, 3*n//4, n-1]
    reps = {}
    for i in rep_idx:
        cfg, wind = configs_with_wind[i]
        sim = create_simulation(**cfg)
        w = wind
        def make_cf(s, wv):
            def _cf():
                thrust = s.sim_model.get_thrust(s.state.time)
                bcd = s.sim_model.get_cd(s.state.mach)
                dens = s.sim_model.get_density(s.state.altitude)
                vr = s.state.velocity - wv
                bd = 0.5 * dens * vr * abs(vr) * bcd * s.rocket.get_reference_area()
                if s.enable_airbrakes and s.airbrake.is_deployed:
                    acd = s.sim_model.get_airbrake_cd(s.state.mach)
                    ad = 0.5 * dens * vr * abs(vr) * acd * s.airbrake.get_area()
                else:
                    ad = 0.0
                return thrust, abs(bd), abs(ad)
            return _cf
        sim._compute_forces = make_cf(sim, w)
        reps[i] = sim.run()

    # All in parallel
    with multiprocessing.Pool(NUM_WORKERS, initializer=_init_worker) as pool:
        results = list(pool.imap(_run_wind_single, configs_with_wind, chunksize=20))
        for j in range(len(results)):
            pbar(j, n, label)

    ap_m = np.array([r[0] for r in results])
    ap_ft = np.array([r[1] for r in results])
    return dict(apogees_m=ap_m, apogees_ft=ap_ft, reps=reps)

def sweep(base, param, values, runs_per=50, label='', seed0=42):
    """Parameter sweep: run all configs in parallel."""
    nv = len(values)
    # Build all configs at once
    all_configs = []
    for j, v in enumerate(values):
        for i in range(runs_per):
            cfg = base.copy()
            cfg[param] = v
            cfg['seed'] = seed0 + j*runs_per + i
            all_configs.append((j, cfg))

    # Run all in parallel
    idx_and_cfgs = [(j, c) for j, c in all_configs]
    just_cfgs = [c for _, c in all_configs]

    with multiprocessing.Pool(NUM_WORKERS, initializer=_init_worker) as pool:
        results = []
        for k, r in enumerate(pool.imap(_run_single, just_cfgs, chunksize=20)):
            results.append(r)
            if k % 50 == 0 or k == len(just_cfgs)-1:
                pbar(k, len(just_cfgs), label)

    # Reorganize by sweep point
    mu, sig, p5, p95 = [np.zeros(nv) for _ in range(4)]
    s500, s1000 = np.zeros(nv), np.zeros(nv)
    all_ap = []
    for j in range(nv):
        start = j * runs_per
        aps = np.array([results[start+i][1] for i in range(runs_per)])  # apogee_ft
        mu[j], sig[j] = np.mean(aps), np.std(aps)
        p5[j], p95[j] = np.percentile(aps, 5), np.percentile(aps, 95)
        s500[j] = np.mean(np.abs(aps - TARGET_FT) <= 500) * 100
        s1000[j] = np.mean(np.abs(aps - TARGET_FT) <= 1000) * 100
        all_ap.append(aps)
    return dict(values=values, mean=mu, std=sig, p5=p5, p95=p95,
                s500=s500, s1000=s1000, all_apogees=all_ap)

def save_fig(fig, name):
    fig.savefig(os.path.join(FDIR, f'{name}.pdf'), bbox_inches='tight')
    plt.close(fig)

def mc_stats(ap_ft):
    return dict(mean=np.mean(ap_ft), std=np.std(ap_ft), median=np.median(ap_ft),
                p5=np.percentile(ap_ft, 5), p95=np.percentile(ap_ft, 95),
                min=np.min(ap_ft), max=np.max(ap_ft),
                s500=np.mean(np.abs(ap_ft - TARGET_FT) <= 500)*100,
                s1000=np.mean(np.abs(ap_ft - TARGET_FT) <= 1000)*100,
                error_mean=np.mean(ap_ft) - TARGET_FT,
                error_std=np.std(ap_ft))

def fig_hist(ap_ft, name, title, color=C[0]):
    fig, ax = plt.subplots(figsize=(8, 5))
    mu, sig = np.mean(ap_ft), np.std(ap_ft)
    ax.hist(ap_ft, bins=50, density=True, alpha=0.7, color=color, edgecolor='white', lw=0.5)
    x = np.linspace(mu-4*sig, mu+4*sig, 300)
    ax.plot(x, stats.norm.pdf(x, mu, sig), 'r-', lw=2,
            label=f'Normal fit ($\\mu$={mu:.0f}, $\\sigma$={sig:.0f} ft)')
    ax.axvline(TARGET_FT, color='red', ls='--', lw=2, label=f'Target ({TARGET_FT:.0f} ft)')
    ax.axvspan(TARGET_FT-500, TARGET_FT+500, alpha=0.08, color='green', label='$\\pm$500 ft')
    ax.set_xlabel('Apogee Altitude (ft)'); ax.set_ylabel('Probability Density')
    ax.set_title(f'{title} (N={len(ap_ft)})')
    ax.legend(); save_fig(fig, name)

def fig_cdf(ap_ft, name, title, color=C[0]):
    fig, ax = plt.subplots(figsize=(8, 5))
    s = np.sort(ap_ft)
    ax.plot(s, np.arange(1, len(s)+1)/len(s)*100, color=color, lw=2)
    ax.axvline(TARGET_FT, color='red', ls='--', lw=2, label='Target')
    ax.axvline(TARGET_FT-500, color='green', ls=':', lw=1.5)
    ax.axvline(TARGET_FT+500, color='green', ls=':', lw=1.5, label='$\\pm$500 ft')
    for p in [5, 50, 95]:
        v = np.percentile(ap_ft, p)
        ax.plot(v, p, 'ko', ms=5)
        ax.annotate(f'P{p}: {v:.0f}', (v, p), xytext=(10, 5), textcoords='offset points', fontsize=8)
    ax.set_xlabel('Apogee (ft)'); ax.set_ylabel('Cumulative %')
    ax.set_title(title); ax.legend(); ax.set_ylim(-2, 102); save_fig(fig, name)

def fig_trajectories(reps, name, title):
    if not reps: return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    for idx, r in sorted(reps.items()):
        lb = f'Run {idx} ({r.apogee_ft:.0f} ft)'
        a1.plot(r.time, r.altitude * M2F, alpha=0.7, label=lb)
        a2.plot(r.time, r.velocity, alpha=0.7, label=lb)
    a1.axhline(TARGET_FT, color='red', ls='--', lw=2, label='Target')
    a1.set_xlabel('Time (s)'); a1.set_ylabel('Altitude (ft)'); a1.set_title('Altitude'); a1.legend(fontsize=7)
    a2.set_xlabel('Time (s)'); a2.set_ylabel('Velocity (m/s)'); a2.set_title('Velocity'); a2.legend(fontsize=7)
    fig.suptitle(title, fontsize=13); fig.tight_layout(); save_fig(fig, name)

def fig_sweep(d, param_label, name, title):
    v, mu, sig, p5i, p95i = d['values'], d['mean'], d['std'], d['p5'], d['p95']
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    a1.fill_between(v, p5i, p95i, alpha=0.15, color=C[0], label='5th-95th pctl')
    a1.fill_between(v, mu-sig, mu+sig, alpha=0.25, color=C[0], label='$\\pm 1\\sigma$')
    a1.plot(v, mu, 'o-', color=C[0], lw=2, ms=4, label='Mean')
    a1.axhline(TARGET_FT, color='red', ls='--', lw=2, label='Target')
    a1.axhspan(TARGET_FT-500, TARGET_FT+500, alpha=0.05, color='green')
    a1.set_xlabel(param_label); a1.set_ylabel('Apogee (ft)'); a1.set_title(f'Apogee vs {param_label}')
    a1.legend(fontsize=7)
    a2.plot(v, d['s500'], 'o-', color=C[2], lw=2, ms=4, label='$\\pm$500 ft')
    a2.plot(v, d['s1000'], 's-', color=C[0], lw=2, ms=4, label='$\\pm$1000 ft')
    a2.set_xlabel(param_label); a2.set_ylabel('Success Rate (%)'); a2.set_title('Success Rate')
    a2.legend(); a2.set_ylim(-5, 105)
    fig.suptitle(title, fontsize=13); fig.tight_layout(); save_fig(fig, name)


# =============================================================================
# STUDY 1: Baseline Performance
# =============================================================================
def study_baseline():
    print("\n" + "="*70)
    print("STUDY 1: Baseline Performance")
    print("="*70)

    r_on = create_simulation(enable_airbrakes=True).run()
    r_off = create_simulation(enable_airbrakes=False).run()

    # Fig: trajectory comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    a = axes[0, 0]
    a.plot(r_on.time, r_on.altitude*M2F, C[0], label=f'With Airbrakes ({r_on.apogee_ft:.0f} ft)')
    a.plot(r_off.time, r_off.altitude*M2F, C[1], label=f'Without ({r_off.apogee_ft:.0f} ft)')
    a.axhline(TARGET_FT, color='red', ls='--', label='Target')
    a.set_xlabel('Time (s)'); a.set_ylabel('Altitude (ft)'); a.set_title('Altitude'); a.legend()

    a = axes[0, 1]
    a.plot(r_on.time, r_on.velocity, C[0], label='With Airbrakes')
    a.plot(r_off.time, r_off.velocity, C[1], label='Without')
    a.set_xlabel('Time (s)'); a.set_ylabel('Velocity (m/s)'); a.set_title('Velocity'); a.legend()

    a = axes[1, 0]
    a.plot(r_on.time, r_on.mach, C[0], label='With Airbrakes')
    a.plot(r_off.time, r_off.mach, C[1], label='Without')
    a.axhline(1.0, color='gray', ls=':', label='Mach 1')
    a.set_xlabel('Time (s)'); a.set_ylabel('Mach'); a.set_title('Mach Number'); a.legend()

    a = axes[1, 1]
    a.plot(r_on.time, r_on.acceleration/9.81, C[0], label='With Airbrakes')
    a.plot(r_off.time, r_off.acceleration/9.81, C[1], label='Without')
    a.set_xlabel('Time (s)'); a.set_ylabel('Acceleration (g)'); a.set_title('Acceleration'); a.legend()

    fig.suptitle('Baseline Trajectory Comparison', fontsize=14)
    fig.tight_layout(); save_fig(fig, 'S01_baseline_comparison')

    # Fig: airbrake deployment
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    a = axes[0, 0]
    a.plot(r_on.time, r_on.airbrake_angle, C[4], lw=2)
    a.set_xlabel('Time (s)'); a.set_ylabel('Angle (deg)'); a.set_title('Airbrake Deployment Angle')

    a = axes[0, 1]
    a.plot(r_on.time, r_on.body_drag, C[0], label='Body Drag')
    a.plot(r_on.time, r_on.airbrake_drag, C[1], label='Airbrake Drag')
    a.set_xlabel('Time (s)'); a.set_ylabel('Force (N)'); a.set_title('Drag Forces'); a.legend()

    a = axes[1, 0]
    a.plot(r_on.time, r_on.thrust, C[2], lw=2)
    a.set_xlabel('Time (s)'); a.set_ylabel('Thrust (N)'); a.set_title('Thrust Profile')

    a = axes[1, 1]
    a.plot(r_on.time, r_on.mass, C[7], lw=2)
    a.set_xlabel('Time (s)'); a.set_ylabel('Mass (kg)'); a.set_title('Mass vs Time')

    fig.suptitle('Airbrake Performance Detail', fontsize=14)
    fig.tight_layout(); save_fig(fig, 'S01_airbrake_detail')

    # Controller telemetry
    tel = r_on.controller_telemetry
    if tel and len(tel.get('time', [])) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        t = np.array(tel['time'])
        a = axes[0, 0]
        a.plot(t, np.array(tel['predicted_apogee'])*M2F, C[0], label='Predicted')
        a.plot(t, np.array(tel['apogee_clean'])*M2F, C[2], alpha=0.5, label='No brakes')
        a.plot(t, np.array(tel['apogee_full_brake'])*M2F, C[1], alpha=0.5, label='Full brakes')
        a.axhline(TARGET_FT, color='red', ls='--', label='Target')
        a.set_xlabel('Time (s)'); a.set_ylabel('Apogee (ft)'); a.set_title('Apogee Predictions'); a.legend()

        a = axes[0, 1]
        a.plot(t, tel['altitude_est'], C[0], label='EKF Altitude')
        a.plot(t, tel['altitude_meas'], C[2], alpha=0.4, label='Measured')
        a.set_xlabel('Time (s)'); a.set_ylabel('Altitude (m)'); a.set_title('State Estimation'); a.legend()

        a = axes[1, 0]
        a.plot(t, tel['velocity_est'], C[0])
        a.set_xlabel('Time (s)'); a.set_ylabel('Velocity (m/s)'); a.set_title('EKF Velocity Estimate')

        a = axes[1, 1]
        a.plot(t, tel['commanded_angle'], C[4])
        a.set_xlabel('Time (s)'); a.set_ylabel('Angle (deg)'); a.set_title('Commanded Angle')

        fig.suptitle('Controller Telemetry', fontsize=14)
        fig.tight_layout(); save_fig(fig, 'S01_controller_telemetry')

    stats_bl = {
        'with_airbrakes_ft': r_on.apogee_ft, 'without_airbrakes_ft': r_off.apogee_ft,
        'delta_ft': r_off.apogee_ft - r_on.apogee_ft,
        'error_ft': r_on.apogee_ft - TARGET_FT,
        'max_mach': r_on.max_mach, 'max_velocity': r_on.max_velocity,
        'burnout_alt_ft': r_on.burnout_altitude * M2F,
        'burnout_vel': r_on.burnout_velocity, 'apogee_time': r_on.apogee_time,
    }
    ALL_STATS['baseline'] = stats_bl
    print(f"  With airbrakes: {r_on.apogee_ft:.0f} ft | Without: {r_off.apogee_ft:.0f} ft | Delta: {stats_bl['delta_ft']:.0f} ft")
    return stats_bl


# =============================================================================
# STUDY 2: Aerodynamic Uncertainty (Cd)
# =============================================================================
def study_cd_uncertainty():
    print("\n" + "="*70)
    print("STUDY 2: Aerodynamic (Cd) Uncertainty - 3000 runs")
    print("="*70)

    results = {}
    for std_pct in [5, 10, 15]:
        std = std_pct / 100.0
        rng = np.random.RandomState(42)
        configs = []
        for i in range(1000):
            cd_err0 = rng.normal(0, std)
            cd_err2 = rng.normal(0, std)
            configs.append(dict(cd_scale_mach0=1+cd_err0, cd_scale_mach2=1+cd_err2, seed=100+i))
        data = run_batch(configs, label=f'Cd {std_pct}%')
        results[std_pct] = data
        s = mc_stats(data['apogees_ft'])
        fig_hist(data['apogees_ft'], f'S02_cd_{std_pct}pct_hist', f'Cd Uncertainty $\\sigma$={std_pct}%')
        fig_cdf(data['apogees_ft'], f'S02_cd_{std_pct}pct_cdf', f'Cd Uncertainty CDF ($\\sigma$={std_pct}%)')
        print(f"  Cd {std_pct}%: mean={s['mean']:.0f} std={s['std']:.0f} s500={s['s500']:.1f}% s1000={s['s1000']:.1f}%")
        ALL_STATS[f'cd_{std_pct}pct'] = s

    # Combined box plot
    fig, ax = plt.subplots(figsize=(8, 6))
    data_bp = [results[s]['apogees_ft'] for s in [5, 10, 15]]
    bp = ax.boxplot(data_bp, labels=['5%', '10%', '15%'], patch_artist=True,
                    boxprops=dict(facecolor=C[0], alpha=0.3))
    ax.axhline(TARGET_FT, color='red', ls='--', lw=2, label='Target')
    ax.axhspan(TARGET_FT-500, TARGET_FT+500, alpha=0.08, color='green')
    ax.set_xlabel('Cd Uncertainty ($\\sigma$)'); ax.set_ylabel('Apogee (ft)')
    ax.set_title('Apogee Distribution vs Cd Uncertainty Level')
    ax.legend(); save_fig(fig, 'S02_cd_boxplot')

    fig_trajectories(results[10]['reps'], 'S02_cd_10pct_traj', 'Cd 10% Uncertainty - Representative Trajectories')
    return results


# =============================================================================
# STUDY 3: Thrust Dispersion
# =============================================================================
def study_thrust_dispersion():
    print("\n" + "="*70)
    print("STUDY 3: Thrust Dispersion - 3000 runs")
    print("="*70)

    results = {}
    for std_pct in [2, 3, 5]:
        std = std_pct / 100.0
        rng = np.random.RandomState(200)
        configs = []
        for i in range(1000):
            ts = max(0.8, rng.normal(1.0, std))
            configs.append(dict(thrust_scale=ts, seed=200+i))
        data = run_batch(configs, label=f'Thrust {std_pct}%')
        results[std_pct] = data
        s = mc_stats(data['apogees_ft'])
        fig_hist(data['apogees_ft'], f'S03_thrust_{std_pct}pct_hist',
                 f'Thrust Dispersion $\\sigma$={std_pct}%', color=C[1])
        fig_cdf(data['apogees_ft'], f'S03_thrust_{std_pct}pct_cdf',
                f'Thrust Dispersion CDF ($\\sigma$={std_pct}%)', color=C[1])
        print(f"  Thrust {std_pct}%: mean={s['mean']:.0f} std={s['std']:.0f} s500={s['s500']:.1f}%")
        ALL_STATS[f'thrust_{std_pct}pct'] = s

    fig, ax = plt.subplots(figsize=(8, 6))
    bp = ax.boxplot([results[s]['apogees_ft'] for s in [2, 3, 5]],
                    labels=['2%', '3%', '5%'], patch_artist=True,
                    boxprops=dict(facecolor=C[1], alpha=0.3))
    ax.axhline(TARGET_FT, color='red', ls='--', lw=2, label='Target')
    ax.axhspan(TARGET_FT-500, TARGET_FT+500, alpha=0.08, color='green')
    ax.set_xlabel('Thrust Uncertainty ($\\sigma$)'); ax.set_ylabel('Apogee (ft)')
    ax.set_title('Apogee Distribution vs Thrust Variability'); ax.legend()
    save_fig(fig, 'S03_thrust_boxplot')

    # Scatter: thrust_scale vs apogee
    d5 = results[5]
    scales = np.array([c['thrust_scale'] for c in [dict(thrust_scale=max(0.8, np.random.RandomState(200).normal(1.0, 0.05))) for _ in range(1000)]])
    # Recompute scales from RNG
    rng = np.random.RandomState(200)
    scales = np.array([max(0.8, rng.normal(1.0, 0.05)) for _ in range(1000)])
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(scales, d5['apogees_ft'], alpha=0.3, s=10, c=C[1])
    ax.axhline(TARGET_FT, color='red', ls='--', lw=2)
    ax.set_xlabel('Thrust Scale Factor'); ax.set_ylabel('Apogee (ft)')
    ax.set_title('Apogee vs Thrust Scale (5% std)')
    save_fig(fig, 'S03_thrust_scatter')

    fig_trajectories(results[5]['reps'], 'S03_thrust_5pct_traj', 'Thrust 5% Dispersion - Trajectories')
    return results


# =============================================================================
# STUDY 4: Wind Perturbation
# =============================================================================
def study_wind():
    print("\n" + "="*70)
    print("STUDY 4: Wind Perturbation - 4000 runs")
    print("="*70)

    results = {}
    for wind_std in [1, 2, 3, 5]:
        rng = np.random.RandomState(300 + wind_std)
        configs_wind = []
        wind_speeds = []
        for i in range(1000):
            w = rng.normal(0, wind_std)
            wind_speeds.append(w)
            cfg = dict(seed=300+i)
            configs_wind.append((cfg, w))
        data = run_wind_batch(configs_wind, label=f'Wind {wind_std}m/s')
        data['wind_speeds'] = np.array(wind_speeds)
        results[wind_std] = data
        s = mc_stats(data['apogees_ft'])
        fig_hist(data['apogees_ft'], f'S04_wind_{wind_std}ms_hist',
                 f'Wind $\\sigma$={wind_std} m/s', color=C[5])
        print(f"  Wind σ={wind_std}m/s: mean={s['mean']:.0f} std={s['std']:.0f} s500={s['s500']:.1f}%")
        ALL_STATS[f'wind_{wind_std}ms'] = s

    # --- Combined figure: violin + scatter + CDF ---
    wind_sigmas = [1, 2, 3, 5]
    wind_colors = [C[0], C[2], C[3], C[9]]

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.28)

    # --- Panel A: Violin + box + strip plot ---
    ax_v = fig.add_subplot(gs[0, 0])
    positions = np.arange(len(wind_sigmas))
    vp = ax_v.violinplot([results[s]['apogees_ft'] for s in wind_sigmas],
                         positions=positions, showextrema=False, widths=0.7)
    for body, col in zip(vp['bodies'], wind_colors):
        body.set_facecolor(col); body.set_alpha(0.25); body.set_edgecolor(col)
    bp = ax_v.boxplot([results[s]['apogees_ft'] for s in wind_sigmas],
                      positions=positions, widths=0.15, patch_artist=True,
                      showfliers=False, medianprops=dict(color='black', lw=1.5),
                      whiskerprops=dict(lw=1.2), capprops=dict(lw=1.2))
    for patch, col in zip(bp['boxes'], wind_colors):
        patch.set_facecolor(col); patch.set_alpha(0.6)
    for k, ws in enumerate(wind_sigmas):
        d = results[ws]
        rng_jitter = np.random.RandomState(42)
        jitter = rng_jitter.uniform(-0.18, 0.18, len(d['apogees_ft']))
        ax_v.scatter(positions[k] + jitter, d['apogees_ft'], s=2, alpha=0.08,
                     color=wind_colors[k], zorder=1)
        ax_v.scatter(positions[k], np.mean(d['apogees_ft']), marker='D', s=40,
                     color='black', zorder=5, edgecolors='white', linewidths=0.5)
    ax_v.axhline(TARGET_FT, color='red', ls='--', lw=1.5, label='Target', zorder=4)
    ax_v.axhspan(TARGET_FT-500, TARGET_FT+500, alpha=0.06, color='green', label='$\\pm$500 ft')
    ax_v.set_xticks(positions)
    ax_v.set_xticklabels([f'$\\sigma$={s}' for s in wind_sigmas])
    ax_v.set_xlabel('Wind Speed Uncertainty (m/s)')
    ax_v.set_ylabel('Apogee (ft)')
    ax_v.set_title('(a) Apogee Distribution by Wind Uncertainty')
    ax_v.legend(loc='upper left', fontsize=7)

    # --- Panel B: 2x2 scatter grid for each wind sigma ---
    gs_inner = gs[0, 1].subgridspec(2, 2, hspace=0.35, wspace=0.3)
    for k, ws in enumerate(wind_sigmas):
        ax_s = fig.add_subplot(gs_inner[k // 2, k % 2])
        d = results[ws]
        within = np.abs(d['apogees_ft'] - TARGET_FT) <= 500
        ax_s.scatter(d['wind_speeds'][~within], d['apogees_ft'][~within],
                     s=6, alpha=0.25, color='#AAAAAA', zorder=2, label='Outside')
        ax_s.scatter(d['wind_speeds'][within], d['apogees_ft'][within],
                     s=6, alpha=0.4, color=wind_colors[k], zorder=3, label='Within $\\pm$500')
        # Trend line
        z = np.polyfit(d['wind_speeds'], d['apogees_ft'], 1)
        x_line = np.linspace(d['wind_speeds'].min(), d['wind_speeds'].max(), 50)
        ax_s.plot(x_line, np.polyval(z, x_line), color='black', lw=1.2, ls='-', alpha=0.7)
        ax_s.axhline(TARGET_FT, color='red', ls='--', lw=1, alpha=0.6)
        ax_s.set_title(f'$\\sigma$={ws} m/s', fontsize=9, pad=3)
        ax_s.tick_params(labelsize=7)
        if k >= 2: ax_s.set_xlabel('Wind (m/s)', fontsize=8)
        if k % 2 == 0: ax_s.set_ylabel('Apogee (ft)', fontsize=8)
    fig.text(0.75, 0.96, '(b) Wind Speed vs Apogee', ha='center', fontsize=11, weight='bold')

    # --- Panel C: CDF with shaded target band ---
    ax_c = fig.add_subplot(gs[1, 0])
    for k, ws in enumerate(wind_sigmas):
        s = np.sort(results[ws]['apogees_ft'])
        cdf = np.arange(1, len(s)+1) / len(s) * 100
        ax_c.plot(s, cdf, color=wind_colors[k], lw=2, label=f'$\\sigma$={ws} m/s')
        # Mark median
        med = np.median(s)
        ax_c.plot(med, 50, 'o', color=wind_colors[k], ms=5, zorder=5)
    ax_c.axvline(TARGET_FT, color='red', ls='--', lw=1.5, label='Target')
    ax_c.axvspan(TARGET_FT-500, TARGET_FT+500, alpha=0.08, color='green', label='$\\pm$500 ft')
    ax_c.set_xlabel('Apogee (ft)'); ax_c.set_ylabel('Cumulative Probability (%)')
    ax_c.set_title('(c) CDF Comparison'); ax_c.legend(fontsize=7, loc='upper left')
    ax_c.set_ylim(-2, 102)

    # --- Panel D: Summary stats table ---
    ax_t = fig.add_subplot(gs[1, 1])
    ax_t.axis('off')
    col_labels = ['$\\sigma$ (m/s)', 'Mean (ft)', 'Std (ft)', '$\\pm$500 ft', '$\\pm$1000 ft']
    table_data = []
    for ws in wind_sigmas:
        s = mc_stats(results[ws]['apogees_ft'])
        table_data.append([f'{ws}', f'{s["mean"]:.0f}', f'{s["std"]:.0f}',
                           f'{s["s500"]:.1f}%', f'{s["s1000"]:.1f}%'])
    tbl = ax_t.table(cellText=table_data, colLabels=col_labels,
                     loc='center', cellLoc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    tbl.scale(1, 1.6)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor('#E0E0E0'); cell.set_text_props(weight='bold')
        else:
            cell.set_facecolor('#F9F9F9')
        cell.set_edgecolor('#CCCCCC')
    ax_t.set_title('(d) Performance Summary', pad=15)

    fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'S04_wind_combined.png'),
                dpi=300, bbox_inches='tight')
    save_fig(fig, 'S04_wind_combined')

    return results


# =============================================================================
# STUDY 5: Combined Uncertainty
# =============================================================================
def study_combined():
    print("\n" + "="*70)
    print("STUDY 5: Combined Uncertainty MC - 1000 runs")
    print("="*70)

    rng = np.random.RandomState(500)
    configs = []
    for i in range(1000):
        configs.append(dict(
            cd_scale_mach0=1 + rng.normal(0, 0.10),
            cd_scale_mach2=1 + rng.normal(0, 0.10),
            airbrake_cd_scale_mach0=1 + rng.normal(0, 0.10),
            airbrake_cd_scale_mach2=1 + rng.normal(0, 0.10),
            thrust_scale=max(0.8, rng.normal(1.0, 0.03)),
            launch_temp_offset_k=rng.normal(0, 10),
            pressure_noise_std_pa=PRESSURE_NOISE_STD * max(0.3, rng.normal(1.0, 0.3)),
            accel_noise_std_mss=ACCEL_Z_NOISE_STD * max(0.3, rng.normal(1.0, 0.3)),
            seed=500+i,
        ))
    data = run_batch(configs, label='Combined MC')
    s = mc_stats(data['apogees_ft'])
    fig_hist(data['apogees_ft'], 'S05_combined_hist', 'Combined Uncertainty', color=C[4])
    fig_cdf(data['apogees_ft'], 'S05_combined_cdf', 'Combined Uncertainty CDF', color=C[4])
    fig_trajectories(data['reps'], 'S05_combined_traj', 'Combined Uncertainty - Trajectories')

    # Correlation: Cd scale vs apogee
    cd_scales = np.array([c['cd_scale_mach0'] for c in configs])
    thrust_scales = np.array([c['thrust_scale'] for c in configs])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    a1.scatter(cd_scales, data['apogees_ft'], alpha=0.2, s=8, c=C[4])
    a1.set_xlabel('Cd Scale (Mach 0)'); a1.set_ylabel('Apogee (ft)'); a1.set_title('Apogee vs Cd Scale')
    a1.axhline(TARGET_FT, color='red', ls='--')
    a2.scatter(thrust_scales, data['apogees_ft'], alpha=0.2, s=8, c=C[1])
    a2.set_xlabel('Thrust Scale'); a2.set_ylabel('Apogee (ft)'); a2.set_title('Apogee vs Thrust Scale')
    a2.axhline(TARGET_FT, color='red', ls='--')
    fig.suptitle('Parameter Correlation Analysis', fontsize=13); fig.tight_layout()
    save_fig(fig, 'S05_combined_correlation')

    print(f"  Combined: mean={s['mean']:.0f} std={s['std']:.0f} s500={s['s500']:.1f}%")
    ALL_STATS['combined'] = s
    return data


# =============================================================================
# STUDY 6: Sensor Noise Sensitivity
# =============================================================================
def study_sensor_noise():
    print("\n" + "="*70)
    print("STUDY 6: Sensor Noise Sensitivity - 1000 runs")
    print("="*70)

    mults = np.array([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0,
                      5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 18.0, 20.0, 25.0])
    nv = len(mults)
    mu, sig, s500 = np.zeros(nv), np.zeros(nv), np.zeros(nv)
    runs_per = 50
    # Build all configs for parallel execution
    all_cfgs = []
    for j, m in enumerate(mults):
        for i in range(runs_per):
            all_cfgs.append(dict(
                pressure_noise_std_pa=PRESSURE_NOISE_STD*m,
                temperature_noise_std_k=TEMPERATURE_NOISE_STD*m,
                accel_noise_std_mss=ACCEL_Z_NOISE_STD*m,
                seed=600+j*runs_per+i))
    with multiprocessing.Pool(NUM_WORKERS, initializer=_init_worker) as pool:
        results = list(pool.imap(_run_single, all_cfgs, chunksize=20))
        pbar(len(results), len(results), 'Noise')
    # Reorganize by sweep point
    for j in range(nv):
        start = j * runs_per
        aps = np.array([results[start+i][1] for i in range(runs_per)])  # apogee_ft
        mu[j], sig[j] = np.mean(aps), np.std(aps)
        s500[j] = np.mean(np.abs(aps-TARGET_FT) <= 500)*100

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(18, 5))
    a1.plot(mults, mu, 'o-', color=C[0], ms=4)
    a1.fill_between(mults, mu-sig, mu+sig, alpha=0.2, color=C[0])
    a1.axhline(TARGET_FT, color='red', ls='--'); a1.set_xlabel('Noise Multiplier')
    a1.set_ylabel('Apogee (ft)'); a1.set_title('Mean Apogee vs Noise Level')

    a2.plot(mults, sig, 'o-', color=C[1], ms=4)
    a2.set_xlabel('Noise Multiplier'); a2.set_ylabel('Apogee Std Dev (ft)')
    a2.set_title('Apogee Variability vs Noise')

    a3.plot(mults, s500, 'o-', color=C[2], ms=4)
    a3.set_xlabel('Noise Multiplier'); a3.set_ylabel('Success Rate (%)')
    a3.set_title('Success Rate ($\\pm$500 ft) vs Noise'); a3.set_ylim(-5, 105)

    fig.suptitle('Sensor Noise Sensitivity Analysis', fontsize=13)
    fig.tight_layout(); save_fig(fig, 'S06_noise_sensitivity')

    ALL_STATS['noise_sensitivity'] = dict(mults=mults.tolist(), means=mu.tolist(),
                                          stds=sig.tolist(), s500=s500.tolist())
    print(f"  Nominal noise s500={s500[3]:.1f}%, 10x noise s500={s500[14]:.1f}%")
    return dict(mults=mults, mu=mu, sig=sig, s500=s500)


# =============================================================================
# STUDY 7: Sensor Bias
# =============================================================================
def study_sensor_bias():
    print("\n" + "="*70)
    print("STUDY 7: Sensor Bias Study - 2000 runs")
    print("="*70)

    # Pressure bias sweep
    p_offsets = np.linspace(-500, 500, 20)
    d_press = sweep(dict(), 'pressure_noise_offset_pa', p_offsets, runs_per=50, label='P bias')
    fig_sweep(d_press, 'Pressure Bias (Pa)', 'S07_pressure_bias', 'Pressure Sensor Bias Effect')

    # Accelerometer bias sweep
    a_offsets = np.linspace(-3, 3, 20)
    d_accel = sweep(dict(), 'accel_noise_offset_mss', a_offsets, runs_per=50, label='A bias', seed0=700)
    fig_sweep(d_accel, 'Accelerometer Bias (m/s²)', 'S07_accel_bias', 'Accelerometer Bias Effect')

    # Temperature bias
    t_offsets = np.linspace(-5, 5, 20)
    d_temp = sweep(dict(), 'temperature_noise_offset_k', t_offsets, runs_per=50, label='T bias', seed0=750)
    fig_sweep(d_temp, 'Temperature Bias (K)', 'S07_temp_bias', 'Temperature Sensor Bias Effect')

    ALL_STATS['pressure_bias'] = dict(offsets=p_offsets.tolist(), means=d_press['mean'].tolist(), stds=d_press['std'].tolist())
    ALL_STATS['accel_bias'] = dict(offsets=a_offsets.tolist(), means=d_accel['mean'].tolist(), stds=d_accel['std'].tolist())
    ALL_STATS['temp_bias'] = dict(offsets=t_offsets.tolist(), means=d_temp['mean'].tolist(), stds=d_temp['std'].tolist())
    return d_press, d_accel, d_temp


# =============================================================================
# STUDY 8: Sensor Failure Modes
# =============================================================================
class FailingSensor(SensorModel):
    """Sensor that fails after N calls."""
    def __init__(self, failure_after, failure_type, **kw):
        super().__init__(**kw)
        self.failure_after = failure_after
        self.failure_type = failure_type
        self.calls = 0
        self._stuck_p = self._stuck_a = self._stuck_t = None

    def get_measurements(self, tp, tt, ta):
        self.calls += 1
        p, t, a = super().get_measurements(tp, tt, ta)
        if self.calls <= self.failure_after:
            self._stuck_p, self._stuck_t, self._stuck_a = p, t, a
            return p, t, a
        if self.failure_type == 'baro_stuck':
            return self._stuck_p or tp, t, a
        elif self.failure_type == 'accel_stuck_zero':
            return p, t, 0.0
        elif self.failure_type == 'accel_stuck':
            return p, t, self._stuck_a or ta
        elif self.failure_type == 'baro_drift':
            drift = (self.calls - self.failure_after) * 5.0  # 5 Pa per call
            return p + drift, t, a
        elif self.failure_type == 'accel_drift':
            drift = (self.calls - self.failure_after) * 0.005
            return p, t, a + drift
        elif self.failure_type == 'all_noisy':
            return p + self.rng.normal(0, 500), t + self.rng.normal(0, 5), a + self.rng.normal(0, 5)
        return p, t, a

def study_sensor_failure():
    print("\n" + "="*70)
    print("STUDY 8: Sensor Failure Modes")
    print("="*70)

    # Failure at ~burnout (around call 500 = 5s at 100Hz)
    scenarios = [
        ('Nominal', None, None),
        ('Baro Stuck at Burnout', 500, 'baro_stuck'),
        ('Accel Stuck at Zero', 500, 'accel_stuck_zero'),
        ('Baro Drift Post-Burnout', 500, 'baro_drift'),
        ('Accel Drift Post-Burnout', 500, 'accel_drift'),
        ('All Sensors Noisy', 500, 'all_noisy'),
        ('Baro Stuck Mid-Coast', 800, 'baro_stuck'),
        ('Accel Stuck Mid-Coast', 800, 'accel_stuck_zero'),
    ]

    results_sf = []
    for name, fail_after, fail_type in scenarios:
        sim = create_simulation(seed=42)
        if fail_after is not None:
            sim.sensors = FailingSensor(fail_after, fail_type, seed=42)
        r = sim.run()
        results_sf.append((name, r))
        print(f"  {name}: {r.apogee_ft:.0f} ft (error: {r.apogee_ft - TARGET_FT:+.0f} ft)")

    # Trajectory comparison
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 6))
    for k, (name, r) in enumerate(results_sf):
        a1.plot(r.time, r.altitude*M2F, color=C[k % len(C)], label=f'{name} ({r.apogee_ft:.0f}ft)', alpha=0.8)
        a2.plot(r.time, r.airbrake_angle, color=C[k % len(C)], alpha=0.8)
    a1.axhline(TARGET_FT, color='red', ls='--'); a1.set_xlabel('Time (s)')
    a1.set_ylabel('Altitude (ft)'); a1.set_title('Trajectories'); a1.legend(fontsize=6)
    a2.set_xlabel('Time (s)'); a2.set_ylabel('Angle (deg)'); a2.set_title('Airbrake Deployment')
    fig.suptitle('Sensor Failure Mode Comparison', fontsize=13)
    fig.tight_layout(); save_fig(fig, 'S08_sensor_failure_traj')

    # Bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    names = [n for n, _ in results_sf]
    errors = [r.apogee_ft - TARGET_FT for _, r in results_sf]
    colors = [C[2] if abs(e) < 500 else (C[3] if abs(e) < 1000 else C[4]) for e in errors]
    ax.barh(range(len(names)), errors, color=colors, alpha=0.8)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
    ax.axvline(0, color='black', lw=1); ax.axvline(-500, color='green', ls=':', alpha=0.5)
    ax.axvline(500, color='green', ls=':', alpha=0.5)
    ax.set_xlabel('Apogee Error (ft)'); ax.set_title('Sensor Failure Impact on Apogee')
    save_fig(fig, 'S08_sensor_failure_bar')

    ALL_STATS['sensor_failure'] = {n: r.apogee_ft - TARGET_FT for n, r in results_sf}
    return results_sf


# =============================================================================
# STUDY 9: Actuator Slew Rate
# =============================================================================
def study_slew_rate():
    print("\n" + "="*70)
    print("STUDY 9: Actuator Slew Rate Sweep - 1000 runs")
    print("="*70)

    rates = np.array([10, 20, 30, 40, 50, 60, 75, 100, 125, 150, 180, 200,
                      250, 300, 400, 500, 600, 800, 1000, 1500])
    d = sweep(dict(), 'airbrake_slew_rate_deg_s', rates, runs_per=50, label='Slew', seed0=900)
    fig_sweep(d, 'Slew Rate (deg/s)', 'S09_slew_rate', 'Actuator Slew Rate Sensitivity')

    # Show deployment profiles at different rates
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    test_rates = [20, 50, 100, 200, 500, 1000]
    for k, rate in enumerate(test_rates):
        ax = axes.flat[k]
        r = create_simulation(airbrake_slew_rate_deg_s=rate, seed=42).run()
        ax.plot(r.time, r.airbrake_angle, C[0], lw=2)
        ax.set_title(f'{rate} deg/s (apo={r.apogee_ft:.0f} ft)')
        ax.set_xlabel('Time (s)'); ax.set_ylabel('Angle (deg)')
    fig.suptitle('Airbrake Deployment Profiles at Different Slew Rates', fontsize=13)
    fig.tight_layout(); save_fig(fig, 'S09_slew_profiles')

    ALL_STATS['slew_rate'] = dict(rates=rates.tolist(), means=d['mean'].tolist(),
                                   stds=d['std'].tolist(), s500=d['s500'].tolist())
    return d


# =============================================================================
# STUDY 10: Actuator Area Degradation
# =============================================================================
def study_area_degradation():
    print("\n" + "="*70)
    print("STUDY 10: Actuator Area Degradation - 1000 runs")
    print("="*70)

    fractions = np.linspace(0.1, 1.0, 20)
    areas = fractions * AIRBRAKE_MAX_AREA_M2
    d = sweep(dict(), 'airbrake_max_area_m2', areas, runs_per=50, label='Area', seed0=1000)
    d['fractions'] = fractions

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    a1.fill_between(fractions*100, d['p5'], d['p95'], alpha=0.15, color=C[0])
    a1.fill_between(fractions*100, d['mean']-d['std'], d['mean']+d['std'], alpha=0.25, color=C[0])
    a1.plot(fractions*100, d['mean'], 'o-', color=C[0], ms=4)
    a1.axhline(TARGET_FT, color='red', ls='--'); a1.set_xlabel('Available Area (%)')
    a1.set_ylabel('Apogee (ft)'); a1.set_title('Apogee vs Available Airbrake Area')

    a2.plot(fractions*100, d['s500'], 'o-', color=C[2], ms=4, label='$\\pm$500 ft')
    a2.plot(fractions*100, d['s1000'], 's-', color=C[0], ms=4, label='$\\pm$1000 ft')
    a2.set_xlabel('Available Area (%)'); a2.set_ylabel('Success Rate (%)')
    a2.set_title('Success Rate vs Available Area'); a2.legend(); a2.set_ylim(-5, 105)
    fig.suptitle('Airbrake Area Degradation Analysis', fontsize=13)
    fig.tight_layout(); save_fig(fig, 'S10_area_degradation')

    ALL_STATS['area_degradation'] = dict(fracs=(fractions*100).tolist(), means=d['mean'].tolist())
    return d


# =============================================================================
# STUDY 11: Actuator Failure Modes
# =============================================================================
def study_actuator_failure():
    print("\n" + "="*70)
    print("STUDY 11: Actuator Failure Modes")
    print("="*70)

    scenarios = [
        ('Nominal', dict(seed=42)),
        ('No Airbrakes', dict(enable_airbrakes=False, seed=42)),
        ('10% Slew Rate', dict(airbrake_slew_rate_deg_s=10, seed=42)),
        ('50% Area', dict(airbrake_max_area_m2=AIRBRAKE_MAX_AREA_M2*0.5, seed=42)),
        ('25% Area', dict(airbrake_max_area_m2=AIRBRAKE_MAX_AREA_M2*0.25, seed=42)),
        ('10% Area', dict(airbrake_max_area_m2=AIRBRAKE_MAX_AREA_M2*0.1, seed=42)),
    ]

    results_af = []
    for name, cfg in scenarios:
        r = create_simulation(**cfg).run()
        results_af.append((name, r))
        print(f"  {name}: {r.apogee_ft:.0f} ft (error: {r.apogee_ft - TARGET_FT:+.0f} ft)")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 6))
    for k, (name, r) in enumerate(results_af):
        a1.plot(r.time, r.altitude*M2F, color=C[k], label=f'{name} ({r.apogee_ft:.0f}ft)')
        if hasattr(r, 'airbrake_angle') and len(r.airbrake_angle) > 0:
            a2.plot(r.time, r.airbrake_angle, color=C[k], label=name)
    a1.axhline(TARGET_FT, color='red', ls='--'); a1.set_xlabel('Time (s)')
    a1.set_ylabel('Altitude (ft)'); a1.set_title('Trajectories'); a1.legend(fontsize=7)
    a2.set_xlabel('Time (s)'); a2.set_ylabel('Angle (deg)'); a2.set_title('Deployment'); a2.legend(fontsize=7)
    fig.suptitle('Actuator Failure Mode Comparison', fontsize=13)
    fig.tight_layout(); save_fig(fig, 'S11_actuator_failure')

    fig, ax = plt.subplots(figsize=(10, 5))
    names = [n for n, _ in results_af]
    errors = [r.apogee_ft - TARGET_FT for _, r in results_af]
    colors_b = [C[2] if abs(e) < 500 else (C[3] if abs(e) < 1000 else C[4]) for e in errors]
    ax.barh(range(len(names)), errors, color=colors_b, alpha=0.8)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
    ax.axvline(0, color='black'); ax.axvline(-500, color='green', ls=':')
    ax.axvline(500, color='green', ls=':')
    ax.set_xlabel('Apogee Error (ft)'); ax.set_title('Actuator Failure Impact')
    save_fig(fig, 'S11_actuator_failure_bar')

    ALL_STATS['actuator_failure'] = {n: r.apogee_ft - TARGET_FT for n, r in results_af}
    return results_af


# =============================================================================
# STUDY 12: Control Latency Sweep
# =============================================================================
def study_latency():
    print("\n" + "="*70)
    print("STUDY 12: Control Latency Sweep - 1250 runs")
    print("="*70)

    latencies = np.array([0, 10, 20, 30, 40, 50, 75, 100, 125, 150, 175, 200,
                          250, 300, 350, 400, 450, 500, 600, 750, 900, 1000,
                          1250, 1500, 2000])
    d = sweep(dict(), 'control_latency_ms', latencies, runs_per=50, label='Latency', seed0=1200)
    fig_sweep(d, 'Control Latency (ms)', 'S12_latency', 'Control System Latency Sensitivity')

    # Deployment profiles at key latencies
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    test_lats = [0, 50, 100, 200, 500, 1000]
    for k, lat in enumerate(test_lats):
        ax = axes.flat[k]
        r = create_simulation(control_latency_ms=lat, seed=42).run()
        ax.plot(r.time, r.airbrake_angle, C[0])
        ax2 = ax.twinx()
        ax2.plot(r.time, r.altitude*M2F, C[1], alpha=0.5)
        ax.set_title(f'{lat}ms (apo={r.apogee_ft:.0f} ft)')
        ax.set_xlabel('Time (s)'); ax.set_ylabel('Angle (deg)')
        ax2.set_ylabel('Alt (ft)')
    fig.suptitle('Deployment Profiles at Different Latencies', fontsize=13)
    fig.tight_layout(); save_fig(fig, 'S12_latency_profiles')

    ALL_STATS['latency'] = dict(latencies=latencies.tolist(), means=d['mean'].tolist(),
                                 stds=d['std'].tolist(), s500=d['s500'].tolist())
    return d


# =============================================================================
# STUDY 13: EKF Tuning Sensitivity
# =============================================================================
def study_ekf():
    print("\n" + "="*70)
    print("STUDY 13: EKF Tuning Sensitivity - 1000 runs")
    print("="*70)

    # 2D sweep: process noise (Q) vs measurement noise (R)
    q_vals = np.logspace(-1, 2, 10)  # 0.1 to 100
    r_vals = np.logspace(-1, 2, 10)  # 0.1 to 100
    runs_per = 10

    grid_mean = np.zeros((len(q_vals), len(r_vals)))
    grid_std = np.zeros_like(grid_mean)
    grid_s500 = np.zeros_like(grid_mean)
    total = len(q_vals) * len(r_vals)
    count = 0

    # Build all configs for parallel execution
    ekf_configs = []
    for qi, q in enumerate(q_vals):
        for ri, rv in enumerate(r_vals):
            for k in range(runs_per):
                ekf_configs.append((qi, ri, q, rv, 1300 + count*runs_per + k))
            count += 1

    with multiprocessing.Pool(NUM_WORKERS, initializer=_init_worker) as pool:
        ekf_results = list(pool.imap(_run_ekf_single, ekf_configs, chunksize=10))
        pbar(len(ekf_results), len(ekf_results), 'EKF')

    # Reorganize results into grids
    from collections import defaultdict
    grid_data = defaultdict(list)
    for qi, ri, apft in ekf_results:
        grid_data[(qi, ri)].append(apft)
    for qi in range(len(q_vals)):
        for ri in range(len(r_vals)):
            aps = np.array(grid_data[(qi, ri)])
            grid_mean[qi, ri] = np.mean(aps)
            grid_std[qi, ri] = np.std(aps)
            grid_s500[qi, ri] = np.mean(np.abs(aps - TARGET_FT) <= 500) * 100

    # Heatmaps
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, data_grid, title, cmap in [
        (axes[0], np.abs(grid_mean - TARGET_FT), 'Mean Apogee Error (ft)', 'Reds'),
        (axes[1], grid_std, 'Apogee Std Dev (ft)', 'Oranges'),
        (axes[2], grid_s500, 'Success Rate ±500ft (%)', 'Greens'),
    ]:
        im = ax.imshow(data_grid, origin='lower', aspect='auto', cmap=cmap,
                       extent=[np.log10(r_vals[0]), np.log10(r_vals[-1]),
                               np.log10(q_vals[0]), np.log10(q_vals[-1])])
        ax.set_xlabel('log10(Measurement Noise R)')
        ax.set_ylabel('log10(Process Noise Q)')
        ax.set_title(title)
        plt.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle('EKF Tuning Sensitivity Analysis', fontsize=13)
    fig.tight_layout(); save_fig(fig, 'S13_ekf_heatmap')

    # Contour
    fig, ax = plt.subplots(figsize=(8, 6))
    cs = ax.contourf(np.log10(r_vals), np.log10(q_vals), grid_s500,
                     levels=np.arange(0, 105, 10), cmap='RdYlGn')
    plt.colorbar(cs, label='Success Rate (%)')
    ax.set_xlabel('log10(Measurement Noise R)'); ax.set_ylabel('log10(Process Noise Q)')
    ax.set_title('EKF Tuning: Success Rate Contour')
    # Mark nominal
    ax.plot(np.log10(10), np.log10(1), 'k*', ms=15, label='Nominal')
    ax.legend(); save_fig(fig, 'S13_ekf_contour')

    ALL_STATS['ekf'] = dict(q_vals=q_vals.tolist(), r_vals=r_vals.tolist(),
                             best_s500=np.max(grid_s500),
                             worst_s500=np.min(grid_s500))
    return dict(q_vals=q_vals, r_vals=r_vals, grid_mean=grid_mean, grid_std=grid_std, grid_s500=grid_s500)


# =============================================================================
# STUDY 14: Temperature / Atmospheric Conditions
# =============================================================================
def study_temperature():
    print("\n" + "="*70)
    print("STUDY 14: Temperature Variation - 1000 runs")
    print("="*70)

    temps = np.linspace(-30, 30, 20)
    d = sweep(dict(), 'launch_temp_offset_k', temps, runs_per=50, label='Temp', seed0=1400)
    fig_sweep(d, 'Temperature Offset (K)', 'S14_temperature', 'Temperature Variation Sensitivity')

    # Show atmospheric profiles
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(16, 5))
    truth = get_truth_model()
    alts = np.linspace(0, 10000, 200)
    for k, temp_off in enumerate([-20, -10, 0, 10, 20]):
        label = f'{temp_off:+.0f}K'
        cfg = SimulationConfig(launch_temp_offset_k=temp_off)
        sm = SimulationModel(truth, cfg)
        dens = [sm.get_density(h) for h in alts]
        pres = [sm.get_pressure(h) for h in alts]
        temps_p = [sm.get_temperature(h) for h in alts]
        a1.plot(dens, alts*M2F, color=C[k], label=label)
        a2.plot(np.array(pres)/1000, alts*M2F, color=C[k], label=label)
        a3.plot(np.array(temps_p)-273.15, alts*M2F, color=C[k], label=label)
    a1.set_xlabel('Density (kg/m³)'); a1.set_ylabel('Altitude (ft)'); a1.set_title('Density')
    a1.legend()
    a2.set_xlabel('Pressure (kPa)'); a2.set_ylabel('Altitude (ft)'); a2.set_title('Pressure'); a2.legend()
    a3.set_xlabel('Temperature (°C)'); a3.set_ylabel('Altitude (ft)'); a3.set_title('Temperature'); a3.legend()
    fig.suptitle('Atmospheric Profiles at Different Temperatures', fontsize=13)
    fig.tight_layout(); save_fig(fig, 'S14_atmo_profiles')

    ALL_STATS['temperature'] = dict(offsets=temps.tolist(), means=d['mean'].tolist(), stds=d['std'].tolist())
    return d


# =============================================================================
# STUDY 15: Launch Altitude Variation
# =============================================================================
def study_altitude():
    print("\n" + "="*70)
    print("STUDY 15: Launch Altitude Variation - 500 runs")
    print("="*70)

    alt_offsets = np.linspace(-300, 1500, 15)  # -300m to +1500m from nominal
    d = sweep(dict(), 'launch_altitude_offset_m', alt_offsets, runs_per=35, label='Alt', seed0=1500)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    actual_alts = (LAUNCH_ALTITUDE_M + alt_offsets) * M2F
    a1.fill_between(actual_alts, d['p5'], d['p95'], alpha=0.15, color=C[0])
    a1.plot(actual_alts, d['mean'], 'o-', color=C[0], ms=4)
    a1.axhline(TARGET_FT, color='red', ls='--')
    a1.set_xlabel('Launch Site Altitude (ft MSL)'); a1.set_ylabel('Apogee (ft AGL)')
    a1.set_title('Apogee vs Launch Altitude')

    a2.plot(actual_alts, d['s500'], 'o-', color=C[2], ms=4, label='$\\pm$500 ft')
    a2.plot(actual_alts, d['s1000'], 's-', color=C[0], ms=4, label='$\\pm$1000 ft')
    a2.set_xlabel('Launch Altitude (ft MSL)'); a2.set_ylabel('Success Rate (%)')
    a2.set_title('Success Rate vs Launch Altitude'); a2.legend(); a2.set_ylim(-5, 105)
    fig.suptitle('Launch Altitude Sensitivity', fontsize=13)
    fig.tight_layout(); save_fig(fig, 'S15_altitude')

    ALL_STATS['altitude'] = dict(offsets=alt_offsets.tolist(), means=d['mean'].tolist())
    return d


# =============================================================================
# STUDY 16: Combined Worst-Case MC
# =============================================================================
def study_worst_case():
    print("\n" + "="*70)
    print("STUDY 16: Combined Worst-Case MC - 1000 runs")
    print("="*70)

    rng = np.random.RandomState(1600)
    configs = []
    for i in range(1000):
        configs.append(dict(
            cd_scale_mach0=1 + rng.normal(0, 0.15),  # 15% Cd uncertainty
            cd_scale_mach2=1 + rng.normal(0, 0.15),
            airbrake_cd_scale_mach0=1 + rng.normal(0, 0.15),
            airbrake_cd_scale_mach2=1 + rng.normal(0, 0.15),
            thrust_scale=max(0.8, rng.normal(1.0, 0.05)),  # 5% thrust
            launch_temp_offset_k=rng.normal(0, 15),  # 15K temp uncertainty
            launch_altitude_offset_m=rng.normal(0, 100),  # 100m altitude uncertainty
            pressure_noise_std_pa=PRESSURE_NOISE_STD * max(0.3, rng.normal(1.5, 0.5)),
            accel_noise_std_mss=ACCEL_Z_NOISE_STD * max(0.3, rng.normal(1.5, 0.5)),
            pressure_noise_offset_pa=rng.normal(0, 100),
            accel_noise_offset_mss=rng.normal(0, 0.5),
            control_latency_ms=max(0, rng.normal(50, 30)),
            airbrake_slew_rate_deg_s=max(20, rng.normal(100, 30)),
            seed=1600+i,
        ))
    data = run_batch(configs, label='Worst Case')
    s = mc_stats(data['apogees_ft'])

    fig_hist(data['apogees_ft'], 'S16_worstcase_hist', 'Worst-Case Combined Uncertainty', color=C[4])
    fig_cdf(data['apogees_ft'], 'S16_worstcase_cdf', 'Worst-Case CDF', color=C[4])
    fig_trajectories(data['reps'], 'S16_worstcase_traj', 'Worst-Case Trajectories')

    # 2D scatter
    cd_s = np.array([c['cd_scale_mach0'] for c in configs])
    ts = np.array([c['thrust_scale'] for c in configs])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    sc = a1.scatter(cd_s, data['apogees_ft'], c=data['apogees_ft'], cmap='RdYlGn_r',
                    alpha=0.3, s=10, vmin=TARGET_FT-3000, vmax=TARGET_FT+3000)
    a1.axhline(TARGET_FT, color='red', ls='--'); a1.set_xlabel('Cd Scale')
    a1.set_ylabel('Apogee (ft)'); a1.set_title('Apogee vs Cd Scale')
    sc = a2.scatter(ts, data['apogees_ft'], c=data['apogees_ft'], cmap='RdYlGn_r',
                    alpha=0.3, s=10, vmin=TARGET_FT-3000, vmax=TARGET_FT+3000)
    a2.axhline(TARGET_FT, color='red', ls='--'); a2.set_xlabel('Thrust Scale')
    a2.set_ylabel('Apogee (ft)'); a2.set_title('Apogee vs Thrust')
    fig.suptitle('Worst-Case Parameter Correlations', fontsize=13); fig.tight_layout()
    save_fig(fig, 'S16_worstcase_scatter')

    # Success rate histogram (by tolerance band)
    fig, ax = plt.subplots(figsize=(8, 5))
    tolerances = [100, 250, 500, 750, 1000, 1500, 2000, 3000]
    rates = [np.mean(np.abs(data['apogees_ft'] - TARGET_FT) <= t)*100 for t in tolerances]
    ax.bar(range(len(tolerances)), rates, color=C[0], alpha=0.8)
    ax.set_xticks(range(len(tolerances)))
    ax.set_xticklabels([f'±{t}' for t in tolerances])
    ax.set_xlabel('Tolerance Band (ft)'); ax.set_ylabel('Success Rate (%)')
    ax.set_title('Worst-Case Success Rate by Tolerance')
    for k, r in enumerate(rates):
        ax.text(k, r+1, f'{r:.0f}%', ha='center', fontsize=8)
    save_fig(fig, 'S16_worstcase_tolerance')

    print(f"  Worst-case: mean={s['mean']:.0f} std={s['std']:.0f} s500={s['s500']:.1f}% s1000={s['s1000']:.1f}%")
    ALL_STATS['worst_case'] = s
    return data


# =============================================================================
# SUMMARY FIGURE: Cross-Study Comparison
# =============================================================================
def make_summary_figures():
    print("\n" + "="*70)
    print("Generating Summary Figures")
    print("="*70)

    # Collect all MC-type studies
    studies = [
        ('Cd 5%', 'cd_5pct'), ('Cd 10%', 'cd_10pct'), ('Cd 15%', 'cd_15pct'),
        ('Thrust 2%', 'thrust_2pct'), ('Thrust 3%', 'thrust_3pct'), ('Thrust 5%', 'thrust_5pct'),
        ('Wind 1m/s', 'wind_1ms'), ('Wind 3m/s', 'wind_3ms'), ('Wind 5m/s', 'wind_5ms'),
        ('Combined', 'combined'), ('Worst Case', 'worst_case'),
    ]

    names, means, stds, s500s, s1000s = [], [], [], [], []
    for label, key in studies:
        if key in ALL_STATS:
            s = ALL_STATS[key]
            names.append(label)
            means.append(s.get('error_mean', s.get('mean', TARGET_FT) - TARGET_FT))
            stds.append(s.get('error_std', s.get('std', 0)))
            s500s.append(s.get('s500', 0))
            s1000s.append(s.get('s1000', 0))

    if not names:
        return

    # Mean error comparison
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 10))
    x = range(len(names))
    colors_e = [C[2] if abs(m) < 500 else C[3] if abs(m) < 1000 else C[4] for m in means]
    a1.bar(x, means, yerr=stds, color=colors_e, alpha=0.8, capsize=3)
    a1.axhline(0, color='black', lw=1); a1.axhline(-500, color='green', ls=':')
    a1.axhline(500, color='green', ls=':')
    a1.set_xticks(x); a1.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    a1.set_ylabel('Mean Apogee Error (ft)'); a1.set_title('Mean Error Across Studies')

    a2.bar(x, s500s, color=C[2], alpha=0.8, label='±500 ft')
    a2.bar(x, s1000s, color=C[0], alpha=0.3, label='±1000 ft')
    a2.set_xticks(x); a2.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    a2.set_ylabel('Success Rate (%)'); a2.set_title('Success Rates Across Studies')
    a2.legend(); a2.set_ylim(0, 105)

    fig.suptitle('Cross-Study Comparison Summary', fontsize=14)
    fig.tight_layout(); save_fig(fig, 'S00_summary_comparison')

    # Std dev comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(x, stds, color=C[0], alpha=0.8)
    ax.set_yticks(x); ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('Apogee Standard Deviation (ft)')
    ax.set_title('Apogee Variability Across Uncertainty Sources')
    save_fig(fig, 'S00_summary_stdev')


# =============================================================================
# MAIN
# =============================================================================
def main():
    global ALL_STATS
    t0 = clock.time()
    resume = '--resume' in sys.argv
    print("=" * 70)
    print("COMPREHENSIVE SIMULATION STUDY")
    print("Stanford Space Initiative - IREC Rocket Airbrake System")
    print(f"Target Apogee: {TARGET_FT:.0f} ft ({TARGET_APOGEE_M:.0f} m)")
    if resume:
        print("RESUMING from Study 6 (loading saved stats)")
        stats_file = os.path.join(DDIR, 'all_stats.json')
        if os.path.exists(stats_file):
            with open(stats_file) as f:
                ALL_STATS = json.load(f)
    print("=" * 70)

    if not resume:
        study_baseline()
        study_cd_uncertainty()
        study_thrust_dispersion()
        study_wind()
        study_combined()
    else:
        # Load cached stats from first run
        ALL_STATS.update({
            'baseline': {'with_airbrakes_ft': 30002, 'without_airbrakes_ft': 32152, 'delta_ft': 2150, 'error_ft': 2},
            'cd_5pct': {'mean': 30034, 'std': 158, 's500': 97.6, 's1000': 99.2, 'error_mean': 34, 'error_std': 158},
            'cd_10pct': {'mean': 30204, 'std': 786, 's500': 79.2, 's1000': 86.7, 'error_mean': 204, 'error_std': 786},
            'cd_15pct': {'mean': 30404, 'std': 1645, 's500': 61.3, 's1000': 70.2, 'error_mean': 404, 'error_std': 1645},
            'thrust_2pct': {'mean': 30041, 'std': 176, 's500': 96.3, 's1000': 99.0, 'error_mean': 41, 'error_std': 176},
            'thrust_3pct': {'mean': 30108, 'std': 470, 's500': 85.8, 's1000': 93.0, 'error_mean': 108, 'error_std': 470},
            'thrust_5pct': {'mean': 30216, 'std': 1229, 's500': 64.7, 's1000': 78.0, 'error_mean': 216, 'error_std': 1229},
            'wind_1ms': {'mean': 30004, 'std': 9, 's500': 100.0, 's1000': 100.0, 'error_mean': 4, 'error_std': 9},
            'wind_3ms': {'mean': 30008, 'std': 23, 's500': 100.0, 's1000': 100.0, 'error_mean': 8, 'error_std': 23},
            'wind_5ms': {'mean': 30020, 'std': 67, 's500': 99.6, 's1000': 100.0, 'error_mean': 20, 'error_std': 67},
            'combined': {'mean': 30372, 'std': 1387, 's500': 62.9, 's1000': 75.0, 'error_mean': 372, 'error_std': 1387},
        })
    if not resume:
        study_sensor_noise()
        study_sensor_bias()
        study_sensor_failure()
        study_slew_rate()
        study_area_degradation()
        study_actuator_failure()
        study_latency()
    study_ekf()
    study_temperature()
    study_altitude()
    study_worst_case()
    make_summary_figures()

    # Save all stats
    # Convert numpy types for JSON
    def convert(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return o

    with open(os.path.join(DDIR, 'all_stats.json'), 'w') as f:
        json.dump(ALL_STATS, f, indent=2, default=convert)

    elapsed = clock.time() - t0
    print(f"\n{'='*70}")
    print(f"STUDY COMPLETE - {elapsed/60:.1f} minutes ({elapsed:.0f}s)")
    print(f"Figures saved to: {FDIR}")
    print(f"Data saved to: {DDIR}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
