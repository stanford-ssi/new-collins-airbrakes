#!/usr/bin/env python3
"""Finish remaining studies (16 + summary) and save all stats."""
import sys, os, json, multiprocessing
multiprocessing.set_start_method('fork', force=True)
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simulation import create_simulation
from config import (TARGET_APOGEE_M, METERS_TO_FEET, PRESSURE_NOISE_STD, ACCEL_Z_NOISE_STD)
from debug import configure_debug, DebugLevel
configure_debug(level=DebugLevel.OFF)

FDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'study_output', 'figures')
DDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'study_output', 'data')
os.makedirs(FDIR, exist_ok=True)
os.makedirs(DDIR, exist_ok=True)
TARGET_FT = TARGET_APOGEE_M * METERS_TO_FEET
M2F = METERS_TO_FEET
NUM_WORKERS = 8

C = ['#2196F3', '#FF5722', '#4CAF50', '#FFC107', '#9C27B0',
     '#00BCD4', '#FF9800', '#795548', '#607D8B', '#E91E63']

plt.rcParams.update({
    'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 11,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 8,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.grid': True, 'grid.alpha': 0.3, 'lines.linewidth': 1.5,
})

def _init_worker():
    import debug; debug.configure_debug(level=debug.DebugLevel.OFF)
    from truth_model import reset_truth_model; reset_truth_model()

def _run_single(cfg):
    sim = create_simulation(**cfg)
    sim._record_truth_predictions = lambda: None
    sim.dt = 0.005; sim.control_dt = 0.05
    r = sim.run()
    return (r.apogee_m, r.apogee_ft, r.max_velocity, r.max_mach,
            r.burnout_altitude, r.burnout_velocity, r.apogee_time)

def save_fig(fig, name):
    fig.savefig(os.path.join(FDIR, f'{name}.pdf'), bbox_inches='tight')
    plt.close(fig)

# ---- Known stats from completed runs ----
ALL_STATS = {
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
}

# ---- STUDY 16: Worst Case MC ----
print("STUDY 16: Worst Case MC - 1000 runs")
rng = np.random.RandomState(1600)
configs = []
for i in range(1000):
    configs.append(dict(
        cd_scale_mach0=1 + rng.normal(0, 0.15),
        cd_scale_mach2=1 + rng.normal(0, 0.15),
        airbrake_cd_scale_mach0=1 + rng.normal(0, 0.15),
        airbrake_cd_scale_mach2=1 + rng.normal(0, 0.15),
        thrust_scale=max(0.8, rng.normal(1.0, 0.05)),
        launch_temp_offset_k=rng.normal(0, 15),
        launch_altitude_offset_m=rng.normal(0, 100),
        pressure_noise_std_pa=PRESSURE_NOISE_STD * max(0.3, rng.normal(1.5, 0.5)),
        accel_noise_std_mss=ACCEL_Z_NOISE_STD * max(0.3, rng.normal(1.5, 0.5)),
        pressure_noise_offset_pa=rng.normal(0, 100),
        accel_noise_offset_mss=rng.normal(0, 0.5),
        control_latency_ms=max(0, rng.normal(50, 30)),
        airbrake_slew_rate_deg_s=max(20, rng.normal(100, 30)),
        seed=1600+i,
    ))

# Get 5 representative trajectories
rep_idx = [0, 250, 500, 750, 999]
reps = {}
for i in rep_idx:
    sim = create_simulation(**configs[i])
    reps[i] = sim.run()
    print(f"  Rep {i}: {reps[i].apogee_ft:.0f} ft")

# Run all in parallel
print("  Running 1000 parallel sims...")
with multiprocessing.Pool(NUM_WORKERS, initializer=_init_worker) as pool:
    results = list(pool.imap(_run_single, configs, chunksize=25))
print(f"  Done! Got {len(results)} results")

ap_ft = np.array([r[1] for r in results])
mu, sig = np.mean(ap_ft), np.std(ap_ft)
s500 = np.mean(np.abs(ap_ft - TARGET_FT) <= 500) * 100
s1000 = np.mean(np.abs(ap_ft - TARGET_FT) <= 1000) * 100
print(f"  Worst-case: mean={mu:.0f} std={sig:.0f} s500={s500:.1f}% s1000={s1000:.1f}%")
ALL_STATS['worst_case'] = {'mean': mu, 'std': sig, 's500': s500, 's1000': s1000,
                           'error_mean': mu - TARGET_FT, 'error_std': sig,
                           'p5': float(np.percentile(ap_ft, 5)), 'p95': float(np.percentile(ap_ft, 95)),
                           'min': float(np.min(ap_ft)), 'max': float(np.max(ap_ft))}

import scipy.stats as stats

# Histogram
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(ap_ft, bins=50, density=True, alpha=0.7, color=C[4], edgecolor='white', lw=0.5)
x = np.linspace(mu-4*sig, mu+4*sig, 300)
ax.plot(x, stats.norm.pdf(x, mu, sig), 'r-', lw=2, label=f'Normal fit ($\\mu$={mu:.0f}, $\\sigma$={sig:.0f} ft)')
ax.axvline(TARGET_FT, color='red', ls='--', lw=2, label=f'Target ({TARGET_FT:.0f} ft)')
ax.axvspan(TARGET_FT-500, TARGET_FT+500, alpha=0.08, color='green', label='$\\pm$500 ft')
ax.set_xlabel('Apogee Altitude (ft)'); ax.set_ylabel('Probability Density')
ax.set_title(f'Worst-Case Combined Uncertainty (N=1000)'); ax.legend()
save_fig(fig, 'S16_worstcase_hist')

# CDF
fig, ax = plt.subplots(figsize=(8, 5))
s = np.sort(ap_ft); ax.plot(s, np.arange(1, len(s)+1)/len(s)*100, color=C[4], lw=2)
ax.axvline(TARGET_FT, color='red', ls='--'); ax.axvline(TARGET_FT-500, color='green', ls=':')
ax.axvline(TARGET_FT+500, color='green', ls=':', label='$\\pm$500 ft')
for p in [5, 50, 95]:
    v = np.percentile(ap_ft, p); ax.plot(v, p, 'ko', ms=5)
    ax.annotate(f'P{p}: {v:.0f}', (v, p), xytext=(10, 5), textcoords='offset points', fontsize=8)
ax.set_xlabel('Apogee (ft)'); ax.set_ylabel('CDF (%)'); ax.set_title('Worst-Case CDF')
ax.legend(); ax.set_ylim(-2, 102); save_fig(fig, 'S16_worstcase_cdf')

# Trajectories
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
for idx, r in sorted(reps.items()):
    a1.plot(r.time, r.altitude*M2F, alpha=0.7, label=f'Run {idx} ({r.apogee_ft:.0f} ft)')
    a2.plot(r.time, r.velocity, alpha=0.7)
a1.axhline(TARGET_FT, color='red', ls='--', lw=2, label='Target')
a1.set_xlabel('Time (s)'); a1.set_ylabel('Altitude (ft)'); a1.set_title('Altitude'); a1.legend(fontsize=7)
a2.set_xlabel('Time (s)'); a2.set_ylabel('Velocity (m/s)'); a2.set_title('Velocity')
fig.suptitle('Worst-Case - Representative Trajectories', fontsize=13); fig.tight_layout()
save_fig(fig, 'S16_worstcase_traj')

# Scatter
cd_s = np.array([c['cd_scale_mach0'] for c in configs])
ts = np.array([c['thrust_scale'] for c in configs])
fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
sc = a1.scatter(cd_s, ap_ft, c=ap_ft, cmap='RdYlGn_r', alpha=0.3, s=10, vmin=TARGET_FT-3000, vmax=TARGET_FT+3000)
a1.axhline(TARGET_FT, color='red', ls='--'); a1.set_xlabel('Cd Scale'); a1.set_ylabel('Apogee (ft)')
a1.set_title('Apogee vs Cd Scale')
a2.scatter(ts, ap_ft, c=ap_ft, cmap='RdYlGn_r', alpha=0.3, s=10, vmin=TARGET_FT-3000, vmax=TARGET_FT+3000)
a2.axhline(TARGET_FT, color='red', ls='--'); a2.set_xlabel('Thrust Scale'); a2.set_ylabel('Apogee (ft)')
a2.set_title('Apogee vs Thrust')
fig.suptitle('Worst-Case Parameter Correlations', fontsize=13); fig.tight_layout()
save_fig(fig, 'S16_worstcase_scatter')

# Tolerance band
fig, ax = plt.subplots(figsize=(8, 5))
tolerances = [100, 250, 500, 750, 1000, 1500, 2000, 3000]
rates = [np.mean(np.abs(ap_ft - TARGET_FT) <= t)*100 for t in tolerances]
ax.bar(range(len(tolerances)), rates, color=C[0], alpha=0.8)
ax.set_xticks(range(len(tolerances))); ax.set_xticklabels([f'$\\pm${t}' for t in tolerances])
ax.set_xlabel('Tolerance Band (ft)'); ax.set_ylabel('Success Rate (%)')
ax.set_title('Worst-Case Success Rate by Tolerance')
for k, r in enumerate(rates): ax.text(k, r+1, f'{r:.0f}%', ha='center', fontsize=8)
save_fig(fig, 'S16_worstcase_tolerance')

# ---- SUMMARY FIGURES ----
print("\nGenerating Summary Figures...")
studies = [
    ('Cd 5%', 'cd_5pct'), ('Cd 10%', 'cd_10pct'), ('Cd 15%', 'cd_15pct'),
    ('Thrust 2%', 'thrust_2pct'), ('Thrust 3%', 'thrust_3pct'), ('Thrust 5%', 'thrust_5pct'),
    ('Wind 1m/s', 'wind_1ms'), ('Wind 3m/s', 'wind_3ms'), ('Wind 5m/s', 'wind_5ms'),
    ('Combined', 'combined'), ('Worst Case', 'worst_case'),
]
names, means, stdev, s500s, s1000s = [], [], [], [], []
for label, key in studies:
    if key in ALL_STATS:
        s = ALL_STATS[key]
        names.append(label); means.append(s.get('error_mean', s.get('mean', TARGET_FT) - TARGET_FT))
        stdev.append(s.get('error_std', s.get('std', 0)))
        s500s.append(s.get('s500', 0)); s1000s.append(s.get('s1000', 0))

fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 10))
x = range(len(names))
colors_e = [C[2] if abs(m) < 500 else C[3] if abs(m) < 1000 else C[4] for m in means]
a1.bar(x, means, yerr=stdev, color=colors_e, alpha=0.8, capsize=3)
a1.axhline(0, color='black', lw=1); a1.axhline(-500, color='green', ls=':'); a1.axhline(500, color='green', ls=':')
a1.set_xticks(x); a1.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
a1.set_ylabel('Mean Apogee Error (ft)'); a1.set_title('Mean Error Across Studies')
a2.bar(x, s500s, color=C[2], alpha=0.8, label='$\\pm$500 ft')
a2.bar(x, s1000s, color=C[0], alpha=0.3, label='$\\pm$1000 ft')
a2.set_xticks(x); a2.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
a2.set_ylabel('Success Rate (%)'); a2.set_title('Success Rates Across Studies'); a2.legend(); a2.set_ylim(0, 105)
fig.suptitle('Cross-Study Comparison Summary', fontsize=14); fig.tight_layout()
save_fig(fig, 'S00_summary_comparison')

fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(x, stdev, color=C[0], alpha=0.8)
ax.set_yticks(x); ax.set_yticklabels(names, fontsize=8)
ax.set_xlabel('Apogee Standard Deviation (ft)')
ax.set_title('Apogee Variability Across Uncertainty Sources')
save_fig(fig, 'S00_summary_stdev')

# Save stats
def convert(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    return o
with open(os.path.join(DDIR, 'all_stats.json'), 'w') as f:
    json.dump(ALL_STATS, f, indent=2, default=convert)

print(f"\nALL DONE! Figures: {FDIR}  Data: {DDIR}")
