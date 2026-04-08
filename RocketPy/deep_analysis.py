#!/usr/bin/env python3
"""
Deep analysis: fine-resolution sweeps at critical thresholds,
interaction effects, and failure timing studies.
"""
import sys, os, json, multiprocessing
multiprocessing.set_start_method('fork', force=True)
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.stats as stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simulation import create_simulation
from config import (TARGET_APOGEE_M, METERS_TO_FEET, PRESSURE_NOISE_STD,
                    ACCEL_Z_NOISE_STD, AIRBRAKE_MAX_AREA_M2)
from debug import configure_debug, DebugLevel
from sensors import SensorModel
configure_debug(level=DebugLevel.OFF)

FDIR = 'study_output/figures'
TARGET_FT = TARGET_APOGEE_M * METERS_TO_FEET
NUM_WORKERS = 8

plt.rcParams.update({
    'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 12,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.grid': True, 'grid.alpha': 0.3, 'lines.linewidth': 1.8,
    'font.family': 'serif',
})

def _init():
    import debug; debug.configure_debug(level=debug.DebugLevel.OFF)
    from truth_model import reset_truth_model; reset_truth_model()

def _run(cfg):
    sim = create_simulation(**cfg)
    sim._record_truth_predictions = lambda: None
    sim.dt = 0.005; sim.control_dt = 0.05
    r = sim.run()
    return r.apogee_ft

def sweep_mc(base, param, values, runs_per=200, label=""):
    """Fine sweep with MC at each point."""
    all_cfgs = []
    for j, v in enumerate(values):
        for i in range(runs_per):
            cfg = base.copy(); cfg[param] = v; cfg['seed'] = 9000 + j*runs_per + i
            all_cfgs.append(cfg)
    print(f"  {label}: running {len(all_cfgs)} sims...")
    with multiprocessing.Pool(NUM_WORKERS, initializer=_init) as pool:
        results = list(pool.imap(_run, all_cfgs, chunksize=25))
    # Reorganize
    mu, sig, s500, s1000 = [], [], [], []
    for j in range(len(values)):
        aps = np.array(results[j*runs_per:(j+1)*runs_per])
        mu.append(np.mean(aps)); sig.append(np.std(aps))
        s500.append(np.mean(np.abs(aps - TARGET_FT) <= 500) * 100)
        s1000.append(np.mean(np.abs(aps - TARGET_FT) <= 1000) * 100)
    return np.array(mu), np.array(sig), np.array(s500), np.array(s1000)

# ============================================================
# 1. FINE Cd UNCERTAINTY SWEEP (1% to 20% in 1% steps)
# ============================================================
print("=== 1. Fine Cd Uncertainty Sweep ===")
cd_pcts = np.arange(1, 21)
cd_stds = cd_pcts / 100.0
cd_cfgs_all = []
runs_per = 300
for j, std in enumerate(cd_stds):
    rng = np.random.RandomState(10000 + j)
    for i in range(runs_per):
        e0, e2 = rng.normal(0, std), rng.normal(0, std)
        cd_cfgs_all.append(dict(cd_scale_mach0=1+e0, cd_scale_mach2=1+e2, seed=10000+j*runs_per+i))
print(f"  Running {len(cd_cfgs_all)} sims...")
with multiprocessing.Pool(NUM_WORKERS, initializer=_init) as pool:
    cd_results = list(pool.imap(_run, cd_cfgs_all, chunksize=25))
cd_mu, cd_sig, cd_s500, cd_s1000 = [], [], [], []
for j in range(len(cd_pcts)):
    aps = np.array(cd_results[j*runs_per:(j+1)*runs_per])
    cd_mu.append(np.mean(aps)); cd_sig.append(np.std(aps))
    cd_s500.append(np.mean(np.abs(aps - TARGET_FT) <= 500) * 100)
    cd_s1000.append(np.mean(np.abs(aps - TARGET_FT) <= 1000) * 100)
cd_mu, cd_sig, cd_s500, cd_s1000 = map(np.array, [cd_mu, cd_sig, cd_s500, cd_s1000])

fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.5))
a1.plot(cd_pcts, cd_sig, 'o-', color='#2196F3', ms=5, label='$\\sigma_{\\mathrm{apogee}}$')
a1.fill_between(cd_pcts, cd_mu - TARGET_FT - cd_sig, cd_mu - TARGET_FT + cd_sig, alpha=0.15, color='#2196F3')
a1.set_xlabel('$C_D$ Uncertainty $\\sigma$ (%)'); a1.set_ylabel('Apogee Scatter $\\sigma$ (ft)')
a1.set_title('Apogee Scatter vs. Drag Coefficient Uncertainty')
a1.axhline(500, color='green', ls=':', alpha=0.7, label='$\\pm$500 ft threshold')
# Mark where sigma crosses 500
cross_idx = np.where(cd_sig > 500)[0]
if len(cross_idx) > 0:
    cross_pct = cd_pcts[cross_idx[0]]
    a1.axvline(cross_pct, color='red', ls='--', alpha=0.7, label=f'$\\sigma > 500$ ft at {cross_pct}%')
a1.legend(fontsize=9)

a2.plot(cd_pcts, cd_s500, 'o-', color='#4CAF50', ms=5, label='Within $\\pm$500 ft')
a2.plot(cd_pcts, cd_s1000, 's-', color='#2196F3', ms=4, label='Within $\\pm$1000 ft')
a2.set_xlabel('$C_D$ Uncertainty $\\sigma$ (%)'); a2.set_ylabel('Success Rate (%)')
a2.set_title('Success Rate vs. Drag Coefficient Uncertainty')
a2.set_ylim(0, 105)
# Mark 90% line
a2.axhline(90, color='gray', ls=':', alpha=0.5)
for pct_thresh in [90, 95]:
    idx = np.where(cd_s500 < pct_thresh)[0]
    if len(idx) > 0:
        a2.annotate(f'{pct_thresh}% at ~{cd_pcts[idx[0]-1]}%', (cd_pcts[idx[0]-1], pct_thresh),
                   fontsize=8, xytext=(5, -15), textcoords='offset points')
a2.legend(fontsize=9)
fig.tight_layout()
fig.savefig(f'{FDIR}/S_deep_cd_fine.pdf', bbox_inches='tight'); plt.close(fig)
print(f"  Done. Sigma crosses 500 ft near Cd uncertainty of {cd_pcts[cross_idx[0]] if len(cross_idx)>0 else '>20'}%")

# ============================================================
# 2. FINE SLEW RATE SWEEP (5 to 100 deg/s in 5 deg/s steps)
# ============================================================
print("\n=== 2. Fine Slew Rate Sweep ===")
slew_rates = np.arange(5, 105, 5)
slew_mu, slew_sig, slew_s500, slew_s1000 = sweep_mc(
    {}, 'airbrake_slew_rate_deg_s', slew_rates, runs_per=150, label='Slew Fine')

fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.5))
a1.plot(slew_rates, slew_mu, 'o-', color='#FF5722', ms=5)
a1.axhline(TARGET_FT, color='red', ls='--', alpha=0.5)
a1.axhspan(TARGET_FT-500, TARGET_FT+500, alpha=0.05, color='green')
a1.set_xlabel('Slew Rate (deg/s)'); a1.set_ylabel('Mean Apogee (ft)')
a1.set_title('Mean Apogee vs. Slew Rate (Zoomed)')
# Annotate the knee
for rate, mean in zip(slew_rates, slew_mu):
    if abs(mean - TARGET_FT) < 50:
        a1.annotate(f'Converges at ~{rate} deg/s', (rate, mean),
                   fontsize=9, xytext=(10, 20), textcoords='offset points',
                   arrowprops=dict(arrowstyle='->', color='black'))
        break

a2.plot(slew_rates, slew_mu - TARGET_FT, 'o-', color='#FF5722', ms=5)
a2.axhline(0, color='black', lw=0.8)
a2.axhline(500, color='green', ls=':', alpha=0.5); a2.axhline(-500, color='green', ls=':', alpha=0.5)
a2.set_xlabel('Slew Rate (deg/s)'); a2.set_ylabel('Mean Apogee Error (ft)')
a2.set_title('Apogee Error vs. Slew Rate')
fig.tight_layout()
fig.savefig(f'{FDIR}/S_deep_slew_fine.pdf', bbox_inches='tight'); plt.close(fig)

# ============================================================
# 3. FINE AREA SWEEP (20% to 80% in 5% steps)
# ============================================================
print("\n=== 3. Fine Area Sweep ===")
area_pcts = np.arange(20, 85, 5)
area_vals = AIRBRAKE_MAX_AREA_M2 * area_pcts / 100
area_mu, area_sig, area_s500, area_s1000 = sweep_mc(
    {}, 'airbrake_max_area_m2', area_vals, runs_per=150, label='Area Fine')

fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.5))
a1.plot(area_pcts, area_mu, 'o-', color='#9C27B0', ms=5)
a1.axhline(TARGET_FT, color='red', ls='--', alpha=0.5)
a1.axhspan(TARGET_FT-500, TARGET_FT+500, alpha=0.05, color='green')
a1.set_xlabel('Available Airbrake Area (%)'); a1.set_ylabel('Mean Apogee (ft)')
a1.set_title('Mean Apogee vs. Available Area (Zoomed)')
# Find where mean exits ±500 band
for i, (pct, m) in enumerate(zip(area_pcts, area_mu)):
    if m > TARGET_FT + 500:
        a1.axvline(pct, color='red', ls='--', alpha=0.5, label=f'Exits $\\pm$500 ft at {pct}%')
        a1.legend(fontsize=9)
        break

a2.plot(area_pcts, area_s500, 'o-', color='#4CAF50', ms=5, label='$\\pm$500 ft')
a2.plot(area_pcts, area_s1000, 's-', color='#2196F3', ms=4, label='$\\pm$1000 ft')
a2.set_xlabel('Available Area (%)'); a2.set_ylabel('Success Rate (%)')
a2.set_title('Success Rate vs. Available Area')
a2.set_ylim(0, 105); a2.legend(fontsize=9)
fig.tight_layout()
fig.savefig(f'{FDIR}/S_deep_area_fine.pdf', bbox_inches='tight'); plt.close(fig)

# ============================================================
# 4. SENSOR FAILURE TIMING SWEEP
# ============================================================
print("\n=== 4. Sensor Failure Timing Sweep ===")

class TimedFailingSensor(SensorModel):
    def __init__(self, failure_call, **kw):
        super().__init__(**kw)
        self.failure_call = failure_call
        self.calls = 0
        self._stuck_p = None
    def get_measurements(self, tp, tt, ta):
        self.calls += 1
        p, t, a = super().get_measurements(tp, tt, ta)
        if self.calls <= self.failure_call:
            self._stuck_p = p
            return p, t, a
        return self._stuck_p or tp, t, a

# Failure at different times (call count = time * 100 Hz)
# Coast phase is roughly t=5s to t=25s, so calls 500-2500
failure_calls = np.arange(200, 2400, 100)  # Every 1 second from t=2s to t=24s
failure_times = failure_calls / 100.0
failure_apogees = []
for fc in failure_calls:
    sim = create_simulation(seed=42)
    sim.sensors = TimedFailingSensor(int(fc), seed=42)
    r = sim.run()
    failure_apogees.append(r.apogee_ft)
    print(f"  Baro stuck at t={fc/100:.1f}s: apogee={r.apogee_ft:.0f} ft")
failure_apogees = np.array(failure_apogees)

fig, ax = plt.subplots(figsize=(10, 5.5))
colors = ['#F44336' if abs(a - TARGET_FT) > 500 else '#4CAF50' for a in failure_apogees]
ax.bar(failure_times, failure_apogees - TARGET_FT, width=0.8, color=colors, alpha=0.8)
ax.axhline(0, color='black', lw=0.8)
ax.axhline(500, color='green', ls=':', alpha=0.5, label='$\\pm$500 ft')
ax.axhline(-500, color='green', ls=':', alpha=0.5)
ax.set_xlabel('Time of Barometer Failure (s)')
ax.set_ylabel('Apogee Error (ft)')
ax.set_title('Impact of Barometer Failure vs. Failure Timing')
# Annotate phases
ax.axvspan(0, 5, alpha=0.05, color='orange', label='Boost phase')
ax.axvspan(5, 8, alpha=0.05, color='blue', label='Transonic lockout')
ax.axvspan(8, 22, alpha=0.05, color='green', label='Active coast')
ax.legend(fontsize=8, loc='upper right')
fig.tight_layout()
fig.savefig(f'{FDIR}/S_deep_failure_timing.pdf', bbox_inches='tight'); plt.close(fig)

# ============================================================
# 5. Cd × THRUST INTERACTION (2D heatmap)
# ============================================================
print("\n=== 5. Cd x Thrust Interaction ===")
cd_levels = np.array([0, 3, 5, 7, 10, 13, 15, 18, 20])
thrust_levels = np.array([0, 1, 2, 3, 4, 5, 7, 10])
runs_per_cell = 100
interaction_cfgs = []
for ci, cd_std in enumerate(cd_levels / 100.0):
    for ti, t_std in enumerate(thrust_levels / 100.0):
        rng = np.random.RandomState(20000 + ci*100 + ti)
        for k in range(runs_per_cell):
            cfg = dict(
                cd_scale_mach0=1 + rng.normal(0, max(cd_std, 0.001)),
                cd_scale_mach2=1 + rng.normal(0, max(cd_std, 0.001)),
                thrust_scale=max(0.7, rng.normal(1.0, max(t_std, 0.001))),
                seed=20000 + ci*1000 + ti*100 + k
            )
            interaction_cfgs.append((ci, ti, cfg))

print(f"  Running {len(interaction_cfgs)} sims...")
just_cfgs = [c for _, _, c in interaction_cfgs]
with multiprocessing.Pool(NUM_WORKERS, initializer=_init) as pool:
    interaction_results = list(pool.imap(_run, just_cfgs, chunksize=25))

# Build grid
grid_mu = np.zeros((len(cd_levels), len(thrust_levels)))
grid_sig = np.zeros_like(grid_mu)
grid_s500 = np.zeros_like(grid_mu)
from collections import defaultdict
cell_data = defaultdict(list)
for idx, (ci, ti, _) in enumerate(interaction_cfgs):
    cell_data[(ci, ti)].append(interaction_results[idx])
for ci in range(len(cd_levels)):
    for ti in range(len(thrust_levels)):
        aps = np.array(cell_data[(ci, ti)])
        grid_mu[ci, ti] = np.mean(aps)
        grid_sig[ci, ti] = np.std(aps)
        grid_s500[ci, ti] = np.mean(np.abs(aps - TARGET_FT) <= 500) * 100

fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.5))
im1 = a1.imshow(grid_sig, origin='lower', aspect='auto', cmap='YlOrRd',
               extent=[thrust_levels[0]-0.5, thrust_levels[-1]+0.5,
                       cd_levels[0]-0.5, cd_levels[-1]+0.5])
a1.set_xlabel('Thrust Uncertainty $\\sigma$ (%)'); a1.set_ylabel('$C_D$ Uncertainty $\\sigma$ (%)')
a1.set_title('Apogee Scatter $\\sigma$ (ft)')
plt.colorbar(im1, ax=a1, label='$\\sigma$ (ft)')
# Add contour lines
for val in [200, 500, 1000]:
    cs = a1.contour(thrust_levels, cd_levels, grid_sig, levels=[val], colors='black', linewidths=1)
    a1.clabel(cs, fmt='%d', fontsize=8)

im2 = a2.imshow(grid_s500, origin='lower', aspect='auto', cmap='RdYlGn',
               extent=[thrust_levels[0]-0.5, thrust_levels[-1]+0.5,
                       cd_levels[0]-0.5, cd_levels[-1]+0.5],
               vmin=0, vmax=100)
a2.set_xlabel('Thrust Uncertainty $\\sigma$ (%)'); a2.set_ylabel('$C_D$ Uncertainty $\\sigma$ (%)')
a2.set_title('Success Rate Within $\\pm$500 ft (%)')
plt.colorbar(im2, ax=a2, label='Success Rate (%)')
for val in [50, 75, 90]:
    cs = a2.contour(thrust_levels, cd_levels, grid_s500, levels=[val], colors='black', linewidths=1)
    a2.clabel(cs, fmt='%d%%', fontsize=8)

fig.suptitle('Interaction Between $C_D$ and Thrust Uncertainty', fontsize=14)
fig.tight_layout()
fig.savefig(f'{FDIR}/S_deep_cd_thrust_interaction.pdf', bbox_inches='tight'); plt.close(fig)

# ============================================================
# 6. CONTROLLER AUTHORITY TIMELINE (from single detailed run)
# ============================================================
print("\n=== 6. Controller Authority Timeline ===")
sim = create_simulation(seed=42)
r = sim.run()
tel = r.controller_telemetry
if tel and len(tel.get('time', [])) > 0:
    t = np.array(tel['time'])
    pred = np.array(tel['predicted_apogee']) * METERS_TO_FEET
    clean = np.array(tel['apogee_clean']) * METERS_TO_FEET
    full = np.array(tel['apogee_full_brake']) * METERS_TO_FEET
    angle = np.array(tel['commanded_angle'])

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    # Authority envelope
    a1.fill_between(t, full, clean, alpha=0.2, color='#2196F3', label='Control authority envelope')
    a1.plot(t, pred, color='#2196F3', lw=2, label='Predicted apogee')
    a1.plot(t, clean, color='#FF5722', lw=1, ls='--', alpha=0.7, label='No-brake apogee')
    a1.plot(t, full, color='#4CAF50', lw=1, ls='--', alpha=0.7, label='Full-brake apogee')
    a1.axhline(TARGET_FT, color='red', lw=2, ls=':', label='Target')
    a1.set_ylabel('Predicted Apogee (ft)')
    a1.set_title('Controller Authority Over Time')
    a1.legend(fontsize=8, loc='upper right')
    # Show where authority shrinks to zero
    authority = clean - full
    authority_mask = authority > 0
    if np.any(authority_mask):
        a1.set_ylim(min(full[authority_mask].min(), TARGET_FT - 1000) - 200,
                    max(clean[authority_mask].max(), TARGET_FT + 1000) + 200)

    # Deployment angle + remaining authority
    a2.plot(t, angle, color='#9C27B0', lw=2, label='Deployment angle')
    a2_twin = a2.twinx()
    a2_twin.plot(t, authority, color='#FF9800', lw=1.5, ls='--', label='Authority (ft)')
    a2.set_xlabel('Time (s)'); a2.set_ylabel('Deployment Angle (deg)')
    a2_twin.set_ylabel('Remaining Authority (ft)', color='#FF9800')
    a2.set_title('Deployment Profile and Remaining Authority')
    lines1, labels1 = a2.get_legend_handles_labels()
    lines2, labels2 = a2_twin.get_legend_handles_labels()
    a2.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper right')

    fig.tight_layout()
    fig.savefig(f'{FDIR}/S_deep_authority_timeline.pdf', bbox_inches='tight')
    plt.close(fig)

print("\n=== All deep analysis complete ===")

# Save key numbers
deep_stats = {
    'cd_fine': {
        'pcts': cd_pcts.tolist(), 'sigma': cd_sig.tolist(),
        's500': cd_s500.tolist(), 's1000': cd_s1000.tolist(),
    },
    'failure_timing': {
        'times': failure_times.tolist(), 'errors': (failure_apogees - TARGET_FT).tolist(),
    },
}
with open('study_output/data/deep_stats.json', 'w') as f:
    json.dump(deep_stats, f, indent=2)
print("Saved deep_stats.json")
