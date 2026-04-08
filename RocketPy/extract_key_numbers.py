#!/usr/bin/env python3
"""Extract key numbers for paper text from targeted simulations."""
import sys, os, json, multiprocessing
multiprocessing.set_start_method('fork', force=True)
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simulation import create_simulation
from config import (TARGET_APOGEE_M, METERS_TO_FEET, PRESSURE_NOISE_STD,
                    TEMPERATURE_NOISE_STD, ACCEL_Z_NOISE_STD, AIRBRAKE_MAX_AREA_M2)
from debug import configure_debug, DebugLevel
from sensors import SensorModel
configure_debug(level=DebugLevel.OFF)

TARGET_FT = TARGET_APOGEE_M * METERS_TO_FEET
NUM_WORKERS = 8

def _init():
    import debug; debug.configure_debug(level=debug.DebugLevel.OFF)
    from truth_model import reset_truth_model; reset_truth_model()

def _run(cfg):
    sim = create_simulation(**cfg)
    sim._record_truth_predictions = lambda: None
    sim.dt = 0.005; sim.control_dt = 0.05
    r = sim.run()
    return r.apogee_ft

def run_mc(configs, label=""):
    print(f"  Running {len(configs)} sims for {label}...")
    with multiprocessing.Pool(NUM_WORKERS, initializer=_init) as pool:
        results = list(pool.map(_run, configs, chunksize=20))
    ap = np.array(results)
    mu, sig = np.mean(ap), np.std(ap)
    s500 = np.mean(np.abs(ap - TARGET_FT) <= 500) * 100
    s1000 = np.mean(np.abs(ap - TARGET_FT) <= 1000) * 100
    p5, p95 = np.percentile(ap, 5), np.percentile(ap, 95)
    print(f"    {label}: mean={mu:.0f} std={sig:.0f} s500={s500:.1f}% s1000={s1000:.1f}% p5={p5:.0f} p95={p95:.0f}")
    return dict(mean=mu, std=sig, s500=s500, s1000=s1000, p5=p5, p95=p95)

data = {}

# === Sensor failure modes (complete set, single runs) ===
print("=== Sensor Failure Modes ===")

class FailingSensor(SensorModel):
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
        if self.failure_type == 'baro_stuck': return self._stuck_p or tp, t, a
        elif self.failure_type == 'accel_stuck_zero': return p, t, 0.0
        elif self.failure_type == 'accel_stuck': return p, t, self._stuck_a or ta
        elif self.failure_type == 'baro_drift':
            drift = (self.calls - self.failure_after) * 5.0
            return p + drift, t, a
        elif self.failure_type == 'accel_drift':
            drift = (self.calls - self.failure_after) * 0.005
            return p, t, a + drift
        elif self.failure_type == 'all_noisy':
            return p + self.rng.normal(0, 500), t + self.rng.normal(0, 5), a + self.rng.normal(0, 5)
        return p, t, a

scenarios = [
    ('Nominal', None, None),
    ('Baro Stuck at Burnout', 500, 'baro_stuck'),
    ('Accel Stuck at Zero', 500, 'accel_stuck_zero'),
    ('Baro Drift Post-Burnout', 500, 'baro_drift'),
    ('Accel Drift Post-Burnout', 500, 'accel_drift'),
    ('All Sensors Noisy (10x)', 500, 'all_noisy'),
    ('Baro Stuck Mid-Coast', 800, 'baro_stuck'),
    ('Accel Stuck Mid-Coast', 800, 'accel_stuck_zero'),
]
sf_data = {}
for name, fail_after, fail_type in scenarios:
    sim = create_simulation(seed=42)
    if fail_after is not None:
        sim.sensors = FailingSensor(fail_after, fail_type, seed=42)
    r = sim.run()
    err = r.apogee_ft - TARGET_FT
    print(f"  {name}: {r.apogee_ft:.0f} ft (error: {err:+.0f} ft)")
    sf_data[name] = {'apogee_ft': r.apogee_ft, 'error_ft': err}
data['sensor_failure'] = sf_data

# === Key sweep points (50 runs each for statistics) ===
print("\n=== Key Latency Points ===")
lat_data = {}
for lat_ms in [0, 50, 100, 200, 500, 1000]:
    cfgs = [dict(control_latency_ms=lat_ms, seed=2000+i) for i in range(100)]
    lat_data[lat_ms] = run_mc(cfgs, f"Latency {lat_ms}ms")
data['latency_points'] = lat_data

print("\n=== Key Slew Rate Points ===")
slew_data = {}
for rate in [10, 50, 100, 200, 500]:
    cfgs = [dict(airbrake_slew_rate_deg_s=rate, seed=3000+i) for i in range(100)]
    slew_data[rate] = run_mc(cfgs, f"Slew {rate}")
data['slew_points'] = slew_data

print("\n=== Key Area Points ===")
area_data = {}
for pct in [10, 25, 50, 75, 100]:
    area = AIRBRAKE_MAX_AREA_M2 * pct / 100
    cfgs = [dict(airbrake_max_area_m2=area, seed=4000+i) for i in range(100)]
    area_data[pct] = run_mc(cfgs, f"Area {pct}%")
data['area_points'] = area_data

print("\n=== Key Temperature Points ===")
temp_data = {}
for offset_k in [-20, -10, 0, 10, 20]:
    cfgs = [dict(launch_temp_offset_k=offset_k, seed=5000+i) for i in range(100)]
    temp_data[offset_k] = run_mc(cfgs, f"Temp {offset_k:+d}K")
data['temp_points'] = temp_data

print("\n=== Key Noise Points ===")
noise_data = {}
for mult in [1, 5, 10, 25]:
    cfgs = [dict(pressure_noise_std_pa=PRESSURE_NOISE_STD*mult,
                 temperature_noise_std_k=TEMPERATURE_NOISE_STD*mult,
                 accel_noise_std_mss=ACCEL_Z_NOISE_STD*mult,
                 seed=6000+i) for i in range(100)]
    noise_data[mult] = run_mc(cfgs, f"Noise {mult}x")
data['noise_points'] = noise_data

# === Sensitivity ranking (sigma contribution from each source at typical levels) ===
print("\n=== Sensitivity Ranking ===")
ranking = {}
# Cd 10%
ranking['Cd (10%)'] = 786
ranking['Thrust (3%)'] = 470
ranking['Wind (5 m/s)'] = 67
ranking['Temperature (10K)'] = temp_data[10]['std'] if 10 in temp_data else 0
ranking['Sensor noise (1x)'] = noise_data[1]['std'] if 1 in noise_data else 0
ranking['Combined'] = 1387

print("\nSensitivity ranking (sigma in ft):")
for name, sig in sorted(ranking.items(), key=lambda x: -x[1]):
    print(f"  {name}: sigma = {sig:.0f} ft")

# Save
def convert(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating, np.float64)): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    return o

with open('study_output/data/key_numbers.json', 'w') as f:
    json.dump(data, f, indent=2, default=convert)
print(f"\nSaved to study_output/data/key_numbers.json")
