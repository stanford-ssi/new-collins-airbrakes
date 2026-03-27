"""
Monte Carlo simulation framework.
Runs multiple simulations with varied parameters to assess system robustness.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import multiprocessing as mp
from functools import partial

from simulation import SimulationEnvironment, SimulationResults, create_simulation
from config import DEFAULT_MONTE_CARLO_RUNS, METERS_TO_FEET


@dataclass
class MonteCarloConfig:
    """Configuration for Monte Carlo simulation."""
    
    num_runs: int = DEFAULT_MONTE_CARLO_RUNS
    
    cd_body_error_mean_low: float = 0.0
    cd_body_error_std_low: float = 0.05
    cd_body_error_mean_high: float = 0.0
    cd_body_error_std_high: float = 0.05
    
    cd_airbrake_error_mean_low: float = 0.0
    cd_airbrake_error_std_low: float = 0.05
    cd_airbrake_error_mean_high: float = 0.0
    cd_airbrake_error_std_high: float = 0.05
    
    sensor_noise_mult_mean: float = 1.0
    sensor_noise_mult_std: float = 0.2
    
    base_seed: int = 42
    
    def generate_parameters(self, run_index: int) -> dict:
        """
        Generate randomized parameters for a single run.
        
        Args:
            run_index: Index of this run
            
        Returns:
            Dictionary of parameters for this run
        """
        rng = np.random.RandomState(self.base_seed + run_index)
        
        cd_body_low = rng.normal(self.cd_body_error_mean_low, self.cd_body_error_std_low)
        cd_body_high = rng.normal(self.cd_body_error_mean_high, self.cd_body_error_std_high)
        
        cd_ab_low = rng.normal(self.cd_airbrake_error_mean_low, self.cd_airbrake_error_std_low)
        cd_ab_high = rng.normal(self.cd_airbrake_error_mean_high, self.cd_airbrake_error_std_high)
        
        noise_mult = max(0.1, rng.normal(self.sensor_noise_mult_mean, self.sensor_noise_mult_std))
        
        return {
            'cd_error_body': (cd_body_low, cd_body_high),
            'cd_error_airbrake': (cd_ab_low, cd_ab_high),
            'sensor_noise_mult': noise_mult,
            'seed': self.base_seed + run_index,
        }


@dataclass
class MonteCarloResults:
    """Results from Monte Carlo simulation."""
    
    num_runs: int = 0
    apogees_m: np.ndarray = field(default_factory=lambda: np.array([]))
    apogees_ft: np.ndarray = field(default_factory=lambda: np.array([]))
    max_velocities: np.ndarray = field(default_factory=lambda: np.array([]))
    max_machs: np.ndarray = field(default_factory=lambda: np.array([]))
    burnout_altitudes: np.ndarray = field(default_factory=lambda: np.array([]))
    
    individual_results: List[SimulationResults] = field(default_factory=list)
    run_parameters: List[dict] = field(default_factory=list)
    
    apogee_mean_m: float = 0.0
    apogee_std_m: float = 0.0
    apogee_mean_ft: float = 0.0
    apogee_std_ft: float = 0.0
    apogee_min_ft: float = 0.0
    apogee_max_ft: float = 0.0
    
    def compute_statistics(self):
        """Compute aggregate statistics."""
        if len(self.apogees_m) > 0:
            self.apogee_mean_m = np.mean(self.apogees_m)
            self.apogee_std_m = np.std(self.apogees_m)
            self.apogee_mean_ft = self.apogee_mean_m * METERS_TO_FEET
            self.apogee_std_ft = self.apogee_std_m * METERS_TO_FEET
            self.apogee_min_ft = np.min(self.apogees_ft)
            self.apogee_max_ft = np.max(self.apogees_ft)
            
    def get_percentile(self, percentile: float) -> float:
        """Get apogee at given percentile in feet."""
        return np.percentile(self.apogees_ft, percentile)
    
    def get_success_rate(self, target_ft: float, tolerance_ft: float = 500) -> float:
        """
        Get fraction of runs within tolerance of target.
        
        Args:
            target_ft: Target apogee in feet
            tolerance_ft: Acceptable tolerance in feet
            
        Returns:
            Fraction of successful runs (0 to 1)
        """
        within_tolerance = np.abs(self.apogees_ft - target_ft) <= tolerance_ft
        return np.mean(within_tolerance)


def _run_single_sim(
    run_index: int,
    base_config: dict,
    mc_config: MonteCarloConfig,
) -> SimulationResults:
    """
    Run a single simulation with Monte Carlo parameters.
    
    Args:
        run_index: Index of this run
        base_config: Base simulation configuration
        mc_config: Monte Carlo configuration
        
    Returns:
        SimulationResults from this run
    """
    params = mc_config.generate_parameters(run_index)
    
    config = base_config.copy()
    config.update(params)
    
    sim = create_simulation(**config)
    results = sim.run()
    
    return results


class MonteCarloSimulation:
    """
    Monte Carlo simulation runner.
    """
    
    def __init__(
        self,
        base_config: dict,
        mc_config: MonteCarloConfig = None,
    ):
        """
        Initialize Monte Carlo simulation.
        
        Args:
            base_config: Base simulation configuration dictionary
            mc_config: Monte Carlo configuration
        """
        self.base_config = base_config
        self.mc_config = mc_config if mc_config else MonteCarloConfig()
        
    def run(
        self,
        num_runs: int = None,
        parallel: bool = False,
        progress_callback: callable = None,
    ) -> MonteCarloResults:
        """
        Run Monte Carlo simulation.
        
        Args:
            num_runs: Number of runs (overrides mc_config if provided)
            parallel: Whether to run in parallel (experimental)
            progress_callback: Callback function(run_index, total_runs)
            
        Returns:
            MonteCarloResults with aggregate statistics
        """
        if num_runs is not None:
            self.mc_config.num_runs = num_runs
            
        results = MonteCarloResults(num_runs=self.mc_config.num_runs)
        
        apogees_m = []
        apogees_ft = []
        max_velocities = []
        max_machs = []
        burnout_altitudes = []
        
        for i in range(self.mc_config.num_runs):
            if progress_callback:
                progress_callback(i, self.mc_config.num_runs)
                
            params = self.mc_config.generate_parameters(i)
            results.run_parameters.append(params)
            
            config = self.base_config.copy()
            config.update(params)
            
            sim = create_simulation(**config)
            sim_results = sim.run()
            
            results.individual_results.append(sim_results)
            apogees_m.append(sim_results.apogee_m)
            apogees_ft.append(sim_results.apogee_ft)
            max_velocities.append(sim_results.max_velocity)
            max_machs.append(sim_results.max_mach)
            burnout_altitudes.append(sim_results.burnout_altitude)
            
        results.apogees_m = np.array(apogees_m)
        results.apogees_ft = np.array(apogees_ft)
        results.max_velocities = np.array(max_velocities)
        results.max_machs = np.array(max_machs)
        results.burnout_altitudes = np.array(burnout_altitudes)
        
        results.compute_statistics()
        
        return results
    
    def run_single(self, run_index: int = 0) -> SimulationResults:
        """
        Run a single simulation with specified run index parameters.
        
        Args:
            run_index: Index to use for parameter generation
            
        Returns:
            SimulationResults
        """
        params = self.mc_config.generate_parameters(run_index)
        config = self.base_config.copy()
        config.update(params)
        
        sim = create_simulation(**config)
        return sim.run()


def run_monte_carlo(
    target_apogee_m: float = None,
    enable_airbrakes: bool = True,
    num_runs: int = 100,
    cd_body_error_std: float = 0.05,
    cd_airbrake_error_std: float = 0.05,
    sensor_noise_std: float = 0.2,
    seed: int = 42,
    progress_callback: callable = None,
) -> MonteCarloResults:
    """
    Convenience function to run Monte Carlo simulation.
    Uses rocket data from tables/ directory.
    
    Args:
        target_apogee_m: Target apogee in meters (default from config)
        enable_airbrakes: Whether airbrakes are enabled
        num_runs: Number of Monte Carlo runs
        cd_body_error_std: Std dev of body Cd error
        cd_airbrake_error_std: Std dev of airbrake Cd error
        sensor_noise_std: Std dev of sensor noise multiplier
        seed: Random seed
        progress_callback: Progress callback function
        
    Returns:
        MonteCarloResults
    """
    base_config = {
        'target_apogee_m': target_apogee_m,
        'enable_airbrakes': enable_airbrakes,
    }
    
    mc_config = MonteCarloConfig(
        num_runs=num_runs,
        cd_body_error_std_low=cd_body_error_std,
        cd_body_error_std_high=cd_body_error_std,
        cd_airbrake_error_std_low=cd_airbrake_error_std,
        cd_airbrake_error_std_high=cd_airbrake_error_std,
        sensor_noise_mult_std=sensor_noise_std,
        base_seed=seed,
    )
    
    mc_sim = MonteCarloSimulation(base_config, mc_config)
    return mc_sim.run(progress_callback=progress_callback)
