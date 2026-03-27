"""
Simulation storage module for saving and loading past simulations.
Stores simulations as JSON files in a saved_simulations/ directory.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import asdict
import numpy as np

from simulation import SimulationResults


STORAGE_DIR = Path(__file__).parent / "saved_simulations"


def _ensure_storage_dir():
    """Ensure the storage directory exists."""
    STORAGE_DIR.mkdir(exist_ok=True)


def _numpy_to_list(obj):
    """Convert numpy arrays to lists for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _numpy_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_numpy_to_list(item) for item in obj]
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    return obj


def _list_to_numpy(obj):
    """Convert lists back to numpy arrays for SimulationResults fields."""
    if isinstance(obj, list) and len(obj) > 0 and isinstance(obj[0], (int, float)):
        return np.array(obj)
    elif isinstance(obj, dict):
        return {k: _list_to_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_list_to_numpy(item) for item in obj]
    return obj


def generate_simulation_id() -> str:
    """Generate a unique simulation ID based on timestamp."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_simulation(
    results: SimulationResults,
    name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    sim_id: Optional[str] = None,
) -> str:
    """
    Save a simulation to disk.
    
    Args:
        results: SimulationResults to save
        name: Optional human-readable name for the simulation
        config: Optional configuration dict used for the simulation
        sim_id: Optional custom simulation ID (auto-generated if not provided)
        
    Returns:
        The simulation ID
    """
    _ensure_storage_dir()
    
    if sim_id is None:
        sim_id = generate_simulation_id()
    
    # Build metadata
    metadata = {
        "id": sim_id,
        "name": name or f"Simulation {sim_id}",
        "timestamp": datetime.now().isoformat(),
        "apogee_ft": results.apogee_ft,
        "apogee_m": results.apogee_m,
        "max_velocity": results.max_velocity,
        "max_mach": results.max_mach,
        "burnout_time": results.burnout_time,
        "apogee_time": results.apogee_time,
    }
    
    if config:
        metadata["config"] = config
    
    # Convert results to serializable dict
    results_dict = {
        "time": results.time,
        "altitude": results.altitude,
        "velocity": results.velocity,
        "acceleration": results.acceleration,
        "mach": results.mach,
        "mass": results.mass,
        "thrust": results.thrust,
        "body_drag": results.body_drag,
        "airbrake_drag": results.airbrake_drag,
        "airbrake_angle": results.airbrake_angle,
        "pressure": results.pressure,
        "temperature": results.temperature,
        "apogee_m": results.apogee_m,
        "apogee_ft": results.apogee_ft,
        "apogee_time": results.apogee_time,
        "burnout_altitude": results.burnout_altitude,
        "burnout_velocity": results.burnout_velocity,
        "burnout_time": results.burnout_time,
        "max_velocity": results.max_velocity,
        "max_mach": results.max_mach,
        "max_acceleration": results.max_acceleration,
        "controller_telemetry": results.controller_telemetry,
    }
    
    data = {
        "metadata": metadata,
        "results": _numpy_to_list(results_dict),
    }
    
    filepath = STORAGE_DIR / f"{sim_id}.json"
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"💾 Simulation saved: {filepath}")
    return sim_id


def load_simulation(sim_id: str) -> Optional[SimulationResults]:
    """
    Load a simulation from disk.
    
    Args:
        sim_id: The simulation ID to load
        
    Returns:
        SimulationResults or None if not found
    """
    filepath = STORAGE_DIR / f"{sim_id}.json"
    
    if not filepath.exists():
        print(f"⚠️ Simulation not found: {sim_id}")
        return None
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    results_dict = _list_to_numpy(data["results"])
    
    results = SimulationResults(
        time=np.array(results_dict.get("time", [])),
        altitude=np.array(results_dict.get("altitude", [])),
        velocity=np.array(results_dict.get("velocity", [])),
        acceleration=np.array(results_dict.get("acceleration", [])),
        mach=np.array(results_dict.get("mach", [])),
        mass=np.array(results_dict.get("mass", [])),
        thrust=np.array(results_dict.get("thrust", [])),
        body_drag=np.array(results_dict.get("body_drag", [])),
        airbrake_drag=np.array(results_dict.get("airbrake_drag", [])),
        airbrake_angle=np.array(results_dict.get("airbrake_angle", [])),
        pressure=np.array(results_dict.get("pressure", [])),
        temperature=np.array(results_dict.get("temperature", [])),
        apogee_m=results_dict.get("apogee_m", 0.0),
        apogee_ft=results_dict.get("apogee_ft", 0.0),
        apogee_time=results_dict.get("apogee_time", 0.0),
        burnout_altitude=results_dict.get("burnout_altitude", 0.0),
        burnout_velocity=results_dict.get("burnout_velocity", 0.0),
        burnout_time=results_dict.get("burnout_time", 0.0),
        max_velocity=results_dict.get("max_velocity", 0.0),
        max_mach=results_dict.get("max_mach", 0.0),
        max_acceleration=results_dict.get("max_acceleration", 0.0),
        controller_telemetry=results_dict.get("controller_telemetry", {}),
    )
    
    return results


def list_simulations() -> List[Dict[str, Any]]:
    """
    List all saved simulations.
    
    Returns:
        List of simulation metadata dicts, sorted by timestamp (newest first)
    """
    _ensure_storage_dir()
    
    simulations = []
    
    for filepath in STORAGE_DIR.glob("*.json"):
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            simulations.append(data["metadata"])
        except (json.JSONDecodeError, KeyError) as e:
            print(f"⚠️ Error loading {filepath}: {e}")
            continue
    
    # Sort by timestamp, newest first
    simulations.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    
    return simulations


def delete_simulation(sim_id: str) -> bool:
    """
    Delete a saved simulation.
    
    Args:
        sim_id: The simulation ID to delete
        
    Returns:
        True if deleted, False if not found
    """
    filepath = STORAGE_DIR / f"{sim_id}.json"
    
    if filepath.exists():
        filepath.unlink()
        print(f"🗑️ Simulation deleted: {sim_id}")
        return True
    
    return False


def get_simulation_metadata(sim_id: str) -> Optional[Dict[str, Any]]:
    """
    Get metadata for a specific simulation without loading full results.
    
    Args:
        sim_id: The simulation ID
        
    Returns:
        Metadata dict or None if not found
    """
    filepath = STORAGE_DIR / f"{sim_id}.json"
    
    if not filepath.exists():
        return None
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    return data.get("metadata")
