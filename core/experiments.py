#!/usr/bin/env python3
"""
Experiment Engine:
Runs specific benchmarking scenarios, collects time-series data, 
and calculates rigorous academic evaluation metrics (TTC, Jerk RMS, etc.).
"""

import time
import numpy as np
from .env import SimulationConfig, DEFAULT_CFG, EgoPlant, IDM_DP_Planner

class ExperimentRunner:
    def __init__(self, cfg: SimulationConfig = DEFAULT_CFG):
        self.cfg = cfg
        # The planner acts as the "Driver Intention", generating reference trajectories
        self.planner = IDM_DP_Planner(cfg=self.cfg)

    def run_single_simulation(self, controller, scenario, ego_v0, t_max=15.0):
        """
        Executes a single run of a specific controller in a specific scenario.
        
        Args:
            controller: Instance of the tracking controller (PID, OSQP, iLQR)
            scenario: Instance of the obstacle scenario
            ego_v0: Ego vehicle's initial velocity [m/s]
            t_max: Maximum simulation time [s]
            
        Returns:
            history: dict of all time-series data
            metrics: dict of aggregated quantitative scores
        """
        
        # 1. Initialize Ego Physics
        ego = EgoPlant(cfg=self.cfg, delay_steps=self.cfg.DELAY_STEPS, v0=ego_v0)
        
        # 2. Reset scenario and controller states
        scenario.reset()
        if hasattr(controller, 'reset'):
            controller.reset()

        # 3. Time-series Data Loggers
        history = {
            't': [], 'ego_x': [], 'ego_v': [], 'ego_a': [],
            'obs_x': [], 'obs_v': [], 'obs_a': [], 'obs_active': [],
            'ref_v': [], 'ref_a': [],
            'ttc': [], 'jerk': [], 'compute_time_ms': []
        }

        # 4. Main Simulation Loop
        steps = int(t_max / self.cfg.DT)
        prev_actual_a = 0.0

        for step in range(steps):
            t = step * self.cfg.DT
            
            # --- A. Update Environment ---
            scenario.step(self.cfg.DT)
            obs_state = scenario.get_state()
            ego_state = ego.get_state()
            
            # --- B. Generate Reference (IDM) ---
            ref_xs, ref_vs, ref_as = self.planner.generate_reference(
                ego_state, obs_state, self.cfg.MPC_HORIZON, self.cfg.DT, v_desired=ego_v0
            )
            
            # --- C. Control Computation (with exact timing) ---
            t_start = time.perf_counter()
            
            a_cmd = controller.compute(
                curr_x=ego_state.x, 
                curr_v=ego_state.v, 
                ref_vs=ref_vs, 
                ref_as=ref_as, 
                delay_buffer=ego.delay_buffer.copy(), # Pass history for compensation
                obs_x=obs_state.x if obs_state.active else None,
                obs_v=obs_state.v if obs_state.active else None,
                obs_active=obs_state.active
            )
            
            t_end = time.perf_counter()
            compute_time = (t_end - t_start) * 1000.0 # Convert to milliseconds
            
            # --- D. Physical Execution ---
            actual_a = ego.step(a_cmd)
            
            # --- E. Instantaneous Metrics Calculation ---
            jerk = (actual_a - prev_actual_a) / self.cfg.DT
            prev_actual_a = actual_a
            
            # TTC Calculation (Time To Collision)
            ttc = float('inf')
            if obs_state.active and ego_state.v > obs_state.v:
                # Relative distance (bumper to bumper, minus min_dist safety buffer if desired, 
                # but physically TTC is based on absolute delta_x)
                dist = obs_state.x - ego_state.x
                if dist > 0:
                    ttc = dist / (ego_state.v - obs_state.v)
                else:
                    ttc = 0.0 # Crash!
            
            # --- F. Data Logging ---
            history['t'].append(t)
            history['ego_x'].append(ego_state.x)
            history['ego_v'].append(ego_state.v)
            history['ego_a'].append(actual_a)
            history['obs_x'].append(obs_state.x)
            history['obs_v'].append(obs_state.v)
            history['obs_a'].append(obs_state.a)
            history['obs_active'].append(obs_state.active)
            history['ref_v'].append(ref_vs[0] if len(ref_vs) > 0 else ego_state.v)
            history['ref_a'].append(ref_as[0] if len(ref_as) > 0 else 0.0)
            history['ttc'].append(ttc)
            history['jerk'].append(jerk)
            history['compute_time_ms'].append(compute_time)
            
            # Early stop if a physical crash occurs
            if obs_state.active and (obs_state.x - ego_state.x) <= 0:
                print(f"      [!] CRASH DETECTED at t={t:.2f}s! Stopping simulation.")
                # Pad remaining arrays to maintain equal length for plotting
                remaining_steps = steps - step - 1
                for _ in range(remaining_steps):
                    for key in history.keys():
                        history[key].append(history[key][-1])
                break

        # 5. Aggregate Quantitative Metrics for the Dashboard
        metrics = self._calculate_metrics(history)
        
        return history, metrics

    def _calculate_metrics(self, h: dict):
        """
        Processes the raw time-series data to extract the 5 Golden Metrics.
        """
        # Convert to numpy arrays for vectorized math
        ego_x = np.array(h['ego_x'])
        ego_v = np.array(h['ego_v'])
        obs_x = np.array(h['obs_x'])
        ref_v = np.array(h['ref_v'])
        jerk = np.array(h['jerk'])
        ttc = np.array(h['ttc'])
        active_mask = np.array(h['obs_active'], dtype=bool)
        
        # 1. Safety: Min Distance
        if np.any(active_mask):
            min_dist = np.min(obs_x[active_mask] - ego_x[active_mask])
        else:
            min_dist = 100.0 # Arbitrary large safe distance if no obstacle
            
        # 2. Safety: Danger Time (Integral of time where TTC < 3.0 seconds)
        # We only count when obstacle is active
        danger_condition = (ttc < 3.0) & active_mask
        danger_time_s = np.sum(danger_condition) * self.cfg.DT
        
        # 3. Comfort: Jerk Metrics
        max_jerk = np.max(np.abs(jerk))
        jerk_rms = np.sqrt(np.mean(jerk**2))
        
        # 4. Efficiency: Tracking RMSE
        tracking_error_rmse = np.sqrt(np.mean((ego_v - ref_v)**2))
        
        # 5. Computation: Average Solve Time
        avg_compute_time = np.mean(h['compute_time_ms'])
        
        return {
            'min_dist': min_dist,
            'danger_time_s': danger_time_s,
            'max_jerk': max_jerk,
            'jerk_rms': jerk_rms,
            'tracking_rmse': tracking_error_rmse,
            'avg_compute_ms': avg_compute_time
        }

# ============================================================
# Pre-defined Scenario Wrappers for easy invocation
# ============================================================

def run_ablation_group(controllers_dict, scenario_class, scenario_kwargs, ego_v0, t_max=15.0):
    """
    Runs a batch of controllers against the exact same scenario.
    Ensures rigorous variable control.
    """
    runner = ExperimentRunner()
    batch_results = {}
    
    print(f"\n🚀 Starting Ablation Group (Scenario: {scenario_class.__name__})")
    print(f"   Ego Initial Speed: {ego_v0:.1f} m/s. Simulating {t_max} seconds.")
    print("-" * 60)
    
    for name, ctrl in controllers_dict.items():
        print(f"   -> Testing Controller: [{name}]...")
        # Create a fresh scenario instance for every run to ensure zero contamination
        scenario = scenario_class(**scenario_kwargs)
        
        history, metrics = runner.run_single_simulation(
            controller=ctrl, 
            scenario=scenario, 
            ego_v0=ego_v0, 
            t_max=t_max
        )
        
        batch_results[name] = {
            'history': history,
            'metrics': metrics
        }
        
    return batch_results