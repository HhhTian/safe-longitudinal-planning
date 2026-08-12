#!/usr/bin/env python3
"""
Visualization Engine:
Renders the ultimate testing dashboard.
Includes 4 time-series subplots (with TTC danger zones) and 1 Radar Chart.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Polygon
import matplotlib.patches as mpatches

# ============================================================
# Radar Chart Helper Functions
# ============================================================
def _normalize_metrics(batch_metrics):
    """
    [Academic-Grade] Global Physical Anchoring Normalization.
    Avoids the misleading exaggeration of local Min-Max scaling.
    Maps each metric to the visual radius [0.1, 1.0] of the radar chart.
    """
    metrics_keys = ['min_dist', 'max_jerk', 'jerk_rms', 'tracking_rmse', 'avg_compute_ms']
    
    # Define absolute physical bounds for each metric: (Worst Performance, Best Performance)
    # These global anchors ensure fair cross-scenario comparisons.
    global_bounds = {
        'min_dist':       (0.0,  40.0),  # 0.0m is a crash (Worst), 40.0m is extremely safe (Best)
        'max_jerk':       (15.0, 0.0),   # 15.0 m/s^3 causes severe discomfort (Worst), 0.0 is perfectly smooth (Best)
        'jerk_rms':       (5.0,  0.0),   # Overall bumpiness: 5.0 is poor (Worst), 0.0 is optimal (Best)
        'tracking_rmse':  (5.0,  0.0),   # Velocity error: 5.0 m/s deviation is poor (Worst), 0.0 is perfect tracking (Best)
        'avg_compute_ms': (60.0, 0.0)    # Computation cost: 60ms is too slow (Worst), 0ms is ideal (Best)
    }
    
    ctrl_names = list(batch_metrics.keys())
    normalized_scores = {name: [] for name in ctrl_names}
    
    # Iterate over metrics first, then controllers, to ensure exact data matching
    for key in metrics_keys:
        worst, best = global_bounds[key]
        
        for name in ctrl_names:
            val = batch_metrics[name][key]
            
            # 1. Clip data to prevent exceeding defined physical/cognitive bounds
            if best > worst: 
                # "Bigger is Better" logic (e.g., min_dist)
                val_clipped = max(worst, min(val, best))
            else:            
                # "Smaller is Better" logic (e.g., jerk, compute time)
                val_clipped = max(best, min(val, worst))
                
            # 2. Linear mapping to [0.0, 1.0] scale
            if best != worst:
                ratio = (val_clipped - worst) / (best - worst)
            else:
                ratio = 1.0
                
            # 3. Map to radar chart visual range [0.1, 1.0] 
            # (0.1 prevents the polygon from completely collapsing into the invisible center)
            score = 0.1 + 0.9 * ratio
            normalized_scores[name].append(score)
            
    return normalized_scores, metrics_keys

# ============================================================
# Main Dashboard Plotter
# ============================================================
def plot_dashboard(batch_results, title="Autonomous Driving Controller Benchmark"):
    """
    Generates a 16x8 dashboard with 4 time-series plots and 1 radar chart.
    """
    # Create Figure and GridSpec
    fig = plt.figure(figsize=(18, 9))
    fig.suptitle(title, fontsize=18, fontweight='bold')
    
    # Grid: 2 rows, 3 columns. (Left 2 cols for timeseries, Right 1 col for radar)
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1, 1, 1.2], wspace=0.3, hspace=0.3)
    
    ax_dist = fig.add_subplot(gs[0, 0])
    ax_vel  = fig.add_subplot(gs[0, 1])
    ax_acc  = fig.add_subplot(gs[1, 0])
    ax_ttc  = fig.add_subplot(gs[1, 1])
    ax_radar = fig.add_subplot(gs[:, 2], polar=True) # Radar spans both rows
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    line_styles = ['-', '--', '-.', ':']
    
    # --- 1. Plot Time-Series Data ---
    for idx, (name, data) in enumerate(batch_results.items()):
        color = colors[idx % len(colors)]
        ls = line_styles[idx % len(line_styles)]
        
        h = data['history']
        m = data['metrics']
        t = np.array(h['t'])
        
        # Legend label with TTC Danger Time included!
        label = f"{name} (Danger: {m['danger_time_s']:.1f}s)"
        
        # (A) Space-Time (S-T) Trajectory Plot
        ego_x = np.array(h['ego_x'])
        ax_dist.plot(t, ego_x, label=label, color=color, linestyle=ls, linewidth=2)
        
        # plot obstacle
        if idx == 0:
            obs_x = np.array(h['obs_x'])
            active = np.array(h['obs_active'], dtype=bool)
            if np.any(active):
                t_obs = t[active]
                obs_x_active = obs_x[active]
                
                ax_dist.plot(t_obs, obs_x_active, color='black', linewidth=2.5, label='Obstacle Trajectory')
                ax_dist.fill_between(t_obs, obs_x_active - 5.0, obs_x_active, 
                                     color='red', alpha=0.2, label='Danger Zone (< 5m)')
            
        # (B) Velocity Plot
        ax_vel.plot(t, h['ref_v'], color=color, linestyle=':', linewidth=1, alpha=0.5)
        ax_vel.plot(t, h['ego_v'], label=label, color=color, linestyle=ls, linewidth=2)
        
        # (C) Acceleration Plot
        ax_acc.plot(t, h['ego_a'], label=label, color=color, linestyle=ls, linewidth=2)
        
        # (D) Inverse TTC Plot (1 / TTC)
        ttc = np.array(h['ttc'])
        inv_ttc = np.zeros_like(ttc)
        valid_ttc = ttc > 0.01 # Prevent divide by zero
        inv_ttc[valid_ttc] = 1.0 / ttc[valid_ttc]
        ax_ttc.plot(t[active], inv_ttc[active], label=label, color=color, linestyle=ls, linewidth=2)

    # Add Reference Lines & Decorate Subplots
    # (A) Distance
    ax_dist.set_title("Space-Time Trajectory (S-T)")
    ax_dist.set_ylabel("Absolute Position $s$ [m]")
    ax_dist.grid(True, alpha=0.3)
    ax_dist.legend(loc='upper left', fontsize=8)
    
    # (B) Velocity
    # Assume ref_v is same for all (just take the last one's ref_v for plotting)
    # ref_v = np.array(list(batch_results.values())[0]['history']['ref_v'])
    # ax_vel.plot(t, ref_v, color='gray', linestyle='--', linewidth=1.5, alpha=0.6, label='IDM Target $v_{ref}$')
    ax_vel.set_title("Velocity Tracking (v-t)")
    ax_vel.set_ylabel("Velocity [m/s]")
    ax_vel.grid(True, alpha=0.3)
    
    # (C) Acceleration
    ax_acc.set_title("Chassis Command (a-t)")
    ax_acc.set_xlabel("Time [s]")
    ax_acc.set_ylabel("Acceleration [m/s²]")
    ax_acc.grid(True, alpha=0.3)
    
    # (D) Inverse TTC
    ax_ttc.set_title("Danger Index ($TTC^{-1}$-t)")
    ax_ttc.set_xlabel("Time [s]")
    ax_ttc.set_ylabel("$TTC^{-1}$ [1/s]")
    ax_ttc.set_ylim(-0.05, 1.0) # Max 1.0 means TTC = 1.0s
    # **DANGER ZONE SHADOW**
    ax_ttc.axhspan(0.33, 1.0, facecolor='red', alpha=0.15, label='Danger Zone (TTC < 3s)')
    ax_ttc.grid(True, alpha=0.3)
    ax_ttc.legend(loc='upper right', fontsize=8)

    ax_dist.legend(loc='best', fontsize=8)
    
    # --- 2. Plot Radar Chart ---
    # Prepare data
    ctrl_metrics = {name: data['metrics'] for name, data in batch_results.items()}
    norm_scores, labels_keys = _normalize_metrics(ctrl_metrics)
    
    # Human-readable labels for the radar chart axes
    display_labels = ['Min Dist', 'Max Jerk', 'Jerk RMS', 'Tracking RMSE', 'Compute Time']
    num_vars = len(labels_keys)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    
    # Complete the loop for polar plotting
    angles += angles[:1]
    
    ax_radar.set_theta_offset(np.pi / 2)
    ax_radar.set_theta_direction(-1)
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(display_labels, fontsize=10, fontweight='bold')
    ax_radar.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax_radar.set_yticklabels([]) # Hide radial ticks
    
    # Plot each controller on the radar chart
    radar_legend_elements = []
    radar_legend_handles = []
    for idx, name in enumerate(batch_results.keys()):
        color = colors[idx % len(colors)]
        scores = norm_scores[name]
        scores += scores[:1] # Close the polygon
        
        ax_radar.plot(angles, scores, color=color, linewidth=2, linestyle='solid')
        ax_radar.fill(angles, scores, color=color, alpha=0.1)
        
        # Build detailed legend label with actual raw values!
        m = ctrl_metrics[name]
        raw_text = (f"{name}\n"
                    f" Dist: {m['min_dist']:.1f}m\n"
                    f" MaxJ: {m['max_jerk']:.1f}\n"
                    f" JRMS: {m['jerk_rms']:.2f}\n"
                    f" Err:  {m['tracking_rmse']:.2f}\n"
                    f" Time: {m['avg_compute_ms']:.1f}ms")
        patch = mpatches.Patch(color=color, alpha=0.5, label=raw_text)
        radar_legend_handles.append(patch)
        # radar_legend_elements.append(raw_text)

    # Add custom legend for Radar Chart
    ax_radar.legend(handles=radar_legend_handles, loc='upper right', bbox_to_anchor=(1.4, 1.1), 
                    fontsize=9, frameon=True, title="Raw Performance Data", title_fontsize=10)
    
    plt.subplots_adjust(left=0.05, right=0.82, top=0.90, bottom=0.08, wspace=0.25, hspace=0.35)
    plt.show()