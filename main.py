#!/usr/bin/env python3
"""
Ultimate Evaluation Matrix for Transactions Benchmark
Executes a 3x3 Cross-Validation: 3 Controller Batches across 3 Distinct Scenarios.
Generates 9 comprehensive dashboards (S-T, v-t, a-t, TTC^-1, and Radar).
"""

from core.env import IDM_DP_Planner, VehicleState, ObstacleState, HighwayCutInObstacleSim, UrbanStopObstacleSim, GhostCutOutObstacleSim, AdversarialStopGoObstacleSim, TeleportingBrickObstacleSim
from core.controllers import get_all_controllers
from core.experiments import run_ablation_group
from core.visualization import plot_dashboard

def main():
    # 0. Plot reference lines
    print("\n" + "="*50)
    print("FIGURE 0: IDM Planner Reference Generation")
    print("="*50)
    
    demo_planner = IDM_DP_Planner()
    demo_ego = VehicleState(x=0.0, v=25.0, a=0.0)
    demo_obs = ObstacleState(x=50.0, v=25.0, a=0.0, active=True)
    
    _ = demo_planner.generate_reference(demo_ego, demo_obs, horizon=40, dt=0.1)
    demo_planner.plot_debug()
    # 1. Fetch all rigorously configured controllers
    all_ctrls = get_all_controllers()

    # 2. Define the 3 Logical Batches (The "Who")
    controller_batches = {
        "Batch 1: Delay Immunity": {
            "PID_Standard": all_ctrls["PID_Standard"],       # Blind to delay
            "PID_Forward": all_ctrls["PID_Forward"],      # Kinematic extrapolation (Noise susceptible)
            "MPC_NoAug": all_ctrls["MPC_NoAug_Track"],    # MPC without delay buffer
            "MPC_Aug_Baseline": all_ctrls["MPC_Aug_Track"]# MPC with strictly augmented states
        },
        "Batch 2: Constraint Paradigm": {
            "MPC_Track_Only": all_ctrls["MPC_Aug_Track"], # No safety boundaries (Will crash)
            "MPC_Hard_Bound": all_ctrls["MPC_Aug_Hard"],  # Rigid KKT boundaries (Prone to infeasibility)
            "MPC_Soft_Bound": all_ctrls["MPC_Aug_Soft"]   # Slack variables (Survives but jerky)
        },
        "Batch 3: Nonlinear Elegance": {
            "OSQP_Soft": all_ctrls["MPC_Aug_Soft"],       # Industry standard baseline
            "iLQR_Quad": all_ctrls["iLQR_Aug_Quad"],      # Mathematical equivalent in DDP
            "iLQR_Exp": all_ctrls["iLQR_Aug_Exp"]         # The Ultimate Solution (Exponential repulsion)
        }
    }

    # 3. Define the 3 Extreme Physical Scenarios (The "Where")
    # Note: Physics bounds and times have been optimized to prevent t=0 crashes and allow full stops
    scenarios = {
        "Scenario A: Highway Cut-in & Flee": {
            "class": HighwayCutInObstacleSim,
            "kwargs": {'x0': 30.0, 'v0': 27.7, 't_cutin': 1.5},
            "ego_v0": 33.3,  # 120 km/h
            "t_max": 12.0
        },
        "Scenario B: Urban Stop & Go": {
            "class": UrbanStopObstacleSim,
            "kwargs": {'x0': 15.0, 'v0': 14.0, 't_cutin': 1.0},
            "ego_v0": 14.0,  # 50 km/h
            "t_max": 30.0    # Extended to show the complete standstill process
        },
        "Scenario C: Ghost Cut-out to Stationary": {
            "class": GhostCutOutObstacleSim,
            "kwargs": {'leader_v0': 22.2, 'stat_x': 180.0, 't_cutout': 1.5}, # 180m allows natural IDM deceleration
            "ego_v0": 22.2,  # 80 km/h
            "t_max": 15.0
        },
        "Scenario D: Ghost Cut-out (Panic Braking Limit)": {
            "class": GhostCutOutObstacleSim,
            "kwargs": {'leader_v0': 22.2, 'leader_x0': 40.0, 'stat_x': 75, 't_cutout': 1.5},
            "ego_v0": 22.2,  # 80 km/h
            "t_max": 20.0
        },
        "Scenario E: Adversarial Stop & Go (Aggressive Cut-in & Brake/Accel)": {
            "class": AdversarialStopGoObstacleSim,
            # ego 20m/s, obs 15m/s，cuts in 15m, then aggressively brakes, accelerates, brakes again, and flees.
            "kwargs": {'ego_v0': 20.0, 'v0': 15.0, 'gap_at_cutin': 15.0, 't_cutin': 1.0},
            "ego_v0": 20.0,
            "t_max": 12.0
        }
    }

    # ====================================================================
    # Execute the 3x3 Cross-Validation Matrix
    # ====================================================================
    total_runs = len(scenarios) * len(controller_batches)
    current_run = 1

    for scen_name, scen_cfg in scenarios.items():
        
        print(f"=== INITIATING {scen_name.upper()} ===")
        
        for batch_name, batch_dict in controller_batches.items():
            print(f"\n>>> [Progress: {current_run}/{total_runs}] Running {batch_name}...")
            
            # Run the simulation group
            batch_results = run_ablation_group(
                controllers_dict=batch_dict,
                scenario_class=scen_cfg["class"],
                scenario_kwargs=scen_cfg["kwargs"],
                ego_v0=scen_cfg["ego_v0"],
                t_max=scen_cfg["t_max"]
            )
            
            # Generate the specific title for this dashboard
            dashboard_title = f"{scen_name} | {batch_name}"
            
            # Render and show the plot
            plot_dashboard(batch_results, title=dashboard_title)
            
            current_run += 1

    print("\n✅ All Cross-Validation Matrix simulations completed successfully!")

if __name__ == "__main__":
    main()




# #!/usr/bin/env python3
# """
# Main Execution Script 
# Runs the ablation studies and plots the results.
# """

# from env import IDM_DP_Planner, VehicleState, ObstacleState, HighwayCutInObstacleSim, UrbanStopObstacleSim, GhostCutOutObstacleSim
# from controllers import get_all_controllers
# from experiments import run_ablation_group
# from visualization import plot_dashboard

# def main():
#     # 0. Plot reference lines
#     print("\n" + "="*50)
#     print("FIGURE 0: IDM Planner Reference Generation")
#     print("="*50)
    
#     demo_planner = IDM_DP_Planner()
#     demo_ego = VehicleState(x=0.0, v=25.0, a=0.0)
#     demo_obs = ObstacleState(x=50.0, v=25.0, a=0.0, active=True)
    
#     _ = demo_planner.generate_reference(demo_ego, demo_obs, horizon=40, dt=0.1)
#     demo_planner.plot_debug()
    
#     # 1. Fetch all 8 rigorously configured controllers
#     all_ctrls = get_all_controllers()

#     # ====================================================================
#     # Phase 1: Delay Immunity Test (Dimension A)
#     # Target: Highway Cut-in & Flee (High speed emphasizes delay effects)
#     # ====================================================================
#     print("\n" + "="*50)
#     print("PHASE 1: Delay Immunity Analysis")
#     print("="*50)
#     phase1_ctrls = {
#         "PID_Standard": all_ctrls["PID_Standard"],
#         "PID_Forward": all_ctrls["PID_Forward"],
#         "MPC_NoAug": all_ctrls["MPC_NoAug_Track"],
#         "MPC_Aug_Baseline": all_ctrls["MPC_Aug_Track"]
#     }
#     res_phase1 = run_ablation_group(
#         controllers_dict=phase1_ctrls,
#         scenario_class=HighwayCutInObstacleSim,
#         scenario_kwargs={'x0': 30.0, 'v0': 27.7, 't_cutin': 1.5},
#         ego_v0=33.3, # 120 km/h
#         t_max=10.0
#     )
#     plot_dashboard(res_phase1, title="Phase 1: Delay Immunity (Highway Cut-in & Flee)")


#     # ====================================================================
#     # Phase 2: Constraint Paradigm Crash Test (Dimension B)
#     # Target: Ghost Cut-out (Sudden stationary obstacle pushes state to limits)
#     # ====================================================================
#     print("\n" + "="*50)
#     print("PHASE 2: Constraint Paradigm Analysis (Hard vs Soft)")
#     print("="*50)
#     phase2_ctrls = {
#         "MPC_Track_Only": all_ctrls["MPC_Aug_Track"], # No safety
#         "MPC_Hard_Bound": all_ctrls["MPC_Aug_Hard"],  # Will likely crash solver
#         "MPC_Soft_Bound": all_ctrls["MPC_Aug_Soft"]   # Survives but jerky
#     }
#     res_phase2 = run_ablation_group(
#         controllers_dict=phase2_ctrls,
#         scenario_class=GhostCutOutObstacleSim,
#         scenario_kwargs={'leader_v0': 22.2, 'stat_x': 85.0, 't_cutout': 1.5},
#         ego_v0=22.2, # 80 km/h
#         t_max=15.0
#     )
#     plot_dashboard(res_phase2, title="Phase 2: Constraint Paradigm (Ghost Cut-out to Stationary)")


#     # ====================================================================
#     # Phase 3: Nonlinear Elegance (Dimension C)
#     # Target: Urban Stop (Tests ultimate comfort and stopping precision)
#     # ====================================================================
#     print("\n" + "="*50)
#     print("PHASE 3: Nonlinear Cost Shaping (iLQR vs OSQP)")
#     print("="*50)
#     phase3_ctrls = {
#         "OSQP_Soft": all_ctrls["MPC_Aug_Soft"],
#         "iLQR_Quad (Math Equiv)": all_ctrls["iLQR_Aug_Quad"],
#         "iLQR_Exp (Ultimate)": all_ctrls["iLQR_Aug_Exp"]
#     }
#     res_phase3 = run_ablation_group(
#         controllers_dict=phase3_ctrls,
#         scenario_class=UrbanStopObstacleSim,
#         scenario_kwargs={'x0': 15.0, 'v0': 14.0, 't_cutin': 1.0},
#         ego_v0=14.0, # 50 km/h
#         t_max=30.0
#     )
#     plot_dashboard(res_phase3, title="Phase 3: Nonlinear Elegance (Urban Stop & Go)")


#     # ====================================================================
#     # Phase 4: The Ultimate Generalization Benchmark
#     # ====================================================================
#     print("\n" + "="*50)
#     print("PHASE 4: Global Benchmark across Eras")
#     print("="*50)
#     global_ctrls = {
#         "Classical (PID_Forward)": all_ctrls["PID_Forward"],
#         "Industry (OSQP_Soft)": all_ctrls["MPC_Aug_Soft"],
#         "Next-Gen (iLQR_Exp)": all_ctrls["iLQR_Aug_Exp"]
#     }
#     # Run the toughest scenario (Ghost Cut-out) for the global benchmark
#     res_global = run_ablation_group(
#         controllers_dict=global_ctrls,
#         scenario_class=GhostCutOutObstacleSim,
#         scenario_kwargs={'leader_v0': 22.2, 'stat_x': 85.0, 't_cutout': 1.5},
#         ego_v0=22.2,
#         t_max=15.0
#     )
#     plot_dashboard(res_global, title="Global Benchmark: Classical vs Industry vs Next-Gen")


# if __name__ == "__main__":
#     main()