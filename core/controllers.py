#!/usr/bin/env python3
"""
Controllers: 
Instantiates the 8 rigorously controlled benchmarking candidates for the Ablation Study.
Isolates variables across three dimensions: 
A. Delay Mitigation (Tracking Only)
B. Safety Paradigm (Hard vs Soft Constraints)
C. Solver & Cost Shaping (Linear OSQP vs Nonlinear iLQR)
"""

from .env import SimulationConfig, DEFAULT_CFG
from .solvers import (
    CompensatedQPPID, 
    TrackingMPC_OSQP, 
    TrackingMPC_iLQR, 
    QuadraticSafetyCost, 
    ExponentialSafetyCost
)

def get_all_controllers(cfg: SimulationConfig = DEFAULT_CFG):
    """
    Returns a dictionary containing all 8 controller instances.
    Each controller is rigorously configured to isolate specific algorithmic variables.
    """
    
    controllers = {}

    # =========================================================================
    # Dimension A: Delay Mitigation (Tracking Only, No Safety Bounds)
    # Goal: Prove the necessity of State Augmentation for delay handling.
    # =========================================================================
    
    # 1. Naive PID: Assumes 0 delay in a system that actually has a 0.3s delay. (Baseline to fail)
    controllers["PID_Standard"] = CompensatedQPPID(
        dt=cfg.DT, 
        delay_steps=0,  # Blind to actual delay
        qp_params={'w_smooth': 10.0, 'w_track': 1.0, 'cfg': cfg},
        pid_params={'kp': 1.5, 'ki': 0.1, 'kd': 0.5, 'cfg': cfg}
    )

    # 2. Forward PID: Uses kinematic forward simulation to guess future state. (Industrial fallback)
    controllers["PID_Forward"] = CompensatedQPPID(
        dt=cfg.DT, 
        delay_steps=cfg.DELAY_STEPS, # Aware of delay, uses forward integration
        qp_params={'w_smooth': 10.0, 'w_track': 1.0, 'cfg': cfg},
        pid_params={'kp': 1.5, 'ki': 0.1, 'kd': 0.5, 'cfg': cfg}
    )

    # 3. MPC OSQP (No Aug): No augmented matrix, no safety constraints (Mode A).
    controllers["MPC_NoAug_Track"] = TrackingMPC_OSQP(
        q_v=1.0,
        # w_jerk=0.5,
        w_jerk=0.0113,
        delay_steps=0, 
        safety_mode='A', # Pure tracking, no obstacle avoidance
        cfg=cfg
    )

    # 4. MPC OSQP (Aug): *TRACKING BASELINE* - Perfect delay handling via augmented matrix.
    controllers["MPC_Aug_Track"] = TrackingMPC_OSQP(
        q_v=1.0,
        w_jerk=0.0113,
        # w_jerk=0.5,
        delay_steps=cfg.DELAY_STEPS, 
        safety_mode='A', 
        cfg=cfg
    )


    # =========================================================================
    # Dimension B: Safety Paradigm (Hard vs Soft Constraints)
    # Goal: Prove that Hard constraints crash under unexpected cut-ins.
    # Note: Both use Augmented Matrix to ensure perfect tracking foundation.
    # =========================================================================
    
    # 5. MPC OSQP (Hard): Hard constraints on obstacle distance (Prone to Infeasibility).
    controllers["MPC_Aug_Hard"] = TrackingMPC_OSQP(
        q_v=1.0,
        w_jerk=0.0582,
        # w_jerk=0.5,
        delay_steps=cfg.DELAY_STEPS, 
        safety_mode='B', 
        cfg=cfg
    )

    # 6. MPC OSQP (Soft): Industry standard. Slack variables with quadratic penalties.
    controllers["MPC_Aug_Soft"] = TrackingMPC_OSQP(
        q_v=1.0,
        w_jerk=0.0961,
        # w_jerk=0.5,
        delay_steps=cfg.DELAY_STEPS, 
        safety_mode='C', 
        w_slack_quad=1392.6779,  #10000
        w_slack_lin=1000.0, 
        cfg=cfg
    )


    # =========================================================================
    # Dimension C: Solver Alignment & Cost Shaping (Nonlinear iLQR vs Linear OSQP)
    # Goal: Prove nonlinear exponential cost yields superior comfort (lower jerk).
    # =========================================================================
    
    # 7. iLQR (Quadratic): Mathematically equivalent to MPC_Aug_Soft. 
    # Used for solver validation (curves must perfectly overlap with MPC_Aug_Soft).
    quad_cost = QuadraticSafetyCost(
        nx=2 + cfg.DELAY_STEPS, 
        nu=1, 
        q_v=1.0, 
        r_a=0.1, 
        w_jerk=0.0631,
        # w_jerk=0.5, 
        w_obs=3144.8759, #10000
        cfg=cfg
    )
    controllers["iLQR_Aug_Quad"] = TrackingMPC_iLQR(
        cost_fn=quad_cost, 
        cfg=cfg, 
        delay_steps=cfg.DELAY_STEPS, 
        max_iter=15
    )

    # 8. iLQR (Exponential): The ultimate nonlinear solution with APF-like smooth repulsion.
    exp_cost = ExponentialSafetyCost(
        nx=2 + cfg.DELAY_STEPS, 
        nu=1, 
        q_v=1.0, 
        r_a=0.1, 
        w_jerk=0.5,
        # w_jerk=0.5, 
        w_obs=800,  #2658.3872 800
        length_scale=0.5086,  #3
        cfg=cfg
    )
    controllers["iLQR_Aug_Exp"] = TrackingMPC_iLQR(
        cost_fn=exp_cost, 
        cfg=cfg, 
        delay_steps=cfg.DELAY_STEPS, 
        max_iter=15
    )

    return controllers

# Simple test block to ensure all controllers initialize properly
if __name__ == "__main__":
    ctrls = get_all_controllers()
    print(f"Successfully initialized {len(ctrls)} controllers for the benchmark:")
    for name in ctrls.keys():
        print(f" - {name}")