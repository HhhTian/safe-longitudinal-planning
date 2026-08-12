#!/usr/bin/env python3
"""
Automated Hyperparameter Tuning (Bayesian Optimization) for Advanced Controllers.
Finds the absolute BEST parameter combination for each MPC variant before the final Benchmark.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import optuna
import numpy as np
from core.env import SimulationConfig, DEFAULT_CFG, UrbanStopObstacleSim
from core.experiments import ExperimentRunner
from core.solvers import TrackingMPC_OSQP, TrackingMPC_iLQR, QuadraticSafetyCost, ExponentialSafetyCost
from core.controllers import CompensatedQPPID

# 屏蔽 Optuna 的刷屏日志，只看核心结果
optuna.logging.set_verbosity(optuna.logging.WARNING)

# 全局锚点变量（将在 main 函数启动时自动计算）
ANCHOR_RMSE = 1.0
ANCHOR_JERK = 1.0

# ==========================================
# 1. 核心裁判系统：学术级平滑代价函数 (Smooth Barrier Fitness)
# ==========================================
def evaluate_fitness(metrics):
    """
    Rigorously designed objective function for Bayesian Optimization.
    Uses Baseline Anchoring to normalize dimensions, avoiding arbitrary magic numbers.
    """
    rmse = metrics['tracking_rmse']
    jerk = metrics['jerk_rms']
    min_dist = metrics['min_dist']
    
    # 1. 严谨的相对归一化 (Relative Normalization)
    # 除以基准值，将性能转化为“相对于基准的倍数”
    norm_rmse = rmse / ANCHOR_RMSE      
    norm_jerk = jerk / ANCHOR_JERK     
    
    # 我们认为“追踪”和“平顺”在这个测试中同等重要 (各占 50% 权重)
    # 如果 base_score < 1.0，说明综合性能超越了基准！
    base_score = 0.5 * norm_rmse + 0.5 * norm_jerk
    
    # 2. 平滑二次屏障惩罚 (Smooth Quadratic Barrier for Safety)
    SAFE_MARGIN = 2.0
    if min_dist < SAFE_MARGIN:
        # 侵入越深，惩罚呈二次方指数级增长
        violation = SAFE_MARGIN - min_dist
        # 因为 base_score 的量级现在在 1.0 左右，所以惩罚系数 100 已经极其巨大了
        safety_penalty = (violation ** 2) * 100.0  
    else:
        safety_penalty = 0.0
        
    total_score = base_score + safety_penalty
    
    return total_score

# ==========================================
# 2. 跑单次仿真的闭环函数
# ==========================================
def run_trial(controller):
    runner = ExperimentRunner(cfg=DEFAULT_CFG)
    # 使用城市跟车刹停作为“标定测试道”
    scenario = UrbanStopObstacleSim(x0=15.0, v0=14.0, t_cutin=1.0)
    
    try:
        _, metrics = runner.run_single_simulation(
            controller=controller, scenario=scenario, ego_v0=14.0, t_max=18.0
        )
        return metrics # 注意：这里改为返回完整的 metrics 字典，让外部决定怎么处理
    except Exception as e:
        return None
    
def trial_wrapper(controller):
    """包裹 run_trial，处理异常并调用打分"""
    metrics = run_trial(controller)
    if metrics is None:
        return 99999.0
    return evaluate_fitness(metrics)

# ==========================================
# 3. 为每个控制器定义 AI 探索的参数空间
# ==========================================

def objective_mpc_track(trial):
    # Track_Only 没有安全权重，只调 q_v 和 w_jerk
    q_v = 1.0
    # q_v = trial.suggest_float("q_v", 0.1, 10.0, log=True)
    w_jerk = trial.suggest_float("w_jerk", 0.01, 10.0, log=True)
    
    ctrl = TrackingMPC_OSQP(delay_steps=DEFAULT_CFG.DELAY_STEPS, safety_mode='A', 
                            q_v=q_v, r_a=0.1, w_jerk=w_jerk, cfg=DEFAULT_CFG)
    return trial_wrapper(ctrl)

def objective_mpc_hard(trial):
    q_v = 1.0
    # q_v = trial.suggest_float("q_v", 0.1, 10.0, log=True)
    w_jerk = trial.suggest_float("w_jerk", 0.01, 10.0, log=True)
    # Hard bound 没有软松弛权重
    
    ctrl = TrackingMPC_OSQP(delay_steps=DEFAULT_CFG.DELAY_STEPS, safety_mode='B', 
                            q_v=q_v, r_a=0.1, w_jerk=w_jerk, cfg=DEFAULT_CFG)
    return trial_wrapper(ctrl)

def objective_mpc_soft(trial):
    q_v = 1.0
    # q_v = trial.suggest_float("q_v", 0.1, 10.0, log=True)
    w_jerk = trial.suggest_float("w_jerk", 0.01, 10.0, log=True)
    w_slack = trial.suggest_float("w_slack_quad", 1000.0, 10000.0, log=True)
    
    ctrl = TrackingMPC_OSQP(delay_steps=DEFAULT_CFG.DELAY_STEPS, safety_mode='C', 
                            q_v=q_v, r_a=0.1, w_jerk=w_jerk, w_slack_quad=w_slack, cfg=DEFAULT_CFG)
    return trial_wrapper(ctrl)

def objective_ilqr_quad(trial):
    q_v = 1.0
    # q_v = trial.suggest_float("q_v", 0.1, 10.0, log=True)
    w_jerk = trial.suggest_float("w_jerk", 0.01, 10.0, log=True)
    w_obs = trial.suggest_float("w_obs", 1000.0, 10000.0, log=True)
    
    cost = QuadraticSafetyCost(nx=2+DEFAULT_CFG.DELAY_STEPS, nu=1, 
                               q_v=q_v, r_a=0.1, w_jerk=w_jerk, w_obs=w_obs, cfg=DEFAULT_CFG)
    ctrl = TrackingMPC_iLQR(cost_fn=cost, delay_steps=DEFAULT_CFG.DELAY_STEPS, max_iter=15, cfg=DEFAULT_CFG)
    return trial_wrapper(ctrl)

def objective_ilqr_exp(trial):
    q_v = 1.0
    # q_v = trial.suggest_float("q_v", 0.1, 10.0, log=True)
    w_jerk = trial.suggest_float("w_jerk", 0.01, 10.0, log=True)
    w_obs = trial.suggest_float("w_obs", 1000.0, 10000.0, log=True)
    length_scale = trial.suggest_float("length_scale", 0.5, 5.0) # 指数函数特有参数
    
    cost = ExponentialSafetyCost(nx=2+DEFAULT_CFG.DELAY_STEPS, nu=1, 
                                 q_v=q_v, r_a=0.1, w_jerk=w_jerk, w_obs=w_obs, length_scale=length_scale, cfg=DEFAULT_CFG)
    ctrl = TrackingMPC_iLQR(cost_fn=cost, delay_steps=DEFAULT_CFG.DELAY_STEPS, max_iter=15, cfg=DEFAULT_CFG)
    return trial_wrapper(ctrl)

# ==========================================
# 4. 自动化调度主程序
# ==========================================
def main():
    global ANCHOR_RMSE, ANCHOR_JERK
    
    print("🚀 [Step 1] Computing Scientific Baselines (Anchors)...")
    # 我们用最原始的 PID_Standard 作为宇宙的基准锚点
    baseline_ctrl = CompensatedQPPID(dt=DEFAULT_CFG.DT, delay_steps=0, 
                                     pid_params={'kp':2.0, 'ki':0.1, 'kd':0.5})
    
    baseline_metrics = run_trial(baseline_ctrl)
    
    if baseline_metrics is None:
        raise ValueError("Baseline controller crashed! Cannot establish anchors.")
        
    ANCHOR_RMSE = baseline_metrics['tracking_rmse']
    ANCHOR_JERK = baseline_metrics['jerk_rms']
    
    print(f"   -> Anchor Established: RMSE = {ANCHOR_RMSE:.4f}, Jerk = {ANCHOR_JERK:.4f}")
    print("   -> All BO scores will now represent 'Relative Performance vs Baseline'.")
    print("   -> A score of < 1.0 means it has defeated the baseline!\n")
    
    print("🚀 [Step 2] Initiating Bayesian Optimization for Advanced Controllers...")
    print("This will find the absolute best parameter combinations. Please wait...\n")
    
    tasks = [
        ("MPC_Track_Only", objective_mpc_track),
        ("MPC_Hard_Bound", objective_mpc_hard),
        ("MPC_Soft_Bound", objective_mpc_soft),
        ("iLQR_Quad", objective_ilqr_quad),
        ("iLQR_Exp (Ultimate)", objective_ilqr_exp)
    ]
    
    best_params_dict = {}
    
    for name, objective in tasks:
        print(f"[{name}] AI is hunting for the best weights (30 trials)...")
        # 创建一个寻找最小分数的 Optuna Study
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=100) # 每个控制器尝试 30 种不同组合
        
        best_params_dict[name] = study.best_params
        best_score = study.best_value
        
        print(f"   -> BEST SCORE: {best_score:.4f}")
        print(f"   -> BEST PARAMS: {study.best_params}\n")

    print("="*60)
    print("🏆 OPTIMIZATION COMPLETE! Use these parameters in your main script:")
    print("="*60)
    for name, params in best_params_dict.items():
        print(f"{name}:")
        for k, v in params.items():
            print(f"  {k} = {v:.4f}")
        print("-" * 30)

if __name__ == "__main__":
    main()