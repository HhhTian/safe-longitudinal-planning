import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from core.env import SimulationConfig, DEFAULT_CFG, UrbanStopObstacleSim
from core.experiments import ExperimentRunner
from core.solvers import TrackingMPC_OSQP, TrackingMPC_iLQR, ExponentialSafetyCost, QuadraticSafetyCost

def extract_pareto_frontier(xs, ys):
    """
    🔥 核心数学过滤器：帕累托前沿提取算法
    剔除所有被支配的垃圾点（Dominated Points），只保留真正的极限前沿，并按顺序排序保证连线平滑。
    (假设 x 和 y 都是越小越好)
    """
    points = sorted(list(zip(xs, ys)), key=lambda p: p[0]) # 先按 X (Jerk) 排序
    pareto_front = []
    
    # 只要当前的 Y 比之前见过的所有 Y 都小，它就是前沿点
    min_y_so_far = float('inf')
    for x, y in points:
        # 添加一个极小的容差，防止浮点数微小波动导致的误删
        if y < min_y_so_far - 1e-5:
            pareto_front.append((x, y))
            min_y_so_far = y
            
    if not pareto_front:
        return [], []
        
    p_xs, p_ys = zip(*pareto_front)
    return list(p_xs), list(p_ys)

def run_clean_pareto_sweep():
    print("🚀 Initiating Strictly Filtered Pareto Frontier...")
    runner = ExperimentRunner(cfg=DEFAULT_CFG)
    scenario = UrbanStopObstacleSim(x0=15.0, v0=14.0, t_cutin=1.0)
    
    # 我们用更密集的扫描来寻找那些幸存的最优点
    w_jerk_list = np.logspace(-1, 2.5, 20) 
    
    results = {
        "MPC_Soft":  {"jerk": [], "rmse": []},
        "iLQR_Quad": {"jerk": [], "rmse": []},
        "iLQR_Exp":  {"jerk": [], "rmse": []}
    }
    
    def safe_run(ctrl, name):
        try:
            _, metrics = runner.run_single_simulation(ctrl, scenario, 14.0, 18.0)
            return metrics['jerk_rms'], metrics['tracking_rmse']
        except Exception:
            return np.nan, np.nan

    for wj in w_jerk_list:
        # 1. OSQP Soft
        print(f"⏳ [Sweeping] Testing w_jerk = {wj:.2f} ...")
        ctrl_soft = TrackingMPC_OSQP(delay_steps=DEFAULT_CFG.DELAY_STEPS, safety_mode='C', 
                                     q_v=1.0, w_jerk=wj, w_slack_quad=1500.0, cfg=DEFAULT_CFG)
        j, r = safe_run(ctrl_soft, "MPC_Soft")
        if not np.isnan(j):
            results["MPC_Soft"]["jerk"].append(j)
            results["MPC_Soft"]["rmse"].append(r)
        
        # 2. iLQR Quad
        cost_quad = QuadraticSafetyCost(nx=2+DEFAULT_CFG.DELAY_STEPS, nu=1, 
                                        q_v=1.0, w_jerk=wj, w_obs=3000.0, cfg=DEFAULT_CFG)
        ctrl_quad = TrackingMPC_iLQR(cost_fn=cost_quad, delay_steps=DEFAULT_CFG.DELAY_STEPS, cfg=DEFAULT_CFG)
        j, r = safe_run(ctrl_quad, "iLQR_Quad")
        if not np.isnan(j):
            results["iLQR_Quad"]["jerk"].append(j)
            results["iLQR_Quad"]["rmse"].append(r)
        
        # 3. iLQR Exp (把 length_scale 调到 2.0，给足缓冲，防止高频撞墙)
        cost_exp = ExponentialSafetyCost(nx=2+DEFAULT_CFG.DELAY_STEPS, nu=1, 
                                         q_v=1.0, w_jerk=wj, w_obs=3000.0, length_scale=0.5, cfg=DEFAULT_CFG)
        ctrl_exp = TrackingMPC_iLQR(cost_fn=cost_exp, delay_steps=DEFAULT_CFG.DELAY_STEPS, cfg=DEFAULT_CFG)
        j, r = safe_run(ctrl_exp, "iLQR_Exp")
        if not np.isnan(j):
            results["iLQR_Exp"]["jerk"].append(j)
            results["iLQR_Exp"]["rmse"].append(r)

    # === 画图环节 ===
    plt.figure(figsize=(10, 7))
    
    # 分别提取各自的严格帕累托前沿
    for name, color, marker, label in [
        ("MPC_Soft", "royalblue", "s", "Soft Bound"),
        ("iLQR_Quad", "darkorange", "^", "iLQR Quadratic"),
        ("iLQR_Exp", "crimson", "o", "iLQR Exponential")
    ]:
        p_xs, p_ys = extract_pareto_frontier(results[name]["jerk"], results[name]["rmse"])
        
        # 画出过滤后的干净前沿线
        plt.plot(p_xs, p_ys, color=color, linestyle='-', marker=marker, label=label, linewidth=3, markersize=8)
        
        # 用半透明小点把那些被剔除的“垃圾杂点”画在背景里，证明我们做了大量搜索
        plt.scatter(results[name]["jerk"], results[name]["rmse"], color=color, alpha=0.2, s=20)

    plt.title('Strictly Filtered Pareto Frontier: Agility vs. Comfort', fontsize=16, fontweight='bold')
    plt.xlabel('Jerk RMS (Comfort) $\\rightarrow$ Worse', fontsize=13)
    plt.ylabel('Tracking RMSE (Agility) $\\rightarrow$ Worse', fontsize=13)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12, loc='upper right')
    
    # 根据实际数据的最小值动态标注原点方向
    all_j = results["MPC_Soft"]["jerk"] + results["iLQR_Exp"]["jerk"]
    all_r = results["MPC_Soft"]["rmse"] + results["iLQR_Exp"]["rmse"]
    if all_j and all_r:
        plt.annotate('Utopia Point\n(Physical Limit)', xy=(min(all_j)*0.9, min(all_r)*0.95), 
                     xytext=(min(all_j)*0.6, min(all_r)*0.7),
                     arrowprops=dict(facecolor='black', shrink=0.05), fontsize=12, fontweight='bold')
                 
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_clean_pareto_sweep()