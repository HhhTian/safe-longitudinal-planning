#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 导入你框架里的基础组件
from core.env import UrbanStopObstacleSim
from core.controllers import get_all_controllers
from core.experiments import run_ablation_group

def calc_cost_z(D_actual, V_actual, cost_type):
    """复刻底层的惩罚计算逻辑，用于生成曲面和计算轨迹点的 Z 轴高度"""
    MIN_DIST = 2.0
    TIME_GAP = 1.0
    
    # 侵入深度计算: g(x) = D_safe - D_actual
    D_safe = MIN_DIST + TIME_GAP * V_actual
    G = D_safe - D_actual
    
    Z = np.zeros_like(G)
    
    if cost_type == "MPC_Soft":
        mask = G > 0
        Z[mask] = 1000.0 * G[mask] + 1392.7 * (G[mask]**2)
        
    elif cost_type == "iLQR_Quad":
        mask = G > 0
        Z[mask] = 0.5 * 3144.9 * (G[mask]**2)
        
    elif cost_type == "iLQR_Exp_New":
        # 这是我们重构的神级 Exp-Lin 混合样条函数
        EXP_LIMIT = 5.0
        w_obs = 500.0
        L = 1.0
        exponent = G / L
        
        # 区域 A: 指数缓坡 (平滑预警)
        mask_A = exponent <= EXP_LIMIT
        Z[mask_A] = w_obs * np.exp(exponent[mask_A])
        
        # 区域 B: 线性陡坡 (防止海森矩阵崩溃，保持恒定自救推力)
        mask_B = exponent > EXP_LIMIT
        c_max = w_obs * np.exp(EXP_LIMIT)
        grad_max = c_max / L
        Z[mask_B] = c_max + grad_max * (G[mask_B] - EXP_LIMIT * L)
        
    return Z

def plot_trajectories_on_landscape():
    print("🚀 Running Simulation to extract Real Trajectories...")
    
    # 1. 配置并运行 Scenario B (城市平滑跟停)
    scen_cfg = {
        "class": UrbanStopObstacleSim,
        "kwargs": {'x0': 15.0, 'v0': 14.0, 't_cutin': 1.0},
        "ego_v0": 14.0,
        "t_max": 25.0
    }
    
    all_ctrls = get_all_controllers()
    
    # 我们只挑选最核心的三个算法进行 3D 对比
    test_batch = {
        "OSQP_Soft": all_ctrls["MPC_Aug_Soft"],
        "iLQR_Quad": all_ctrls["iLQR_Aug_Quad"],
        "iLQR_Exp": all_ctrls["iLQR_Aug_Exp"]
    }
    
    # 调用你的 experiments 引擎运行仿真
    results = run_ablation_group(
        controllers_dict=test_batch,
        scenario_class=scen_cfg["class"],
        scenario_kwargs=scen_cfg["kwargs"],
        ego_v0=scen_cfg["ego_v0"],
        t_max=scen_cfg["t_max"]
    )
    
    print("✅ Simulation complete! Generating 3D Topographic Trajectory Maps...")

    # 2. 准备 3D 网格地貌基底 (D, V 坐标系)
    dist_vals = np.linspace(0, 40, 100)
    vel_vals = np.linspace(0, 15, 100)
    D_grid, V_grid = np.meshgrid(dist_vals, vel_vals)
    
    fig = plt.figure(figsize=(18, 6))
    
    plot_configs = [
        (1, "OSQP_Soft", "MPC_Soft", 'Blues'),
        (2, "iLQR_Quad", "iLQR_Quad", 'Oranges'),
        (3, "iLQR_Exp", "iLQR_Exp_New", 'Greens')
    ]
    
    Z_MAX = 1000 # 统一天花板，防止图形比例失调
    
    for idx, result_key, cost_type, cmap in plot_configs:
        ax = fig.add_subplot(1, 3, idx, projection='3d')
        
        # --- A. 画底层势场地貌 ---
        Z_grid = calc_cost_z(D_grid, V_grid, cost_type)
        Z_grid_clipped = np.clip(Z_grid, 0, Z_MAX)
        ax.plot_surface(D_grid, V_grid, Z_grid_clipped, cmap=cmap, edgecolor='none', alpha=0.5)
        
        # --- B. 提取该算法的真实物理轨迹 (完美适配 experiments.py 结构) ---
        history = results[result_key]['history']
        
        ego_v = np.array(history['ego_v'])
        ego_x = np.array(history['ego_x'])
        obs_x = np.array(history['obs_x'])
        obs_active = np.array(history['obs_active'], dtype=bool)
        
        # 计算每一步的真实相对距离
        delta_x = np.zeros_like(ego_x)
        for i in range(len(ego_x)):
            if obs_active[i]:
                delta_x[i] = obs_x[i] - ego_x[i]
            else:
                delta_x[i] = 100.0  # 没有障碍物时假设在绝对安全距离
                
        # 计算这条轨迹在 Z 轴上的真实惩罚高度
        traj_z = calc_cost_z(delta_x, ego_v, cost_type)
        traj_z = np.clip(traj_z, 0, Z_MAX)
        
        # --- C. 将红线轨迹烙印在势场上 ---
        # 1. 画 3D 空中轨迹
        ax.plot3D(delta_x, ego_v, traj_z, color='red', linewidth=4, label='Ego Trajectory')
        
        # 2. 标记起点 (绿色 O) 和 终点 (黑色 X)
        ax.scatter(delta_x[0], ego_v[0], traj_z[0], color='lime', s=100, marker='o', label='Start', zorder=5)
        ax.scatter(delta_x[-1], ego_v[-1], traj_z[-1], color='black', s=100, marker='X', label='Stop', zorder=5)
        
        # 3. 投影到底面 (影子轨迹，方便看二维 X-Y 平面的走势)
        ax.plot3D(delta_x, ego_v, np.zeros_like(traj_z), color='gray', linewidth=2, linestyle='dashed', alpha=0.8)
        
        # --- D. 图表装饰与视角 ---
        ax.set_title(f"{result_key} Trajectory", fontsize=14, fontweight='bold')
        ax.set_xlabel('$\Delta x$ (Distance to Obs) [m]')
        ax.set_ylabel('$v$ (Velocity) [m/s]')
        ax.set_zlabel('Safety Cost')
        ax.set_zlim(0, Z_MAX)
        
        # 这是一个极佳的观察视角：能同时看到地面的平坦和绝壁的高耸
        ax.view_init(elev=30, azim=130) 
        ax.legend()
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_trajectories_on_landscape()