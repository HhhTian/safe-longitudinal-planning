import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_cost_landscapes():
    print("🚀 Generating 3D Cost Landscapes...")
    
    # === 1. 定义物理空间 (网格扫描) ===
    # 距离障碍物: 0m 到 40m
    dist_vals = np.linspace(0, 40, 100)
    # 自车速度: 0 m/s 到 20 m/s (约 72 km/h)
    vel_vals = np.linspace(0, 20, 100)
    
    D, V = np.meshgrid(dist_vals, vel_vals)
    
    # 基础物理参数
    MIN_DIST = 2.0
    TIME_GAP = 1.0
    
    # 侵入深度计算: g(x) = D_safe - D_actual
    # 如果 g(x) > 0，说明越界了
    D_safe = MIN_DIST + TIME_GAP * V
    G = D_safe - D
    
    # === 2. 初始化四种代价函数矩阵 ===
    Cost_Soft = np.zeros_like(G)
    Cost_Quad = np.zeros_like(G)
    Cost_Exp_Old = np.zeros_like(G)
    Cost_Exp_New = np.zeros_like(G)
    
    # === 3. 填充算法逻辑 (完全复刻你的代码) ===
    
    # 3.1 OSQP Soft Bound (线性 + 二次)
    w_slack_lin = 1000.0
    w_slack_quad = 1392.7
    mask_soft = G > 0
    Cost_Soft[mask_soft] = w_slack_lin * G[mask_soft] + w_slack_quad * (G[mask_soft]**2)
    
    # 3.2 iLQR Quad (纯二次型)
    w_obs_quad = 3144.9
    mask_quad = G > 0
    # 注意公式里有 0.5
    Cost_Quad[mask_quad] = 0.5 * w_obs_quad * (G[mask_quad]**2)
    
    # 3.3 iLQR Exp (老版本：带有导致木讷的 np.clip)
    w_obs_exp_old = 800.0
    length_scale_old = 0.5086
    exponent_old = G / length_scale_old
    exponent_old_clipped = np.clip(exponent_old, -20.0, 10.0) # 致命的截断
    Cost_Exp_Old = w_obs_exp_old * np.exp(exponent_old_clipped)
    
    # 3.4 iLQR Exp-Lin (我们刚刚推演的新版本：平滑样条)
    w_obs_exp_new = 500.0
    length_scale_new = 1.0
    EXP_LIMIT = 5.0
    exponent_new = G / length_scale_new
    
    # 区域 A: 指数预警区 (exponent <= 5.0)
    mask_A = exponent_new <= EXP_LIMIT
    Cost_Exp_New[mask_A] = w_obs_exp_new * np.exp(exponent_new[mask_A])
    
    # 区域 B: 线性保命区 (exponent > 5.0)
    mask_B = exponent_new > EXP_LIMIT
    c_max = w_obs_exp_new * np.exp(EXP_LIMIT)
    grad_max = c_max / length_scale_new
    Cost_Exp_New[mask_B] = c_max + grad_max * (G[mask_B] - EXP_LIMIT * length_scale_new)
    
    # === 4. 开始画图 ===
    fig = plt.figure(figsize=(16, 10))
    
    # 为了防止老版本的 1700万 把整个图的比例撑爆，我们把 Z 轴最高点限制在 50 万
    Z_MAX = 500000 
    
    plots = [
        (1, "MPC_Aug_Soft (Quadratic Slack)", Cost_Soft, 'Blues'),
        (2, "iLQR_Aug_Quad (Pure Quadratic)", Cost_Quad, 'Oranges'),
        (3, "iLQR_Exp_Old (with Clip - The Flatline)", Cost_Exp_Old, 'Reds'),
        (4, "iLQR_Exp_New (Spline - The Ultimate)", Cost_Exp_New, 'Greens')
    ]
    
    for idx, title, Z, cmap in plots:
        ax = fig.add_subplot(2, 2, idx, projection='3d')
        
        # 截断 Z 轴以便于观察地貌细节
        Z_plot = np.clip(Z, 0, Z_MAX)
        
        surf = ax.plot_surface(D, V, Z_plot, cmap=cmap, edgecolor='none', alpha=0.8)
        
        # 画一条 Z=0 的地平线（安全与危险的边界）
        ax.contour(D, V, Z_plot, levels=[1.0], colors='black', linewidths=2, linestyles='dashed')
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Distance to Obstacle $\Delta x$ (m)')
        ax.set_ylabel('Ego Velocity $v$ (m/s)')
        ax.set_zlabel('Safety Cost')
        ax.set_zlim(0, Z_MAX)
        ax.view_init(elev=25, azim=135) # 调整绝佳的观察视角
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_cost_landscapes()