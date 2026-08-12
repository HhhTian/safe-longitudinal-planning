#!/usr/bin/env python3
"""
The 3-Stage Disturbance Evolution Experiment:
Stage 1: Utopia (No Delay, No Noise)
Stage 2: The Physical Lag (0.3s Delay, No Noise)
Stage 3: The Harsh Reality (0.3s Delay + Sensor Noise)

Note: Uses in-place mutation of DEFAULT_CFG to preserve the core API.
"""

# 直接引入全局配置单例 DEFAULT_CFG
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.env import DEFAULT_CFG, UrbanStopObstacleSim, GhostCutOutObstacleSim, HighwayCutInObstacleSim
from core.controllers import get_all_controllers
from core.experiments import run_ablation_group
from core.visualization import plot_dashboard

def run_progressive_stages():
    print("\n" + "★"*70)
    print("★★★ INITIATING 3-STAGE EVOLUTION EXPERIMENT ★★★")
    print("★"*70)
    
    base_scenario = HighwayCutInObstacleSim
    scenario_kwargs = {
        'x0': 30.0, 
        'v0': 27.7,  
        't_cutin': 1.5
    }
    ego_v0 = 33.3
    t_max = 12.0
    
    # base_scenario = GhostCutOutObstacleSim
    # scenario_kwargs = {
    #     'leader_v0': 22.2, 
    #     'leader_x0': 40.0,  
    #     'stat_x': 75,     # 1.5s 暴雷时恰好相距 40m
    #     't_cutout': 1.5
    # }
    # ego_v0 = 22.2
    # t_max = 8.0
    
    # base_scenario = UrbanStopObstacleSim
    # scenario_kwargs = {
    #     'x0': 15.0, 
    #     'v0': 14.0,  
    #     't_cutin': 1.0
    # }
    # ego_v0 = 14.0
    # t_max = 30.0

    # ==========================================
    # Stage 1: Utopia (无延迟，无噪声)
    # ==========================================
    print("\n>>> [Stage 1/3] UTOPIA: No Delay, No Noise...")
    # 🔥 核心操作：直接覆写全局配置对象 DEFAULT_CFG
    DEFAULT_CFG.DT = 0.05
    DEFAULT_CFG.DELAY_STEPS = 0         # 无延迟
    DEFAULT_CFG.MPC_HORIZON = 80
    DEFAULT_CFG.POS_NOISE_BASE = 0.0    # 完美传感器
    DEFAULT_CFG.POS_NOISE_DIST_FACTOR = 0.0
    DEFAULT_CFG.VEL_NOISE_STD = 0.0
    
    # 获取控制器（内部会自动读取已修改的 DEFAULT_CFG）
    ctrls_stage1 = get_all_controllers()
    batch_stage1 = {
        "PID_Standard": ctrls_stage1["PID_Standard"],
        "PID_Forward": ctrls_stage1["PID_Forward"],
        "MPC_NoAug": ctrls_stage1["MPC_NoAug_Track"],
        "MPC_Aug_Baseline": ctrls_stage1["MPC_Aug_Track"]
    }
    
    # 直接调用原汁原味的 run_ablation_group，不传 cfg
    res_stage1 = run_ablation_group(batch_stage1, base_scenario, scenario_kwargs, ego_v0, t_max)
    plot_dashboard(res_stage1, title="STAGE 1: Utopia (Delay=0s, Noise=OFF)")

    # ==========================================
    # Stage 2: The Physical Lag (有延迟 0.3s，无噪声)
    # ==========================================
    print("\n>>> [Stage 2/3] PURE DELAY: 0.3s Delay, No Noise...")
    # 🔥 覆写全局配置：加上物理延迟，依然保持无噪
    DEFAULT_CFG.DELAY_STEPS = 6         # 恢复 0.3s 物理迟滞 (6 * 0.05)
    
    # 重新获取控制器（加载新的延迟设定）
    ctrls_stage2 = get_all_controllers()
    batch_stage2 = {
        "PID_Standard": ctrls_stage2["PID_Standard"],
        "PID_Forward": ctrls_stage2["PID_Forward"],
        "MPC_NoAug": ctrls_stage2["MPC_NoAug_Track"],
        "MPC_Aug_Baseline": ctrls_stage2["MPC_Aug_Track"]
    }
    
    res_stage2 = run_ablation_group(batch_stage2, base_scenario, scenario_kwargs, ego_v0, t_max)
    plot_dashboard(res_stage2, title="STAGE 2: Pure Delay (Delay=0.3s, Noise=OFF)")

    # ==========================================
    # Stage 3: The Harsh Reality (有延迟 0.3s，有传感器噪声)
    # ==========================================
    print("\n>>> [Stage 3/3] HARSH REALITY: 0.3s Delay + Sensor Noise...")
    # 🔥 覆写全局配置：开启残酷现实的雷达噪声
    DEFAULT_CFG.POS_NOISE_BASE = 0.5    
    DEFAULT_CFG.POS_NOISE_DIST_FACTOR = 0.02
    DEFAULT_CFG.VEL_NOISE_STD = 0.5
    
    # 重新获取控制器
    ctrls_stage3 = get_all_controllers()
    batch_stage3 = {
        "PID_Standard": ctrls_stage3["PID_Standard"],
        "PID_Forward": ctrls_stage3["PID_Forward"],
        "MPC_NoAug": ctrls_stage3["MPC_NoAug_Track"],
        "MPC_Aug_Baseline": ctrls_stage3["MPC_Aug_Track"]
    }
    
    res_stage3 = run_ablation_group(batch_stage3, base_scenario, scenario_kwargs, ego_v0, t_max)
    plot_dashboard(res_stage3, title="STAGE 3: Harsh Reality (Delay=0.3s, Noise=ON)")

if __name__ == "__main__":
    run_progressive_stages()