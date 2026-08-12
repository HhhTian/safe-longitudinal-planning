#!/usr/bin/env python3
"""
Part 1: Core components - data structures, dynamics, reference generators, cost functions

Features: 
- Trapezoidal Integration for physics accuracy (Forward Euler Method would accumulate Error)
- IDM (Intelligent Driver Model) for scientific reference generation, can be replaced by DP, S-T Gram, DeepLearning/Reinforcement Learning approches
- Triangular acceleration profile for obstacle
"""

import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
import os

# ============================================================
# 1. Simulation Parameters
# ============================================================
class SimulationConfig:
    DT = 0.05
    MPC_HORIZON = 80  

    # System Delay Parameters
    DELAY = 0.3
    DELAY_STEPS = int(DELAY / DT)

    # Physics Limits
    A_MIN = -9.0
    A_MAX = 4.0
    V_MAX = 35.0  # speed limit
    
    # Comfort Limits
    A_COMF_MIN: float = -4.0
    A_COMF_MAX: float = 2.0
    
    # Safety Parameters
    TIME_GAP: float = 1.8      # safe time gap
    MIN_DIST: float = 5.0      # m
    
    # --- Perception Noise Parameters (Sensor Model) ---
    POS_NOISE_BASE = 0.5   # Base noise (meters)
    POS_NOISE_DIST_FACTOR = 0.02 # Distance-dependent noise factor (2%)
    VEL_NOISE_STD = 0.5    # Doppler error (m/s)

DEFAULT_CFG = SimulationConfig()
# Linear ramp duration at every phase boundary to enforce finite jerk.
# 0.3s yields jerk ≤ 12 m/s³ for typical accelerations, consistent
# with measured human driving behavior (5–15 m/s³).
RAMP_DURATION = 0.3  # s

@dataclass
class VehicleState:
    x: float
    v: float
    a: float

@dataclass
class ObstacleState:
    x: float
    v: float
    a: float
    active: bool

# ============================================================
# 2. Obstacle Dynamics (Piecewise-Ramp Acceleration Scenarios)
# ============================================================
"""
Three scenarios covering distinct threat profiles:
  1. HighwayCutInObstacleSim  — High-speed cut-in, brake, then flee
  2. UrbanStopObstacleSim     — Urban cut-in, brake to full stop, then restart
  3. GhostCutOutObstacleSim   — Lead vehicle cut-out exposing a stationary obstacle
"""
def _ramp_transition(t, t_boundary, a_before, a_after, ramp_dur=RAMP_DURATION):
    """
    Linear interpolation between two acceleration values over a short ramp.
    
    Used at every phase boundary to replace instantaneous acceleration jumps
    (infinite jerk) with finite-jerk transitions.
    
    Args:
        t:           current time [s]
        t_boundary:  nominal phase-switch time [s]
        a_before:    acceleration value before the boundary [m/s²]
        a_after:     acceleration value after the boundary [m/s²]
        ramp_dur:    transition duration [s], centered on t_boundary
        
    Returns:
        Interpolated acceleration [m/s²], or None if t is outside the ramp window.
    """
    half = ramp_dur / 2.0
    if t_boundary - half <= t <= t_boundary + half:
        alpha = (t - (t_boundary - half)) / ramp_dur  # 0→1
        return a_before + alpha * (a_after - a_before)
    return None


def _clamp_velocity(v_next, v_current, a, dt, v_min=0.0, v_max=np.inf):
    """
    Clamp velocity to physical bounds and correct acceleration accordingly.
    
    When velocity hits a limit (e.g., vehicle cannot reverse, or reaches top speed),
    the reported acceleration must be adjusted to match the actual velocity change.
    Without this correction, get_state() would return a=-3.5 while v=0, which is
    physically contradictory and confuses downstream MPC/IDM predictions.
    
    Args:
        v_next:    unclamped next velocity [m/s]
        v_current: current velocity [m/s]
        a:         commanded acceleration [m/s²]
        dt:        time step [s]
        v_min:     minimum velocity (default 0, no reversing)
        v_max:     maximum velocity (vehicle top speed)
        
    Returns:
        (v_clamped, a_corrected): physically consistent velocity-acceleration pair
    """
    if v_next < v_min:
        v_clamped = v_min
        a_corrected = (v_clamped - v_current) / dt if dt > 0 else 0.0
    elif v_next > v_max:
        v_clamped = v_max
        a_corrected = (v_clamped - v_current) / dt if dt > 0 else 0.0
    else:
        v_clamped = v_next
        a_corrected = a
    return v_clamped, a_corrected


# ============================================================
# Scenario 1: Highway Cut-in, Brake, then Flee
# ============================================================
class HighwayCutInObstacleSim:
    """
    A vehicle on the adjacent lane (traveling at ~100 km/h) cuts into ego's lane
    (ego at ~120 km/h), brakes briefly, then accelerates past ego and departs.
    
    Timeline (t_active = time since becoming visible):
        Phase 1  [0.0, 1.0)s    Coast at initial speed after completing lane change.
        Phase 2  [1.0, 2.5)s    Brake at -3.0 m/s² (driver sees slower traffic ahead).
        Phase 3  [2.5, 3.0)s    Hold speed (foot off brake, assessing situation).
        Phase 4  [3.0, 7.0)s    Accelerate at +2.95 m/s² (decides to overtake / flee).
        Phase 5  [7.0, ∞)s      Cruise at ~126 km/h, pulling away from ego.
        
    All phase boundaries have 0.3s linear ramps for finite jerk.
        
    Test objective: 
        Controller must first decelerate to avoid rear-ending the braking obstacle,
        then smoothly resume cruise as the obstacle accelerates away.
    """
    def __init__(self, x0=30.0, v0=27.7, v_max=40.0, t_cutin=1.5):
        """
        Args:
            x0:      initial position ahead of ego [m]
            v0:      initial longitudinal speed [m/s] (27.7 ≈ 100 km/h)
            v_max:   vehicle top speed [m/s] (40.0 ≈ 144 km/h)
            t_cutin: time at which the obstacle becomes visible to ego [s]
        """
        self.init_x = x0
        self.init_v = v0
        self.v_max = v_max
        self.t_cutin = t_cutin
        
        # Phase timing (relative to activation)
        self.t1 = 1.0   # end of coast
        self.t2 = 2.5   # end of brake
        self.t3 = 3.0   # end of hold
        self.t4 = 7.0   # end of acceleration
        
        # Phase accelerations
        self.a_brake = -3.0
        self.a_hold  = 0.0
        self.a_accel = 2.95
        
        self.reset()

    def reset(self):
        self.x = self.init_x
        self.v = self.init_v
        self.a = 0.0
        self.active = False
        self.t_active = 0.0
        self.global_t = 0.0

    def _get_accel_at_time(self, t):
        """
        Piecewise-constant acceleration with linear ramps at transitions.
        
        Instead of instantaneous jumps (jerk = ∞), each phase boundary uses a 
        0.3s linear ramp. For example, at t=1.0s the acceleration transitions
        from 0.0 to -3.0 m/s² over 0.15s on each side, yielding 
        jerk = 3.0/0.3 = 10 m/s³ (within human driving range of 5–15 m/s³).
        """
        
        # Ramp: coast → brake
        r = _ramp_transition(t, self.t1, 0.0, self.a_brake)
        if r is not None: return r
        
        # Ramp: brake → hold
        r = _ramp_transition(t, self.t2, self.a_brake, self.a_hold)
        if r is not None: return r
        
        # Ramp: hold → accel
        r = _ramp_transition(t, self.t3, self.a_hold, self.a_accel)
        if r is not None: return r
        
        # Ramp: accel → cruise
        r = _ramp_transition(t, self.t4, self.a_accel, 0.0)
        if r is not None: return r
        
        # Flat regions (outside any ramp)
        if t < self.t1:    return 0.0
        elif t < self.t2:  return self.a_brake
        elif t < self.t3:  return self.a_hold
        elif t < self.t4:  return self.a_accel
        else:              return 0.0

    def step(self, dt):
        self.global_t += dt
        
        if self.global_t < self.t_cutin:
            # Pre-activation: obstacle is on adjacent lane, cruising at v0.
            # Position updates so that when it "appears", x is physically correct.
            self.x += self.v * dt
            return
        
        # Activation latch: once visible, stays visible.
        if not self.active:
            self.active = True
            
        self.t_active += dt
        a_cmd = self._get_accel_at_time(self.t_active)

        # Trapezoidal integration
        v_next = self.v + a_cmd * dt
        v_next, a_actual = _clamp_velocity(v_next, self.v, a_cmd, dt, 
                                           v_min=0.0, v_max=self.v_max)
        self.a = a_actual
        
        self.x += 0.5 * (self.v + v_next) * dt
        self.v = v_next

    def get_state(self):
        return ObstacleState(x=self.x, v=self.v, a=self.a, active=self.active)

    def predict_future(self, steps, dt):
        """
        Roll out future obstacle trajectory for MPC horizon.
        
        Returns:
            xs: predicted positions [m], shape (steps,)
            vs: predicted velocities [m/s], shape (steps,)
        """
        xs, vs = np.zeros(steps), np.zeros(steps)
        curr_x, curr_v = self.x, self.v
        
        for i in range(steps):
            t = self.t_active + (i + 1) * dt
            a = self._get_accel_at_time(t)
            v_next = curr_v + a * dt
            v_next, _ = _clamp_velocity(v_next, curr_v, a, dt, 
                                        v_min=0.0, v_max=self.v_max)
            xs[i] = curr_x + 0.5 * (curr_v + v_next) * dt
            vs[i] = v_next
            curr_x, curr_v = xs[i], v_next
            
        return xs, vs


# ============================================================
# Scenario 2: Urban Cut-in, Brake to Stop, then Restart
# ============================================================
class UrbanStopObstacleSim:
    """
    On an urban road, a vehicle cuts into ego's lane at ~43 km/h (ego at ~50 km/h),
    then brakes to a full stop (e.g., red light or traffic jam), waits, and restarts.
    
    Timeline (t_active = time since becoming visible):
        Phase 1  [0.0, 1.0)s     Coast at initial speed after lane change.
        Phase 2  [1.0, 5.0)s     Brake at -3.5 m/s² (approaching red light).
        Phase 3  [5.0, 7.0)s     Stopped, waiting for green / traffic to clear.
        Phase 4  [7.0, 12.0)s    Accelerate at +2.0 m/s² (light turns green).
        Phase 5  [12.0, ∞)s      Cruise at new speed.
        
    All phase boundaries have 0.3s linear ramps for finite jerk.
        
    Test objective:
        Controller must decelerate to a stop behind the obstacle, maintain safe
        standstill distance, then smoothly accelerate when the obstacle restarts.
        Tests the full speed range from cruise → stop → restart.
    """

    def __init__(self, x0=15.0, v0=12.0, v_max=20.0, t_cutin=1.5):
        """
        Args:
            x0:      initial position ahead of ego [m]
            v0:      initial speed [m/s] (12.0 ≈ 43 km/h, slower than ego's ~50 km/h)
            v_max:   speed cap for urban road [m/s] (20.0 ≈ 72 km/h)
            t_cutin: time at which the obstacle becomes visible to ego [s]
        """
        self.init_x = x0
        self.init_v = v0
        self.v_max = v_max
        self.t_cutin = t_cutin
        
        # Phase timing
        self.t1 = 1.0    # end of coast
        self.t2 = 5.0    # end of brake
        self.t3 = 15.0    # end of standstill
        self.t4 = 20.0   # end of acceleration
        
        # Phase accelerations
        self.a_brake = -3.5
        self.a_accel = 2.0
        
        self.reset()

    def reset(self):
        self.x = self.init_x
        self.v = self.init_v
        self.a = 0.0
        self.active = False
        self.t_active = 0.0
        self.global_t = 0.0

    def _get_accel_at_time(self, t):
        """Piecewise acceleration with linear ramps at all phase boundaries."""      
        # Ramp: coast → brake
        r = _ramp_transition(t, self.t1, 0.0, self.a_brake)
        if r is not None: return r
        
        # Ramp: brake → stop
        r = _ramp_transition(t, self.t2, self.a_brake, 0.0)
        if r is not None: return r
        
        # Ramp: stop → accel
        r = _ramp_transition(t, self.t3, 0.0, self.a_accel)
        if r is not None: return r
        
        # Ramp: accel → cruise
        r = _ramp_transition(t, self.t4, self.a_accel, 0.0)
        if r is not None: return r
        
        # Flat regions
        if t < self.t1:    return 0.0
        elif t < self.t2:  return self.a_brake
        elif t < self.t3:  return 0.0
        elif t < self.t4:  return self.a_accel
        else:              return 0.0

    def step(self, dt):
        self.global_t += dt
        
        if self.global_t < self.t_cutin:
            self.x += self.v * dt
            return
        
        if not self.active:
            self.active = True
            
        self.t_active += dt
        a_cmd = self._get_accel_at_time(self.t_active)

        v_next = self.v + a_cmd * dt
        v_next, a_actual = _clamp_velocity(v_next, self.v, a_cmd, dt,
                                           v_min=0.0, v_max=self.v_max)
        self.a = a_actual
        
        self.x += 0.5 * (self.v + v_next) * dt
        self.v = v_next

    def get_state(self):
        return ObstacleState(x=self.x, v=self.v, a=self.a, active=self.active)

    def predict_future(self, steps, dt):
        """Roll out future trajectory for MPC horizon."""
        xs, vs = np.zeros(steps), np.zeros(steps)
        curr_x, curr_v = self.x, self.v
        
        for i in range(steps):
            t = self.t_active + (i + 1) * dt
            a = self._get_accel_at_time(t)
            v_next = curr_v + a * dt
            v_next, _ = _clamp_velocity(v_next, curr_v, a, dt, 
                                        v_min=0.0, v_max=self.v_max)
            xs[i] = curr_x + 0.5 * (curr_v + v_next) * dt
            vs[i] = v_next
            curr_x, curr_v = xs[i], v_next
            
        return xs, vs


# ============================================================
# Scenario 3: Ghost Cut-out Exposing Stationary Obstacle
# ============================================================
class GhostCutOutObstacleSim:
    """
    Classic "ghost cut-out" scenario:
    A lead vehicle traveling at the same speed as ego suddenly swerves out,
    revealing a stationary disabled vehicle directly ahead.
    
    From ego's perspective, the perceived obstacle jumps:
      - Before cut-out: a moving lead vehicle at ~35m, traveling at 80 km/h
      - After cut-out:  a stationary vehicle at ~85m, velocity = 0
    
    This scenario uniquely models a SENSOR TARGET SWITCH rather than a single
    vehicle's dynamics. The `active` flag represents whether ego's forward path 
    is obstructed at all (before the lead vehicle exists, nothing is blocking ego).
    
    Timeline (referenced to global_t):
        Phase 1  [0, t_cutout)s    Ego follows the lead vehicle (steady-state car following).
                                   Sensor tracks lead vehicle: reports its x and v.
        Phase 2  [t_cutout, ∞)s    Lead vehicle swerves out. Sensor locks onto the
                                   stationary obstacle: reports (stat_x, 0).
                                   
    Test objective:
        Controller must execute emergency braking within ~2.3s TTC and ~51m to 
        stop before hitting the stationary obstacle. Tests AEB-level response
        and the controller's ability to handle sudden target switches.
    """
    def __init__(self, leader_v0=22.2, leader_x0=50.0, stat_x=85.0, t_cutout=1.5):
        """
        Args:
            leader_v0:  initial speed of the lead vehicle [m/s] (22.2 ≈ 80 km/h)
            leader_x0:  initial gap between ego and lead vehicle [m]
            stat_x:     absolute position of the stationary obstacle [m]
            t_cutout:   time at which the lead vehicle swerves out [s]
        """
        self.leader_v0 = leader_v0
        self.leader_x0 = leader_x0
        self.stat_x = stat_x
        self.t_cutout = t_cutout
        self.reset()

    def reset(self):
        self.global_t = 0.0
        self.active = False
        
        # Lead vehicle state (always updates, even before activation)
        self.leader_x = self.leader_x0
        self.leader_v = self.leader_v0
        
        # Perceived obstacle (what ego's sensor reports)
        self.current_obs_x = self.leader_x0
        self.current_obs_v = self.leader_v0
        self.current_obs_a = 0.0
        
        # Internal flag: has the lead vehicle departed?
        self._leader_departed = False

    def step(self, dt):
        self.global_t += dt
        
        # [FIX] Lead vehicle always moves, regardless of `active` flag.
        #       The lead vehicle exists in the world before ego can see anything.
        self.leader_x += self.leader_v * dt
        
        # [FIX] Unified activation: obstacle becomes visible at t=0
        #       (ego starts following the lead vehicle immediately).
        if not self.active:
            self.active = True
        
        if self.global_t < self.t_cutout:
            # Stage 1: Lead vehicle is ahead, sensor tracks it.
            self.current_obs_x = self.leader_x
            self.current_obs_v = self.leader_v
            self.current_obs_a = 0.0
        else:
            # Stage 2: Lead vehicle has swerved out.
            # Sensor now reports the stationary obstacle.
            if not self._leader_departed:
                self._leader_departed = True
            self.current_obs_x = self.stat_x
            self.current_obs_v = 0.0
            self.current_obs_a = 0.0

    def get_state(self):
        return ObstacleState(
            x=self.current_obs_x, 
            v=self.current_obs_v, 
            a=self.current_obs_a, 
            active=self.active
        )

    def predict_future(self, steps, dt):
        """
        Sensors cannot predict when will front vehicle cut-out, only focus on current lead vehicle.
        """
        xs = np.zeros(steps)
        vs = np.zeros(steps)
        
        curr_x = self.current_obs_x
        curr_v = self.current_obs_v
        
        for i in range(steps):
            curr_x += curr_v * dt
            
            xs[i] = curr_x
            vs[i] = curr_v
            
        return xs, vs
    
# ============================================================
# Scenario 4: Adversarial Stop-and-Go (Aggressive Cut-in & Brake/Accel)
# ============================================================
class AdversarialStopGoObstacleSim:
    """
    Adversarial Highly-Dynamic Scenario:
    An obstacle cuts in closely, then aggressively brakes, accelerates, brakes again, and flees.
    
    Objectives:
    1. Break constant-velocity/acceleration prediction models to test robustness against model mismatch.
    2. Validate the jerk attenuation (spring-like damping) of soft-constrained MPC (iLQR_Exp).
    
    Timeline (t_active):
        [0.0, 0.5)s:  Coast after cut-in
        [0.5, 2.0)s:  1st aggressive brake (-5.0 m/s²)
        [2.0, 3.5)s:  Sudden hard acceleration (+3.5 m/s²)
        [3.5, 5.0)s:  2nd aggressive brake (-4.5 m/s²)
        [5.0, ∞)s:    Accelerate to cruise (+2.0 m/s²)
    """
    def __init__(self, ego_v0=20.0, v0=15.0, gap_at_cutin=15.0, v_max=30.0, t_cutin=1.0):
        # Back-calculate the obstacle's initial absolute position to guarantee the exact gap at t_cutin
        ego_x_at_cutin = ego_v0 * t_cutin
        obs_x_at_cutin = ego_x_at_cutin + gap_at_cutin
        self.init_x = obs_x_at_cutin - v0 * t_cutin
        
        self.init_v = v0
        self.v_max = v_max
        self.t_cutin = t_cutin
        
        # Phase boundaries [s]
        self.t1 = 0.5   # end coast
        self.t2 = 2.0   # end brake 1
        self.t3 = 3.5   # end accel 1
        self.t4 = 5.0   # end brake 2
        self.t5 = 8.0   # end accel to cruise
        
        # Extreme acceleration values [m/s²]
        self.a_brake1 = -5.0
        self.a_accel1 =  3.5
        self.a_brake2 = -4.5
        self.a_accel2 =  2.0
        
        self.reset()

    def reset(self):
        self.x = self.init_x
        self.v = self.init_v
        self.a = 0.0
        self.active = False
        self.t_active = 0.0
        self.global_t = 0.0

    def _get_accel_at_time(self, t):
        # Apply linear ramps to avoid infinite jerk at phase transitions
        r = _ramp_transition(t, self.t1, 0.0, self.a_brake1)
        if r is not None: return r
        
        r = _ramp_transition(t, self.t2, self.a_brake1, self.a_accel1)
        if r is not None: return r
        
        r = _ramp_transition(t, self.t3, self.a_accel1, self.a_brake2)
        if r is not None: return r
        
        r = _ramp_transition(t, self.t4, self.a_brake2, self.a_accel2)
        if r is not None: return r
        
        r = _ramp_transition(t, self.t5, self.a_accel2, 0.0)
        if r is not None: return r
        
        # Constant acceleration zones
        if t < self.t1:   return 0.0
        elif t < self.t2: return self.a_brake1
        elif t < self.t3: return self.a_accel1
        elif t < self.t4: return self.a_brake2
        elif t < self.t5: return self.a_accel2
        else:             return 0.0

    def step(self, dt):
        self.global_t += dt
        
        # Pre-activation: Move blindly before cut-in
        if self.global_t < self.t_cutin:
            self.x += self.v * dt
            return
        
        # Trigger activation
        if not self.active: self.active = True
        self.t_active += dt
        a_cmd = self._get_accel_at_time(self.t_active)

        # Kinematic update with velocity clamping
        v_next = self.v + a_cmd * dt
        v_next, a_actual = _clamp_velocity(v_next, self.v, a_cmd, dt, v_min=0.0, v_max=self.v_max)
        self.a = a_actual
        self.x += 0.5 * (self.v + v_next) * dt
        self.v = v_next

    def get_state(self):
        return ObstacleState(x=self.x, v=self.v, a=self.a, active=self.active)

    def predict_future(self, steps, dt):
        # Forward rollout for MPC prediction horizon
        xs, vs = np.zeros(steps), np.zeros(steps)
        curr_x, curr_v = self.x, self.v
        for i in range(steps):
            t = self.t_active + (i + 1) * dt
            a = self._get_accel_at_time(t)
            v_next = curr_v + a * dt
            v_next, _ = _clamp_velocity(v_next, curr_v, a, dt, v_min=0.0, v_max=self.v_max)
            xs[i] = curr_x + 0.5 * (curr_v + v_next) * dt
            vs[i] = v_next
            curr_x, curr_v = xs[i], v_next
        return xs, vs


# ============================================================
# Scenario 5: The Teleporting Brick (Failure Mode Testing)
# ============================================================
class TeleportingBrickObstacleSim:
    """
    The "Teleporting Brick" Scenario:
    Designed specifically to break MPC_Track_Only (blind) and MPC_Hard (rigid).
    
    Setup: Ego cruises at high speed (30 m/s). 
           At t_cutin, a very slow vehicle (5 m/s) suddenly "teleports" (cuts in) 
           just 12 meters ahead.
             
    Expected Outcomes:
    1. MPC_Track_Only: Ignores the obstacle and crashes at 30 m/s.
    2. MPC_Hard: OSQP solver instantly returns `Primal Infeasible` due to 
                 initial state violation, causing system collapse.
    3. iLQR_Exp (Soft CBF): Survives the numerical explosion, issues maximum 
                 braking, avoids crash or mitigates it gracefully without freezing.
    """
    def __init__(self, ego_v0=30.0, v0=5.0, gap_at_cutin=12.0, v_max=20.0, t_cutin=1.5):
        # Back-calculate position to enforce an exact 12m gap upon sudden cut-in
        ego_x_at_cutin = ego_v0 * t_cutin
        obs_x_at_cutin = ego_x_at_cutin + gap_at_cutin
        self.init_x = obs_x_at_cutin - v0 * t_cutin
        
        self.init_v = v0
        self.v_max = v_max
        self.t_cutin = t_cutin
        self.reset()

    def reset(self):
        self.x = self.init_x
        self.v = self.init_v
        self.a = 0.0
        self.active = False
        self.global_t = 0.0

    def step(self, dt):
        self.global_t += dt
        
        # Pre-activation
        if self.global_t < self.t_cutin:
            self.x += self.v * dt
            return
            
        if not self.active: self.active = True
        
        # Obstacle crawls at a steady slow speed (e.g., a slow truck)
        a_cmd = 0.0
        v_next, a_actual = _clamp_velocity(self.v + a_cmd*dt, self.v, a_cmd, dt, v_min=0.0, v_max=self.v_max)
        self.a = a_actual
        self.x += 0.5 * (self.v + v_next) * dt
        self.v = v_next

    def get_state(self):
        return ObstacleState(x=self.x, v=self.v, a=self.a, active=self.active)

    def predict_future(self, steps, dt):
        # Simple Constant Velocity (CV) prediction
        xs, vs = np.zeros(steps), np.zeros(steps)
        curr_x, curr_v = self.x, self.v
        for i in range(steps):
            xs[i] = curr_x + curr_v * dt
            vs[i] = curr_v
            curr_x = xs[i]
        return xs, vs


# ============================================================
# 3. Ego Plant (Physics + Delay)
# ============================================================
class EgoPlant:
    def __init__(self, cfg: SimulationConfig = DEFAULT_CFG, delay_steps=0, v0=20.0):
        """
        :param delay_steps: 0-zero-latency system
        """
        self.dt = cfg.DT
        self.delay_steps = int(delay_steps)
        self.v0 = v0
        self.a_min = cfg.A_MIN
        self.a_max = cfg.A_MAX
        self.v_max = cfg.V_MAX
        self.reset()

    def reset(self):
        self.x = 0.0
        self.v = self.v0
        self.a_actual = 0.0
        
        # Initialization
        if self.delay_steps > 0:
            self.delay_buffer = [0.0] * self.delay_steps
        else:
            self.delay_buffer = []

    def step(self, a_cmd):
        """
        Receives control commands and outputs realistic physical acceleration.
        """
        # 0. clipping
        clipped_cmd = np.clip(a_cmd, self.a_min, self.a_max)

        # 1. Delay Simulation
        if self.delay_steps > 0:
            self.delay_buffer.append(clipped_cmd)
            # receives commands from N steps earlier
            a_target = self.delay_buffer.pop(0)
        else:
            # no delay
            a_target = clipped_cmd
            
        self.a_actual = a_target
        
        # 2. Kinematic integral (Trapezoidal Integration)
        v_next = self.v + self.a_actual * self.dt
        
        # 3. physical modification (Engine/Brake limits)
        if v_next < 0.0:
            v_next = 0.0
            # The acceleration required to reduce the velocity to 0
            self.a_actual = -self.v / self.dt if self.dt > 0 else 0.0 
        elif v_next > self.v_max:
            v_next = self.v_max 
            # reach the maximum velocity
            self.a_actual = (self.v_max - self.v) / self.dt if self.dt > 0 else 0.0

        v_avg = 0.5 * (self.v + v_next)
        self.x += v_avg * self.dt
        self.v = v_next
        
        return self.a_actual

    def get_state(self) -> VehicleState:
        return VehicleState(x=self.x, v=self.v, a=self.a_actual)

# ============================================================
# 4. Reference Generator (IDM with DP Simulation)
# ============================================================

"""
Simulates a Grid-based Planner (DP) with IDM Reference.
It encapsulates the logic of perception noise, decision quantization, and latency simulation into a reusable class.
Can be replaced by other Planning Aproches like DeepLearning/Reinforcement Learning reference trajectory generator
"""

class IDM_DP_Planner:
    def __init__(self, cfg: SimulationConfig = DEFAULT_CFG):
        # --------------------------------------------------------
        # A. Internal Parameters (DP & Noise Simulation Config)
        # --------------------------------------------------------
        
        # --- IDM Physics Parameters ---
        self.A_MAX_COMF = cfg.A_COMF_MAX       # comfort acceleration m/s^2
        self.B_COMF = 2.0           # comfort deceleration m/s^2
        self.DELTA = 4.0            # acceleration exponent
        # self.V_DESIRED = 25.0       # speed limit
        self.desired_time_gap = cfg.TIME_GAP # seconds
        self.min_dist = cfg.MIN_DIST         # meters

        # --- Physical Limits (Hard constraints for the planner) ---
        self.A_PHYS_LIMIT = cfg.A_MIN    # Physical limit (AEB / Max Braking)
        self.A_COMF_LIMIT = cfg.A_COMF_MIN # Comfort limit (Planner soft constraint)
        self.A_MAX_LIMIT = cfg.A_MAX      # Max acceleration

        # --- Perception Noise Parameters (Sensor Model) ---
        self.POS_NOISE_BASE = getattr(cfg, 'POS_NOISE_BASE', 0.5)   # Base noise (meters)
        self.POS_NOISE_DIST_FACTOR = getattr(cfg, 'POS_NOISE_DIST_FACTOR', 0.02) # Distance-dependent noise factor (2%)
        self.VEL_NOISE_STD = getattr(cfg, 'VEL_NOISE_STD', 0.5)    # Doppler error (m/s)

        # --- DP Solver Characteristics (Simulating Discrete Planner) ---
        self.PLAN_FREQ = 5.0        # Hz (Planner runs every 0.2s)
        self.DROP_PROB = 0.05       # 5% Probability of frame drop (Planner timeout)
        
        # Quantization Grids
        self.ACC_GRID_FINE = 0.5    # Fine grid for normal driving
        self.ACC_GRID_COARSE = 2.0  # Coarse grid for emergency braking

        # --- Data Container for Debugging/Plotting ---
        # Stores the internal state of the LAST planning cycle
        self.debug_data = {
            "t": [], 
            "a_ideal": [], 
            "a_noisy": [], 
            "a_final": []
        }

    def generate_reference(self, ego_state, obs_state, horizon, dt, v_desired=None):
        """
        Main execution method.
        Generates reference trajectory using IDM but simulates a Grid-based Planner (DP).
        
        Returns:
            ref_xs, ref_vs, ref_as (Arrays for MPC/Control)
        """
        self.current_v_desired = v_desired if v_desired is not None else max(ego_state.v, 1.0)
        # Initialization
        ref_xs, ref_vs, ref_as = [], [], []
        
        # Reset debug data for this planning cycle
        self.debug_data = {"t": [], "a_ideal": [], "a_noisy": [], "a_final": []}

        # Current State
        curr_x = ego_state.x
        curr_v = ego_state.v
        
        # Ground Truth Obstacle State (for physics update)
        obs_gt_x = obs_state.x
        obs_gt_v = obs_state.v 
        
        # DP Simulation State helpers
        steps_per_plan = int((1.0 / self.PLAN_FREQ) / dt)
        last_planned_acc = 0.0 # Holds the command between planning steps (ZOH)

        # --------------------------------------------------------
        # Main Simulation Loop (Prediction Horizon)
        # --------------------------------------------------------
        for i in range(horizon):
            t_now = i * dt
            
            # 1. Update Ground Truth Environment (Constant Velocity Model)
            obs_gt_x += obs_gt_v * dt
            
            # 2. Perception Injection (The "Input" Noise)
            # Calculate distance-dependent noise
            if not obs_state.active:
                acc_ideal = self._calculate_idm_acc(curr_v, np.inf, obs_gt_v, curr_x)
                acc_noisy_raw = self._calculate_idm_acc(curr_v, np.inf, obs_gt_v, curr_x)
            else:
                # 2. Perception Injection (The "Input" Noise)
                distance = max(0.0, obs_gt_x - curr_x)
                current_pos_std = self.POS_NOISE_BASE + self.POS_NOISE_DIST_FACTOR * distance
                
                noise_p = np.random.normal(0, current_pos_std)
                noise_v = np.random.normal(0, self.VEL_NOISE_STD)
                
                obs_meas_x = obs_gt_x + noise_p
                obs_meas_v = obs_gt_v + noise_v
                
                # 3. Compute Accelerations 
                acc_ideal = self._calculate_idm_acc(curr_v, obs_gt_x, obs_gt_v, curr_x)
                # Noisy Reference (What IDM calculates based on bad sensors)
                acc_noisy_raw = self._calculate_idm_acc(curr_v, obs_meas_x, obs_meas_v, curr_x)
                
            # Ideal Reference (Ground Truth - for Debugging)
            acc_ideal = np.clip(acc_ideal, self.A_PHYS_LIMIT, self.A_MAX_LIMIT)
            
            # 4. DP Solver Logic (Discretization & Hold)
            is_planning_frame = (i % steps_per_plan == 0)
            
            if is_planning_frame:
                # Simulate Packet Loss / Computation Timeout
                if np.random.rand() < self.DROP_PROB:
                    # Keep previous command (Packet Lost)
                    current_cmd = last_planned_acc
                else:
                    # Quantization: Snap to grid
                    # A. Physics Clipping (Hard Limit)
                    solver_input = np.clip(acc_noisy_raw, self.A_PHYS_LIMIT, self.A_MAX_LIMIT)
                    
                    # B. Dual-Mode Quantization Logic
                    if solver_input >= self.A_COMF_LIMIT:
                        # Case 1: Normal Braking -> Fine Grid
                        current_cmd = round(solver_input / self.ACC_GRID_FINE) * self.ACC_GRID_FINE
                    else:
                        # Case 2: Hard Braking (Panic) -> Coarse Grid
                        current_cmd = round(solver_input / self.ACC_GRID_COARSE) * self.ACC_GRID_COARSE
                        # Ensure we don't accidentally clip above physical limit due to rounding
                        current_cmd = max(current_cmd, self.A_PHYS_LIMIT)
                    
                    # Final safety clip
                    current_cmd = np.clip(current_cmd, self.A_PHYS_LIMIT, self.A_MAX_LIMIT)
                
                last_planned_acc = current_cmd
            
            # Apply Zero-Order Hold (ZOH)
            final_acc = last_planned_acc
            
            # 5. Physics Integration (Ego moves based on Final DP Output)
            v_next = max(0.0, curr_v + final_acc * dt)
            x_next = curr_x + 0.5 * (curr_v + v_next) * dt
            
            # 6. Store Data
            ref_xs.append(x_next)
            ref_vs.append(v_next)
            ref_as.append(final_acc)
            
            # Store Debug Data (Internal History)
            self.debug_data["t"].append(t_now)
            self.debug_data["a_ideal"].append(acc_ideal)
            self.debug_data["a_noisy"].append(acc_noisy_raw)
            self.debug_data["a_final"].append(final_acc)
            
            curr_x, curr_v = x_next, v_next

        return np.array(ref_xs), np.array(ref_vs), np.array(ref_as)

    def _calculate_idm_acc(self, c_v, o_x, o_v, c_x):
        """
        Internal Helper: IDM Formula with AEB Override.
        """
        # Free road term
        acc_term = 1.0 - (c_v / self.current_v_desired) ** self.DELTA
        
        # Interaction term
        delta_v = c_v - o_v
        s_current = o_x - c_x - self.min_dist
        
        # --- AEB Logic (Industrial Standard) ---
        # If collision implies or penetration occurs, request max physical braking immediately.
        if s_current <= 0:
            return self.A_PHYS_LIMIT
            
        s_star = self.min_dist + c_v * self.desired_time_gap + \
                 (c_v * delta_v) / (2 * np.sqrt(self.A_MAX_COMF * self.B_COMF))
                 
        dec_term = (s_star / s_current) ** 2
        
        return self.A_MAX_COMF * (acc_term - dec_term)

    def plot_debug(self, save_path=None):
        """
        Visualization method.
        Can be called optionally at the end of a simulation step or episode.
        """
        if not self.debug_data["t"]:
            print("Planner: No debug data to plot.")
            return

        plt.figure(figsize=(10, 6))
        
        # Plot 1: Ideal Ground Truth (Green Dashed)
        plt.plot(self.debug_data['t'], self.debug_data['a_ideal'], 'g--', 
                 linewidth=2, alpha=0.8, label='Ideal Reference (Ground Truth)')
        
        # Plot 2: Raw Perception Input (Gray Thin)
        plt.plot(self.debug_data['t'], self.debug_data['a_noisy'], 'gray', 
                 linewidth=0.5, alpha=0.4, label='Raw IDM Output (Noisy Perception)')
        
        # Plot 3: DP Output (Red Step)
        plt.step(self.debug_data['t'], self.debug_data['a_final'], 'r-', 
                 linewidth=2.5, where='post', label='Simulated DP Output (Quantized + ZOH)')
        
        plt.title('Planner Simulation Debug View\n(Noise -> IDM -> Discretization)')
        plt.xlabel('Prediction Horizon (s)')
        plt.ylabel('Acceleration (m/s²)')
        plt.legend(loc='lower left')
        plt.grid(True, linestyle='--', alpha=0.6)
        
        if save_path:
            # Create directory if not exists
            directory = os.path.dirname(save_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
            plt.savefig(save_path)
            plt.close() # Close memory
            # print(f"Planner debug plot saved to {save_path}")
        else:
            plt.show()


