#!/usr/bin/env python3
"""
Part 2: Solvers - QP smoother, PID, MPC-QP, MPC-iLQR, delay compensation
"""
import numpy as np
import scipy.sparse as sparse
import osqp
from collections import deque
from .env import SimulationConfig, DEFAULT_CFG

# ============================================================
# Delay Compensation Methods
# ============================================================
def kinematic_forward_simulate(x, v, delay_buffer, dt):
    """
    Extrapolate the actual future state based on the historical command pipeline
    """
    x_pred, v_pred = x, v
    for a_buf in delay_buffer:
        v_next = max(0.0, v_pred + a_buf * dt)
        v_avg = 0.5 * (v_pred + v_next)
        x_pred += v_avg * dt
        v_pred = v_next
    return x_pred, v_pred

# def taylor_expansion_predict(x, v, a, delay):
#     x_pred = x + v * delay + 0.5 * a * delay ** 2
#     v_pred = max(0.0, v + a * delay)
#     return x_pred, v_pred

"""
Augmented State Matrix Builders
Rigorous mathematical modeling for Input Delay.
Model: x_{k+1} = A x_k + B u_{k-tau}
Augmented State: xi = [x, v, u_{k-1}, ..., u_{k-tau}]^T
"""

def get_linear_model_matrices(dt):
    """
    Returns standard Double Integrator matrices.
    State: z = [x, v]^T, Input: u = [a]
    """
    A = np.array([[1.0, dt],
                  [0.0, 1.0]])
    
    B = np.array([[0.5 * dt**2],
                  [dt]])
    return A, B

def build_augmented_matrices(A, B, delay_steps):
    """
    Constructs the augmented A and B matrices.
    Dimension: (2 + delay_steps) x (2 + delay_steps)
    """
    n_states = A.shape[0]   # 2
    n_controls = B.shape[1] # 1
    
    if delay_steps <= 0:
        return A, B

    total_dim = n_states + delay_steps * n_controls
    
    A_aug = np.zeros((total_dim, total_dim))
    B_aug = np.zeros((total_dim, n_controls))
    
    # 1. Physics Block (Top-Left)
    A_aug[0:n_states, 0:n_states] = A
    
    # 2. Delayed Actuation Effect (Top-Right)
    # The force is applied by the OLDEST command in buffer (u_{k-tau})
    # Located at the very end of the state vector
    idx_delayed_u = n_states + (delay_steps - 1) * n_controls
    A_aug[0:n_states, idx_delayed_u : idx_delayed_u + n_controls] = B
    
    # 3. Delay Chain / Shift Register (Bottom-Right)
    # Shift: u_{i} (next) = u_{i-1} (current)
    if delay_steps > 1:
        for i in range(delay_steps - 1):
            # Map u_{k-1-i} -> u_{k-2-i}
            row = n_states + (i + 1) * n_controls
            col = n_states + i * n_controls
            A_aug[row : row + n_controls, col : col + n_controls] = np.eye(n_controls)
            
    # 4. Input Entry (Bottom-Left of B_aug)
    # New u_k enters into u_{k-1} (first slot of buffer)
    B_aug[n_states : n_states + n_controls, :] = np.eye(n_controls)
    
    return A_aug, B_aug

def pack_augmented_state(x, v, delay_buffer, delay_steps):
    """
    Packs physical state and history buffer into the augmented vector.
    
    Args:
        delay_buffer: List of past commands [u_{k-1}, u_{k-2}, ...]
                      MUST be length == delay_steps.
    """
    if delay_steps == 0:
        return np.array([x, v])
        
    # Validation
    if len(delay_buffer) != delay_steps:
        # If buffer is too short (start of sim), pad with zeros
        padded_buffer = list(delay_buffer)
        while len(padded_buffer) < delay_steps:
            padded_buffer.append(0.0)
        # If buffer is too long, take the most recent ones
        if len(padded_buffer) > delay_steps:
             padded_buffer = padded_buffer[:delay_steps]
        x_delay = np.array(padded_buffer)
    else:
        x_delay = np.array(delay_buffer)
    
    # Structure: [x, v, u_{k-1}, u_{k-2}, ..., u_{k-tau}]
    return np.concatenate([np.array([x, v]), x_delay])


# ============================================================
# Constrained QP Smoother (Native OSQP + Sparse Matrices)
# ============================================================
class ConstrainedQPSmoother:
    """
    Solves a constrained Quadratic Programming problem for trajectory smoothing.
    Utilizes OSQP C-API with sparse matrices and warm-starting for real-time performance.
    """
    def __init__(self, cfg: SimulationConfig = DEFAULT_CFG, w_smooth=10.0, w_track=1.0):
        self.w_smooth = w_smooth
        self.w_track = w_track
        self.a_min = cfg.A_MIN
        self.a_max = cfg.A_MAX
        
        self.solver = None
        self.n = 0 # Horizon length tracker
        
    def smooth(self, ref_as):
        n = len(ref_as)
        if n == 0:
            return np.array([])
            
        # 1. Update gradient vector q (Linear term)
        # Cost: 1/2 * x^T * P * x + q^T * x
        q = -2.0 * self.w_track * np.array(ref_as)
        
        # 2. Re-initialize solver ONLY if horizon length changes (e.g., first run)
        if self.solver is None or self.n != n:
            self.n = n
            
            # Construct Hessian matrix P (Sparse Tridiagonal)
            # Multiply by 2 because standard QP form has 1/2 outside
            main_diag = np.full(n, 2.0 * self.w_track + 4.0 * self.w_smooth)
            main_diag[0] = 2.0 * self.w_track + 2.0 * self.w_smooth
            main_diag[-1] = 2.0 * self.w_track + 2.0 * self.w_smooth
            off_diag = np.full(n - 1, -2.0 * self.w_smooth)
            
            P = sparse.diags([off_diag, main_diag, off_diag], [-1, 0, 1], format='csc')
            
            # Construct inequality constraints: l <= A * x <= u
            A = sparse.eye(n, format='csc')
            l = np.full(n, self.a_min)
            u = np.full(n, self.a_max)
            
            # Setup OSQP solver instance
            self.solver = osqp.OSQP()
            self.solver.setup(P=P, q=q, A=A, l=l, u=u, 
                              verbose=False, eps_abs=1e-4, eps_rel=1e-4)
        else:
            # 3. Fast Update: Only update q vector to exploit warm-starting
            self.solver.update(q=q)
            
        # 4. Solve the QP problem
        results = self.solver.solve()
        
        if results.info.status_val == osqp.constant('OSQP_SOLVED'):
            return results.x
        else:
            # Fallback to hard clipping if infeasible or failed
            # print("[Warning] OSQP failed, applying hard clip.")
            return np.clip(ref_as, self.a_min, self.a_max)


# ============================================================
# PID Controller (Feedforward + Feedback Architecture)
# ============================================================
class PIDController:
    """
    A robust PID controller designed to work with a trajectory planner.
    Combines reference Feedforward with error-driven Feedback.
    """
    def __init__(self, kp=2.0, ki=0.1, kd=0.5, cfg: SimulationConfig = DEFAULT_CFG):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.a_min = cfg.A_MIN
        self.a_max = cfg.A_MAX
        self.reset()

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.first_run = True # Prevents derivative kick on startup

    def compute(self, error, dt, feedforward_a=0.0):
        if dt <= 1e-6:
            return feedforward_a
            
        # 1. Proportional Term
        p_term = self.kp * error
        
        # 2. Integral Term (with Anti-windup)
        # Limit the maxi compensation of the integral term to prevent long-term errors
        self.integral = np.clip(self.integral + error * dt, -5.0, 5.0)
        i_term = self.ki * self.integral
        
        # 3. Derivative Term (with Kick Prevention)
        if self.first_run:
            d_term = 0.0
            self.first_run = False
        else:
            d_term = self.kd * (error - self.prev_error) / dt
            
        self.prev_error = error
        
        # 4. Total Feedback Correction
        feedback_a = p_term + i_term + d_term
        
        # 5. Combine and constrain
        # final = QP + PID
        total_a = feedforward_a + feedback_a
        
        return np.clip(total_a, self.a_min, self.a_max)


# ============================================================
# Delay-compensated QP + PID
# ============================================================
class CompensatedQPPID:
    """
    Integrates QP smoother and PID tracker with the switch of delay compensation
    """
    def __init__(self, dt=0.1, delay_steps=0, 
                 qp_params=None, pid_params=None):
        """
        :param delay_steps: 0-no delay compensation
        """
        self.dt = dt
        self.delay_steps = int(delay_steps)
        
        # init
        qp_params = qp_params or {}
        pid_params = pid_params or {}
        self.qp = ConstrainedQPSmoother(**qp_params)
        self.pid = PIDController(**pid_params)
        
        # # history list
        # self.cmd_history = deque([0.0] * max(1, self.delay_steps), 
        #                          maxlen=max(1, self.delay_steps))

    def reset(self):
        """Reset the controller's internal state."""
        self.pid.reset()
        # self.cmd_history.clear()
        # if self.delay_steps > 0:
        #     self.cmd_history.extend([0.0] * self.delay_steps)

    def compute(self, curr_x, curr_v, ref_vs, ref_as, delay_buffer, 
                obs_x=None, obs_v=None, obs_active=False):
        """
        Main flow
        :param ref_vs: The target velocity sequence given by IDM
        :param ref_as: The rough target acceleration sequence given by IDM
        :param curr_x, curr_v: The current actual physical state given by the sensors
        :return: The final a_cmd sent to the chassis
        """
        # ----------------------------------------------------
        # Step 1：Kinematic Forward Simulation
        # ----------------------------------------------------
        if self.delay_steps > 0 and len(delay_buffer) > 0:
            x_proj, v_proj = kinematic_forward_simulate(curr_x, curr_v, delay_buffer, self.dt)
        else:
            x_proj, v_proj = curr_x, curr_v

        # ----------------------------------------------------
        # Step 2：QP Feedforward
        # ----------------------------------------------------
        smooth_as = self.qp.smooth(ref_as)
        ff_a = smooth_as[0] if len(smooth_as) > 0 else 0.0

        # ----------------------------------------------------
        # Step 3：PID Feedback
        # ----------------------------------------------------
        # PID uses v_proj instead of curr_v
        target_v = ref_vs[0] if len(ref_vs) > 0 else curr_v
        v_error = target_v - v_proj 
        
        return self.pid.compute(error=v_error, dt=self.dt, feedforward_a=ff_a)


# ============================================================
# MPC QP Solver
# ============================================================
class TrackingMPC_OSQP:
    def __init__(self, delay_steps=0, safety_mode='A',
                 q_v=10.0, r_a=0.1, w_jerk=0.5, 
                 w_slack_quad=10000.0, w_slack_lin=1000.0, cfg: SimulationConfig = DEFAULT_CFG):
        """
        :param safety_mode: 'A'(no obsticle avoidance), 'B'(hard constrain), 'C'(soft constrain with slack variables)
        """
        self.dt = cfg.DT
        self.N = cfg.MPC_HORIZON
        self.delay_steps = delay_steps
        self.safety_mode = safety_mode.upper()
        
        # weights and physical parameters
        self.q_v = q_v
        self.r_a = r_a
        self.w_jerk = w_jerk
        self.a_min = cfg.A_MIN
        self.a_max = cfg.A_MAX
        
        # safety parameters
        self.w_slack_quad = w_slack_quad
        self.w_slack_lin = w_slack_lin
        self.time_gap = cfg.TIME_GAP
        self.min_dist = cfg.MIN_DIST
        
        # 1. system matrix
        A_base, B_base = get_linear_model_matrices(self.dt)
        self.A, self.B = build_augmented_matrices(A_base, B_base, self.delay_steps)
        self.nx = self.A.shape[0]
        self.nu = self.B.shape[1]
        
        # dimension calculation
        self.n_base_vars = (self.N + 1) * self.nx + self.N * self.nu
        self.n_slack = self.N if self.safety_mode == 'C' else 0
        self.n_z = self.n_base_vars + self.n_slack # Z 向量总长度
        
        # 2. OSQP matrix
        self.solver = osqp.OSQP()
        self.is_setup = False
        self.prev_a = 0.0
        
        self._build_sparse_matrices()

    def _build_sparse_matrices(self):
        """Constructing P and A_mat, whose topology remains unchanged once established (depending on physical characteristics)"""
        # ==========================================
        # 1. Cost Hessian (P)
        # ==========================================
        P = sparse.dok_matrix((self.n_z, self.n_z))
        
        for k in range(self.N):
            idx_x = k * (self.nx + self.nu)
            idx_u = idx_x + self.nx
            
            # Tracking Cost
            P[idx_x + 1, idx_x + 1] = 2.0 * self.q_v # velocity
            P[idx_u, idx_u] += 2.0 * self.r_a        # acceleration
            
            # Jerk
            jerk_w = 2.0 * self.w_jerk / (self.dt ** 2)
            if k > 0:
                idx_u_prev = (k - 1) * (self.nx + self.nu) + self.nx
                P[idx_u, idx_u] += jerk_w
                P[idx_u_prev, idx_u_prev] += jerk_w
                P[idx_u, idx_u_prev] -= jerk_w
                P[idx_u_prev, idx_u] -= jerk_w
            else:
                P[idx_u, idx_u] += jerk_w
                
        # final v cost
        P[self.N * (self.nx + self.nu) + 1, self.N * (self.nx + self.nu) + 1] = 2.0 * self.q_v
        
        # mode C：soft constrain with slack variables in quadratic form
        if self.safety_mode == 'C':
            for i in range(self.N):
                idx_eps = self.n_base_vars + i
                P[idx_eps, idx_eps] = 2.0 * self.w_slack_quad
                
        self.P = P.tocsc()
        
        # ==========================================
        # 2. Constrain Matrix A_mat
        # ==========================================
        n_eq = (self.N + 1) * self.nx          # initial state + dynamics
        n_ineq_ctrl = self.N * self.nu         # upper and lower bounds
        n_ineq_obs = self.N if self.safety_mode in ['B', 'C'] else 0  # obstacle hard constrain
        n_ineq_slack = self.N if self.safety_mode == 'C' else 0       # slack variables are non-negative
        
        self.total_rows = n_eq + n_ineq_ctrl + n_ineq_obs + n_ineq_slack
        A_mat = sparse.dok_matrix((self.total_rows, self.n_z))
        
        row = 0
        
        # 2.1 initial state: x_0 = x_init
        for i in range(self.nx):
            A_mat[row + i, i] = 1.0
        row += self.nx
        
        # 2.2 dynamic: x_{k+1} = A x_k + B u_k
        for k in range(self.N):
            idx_x = k * (self.nx + self.nu)
            idx_u = idx_x + self.nx
            idx_x_next = (k + 1) * (self.nx + self.nu)
            
            for i in range(self.nx):
                for j in range(self.nx):
                    if self.A[i, j] != 0: A_mat[row + i, idx_x + j] = -self.A[i, j]
                for j in range(self.nu):
                    if self.B[i, j] != 0: A_mat[row + i, idx_u + j] = -self.B[i, j]
                A_mat[row + i, idx_x_next + i] = 1.0
            row += self.nx
            
        # 2.3 a_min <= u_k <= a_max
        for k in range(self.N):
            idx_u = k * (self.nx + self.nu) + self.nx
            A_mat[row, idx_u] = 1.0
            row += self.nu 
            # current only for nu = 1, single input
            
        # 2.4 obstacle wall (mode B & C)
        self.obs_row_start = row
        if self.safety_mode in ['B', 'C']:
            for k in range(self.N):
                idx_x_pos = (k + 1) * (self.nx + self.nu) # x_{k+1} index
                A_mat[row, idx_x_pos] = 1.0               # 1.0 * pos
                
                if self.safety_mode == 'C':
                    idx_eps = self.n_base_vars + k
                    A_mat[row, idx_eps] = -1.0            # pos - eps <= obs_pos
                row += 1
                
        # 2.5 non-negative slack variable: eps >= 0 (mode C)
        self.slack_row_start = row
        if self.safety_mode == 'C':
            for k in range(self.N):
                idx_eps = self.n_base_vars + k
                A_mat[row, idx_eps] = 1.0
                row += 1
                
        self.A_mat = A_mat.tocsc()

    def compute(self, curr_x, curr_v, ref_vs, ref_as, delay_buffer, 
                obs_x=None, obs_v=None, obs_active=False):
        """main flow"""
        # 1. augmented state
        x0 = pack_augmented_state(curr_x, curr_v, delay_buffer, self.delay_steps)
        
        # 2. q vector
        q = np.zeros(self.n_z)
        jerk_w = 2.0 * self.w_jerk / (self.dt ** 2)
        
        for k in range(self.N):
            idx_x = k * (self.nx + self.nu)
            v_ref = ref_vs[k] if k < len(ref_vs) else ref_vs[-1]
            q[idx_x + 1] = -2.0 * self.q_v * v_ref
            
            if k == 0:
                idx_u = idx_x + self.nx
                q[idx_u] = -jerk_w * self.prev_a
                
        q[self.N * (self.nx + self.nu) + 1] = -2.0 * self.q_v * (ref_vs[-1] if len(ref_vs)>0 else curr_v)
        
        # linear penalty for mode C
        if self.safety_mode == 'C':
            for i in range(self.N):
                q[self.n_base_vars + i] = self.w_slack_lin

        # 3. bound l, u
        l = np.zeros(self.total_rows)
        u = np.zeros(self.total_rows)
        
        # constraints (initial state + dynamics + control)
        l[:self.nx] = x0
        u[:self.nx] = x0
        
        row = (self.N + 1) * self.nx
        for _ in range(self.N):
            l[row] = self.a_min
            u[row] = self.a_max
            row += 1
            
        # obstacle constrains update
        # obstacle constrains update
        if self.safety_mode in ['B', 'C']:
            obs_vel = obs_v if (obs_v is not None) else 0.0
            
            # 🌟 Mobileye RSS 物理运动学边界
            t_rho = self.time_gap         # 反应延迟时间
            a_max_brake = abs(self.a_min) # 物理极限减速度 (9.0)
            
            # 分别计算自车和前车的极限刹停距离
            d_ego_stop = curr_v * t_rho + (curr_v ** 2) / (2.0 * a_max_brake)
            d_obs_stop = (obs_vel ** 2) / (2.0 * a_max_brake)
            
            # 动态计算绝对安全红线
            kinematic_buffer = max(0.0, d_ego_stop - d_obs_stop)
            d_safe = self.min_dist + kinematic_buffer
            
            for k in range(self.N):
                l[row] = -np.inf
                if obs_active and obs_x is not None:
                    # future obstacle position (assuming unchanged velocity)
                    obs_x_future = obs_x + (k + 1) * self.dt * obs_vel
                    u[row] = obs_x_future - d_safe
                else:
                    u[row] = np.inf # no obs
                row += 1
        # if self.safety_mode in ['B', 'C']:
        #     # safety distance based on the current velocity
        #     d_safe = self.min_dist + self.time_gap * max(curr_v, 0.1)
            
        #     for k in range(self.N):
        #         l[row] = -np.inf
        #         if obs_active and obs_x is not None:
        #             # future obstacle position (assuming unchanged velocity)
        #             obs_x_future = obs_x + (k + 1) * self.dt * (obs_v if obs_v is not None else 0.0)
        #             u[row] = obs_x_future - d_safe
        #         else:
        #             u[row] = np.inf # no obs
        #         row += 1
                
        # non negative slack variable
        if self.safety_mode == 'C':
            for k in range(self.N):
                l[row] = 0.0
                u[row] = np.inf
                row += 1

        # 4. solve
        if not self.is_setup:
            self.solver.setup(P=self.P, q=q, A=self.A_mat, l=l, u=u,
                              verbose=False, eps_abs=1e-3, eps_rel=1e-3)
            self.is_setup = True
        else:
            self.solver.update(q=q, l=l, u=u)
            
        res = self.solver.solve()
        
        if res.info.status_val == osqp.constant('OSQP_SOLVED'):
            u_opt = res.x[self.nx] # take the first step control variable
            self.prev_a = u_opt
            return u_opt
        else:
            # print(f"[Warning] OSQP Failed (Mode {self.safety_mode})! Status: {res.info.status}")
            return self.a_min # Infeasible: start AEB


# ============================================================
# MPC iLQR Solver
# ============================================================
class TrackingMPC_iLQR:
    """
    Non-linear MPC solver utilizing Iterative LQR (DDP framework).
    Handles arbitrary non-linear costs and integrates pure delay state augmentation.
    """
    def __init__(self, cost_fn, cfg: SimulationConfig = DEFAULT_CFG, delay_steps=0, max_iter=15):
        self.cost_fn = cost_fn
        self.dt = cfg.DT
        self.N = cfg.MPC_HORIZON
        self.delay_steps = delay_steps
        self.max_iter = max_iter
        
        # 1. Build fixed System Matrices (A and B are constant for linear kinematics)
        A_base, B_base = get_linear_model_matrices(self.dt)
        self.A, self.B = build_augmented_matrices(A_base, B_base, self.delay_steps)
        self.nx = self.A.shape[0]
        self.nu = self.B.shape[1]
        
        # 2. Memory allocations for nominal trajectory
        self.X_bar = np.zeros((self.N + 1, self.nx))
        self.U_bar = np.zeros((self.N, self.nu))
        self.prev_a = 0.0

    def compute(self, curr_x, curr_v, ref_vs, ref_as, delay_buffer, 
                obs_x=None, obs_v=None, obs_active=False):
        """
        Main execution step.
        """
        # 1. Pack augmented initial state
        x0 = pack_augmented_state(curr_x, curr_v, delay_buffer, self.delay_steps)
        self.X_bar[0] = x0
        
        # 2. Shift previous controls for Warm-Start (Receding Horizon trick)
        self.U_bar[:-1] = self.U_bar[1:]
        self.U_bar[-1] = self.U_bar[-2]
        self._forward_rollout()
        
        # 3. iLQR Optimization Loop
        for iteration in range(self.max_iter):
            # --- Backward Pass: Solve Riccati backwards ---
            K, k, expected_reduction = self._backward_pass(ref_vs, obs_x, obs_v, obs_active)
            
            # --- Forward Pass: Line search & trajectory update ---
            X_new, U_new = self._forward_pass(x0, K, k)
            
            self.X_bar = X_new
            self.U_bar = U_new
            
            # Early stopping if convergence is reached
            if expected_reduction < 1e-4:
                break
                
        # 4. Extract current optimal action
        u_opt = self.U_bar[0, 0]
        self.prev_a = u_opt
        return u_opt

    def _forward_rollout(self):
        """Simulates the system forward using the current U_bar"""
        for i in range(self.N):
            self.X_bar[i+1] = self.A @ self.X_bar[i] + self.B @ self.U_bar[i]

    def _backward_pass(self, ref_vs, obs_x_init, obs_v, obs_active):
        """
        Dynamic Programming step. Computes optimal feedback (K) and feedforward (k) gains.
        Includes Tikhonov Regularization to prevent Hessian collapse.
        """
        K = np.zeros((self.N, self.nu, self.nx))
        k = np.zeros((self.N, self.nu))
        
        V_x = np.zeros(self.nx)
        V_xx = np.zeros((self.nx, self.nx))
        
        expected_reduction = 0.0
        
        # Traverse backwards from N-1 to 0
        for i in range(self.N - 1, -1, -1):
            x_i = self.X_bar[i]
            u_i = self.U_bar[i]
            u_prev = self.U_bar[i-1, 0] if i > 0 else self.prev_a
            v_ref = ref_vs[i] if i < len(ref_vs) else ref_vs[-1]
            
            # Predict dynamic obstacle position (Spatiotemporal prediction!)
            curr_obs_x = None
            if obs_active and obs_x_init is not None:
                obs_velocity = obs_v if obs_v is not None else 0.0
                curr_obs_x = obs_x_init + i * self.dt * obs_velocity
                
            # Get local Taylor expansion from Cost Function
            _, lx, lu, lxx, luu = self.cost_fn.get_derivatives(
                x_i, u_i, u_prev, v_ref, curr_obs_x, obs_v, obs_active, self.dt
            )
            
            # Q-function partial derivatives
            Q_x = lx + self.A.T @ V_x
            Q_u = lu + self.B.T @ V_x
            
            Q_xx = lxx + self.A.T @ V_xx @ self.A
            Q_uu = luu + self.B.T @ V_xx @ self.B
            Q_ux = self.B.T @ V_xx @ self.A
            
            # Tikhonov Regularization (Safeguard against non-convex costs like Exponential)
            eigenvals = np.linalg.eigvalsh(Q_uu)
            if np.min(eigenvals) <= 0:
                Q_uu += (abs(np.min(eigenvals)) + 1e-4) * np.eye(self.nu)
                
            Q_uu_inv = np.linalg.inv(Q_uu)
            
            # Compute gains
            K[i] = -Q_uu_inv @ Q_ux
            k[i] = -Q_uu_inv @ Q_u
            
            # Update Value function for the previous step
            V_x = Q_x + K[i].T @ Q_uu @ k[i] + K[i].T @ Q_u + Q_ux.T @ k[i]
            V_xx = Q_xx + K[i].T @ Q_uu @ K[i] + K[i].T @ Q_ux + Q_ux.T @ K[i]
            
            expected_reduction += 0.5 * k[i].T @ Q_uu @ k[i]
            
        return K, k, expected_reduction

    def _forward_pass(self, x0, K, k, alpha=1.0):
        """
        Applies calculated gains to generate a new optimized trajectory.
        alpha is the line-search step size (1.0 for full step).
        """
        X_new = np.zeros_like(self.X_bar)
        U_new = np.zeros_like(self.U_bar)
        X_new[0] = x0
        
        for i in range(self.N):
            dx = X_new[i] - self.X_bar[i]
            # u_new = u_old + alpha * feedforward + feedback * state_error
            U_new[i] = self.U_bar[i] + alpha * k[i] + K[i] @ dx
            X_new[i+1] = self.A @ X_new[i] + self.B @ U_new[i]
            
        return X_new, U_new


# ============================================================
# Cost Functions
# ============================================================

class QuadraticSafetyCost:
    """
    Cost 1: Quadratic Safety Cost ("The Sluggish Spring").
    Equivalent to OSQP's slack variable. Provides 0 penalty until the 
    boundary is breached, then applies a quadratic penalty.
    """
    def __init__(self, nx, nu, q_v=10.0, r_a=0.1, w_jerk=0.5, 
                 w_obs=5000.0, cfg: SimulationConfig = DEFAULT_CFG):
        self.nx = nx
        self.nu = nu
        
        # Tracking weights
        self.q_v = q_v
        self.r_a = r_a
        self.w_jerk = w_jerk
        
        # Safety parameters
        self.w_obs = w_obs
        self.time_gap = cfg.TIME_GAP
        self.min_dist = cfg.MIN_DIST
        
        # Physical bounds
        self.a_min = cfg.A_MIN
        self.a_max = cfg.A_MAX
        self.w_bound = 1000.0  # Soft penalty weight for control bounds

    def get_derivatives(self, x, u, u_prev, ref_v, obs_x, obs_v, obs_active, dt):
        """
        Computes scalar cost (l), Jacobians (lx, lu), and Hessians (lxx, luu).
        """
        l = 0.0
        lx = np.zeros(self.nx)
        lu = np.zeros(self.nu)
        lxx = np.zeros((self.nx, self.nx))
        luu = np.zeros((self.nu, self.nu))

        # ---------------------------------------------------------
        # 1. Tracking Cost (Velocity + Accel + Jerk)
        # ---------------------------------------------------------
        v_err = x[1] - ref_v
        l += 0.5 * self.q_v * v_err**2 + 0.5 * self.r_a * u[0]**2
        
        lx[1] += self.q_v * v_err
        lxx[1, 1] += self.q_v
        
        lu[0] += self.r_a * u[0]
        luu[0, 0] += self.r_a

        jerk = (u[0] - u_prev) / dt
        l += 0.5 * self.w_jerk * jerk**2
        lu[0] += self.w_jerk * jerk / dt
        luu[0, 0] += self.w_jerk / (dt**2)

        # ---------------------------------------------------------
        # 2. Soft Control Bounds (Crucial since iLQR has no hard constraints)
        # ---------------------------------------------------------
        if u[0] > self.a_max:
            viol = u[0] - self.a_max
            l += 0.5 * self.w_bound * viol**2
            lu[0] += self.w_bound * viol
            luu[0, 0] += self.w_bound
        elif u[0] < self.a_min:
            viol = self.a_min - u[0]
            l += 0.5 * self.w_bound * viol**2
            lu[0] -= self.w_bound * viol
            luu[0, 0] += self.w_bound

        # ---------------------------------------------------------
        # 3. Obstacle Avoidance: RSS 动态运动学惩罚 (高斯-牛顿近似)
        # ---------------------------------------------------------
        if obs_active and obs_x is not None:
            obs_vel = obs_v if obs_v is not None else 0.0
            t_rho = self.time_gap
            a_max_brake = abs(self.a_min)
            
            # RSS 极限刹车距离推演
            d_ego_stop = x[1] * t_rho + (x[1] ** 2) / (2.0 * a_max_brake)
            d_obs_stop = (obs_vel ** 2) / (2.0 * a_max_brake)
            kinematic_buffer = d_ego_stop - d_obs_stop
            
            # 🌟 动态计算侵入深度和速度雅可比 (Jacobian)
            if kinematic_buffer > 0:
                g_x = x[0] + self.min_dist + kinematic_buffer - obs_x
                # 速度导数: d(buffer)/d(v) = t_rho + v / a_max
                grad_v = t_rho + x[1] / a_max_brake
            else:
                g_x = x[0] + self.min_dist - obs_x
                grad_v = 0.0  # 自车比前车慢足够多，允许放宽跟车
                
            if g_x > 0: # 发生红线侵入
                l += 0.5 * self.w_obs * (g_x**2)
                
                # 构造梯度向量
                grad_g = np.zeros(self.nx)
                grad_g[0] = 1.0       # 对位置的导数
                grad_g[1] = grad_v    # 对速度的物理极限导数
                
                lx += self.w_obs * g_x * grad_g
                # 使用 Gauss-Newton 近似保证海森矩阵半正定，求解极度稳定
                lxx += self.w_obs * np.outer(grad_g, grad_g)
        # # ---------------------------------------------------------
        # # 3. Obstacle Avoidance: Quadratic Penalty
        # # ---------------------------------------------------------
        # if obs_active and obs_x is not None:
        #     # Penetration depth: g(x) = pos + d_min + t_gap * vel - obs_pos
        #     g_x = x[0] + self.min_dist + self.time_gap * x[1] - obs_x
            
        #     if g_x > 0: # Penetration occurred!
        #         l += 0.5 * self.w_obs * (g_x**2)
                
        #         # Gradient of g(x) wrt state x
        #         grad_g = np.zeros(self.nx)
        #         grad_g[0] = 1.0              # d(g)/d(pos)
        #         grad_g[1] = self.time_gap    # d(g)/d(vel)
                
        #         lx += self.w_obs * g_x * grad_g
        #         lxx += self.w_obs * np.outer(grad_g, grad_g)

        return l, lx, lu, lxx, luu


class ExponentialSafetyCost(QuadraticSafetyCost):
    """
    Cost 2: Exponential Safety Cost ("The Premature Repulsion").
    APF-inspired. Extremely smooth, acts early, but risks quantum tunneling.
    """
    def __init__(self, nx, nu, length_scale=2.0, **kwargs):
        super().__init__(nx, nu, **kwargs)
        self.length_scale = length_scale

    def get_derivatives(self, x, u, u_prev, ref_v, obs_x, obs_v, obs_active, dt):
        # 1. Reuse parent class for Tracking and Bounds (Pass obs_active=False to skip quad obstacle)
        l, lx, lu, lxx, luu = super().get_derivatives(x, u, u_prev, ref_v, obs_x, obs_v, False, dt)

        # 2. Override with Exponential Obstacle Penalty
        if obs_active and obs_x is not None:
            # 获取前车速度 (如果仿真器传了前车速度的话，没传默认静止)
            obs_vel = obs_v if obs_v is not None else 0.0
            
            # 物理常量定义
            t_rho = self.time_gap         # 借用 time_gap 作为系统反应延迟
            a_max_brake = abs(self.a_min) # 自车底盘的物理极限减速度
            
            # RSS 极限刹车距离推演
            d_ego_stop = x[1] * t_rho + (x[1] ** 2) / (2.0 * a_max_brake)
            d_obs_stop = (obs_vel ** 2) / (2.0 * a_max_brake)
            kinematic_buffer = d_ego_stop - d_obs_stop
            
            # 🌟 动态计算侵入深度 (g_x) 和 速度雅可比 (grad_v)
            if kinematic_buffer > 0:
                g_x = x[0] + self.min_dist + kinematic_buffer - obs_x
                # 对 v_ego 的偏导数: d(buffer)/d(v) = t_rho + v / a_max
                grad_v = t_rho + x[1] / a_max_brake
            else:
                g_x = x[0] + self.min_dist - obs_x
                grad_v = 0.0  # 前车比自车快，或者自车极慢，速度梯度归零
                
            exponent = g_x / self.length_scale
            
            # 依然保留原版的硬截断 (防止海森矩阵 10^7 数值爆炸)
            exponent = np.clip(exponent, -20.0, 10.0) 
            
            cost_obs = self.w_obs * np.exp(exponent)
            l += cost_obs
            
            # 动态梯度计算
            grad_g = np.zeros(self.nx)
            grad_g[0] = 1.0       # 对位置的导数
            grad_g[1] = grad_v    # 🌟 融入了物理刹车极限的动态梯度
            
            grad_scalar = 1.0 / self.length_scale
            hess_scalar = grad_scalar ** 2
            
            lx += cost_obs * grad_scalar * grad_g
            # 使用 Gauss-Newton 近似保护矩阵正定性
            lxx += cost_obs * hess_scalar * np.outer(grad_g, grad_g)
        # if obs_active and obs_x is not None:
        #     # g(x) = pos + d_min + t_gap * vel - obs_pos
        #     g_x = x[0] + self.min_dist + self.time_gap * x[1] - obs_x
            
        #     # exponent = g(x) / L
        #     exponent = g_x / self.length_scale
        #     # Clip to prevent numerical overflow (Math Domain Error)
        #     exponent = np.clip(exponent, -20.0, 10.0) 
            
        #     cost_obs = self.w_obs * np.exp(exponent)
        #     l += cost_obs
            
        #     # Gradients
        #     grad_g = np.zeros(self.nx)
        #     grad_g[0] = 1.0
        #     grad_g[1] = self.time_gap
            
        #     grad_scalar = 1.0 / self.length_scale
        #     hess_scalar = grad_scalar ** 2
            
        #     # lx = J_obs * (1/L) * grad_g
        #     lx += cost_obs * grad_scalar * grad_g
        #     # lxx = J_obs * (1/L^2) * (grad_g * grad_g^T)
        #     lxx += cost_obs * hess_scalar * np.outer(grad_g, grad_g)

        return l, lx, lu, lxx, luu




