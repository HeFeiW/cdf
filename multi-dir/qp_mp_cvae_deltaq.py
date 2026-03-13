# -----------------------------------------------------------------------------
# 2D QP Motion Planner with CVAE-sampled goals
# Uses CVAE to generate contact configurations, then tracks them with QP+CDF
# -----------------------------------------------------------------------------

import casadi as ca
import numpy as np
import torch
import torch.nn as nn
import os
import sys
import time
import matplotlib.pyplot as plt

CUR_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(CUR_PATH)
sys.path.append(os.path.join(CUR_PATH, '../2Dexamples'))

from mlp import MLPRegression
from robot2D_torch import Robot2D
from primitives2D_torch import Circle, Box
from cdf import CDF2D
from CVAE import CVAE
import sys
sys.path.append('../../RDF')
from Siren import Siren

# Set default dtype
torch.set_default_dtype(torch.float32)
np.set_printoptions(precision=4, suppress=True)
PI = np.pi


def create_system_matrices(n, dt):
    """
    创建系统矩阵 A 和 B
    对于单积分器系统: x_{k+1} = A*x_k + B*u_k
    
    Args:
        n: 系统维度 (DoF)
        dt: 时间步长
    
    Returns:
        A_d: 状态转移矩阵 (n×n)
        B_d: 控制输入矩阵 (n×n)
    """
    A_d = torch.eye(n)
    B_d = dt * torch.eye(n)
    return A_d, B_d


class QPPlannerCVAE:
    """2D Robot QP planner using CVAE-sampled goals and CDF for obstacle avoidance.

    Core idea:
    1. At each step, use CVAE to sample a contact configuration q_goal given target point x_goal
    2. Solve QP to track q_goal while avoiding obstacles using CDF
    
    QP Formulation:
        min  (q_{k+1} - q_goal)^T Q (q_{k+1} - q_goal) + u_k^T R u_k
        s.t. q_{k+1} = A q_k + B u_k
             -∇_q f_c(p_obs, q_k) B u_k Δt ≤ ln(f_c(p_obs, q_k) + 1 - γ)
             
    where:
        - q_goal: sampled from CVAE given x_goal (target workspace point)
        - f_c: CDF distance to obstacles
        - γ: safety buffer

    Usage:
      planner = QPPlannerCVAE(robot, cvae_model, cdf_model, dt=0.01, ...)
      q_next = planner.step(q_current, x_goal, obs_objs)
    """
    
    def __init__(self, robot, cvae_model, cdf_model, dt=0.01, cons_u=2.7,
                 solver='ipopt', safety_buffer=0.01, device=None, 
                 n_cvae_samples=5, resample_interval=1):
        """
        Args:
            robot: Robot2D instance
            cvae_model: trained CVAE model for sampling contact configs
            cdf_model: trained MLP model for CDF inference (obstacle avoidance)
            dt: time step
            cons_u: control input constraint
            solver: QP solver type ('ipopt', 'osqp', 'qpOASES', 'qrqp')
            safety_buffer: safety distance buffer for obstacles
            device: torch device
            n_cvae_samples: number of samples to draw from CVAE
            resample_interval: how often to resample q_goal (in steps)
        """
        self.robot = robot
        self.cvae_model = cvae_model
        self.cdf_model = cdf_model
        self.dof = robot.num_links
        self.dt = dt
        self.cons_u = cons_u
        self.cons_x = robot.joint_limits
        self.solver = solver
        self.safety_buffer = safety_buffer
        self.device = device if device is not None else torch.device('cpu')
        self.n_cvae_samples = n_cvae_samples
        self.resample_interval = resample_interval
        
        # System matrices
        self.A, self.B = create_system_matrices(self.dof, dt)
        self.A = self.A.to(self.device)
        self.B = self.B.to(self.device)
        self.cdf = CDF2D(device=self.device)
        
        # Track current goal and step counter
        self.current_q_goal = None
        self.step_counter = 0
        
    def sample_q_goal(self, q_current, x_goal):
        """
        Sample contact configuration from CVAE given current q and target workspace point
        
        Args:
            q_current: current joint configuration (dof,)
            x_goal: target workspace point (2,)
            
        Returns:
            q_goal: sampled contact configuration (dof,)
        """
        self.cvae_model.eval()
        
        with torch.no_grad():
            # Construct condition: [q_0, x_idx]
            condition = torch.cat([q_current, x_goal], dim=0).to(self.device)
            
            # Sample multiple candidates from CVAE
            delta_q_samples = self.cvae_model.sample(condition, n_samples=self.n_cvae_samples)  # (n_samples, dof)
            q_goal_candidates = q_current.unsqueeze(0) + delta_q_samples  # (n_samples, dof)
            
            # Select the closest one to current configuration (greedy selection)
            # Alternative: could select based on some other criterion
            distances = torch.norm(q_goal_candidates - q_current.unsqueeze(0), dim=-1)
            best_idx = torch.argmin(distances)
            q_goal = q_goal_candidates[best_idx]
            
            print(f"  CVAE sampled {self.n_cvae_samples} candidates, selected one with distance {distances[best_idx]:.4f}")
            
        return q_goal
    
    def inference_cdf(self, q, obj_list, return_grad=False):
        """
        使用训练好的MLP模型推理CDF距离场
        
        Args:
            q: joint configuration, shape (batch, dof) or (dof,)
            obj_list: list of obstacle/target objects
            return_grad: whether to return gradient w.r.t. q
            
        Returns:
            distance: CDF distance, shape (batch,) or scalar
            gradient: (optional) gradient w.r.t. q, shape (batch, dof) or (dof,)
        """
        if q.dim() == 1:
            q = q.unsqueeze(0)  # (1, dof)
        
        q.requires_grad_(return_grad)
        batch_size = q.shape[0]
        
        # Sample points on object surfaces
        pts_list = []
        for obj in obj_list:
            if hasattr(obj, 'sample_surface'):
                pts = obj.sample_surface(100)  # (N_pts, 2)
                pts_list.append(pts)
            else:
                raise ValueError("Object does not have 'sample_surface' method.")
        
        if len(pts_list) == 0:
            # No objects, return large distance
            dist = torch.ones(batch_size, device=self.device) * 10.0
            if return_grad:
                grad = torch.zeros(batch_size, self.dof, device=self.device)
                return dist.squeeze(), grad.squeeze()
            return dist.squeeze()
        
        pts = torch.cat(pts_list, dim=0)  # (N_total, 2)
        N_pts = pts.shape[0]
        
        # Expand q and pts for batch inference
        # Input to MLP: [p_x, p_y, q1, q2, ..., q_dof]
        q_expanded = q.unsqueeze(1).expand(batch_size, N_pts, self.dof)  # (batch, N_pts, dof)
        pts_expanded = pts.unsqueeze(0).expand(batch_size, N_pts, 2)  # (batch, N_pts, 2)
        
        # Concatenate: (batch, N_pts, 2+dof)
        mlp_input = torch.cat([pts_expanded, q_expanded], dim=-1)
        mlp_input = mlp_input.reshape(-1, 2 + self.dof)  # (batch*N_pts, 2+dof)
        
        # Inference
        dist_pred = self.cdf_model(mlp_input)  # (batch*N_pts, 1)
        if isinstance(dist_pred, tuple):
            dist_pred = dist_pred[0]
        dist_pred = dist_pred.reshape(batch_size, N_pts)  # (batch, N_pts)
        
        # Take minimum distance across all points
        dist_min, _ = torch.min(dist_pred, dim=1)  # (batch,)
        
        if return_grad:
            # Compute gradient w.r.t. q
            grad = torch.autograd.grad(dist_min.sum(), q, create_graph=False)[0]  # (batch, dof)
            return dist_min.squeeze(), grad.squeeze()
        
        return dist_min.squeeze()

    def step(self, q, x_goal, obs_objs, noise_factor=0.0):
        """Run one planning step.

        Args:
            q: current joint configuration, torch.Tensor shape (dof,)
            x_goal: target workspace point, torch.Tensor shape (2,)
            obs_objs: list of obstacle objects
            noise_factor: noise factor for exploration (default: 0.0)

        Returns:
            q_next: next joint configuration, torch.Tensor shape (dof,)
            q_goal: sampled goal configuration (for visualization)
        """
        q = q.detach().clone().to(self.device)
        x_goal = x_goal.detach().clone().to(self.device)
        q.requires_grad = True
        
        if q.dim() == 0 or q.shape[0] != self.dof:
            raise ValueError(f"q must have shape ({self.dof},), got {q.shape}")
        if x_goal.dim() == 0 or x_goal.shape[0] != 2:
            raise ValueError(f"x_goal must have shape (2,), got {x_goal.shape}")
        
        # Sample q_goal from CVAE (resample periodically)
        if self.current_q_goal is None or self.step_counter % self.resample_interval == 0:
            print(f"Step {self.step_counter}: Resampling q_goal from CVAE...")
            valid_goal = False
            while not valid_goal:
                self.current_q_goal = self.sample_q_goal(q, x_goal)
                # debug: 检查采样的 q_goal 是否合理（是否与obs_objs碰撞)
                # 利用 CDF 计算距离
                goal_dist, _ = self.inference_cdf(self.current_q_goal, obs_objs, return_grad=True)
                if goal_dist.item() > self.safety_buffer:
                    valid_goal = True
        else:
            print(f"Step {self.step_counter}: Reusing previous q_goal")
        
        q_goal = self.current_q_goal
        
        # Compute obstacle distance and gradient using CDF
        if len(obs_objs) > 0:
            obs_dist, obs_dist_grad = self.inference_cdf(q, obs_objs, return_grad=True)
            obs_dist = obs_dist.detach().cpu().numpy()
            obs_dist_grad = obs_dist_grad.detach().cpu().numpy()
            if obs_dist_grad.ndim == 1:
                obs_dist_grad = obs_dist_grad.reshape(1, -1)  # (1, dof)
        else:
            print("  No obstacles provided, skipping obstacle avoidance.")
            obs_dist = np.array([1e6])
            obs_dist_grad = np.zeros((1, self.dof))
        
        # Solve QP for tracking q_goal
        print(f"  Solving QP...")
        print(f"    Current q: {q.detach().cpu().numpy()}")
        print(f"    Goal q: {q_goal.detach().cpu().numpy()}")
        print(f"    Obstacle distance: {obs_dist}")
        
        opt_u = solve_qp_tracking(
            n_dimensions=self.dof,
            x0=q.detach().cpu().numpy(),
            q_goal=q_goal.detach().cpu().numpy(),
            cons_u=self.cons_u,
            cons_x=self.cons_x,
            A=self.A,
            B=self.B,
            obs_dist=torch.from_numpy(obs_dist).float(),
            obs_dist_grad=torch.from_numpy(obs_dist_grad).float(),
            dt=self.dt,
            solver=self.solver,
            safety_buffer=self.safety_buffer
        )
        
        print(f"    Optimal control u: {opt_u.flatten()}")
        
        # Update state: q_next = A * q + B * u
        q_np = q.detach().cpu().numpy()
        A_np = self.A.detach().cpu().numpy()
        B_np = self.B.detach().cpu().numpy()
        q_next = A_np @ q_np + B_np @ opt_u[:, 0]
        
        # Add noise for exploration (scaled by distance to goal)
        # Helps avoid local minima, noise magnitude proportional to goal distance
        if noise_factor > 0.0:
            goal_dist = np.linalg.norm(q_next - q_goal.detach().cpu().numpy())
            noise = np.random.randn(*q_next.shape) * noise_factor * goal_dist
            q_next += noise
            print(f"    Added noise (factor={noise_factor}, goal_dist={goal_dist:.4f}): {noise}")
        
        self.step_counter += 1
        
        return torch.from_numpy(q_next).float().to(self.device), q_goal


def solve_qp_tracking(n_dimensions, x0, q_goal, cons_u, cons_x, A, B, obs_dist, obs_dist_grad, dt,
                      solver='ipopt', safety_buffer=0.01, Q_weight=10.0, R_weight=0.01):
    """
    设置并求解 QP 跟踪问题
    
    目标函数:
        min 1/2 (q_{k+1} - q_goal)^T Q (q_{k+1} - q_goal) + 1/2 u^T R u
    
    其中:
        q_{k+1} = A x0 + B u
    
    展开后:
        min 1/2 u^T (B^T Q B) u + (B^T Q (A x0 - q_goal))^T u + const
    
    约束:
        -obs_dist_grad * B * u * dt - log(obs_dist + 1 - safety_buffer) <= 0
        -cons_u <= u <= cons_u
        -cons_x <= q_{k+1} <= cons_x
    
    Args:
        n_dimensions: 系统维度 (DoF)
        x0: 当前状态 (n,)
        q_goal: 目标配置 (n,)
        cons_u: 控制输入约束
        cons_x: 状态约束 (2, n)
        A: 状态转移矩阵 (n, n)
        B: 控制输入矩阵 (n, n)
        obs_dist: 到障碍物的距离 (scalar or (1,))
        obs_dist_grad: 到障碍物距离的梯度 (1, dof)
        dt: 时间步长
        solver: 求解器类型
        safety_buffer: 安全缓冲距离
        Q_weight: 跟踪误差权重
        R_weight: 控制代价权重
    
    Returns:
        opt_u: 优化后的控制输入, shape (dof, 1)
    """
    n_controls = n_dimensions
    cons_x = np.array(cons_x)  # shape (2, dof)
    
    # Convert to numpy
    A_np = A.detach().cpu().numpy()
    B_np = B.detach().cpu().numpy()
    obs_dist = obs_dist.detach().cpu().numpy() if torch.is_tensor(obs_dist) else obs_dist
    obs_dist_grad = obs_dist_grad.detach().cpu().numpy() if torch.is_tensor(obs_dist_grad) else obs_dist_grad
    
    # Ensure correct shapes
    if obs_dist_grad.ndim == 1:
        obs_dist_grad = obs_dist_grad.reshape(1, -1)
    # 确保 obs_dist 是 1D 数组
    if np.isscalar(obs_dist):
        obs_dist = np.array([obs_dist])
    elif obs_dist.ndim == 0:  # 0-dimensional array
        obs_dist = obs_dist.reshape(1)
    elif obs_dist.ndim > 1:
        obs_dist = obs_dist.flatten()
    
    x0 = x0.reshape(-1, 1)  # (n, 1)
    q_goal = q_goal.reshape(-1, 1)  # (n, 1)
    
    # Weight matrices
    Q = np.eye(n_dimensions) * Q_weight
    R = np.eye(n_dimensions) * R_weight
    
    # Decision variables
    U = ca.MX.sym('U', n_controls, 1)
    
    # q_{k+1} = A x0 + B u
    q_next = ca.mtimes(A_np, x0) + ca.mtimes(B_np, U)
    
    # Tracking error
    error = q_next - q_goal
    
    # Objective function: 1/2 error^T Q error + 1/2 u^T R u
    obj = 0.5 * ca.mtimes(error.T, ca.mtimes(Q, error)) + 0.5 * ca.mtimes(U.T, ca.mtimes(R, U))
    
    # Constraints
    g_list = []
    
    # Collision avoidance constraints
    # g(u) = -obs_dist_grad * B * u * dt - log(obs_dist + 1 - safety_buffer) <= 0
    collision_constraint = -ca.mtimes(ca.mtimes(obs_dist_grad, ca.mtimes(B_np, U)), dt) - np.log(obs_dist + 1 - safety_buffer)
    g_list.append(collision_constraint)
    
    # State constraints: cons_x[0] <= q_{k+1} <= cons_x[1]
    # Lower bound: q_{k+1} - cons_x[0] >= 0  =>  -(q_{k+1} - cons_x[0]) <= 0
    # Upper bound: q_{k+1} - cons_x[1] <= 0
    g_list.append(q_next - cons_x[:, 0].reshape(-1, 1))  # Lower bound
    g_list.append(q_next - cons_x[:, 1].reshape(-1, 1))  # Upper bound
    
    # Flatten constraints
    if len(g_list) > 0:
        g_vec = ca.vertcat(*g_list)
    else:
        g_vec = ca.MX.sym('g_empty', 0, 1)
    
    # Bounds for constraints
    print('obs_dist:', obs_dist)
    n_collision = obs_dist.shape[0]
    lbg = [-np.inf] * n_collision  # Collision constraints
    ubg = [0] * n_collision
    
    lbg += [0] * n_dimensions  # State lower bound constraints
    ubg += [np.inf] * n_dimensions
    
    lbg += [-np.inf] * n_dimensions  # State upper bound constraints
    ubg += [0] * n_dimensions
    
    # Bounds for control inputs
    lb_u = [-cons_u] * n_controls
    ub_u = [cons_u] * n_controls
    
    # QP structure
    qp = {'x': U, 'f': obj, 'g': g_vec}
    
    # Solver options
    opts = {'print_time': 0, 'error_on_fail': False, 'verbose': False}
    
    # Create the solver
    try:
        if solver == 'ipopt':
            opts_ipopt = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
            solver_qp = ca.nlpsol('solver', 'ipopt', qp, opts_ipopt)
        elif solver == 'osqp':
            solver_qp = ca.qpsol('solver', 'osqp', qp, opts)
        elif solver == 'qpOASES':
            solver_qp = ca.qpsol('solver', 'qpoases', qp, opts)
        elif solver == 'qrqp':
            solver_qp = ca.qpsol('solver', 'qrqp', qp, opts)
        else:
            solver_qp = ca.nlpsol('solver', 'ipopt', qp, opts)
    except Exception as e:
        print(f"    Solver creation failed: {e}")
        return np.zeros((n_controls, 1))
    
    # Solve the problem
    try:
        sol = solver_qp(lbx=lb_u, ubx=ub_u, lbg=lbg, ubg=ubg, x0=np.zeros(n_controls))
        opt_u = sol['x'].full().reshape(n_controls, 1)
        
        # Check constraints
        g_val = sol['g'].full().flatten()
        print(f'    Constraint values at solution: {g_val}')
        
    except Exception as e:
        print(f"    Solver failed: {e}, returning zero control")
        opt_u = np.zeros((n_controls, 1))
    
    return opt_u


def build_planner_cvae(robot, cvae_model, cdf_model, device, dt=0.01, cons_u=2.7, 
                       solver='ipopt', safety_buffer=0.01, n_cvae_samples=5, resample_interval=1):
    """Factory to create a QPPlannerCVAE.

    Args:
        robot: Robot2D instance
        cvae_model: trained CVAE model
        cdf_model: trained MLP model for CDF
        device: torch.device
        dt: time step
        cons_u: control constraint
        solver: QP solver type
        safety_buffer: safety distance buffer
        n_cvae_samples: number of samples from CVAE
        resample_interval: how often to resample q_goal
        
    Returns:
        QPPlannerCVAE instance
    """
    planner = QPPlannerCVAE(robot, cvae_model, cdf_model, dt=dt, cons_u=cons_u, 
                            solver=solver, safety_buffer=safety_buffer, device=device,
                            n_cvae_samples=n_cvae_samples, resample_interval=resample_interval)
    return planner


def plot_qp_planning_cvae(obj_lists, filename, cvae_model_path, cdf_model_path, 
                          q_start=None, x_goal=None, max_steps=200, rounds=1,
                          dt=0.05, cons_u=1.0, safety_buffer=0.1,
                          n_cvae_samples=5, resample_interval=5, noise_factor=0.0):
    """
    使用CVAE-based QP规划器进行轨迹规划并可视化
    
    Args:
        obj_lists: 障碍物和目标的列表 (包含Circle和Box对象)
        filename: 保存文件名
        cvae_model_path: CVAE模型路径
        cdf_model_path: CDF模型路径
        q_start: 起始关节配置, 如果为None则随机生成
        x_goal: 目标工作空间点 (2,)
        max_steps: 最大规划步数
        rounds: 规划轮数
        dt: 时间步长
        cons_u: 控制输入约束
        safety_buffer: 安全缓冲距离
        n_cvae_samples: CVAE采样数量
        resample_interval: 重采样间隔
        noise_factor: 探索噪声因子 (default: 0.0)
    """
    from plot_utils import plot_planning_results, plot_cvae_goals
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 分离障碍物和目标
    obs_objs = [obj for obj in obj_lists if not obj.attract]
    targ_objs = [obj for obj in obj_lists if obj.attract]
    
    print(f"Number of obstacle objects: {len(obs_objs)}")
    print(f"Number of target objects: {len(targ_objs)}")
    
    # Load CVAE model
    if not os.path.exists(cvae_model_path):
        raise FileNotFoundError(f"CVAE model not found at {cvae_model_path}")
    
    print(f"Loading CVAE model from {cvae_model_path}")
    checkpoint = torch.load(cvae_model_path, map_location=device)
    cvae_model = CVAE(data_dim=2, condition_dim=4, latent_dim=32, hidden_dims=[256, 128]).to(device)
    cvae_model.load_state_dict(checkpoint['model_state_dict'])
    cvae_model.eval()
    
    # Load CDF model
    if not os.path.exists(cdf_model_path):
        raise FileNotFoundError(f"CDF model not found at {cdf_model_path}")
    
    print(f"Loading CDF model from {cdf_model_path}")
    cdf_model = torch.load(cdf_model_path, map_location=device)
    cdf_model.eval()
    
    # Initialize CDF and robot
    cdf = CDF2D(device=device)
    
    # 设置目标点
    if x_goal is None:
        # 从目标对象中提取中心点
        if len(targ_objs) > 0:
            if hasattr(targ_objs[0], 'center'):
                x_goal = targ_objs[0].center
            else:
                x_goal = torch.tensor([2.0, -1.5]).to(device)
        else:
            x_goal = torch.tensor([2.0, -1.5]).to(device)
    else:
        x_goal = torch.tensor(x_goal).to(device)
    
    print(f"Target workspace point: {x_goal}")
    
    # 创建QP规划器
    planner = build_planner_cvae(cdf.robot, cvae_model, cdf_model, device, 
                                 dt=dt, cons_u=cons_u, solver='ipopt', 
                                 safety_buffer=safety_buffer,
                                 n_cvae_samples=n_cvae_samples,
                                 resample_interval=resample_interval)
    
    # 设置起始配置
    if q_start is None:
        q_start = torch.tensor([-np.pi/2, np.pi/4]).to(device)
    else:
        q_start = torch.tensor(q_start).to(device)
    
    # 执行QP规划
    print(f"\nStarting CVAE-based QP planning for {rounds} rounds...")
    q_trajectories = []
    q_goals_list = []
    
    for round_idx in range(rounds):
        print(f"\n{'='*60}")
        print(f"Round {round_idx + 1}/{rounds}")
        print(f"{'='*60}")
        
        q_trajectory = [q_start.detach().cpu().numpy()]
        q_goals = []
        q_current = q_start.clone()
        
        # Reset planner for new round
        planner.current_q_goal = None
        planner.step_counter = 0
        
        for step in range(max_steps - 1):
            try:
                q_next, q_goal = planner.step(q_current, x_goal, obs_objs, noise_factor=noise_factor)
                
                q_trajectory.append(q_next.detach().cpu().numpy())
                q_goals.append(q_goal.detach().cpu().numpy())
                q_current = q_next
                
                # Check if reached target
                dist_to_target = torch.norm(q_current - q_goal)
                if dist_to_target < 0.1:
                    print(f"  Reached target at step {step}")
                    break
                    
            except Exception as e:
                print(f"  Planning failed at step {step}: {e}")
                break
        
        q_trajectories.append(np.array(q_trajectory))
        q_goals_list.append(np.array(q_goals))
        print(f"Round {round_idx + 1} completed with {len(q_trajectory)} steps")
    
    q_trajectories = np.array(q_trajectories)  # (rounds, T, dof)
    print(f"\nPlanning finished with {len(q_trajectories)} rounds")
    
    # 可视化
    save_path = os.path.join(CUR_PATH, '../2Dexamples/image')
    plot_planning_results(cdf, cdf.robot, q_trajectories, obj_lists, device,
                         save_path, filename_prefix=filename)
    
    # Additional plot with CVAE goals
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot C-space with CVAE goals
    q_concat = np.concatenate(q_trajectories, axis=0)
    q_goals_concat = np.concatenate(q_goals_list, axis=0) if q_goals_list else None
    
    cdf.plot_cdf(ax=axes[0], obj_lists=obj_lists)
    axes[0].plot(q_concat[:, 0], q_concat[:, 1], 'r-', linewidth=2, label='Trajectory')
    axes[0].plot(q_concat[0, 0], q_concat[0, 1], 'go', markersize=10, label='Start')
    axes[0].plot(q_concat[-1, 0], q_concat[-1, 1], 'r*', markersize=15, label='End')
    
    if q_goals_concat is not None:
        plot_cvae_goals(axes[0], q_goals_concat, q_concat, 
                       title='CVAE-based QP Planning in C-Space')
    
    # Plot task space
    from plot_utils import plot_task_space_trajectory
    plot_task_space_trajectory(axes[1], cdf.robot, q_concat, obj_lists, device,
                               title='CVAE-based QP Planning in Task Space')
    
    fig.tight_layout()
    save_file = os.path.join(save_path, f'{filename}_cvae_goals.png')
    fig.savefig(save_file, dpi=300, bbox_inches='tight')
    print(f"Saved CVAE goals visualization to {save_file}")
    
    return q_trajectories, q_goals_list


if __name__ == '__main__':
    import argparse
    import matplotlib.pyplot as plt
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    parser = argparse.ArgumentParser(description='CVAE-based QP Motion Planning')
    parser.add_argument('--cvae_model', type=str, default='./checkpoints/cvae_model.pth',
                       help='Path to trained CVAE model')
    parser.add_argument('--cdf_model', type=str, default='../2Dexamples/model_dict/siren_model22.pth',
                       help='Path to trained CDF model')
    parser.add_argument('--scene', type=str, default='scene4', choices=['scene4', 'scene5'],
                       help='Scene to test')
    parser.add_argument('--max_steps', type=int, default=300,
                       help='Maximum planning steps')
    parser.add_argument('--rounds', type=int, default=1,
                       help='Number of planning rounds')
    parser.add_argument('--dt', type=float, default=0.05,
                       help='Time step')
    parser.add_argument('--cons_u', type=float, default=1.0,
                       help='Control input constraint')
    parser.add_argument('--safety_buffer', type=float, default=0.1,
                       help='Safety buffer distance')
    parser.add_argument('--n_cvae_samples', type=int, default=5,
                       help='Number of CVAE samples')
    parser.add_argument('--resample_interval', type=int, default=5,
                       help='CVAE goal resampling interval')
    parser.add_argument('--noise_factor', type=float, default=0.0,
                       help='Noise factor for exploration (default: 0.0)')
    
    args = parser.parse_args()
    
    # Define scenes (same as example.py)
    scene_4_object = [
        Box(center=torch.tensor([0.75, -1.5]).to(device), w=0.5, h=0.5, attract=False, device=device),
        Box(center=torch.tensor([0.75, -2.5]).to(device), w=0.5, h=0.5, attract=False, device=device)
    ]
    scene_4_target = [
        Box(center=torch.tensor([1.25, -1.5]).to(device), w=0.5, h=0.5, attract=True, device=device),
        Box(center=torch.tensor([1.25, -2.5]).to(device), w=0.5, h=0.5, attract=True, device=device)
    ]
    
    scene_5_object = [
        Box(center=torch.tensor([2.0, 2.0]).to(device), w=0.5, h=0.5, attract=False, device=device)
    ]
    scene_5_target = [
        Circle(center=torch.tensor([0.0, -2.25]).to(device), radius=0.25, attract=True, device=device)
    ]
    
    # Select scene
    if args.scene == 'scene4':
        obj_lists = scene_4_target + scene_4_object
        q_start = [-np.pi + 0.1, 2.2]
        # q_start = [-2.0, -2.0]
        x_goal = [1.25, -2.0]  # Center of target boxes
    else:  # scene5
        obj_lists = scene_5_object + scene_5_target
        q_start = [0.5, 0.5]
        x_goal = [0.0, -2.25]  # Center of target circle
    
    print("\n" + "="*60)
    print("CVAE-based QP Motion Planning Test")
    print("="*60)
    print(f"Scene: {args.scene}")
    print(f"CVAE model: {args.cvae_model}")
    print(f"CDF model: {args.cdf_model}")
    print(f"Start config: {q_start}")
    print(f"Target point: {x_goal}")
    
    try:
        q_trajectories, q_goals = plot_qp_planning_cvae(
            obj_lists=obj_lists,
            filename=f'{args.scene}_cvae_qp',
            cvae_model_path=args.cvae_model,
            cdf_model_path=args.cdf_model,
            q_start=q_start,
            x_goal=x_goal,
            max_steps=args.max_steps,
            rounds=args.rounds,
            dt=args.dt,
            cons_u=args.cons_u,
            safety_buffer=args.safety_buffer,
            n_cvae_samples=args.n_cvae_samples,
            resample_interval=args.resample_interval
        )
        print("\nPlanning completed successfully!")
        
    except Exception as e:
        print(f"\nPlanning failed: {e}")
        import traceback
        traceback.print_exc()
