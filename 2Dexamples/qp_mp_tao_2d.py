# -----------------------------------------------------------------------------
# 2D QP Motion Planner with CDF for 2D Robots
# Based on qp_mp_tao.py but adapted for 2D robot scenarios

# 使用：
# 如果想要把cdf模型集成到planner中，可以把step函数中的online_calculate_cdf替换为inference_cdf.
# -----------------------------------------------------------------------------

import casadi as ca
import numpy as np
import torch
import torch.nn as nn
import os
import sys
import time

CUR_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(CUR_PATH)

from mlp import MLPRegression
from robot2D_torch import Robot2D
from primitives2D_torch import Circle, Box
from cdf import CDF2D

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


class QPPlanner2D:
    """2D Robot QP planner using CDF.

    Usage:
      planner = QPPlanner2D(robot, cdf_model, dt=0.01, ...)
      q_next = planner.step(q_current, obs_objs, targ_objs)

    Methods:
      step(q, obs_objs, targ_objs) -> q_next (torch.Tensor)
        - q: current joint configuration, shape (dof,)
        - obs_objs: list of obstacle objects (Circle/Box with attract=False)
        - targ_objs: list of target objects (Circle/Box with attract=True)
    """
    def __init__(self, robot, cdf_model, dt=0.01, cons_u=2.7,
                 solver='ipopt', safety_buffer=0.01, device=None):
        """
        Args:
            robot: Robot2D instance
            cdf_model: trained MLP model for CDF inference
            dt: time step
            cons_u: control input constraint
            solver: QP solver type ('ipopt', 'osqp', 'qpOASES', 'qrqp')
            safety_buffer: safety distance buffer
            device: torch device
        """
        self.robot = robot
        self.cdf_model = cdf_model
        self.dof = robot.num_links
        self.dt = dt
        self.cons_u = cons_u
        self.cons_x = robot.joint_limits
        self.solver = solver
        self.safety_buffer = safety_buffer
        self.device = device if device is not None else torch.device('cpu')
        
        # System matrices
        self.A, self.B = create_system_matrices(self.dof, dt)
        self.A = self.A.to(self.device)
        self.B = self.B.to(self.device)
        self.cdf = CDF2D(device=self.device)
        
    def online_calculate_cdf(self, q, obj_list, return_grad=False):
        return self.cdf.calculate_cdf(q.unsqueeze(0), obj_list, method='online_computation', return_grad=return_grad)
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
        dist_pred, _ = self.cdf_model(mlp_input)  # (batch*N_pts, 1)
        dist_pred = dist_pred.reshape(batch_size, N_pts)  # (batch, N_pts)
        
        # Take minimum distance across all points
        dist_min, _ = torch.min(dist_pred, dim=1)  # (batch,)
        
        if return_grad:
            # Compute gradient w.r.t. q
            grad = torch.autograd.grad(dist_min.sum(), q, create_graph=False)[0]  # (batch, dof)
            return dist_min.squeeze(), grad.squeeze()
        
        return dist_min.squeeze()

    def step(self, q, obs_objs, targ_objs=None, noise_factor=0.0):
        """Run one planning step.

        Args:
            q: current joint configuration, torch.Tensor shape (dof,)
            obs_objs: list of obstacle objects
            targ_objs: (optional) list of target objects

        Returns:
            q_next: next joint configuration, torch.Tensor shape (dof,)
        """
        q = q.detach().clone().to(self.device)
        q.requires_grad = True
        if q.dim() == 0 or q.shape[0] != self.dof:
            raise ValueError(f"q must have shape ({self.dof},), got {q.shape}")
        
        # Compute obstacle distance and gradient
        if len(obs_objs) > 0:
            obs_dist, obs_dist_grad = self.inference_cdf(q, obs_objs, return_grad=True)
            obs_dist = obs_dist.detach().cpu().numpy()
            obs_dist_grad = obs_dist_grad.detach().cpu().numpy()
            if obs_dist_grad.ndim == 1:
                obs_dist_grad = obs_dist_grad.reshape(1, -1)  # (1, dof)
        else:
            print("No obstacles provided, skipping obstacle avoidance.")
            obs_dist = np.array([1e6])
            obs_dist_grad = np.zeros((1, self.dof))
        
        # Compute target distance and gradient
        if targ_objs is not None and len(targ_objs) > 0:
            targ_dist, targ_dist_grad = self.inference_cdf(q, targ_objs, return_grad=True)
            targ_dist = targ_dist.detach().cpu().numpy()
            targ_dist_grad = targ_dist_grad.detach().cpu().numpy()
            if targ_dist_grad.ndim == 1:
                targ_dist_grad = targ_dist_grad.reshape(1, -1)  # (1, dof)
        else:
            # No target, set to zero (no attraction)
            print("No targets provided, skipping target attraction.")
            targ_dist = np.array([0.0])
            targ_dist_grad = np.zeros((1, self.dof))
        
        # Solve QP
        # 使用siren近似cdf时，偶尔会出现一些异常的梯度，过小的target_grad会导致求解器被困在原地， 因此这里加入噪声，在target_grad过小的时候对梯度进行扰动，引导求解器前进。
        if np.linalg.norm(targ_dist_grad) < 1e-4:
            print("Target gradient is near zero, add some noise.")
            targ_dist_grad += np.random.randn(*targ_dist_grad.shape) * 1e-1
        print("Solving QP...")
        print(f"  Obstacle distance: {obs_dist}, Target distance: {targ_dist}")
        print(f"  Obstacle gradient: {obs_dist_grad}, Target gradient: {targ_dist_grad}")
        print(f"  Current q: {q.detach().cpu().numpy()}")
        print(f"  Control input bounds: {self.cons_u}")
        print(f"  Safety buffer: {self.safety_buffer}")
        opt_u = solve_optimization_problem(
            n_dimensions=self.dof,
            x0=q.detach().cpu().numpy(),
            cons_u=self.cons_u,
            cons_x=self.cons_x,
            A=self.A,
            B=self.B,
            targ_dist=torch.from_numpy(targ_dist).float(),
            targ_dist_grad=torch.from_numpy(targ_dist_grad).float(),
            obs_dist=torch.from_numpy(obs_dist).float(),
            obs_dist_grad=torch.from_numpy(obs_dist_grad).float(),
            dt=self.dt,
            solver=self.solver,
            safety_buffer=self.safety_buffer
        )
        print(f"Optimal control u: {opt_u.flatten()}")
        # Update state: q_next = A * q + B * u
        q_np = q.detach().cpu().numpy()
        A_np = self.A.detach().cpu().numpy()
        B_np = self.B.detach().cpu().numpy()
        q_next = A_np @ q_np + B_np @ opt_u[:, 0]
        # debug: 加入噪声以避免局部最优, 噪声的大小与到目标/障碍物的距离成正比
        # min_dist = np.min([targ_dist, obs_dist])
        # 不能用obs_dist，否则靠近障碍物时噪声小，容易被困住。还是得用targ_dist
        # 在obstacles附近加噪声所造成的风险是可能会撞上障碍物，可以用其他方法规避
        noise = np.random.randn(*q_next.shape) * noise_factor * (targ_dist)
        # 如果容易卡在obstacle & targert 交界处，可以考虑用下面的方式加噪声，不过又出现一个要调的参数感觉不是很优雅
        # noise = np.random.randn(*q_next.shape) * noise_factor * (targ_dist+0.1)
        q_next += noise
        
        return torch.from_numpy(q_next).float().to(self.device)


def solve_optimization_problem(n_dimensions, x0, cons_u, cons_x, A, B, targ_dist, targ_dist_grad, obs_dist, obs_dist_grad, dt,
                                solver='ipopt', safety_buffer=0.01, cost_mat_R=None, seed=42):
    """
    设置并求解优化问题 (支持任意维度)
    
    目标函数:
        min 1/2 u^T H u + h^T u
    其中:
        H = (B^T * targ_dist_grad^T * targ_dist_grad * B) * dt^2 + R
        h = 2 * B^T * targ_dist_grad^T * targ_dist * dt

    约束:
        -obs_dist_grad * B * u * dt - log(obs_dist + 1 - safety_buffer) <= 0
        -cons_u <= u <= cons_u
        -cons_x <= x <= cons_x
    其中:
        x = A * x0 + B * u
    
    Args:
        n_dimensions: 系统维度 (DoF)
        x0: 当前状态 (n×1)
        cons_u: 控制输入约束
        cons_x: 状态约束
        B: 控制输入矩阵 (n×n)
        targ_dist: 目标点距离场 (scalar or (1,))
        targ_dist_grad: 目标点距离场梯度 (1, dof)
        obs_dist: 到障碍物的距离 (scalar or (1,))
        obs_dist_grad: 到障碍物距离的梯度 (1, dof) or (N_obs, dof)
        dt: 时间步长
        solver: 求解器类型
        safety_buffer: 安全缓冲距离
        cost_mat_R: 控制代价矩阵 (可选)
    
    Returns:
        opt_u: 优化后的控制输入, shape (dof, 1)
    """
    n_controls = n_dimensions
    cons_x = np.array(cons_x)  # shape (2, dof)
    if cost_mat_R is None:
        cost_mat_R = torch.diag(torch.tensor([0.01] * n_dimensions))
    
    # Convert to numpy
    B_np = B.detach().cpu().numpy()
    targ_dist = float(targ_dist.item() if torch.is_tensor(targ_dist) else targ_dist)
    targ_dist_grad = targ_dist_grad.detach().cpu().numpy() if torch.is_tensor(targ_dist_grad) else targ_dist_grad
    obs_dist = obs_dist.detach().cpu().numpy() if torch.is_tensor(obs_dist) else obs_dist
    obs_dist_grad = obs_dist_grad.detach().cpu().numpy() if torch.is_tensor(obs_dist_grad) else obs_dist_grad
    cost_mat_R_np = cost_mat_R.detach().cpu().numpy()
    
    # Ensure correct shapes
    if targ_dist_grad.ndim == 1:
        targ_dist_grad = targ_dist_grad.reshape(1, -1)
    if obs_dist_grad.ndim == 1:
        obs_dist_grad = obs_dist_grad.reshape(1, -1)
    if np.isscalar(obs_dist):
        obs_dist = np.array([obs_dist])
    
    # Decision variables
    U = ca.MX.sym('U', n_controls, 1)

    # Compute H and h for the objective function
    # H = (B^T * targ_dist_grad^T * targ_dist_grad * B) * dt^2 + R
    # h = 2 * B^T * targ_dist_grad^T * targ_dist * dt
    pre_H = np.matmul(targ_dist_grad, B_np) * dt  # (1, dof)
    H = 0.5 * np.matmul(pre_H.T, pre_H) + cost_mat_R_np  # (dof, dof)
    h = 2 * np.matmul(B_np.T, targ_dist_grad.T) * targ_dist * dt  # (dof, 1)
    
    # Objective function
    obj = ca.mtimes(U.T, ca.mtimes(H, U)) + ca.mtimes(h.T, U)
    
    # Constraints
    g_2d = []
    
    # Collision avoidance constraints (for each obstacle point)
    # g(u) = -obs_dist_grad * u * dt - log(obs_dist + 1- safety_buffer) <= 0
    g_2d.append(-ca.mtimes(ca.mtimes(obs_dist_grad, U), dt) - np.log(obs_dist + 1 - safety_buffer))
    
    # State constraints: -cons_x <= A*x0 + B*u <= cons_x
    # 转化为不等式约束:
    # g1(u) = - A*x0 - B*u + cons_x[0] <= 0
    # g2(u) = A*x0 + B*u - cons_x[1] <= 0
    A_x0 = np.matmul(A.detach().cpu().numpy(), x0.reshape(-1, 1))  # (dof, 1)
    g_2d.append(-ca.mtimes(B_np, U) - A_x0 + cons_x[:,0])
    g_2d.append(ca.mtimes(B_np, U) + A_x0 - cons_x[:,1])

    # Flatten constraints
    if len(g_2d) > 0:
        g_vec = ca.vertcat(*g_2d)
    else:
        g_vec = ca.MX.sym('g_empty', 0, 1)

    # Bounds for constraints
    if len(g_2d) > 0:
        lbg = [-np.inf] * g_vec.size()[0]
        ubg = [0] * g_vec.size()[0]
    else:
        lbg = []
        ubg = []
    # Bounds for control inputs
    lb_u = [-cons_u] * n_controls
    ub_u = [cons_u] * n_controls
    # QP structure
    qp = {'x': U, 'f': obj, 'g': g_vec}
    # debug: 打印信息
    opts = {'print_time': 0, 'error_on_fail': False, 'verbose': False, 'printLevel': 'none'}
    
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
        print(f"Solver creation failed: {e}")
        # Fallback to zero control
        return np.zeros((n_controls, 1))
    
    # Solve the problem
    try:
        sol = solver_qp(lbx=lb_u, ubx=ub_u, lbg=lbg, ubg=ubg, x0=np.zeros(n_controls))
        opt_u = sol['x'].full().reshape(n_controls, 1)
    except Exception as e:
        print(f"Solver failed: {e}, returning zero control")
        opt_u = np.zeros((n_controls, 1))
        
    print(f'constraint value at solution: {sol["g"].full().flatten()}')
    
    return opt_u


def build_planner_2d(robot, cdf_model, device, dt=0.01, cons_u=2.7, solver='ipopt', safety_buffer=0.01):
    """Factory to create a QPPlanner2D.

    Args:
        robot: Robot2D instance
        cdf_model: trained MLP model
        device: torch.device
        dt: time step
        cons_u: control constraint
        solver: QP solver type
        safety_buffer: safety distance buffer
        
    Returns:
        QPPlanner2D instance
    """
    planner = QPPlanner2D(robot, cdf_model, dt=dt, cons_u=cons_u, solver=solver,
                          safety_buffer=safety_buffer, device=device)
    return planner


if __name__ == '__main__':
    # Simple test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create a 2-link robot
    link_lengths = torch.tensor([[2.0, 2.0]]).float()
    link_parent_map = {1: 0, 2: 1}
    init_states = torch.zeros((1, 2))
    robot = Robot2D(num_links=2, init_states=init_states, link_lengths=link_lengths,
                    link_parent_map=link_parent_map, device=device)
    
    # Create a dummy MLP model
    cdf_model = MLPRegression(input_dims=4, output_dims=1, mlp_layers=[128, 64, 32],
                               skips=[], act_fn=torch.nn.ReLU, nerf=True).to(device)
    
    # Create planner
    planner = build_planner_2d(robot, cdf_model, device, dt=0.01, cons_u=2.0, solver='ipopt')
    
    # Test step
    q = torch.tensor([0.0, 0.0]).to(device)
    obs_objs = [Circle(center=torch.tensor([1.0, 1.0]).to(device), radius=0.3, attract=False, device=device)]
    targ_objs = [Circle(center=torch.tensor([2.0, 0.0]).to(device), radius=0.1, attract=True, device=device)]
    
    print("Testing QPPlanner2D...")
    print(f"Initial q: {q}")
    
    try:
        q_next = planner.step(q, obs_objs, targ_objs)
        print(f"Next q: {q_next}")
        print("Test passed!")
    except Exception as e:
        print(f"Test failed: {e}")
