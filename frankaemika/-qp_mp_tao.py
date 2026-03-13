# Multi-serial (multi-finger) QP planner utility supporting arbitrary DoF per serial.
# Target points reaching And Obstacle avoidance via CDF distance fields.
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os
import sys
import time
import matplotlib.cm as cm
import argparse
CUR_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(CUR_PATH,'../RDF'))

from mlp import MLPRegression
import trimesh
from para_nn_cdf import CDF
from parallel_robot_layer import ParallelRobotLayer
import pybullet as p
import pybullet_data as pd
from pybullet_panda_sim import PandaSim, SphereManager

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
    # For a single integrator, A is an identity matrix and B is a dt scaled identity matrix
    A_d = torch.eye(n)
    B_d = torch.eye(n)
    return A_d, B_d


class QPPlanner:
    """Multi-serial (multi-finger) QP planner utility.

    Usage:
      planner = build_planner(paths, args, device, dt=0.01)
      q_next = planner.step(q_full, pts)

    Methods:
      step(q_full, pts, xf_full=None) -> q_next_full (numpy array)
        - q_full: 1D numpy array of full-robot joint positions (length = total_dofs)
        - pts: torch.Tensor of obstacle points, shape (N_pts, 3)
        - xf_full: optional 1D numpy array of target full-robot joint positions
    """
    def __init__(self, robot_layer, cdf, cdf_models, dt=0.0001, cons_u=2.7,
                 solver='ipopt', safety_buffer=0.01, device=None):
        self.robot_layer = robot_layer
        self.cdf = cdf
        self.cdf_models = cdf_models
        self.dofs = self.robot_layer.dof
        self.dt = dt
        self.cons_u = cons_u
        self.solver = solver
        self.safety_buffer = safety_buffer
        self.device = device if device is not None else torch.device('cpu')
        self.As = []
        self.Bs = []
        for serial in robot_layer.serials:
            A_i, B_i = create_system_matrices(serial.dof, dt)
            self.As.append(A_i.to(self.device))
            self.Bs.append(B_i.to(self.device))
        # Build a list of joint name orders for each serial to map between full q and theta
        # serial_joint_order[i] is a list of joint names in the order used by serial.Joint2Idx.keys()
        self.serial_joint_order = [list(serial.Joint2Idx.keys()) for serial in robot_layer.serials]

    def step(self, q_full, obs_pts, targ_pts):
        """Run one planning step for the whole hand.

        Args:
            q_full: numpy array shape (total_dofs,) current full robot joint positions
            obs_pts: torch.Tensor shape (N_pts, 3) obstacle points in robot frame
            targ_pts: torch.Tensor shape (N_pts, 3) target points in robot frame

        Returns:
            q_next_full: tensor shape (total_dofs,) next-step full robot joint positions
        """
        q_full = np.asarray(q_full.detach().cpu().numpy()).copy()
        total_dofs = q_full.shape[0]
        q_next = q_full.copy()

        # For each serial, extract theta from q_full, run the single-step QP, and write back
        for i, serial in enumerate(self.robot_layer.serials):
            joint_order = self.serial_joint_order[i]
            dof_i = serial.dof

            # Build theta (1D numpy) from q_full using robot_layer.Joint2Idx mapping
            theta_idxs = [self.robot_layer.Joint2Idx[joint] for joint in joint_order]
            x0_theta = q_full[theta_idxs].astype(np.float32)

            # Convert to torch and run CDF inference
            x0_torch = torch.from_numpy(x0_theta).to(self.device).reshape(1, dof_i).float()
            x0_torch.requires_grad = True

            # distance_input: torch Tensor (N_pts,) or (N_pts,1); gradient_input: (N_pts, dof_i)
            obs_dist_torch, obs_dist_grad_torch = self.cdf.inference_d_wrt_q(obs_pts, x0_torch, self.cdf_models[i], return_grad=True)
            targ_dist_torch, targ_dist_grad_torch = self.cdf.inference_d_wrt_q(targ_pts, x0_torch, self.cdf_models[i], return_grad=True)
            
            # Solve per-serial QP (returns opt_u: shape (dof_i, 1))
           
            opt_u_f = solve_optimization_problem(n_dimensions=dof_i,
                                                 cons_u=self.cons_u,
                                                 B=self.Bs[i],
                                                 targ_dist=targ_dist_torch,
                                                 targ_dist_grad=targ_dist_grad_torch,
                                                 obs_dist=obs_dist_torch,
                                                 obs_dist_grad=obs_dist_grad_torch,
                                                 dt=self.dt,
                                                 solver=self.solver,
                                                 safety_buffer=self.safety_buffer)
            # Next theta: use system update x_{k+1} = A * x_k + B * u_k
            # theta_next = (self.As[i] @ x0_theta + self.Bs[i] @ opt_u_f[:, 0]).astype(np.float32)
            print('opt_u_f:', opt_u_f)
            theta_next = torch.matmul(self.As[i], torch.from_numpy(x0_theta).to(self.device).float()) + \
                torch.matmul(self.Bs[i], torch.from_numpy(opt_u_f[:, 0]).to(self.device).float())
            # Write back into q_next
            for idx_j, joint_idx in enumerate(theta_idxs):
                q_next[joint_idx] = theta_next[idx_j]

        return torch.from_numpy(q_next).to(self.device).float()


def build_planner(paths, args, device, dt=0.01, cons_u=2.7, solver='ipopt', safety_buffer=0.3):
    """Factory to create a QPPlanner using the same conventions as sdf_grad_grasp.py

    Args:
        paths: dict of paths (same structure used in main)
        args: argparse.Namespace (must contain robot, model_dict, etc.)
        device: torch.device
    Returns:
        QPPlanner instance
    """
    robot_layer = ParallelRobotLayer(device=args.device, paths=paths, robot=args.robot)
    cdf = CDF(device=device, paths=paths, robot=args.robot, writer=None, signed_distance=False)

    cdf_models = []
    dofs = []
    As = []
    Bs = []

    # model_dicts mapping (like in main)
    model_dicts = {}
    if isinstance(paths.get('model_dict'), dict):
        model_dicts = paths['model_dict']
    else:
        for idx in range(len(robot_layer.serials)):
            model_dicts[idx] = os.path.join(CUR_PATH, f'model_dict/{args.robot}/{args.model_dict}')

    for i, serial in enumerate(robot_layer.serials):
        try:
            dof_i = len(serial.Joint2Idx)
        except Exception:
            dof_i = getattr(serial, 'dof', 1)
        dofs.append(dof_i)

        model_i = MLPRegression(input_dims=3 + dof_i, output_dims=1, mlp_layers=[1024, 512, 256, 128, 128],
                                skips=[], act_fn=torch.nn.ReLU, nerf=True)
        try:
            model_path = model_dicts.get(i, None)
            if model_path is not None and os.path.exists(model_path):
                sd = torch.load(model_path)
                if isinstance(sd, dict) and len(sd) > 0:
                    try:
                        last_val = list(sd.values())[-1]
                        model_i.load_state_dict(last_val)
                    except Exception:
                        try:
                            model_i.load_state_dict(sd[0])
                        except Exception:
                            if 49900 in sd:
                                model_i.load_state_dict(sd[49900])
                else:
                    model_i.load_state_dict(sd)
        except Exception:
            pass
        model_i.to(device)
        cdf_models.append(model_i)

        A_i, B_i = create_system_matrices(dof_i, dt)
        As.append(A_i)
        Bs.append(B_i)

    planner = QPPlanner(robot_layer, cdf, cdf_models, dofs, As, Bs, dt=dt, cons_u=cons_u, solver=solver, safety_buffer=safety_buffer, device=device)
    return planner

# set the dtype of numpy and pytorch to float32
torch.set_default_dtype(torch.float32)
np.set_printoptions(precision=4, suppress=True)
PI = 3.14

def solve_optimization_problem(n_dimensions, cons_u, B, targ_dist, targ_dist_grad, obs_dist, obs_dist_grad, dt, 
                                solver=None, safety_buffer=0.6, cost_mat_R=None, seed=42):
    """
    设置并求解优化问题 (支持任意维度)
    
    Args:
        n_dimensions: 系统维度 (DoF)
        cons_u: 控制输入约束
        B: 控制输入矩阵 (n×n)
        targ_dist: 目标点距离场
        targ_dist_grad: 目标点距离场梯度
        obs_dist: 到障碍物的距离
        obs_dist_grad: 到障碍物距离的梯度
        dt: 时间步长
        solver: 求解器类型
        safety_buffer: 安全缓冲距离
        cost_mat_R: 控制代价矩阵 (可选)
    
    Returns:
        opt_u: 优化后的控制输入
    """
    # 问题：
    # min 1/2 u^T H u + h^T u
    # s.t. g(u) <= 0
    # 其中 
    # H = (B^T * targ_dist_grad^T * targ_dist_grad * B) * dt^2 + R
    # h = 2 * B^T * targ_dist_grad^T * targ_dist * dt
    # g(u) = -obs_dist_grad * u * dt - log(obs_dist + 1- safety_buffer) <= 0
    np.random.seed(seed)
    n_controls = n_dimensions
    if cost_mat_R is None:
        # 默认R矩阵: 控制输入的惩罚
        cost_mat_R = torch.diag(torch.tensor([0.01] * n_dimensions))

    # Decision variables
    U_2d = ca.MX.sym('U', n_controls, 1)

    # Compute H and h for the objective function
    pre_H = 1/2 * torch.matmul(targ_dist_grad, B) * dt
    pre_H = pre_H.detach().cpu()
    H = 0.5* torch.matmul(pre_H.T, pre_H) + cost_mat_R
    h = 2 * torch.matmul(B.T, targ_dist_grad.T) * targ_dist * dt

    H = H.detach().cpu().numpy()
    h = h.detach().cpu().numpy()
    obs_dist = obs_dist.detach().cpu().numpy()
    obs_dist_grad = obs_dist_grad.detach().cpu().numpy()
    print('H:', H)
    print('h:', h)
    print('B:', B)
    print('targ_dist:', targ_dist)
    print('targ_dist_grad:', targ_dist_grad)
    print('obs_dist:', obs_dist)
    print('obs_dist_grad:', obs_dist_grad)

    # Objective function
    obj_2d = ca.mtimes(U_2d[:, 0].T, ca.mtimes(H, U_2d[:, 0])) + ca.mtimes(U_2d[:, 0].T, h)

    # Constraints
    g_2d = []
    
    # inequality constraints for the collision avoidance (避碰不等式约束)
    # grad * u * dt <= log(dist + 1- safety_buffer)
    # 转换为标准形式: -grad * u * dt - log(dist + 1- safety_buffer) <= 0
    g_2d.append(-ca.mtimes(ca.mtimes(obs_dist_grad, U_2d), dt) - np.log(obs_dist + 1 - safety_buffer))

    # Flatten constraints
    g_sys_vector = ca.vertcat(*g_2d)

    # Bounds for constraints
    lbg = [-np.inf] * g_sys_vector.size()[0]
    ubg = [0] * g_sys_vector.size()[0]

    # Control input bounds
    lb_u = [-cons_u] * n_controls
    ub_u = [cons_u] * n_controls

    # QP structure
    qp_x = ca.reshape(U_2d, n_controls, 1)
    qp_2d = {'x': qp_x, 'f': obj_2d, 'g': g_sys_vector}
    
    opts = {'print_time': 0, 'error_on_fail': False, 'verbose': False}

    # Create the solver
    if solver == 'ipopt':
        solver_2d = ca.nlpsol('solver', 'ipopt', qp_2d, opts)
    elif solver == 'osqp':
        solver_2d = ca.qpsol('solver', 'osqp', qp_2d, opts)
    elif solver == 'qpOASES':
        solver_2d = ca.qpsol('solver', 'qpoases', qp_2d, opts)
    elif solver == 'qrqp':
        solver_2d = ca.qpsol('solver', 'qrqp', qp_2d, opts)
    # Solve the problem
    sol_2d = solver_2d(lbg=lbg, ubg=ubg, lbx=lb_u, ubx=ub_u)

    # Extract the optimal solution
    opt_u_2d = sol_2d['x'][:].full().reshape(1, n_controls).T
    # debug: 检查 constraints 对结果的影响
    print('Optimal control input (opt_u_2d):', opt_u_2d.T)
    print('Constraint values at optimal (g_sys_vector):', ca.evalf(ca.mtimes(obs_dist_grad, opt_u_2d) * dt + np.log(obs_dist + 1 - safety_buffer)).T)
    return opt_u_2d


def ring(radius, center, rot, device):
    """
    生成环形障碍物
    
    Args:
        radius: 环的半径
        center: 环心位置
        rot: 旋转矩阵
        device: torch设备
    
    Returns:
        points: 障碍物点云
        obstacle_array: 障碍物数组 (N×4, 前3列为位置，第4列为半径)
    """
    theta = torch.arange(0, 2*PI, 0.2).to(device)
    x = radius*torch.cos(theta)
    y = radius*torch.sin(theta)
    z = torch.zeros_like(x).to(device)
    points = torch.stack([x,y,z],dim=-1)
    points = torch.matmul(points,rot.transpose(0,1)) + center

    obstacle_array = np.zeros((len(points),4))
    obstacle_array[:,:3] = points.detach().cpu().numpy()
    obstacle_array[:,3] = 0.05
    return points, obstacle_array


def wall(size, center, rot, device):
    """
    生成墙壁障碍物
    
    Args:
        size: 墙的尺寸 [宽度, 高度]
        center: 墙心位置
        rot: 旋转矩阵
        device: torch设备
    
    Returns:
        points: 障碍物点云
        obstacle_array: 障碍物数组 (N×4, 前3列为位置，第4列为半径)
    """
    x = torch.arange(-size[0]/2, size[0]/2, 0.05).to(device)
    y = torch.arange(-size[1]/2, size[1]/2, 0.05).to(device)
    x,y = torch.meshgrid(x,y)
    x,y = x.reshape(-1),y.reshape(-1)
    z = torch.zeros_like(x).to(device)
    points = torch.stack([x,y,z],dim=-1)
    points = torch.matmul(points,rot.transpose(0,1)) + center

    obstacle_array = np.zeros((len(points),4))
    obstacle_array[:,:3] = points.detach().cpu().numpy()
    obstacle_array[:,3] = 0.05
    return points, obstacle_array


def main():
    # ===================== 参数解析 =====================
    parser = argparse.ArgumentParser(description='Panda CDF Model Training and Evaluation with N-DoF support')
    parser.add_argument('--data_path', type=str, default='data_thumb_good_fingertip.pt', help='Path to the data file')
    parser.add_argument('--raw', type=str, default='data_thumb_good_fingertip.npy', help='Path to the raw data file')
    parser.add_argument('--signed_distance', action='store_true', help='Whether to use signed distance')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for training/evaluation')
    parser.add_argument('--model_dict', type=str, default='thumb_good_fingertip.pt', help='Path to save/load the model dictionary')
    parser.add_argument('--robot', type=str, default='leaphand', help='Robot type (e.g., panda)',choices=['panda','dexhand','leaphand'])
    # 可选: 指定多个serial robots（例如多根手指），以逗号分隔，例如: --finger_list thumb,index,middle
    parser.add_argument('--finger_list', type=str, default='', help='Comma-separated list of robot names for multi-finger planning')
    
    # ===================== 新增参数: 自由度设置 =====================
    parser.add_argument('--use_pybullet', action='store_true', help='Whether to visualize with PyBullet (only for 7DoF)')
    
    args = parser.parse_args()
    print(f'args:{args}')
    
    # ===================== 路径设置 =====================
    CUR_DIR = os.path.dirname(os.path.abspath(__file__))
    paths = {
        'urdf': os.path.join(CUR_DIR,f'../../RDF/descriptions/{args.robot}/*.urdf'),
        'meshes': os.path.join(CUR_DIR,f'../../RDF/descriptions/{args.robot}/meshes/*.stl'),
        'points': os.path.join(CUR_DIR,f'../../RDF/data/{args.robot}/sdf_points/'),
        'model':os.path.join(CUR_DIR, f'../../RDF/models/{args.robot}/BP_8.pt'),
        'data': os.path.join(CUR_DIR,f'data/{args.robot}/{args.data_path}'),
        'model_dict': os.path.join(CUR_DIR,f'model_dict/{args.robot}/{args.model_dict}'),
        'raw_data': os.path.join(CUR_DIR,f'data/{args.robot}/{args.raw}'),
    }
    
    # ===================== 优化问题参数设置 =====================
    N = 500  # 总步数
    dt = 0.05  # 时间步长
    T = N * dt
    print(f'The total time is: {T}')
    
    # For multi-finger case we already created per-finger A/B matrices and models above
    distance_filed = 'cdf'
    qp_solver_dict = {0: 'ipopt', 1: 'osqp', 2: 'qpOASES', 3: 'qrqp'}
    solver = 0  # 选择求解器(默认使用ipopt)
    solver = qp_solver_dict[solver]
    cons_u = 2.7  # 控制输入约束
    
    # ===================== 设备和模型初始化 =====================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create a ParallelRobotLayer to discover serials (fingers) automatically (like in sdf_grad_grasp.py)
    robot_layer = ParallelRobotLayer(device=args.device, paths=paths, robot=args.robot)

    # Single CDF instance (we will pass per-serial MLP models into its inference method)
    cdf = CDF(device, paths=paths, robot=args.robot, writer=None, signed_distance=False)

    # Build per-serial (per-finger) MLP models and system matrices
    cdf_models = []  # list of MLPRegression models (one per serial)
    dofs = []        # dof per serial
    As = []
    Bs = []

    # Prepare model_dict paths: allow a dict or single path. If single, reuse it for all serials.
    # If user provided a folder of model_dicts, they should be named consistently; fallback to same file.
    model_dicts = {}
    # If paths['model_dict'] is a dict (user provided), use it; else create a mapping with same file for all serials
    if isinstance(paths.get('model_dict'), dict):
        model_dicts = paths['model_dict']
    else:
        # map each serial index to the same model_dict file
        for idx in range(len(robot_layer.serials)):
            model_dicts[idx] = os.path.join(CUR_DIR, f'model_dict/{args.robot}/{args.model_dict}')

    for i, serial in enumerate(robot_layer.serials):
        # infer dof from serial Joint2Idx mapping (number of joints in this serial)
        try:
            dof_i = len(serial.Joint2Idx)
        except Exception:
            # fallback: try attribute 'dof'
            dof_i = getattr(serial, 'dof', 1)
        dofs.append(dof_i)

        # Build MLP per serial: input dims = 3 (point coords) + dof
        model_i = MLPRegression(input_dims=3 + dof_i, output_dims=1, mlp_layers=[1024, 512, 256, 128, 128],
                                skips=[], act_fn=torch.nn.ReLU, nerf=True)
        # Try to load per-serial model dict if available
        try:
            model_path = model_dicts.get(i, None)
            if model_path is not None and os.path.exists(model_path):
                # follow sdf_grad_grasp.py style: some checkpoints index into a state dict
                sd = torch.load(model_path)
                # if sd is a dict and contains many checkpoints, try to pick a sensible key
                if isinstance(sd, dict) and len(sd) > 0:
                    # pick the last item if it's an ordered mapping, else try numeric index
                    try:
                        # attempt to get last value
                        last_val = list(sd.values())[-1]
                        model_i.load_state_dict(last_val)
                    except Exception:
                        # try index 0
                        try:
                            model_i.load_state_dict(sd[0])
                        except Exception:
                            # try numeric last key
                            if 49900 in sd:
                                model_i.load_state_dict(sd[49900])
                else:
                    # If the file is a state_dict itself
                    model_i.load_state_dict(sd)
            else:
                print(f'Warning: model path for serial {i} not found: {model_path}; using random init')
        except Exception as e:
            print(f'Warning: failed to load model for serial {i}: {e}; using random init')
        model_i.to(device)
        cdf_models.append(model_i)

        # system matrices
        A_i, B_i = create_system_matrices(dof_i, dt)
        As.append(A_i)
        Bs.append(B_i)


    # ===================== 设置起始和目标配置 (per finger) =====================
    n_fingers = len(dofs)
    x0_list = []  # list of np arrays, one per finger, each shape = (dof_f,)
    xf_list = []
    for dof_f in dofs:
        if dof_f == 2:
            x0_f = np.array([0.0, 0.0], dtype=np.float32)
            xf_f = np.array([1.0, 1.0], dtype=np.float32)
        elif dof_f == 3:
            x0_f = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            xf_f = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        elif dof_f == 7:
            x0_f = np.array([-0.03610672,  0.14759123,  0.60442339, -2.45172895, -0.06231244, 2.53993935,  1.10256184], dtype=np.float32)
            xf_f = np.array([-0.25802498, -0.01593395, -0.35283275, -2.24489454, -0.06160258, 2.35934126,  0.34169443], dtype=np.float32)
        else:
            x0_f = np.zeros(dof_f, dtype=np.float32)
            xf_f = np.ones(dof_f, dtype=np.float32)
        x0_list.append(x0_f)
        xf_list.append(xf_f)

    for idx in range(n_fingers):
        print(f"Finger {idx} DoF: {dofs[idx]}")
        print(f"Initial configuration (finger {idx}): {x0_list[idx]}")
        print(f"Goal configuration (finger {idx}): {xf_list[idx]}")

    # ===================== 障碍物设置 =====================
    wall_size = torch.tensor([0.5, 0.5]).to(device)
    wall_center = torch.tensor([0.5, 0.0, 0.2]).to(device)
    wall_rot = torch.tensor([[1.0, 0.0, 0.0],
                             [0.0, 0.0, -1.0],
                             [0.0, 1.0, 0.0],]).to(device)
    ring_radius = 0.25
    ring_center = torch.tensor([0.5, 0.0, 0.45]).to(device)
    ring_rot = torch.tensor([[0.0, 0.0, -1.0],
                            [0.0, 1.0, 0.0],
                            [1.0, 0.0, 0.0],]).to(device)
    
    # ===================== QP优化循环 (per-finger) =====================
    # 每根手指独立规划: 对每根手指维护单独的状态、控制和日志
    safety_buffer = 0.3

    # per-finger logs: each element is a list over timesteps
    log_opt_x = [[] for _ in range(n_fingers)]  # each entry: list of states, states are np arrays shape=(dof_f,)
    log_opt_u = [[] for _ in range(n_fingers)]  # each entry: list of controls, controls shape=(dof_f,)
    log_dis_to_obstacle = [[] for _ in range(n_fingers)]

    # store last optimization solutions for reporting
    last_opt_x = [None] * n_fingers
    last_opt_u = [None] * n_fingers

    for t in range(N):
        # shared environment point cloud (same for all fingers)
        pts1, obstacle1 = wall(wall_size, wall_center, wall_rot, device)
        ring_center_current = torch.tensor([0.3, 0.0, 0.45]).to(device)
        pts2, obstacle2 = ring(0.4, ring_center_current, ring_rot, device)
        pts = torch.cat([pts1, pts2], dim=0)

        for f in range(n_fingers):
            dof_f = dofs[f]
            A_f = As[f]
            B_f = Bs[f]

            # log current state
            log_opt_x[f].append(x0_list[f].copy())

            # 推理CDF及其梯度 for this finger
            # x0_torch shape: (1, dof_f)
            x0_torch = torch.from_numpy(x0_list[f]).to(device).reshape(1, dof_f).float()
            x0_torch.requires_grad = True
            # Use the shared CDF object and the per-serial MLP (cdf_models[f]) for inference
            distance_input, gradient_input = cdf.inference_d_wrt_q(pts, x0_torch, cdf_models[f], return_grad=True)
            # distance_input: (num_points,) or (num_points,1) -> convert to numpy
            # gradient_input: (num_points, dof_f)
            distance_input = distance_input.cpu().detach().numpy()
            gradient_input = gradient_input.cpu().detach().numpy()
            log_dis_to_obstacle[f].append(distance_input)
            print(f'[t={t}] finger={f} distance_input.shape={distance_input.shape}, gradient_input.shape={gradient_input.shape}')

            # Solve per-finger QP
            opt_x_f, opt_u_f = solve_optimization_problem(dof_f, x0_list[f], xf_list[f], cons_u, A_f, B_f,
                                                         distance_input, gradient_input, dt,
                                                         solver, safety_buffer)

            # update state: x_{k+1} = A * x_k + B * u_k
            x_next = A_f @ opt_x_f[:, 0] + B_f @ opt_u_f[:, 0]
            x0_list[f] = x_next

            # store logs
            log_opt_u[f].append(opt_u_f[:, 0])
            last_opt_x[f] = opt_x_f
            last_opt_u[f] = opt_u_f

        # optional: early stop if all fingers have converged (simple heuristic)
        all_converged = True
        for f in range(n_fingers):
            if len(log_opt_u[f]) < 2:
                all_converged = False
                break
            if np.linalg.norm(log_opt_u[f][-1] - log_opt_u[f][-2]) >= 0.01:
                all_converged = False
                break
        if all_converged and t > 2:
            print(f'All fingers control inputs converged, stopping at step: {t}')
            break

    # ===================== 结果处理 (per-finger) =====================
    errors = []
    for f in range(n_fingers):
        if last_opt_x[f] is not None:
            # last_opt_x[f] shape: (dof_f, 1+1)
            err = np.linalg.norm(last_opt_x[f][:, 0] - xf_list[f])
        else:
            # fallback: distance between final state and goal
            err = np.linalg.norm(x0_list[f] - xf_list[f])
        errors.append(err)
        print(f'Finger {f} final error to goal: {err:.6f}')

    # Convert logs to arrays for easier indexing and printing
    for f in range(n_fingers):
        log_opt_x[f] = np.array(log_opt_x[f])  # shape = (steps_f, dof_f)
        log_opt_u[f] = np.array(log_opt_u[f])  # shape = (steps_f, dof_f)
        log_dis_to_obstacle[f] = np.array(log_dis_to_obstacle[f])
        print(f'Finger {f} trajectory shape: {log_opt_x[f].shape}, control shape: {log_opt_u[f].shape}')

    # ===================== 可视化 =====================
    # If single-finger and user requested pybullet and it's a 7DoF finger, use PyBullet visualization similar to before
    if args.use_pybullet and n_fingers == 1 and dofs[0] == 7:
        p.connect(p.GUI, options='--background_color_red=1 --background_color_green=1' +
                                 ' --background_color_blue=1 --width=1000 --height=1000')
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(lightPosition=[5, 5, 5])
        p.setPhysicsEngineParameter(maxNumCmdPer1ms=1000)
        p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=145, cameraPitch=-10, 
                                     cameraTargetPosition=[0, 0, 0.6])

        p.setAdditionalSearchPath(pd.getDataPath())
        timeStep = 0.01
        p.setTimeStep(timeStep)
        p.setGravity(0, 0, -9.81)
        p.setRealTimeSimulation(1)
        
        # 生成Franka机器人 (or the single robot)
        base_pos = [0, 0, 0]
        base_rot = p.getQuaternionFromEuler([0, 0, 0])
        panda = PandaSim(p, base_pos, base_rot)
        q0 = panda.get_joint_positions()
        sphere_manager = SphereManager(p)
        obstacle = np.concatenate([obstacle1, obstacle2], axis=0)
        sphere_manager.initialize_spheres(obstacle)

        # 执行轨迹
        for k in range(len(log_opt_x[0])):
            panda.set_joint_positions(log_opt_x[0][k])
            time.sleep(0.1)
    else:
        # For single-finger keep the original plotting behavior (2D/3D), otherwise plot per-finger joint traces
        if n_fingers == 1:
            dof_plot = dofs[0]
            traj = log_opt_x[0]
            x0_final = x0_list[0]
            xf_final = xf_list[0]
            if dof_plot == 2:
                plt.figure(figsize=(10, 8))
                plt.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=2, label='Trajectory')
                plt.plot(traj[0, 0], traj[0, 1], 'go', markersize=10, label='Start')
                plt.plot(xf_final[0], xf_final[1], 'r*', markersize=15, label='Goal')
                plt.xlabel('Joint 1 [rad]')
                plt.ylabel('Joint 2 [rad]')
                plt.title(f'{dof_plot}-DoF Robot Motion Planning')
                plt.legend()
                plt.grid(True)
                plt.axis('equal')
                plt.show()
            elif dof_plot == 3:
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')
                ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 'b-', linewidth=2, label='Trajectory')
                ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2], c='g', s=100, marker='o', label='Start')
                ax.scatter(xf_final[0], xf_final[1], xf_final[2], c='r', s=150, marker='*', label='Goal')
                ax.set_xlabel('Joint 1 [rad]')
                ax.set_ylabel('Joint 2 [rad]')
                ax.set_zlabel('Joint 3 [rad]')
                ax.set_title(f'{dof_plot}-DoF Robot Motion Planning')
                ax.legend()
                plt.show()
            else:
                # plot each joint over time
                fig, axes = plt.subplots(dof_plot, 1, figsize=(12, 2*dof_plot))
                if dof_plot == 1:
                    axes = [axes]
                for j in range(dof_plot):
                    axes[j].plot(traj[:, j], 'b-', linewidth=2)
                    axes[j].axhline(y=xf_final[j], color='r', linestyle='--', label='Goal')
                    axes[j].set_ylabel(f'Joint {j+1} [rad]')
                    axes[j].grid(True)
                    if j == 0:
                        axes[j].set_title(f'{dof_plot}-DoF Robot Motion Planning')
                    if j == dof_plot - 1:
                        axes[j].set_xlabel('Time Step')
                plt.tight_layout()
                plt.show()
        else:
            # Multi-finger joint traces: create one figure per finger
            for f in range(n_fingers):
                traj = log_opt_x[f]
                dof_f = dofs[f]
                fig, axes = plt.subplots(dof_f, 1, figsize=(12, 2*dof_f))
                if dof_f == 1:
                    axes = [axes]
                for j in range(dof_f):
                    axes[j].plot(traj[:, j], 'b-', linewidth=2)
                    axes[j].axhline(y=xf_list[f][j], color='r', linestyle='--', label='Goal')
                    axes[j].set_ylabel(f'Finger{f} Joint {j+1} [rad]')
                    axes[j].grid(True)
                    if j == 0:
                        axes[j].set_title(f'Finger {f} - {dof_f}-DoF Motion')
                    if j == dof_f - 1:
                        axes[j].set_xlabel('Time Step')
                plt.tight_layout()
                plt.show()

if __name__ == '__main__':
    main()
