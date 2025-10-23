# -----------------------------------------------------------------------------
# SPDX-License-Identifier: MIT
# This file is part of the CDF project.
# Copyright (c) 2024 Idiap Research Institute <contact@idiap.ch>
# Contributor: Yimming Li <yiming.li@idiap.ch>
# Modified: Support for customizable N-DoF (from 2DoF to higher dimensions)
# -----------------------------------------------------------------------------


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
from nn_cdf import CDF
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
    A_d = np.eye(n)
    B_d = np.eye(n) * dt
    return A_d, B_d

# set the dtype of numpy and pytorch to float32
torch.set_default_dtype(torch.float32)
np.set_printoptions(precision=4, suppress=True)
PI = 3.14

def solve_optimization_problem(n_dimensions, x0, xf, cons_u, A, B, distance, gradient, dt, 
                                solver=None, safety_buffer=0.6, cost_mat_Q=None, cost_mat_R=None):
    """
    设置并求解优化问题 (支持任意维度)
    
    Args:
        n_dimensions: 系统维度 (DoF)
        x0: 初始状态 (n_dimensions,)
        xf: 目标状态 (n_dimensions,)
        cons_u: 控制输入约束
        A: 状态转移矩阵 (n×n)
        B: 控制输入矩阵 (n×n)
        distance: 到障碍物的距离
        gradient: 距离场梯度
        dt: 时间步长
        solver: 求解器类型
        safety_buffer: 安全缓冲距离
        cost_mat_Q: 状态代价矩阵 (可选)
        cost_mat_R: 控制代价矩阵 (可选)
    
    Returns:
        opt_x: 优化后的状态轨迹
        opt_u: 优化后的控制输入
    """
    n_states = n_dimensions
    n_controls = n_dimensions

    # Decision variables (states and control inputs)
    X_2d = ca.MX.sym('X', n_states, 1+1)  # shape = (n_dimensions, 2)
    U_2d = ca.MX.sym('U', n_controls, 1)  # shape = (n_dimensions, 1)
    
    # 如果未提供代价矩阵，使用默认值
    if cost_mat_Q is None:
        # 默认Q矩阵: 对于不同维度使用不同的权重策略
        if n_dimensions == 2:
            cost_mat_Q = np.diag([100, 100])
        elif n_dimensions == 3:
            cost_mat_Q = np.diag([100, 100, 100])
        elif n_dimensions == 7:  # 保持原始7DoF的权重
            cost_mat_Q = np.diag([150, 190, 80, 70, 70, 90, 100])
        else:
            # 对于其他维度，使用统一权重
            cost_mat_Q = np.diag([100] * n_dimensions)
    
    if cost_mat_R is None:
        # 默认R矩阵: 控制输入的惩罚
        cost_mat_R = np.diag([0.01] * n_dimensions)

    # Objective function (minimize control effort)
    obj_2d = 0
    x_diff = X_2d[:, 1] - xf
    obj_2d += ca.mtimes(x_diff.T, ca.mtimes(cost_mat_Q, x_diff)) + ca.mtimes(U_2d[:, 0].T, ca.mtimes(cost_mat_R, U_2d[:, 0]))

    cons_x = 10  # 状态约束（速度限制）
    lb_x = []  # Lower bound for the constraints
    ub_x = []  # Upper bound for the constraints
    lb_u = []
    ub_u = []
    
    # Adding constraints
    g_2d = []
    g_2d.append(X_2d[:, 0] - x0)  # 初始条件等式约束
    # System dynamics constraint (系统动力学约束)
    g_2d.append(X_2d[:, 1] - (ca.mtimes(A, X_2d[:, 0]) + ca.mtimes(B, U_2d[:, 0])))
    
    # inequality constraints for the collision avoidance (避碰不等式约束)
    # grad * u * dt <= log(dist + safety_buffer)
    # 转换为标准形式: -grad * u * dt - log(dist + safety_buffer) <= 0
    g_2d.append(-ca.mtimes(ca.mtimes(gradient, U_2d), dt) - np.log(distance + safety_buffer))
    
    # 等式约束的上下界
    lbg = [0] * n_states * (1+1)  # Lower bound of the constraints
    ubg = [0] * n_states * (1+1)  # Upper bound of the constraints
    lbg.append(-np.inf)
    ubg.append(0)
    
    # =============================================== #
    # inequality constraints for the velocity (速度不等式约束)
    lb_x.append([-cons_x] * n_dimensions)
    ub_x.append([cons_x] * n_dimensions)
    
    # inequality constraints for the control inputs (控制输入不等式约束)
    lb_u.append([-cons_u] * n_dimensions)
    ub_u.append([cons_u] * n_dimensions)
    
    # state is one more than control (第二个状态的约束)
    lb_x.append([-cons_x] * n_dimensions)
    ub_x.append([cons_x] * n_dimensions)
    
    lbx = ca.vertcat(*lb_x, *lb_u)
    ubx = ca.vertcat(*ub_x, *ub_u)
    
    # QP structure
    X_2d_long_vector = ca.reshape(X_2d, n_states*(1+1), 1)
    U_2d_long_vector = ca.reshape(U_2d, n_controls*1, 1)
    qp_x = ca.vertcat(X_2d_long_vector, U_2d_long_vector)
    g_sys_vector = ca.vertcat(*g_2d)

    # Create the QP
    qp_2d = {'x': qp_x,
            'f': obj_2d,
            'g': g_sys_vector}

    opts = {'print_time': 0,'error_on_fail': False, 'verbose': False}

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
    sol_2d = solver_2d(lbg=lbg, ubg=ubg, lbx=lbx, ubx=ubx)
    
    # Extract the optimal solution
    opt_x_2d = sol_2d['x'][:n_states*(1+1)].full().reshape(1+1, n_states).T
    opt_u_2d = sol_2d['x'][n_states*(1+1):].full().reshape(1, n_controls).T

    return opt_x_2d, opt_u_2d


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
    parser.add_argument('--with_writer', action='store_true', help='Whether to use TensorBoard writer')
    parser.add_argument('--eval', action='store_true', help='Whether to evaluate the model')
    parser.add_argument('--train', action='store_true', help='Whether to train the model')
    parser.add_argument('--epoches', type=int, default=50000, help='Number of training epochs')
    parser.add_argument('--batch_x', type=int, default=10, help='Batch size for x')
    parser.add_argument('--batch_q', type=int, default=100, help='Batch size for q')
    parser.add_argument('--signed_distance', action='store_true', help='Whether to use signed distance')
    parser.add_argument('--max_q_per_link', type=int, default=100, help='Maximum number of q samples per link')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for training/evaluation')
    parser.add_argument('--model_dict', type=str, default='thumb_good_fingertip.pt', help='Path to save/load the model dictionary')
    parser.add_argument('--robot', type=str, default='panda', help='Robot type (e.g., panda)',choices=['panda','dexhand','leaphand'])
    parser.add_argument('--serial_idx', type=int, default=0, help='Serial index for different runs')
    
    # ===================== 新增参数: 自由度设置 =====================
    parser.add_argument('--dof', type=int, default=7, help='Degrees of freedom (2 to higher dimensions, default=7)')
    parser.add_argument('--use_pybullet', action='store_true', help='Whether to visualize with PyBullet (only for 7DoF)')
    
    args = parser.parse_args()
    print(f'args:{args}')
    
    # 验证DoF参数
    if args.dof < 2:
        raise ValueError("DoF must be at least 2")
    
    # 如果不是7DoF，关闭PyBullet可视化
    if args.dof != 7 and args.use_pybullet:
        print(f"Warning: PyBullet visualization only supports 7DoF. Disabling visualization.")
        args.use_pybullet = False
    
    c = input('press enter to continue')
    
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
    
    # ===================== 设备和模型初始化 =====================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cdf = CDF(device, paths=paths, robot='panda', writer=None, signed_distance=False)
    
    # 注意: MLP的输入维度需要根据DoF调整 (3 for obstacle position + dof for joint angles)
    model = MLPRegression(input_dims=3+args.dof, output_dims=1, mlp_layers=[1024, 512, 256, 128, 128],
                          skips=[], act_fn=torch.nn.ReLU, nerf=True)
    model.load_state_dict(torch.load(os.path.join(CUR_DIR,'my_model_dict.pt'))[390])
    model.to(device)

    # ===================== 优化问题参数设置 =====================
    N = 500  # 总步数
    dt = 0.01  # 时间步长
    T = N * dt
    print(f'The total time is: {T}')
    
    n_dimensions = args.dof  # 使用命令行参数指定的自由度
    A, B = create_system_matrices(n_dimensions, dt)

    distance_filed = 'cdf'
    qp_solver_dict = {0: 'ipopt', 1: 'osqp', 2: 'qpOASES', 3: 'qrqp'}
    solver = 0  # 选择求解器
    solver = qp_solver_dict[solver]
    cons_u = 2.7  # 控制输入约束

    # ===================== 设置起始和目标配置 =====================
    # 根据不同的DoF设置不同的起始和目标配置
    if n_dimensions == 2:
        x0 = np.array([0.0, 0.0])
        xf = np.array([1.0, 1.0])
    elif n_dimensions == 3:
        x0 = np.array([0.0, 0.0, 0.0])
        xf = np.array([1.0, 1.0, 1.0])
    elif n_dimensions == 7:
        # 保持原始7DoF配置
        x0 = np.array([-0.03610672,  0.14759123,  0.60442339, -2.45172895, -0.06231244, 2.53993935,  1.10256184])
        xf = np.array([-0.25802498, -0.01593395, -0.35283275, -2.24489454, -0.06160258, 2.35934126,  0.34169443])
    else:
        # 对于其他维度，使用简单的线性插值
        x0 = np.zeros(n_dimensions)
        xf = np.ones(n_dimensions)
    
    print(f"DoF: {n_dimensions}")
    print(f"Initial configuration: {x0}")
    print(f"Goal configuration: {xf}")

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
    
    # ===================== QP优化循环 =====================
    log_opt_x = []
    log_opt_u = []
    log_dis_to_obstacle = []
    safety_buffer = 0.3

    for i in range(N):
        log_opt_x.append(x0.copy())

        # 设置障碍物
        pts1, obstacle1 = wall(wall_size, wall_center, wall_rot, device)
        ring_center_current = torch.tensor([0.3, 0.0, 0.45]).to(device)
        pts2, obstacle2 = ring(0.4, ring_center_current, ring_rot, device)
        pts = torch.cat([pts1, pts2], dim=0)

        # 推理CDF及其梯度
        x0_torch = torch.from_numpy(x0).to(device).reshape(1, n_dimensions).float()
        x0_torch.requires_grad = True
        distance_input, gradient_input = cdf.inference_d_wrt_q(pts, x0_torch, model, return_grad=True)
        distance_input = distance_input.cpu().detach().numpy()
        gradient_input = gradient_input.cpu().detach().numpy()
        log_dis_to_obstacle.append(distance_input)

        # 优化求解
        opt_x, opt_u = solve_optimization_problem(n_dimensions, x0, xf, cons_u, A, B, 
                                                   distance_input, gradient_input, dt, 
                                                   solver, safety_buffer)
        
        # 更新系统状态
        x0 = A @ opt_x[:, 0] + B @ opt_u[:, 0]
        log_opt_u.append(opt_u[:, 0])

        # 提前终止条件: 如果控制输入变化很小，提前停止
        if i > 2:
            if np.linalg.norm(opt_u - log_opt_u[-2]) < 0.01:
                print(f'Control input converged, stopping at step: {i}')
                break

    # ===================== 结果处理 =====================
    log_dis_to_obstacle = np.array(log_dis_to_obstacle)
    error = np.linalg.norm(opt_x[:, 0] - xf)
    print(f'Final error to goal: {error:.6f}')
  
    log_opt_x = np.array(log_opt_x)  # shape = (steps, n_dimensions)
    log_opt_u = np.array(log_opt_u)  # shape = (steps, n_dimensions)

    print(f"Trajectory shape: {log_opt_x.shape}")
    print(f"Control inputs shape: {log_opt_u.shape}")

    # ===================== PyBullet可视化 (仅7DoF) =====================
    if args.use_pybullet and n_dimensions == 7:
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
        
        # 生成Franka机器人
        base_pos = [0, 0, 0]
        base_rot = p.getQuaternionFromEuler([0, 0, 0])
        panda = PandaSim(p, base_pos, base_rot)
        q0 = panda.get_joint_positions()
        sphere_manager = SphereManager(p)
        obstacle = np.concatenate([obstacle1, obstacle2], axis=0)
        sphere_manager.initialize_spheres(obstacle)

        # 执行轨迹
        for i in range(len(log_opt_x)):
            panda.set_joint_positions(log_opt_x[i])
            time.sleep(0.1)
    else:
        # ===================== 简单的matplotlib可视化 (2D/3D) =====================
        if n_dimensions == 2:
            plt.figure(figsize=(10, 8))
            plt.plot(log_opt_x[:, 0], log_opt_x[:, 1], 'b-', linewidth=2, label='Trajectory')
            plt.plot(x0[0], x0[1], 'go', markersize=10, label='Start')
            plt.plot(xf[0], xf[1], 'r*', markersize=15, label='Goal')
            plt.xlabel('Joint 1 [rad]')
            plt.ylabel('Joint 2 [rad]')
            plt.title(f'{n_dimensions}-DoF Robot Motion Planning')
            plt.legend()
            plt.grid(True)
            plt.axis('equal')
            plt.show()
        elif n_dimensions == 3:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.plot(log_opt_x[:, 0], log_opt_x[:, 1], log_opt_x[:, 2], 'b-', linewidth=2, label='Trajectory')
            ax.scatter(x0[0], x0[1], x0[2], c='g', s=100, marker='o', label='Start')
            ax.scatter(xf[0], xf[1], xf[2], c='r', s=150, marker='*', label='Goal')
            ax.set_xlabel('Joint 1 [rad]')
            ax.set_ylabel('Joint 2 [rad]')
            ax.set_zlabel('Joint 3 [rad]')
            ax.set_title(f'{n_dimensions}-DoF Robot Motion Planning')
            ax.legend()
            plt.show()
        else:
            # 对于高维度，绘制每个关节随时间的变化
            fig, axes = plt.subplots(n_dimensions, 1, figsize=(12, 2*n_dimensions))
            if n_dimensions == 1:
                axes = [axes]
            for i in range(n_dimensions):
                axes[i].plot(log_opt_x[:, i], 'b-', linewidth=2)
                axes[i].axhline(y=xf[i], color='r', linestyle='--', label='Goal')
                axes[i].set_ylabel(f'Joint {i+1} [rad]')
                axes[i].grid(True)
                if i == 0:
                    axes[i].set_title(f'{n_dimensions}-DoF Robot Motion Planning')
                if i == n_dimensions - 1:
                    axes[i].set_xlabel('Time Step')
            plt.tight_layout()
            plt.show()

if __name__ == '__main__':
    main()
