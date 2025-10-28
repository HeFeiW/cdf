# -----------------------------------------------------------------------------
# SPDX-License-Identifier: MIT
# This file is part of the CDF project.
# Copyright (c) 2024 Idiap Research Institute <contact@idiap.ch>
# Contributor: Yimming Li <yiming.li@idiap.ch>
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
    # For a single integrator, A is an identity matrix and B is a dt scaled identity matrix
    # x_{k+1} = A*x_k + B*u_k, x for position, u for velocity
    A_d = np.eye(n)
    B_d = np.eye(n) * dt
    return A_d, B_d

# set the dtype of numpy and pytorch to float32
torch.set_default_dtype(torch.float32)
np.set_printoptions(precision=4, suppress=True)
PI = 3.14

def solve_optimization_problem(n_dimensions, x0_2d, xf_2d, cons_u, A, B, distance, gradient, dt, solver=None, safety_buffer=0.6):
    """
    Set up and solve the optimization problem.
    inputs:
        n_dimensions: int, number of dimensions
        x0_2d: np.array, shape = (n_dimensions,), initial state
        xf_2d: np.array, shape = (n_dimensions,), final state
        cons_u: float, control input constraint
        A: np.array, shape = (n_dimensions, n_dimensions), system matrix
        B: np.array, shape = (n_dimensions, n_dimensions), input matrix
        distance: float, distance to the obstacle
        gradient: np.array, shape = (1, n_dimensions), gradient of the distance field
        dt: float, time step
        solver: str, solver name
    returns:
        opt_x_2d: np.array, shape = (n_dimensions, 2), optimal states
        opt_u_2d: np.array, shape = (n_dimensions, 1), optimal control inputs
    objective function: minimize (x - xf).T*Q*(x - xf) + u.T*R*u 目标位姿的跟随误差 + 控制量的能量消耗
    constraints: 
        x[:,0] = x0  系统初始状态约束
        x[:,1] = A*x[:,0] + B*u[:,0] 系统动力学约束 (x_k+1 = A*x_k + B*u_k)
        - gradient*u*dt - log(distance + safety_buffer) <= 0  避障约束
        -cons_x <= x <= cons_x  状态约束
        -cons_u <= u <= cons_u  控制量约束
    """
    n_states =  n_dimensions
    n_controls = n_dimensions

    # Decision variables (states and control inputs)
    X_2d = ca.MX.sym('X', n_states, 1+1)  # shape = (7, 2)
    U_2d = ca.MX.sym('U', n_controls, 1)  # shape = (7, 1)
    cost_mat_Q = np.diag([150, 190, 80, 70, 70, 90, 100]) # Q matrix 偏离目标的惩罚
    # cost_mat_Q = np.diag([100, 100, 100, 100, 100, 90, 100]) # Q matrix
    cost_mat_R = np.diag([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]) # R matrix 控制量的惩罚

    # Objective function (minimize control effort)
    obj_2d = 0
    x_diff = X_2d[:, 1] - xf_2d
    obj_2d += ca.mtimes(x_diff.T, ca.mtimes(cost_mat_Q, x_diff)) + ca.mtimes(U_2d[:, 0].T, ca.mtimes(cost_mat_R, U_2d[:, 0]))

    cons_x = 10
    lb_x = []  # Lower bound for the constraints
    ub_x = []  # Upper bound for the constraints
    lb_u = []
    ub_u = []
    # Adding constraints
    g_2d = []
    g_2d.append(X_2d[:, 0] - x0_2d)  # Equality constraint
    # System dynamics constraint
    g_2d.append(X_2d[:, 1] - (ca.mtimes(A, X_2d[:, 0]) + ca.mtimes(B, U_2d[:, 0])))
    # inequality constraints for the collision avoidance
    # grad * u * dt <= log(dist + 1)
    # grad * u * dt - log(dist + 1) <= 0
    g_2d.append(-ca.mtimes(ca.mtimes(gradient, U_2d), dt) - np.log(distance + safety_buffer))
    lbg = [0] * n_states * (1+1)  # Lower bound of the constraints, 1 for equality constraints, and 1 for the initial condition
    ubg = [0] * n_states * (1+1)  # Upper bound of the constraints, 1 for equality constraints, and 1 for the initial condition
    lbg.append(-np.inf)
    ubg.append(0)
    # =============================================== #
    # inequality constraints for the velocity
    lb_x.append([-cons_x, -cons_x, -cons_x, -cons_x, -cons_x, -cons_x, -cons_x])
    ub_x.append([cons_x, cons_x, cons_x, cons_x, cons_x, cons_x, cons_x])
    # inequality constraints for the control inputs
    lb_u.append([-cons_u, -cons_u, -cons_u, -cons_u, -cons_u, -cons_u, -cons_u])
    ub_u.append([cons_u, cons_u, cons_u, cons_u, cons_u, cons_u, cons_u])
    # state is one more than control
    lb_x.append([-cons_x, -cons_x, -cons_x, -cons_x, -cons_x, -cons_x, -cons_x])
    ub_x.append([cons_x, cons_x, cons_x, cons_x, cons_x, cons_x, cons_x])
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


def main():
    # debug
    parser = argparse.ArgumentParser(description='Panda CDF Model Training and Evaluation')
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
    args = parser.parse_args()
    print(f'args:{args}')
    c = input('press enter to continue')
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
    # debug end
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    # cdf = CDF(device)
    cdf = CDF(device,paths=paths,robot='panda',writer=None,signed_distance=False)
    # trainer.train_nn(epoches=20000)
    model = MLPRegression(input_dims=10, output_dims=1, mlp_layers=[1024, 512, 256, 128, 128],skips=[], act_fn=torch.nn.ReLU, nerf=True)
    # model.load_state_dict(torch.load(os.path.join(CUR_PATH,'my_model_dict.pt'))[19900])
    model.load_state_dict(torch.load(os.path.join(CUR_PATH,'model_dict.pt'))[49900])
    model.to(device)

    # Parameters of the setup
    N = 500
    dt = 0.01
    T = N * dt
    print('The total time is: ', T)
    n_dimensions = 7
    A, B = create_system_matrices(n_dimensions, dt)

    distance_filed = 'cdf'

    qp_solver_dict = {0: 'ipopt', 1: 'osqp', 2: 'qpOASES', 3: 'qrqp'}
    solver = 0

    # set start and goal configuration
    x0_7d = np.array([-0.03610672,  0.14759123,  0.60442339, -2.45172895, -0.06231244,2.53993935,  1.10256184])
    xf_7d = np.array([-0.25802498, -0.01593395, -0.35283275, -2.24489454, -0.06160258,2.35934126,  0.34169443])
    
    # choose the solver by the number in the dict
    solver = qp_solver_dict[solver]

    cons_u = 2.7  # constraints 

    #ring obstacle
    def ring(radius,center,rot):
        theta = torch.arange(0, 2*PI, 0.2).to(device)
        x = radius*torch.cos(theta)
        y = radius*torch.sin(theta)
        z = torch.zeros_like(x).to(device)
        points = torch.stack([x,y,z],dim=-1)
        points = torch.matmul(points,rot.transpose(0,1)) + center

        obstacle_array = np.zeros((len(points),4))
        obstacle_array[:,:3] = points.detach().cpu().numpy()
        obstacle_array[:,3] = 0.05
        return points,obstacle_array
    #wall obstacle
    def wall(size,center,rot):
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
        return points,obstacle_array

    wall_size = torch.tensor([0.5,0.5]).to(device)
    wall_center = torch.tensor([0.5,0.0,0.2]).to(device)
    wall_rot = torch.tensor([[1.0,0.0,0.0],
                             [0.0,0.0,-1.0],
                             [0.0,1.0,0.0],]).to(device)
    ring_radius = 0.25
    ring_center = torch.tensor([0.5,0.0,0.45]).to(device)
    ring_rot = torch.tensor([[0.0,0.0,-1.0],
                            [0.0,1.0,0.0],
                            [1.0,0.0,0.0],]).to(device)
    
    # QP problem for one time step
    log_opt_x_7d = []
    log_opt_u_7d = []
    log_dis_to_obstacle = []
    safety_buffer = 0.3

    # 这实际上是一个 基于 QP 的 Model Predictive Control (MPC) 框架，每步只求一时刻的最优控制。
    # 即，每次迭代只解决一个时间步长的优化问题，然后将系统状态更新为下一个时间步长，重复该过程直到达到目标或满足停止条件。
    for i in range(N):
        log_opt_x_7d.append(x0_7d)

        # set obstacle
        pts1,obstacle1 = wall(wall_size,wall_center,wall_rot)
        ring_center = torch.tensor([0.3,0.0,0.45]).to(device)
        pts2,obstacle2 = ring(0.4,ring_center,ring_rot)
        pts = torch.cat([pts1,pts2],dim=0)

        # infer cdf and its gradient
        x0_7d_torch = torch.from_numpy(x0_7d).to(device).reshape(1, 7).float()
        x0_7d_torch.requires_grad = True
        distance_input, gradient_input  = cdf.inference_d_wrt_q(pts,x0_7d_torch,model,return_grad = True)
        distance_input = distance_input.cpu().detach().numpy()
        gradient_input = gradient_input.cpu().detach().numpy()
        log_dis_to_obstacle.append(distance_input)

        # optimization
        opt_x_7d, opt_u_7d = solve_optimization_problem(n_dimensions, x0_7d, xf_7d, cons_u, A, B, distance_input, gradient_input,dt, solver,safety_buffer)
        # update the double integrator system
        x0_7d = A @ opt_x_7d[:, 0] + B @ opt_u_7d[:, 0]
        log_opt_u_7d.append(opt_u_7d[:, 0])

        # if the difference between current u and last u is small, we stop
        if i > 2:
            if np.linalg.norm(opt_u_7d - log_opt_u_7d[-2]) < 0.01:
                print(f'The difference is small, we stop, the current step is: {i}')
                break

    log_dis_to_obstacle = np.array(log_dis_to_obstacle)
    # compute the norm of the final state and the initial state
    error = np.linalg.norm(opt_x_7d[:, 0] - xf_7d)
    # print('The error is: ', error)
  
    log_opt_x_7d = np.array(log_opt_x_7d)  # shape = (N, 4)
    log_opt_u_7d = np.array(log_opt_u_7d)  # shape = (N, 2)

    print(log_opt_x_7d.shape)

    p.connect(p.GUI, options='--background_color_red=1 --background_color_green=1' +
                             ' --background_color_blue=1 --width=1000 --height=1000')
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.configureDebugVisualizer(lightPosition=[5, 5, 5])
    p.setPhysicsEngineParameter(maxNumCmdPer1ms=1000)

    p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=145, cameraPitch=-10, cameraTargetPosition=[0, 0, 0.6])


    p.setAdditionalSearchPath(pd.getDataPath())
    timeStep = 0.01
    p.setTimeStep(timeStep)
    p.setGravity(0, 0, -9.81)
    p.setRealTimeSimulation(1)
    ## spawn franka robot
    base_pos = [0, 0, 0]
    base_rot = p.getQuaternionFromEuler([0, 0, 0])
    panda = PandaSim(p, base_pos, base_rot)
    # panda_goal = PandaSim(p, base_pos, base_rot)
    # panda_goal = panda_goal.set_joint_positions(log_opt_x_7d[-1])
    # time.sleep(5)
    q0 = panda.get_joint_positions()
    sphere_manager = SphereManager(p)
    obstacle = np.concatenate([obstacle1,obstacle2],axis=0)
    sphere_manager.initialize_spheres(obstacle)

    for i in range(len(log_opt_x_7d)):
        panda.set_joint_positions(log_opt_x_7d[i])
        time.sleep(0.1)

if __name__ == '__main__':
    main()