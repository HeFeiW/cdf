# -----------------------------------------------------------------------------
# SPDX-License-Identifier: MIT
# This file is part of the CDF project.
# Copyright (c) 2024 Idiap Research Institute <contact@idiap.ch>
# Contributor: Yimming Li <yiming.li@idiap.ch>
# -----------------------------------------------------------------------------


import pybullet as p
import pybullet_data as pd
import numpy as np
import sys
sys.path.append("../../RDF")
sys.path.append("../../RDF/panda_layer")
from parallel_bf_sdf import ParallelBPSDF
from parallel_robot_layer import ParallelRobotLayer
from para_nn_cdf import CDF

import time
from pybullet_robot_sim import PandaSim, SphereManager
import torch
import os
CUR_PATH = os.path.dirname(os.path.realpath(__file__))
from mlp import MLPRegression
import argparse
def sample_points_from_box(obj_id, num_samples=100):
    """Sample points on the surface of the object."""
    position, orientation = p.getBasePositionAndOrientation(obj_id)
    half_extents = np.array(p.getVisualShapeData(obj_id)[0][3])  # Get half extents from visual shape data
    points = []
    for _ in range(num_samples):
        x = np.random.uniform(-half_extents[0], half_extents[0])
        y = np.random.uniform(-half_extents[1], half_extents[1])
        z = np.random.uniform(-half_extents[2], half_extents[2])
    #    Transform the point to the object's local space and orient it
        point = np.array([x, y, z])
        point = np.dot(np.array(p.getMatrixFromQuaternion(orientation)).reshape(-1, 3), point) + np.array(position)
        points.append(point)
    # Convert to numpy array
    return np.array(points)
def main_loop():
    # 处理命令行参数
    parser = argparse.ArgumentParser(description='Franka Panda CDF Example')
    parser.add_argument('--step_size', type=float, default=1, help='Step size for moving on the zero-level set')
    parser.add_argument('--display_mode', type=str, default='GUI', choices=['GUI', 'DIRECT'], help='Display mode: GUI or DIRECT')
    parser.add_argument('--robot', type=str, default='panda', help='Robot name (default: panda)')
    parser.add_argument('--model_dict', type=str, default='finger_no_base.pt', help='Path to save/load the model dictionary')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (default: cuda)')
    parser.add_argument('--data_path', type=str, default='data_finger_no_base.pt', help='Sub-directory in data/ to save the results (default: test)')
    parser.add_argument('--cdf', action='store_true', help='Use CDF for collision checking')
    args = parser.parse_args()
    N_FUNC = 8
    DOMAIN_MIN = -1.0
    DOMAIN_MAX = 1.0
    # 在zero-level-set上移动的step_size
    step_size = args.step_size
    # 设置显示模式
    display_mode = args.display_mode
    if display_mode == 'GUI':
        p.connect(p.GUI, options='--background_color_red=0.5 --background_color_green=0.5' +
                                    ' --background_color_blue=0.5 --width=1600 --height=1000')
    else:
        p.connect(p.DIRECT)
    loop_num = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    # --------- load model and cdf -----------
    CUR_DIR = os.path.dirname(os.path.realpath(__file__))
    SDF_DIR = os.path.dirname(os.path.realpath(__file__))+"/../../RDF"
    paths = {
        'urdf': os.path.join(SDF_DIR,f'descriptions/{args.robot}/*.urdf'),
        'meshes': os.path.join(SDF_DIR,f'descriptions/{args.robot}/meshes/*.stl'),
        'points': os.path.join(SDF_DIR,f'data/{args.robot}/sdf_points/'),
        'model':os.path.join(SDF_DIR, f'models/{args.robot}/BP_{N_FUNC}.pt'),
        'data': os.path.join(CUR_DIR,f'data/{args.robot}/{args.data_path}'),
        'model_dict': os.path.join(CUR_DIR,f'model_dict/{args.robot}/{args.model_dict}')
        }

    robot_layer = ParallelRobotLayer(device=args.device,paths=paths,robot=args.robot)
    bp_sdf = ParallelBPSDF(n_func=N_FUNC,
                           domain_min=DOMAIN_MIN,
                           domain_max=DOMAIN_MAX,
                           robot=robot_layer,
                           device=args.device,
                           paths=paths)
    cdf = CDF(device=args.device, paths=paths,robot=args.robot,signed_distance=True,serial_idx=0)
    sdf_model = torch.load(paths['model'])
    # --- TODO: temporarily load the cdf model like this, with input_dims=4+3=7 ---
    cdf_model = MLPRegression(input_dims=7, output_dims=1, mlp_layers=[1024, 512, 256, 128, 128],skips=[], act_fn=torch.nn.ReLU, nerf=True)
    cdf_model.load_state_dict(torch.load(paths['model_dict'])[49900])
    cdf_model.to(device)
    # --- use sdf or cdf for collision checking ---
    use_cdf = args.cdf
    # --------- pybullet setup -----------
    p.setPhysicsEngineParameter(maxNumCmdPer1ms=1000)
    p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)
    #p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=110, cameraPitch=-10, cameraTargetPosition=[0, 0, 0.5])
    # p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=90, cameraPitch=0, cameraTargetPosition=[0, 0, 0.5])
    #p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=110, cameraPitch=-25, cameraTargetPosition=[0, 0, 0.5])
    if display_mode == 'GUI':
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(lightPosition=[5, 5, 5])
        p.resetDebugVisualizerCamera(cameraDistance=0.25, cameraYaw=180, cameraPitch=0, cameraTargetPosition=[0, 0, 0.45])
    p.setAdditionalSearchPath(pd.getDataPath())
    
    # NOTE: need high frequency
    hz = 1000
    delta_t = 1.0 / hz
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(delta_t)
    p.setRealTimeSimulation(0)
    
    # --------- load robot -----------
    base_pos = [0, 0, 0.5]
    base_rot = p.getQuaternionFromEuler([0, 0, 0])
    robot = PandaSim(p, base_pos, base_rot)
    q0 = robot.get_joint_positions()
    print('q0=',q0)
    q_init = torch.tensor([q0],requires_grad=True).to(device).float()

    pandaNumDofs = robot.dof
    for joint_idx in range(pandaNumDofs):
        p.enableJointForceTorqueSensor(robot.panda, joint_idx, enableSensor=True)
    # 用idx_mask来将sdf的关节值映射到robot的关节值
    idx_mask = torch.zeros(pandaNumDofs).int()
    for joint in robot.Joint2Idx.keys():
        idx_mask[robot.Joint2Idx[joint][0]] = robot_layer.Joint2Idx[joint]
    # # # ----------- load plane -----------
    # plane_id = p.createCollisionShape(p.GEOM_PLANE)
    # p.createMultiBody(baseMass=0, baseCollisionShapeIndex=plane_id, basePosition=[0, 0, 0])
    # # ----------- load box -----------
    # box_size = np.array([0.6,0.01,0.3])
    # box_center = np.array([0.0,0.3,0.3])
    # box = p.createVisualShape(p.GEOM_BOX, halfExtents=box_size, rgbaColor=[0.8500, 0.3250, 0.0980, 1.0])
    # p.createMultiBody(baseVisualShapeIndex=box,
    #                                     basePosition=box_center)

    # ------------- load a object to grasp -----------
    obj_size = np.array([0.02,0.02,0.02])
    obj_center = np.array([-0.05, -0.05, 0.45])
    obj_orientation = [0, 0, 0, 1]  # No rotation
    obj_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=obj_size, rgbaColor=[0.8500, 0.3250, 0.0980, 1.0])
    obj_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=obj_size)
    # set baseMass=0 to make the object static
    obj = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=obj_collision, baseVisualShapeIndex=obj_visual, basePosition=obj_center)
    p.changeDynamics(obj, -1, lateralFriction=0.5, spinningFriction=0.1, rollingFriction=0.1)
    
    # --- initialize the task space  ---
    task_space = np.array([[-0.5, 0.5], # x-axis
                          [-0.5, 0.5], # y-axis
                          [ 0.0, 1.0]]) # z-axis
    base_pos = np.array([0.0, 0.0, 0.0])
    c=input('Press any key to continue')
    
    ############ main loop ############
    for _ in range(loop_num):
        for _ in range(300):
            robot.set_joint_positions(q0)
            p.stepSimulation()
        # ---- initialize the object position and orientation ----
        # obj_center = np.random.rand(3) * (task_space[:, 1] - task_space[:, 0]) + task_space[:, 0]
        # obj_center = np.array([0.0, 0.0, 0.0])
        # obj_center[2] = task_space[2, 0] + obj_size[2] / 2.0 + 0.1  # Ensure the object is above the ground
        # obj_orientation_Euler = np.random.rand(3) * np.pi * 2.0 - np.pi
        # obj_orientation = p.getQuaternionFromEuler(obj_orientation_Euler)
        p.resetBasePositionAndOrientation(obj, obj_center, obj_orientation)
        p.resetBaseVelocity(obj, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0])
        # while(True):
        #         keys = p.getKeyboardEvents()
        #         if p.B3G_SPACE in keys and keys[p.B3G_SPACE] & p.KEY_WAS_TRIGGERED:
        #             print('space key pressed')
        #             break
        q = q_init
        while(True):
            # print(f'joint positions:{robot.get_joint_positions()}')
            torques = p.getJointStates(robot.panda, range(pandaNumDofs))
            # print(f'torques: {torques}')
            print('')
            position, orientation = p.getBasePositionAndOrientation(obj)
            # --- debug: for clarity, sample one point only ---
            x = sample_points_from_box(obj, num_samples=1)
            x = torch.from_numpy(np.array(x)).to(device).float()
            # 获得robot base link的位置
            robot_base_pos, robot_base_orn = p.getBasePositionAndOrientation(robot.panda)
            matrix = torch.tensor(p.getMatrixFromQuaternion(robot_base_orn)).reshape(3,3).to(device)
            x_in_robot_frame = (torch.matmul(matrix.T, (x - torch.tensor(robot_base_pos).to(device).float()).T)).T
            q = torch.tensor([robot.get_joint_positions()],requires_grad=True).to(device).float()
            print(f'q:', q)
            # check if the object is out of the task space
            # if (position[0] < task_space[0, 0] or position[0] > task_space[0, 1] or
            #     position[1] < task_space[1, 0] or position[1] > task_space[1, 1] or
            #     position[2] < task_space[2, 0] or position[2] > task_space[2, 1]):
            #     print('object out of task space')
            #     break
            in_contact = bool(p.getContactPoints(bodyA=robot.panda, bodyB=obj))
            if not in_contact:
                print('not in contact')
                q_next = q.clone()
                # print(f'q_init: {q_init}')
                for i,serial in enumerate(robot_layer.serials):
                    if i != 0:
                        continue
                    pose = torch.eye(4).unsqueeze(0).to(device).float()
                    pose[:, :3, 3] = torch.tensor(base_pos).to(device).float()
                    theta = torch.stack([q[:,robot_layer.Joint2Idx[joint]] for joint in serial.Joint2Idx.keys()],dim=-1)
                    if use_cdf:
                        cdf_min, cdf_grad = cdf.inference_d_wrt_q(x_in_robot_frame, theta,cdf_model)
                        q_proj = cdf.projection(theta, cdf_min, cdf_grad)
                        print('cdf_min:', cdf_min)
                        print('cdf_grad:', cdf_grad)
                        print('theta before:', theta)
                        theta = q_proj
                        print('theta after:', theta)
                    else:
                        sdf,grad = bp_sdf.get_serial_sdf_with_joints_grad_batch(x_in_robot_frame,pose,theta,sdf_model,used_links = None, serial_idx = i)
                        sdf_min, min_idx = torch.min(sdf, dim=1)
                        grad = grad[:, min_idx, :].squeeze(1)
                        theta = theta - grad * step_size
                        print('sdf_min:', sdf_min)
                        print('grad:', grad)
                        print('theta:', theta)
                        print('q_next before:', q_next)
                    for j,joint in enumerate(serial.Joint2Idx.keys()):
                        q_next[:,robot_layer.Joint2Idx[joint]] = theta[:,j]
                    print('serial_idx:', i)
                print(f'q_next: {q_next}')
                robot.set_joint_positions(q_next[0].data.cpu().numpy())
                c=input('Press any key to continue')
            else:
                
                print(f'in contact')
                c=input('press any key to continue')
                q = q_init
                d,grad = cdf.inference_d_wrt_q(x_in_robot_frame,q,sdf_model)
                # 找到与grad正交的方向
                n_q = len(grad)

                q_normal = torch.ones(n_q).to(device) - (torch.sum(grad, dim=-1) / (torch.norm(grad, dim=-1)*torch.norm(grad, dim=-1))) * grad
                q_normal = q_normal / torch.norm(q_normal, dim=-1, keepdim=True)
                q_next = q - step_size * q_normal
                q_init = q_next
                # while(True):
                #     keys = p.getKeyboardEvents()
                #     if p.B3G_SPACE in keys and keys[p.B3G_SPACE] & p.KEY_WAS_TRIGGERED:
                #         print('space key pressed')
                #         break
                time.sleep(0.1)
                print(f'd: {d.cpu().detach().numpy()[0]}')
                # print(f'q_next: {q_next}, q_normal: {q_normal}')
                robot.set_joint_positions(q_next[0].data.cpu().numpy())
            p.stepSimulation()

    p.disconnect()


if __name__ == '__main__':
    main_loop()