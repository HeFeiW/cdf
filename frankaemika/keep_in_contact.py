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
import time
from pybullet_panda_sim import PandaSim, SphereManager
import torch
import os
CUR_PATH = os.path.dirname(os.path.realpath(__file__))
from mlp import MLPRegression
from nn_cdf import CDF

def main_loop():
    # 在zero-level-set上移动的step_size
    step_size = 0.01
    # 设置显示模式
    display_mode = 'GUI'  # 可选 'GUI' 或 'DIRECT'
    if display_mode == 'GUI':
        p.connect(p.GUI, options='--background_color_red=0.5 --background_color_green=0.5' +
                                    ' --background_color_blue=0.5 --width=1600 --height=1000')
    else:
        p.connect(p.DIRECT)
    loop_num = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    # --------- load model and cdf -----------
    cdf = CDF(device)
    model = MLPRegression(input_dims=10, output_dims=1, mlp_layers=[1024, 512, 256, 128, 128],skips=[], act_fn=torch.nn.ReLU, nerf=True)
    model.load_state_dict(torch.load(os.path.join(CUR_PATH,'model_dict.pt'))[49900])
    model.to(device)
    # --------- pybullet setup -----------
    # p.connect(p.GUI, options='--background_color_red=0.5 --background_color_green=0.5' +
                                # ' --background_color_blue=0.5 --width=1600 --height=1000')
    


    p.setPhysicsEngineParameter(maxNumCmdPer1ms=1000)
    #p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=110, cameraPitch=-10, cameraTargetPosition=[0, 0, 0.5])
    # p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=90, cameraPitch=0, cameraTargetPosition=[0, 0, 0.5])
    #p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=110, cameraPitch=-25, cameraTargetPosition=[0, 0, 0.5])
    if display_mode == 'GUI':
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(lightPosition=[5, 5, 5])
        p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=90, cameraPitch=0, cameraTargetPosition=[0, 0, 0.6])
    p.setAdditionalSearchPath(pd.getDataPath())
        
    ## spawn franka robot
    base_pos = [0, 0, 0]
    base_rot = p.getQuaternionFromEuler([0, 0, 0])
    robot = PandaSim(p, base_pos, base_rot)
    q0 = robot.get_joint_positions()
    q_init = torch.tensor([q0],requires_grad=True).to(device).float()
    # NOTE: need high frequency
    hz = 1000
    delta_t = 1.0 / hz
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(delta_t)
    p.setRealTimeSimulation(0)

    # # ----------- load box -----------
    # box_size = np.array([0.6,0.01,0.3])
    # box_center = np.array([0.0,0.3,0.3])
    # box = p.createVisualShape(p.GEOM_BOX, halfExtents=box_size, rgbaColor=[0.8500, 0.3250, 0.0980, 1.0])
    # p.createMultiBody(baseVisualShapeIndex=box,
    #                                     basePosition=box_center)

    # ------------- load a object to push -----------
    obj_size = np.array([0.2,0.2,0.2])
    obj_center = np.array([0.5, 0.0, 0.0])
    obj = p.createVisualShape(p.GEOM_BOX, halfExtents=obj_size, rgbaColor=[0.8500, 0.3250, 0.0980, 0.1])
    p.createMultiBody(baseVisualShapeIndex=obj,
                                        basePosition=obj_center)
    
    # --- initialize the task space  ---
    task_space = np.array([[-0.5, 0.5], # x-axis
                          [-0.5, 0.5], # y-axis
                          [ 0.0, 1.0]]) # z-axis
    base_pos = np.array([0.0, 0.0, 0.0])
    
    for _ in range(loop_num):
        for _ in range(300):
            robot.set_joint_positions(q0)
            p.stepSimulation()
        # ---- initialize the object position and orientation ----
        obj_center = np.random.rand(3) * (task_space[:, 1] - task_space[:, 0]) + task_space[:, 0]
        obj_center[2] = task_space[2, 0] + obj_size[2] / 2.0
        obj_orientation_Euler = np.random.rand(3) * np.pi * 2.0 - np.pi
        obj_orientation = p.getQuaternionFromEuler(obj_orientation_Euler)
        p.resetBasePositionAndOrientation(obj, obj_center, obj_orientation)
        p.resetBaseVelocity(obj, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0])
        # ---- initialize a goal area for the object ----
        goal_area_size = np.array([0.2, 0.2, 0.2])
        goal_area_center = np.random.rand(2) * (task_space[:2, 1] - task_space[2:, 0]) + task_space[:2, 0]
        
        reached_goal,proj = False, False
        while(True):
            print(f'joint positions:{robot.get_joint_positions()}')
            position, orientation = p.getBasePositionAndOrientation(obj)
            # check if the object is out of the task space
            if (position[0] < task_space[0, 0] or position[0] > task_space[0, 1] or
                position[1] < task_space[1, 0] or position[1] > task_space[1, 1] or
                position[2] < task_space[2, 0] or position[2] > task_space[2, 1]):
                print('object out of task space')
                break
            in_contact = p.getContactPoints(bodyA=robot.panda, bodyB=obj)
            print(f'contact points: {in_contact}')
            if len(in_contact) == 0:
                print('not in contact')
                q = q_init
                x = torch.from_numpy(np.array([position])).to(device).float()
                d,grad = cdf.inference_d_wrt_q(x,q,model)
                q_next = cdf.projection(q,d,grad)
                q_init = q_next
                print(f'd: {d}, grad: {grad}, q_next: {q_next}')
                robot.set_joint_positions(q_next[0].data.cpu().numpy())
            else:
                print('not reached goal')
                q = q_init
                x = torch.from_numpy(np.array([position])).to(device).float()
                d,grad = cdf.inference_d_wrt_q(x,q,model)
                # 找到与grad正交的方向
                n_q = len(grad)

                q_normal = torch.ones(n_q).to(device) - (torch.sum(grad, dim=-1) / (torch.norm(grad, dim=-1)*torch.norm(grad, dim=-1))) * grad
                q_normal = q_normal / torch.norm(q_normal, dim=-1, keepdim=True)
                q_next = q - step_size * q_normal.unsqueeze(0)
                q_init = q_next
                print(f'd: {d}, grad: {grad}, q_next: {q_next}')
                robot.set_joint_positions(q_next[0].data.cpu().numpy())
            p.stepSimulation()
            time.sleep(delta_t*2.0)

    p.disconnect()


if __name__ == '__main__':

    main_loop()