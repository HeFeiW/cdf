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
    loop_num = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    # --------- load model and cdf -----------
    cdf = CDF(device)
    model = MLPRegression(input_dims=10, output_dims=1, mlp_layers=[1024, 512, 256, 128, 128],skips=[], act_fn=torch.nn.ReLU, nerf=True)
    model.load_state_dict(torch.load(os.path.join(CUR_PATH,'model_dict.pt'))[49900])
    model.to(device)
    # --------- pybullet setup -----------
    p.connect(p.GUI, options='--background_color_red=0.5 --background_color_green=0.5' +
                                ' --background_color_blue=0.5 --width=1600 --height=1000')
    

    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.configureDebugVisualizer(lightPosition=[5, 5, 5])
    p.setPhysicsEngineParameter(maxNumCmdPer1ms=1000)
    #p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=110, cameraPitch=-10, cameraTargetPosition=[0, 0, 0.5])
    # p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=90, cameraPitch=0, cameraTargetPosition=[0, 0, 0.5])
    #p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=110, cameraPitch=-25, cameraTargetPosition=[0, 0, 0.5])
    p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=145, cameraPitch=0, cameraTargetPosition=[0, 0, 0.6])
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

    # ------------- load a object to push -----------
    obj_size = np.array([0.05, 0.05, 0.05]) # 正好是一个grid的大小
    obj_center = np.array([0.0, 0.0, 0.5])
    obj = p.createVisualShape(p.GEOM_BOX, halfExtents=obj_size/2, rgbaColor=[0.8500, 0.3250, 0.0980, 1.0])
    p.createMultiBody(baseVisualShapeIndex=obj,
                                        basePosition=obj_center)
    
    # --- initialize the task space  ---
    # #TODO 初始的位置和目标位置先定死，保证是一条简单的直线，之后再改成随机的
    task_space = np.array([-0.5, 0.5], # x-axis
                          [-0.5, 0.5], # y-axis
                          [ 0.0, 1.0]) # z-axis
    base_pos = np.array([0.0, 0.0, 0.0])
     # --- initialize the start and goal position of the object(both 2D on the ground) ---
    # obj_start_pos = np.random.rand(2) * (task_space[:2, 1] - task_space[:2, 0]) + task_space[:2, 0]
    obj_start_pos = np.array([-0.2, 0.2])
    obj_start_pos = np.append(obj_start_pos, obj_size[2] / 2.0)  # z-axis is half of the object height
    # obj_goal_pos = np.random.rand(2) * (task_space[:2, 1] - task_space[:2, 0]) + task_space[:2, 0]
    obj_goal_pos = np.array([0.2, -0.2])
    obj_goal_pos = np.append(obj_goal_pos, obj_size[2] / 2.0)  # z-axis is half of the object height
    # obj_start_orientation_Euler = (np.random.rand(1) * np.pi * 2.0 - np.pi,0.0, 0.0)  # only rotate around z-axis
    obj_start_orientation_Euler = (0.0, 0.0, 0.0)
    obj_start_orientation = p.getQuaternionFromEuler(obj_start_orientation_Euler)
    # obj_goal_orientation_Euler = (np.random.rand(1) * np.pi * 2.0 - np.pi,0.0, 0.0)
    obj_goal_orientation_Euler = (0.0, 0.0, 0.0)
    obj_goal_orientation = p.getQuaternionFromEuler(obj_goal_orientation_Euler)
    
    # --- 根据初始位置和目标位置，计算直线轨迹 ---
    key_frame_num = 40
    obj_traj_pos = np.linspace(obj_start_pos, obj_goal_pos, key_frame_num)
    obj_traj_orientation = np.linspace(obj_start_orientation_Euler, obj_goal_orientation_Euler, key_frame_num)
    
    for _ in loop_num:
        for _ in range(300):
            robot.set_joint_positions(q0)
            p.stepSimulation()
        # ---- initialize the object position and orientation ----
        p.resetBasePositionAndOrientation(obj, obj_start_pos, obj_start_orientation)
        p.resetBaseVelocity(obj, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0])
        # ---- initialize a goal area for the object ----
        goal_area_size = np.array([0.2, 0.2, 0.2])
        goal_area_center = np.random.rand(2) * (task_space[:2, 1] - task_space[2:, 0]) + task_space[:2, 0]
        
        reached_goal,proj = False, False
        for frame in range(key_frame_num):
            print(robot.get_joint_positions())
            position, orientation = p.getBasePositionAndOrientation(obj)
            frame_gt_pos = obj_traj_pos[frame]
            frame_gt_orientation = p.getQuaternionFromEuler(obj_traj_orientation[frame])
            print(f'frame: {frame}, object position: {position}, orientation: {orientation}')
            print(f'frame: {frame}, gt position: {frame_gt_pos}, orientation: {frame_gt_orientation}')
            # check if the object is out of the task space
            if (position[0] < task_space[0, 0] or position[0] > task_space[0, 1] or
                position[1] < task_space[1, 0] or position[1] > task_space[1, 1] or
                position[2] < task_space[2, 0] or position[2] > task_space[2, 1]):
                print('object out of task space')
                break
            # check if the object is in the goal area
            if (position[0] > goal_area_center[0]-goal_area_size[0]/2) & (position[0] < goal_area_center[0] + goal_area_size[0]/2) & \
                (position[1] > goal_area_center[1]-goal_area_size[1]/2) & (position[1] < goal_area_center[1] + goal_area_size[1]/2):
                reached_goal = True
                break
            else:
                print('not reached goal')
                q = q_init
                x = torch.from_numpy(np.array([frame_gt_pos])).to(device).float()
                d,grad = cdf.inference_d_wrt_q(x,q,model)
                print(f'q_init: {q_init}')
                print(f'd: {d}, grad: {grad}')
                q_proj = cdf.projection(q,d,grad)
                q_init = q_proj
                robot.set_joint_positions(q_proj[0].data.cpu().numpy())
                proj = True

            p.stepSimulation()
            time.sleep(delta_t*2.0)

    p.disconnect()


if __name__ == '__main__':

    main_loop()