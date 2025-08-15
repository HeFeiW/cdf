# -----------------------------------------------------------------------------
# SPDX-License-Identifier: MIT
# This file is part of the CDF project.
# Copyright (c) 2024 Idiap Research Institute <contact@idiap.ch>
# Contributor: Yimming Li <yiming.li@idiap.ch>
# -----------------------------------------------------------------------------


import time
import numpy as np
from math import pi
import os
CUR_PATH = os.path.dirname(os.path.realpath(__file__))
import argparse
dexhand_dof = 7#TODO

# restpose


class dexhandSim():
    def __init__(self, bullet_client, base_pos, base_rot):
        self.bullet_client = bullet_client
        self.bullet_client.setAdditionalSearchPath('content/urdfs')
        self.base_pos = np.array(base_pos)
        self.base_rot = np.array(base_rot)
        # print("offset=",offset)
        flags = self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
        table_rot = self.bullet_client.getQuaternionFromEuler([pi / 2, 0, pi])
        # self.rp = rp
        # self.bullet_client.loadURDF('LAB/lab.urdf', np.array([1.01, -0.28, 0.45]), table_rot, flags=flags)
        #self.bullet_client.loadURDF('lab_table/table.urdf', np.array([-0.15, 0.02, -0.1]), table_rot, flags=flags)
        # self.bullet_client.loadURDF('plane.urdf', np.array([0, 0, 0]), np.array([0, 0, 0, 1]), flags=flags)
        path = os.path.join(CUR_PATH,"dexhand_description/urdf/dexhand-right.urdf")
        print("Loading dexhand from ", path)
        self.dexhand = self.bullet_client.loadURDF(path, self.base_pos,
                                                 self.base_rot, useFixedBase=True, flags=flags)
        print("dexhand=", self.dexhand)
        # self.set_joint_positions(rp)
        self.dexhand_dof = self.bullet_client.getNumJoints(self.dexhand)
        print("dexhand_dof=", self.dexhand_dof)
        self.rp = np.zeros(self.dexhand_dof)
        self.reset()
        self.t = 0.
    def reset(self):
        index = 0
        for j in range(self.bullet_client.getNumJoints(self.dexhand)):
            self.bullet_client.changeDynamics(self.dexhand, j, linearDamping=0, angularDamping=0)
            info = self.bullet_client.getJointInfo(self.dexhand, j)
            jointName = info[1]
            jointType = info[2]
            if (jointType == self.bullet_client.JOINT_PRISMATIC):
                self.bullet_client.resetJointState(self.dexhand, j, self.rp[index])
                index = index + 1
            if (jointType == self.bullet_client.JOINT_REVOLUTE):
                self.bullet_client.resetJointState(self.dexhand, j, self.rp[index])
                index = index + 1

    def set_joint_positions(self, joint_positions):
        for i in range(self.dexhand_dof):
            self.bullet_client.setJointMotorControl2(self.dexhand, i, self.bullet_client.POSITION_CONTROL,
                                                     joint_positions[i], force=240.)
        # self.set_finger_positions(0.04)

    def set_finger_positions(self, gripper_opening):
        self.bullet_client.setJointMotorControl2(self.dexhand, 9, self.bullet_client.POSITION_CONTROL,
                                                 gripper_opening/2, force=5 * 240.)
        self.bullet_client.setJointMotorControl2(self.dexhand, 10, self.bullet_client.POSITION_CONTROL,
                                                 -gripper_opening/2, force=5 * 240.)


    def get_joint_positions(self):
        joint_state = []
        for i in range(self.dexhand_dof):
            joint_state.append(self.bullet_client.getJointState(self.dexhand, i)[0])
        return joint_state

class SphereManager:
    def __init__(self, pybullet_client):
        self.pb = pybullet_client
        self.spheres = []
        self.color = [.7, .1, .1, 1]
        self.color = [.63, .07, .185, 1]
        # self.color = [0.8500, 0.3250, 0.0980, 1]
    def create_sphere(self, position, radius, color):
        sphere = self.pb.createVisualShape(self.pb.GEOM_SPHERE,
                                           radius=radius,
                                           rgbaColor=color, specularColor=[0, 0, 0, 1])
        sphere = self.pb.createCollisionShape(self.pb.GEOM_SPHERE,
                                                radius=radius)
        
        sphere = self.pb.createMultiBody(baseVisualShapeIndex=sphere,
                                         basePosition=position)
        self.spheres.append(sphere)

    def initialize_spheres(self, obstacle_array):
        for obstacle in obstacle_array:
            self.create_sphere(obstacle[0:3], obstacle[3], self.color)

    def delete_spheres(self):
        for sphere in self.spheres:
            self.pb.removeBody(sphere)
        self.spheres = []

    def update_spheres(self, obstacle_array):
        if (obstacle_array is not None) and (len(self.spheres) == len(obstacle_array)):
            for i, sphere in enumerate(self.spheres):
                self.pb.resetBasePositionAndOrientation(sphere,
                                                        obstacle_array[i, 0:3],
                                                        [1, 0, 0, 0])
        else:
            print("Number of spheres and obstacles do not match")
            self.delete_spheres()
            self.initialize_spheres(obstacle_array)
def main():
    parser = argparse.ArgumentParser(description='DexHand simulation')
    parser.add_argument('--display',type=str,default='GUI',choices=['GUI','DIRECT'],help='Display mode: GUI or DIRECT')
    args = parser.parse_args()
    display = args.display
    import pybullet as p
    import pybullet_data
    if display == 'GUI':
        physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
    else:
        physicsClient = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
    p.setGravity(0, 0, -10)
    # planeId = p.loadURDF("plane.urdf")
    hand = dexhandSim(p,[0,0,0],[0,0,0,1])
    # hand.set_joint_positions([0,0,-1.57,-1.57,-0.78,1.57,-0.78])
    while True:
        p.stepSimulation()
        p.stepSimulation()
        p.stepSimulation()
        p.stepSimulation()
        time.sleep(1./240.)
        # Get the current joint positions
        joint_positions = hand.get_joint_positions()
        print("Current joint positions:", joint_positions)
        # 用交互式方式设置关节位置
        index = input(f"Enter joint index to modify (0-{hand.dexhand_dof}) or 'q' to quit: ")
        new_positions = input("Enter new joint positions (comma-separated): ")
        if index.lower() == 'q':
            break
        joint_positions[int(index)] = float(new_positions)
        hand.set_joint_positions(joint_positions)
        
    p.disconnect()

if __name__ == "__main__":
    main()
