# -----------------------------------------------------------------------------
# SPDX-License-Identifier: MIT
# This file is part of the CDF project.
# Copyright (c) 2024 Idiap Research Institute <contact@idiap.ch>
# Contributor: Yimming Li <yiming.li@idiap.ch>
# -----------------------------------------------------------------------------


import copy
import math
import numpy as np
import matplotlib.pyplot as plt
import torch

class Robot2D:
    def __init__(self,
                 num_links=3,
                 init_states=torch.tensor([[2*np.pi/3, -np.pi/3, -np.pi/3]]),
                 link_lengths=torch.tensor([[2, 2, 1]]).float(),
                 link_parent_map=None,
                 base_frame='default',
                 device='cpu'):
        self.device = device
        self.num_links = num_links
        self.init_states = init_states.to(self.device)
        self.link_lengths = link_lengths.to(self.device)
        self.B = init_states.size(0)
        self.link_parent_map = link_parent_map if link_parent_map else {i: i - 1 for i in range(1, num_links)}
        if base_frame == 'default':
            self.base_frame = torch.zeros((self.B, 2)).to(self.device)
        else:
            self.base_frame = base_frame.to(self.device)

    def forward_kinematics_all_links(self, x):
        self.B = x.size(0)
        L = torch.eye(self.num_links).unsqueeze(0).expand(self.B, -1, -1).to(self.device)
        
        for i in range(self.num_links):
            parent_idx = i
            while parent_idx != 0:
                parent_idx = self.link_parent_map[parent_idx + 1] - 1
                L[:, i, parent_idx] = 1
        x = x.unsqueeze(2)
        diag_length = torch.diag(self.link_lengths.squeeze()).unsqueeze(0)
        f = torch.stack([
            torch.matmul(L, torch.matmul(diag_length, torch.cos(torch.matmul(L, x)))),
            torch.matmul(L, torch.matmul(diag_length, torch.sin(torch.matmul(L, x))))
        ], dim=0).transpose(0, 1).squeeze()
        if self.B == 1:
            f = f.unsqueeze(0)
        f = torch.cat([torch.zeros([self.B, 2, 1]).to(self.device), f], dim=-1)
        return f + torch.zeros((self.B, 2)).to(self.device).unsqueeze(2).expand(-1, -1, self.num_links + 1)

    # Forward kinematics for end-effector (in robot coordinate system)
    def forward_kinematics_eef(self,x):
        link_positions = self.forward_kinematics_all_links(x)
        return link_positions[:, :, -1]

    def surface_points_sampler(self,x,n=100):
        # 在机器人每个关节之间采样n个点
        # 输入x，形状为(B, num_joints),表示机器人每个关节的角度
        # 返回一个张量kpts，形状为(B, 2, n * (num_joints - 1))，其中B是批次大小，2表示二维坐标，n是每个关节之间采样的点数
        # 每个点的坐标是相对于机器人基座的
        self.B = x.size(0)
        f_rob = self.forward_kinematics_all_links(x) # B,2,N
        N = f_rob.size(2)
        t = torch.linspace(0,1,n).unsqueeze(0).expand(self.B,-1).to(self.device)
        kpts_list = []
        for i in range (N-1):
            parent_idx = self.link_parent_map[i+1]
            parent_pos = f_rob[:,:,parent_idx]
            self_pos = f_rob[:,:,i+1]
            offset = torch.stack([
                self_pos[:,0]-parent_pos[:,0],
                self_pos[:,1]-parent_pos[:,1]
            ], dim=1)
            kpts = torch.einsum('ij,ik->ijk',offset,t) + parent_pos.unsqueeze(-1).expand(-1,-1,n)
            kpts_list.append(kpts)
        kpts = torch.cat(kpts_list,dim=-1).transpose(1,2)
        return kpts
    
    def distance(self,x,p):
        B = x.size(0)
        kpts = self.surface_points_sampler(x,n=200)
        Ns = kpts.size(1)
        p = p.unsqueeze(0).unsqueeze(2).expand(B,-1,Ns,-1)
        kpts = kpts.unsqueeze(1).expand(-1,p.size(1),-1,-1)
        dist = torch.norm(kpts-p,dim=-1).min(dim=-1)[0]
        return dist

    def plot_trajectory(self, ax, joint_trajectory, n=100):
        """
        Plot the trajectory of the robot given a joint angle trajectory.

        Args:
            joint_trajectory: Tensor of shape (T, num_links), where T is the number of time steps.
            n: Number of points to sample along each link for visualization.
        """
        T = joint_trajectory.size(0)
        all_kpts = []

        for t in range(T):
            x = joint_trajectory[t].unsqueeze(0)  # Shape (1, num_links)
            kpts = self.surface_points_sampler(x, n=n).squeeze(0).cpu().numpy()
            all_kpts.append(kpts)

        all_kpts = np.concatenate(all_kpts, axis=0)  # Combine all time steps

        # Plot the trajectory
        ax.scatter(all_kpts[:, 0], all_kpts[:, 1], s=1, c='blue', label='Trajectory')
        ax.axis("equal")
        ax.set_title("Robot Trajectory")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

    def interact(self):
        """
        Create an interactive plot where users can adjust joint angles using sliders
        and see the robot's shape update in real-time.
        """
        from matplotlib.widgets import Slider

        # Initialize the figure and axis
        fig, ax = plt.subplots()
        plt.subplots_adjust(left=0.25, bottom=0.25)
        ax.set_aspect('equal', 'box')
        ax.set_xlim(-sum(self.link_lengths[0].cpu().numpy()) - 1, sum(self.link_lengths[0].cpu().numpy()) + 1)
        ax.set_ylim(-sum(self.link_lengths[0].cpu().numpy()) - 1, sum(self.link_lengths[0].cpu().numpy()) + 1)
        plt.title("Interactive Robot Shape")

        # Initial joint angles
        joint_angles = self.init_states[0].clone()

        # Plot the initial robot shape
        link_positions = self.forward_kinematics_all_links(joint_angles.unsqueeze(0)).squeeze(0).cpu().numpy()
        lines = []
        for i in range(link_positions.shape[1] - 1):
            parent_idx = self.link_parent_map[i + 1]
            line, = ax.plot(
                [link_positions[0, parent_idx], link_positions[0, i + 1]],
                [link_positions[1, parent_idx], link_positions[1, i + 1]],
                marker="o",
                color="blue"
            )
            lines.append(line)

        # Create sliders for each joint angle
        sliders = []
        ax_sliders = []
        for i in range(self.num_links):
            ax_slider = plt.axes([0.25, 0.1 - i * 0.05, 0.65, 0.03])
            slider = Slider(ax_slider, f'Joint {i + 1}', -np.pi, np.pi, valinit=joint_angles[i].item())
            sliders.append(slider)
            ax_sliders.append(ax_slider)

        # Update function for sliders
        def update(val):
            for i, slider in enumerate(sliders):
                joint_angles[i] = slider.val
            link_positions = self.forward_kinematics_all_links(joint_angles.unsqueeze(0)).squeeze(0).cpu().numpy()
            for i, line in enumerate(lines):
                parent_idx = self.link_parent_map[i + 1]
                line.set_xdata([link_positions[0, parent_idx], link_positions[0, i + 1]])
                line.set_ydata([link_positions[1, parent_idx], link_positions[1, i + 1]])
            fig.canvas.draw_idle()

        # Attach the update function to sliders
        for slider in sliders:
            slider.on_changed(update)

        plt.show()


if __name__ == "__main__":
    # Define the parent-child relationship for the links
    link_parent_map = {
        1: 0,  # Link 1 connects to the base
        2: 1,  # Link 2 connects to Link 1
        3: 1   # Link 3 connects to Link 1
    }

    # Initial joint angles for the robot
    x = torch.tensor([[0.0, -np.pi / 4, np.pi / 4]])

    # Create the Robot2D instance
    rbt = Robot2D(
        num_links=3,
        init_states=x,
        link_lengths=torch.tensor([[2, 2, 2]]).float(),
        link_parent_map=link_parent_map
    )

    # Forward kinematics to get all link positions
    link_positions = rbt.forward_kinematics_all_links(x).numpy()
    kpts = rbt.surface_points_sampler(x,n=50).numpy().squeeze(0)
    # Plot the robot structure
    # plt.figure()
    # plt.scatter(link_positions[0, 0, :], link_positions[0, 1, :], c='red')
    # plt.scatter(kpts[:,0], kpts[:,1], c='blue', s=1)
    # plt.axis("equal")
    # plt.title("Robot2D with Link-Based Structure")
    # plt.show()
    # trajectory = torch.tensor([
    #     [0.0, -np.pi / 4, np.pi / 4, -np.pi / 6],
    #     [np.pi / 12, -np.pi / 6, np.pi / 3, -np.pi / 12],
    #     [np.pi / 6, -np.pi / 8, np.pi / 6, 0.0],
    #     [np.pi / 4, 0.0, np.pi / 8, np.pi / 6]
    # ])
    # rbt.plot_trajectory(trajectory, n=50)
    rbt.interact()

