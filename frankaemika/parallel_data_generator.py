# generate parallel robot data for all serial robots in the parallel robot
# for each serial robot, generate data within the workspace
# for each point in the workspace, find multiple q that sdf(x,q)=0
# save data as a dictionary: {point_idx: {'x':x, 'q':[...], 'idx':[...]}}
# point_idx: index of the point in the workspace grid
# x: point coordinates
# q: list of joint configurations that satisfy sdf(x,q)=0
# idx: list of link indices that are closest to x for each q

import torch
import os
CUR_DIR = os.path.dirname(os.path.realpath(__file__))
import numpy as np
import sys
sys.path.append(os.path.join(CUR_DIR,'../../RDF'))
from panda_layers.robot_layer import RobotLayer
from panda_layers.parallel_robot_layer import ParallelRobotLayer
from panda_layers.serial_robot_layer import SerialRobotLayer
from parallel_bf_sdf import ParallelBPSDF
from torchmin import minimize
import argparse
import time
import math
import copy
import utils
PI = math.pi

class DataGenerator():
    def __init__(self, device, robot, paths, serial_idx=None,with_base=False):
        # panda model
        self.robot = robot
        self.paths = paths
        self.bp_sdf_model_path = paths['model']
        self.bp_sdf = ParallelBPSDF(8, -1.0, 1.0, self.robot, self.bp_sdf_model_path, device)
        self.model = torch.load(self.bp_sdf_model_path)
        # device
        self.device = device
        self.serial_idx = serial_idx
        # data generation
        # workspace大小：1m x 1m x 1m
        # 20 x 20 x 20 = 8000个点
        # 相当于每个grid是 5cm x 5cm x 5cm
        self.workspace = self.robot.space_limits.cpu().numpy()
        self.n_disrete = 20  # total number of x: n_discrete**3
        self.batchsize = 20000  # batch size of q
        self.epsilon = 1e-3  # distance threshold to filter data
        self.used_links = self.robot.serials[self.serial_idx].all_links.copy()
        self.with_base = with_base
        print('used_links:', self.used_links)

        # 保留 palm_lower_left，不再删除
        if 'palm_lower_left' in self.used_links:
            print('palm_lower_left is included in used_links')

        # # 添加 palm_base 的自由度
        # self.base_dof = 6  # 6DoF for palm_base
        # self.total_dof = self.robot.serials[self.serial_idx].dof + self.base_dof
        # print(f'Total DoF including palm_base: {self.total_dof}')

    def compute_sdf(self, x, q, pose=None, return_index=False):
        # x : (Nx,3)
        # q : (Nq,dof)
        # return_index : if True, return the index of link that is closest to x
        # return d : (Nq)
        # return idx : (Nq) optional
        
        if not return_index:
            d, _ = self.bp_sdf.get_serial_sdf_batch(x, pose, q, self.model, use_derivative=False, serial_idx=self.serial_idx,
                                                     used_links=self.used_links)
            d = d.min(dim=1)[0]
            return d
        else:
            d, _, idx = self.bp_sdf.get_serial_sdf_batch(x, pose, q, self.model, use_derivative=False, serial_idx=self.serial_idx,
                                                         return_index=True, used_links=self.used_links)
            # d: (Nq,Nx)
            # idx: (Nq,Nx)
            d, pts_idx = d.min(dim=1)
            idx = idx[torch.arange(len(idx)), pts_idx]
            return d, idx
    def compute_sdf_with_pose(self, x, q, return_index=False):
        # x : (Nx,3)
        # q : (Nq,dof+6)
        # return_index : if True, return the index of link that is closest to x
        # return d : (Nq)
        # return idx : (Nq) optional
        
        q_pose = q[:, -6:]  # (Nq,6) 前三位是平移，后三位是旋转
        q_joint = q[:, :-6]  # (Nq,dof)
        pose = utils.q_to_poseMatrix(self, q_pose).to(self.device)  # (Nq,4,4)
        if not return_index:
            d, _ = self.bp_sdf.get_serial_sdf_batch(x, pose, q_joint, self.model, use_derivative=False, serial_idx=self.serial_idx,
                                                     used_links=self.used_links)
            d = d.min(dim=1)[0]
            return d
        else:
            d, _, idx = self.bp_sdf.get_serial_sdf_batch(x, pose, q_joint, self.model, use_derivative=False, serial_idx=self.serial_idx,
                                                         return_index=True, used_links=self.used_links)
            # d: (Nq,Nx)
            # idx: (Nq,Nx)
            d, pts_idx = d.min(dim=1)
            idx = idx[torch.arange(len(idx)), pts_idx]
            return d, idx

    def given_x_find_q(self, x, q=None, batchsize=None, return_mask=False, epsilon=1e-3, serial_idx=None):
        # x : (N,3)
        # scale x to workspace
        if not batchsize:
            batchsize = self.batchsize
        if serial_idx is None:
            serial_idx = self.serial_idx
        serial = self.robot.serials[serial_idx]
        q_min = serial.theta_min_soft
        q_max = serial.theta_max_soft
        def cost_function(q):
            #  find q that d(x,q) = 0
            # q : B,2
            # x : N,3

            d = self.compute_sdf(x, q)
            cost = torch.sum(d ** 2)
            return cost

        # t0 = time.time()
        # optimizer for data generation
        if q is None:
            q = torch.rand(batchsize, serial.dof).to(self.device) * (q_max - q_min) + q_min
        q0 = copy.deepcopy(q)
        res = minimize(
            cost_function,
            q,
            method='l-bfgs',
            options=dict(line_search='strong-wolfe'),
            max_iter=50,
            disp=0
        )

        d, idx = self.compute_sdf(x, res.x, return_index=True)
        d, idx = d.squeeze(), idx.squeeze()
        mask = torch.abs(d) < epsilon
        # q_valid,d,idx = res.x[mask],d[mask],idx[mask]
        boundary_mask = ((res.x > q_min) & (res.x < q_max)).all(dim=1)
        final_mask = mask & boundary_mask
        final_q, idx = res.x[final_mask], idx[final_mask]
        # q0 = q0[mask][boundary_mask]

        # print('number of q_valid: \t{} \t time cost:{}'.format(len(final_q), time.time() - t0))

        if return_mask:
            return final_mask, final_q, idx
        else:
            return final_q, idx
    
    def given_x_find_q_with_pose(self, x, q=None, batchsize=None, return_mask=False, epsilon=1e-3, serial_idx=None):
        # x : (N,3)
        # pose : (B,4,4)
        # scale x to workspace
        if not batchsize:
            batchsize = self.batchsize
        if serial_idx is None:
            serial_idx = self.serial_idx
        serial = self.robot.serials[serial_idx]
        q_min = serial.theta_min_soft
        q_max = serial.theta_max_soft
        base_dof = 6  # 6DoF for palm_base
        base_min = serial.theta_min_base
        base_max = serial.theta_max_base
        q_full_min = torch.cat([q_min, base_min], dim=0)
        q_full_max = torch.cat([q_max, base_max], dim=0)
        def cost_function(q):
            #  find q that d(x,q) = 0
            # q : B,2
            # x : N,3

            d = self.compute_sdf_with_pose(x, q)
            cost = torch.sum(d ** 2)
            return cost

        # t0 = time.time()
        # optimizer for data generation
        if q is None:
            q = torch.rand(batchsize, serial.dof+6).to(self.device) * (q_full_max - q_full_min) + q_full_min

        q0 = copy.deepcopy(q)
        res = minimize(
            cost_function,
            q,
            method='l-bfgs',
            options=dict(line_search='strong-wolfe'),
            max_iter=50,
            disp=0
        )
        d, idx = self.compute_sdf_with_pose(x, res.x, return_index=True)
        d, idx = d.squeeze(), idx.squeeze()
        mask = torch.abs(d) < epsilon
        # q_valid,d,idx = res.x[mask],d[mask],idx[mask]
        boundary_mask = ((res.x > q_full_min) & (res.x < q_full_max)).all(dim=1)
        final_mask = mask & boundary_mask
        final_q, idx = res.x[final_mask], idx[final_mask]
        # q0 = q0[mask][boundary_mask]

        # print('number of q_valid: \t{} \t time cost:{}'.format(len(final_q), time.time() - t0))

        if return_mask:
            return final_mask, final_q, idx
        else:
            return final_q, idx

    def distance_q(self, x, q):
        # x : (Nx,3)
        # q : (Np,7)
        # return d : (Np) distance between q and x in C space. d = min_{q*}{L2(q-q*)}. sdf(x,q*)=0

        # compute d
        Np = q.shape[0]
        if self.with_base:
            q_template, link_idx = self.given_x_find_q_with_pose(x)
        else:
            q_template, link_idx = self.given_x_find_q(x)

        if link_idx.min() == 0:  # TODO why?
            return torch.zeros(Np).to(self.device)
        else:
            # link_idx[link_idx==7] = 6
            # link_idx[link_idx==8] = 7 #TODO why?
            d = torch.inf * torch.ones(Np, self.robot.dof).to(self.device)
            for i in range(link_idx.min(), link_idx.max() + 1):
                mask = (link_idx == i)
                d_norm = torch.norm(q[:, :i].unsqueeze(1) - q_template[mask][:, :i].unsqueeze(0), dim=-1)
                if d_norm.shape[1] == 0:
                    d[:, i - 1] = torch.inf
                else:
                    d[:, i - 1] = torch.min(d_norm, dim=-1)[0]
        d = torch.min(d, dim=-1)[0]

        # compute sign of d
        d_ts = self.compute_sdf(x, q)
        mask = (d_ts < 0)
        d[mask] = -d[mask]
        return d

    def projection(self, x, q):
        q.requires_grad = True
        d = self.distance_q(x, q)
        grad = torch.autograd.grad(d, q, torch.ones_like(d), create_graph=True)[0]
        q_new = q - grad * d.unsqueeze(-1)
        return q_new

    def generate_offline_data(self, save_path=CUR_DIR, serial_idx=None):
        x = torch.linspace(self.workspace[0][0], self.workspace[1][0], self.n_disrete).to(self.device)
        y = torch.linspace(self.workspace[0][1], self.workspace[1][1], self.n_disrete).to(self.device)
        z = torch.linspace(self.workspace[0][2], self.workspace[1][2], self.n_disrete).to(self.device)
        x, y, z = torch.meshgrid(x, y, z)
        print('generate offline data within workspace:', self.workspace)
        pts = torch.stack([x, y, z], dim=-1).reshape(-1, 3)
        data = {}
        for i, p in enumerate(pts):
            q, idx = self.given_x_find_q_with_pose(p.unsqueeze(0), serial_idx=serial_idx)
            data[i] = {
                'x': p.detach().cpu().numpy(),
                'q': q.detach().cpu().numpy(),
                'idx': idx.detach().cpu().numpy(),
            }
            # print(f'point {i} finished, number of q: {len(q)}')
        np.save(os.path.join(save_path, f'data_with_base_dof_{serial_idx}.npy'), data)

def analysis_data(x):
    # Compute the squared Euclidean distance between each row
    diff = x.unsqueeze(1) - x.unsqueeze(0)
    diff = diff.pow(2).sum(-1)

    # Set the diagonal elements to a large value to exclude self-distance
    diag_indices = torch.arange(x.shape[0])
    diff[diag_indices, diag_indices] = float('inf')

    # Compute the Euclidean distance by taking the square root
    diff = diff.sqrt()
    min_dist = torch.min(diff, dim=1)[0]
    print(f'distance\tmax:{min_dist.max()}\tmin:{min_dist.min()}\taverage:{min_dist.mean()}')


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot', default='panda', type=str, choices=['panda', 'leaphand', 'dexhand'])
    args = parser.parse_args()

    robot = args.robot
    CUR_DIR = os.path.dirname(os.path.abspath(__file__))
    paths = {
        'urdf': os.path.join(CUR_DIR, f'../../RDF/descriptions/{args.robot}/*.urdf'),
        'meshes': os.path.join(CUR_DIR, f'../../RDF/descriptions/{args.robot}/meshes/*.stl'),
        'points': os.path.join(CUR_DIR, f'../../RDF/data/{args.robot}/sdf_points/'),
        'model': os.path.join(CUR_DIR, f'../../RDF/models/{args.robot}/BP_8.pt'),
        'data': os.path.join(CUR_DIR, f'data/{args.robot}/data.pt'),
    }
    parallel_robot = ParallelRobotLayer(device=device, robot=robot, paths=paths)
    for i, serial_robot in enumerate(parallel_robot.serials):
    # x = torch.tensor([[0.5,0.5,0.5]]).to(device)
    # gen.single_point_generation(x)
        if i != 1 and i != 2:
            continue
        gen = DataGenerator(device, parallel_robot, paths, serial_idx=i)
        print(f'Generating data for serial robot {i}, ee_link: {serial_robot.all_links}')
        t0 = time.time()
        gen.generate_offline_data(serial_idx=i)
        print(f'Finished generating data for serial robot {i} in {time.time() - t0:.2f} seconds')