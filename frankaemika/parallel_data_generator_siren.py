"""
Parallel Robot Data Generator using SIREN SDF

This module generates training data for parallel robots using SIREN neural implicit SDF.
For each point in the workspace, it finds multiple joint configurations q where sdf(x,q)=0.

Key differences from BP-SDF version:
- Uses SIREN neural networks for SDF queries (faster, more accurate)
- Supports optional 6-DOF base transformations
- Compatible with SerialRobotLayer from parallel_robot_layer

Data format: {point_idx: {'x': x, 'q': [...], 'idx': [...]}}
- point_idx: index of the point in the workspace grid
- x: point coordinates (3D)
- q: list of joint configurations that satisfy sdf(x,q)=0
- idx: list of link indices closest to x for each q
"""

import torch
import os
CUR_DIR = os.path.dirname(os.path.realpath(__file__))
import numpy as np
import sys
sys.path.append(os.path.join(CUR_DIR, '../../RDF'))
from panda_layers.robot_layer import RobotLayer
from panda_layers.parallel_robot_layer import ParallelRobotLayer
from panda_layers.serial_robot_layer import SerialRobotLayer
from siren_sdf import SirenSDF
from torchmin import minimize
import argparse
import time
import math
import copy
import utils

PI = math.pi


class DataGeneratorSiren():
    """
    Data generator using SIREN SDF for fast and accurate distance queries.
    
    Purpose:
        Generate training data by finding joint configurations that place robot
        links on the surface of workspace query points (sdf = 0).
    
    Usage:
        gen = DataGeneratorSiren(device, robot, paths, serial_idx=0, with_base=False)
        gen.generate_offline_data(save_path='./data', serial_idx=0)
    """
    
    def __init__(self, device, robot, paths, serial_idx=None, with_base=False):
        """
        Initialize SIREN-based data generator.
        
        Args:
            device: torch device (cuda/cpu)
            robot: ParallelRobotLayer instance
            paths: dict with keys ['urdf', 'meshes', 'points', 'model']
            serial_idx: index of serial robot to use (0-based)
            with_base: if True, include 6-DOF base transformation in optimization
        """
        self.robot = robot
        self.paths = paths
        self.device = device
        self.serial_idx = serial_idx
        self.with_base = with_base
        
        # Initialize SIREN SDF model
        self.siren_sdf = SirenSDF(
            robot=robot,
            paths=paths,
            device=device,
            domain_min=-1.0,
            domain_max=1.0
        )
        
        # Load trained SIREN models
        siren_model_path = paths['model']
        if not os.path.exists(siren_model_path):
            raise FileNotFoundError(f"SIREN model not found at {siren_model_path}")
        
        self.model = torch.load(siren_model_path, map_location=device, weights_only=False)
        print(f"Loaded SIREN model from: {siren_model_path}")
        print(f"Model contains meshes: {list(self.model.keys())}")
        
        # Workspace configuration
        self.workspace = self.robot.space_limits.cpu().numpy()
        self.n_disrete = 20  # 20x20x20 = 8000 points (5cm grid spacing)
        self.batchsize = 20000  # batch size for optimization
        self.epsilon = 1e-3  # distance threshold to filter valid configurations
        
        # Links to use for this serial robot
        self.used_links = self.robot.serials[self.serial_idx].all_links.copy()
        print(f'Used links for serial {serial_idx}: {self.used_links}')
        
        # Keep palm_lower_left if present
        if 'palm_lower_left' in self.used_links:
            print('palm_lower_left is included in used_links')

    def compute_sdf(self, x, q, pose=None, return_index=False):
        """
        Compute SDF using SIREN for given query points and configurations.
        
        This is the core SDF query function without base DOF.
        
        Args:
            x: (Nx, 3) query points in world frame
            q: (Nq, dof) joint configurations
            pose: (Nq, 4, 4) base poses (default: identity)
            return_index: if True, return link index closest to each point
            
        Returns:
            d: (Nq,) minimum distances
            idx: (Nq,) optional, index of closest link for each configuration
        """
        Nq = q.shape[0]
        
        # Default to identity pose if not provided
        if pose is None:
            pose = torch.eye(4, device=self.device).unsqueeze(0).expand(Nq, 4, 4)
        
        # Query SIREN SDF for serial robot
        if not return_index:
            d, _ = self.siren_sdf.get_serial_sdf_batch(
                x, pose, q, self.model,
                serial_idx=self.serial_idx,
                use_derivative=False,
                used_links=self.used_links
            )
            # d shape: (Nq, Nx), take minimum over query points
            d = d.min(dim=1)[0]  # (Nq,)
            return d
        else:
            d, _, idx = self.siren_sdf.get_serial_sdf_batch(
                x, pose, q, self.model,
                serial_idx=self.serial_idx,
                use_derivative=False,
                used_links=self.used_links,
                return_index=True
            )
            # d: (Nq, Nx), idx: (Nq, Nx)
            d, pts_idx = d.min(dim=1)  # (Nq,)
            idx = idx[torch.arange(len(idx)), pts_idx]  # (Nq,)
            return d, idx
    
    def compute_sdf_with_pose(self, x, q, return_index=False):
        """
        Compute SDF with 6-DOF base transformation included in q.
        
        This function is used when with_base=True, treating the last 6 DOF
        of q as base pose parameters [tx, ty, tz, rx, ry, rz].
        
        Args:
            x: (Nx, 3) query points
            q: (Nq, dof+6) joint configs + base pose [joints..., tx, ty, tz, rx, ry, rz]
            return_index: if True, return link indices
            
        Returns:
            d: (Nq,) distances
            idx: (Nq,) optional link indices
        """
        # Split joint angles and base pose
        q_pose = q[:, -6:]  # (Nq, 6) - last 6 DOF are base [translation, rotation]
        q_joint = q[:, :-6]  # (Nq, dof) - joint angles
        
        # Convert 6D pose to 4x4 transformation matrix
        pose = utils.q_to_poseMatrix(self, q_pose).to(self.device)  # (Nq, 4, 4)
        
        # Query SIREN SDF
        if not return_index:
            d, _ = self.siren_sdf.get_serial_sdf_batch(
                x, pose, q_joint, self.model,
                serial_idx=self.serial_idx,
                use_derivative=False,
                used_links=self.used_links
            )
            d = d.min(dim=1)[0]
            return d
        else:
            d, _, idx = self.siren_sdf.get_serial_sdf_batch(
                x, pose, q_joint, self.model,
                serial_idx=self.serial_idx,
                use_derivative=False,
                used_links=self.used_links,
                return_index=True
            )
            d, pts_idx = d.min(dim=1)
            idx = idx[torch.arange(len(idx)), pts_idx]
            return d, idx

    def given_x_find_q(self, x, q=None, batchsize=None, return_mask=False, epsilon=1e-3, serial_idx=None):
        """
        Given a query point x, find joint configurations q where sdf(x, q) = 0.
        
        This is the core inverse kinematics solver for surface sampling.
        Uses L-BFGS optimization to minimize ||sdf(x, q)||^2.
        
        Args:
            x: (N, 3) query point(s) in world frame
            q: (B, dof) initial guess (random if None)
            batchsize: number of random initializations (default: self.batchsize)
            return_mask: if True, return boolean mask of valid solutions
            epsilon: distance threshold for valid solutions
            serial_idx: serial robot index (default: self.serial_idx)
            
        Returns:
            final_q: (M, dof) valid joint configurations where sdf ≈ 0
            idx: (M,) link indices closest to x for each configuration
            
        Usage:
            x = torch.tensor([[0.5, 0.5, 0.5]]).to(device)
            q_valid, link_idx = gen.given_x_find_q(x)
        """
        if not batchsize:
            batchsize = self.batchsize
        if serial_idx is None:
            serial_idx = self.serial_idx
        
        serial = self.robot.serials[serial_idx]
        q_min = serial.theta_min_soft
        q_max = serial.theta_max_soft
        
        def cost_function(q):
            """Minimize squared distance: sum(sdf(x, q)^2)"""
            d = self.compute_sdf(x, q)
            cost = torch.sum(d ** 2)
            return cost
        
        # Random initialization if not provided
        if q is None:
            q = torch.rand(batchsize, serial.dof).to(self.device) * (q_max - q_min) + q_min
        
        q0 = copy.deepcopy(q)
        
        # Optimize using L-BFGS
        res = minimize(
            cost_function,
            q,
            method='l-bfgs',
            options=dict(line_search='strong-wolfe'),
            max_iter=50,
            disp=0
        )
        
        # Filter valid solutions: distance < epsilon and within joint limits
        d, idx = self.compute_sdf(x, res.x, return_index=True)
        d, idx = d.squeeze(), idx.squeeze()
        
        distance_mask = torch.abs(d) < epsilon
        boundary_mask = ((res.x > q_min) & (res.x < q_max)).all(dim=1)
        final_mask = distance_mask & boundary_mask
        
        final_q, idx = res.x[final_mask], idx[final_mask]
        
        if return_mask:
            return final_mask, final_q, idx
        else:
            return final_q, idx
    
    def given_x_find_q_with_pose(self, x, q=None, batchsize=None, return_mask=False, epsilon=1e-3, serial_idx=None):
        """
        Find joint configs + base poses where sdf(x, q) = 0.
        
        Extended version of given_x_find_q that optimizes over joint angles
        AND 6-DOF base transformation simultaneously.
        
        Args:
            x: (N, 3) query point(s)
            q: (B, dof+6) initial guess [joints..., tx, ty, tz, rx, ry, rz]
            batchsize: number of random initializations
            return_mask: if True, return boolean mask
            epsilon: distance threshold
            serial_idx: serial robot index
            
        Returns:
            final_q: (M, dof+6) valid configurations [joints + base pose]
            idx: (M,) link indices
            
        Usage:
            x = torch.tensor([[0.5, 0.5, 0.5]]).to(device)
            q_with_base, link_idx = gen.given_x_find_q_with_pose(x)
            # q_with_base[:, :-6] are joint angles
            # q_with_base[:, -6:] are base poses
        """
        if not batchsize:
            batchsize = self.batchsize
        if serial_idx is None:
            serial_idx = self.serial_idx
        
        serial = self.robot.serials[serial_idx]
        q_min = serial.theta_min_soft
        q_max = serial.theta_max_soft
        
        # Base pose limits: 6-DOF [tx, ty, tz, rx, ry, rz]
        base_dof = 6
        base_min = serial.theta_min_base
        base_max = serial.theta_max_base
        
        # Combined limits: [joint_min..., base_min...] to [joint_max..., base_max...]
        q_full_min = torch.cat([q_min, base_min], dim=0)
        q_full_max = torch.cat([q_max, base_max], dim=0)
        
        def cost_function(q):
            """Minimize squared distance: sum(sdf(x, q)^2)"""
            d = self.compute_sdf_with_pose(x, q)
            cost = torch.sum(d ** 2)
            return cost
        
        # Random initialization
        if q is None:
            q = torch.rand(batchsize, serial.dof + 6).to(self.device) * (q_full_max - q_full_min) + q_full_min
        
        q0 = copy.deepcopy(q)
        
        # Optimize
        res = minimize(
            cost_function,
            q,
            method='l-bfgs',
            options=dict(line_search='strong-wolfe'),
            max_iter=50,
            disp=0
        )
        
        # Filter valid solutions
        d, idx = self.compute_sdf_with_pose(x, res.x, return_index=True)
        d, idx = d.squeeze(), idx.squeeze()
        
        distance_mask = torch.abs(d) < epsilon
        boundary_mask = ((res.x > q_full_min) & (res.x < q_full_max)).all(dim=1)
        final_mask = distance_mask & boundary_mask
        
        final_q, idx = res.x[final_mask], idx[final_mask]
        
        if return_mask:
            return final_mask, final_q, idx
        else:
            return final_q, idx

    def distance_q(self, x, q):
        """
        Compute configuration-space distance from q to surface manifold.
        
        For a query point x and configuration q, finds the minimum distance
        in configuration space to any q* where sdf(x, q*) = 0.
        
        Returns signed distance: negative if robot penetrates surface.
        
        Args:
            x: (Nx, 3) query point
            q: (Np, dof) or (Np, dof+6) joint configurations to evaluate
            
        Returns:
            d: (Np,) signed configuration-space distances
            
        Usage:
            x = torch.tensor([[0.5, 0.5, 0.5]]).to(device)
            q_test = torch.randn(100, robot.dof).to(device)
            dist = gen.distance_q(x, q_test)
            # dist > 0: outside, dist < 0: penetrating
        """
        Np = q.shape[0]
        
        # Find surface configurations q* where sdf(x, q*) = 0
        if self.with_base:
            q_template, link_idx = self.given_x_find_q_with_pose(x)
            # q_template: (Nq, dof+6), link_idx: (Nq)
        else:
            q_template, link_idx = self.given_x_find_q(x)
            # q_template: (Nq, dof), link_idx: (Nq)
        
        # Handle edge case: if base link is closest and no base DOF
        if link_idx.min() == 0 and not self.with_base:
            return torch.zeros(Np).to(self.device)
        
        # Compute minimum L2 distance in configuration space
        base_offset = 6 if self.with_base else 0
        d = torch.inf * torch.ones(Np, self.robot.dof).to(self.device)
        
        for i in range(link_idx.min(), link_idx.max() + 1):
            mask = (link_idx == i)
            # Only compare relevant DOFs (up to link i)
            d_norm = torch.norm(
                q[:, :i + base_offset].unsqueeze(1) - q_template[mask][:, :i + base_offset].unsqueeze(0),
                dim=-1
            )
            if d_norm.shape[1] == 0:
                d[:, i + base_offset - 1] = torch.inf
            else:
                d[:, i + base_offset - 1] = torch.min(d_norm, dim=-1)[0]
        
        d = torch.min(d, dim=-1)[0]  # (Np,)
        
        # Determine sign: negative if penetrating
        if self.with_base:
            d_ts = self.compute_sdf_with_pose(x, q)
        else:
            d_ts = self.compute_sdf(x, q)
        
        mask = (d_ts < 0)
        d[mask] = -d[mask]
        
        return d

    def projection(self, x, q):
        """
        Project configuration q onto the surface manifold sdf(x, q*) = 0.
        
        Uses gradient descent: q* = q - ∇_q distance(x, q) * distance(x, q)
        
        Args:
            x: (N, 3) query point
            q: (B, dof) configurations to project
            
        Returns:
            q_new: (B, dof) projected configurations on surface
            
        Usage:
            x = torch.tensor([[0.5, 0.5, 0.5]]).to(device)
            q_init = torch.randn(10, robot.dof).to(device)
            q_proj = gen.projection(x, q_init)
            # q_proj should satisfy sdf(x, q_proj) ≈ 0
        """
        q.requires_grad = True
        d = self.distance_q(x, q)
        grad = torch.autograd.grad(d, q, torch.ones_like(d), create_graph=True)[0]
        q_new = q - grad * d.unsqueeze(-1)
        return q_new

    def generate_offline_data(self, save_path=CUR_DIR, serial_idx=None):
        """
        Generate and save workspace data for offline training.
        
        Creates a 3D grid in workspace and finds multiple joint configurations
        that place robot links at each grid point (sdf = 0).
        
        Data is saved as .npy file with structure:
        {
            point_idx: {
                'x': np.array([x, y, z]),           # 3D point
                'q': np.array([[q1], [q2], ...]),   # valid joint configs
                'idx': np.array([i1, i2, ...])      # closest link indices
            },
            ...
        }
        
        Args:
            save_path: directory to save data file
            serial_idx: serial robot index (default: self.serial_idx)
            
        Output file:
            {save_path}/data_with_base_dof_{serial_idx}.npy (if with_base=True)
            {save_path}/data_dof_{serial_idx}.npy (if with_base=False)
            
        Usage:
            gen = DataGeneratorSiren(device, robot, paths, serial_idx=0, with_base=False)
            gen.generate_offline_data(save_path='./data', serial_idx=0)
        """
        # Create 3D grid in workspace
        x = torch.linspace(self.workspace[0][0], self.workspace[1][0], self.n_disrete).to(self.device)
        y = torch.linspace(self.workspace[0][1], self.workspace[1][1], self.n_disrete).to(self.device)
        z = torch.linspace(self.workspace[0][2], self.workspace[1][2], self.n_disrete).to(self.device)
        x, y, z = torch.meshgrid(x, y, z, indexing='ij')
        
        print(f'Generating offline data within workspace: {self.workspace}')
        print(f'Grid size: {self.n_disrete}^3 = {self.n_disrete**3} points')
        
        pts = torch.stack([x, y, z], dim=-1).reshape(-1, 3)
        data = {}
        
        # Process each grid point
        for i, p in enumerate(pts):
            if self.with_base:
                q, idx = self.given_x_find_q_with_pose(p.unsqueeze(0), serial_idx=serial_idx)
            else:
                q, idx = self.given_x_find_q(p.unsqueeze(0), serial_idx=serial_idx)
            
            data[i] = {
                'x': p.detach().cpu().numpy(),
                'q': q.detach().cpu().numpy(),
                'idx': idx.detach().cpu().numpy(),
            }
            
            if (i + 1) % 100 == 0:
                print(f'Processed {i+1}/{len(pts)} points, last point found {len(q)} configurations')
        
        # Save data
        if self.with_base:
            filename = f'data_with_base_dof_{serial_idx}.npy'
        else:
            filename = f'data_dof_{serial_idx}.npy'
        
        save_file = os.path.join(save_path, 'data',f'{self.robot.robot}', filename)
        np.save(save_file, data)
        print(f'Data saved to: {save_file}')


def analysis_data(x):
    """
    Analyze configuration space coverage by computing pairwise distances.
    
    Args:
        x: (N, dof) tensor of joint configurations
        
    Prints:
        Maximum, minimum, and average nearest-neighbor distances
        
    Usage:
        q_data = torch.randn(1000, 7).to(device)
        analysis_data(q_data)
    """
    # Compute pairwise Euclidean distances
    diff = x.unsqueeze(1) - x.unsqueeze(0)
    diff = diff.pow(2).sum(-1)
    
    # Exclude self-distances
    diag_indices = torch.arange(x.shape[0])
    diff[diag_indices, diag_indices] = float('inf')
    
    # Nearest neighbor distances
    diff = diff.sqrt()
    min_dist = torch.min(diff, dim=1)[0]
    
    print(f'Nearest neighbor distance statistics:')
    print(f'  Max: {min_dist.max():.6f}')
    print(f'  Min: {min_dist.min():.6f}')
    print(f'  Average: {min_dist.mean():.6f}')


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser(description='Generate training data using SIREN SDF')
    parser.add_argument('--robot', default='leap', type=str, 
                        choices=['panda', 'leaphand', 'dexhand', 'leap'],
                        help='Robot model to use')
    parser.add_argument('--serial_idx', default=None, type=int,
                        help='Serial robot index to process (None = all serials)')
    parser.add_argument('--with_base', action='store_true',
                        help='Include 6-DOF base transformation in optimization')
    args = parser.parse_args()
    
    robot = args.robot
    CUR_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Setup paths
    paths = {
        'urdf': os.path.join(CUR_DIR, f'../../RDF/descriptions/{args.robot}/*.urdf'),
        'meshes': os.path.join(CUR_DIR, f'../../RDF/descriptions/{args.robot}/meshes/*.stl'),
        'points': os.path.join(CUR_DIR, f'../../RDF/data/{args.robot}/sdf_points/'),
        'model': os.path.join(CUR_DIR, f'../../RDF/models/{args.robot}/{args.robot}_siren.pth'),
        'data': os.path.join(CUR_DIR, f'data/{args.robot}/data.pt'),
    }
    
    # Special handling for leap hand (uses .glb meshes)
    if args.robot == 'leap':
        paths['meshes'] = os.path.join(CUR_DIR, f'../../RDF/descriptions/{args.robot}/meshes/visual/*.glb')
    
    # Initialize parallel robot
    parallel_robot = ParallelRobotLayer(device=device, robot=robot, paths=paths)
    
    # Process specified serial robots
    if args.serial_idx is not None:
        # Single serial robot
        serial_indices = [args.serial_idx]
    else:
        # All serial robots
        serial_indices = range(len(parallel_robot.serials))
    
    for i in serial_indices:
        serial_robot = parallel_robot.serials[i]
        print(f'\n{"="*60}')
        print(f'Generating data for serial robot {i}')
        print(f'End-effector links: {serial_robot.all_links}')
        print(f'DOF: {serial_robot.dof}')
        print(f'With base: {args.with_base}')
        print(f'{"="*60}\n')
        
        # Create data generator
        gen = DataGeneratorSiren(
            device=device,
            robot=parallel_robot,
            paths=paths,
            serial_idx=i,
            with_base=args.with_base
        )
        
        # Generate and save data
        t0 = time.time()
        gen.generate_offline_data(serial_idx=i)
        elapsed = time.time() - t0
        
        print(f'\n{"="*60}')
        print(f'Finished serial robot {i} in {elapsed:.2f} seconds')
        print(f'{"="*60}\n')
