# Author: Hefei Wang
# date: 2025-11-03
# CDF model with support for base DoF and choice between MLP and SIREN networks
import torch
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

import numpy as np
import os

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb=256'
import sys
import argparse
import math
import time

CUR_PATH = os.path.dirname(os.path.realpath(__file__))
from mlp import MLPRegression
sys.path.append(os.path.join(CUR_PATH,'../../RDF'))
from Siren import Siren
sys.path.append(os.path.join(CUR_PATH,'../../RDF/panda_layers'))
from parallel_robot_layer import ParallelRobotLayer
from parallel_bf_sdf import ParallelBPSDF
import utils

PI = math.pi
np.random.seed(10)

class CDF_V2:
    def __init__(self, device, paths, robot, network_type='mlp', signed_distance=False, 
                 writer=None, serial_idx=0, use_base=False) -> None:
        """
        CDF model with support for base DoF
        
        Args:
            device: torch device
            paths: dictionary containing paths to data, model, etc.
            robot: robot model
            network_type: 'mlp' or 'siren'
            signed_distance: whether to use signed distance
            writer: tensorboard writer
            serial_idx: index of serial chain
            use_base: whether to include 6DoF base in the configuration space
        """
        # device
        self.device = device  
        self.writer = writer
        self.batch_x = 10
        self.batch_q = 100
        self.max_q_per_link = 100
        self.paths = paths
        self.network_type = network_type
        self.use_base = use_base
        
        # --- initialize robot model ---
        self.robot = ParallelRobotLayer(device=device, paths=paths, robot=robot)
        self.serial_idx = serial_idx
        print(paths)
        # Process raw data if needed (comment out if data is already processed)
        if 'raw_data' in paths and os.path.exists(paths['raw_data']):
            data_path = paths['raw_data'].split('.npy')[0] + f'.pt'
            self.paths['data'] = data_path
            if not os.path.exists(paths['data']):
                print(f"Processing raw data from {paths['raw_data']}")
                self.raw_data = np.load(paths['raw_data'], allow_pickle=True).item()
                self.process_data(self.raw_data)
            else:
                print(f"Processed data already exists at {paths['data']}, skipping processing")
        else:
            raise FileNotFoundError(f"Raw data file not found at {paths['raw_data']}")
        # Load data
        self.data_path = paths['data']
        self.data = self.load_data(self.data_path)
        self.len_data = len(self.data['k'])
        self.signed_distance = signed_distance
        
        # Initialize SDF model
        self.paths = paths
        self.bp_sdf_model_path = paths['model']
        self.bp_sdf = ParallelBPSDF(8, -1.0, 1.0, self.robot, self.bp_sdf_model_path, device)
        self.bp_sdf_model = torch.load(self.bp_sdf_model_path)
        self.model_dict = paths['model_dict']
        
        # Set used joints based on robot type
        if robot == 'panda':
            self.used_joints = [0, 1, 2, 3, 4, 5, 6]
        elif robot == 'leaphand':
            self.used_joints = [0, 1, 2, 3]
        elif robot == 'dexhand':
            self.used_joints = [0, 1, 2, 3]

    def process_data(self, data):
        """
        Process raw data by downsampling joint configurations per link
        
        From raw data, downsample (using pytorch3d.ops.sample_farthest_points) the sample points for each joint,
        "each joint" means that for each sample point, the number of samples where each joint is the 
        "last colliding joint" does not exceed max_q_per_link
        
        Args:
            data: {key: {'x': (N,3), 'q': (N,DoF) or (N,DoF+6), 'idx': (N)}}
                  idx: (N) stands for which link is the last link that causes the collision
        
        Returns:
            final_data: {'x': (G,3), 'q': (G,max_q_per_link,D,D), 'k': (G)}
                        G is the number of grids
                        D is DoF or DoF+6 depending on use_base
        """
        import pytorch3d.ops
        
        serial = self.robot.serials[self.serial_idx]
        DoF = serial.dof
        
        # Determine configuration space dimension
        if self.use_base:
            config_dim = DoF + 6  # joint angles + base pose
        else:
            config_dim = DoF
            
        keys = list(data.keys())  # Create a copy of the keys
        processed_data = {}
        
        print(f"Processing data with {'base DoF' if self.use_base else 'no base'}")
        print(f"Configuration dimension: {config_dim} (DoF: {DoF})")
        
        for k in keys:  # 每个key对应一个采样点，即Cartesian空间中的一个grid点
            if len(data[k]['q']) == 0:
                # 如果没有采样点，就跳过这个key
                data.pop(k)
                continue
                
            q = torch.from_numpy(data[k]['q']).float().to(self.device)
            # q:(N,config_dim) N: number of samples for this key
            
            q_idx = torch.from_numpy(data[k]['idx']).float().to(self.device)
            # q_idx:(N) stands for which link is the last link that causes the collision
            
            # Initialize q_lib with inf
            q_lib = torch.inf * torch.ones(self.max_q_per_link, config_dim, DoF).to(self.device)
            
            # Process each link
            num_links = len(serial.all_links)
            for i in range(1, num_links + 1):
                mask = (q_idx == i)  # find the samples where the last colliding link is i
                
                if mask.sum() == 0:
                    continue
                
                # 如果这个link的采样点多于max_q_per_link，就用farthest point sampling降采样
                if len(q[mask]) > self.max_q_per_link:
                    # For base DoF, we only use joint angles for FPS (not base pose)
                    if self.use_base:
                        q_joint_only = q[mask][:, :DoF]  # Use only joint angles for FPS
                    else:
                        q_joint_only = q[mask]
                    
                    fps_q_joint = pytorch3d.ops.sample_farthest_points(
                        q_joint_only.unsqueeze(0), K=self.max_q_per_link
                    )[0].squeeze()
                    
                    # Find indices of FPS samples in original data
                    # Use joint angles only for matching
                    distances = torch.cdist(fps_q_joint, q_joint_only)
                    fps_indices = distances.argmin(dim=1)
                    
                    # Get full configurations (including base if present)
                    fps_q_full = q[mask][fps_indices]
                    q_lib[:, :config_dim, i-1] = fps_q_full
                    
                elif len(q[mask]) > 0:
                    q_lib[:len(q[mask]), :config_dim, i-1] = q[mask]
            
            # q_lib:(max_q_per_link, config_dim, DoF) 
            # (first dim: index of the q sample,
            #  second dim: configuration dimension (DoF or DoF+6),
            #  third dim: link index, i.e. to get all samples for link i, use q_lib[:,:,i-1])
            
            if torch.isinf(q_lib).all():
                print(f'Warning: all inf in key {k}, remove this key')
                print('corresponding x:', data[k]['x'])
                data.pop(k)
                continue
                
            processed_data[k] = {
                'x': torch.from_numpy(data[k]['x']).float().to(self.device),
                'q': q_lib,
            }
        
        # Combine all processed data
        final_data = {
            'x': torch.cat([processed_data[k]['x'].unsqueeze(0) for k in processed_data.keys()], dim=0),
            'q': torch.cat([processed_data[k]['q'].unsqueeze(0) for k in processed_data.keys()], dim=0),
            'k': torch.tensor([k for k in processed_data.keys()]).to(self.device)
        }
        
        print(f'Processed data shapes:')
        print(f"  x: {final_data['x'].shape}")
        print(f"  q: {final_data['q'].shape}")
        print(f"  k: {final_data['k'].shape}")
        
        # Save processed data
        torch.save(final_data, self.paths['data'])
        print(f"Saved processed data to {self.paths['data']}")
        
        return data

    def load_data(self, path):
        data = torch.load(path)
        return data

    def select_data(self):
        """
        Select batch of data for training
        Returns:
            x_batch: (batch_x, 3) - Cartesian positions
            q_batch: (batch_q, DoF) or (batch_q, DoF+6) if use_base - configurations
            d: (batch_x, batch_q) - distances
            grad: (batch_x, batch_q, DoF) or (batch_x, batch_q, DoF+6) - gradients
        """
        x = self.data['x']
        q = self.data['q']

        idx = torch.randint(0, len(x), (self.batch_x,)) 
        x_batch, q_lib = x[idx], q[idx]
        q_batch = self.sample_q()   
        d, grad = self.decode_distance(q_batch, q_lib)
        # 用decode_distance与distance_q对比
        from parallel_data_generator import DataGenerator
        data_gen = DataGenerator(device=self.device, paths=self.paths, robot=self.robot, 
                                 serial_idx=self.serial_idx, with_base=self.use_base)
        d_check = data_gen.distance_q(x_batch, q_batch)
        diff = torch.abs(d - d_check)
        if torch.max(diff) > 1e-3:
            print('Warning: distance mismatch between decode_distance and distance_q')
            print('Max difference:', torch.max(diff).item(), 'Average difference:', torch.mean(diff).item())
            print('d from decode_distance:', torch.max(d).item(), torch.min(d).item(), torch.mean(d).item())
            print('d from distance_q:', torch.max(d_check).item(), torch.min(d_check).item(), torch.mean(d_check).item())
        exit()
        return x_batch, q_batch, d, grad
    
    def select_data_signed(self):
        """
        Select batch of data with signed distance
        """
        x = self.data['x']
        q = self.data['q']
        idx = torch.randint(0, len(x), (self.batch_x,)) 
        x_batch, q_lib = x[idx], q[idx]
        q_batch = self.sample_q()   
        d, grad = self.decode_distance_signed(q_batch, q_lib, x_batch)
        return x_batch, q_batch, d, grad

    def decode_distance(self, q_batch, q_lib):
        """
        Compute distance from q_batch to the manifold defined by q_lib
        
        Args:
            q_batch: (batch_q, DoF) or (batch_q, DoF+6) - query configurations
            q_lib: (batch_x, max_q_per_link, config_dim, DoF) - library configurations
                   config_dim = DoF or DoF+6 depending on use_base
        Returns:
            d: (batch_x, batch_q) - distances
            grad: (batch_x, batch_q, DoF) or (batch_x, batch_q, DoF+6) - gradients
        """
        DoF = self.robot.serials[self.serial_idx].dof        
        base_offset = 6 if self.use_base else 0

        q_joint = q_batch[:, :DoF+base_offset]
        total_dof = DoF + base_offset

        batch_x = q_lib.shape[0]
        batch_q = q_batch.shape[0]
        d_tensor = torch.ones(batch_x, batch_q, DoF).to(self.device) * torch.inf
        grad_tensor = torch.zeros(batch_x, batch_q, total_dof, DoF).to(self.device)
        for i in range(DoF):
            # Extract joint angles from q_lib (first DoF dimensions)
            # q_lib shape: (batch_x, max_q_per_link, config_dim, DoF)
            # We want: (batch_x, max_q_per_link, i+1) for link i
            q_lib_temp = q_lib[:, :, :i+base_offset+1, i].reshape(batch_x*self.max_q_per_link, -1).unsqueeze(0).expand(batch_q, -1, -1)
            q_joint_temp = q_joint[:, :i+base_offset+1].unsqueeze(1).expand(-1, batch_x*self.max_q_per_link, -1)
            d_norm = torch.norm((q_joint_temp - q_lib_temp), dim=-1).reshape(batch_q, batch_x, self.max_q_per_link)

            d_norm_min, d_norm_min_idx = d_norm.min(dim=-1)
            grad = torch.autograd.grad(d_norm_min.reshape(-1), q_joint_temp, torch.ones_like(d_norm_min.reshape(-1)), 
                                      retain_graph=True)[0]
            grad_min_q = grad.reshape(batch_q, batch_x, self.max_q_per_link, -1).gather(
                2, d_norm_min_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, i+base_offset+1))[:, :, 0, :]
            grad_tensor[:, :, :i+base_offset+1, i] = grad_min_q.transpose(0, 1)
            d_tensor[:, :, i] = d_norm_min.transpose(0, 1)

        d, d_min_idx = d_tensor.min(dim=-1)
        grad_final = grad_tensor.gather(3, d_min_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, total_dof, -1))[:, :, :, 0]
        
        return d, grad_final
    
    def decode_distance_signed(self, q_batch, q_lib, x_batch):
        """
        Compute signed distance
        """
        # DoF = self.robot.serials[self.serial_idx].dof
        # if self.use_base:
        #     q_joint = q_batch[:, :DoF]
        #     total_dof = DoF + 6
        # else:
        #     q_joint = q_batch
        #     total_dof = DoF
            
        # batch_x = q_lib.shape[0]
        # batch_q = q_batch.shape[0]
        # d_tensor = torch.ones(batch_x, batch_q, DoF).to(self.device) * torch.inf
        # grad_tensor = torch.zeros(batch_x, batch_q, total_dof, DoF).to(self.device)
        # print('shape of q_lib:', q_lib.shape)
        # print('Dof:', DoF)
        # exit()
        # for i in range(DoF):
        #     q_lib_temp = q_lib[:, :, :i+1, i].reshape(batch_x*self.max_q_per_link, -1).unsqueeze(0).expand(batch_q, -1, -1)
        #     q_joint_temp = q_joint[:, :i+1].unsqueeze(1).expand(-1, batch_x*self.max_q_per_link, -1)
        #     d_norm = torch.norm((q_joint_temp - q_lib_temp), dim=-1).reshape(batch_q, batch_x, self.max_q_per_link)
        #     d_norm_min, d_norm_min_idx = d_norm.min(dim=-1)
        #     grad = torch.autograd.grad(d_norm_min.reshape(-1), q_joint_temp, torch.ones_like(d_norm_min.reshape(-1)), 
        #                               retain_graph=True)[0]
        #     grad_min_q = grad.reshape(batch_q, batch_x, self.max_q_per_link, -1).gather(
        #         2, d_norm_min_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, i+1))[:, :, 0, :]
        #     grad_tensor[:, :, :i+1, i] = grad_min_q.transpose(0, 1)
        #     d_tensor[:, :, i] = d_norm_min.transpose(0, 1)
            
        # d, d_min_idx = d_tensor.min(dim=-1)
        # d_ts = self.compute_sdf(x_batch, q_batch)
        # mask = (d_ts < 0).transpose(0, 1)
        # d[mask] = -d[mask]
        # grad_final = grad_tensor.gather(3, d_min_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, total_dof, -1))[:, :, :, 0]
        # grad_final[mask] = -grad_final[mask]
        # 复用decode_distance函数
        d, grad_final = self.decode_distance(q_batch, q_lib)
        d_ts = self.compute_sdf(x_batch, q_batch)
        mask = (d_ts < 0).transpose(0, 1)
        d[mask] = -d[mask]
        grad_final[mask] = -grad_final[mask]
        return d, grad_final
    
    def compute_sdf(self, x, q, return_index=False):
        """
        Compute SDF with support for base DoF
        
        Args:
            x: (Nx, 3) - query points
            q: (Nq, DoF) or (Nq, DoF+6) - configurations
            return_index: if True, return the index of closest link
        """
        if self.use_base:
            # Split q into joint angles and base pose
            DoF = self.robot.serials[self.serial_idx].dof
            q_joint = q[:, :DoF]
            q_base = q[:, DoF:]
            # Convert base parameters to pose matrix
            pose = utils.q_to_poseMatrix(None, q_base).to(self.device).float()
        else:
            q_joint = q
            pose = torch.eye(4).unsqueeze(0).to(self.device).expand(len(q), 4, 4).float()
        
        used_links = self.robot.serials[self.serial_idx].all_links.copy()
        if 'palm_lower_left' in used_links:
            used_links.remove('palm_lower_left')
            
        if not return_index:
            d, _ = self.bp_sdf.get_serial_sdf_batch(x, pose, q_joint, self.bp_sdf_model, 
                                                    use_derivative=False, serial_idx=self.serial_idx, 
                                                    used_links=used_links)
            return d
        else:
            d, _, idx = self.bp_sdf.get_serial_sdf_batch(x, pose, q_joint, self.bp_sdf_model, 
                                                         use_derivative=False, return_index=True, 
                                                         serial_idx=self.serial_idx, used_links=used_links)
            d, pts_idx = d.min(dim=1)
            idx = idx[torch.arange(len(idx)), pts_idx]
            return d, idx
    
    def sample_q(self, batch_q=None):
        """
        Sample random configurations
        Returns:
            q_sampled: (batch_q, DoF) or (batch_q, DoF+6) if use_base
        """
        serial = self.robot.serials[self.serial_idx]
        DoF = serial.dof
        if batch_q is None:
            batch_q = self.batch_q
            
        # Sample joint angles
        q_joint = serial.theta_min + torch.rand(batch_q, DoF).to(self.device) * (serial.theta_max - serial.theta_min)
        
        if self.use_base:
            # Sample base DoF (6DoF: 3 translation + 3 rotation)
            # Assuming base limits are defined in serial
            if hasattr(serial, 'theta_min_base') and hasattr(serial, 'theta_max_base'):
                base_min = serial.theta_min_base
                base_max = serial.theta_max_base
            else:
                # Default base limits if not defined
                print('[Warning]: base limits not defined, using default base limits [-0.5,0.5] for translation and [-pi,pi] for rotation')
                base_min = torch.tensor([-0.5, -0.5, -0.5, -PI, -PI, -PI]).to(self.device)
                base_max = torch.tensor([0.5, 0.5, 0.5, PI, PI, PI]).to(self.device)
            
            q_base = base_min + torch.rand(batch_q, 6).to(self.device) * (base_max - base_min)
            q_sampled = torch.cat([q_joint, q_base], dim=-1)
        else:
            q_sampled = q_joint
            
        q_sampled.requires_grad = True
        return q_sampled
    
    def projection(self, q, d, grad):
        q_new = q - grad * d.unsqueeze(-1)
        return q_new

    def create_network(self, input_dims, output_dims=1):
        """
        Create either MLP or SIREN network based on network_type
        
        Args:
            input_dims: input dimension (3 + DoF or 3 + DoF + 6)
            output_dims: output dimension (default 1)
        Returns:
            model: neural network
        """
        if self.network_type == 'mlp':
            model = MLPRegression(
                input_dims=input_dims, 
                output_dims=output_dims, 
                mlp_layers=[1024, 512, 256, 128, 128],
                skips=[], 
                act_fn=torch.nn.ReLU, 
                nerf=True
            )
        elif self.network_type == 'siren':
            model = Siren(
                in_features=input_dims,
                out_features=output_dims,
                hidden_features=256,
                hidden_layers=3,
                outermost_linear=True,
                first_omega_0=30,
                hidden_omega_0=30
            )
        else:
            raise ValueError(f"Unknown network type: {self.network_type}")
        
        return model

    def train_nn(self, epoches=500):
        """
        Train the neural network CDF model
        """
        # Determine input dimensions
        DoF = self.robot.serials[self.serial_idx].dof
        if self.use_base:
            input_dims = 3 + DoF + 6  # x + joint angles + base pose
        else:
            input_dims = 3 + DoF  # x + joint angles
            
        print(f"Training {self.network_type.upper()} network with input dims: {input_dims}")
        print(f"Base DoF: {'Enabled' if self.use_base else 'Disabled'}")
        
        # Create model
        model = self.create_network(input_dims)
        model.to(self.device)
        
        # Optimizer and scheduler
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5000,
            threshold=0.01, threshold_mode='rel',
            cooldown=0, min_lr=0, eps=1e-04, verbose=True
        )
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        COSLOSS = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        model_dict = {}
        
        for iter in range(epoches):
            model.train()
            with torch.cuda.amp.autocast():
                if self.signed_distance:
                    x_batch, q_batch, d, gt_grad = self.select_data_signed()
                else:
                    x_batch, q_batch, d, gt_grad = self.select_data()
                
                # Prepare inputs
                # x_batch: (batch_x, 3)
                # q_batch: (batch_q, DoF) or (batch_q, DoF+6)
                # d: (batch_x, batch_q)
                # gt_grad: (batch_x, batch_q, DoF) or (batch_x, batch_q, DoF+6)
                
                total_dof = DoF + 6 if self.use_base else DoF
                x_inputs = x_batch.unsqueeze(1).expand(-1, self.batch_q, -1).reshape(-1, 3)
                q_inputs = q_batch.unsqueeze(0).expand(self.batch_x, -1, -1).reshape(-1, total_dof)

                inputs = torch.cat([x_inputs, q_inputs], dim=-1)
                outputs = d.reshape(-1, 1)
                gt_grad = gt_grad.reshape(-1, total_dof)
                weights = torch.ones_like(outputs).to(self.device)

                # Forward pass - handle SIREN differently
                if self.network_type == 'siren':
                    d_pred, _ = model.forward(inputs)
                else:
                    d_pred = model.forward(inputs)
                
                d_grad_pred = torch.autograd.grad(
                    d_pred, q_inputs, torch.ones_like(d_pred), 
                    retain_graph=True, create_graph=True
                )[0]
                
                # Compute losses
                eikonal_loss = torch.abs(d_grad_pred.norm(2, dim=-1) - 1).mean()
                dd_grad_pred = torch.autograd.grad(
                    d_grad_pred, q_inputs, torch.ones_like(d_grad_pred), 
                    retain_graph=True, create_graph=True
                )[0]
                gradient_loss = (1 - COSLOSS(d_grad_pred, gt_grad)).mean()
                tension_loss = dd_grad_pred.square().sum(dim=-1).mean()
                d_loss = ((d_pred - outputs) ** 2 * weights).mean()

                # Combined loss
                w0 = 5.0
                w1 = 0.01
                w2 = 0.01
                w3 = 0.1
                loss = w0 * d_loss + w1 * eikonal_loss + w2 * tension_loss + w3 * gradient_loss

                # Backward pass
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step(loss)
                
                if iter % 10 == 0:
                    print(f"Epoch:{iter}\tMSE Loss: {d_loss.item():.3f}\t"
                          f"Eikonal Loss: {eikonal_loss.item():.3f}\t"
                          f"Tension Loss: {tension_loss.item():.3f}\t"
                          f"Gradient Loss: {gradient_loss.item():.3f}\t"
                          f"Total loss:{loss.item():.3f}\t"
                          f"Time: {time.strftime('%H:%M:%S', time.gmtime())}")
                    model_dict[iter] = model.state_dict()
                    
                    if self.writer is not None:
                        self.writer.add_scalar('loss/d_loss', d_loss.item(), iter)
                        self.writer.add_scalar('loss/eikonal_loss', eikonal_loss.item(), iter)
                        self.writer.add_scalar('loss/tension_loss', tension_loss.item(), iter)
                        self.writer.add_scalar('loss/gradient_loss', gradient_loss.item(), iter)
                        self.writer.add_scalar('loss/total_loss', loss.item(), iter)
                    
                    torch.save(model_dict, os.path.join(CUR_PATH, self.model_dict))
        
        return model
    
    def inference(self, x, q, model):
        """
        Inference with the trained model
        
        Args:
            x: (len(x), 3)
            q: (len(q), DoF) or (len(q), DoF+6)
            model: trained network
        Returns:
            cdf_pred: (len(x)*len(q), 1)
        """
        DoF = self.robot.serials[self.serial_idx].dof
        total_dof = DoF + 6 if self.use_base else DoF
        
        model.eval()
        x, q = x.to(self.device), q.to(self.device)
        x_cat = x.unsqueeze(1).expand(-1, len(q), -1).reshape(-1, 3)
        q_cat = q.unsqueeze(0).expand(len(x), -1, -1).reshape(-1, total_dof)
        inputs = torch.cat([x_cat, q_cat], dim=-1)
        
        if self.network_type == 'siren':
            cdf_pred, _ = model.forward(inputs)
        else:
            cdf_pred = model.forward(inputs)
            
        return cdf_pred
    
    def inference_d_wrt_q(self, x, q, model, return_grad=True):
        """
        Compute distance and gradient with respect to q
        """
        cdf_pred = self.inference(x, q, model)
        d = cdf_pred.abs().reshape(len(x), len(q)).min(dim=0)[0]
        
        if return_grad:
            DoF = self.robot.serials[self.serial_idx].dof
            total_dof = DoF + 6 if self.use_base else DoF
            # q_cat = q.unsqueeze(0).expand(len(x), -1, -1).reshape(-1, total_dof)
            grad = torch.autograd.grad(
                d, q, torch.ones_like(d), 
                retain_graph=True, create_graph=True
            )[0]
            return d, grad
        else:
            return d

    def eval_nn(self, model, num_iter=3):
        """
        Evaluate the trained model
        """
        eval_acc = True
        
        if eval_acc:
            bp_sdf = self.bp_sdf
            bp_sdf_model = torch.load(self.bp_sdf_model_path)

            res = []
            for i in range(1000):
                x = torch.rand(1, 3).to(self.device) - torch.tensor([[0.5, 0.5, 0]]).to(self.device)
                q = self.sample_q(batch_q=1000)
                for _ in range(num_iter):
                    d, grad = self.inference_d_wrt_q(x, q, model)
                    q = self.projection(q, d, grad)
                
                # Compute actual SDF
                d_actual = self.compute_sdf(x, q)
                
                error = d_actual.reshape(-1).abs()
                MAE = error.mean()
                RMSE = torch.sqrt(torch.mean(error ** 2))
                SR = (error < 0.03).sum().item() / len(error)
                res.append([MAE.item(), RMSE.item(), SR])
                print(f'iter {i} finished, MAE:{MAE}\tRMSE:{RMSE}\tSR:{SR}')
                
            res = np.array(res)
            print(f'MAE:{res[:, 0].mean()}\tRMSE:{res[:, 1].mean()}\tSR:{res[:, 2].mean()}')
            print(f'MAE:{res[:, 0].std()}\tRMSE:{res[:, 1].std()}\tSR:{res[:, 2].std()}')
    def my_eval_1(self,model):
        # model
        # input: [x,q] (B,3+DoF)
        # 在data中取出8个构成cube的点
        n_cube=8
        idx = torch.random.randint(0,len(self.data['x']),(n_cube,))
        cube_poses = self.data['x'][idx] 
        cube_ground_truth_q = self.data['q'][idx]
        # cube_poses=[[10,10,0]]
        for cube_points,gt_q in zip(cube_poses,cube_ground_truth_q):
        
            cube_edge = 0 # cube的边长
            print(f'data_x_shape:{self.data["x"].shape}')
            # cube_points  = torch.stack([self.data['x'][20*20*x+20*y+z] \
            #                                             for x in [cube_pos[0],cube_pos[0]+cube_edge] \
            #                                             for y in [cube_pos[1],cube_pos[1]+cube_edge]\
            #                                             for z in [cube_pos[2],cube_pos[2]+cube_edge]]).float().to(device)
            # cube_ground_truth_q  = torch.stack([self.data['q'][20*20*x+20*y+z] \
            #                                             for x in [cube_pos[0],cube_pos[0]+cube_edge] \
            #                                             for y in [cube_pos[1],cube_pos[1]+cube_edge]\
            #                                             for z in [cube_pos[2],cube_pos[2]+cube_edge]]).float().to(device)
            print(f'cube_points:{gt_q.shape}')
            serial = self.robot.serials[self.serial_idx]
            DoF = serial.dof
            bp_sdf = self.bp_sdf
            bdf_model = torch.load(self.bp_sdf_model_path)
            q_max = serial.theta_max[self.used_joints]
            q_min = serial.theta_min[self.used_joints]
            if self.use_base:
                base_min = serial.theta_min_base
                base_max = serial.theta_max_base
                q_max = torch.cat([q_max,base_max],dim=0)
                q_min = torch.cat([q_min,base_min],dim=0)
                DoF +=6
            
            # device
            self.device = device
            # 在DoF维度上采样test_sample_num个点
            test_sample_num = 1000
            q_sampled = q_min + torch.rand(test_sample_num,DoF).to(self.device)*(q_max-q_min)
            # 在关键代码段添加
            print(f"Allocated memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            print(f"Reserved memory: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
            q_sampled.requires_grad = True
            # 获得模型预测的距离和梯度
            print(f'q_sampled:{q_sampled.shape}')
            pred_d, pred_grad = self.inference_d_wrt_q(cube_points,q_sampled,model,return_grad = True)
            # 计算ground truth的距离和梯度
            from parallel_data_generator import DataGenerator
            data_generator = DataGenerator(self.device,self.robot,self.paths,serial_idx=self.serial_idx,with_base=self.use_base)
            data_generator.serial_idx = self.serial_idx
            gt_d = data_generator.distance_q(cube_points,q_sampled)
            # print(f'pred_d:{pred_d.shape}, gt_d:{gt_d.shape}')
            d_error = pred_d - gt_d
            # 把q_sampled, d_error, pred_grad, pred_d, gt_d reshape成(batch_size, DoF), 
            # 叠在一起，然后按照d_error的大小排序
            q_sampled = q_sampled.reshape(-1,DoF)
            d_error = d_error.reshape(-1,1)
            print(f'pred_grad:{pred_grad.shape}, pred_d:{pred_d.shape}, gt_d:{gt_d.shape}')
            pred_grad = pred_grad.reshape(-1,DoF)
            pred_d = pred_d.reshape(-1,1)
            gt_d = gt_d.reshape(-1,1)
            # 按照gt_d的大小排序
            sorted_idx = torch.argsort(gt_d,dim=0)
            sorted_idx = torch.linspace(0,gt_d.shape[0]-1,gt_d.shape[0],dtype=torch.long).to(self.device)
            q_sampled = q_sampled[sorted_idx]
            d_error = d_error[sorted_idx]
            pred_grad = pred_grad[sorted_idx]
            pred_grad_norm = torch.norm(pred_grad,dim=-1,keepdim=True)
            pred_d = pred_d[sorted_idx]
            gt_d = gt_d[sorted_idx]
            print(f'q_sampled:{q_sampled.shape}, d_error:{d_error.shape}, pred_grad:{pred_grad.shape}, pred_d:{pred_d.shape}, gt_d:{gt_d.shape}, pred_grad_norm:{pred_grad_norm.shape}')
            # 取前100个点，画出pred_d, gt_d, pred_grad, d_error的散点图，
            # 并保存到当前目录下的pred_gt_distance_gradient_scatter.png
            import matplotlib.pyplot as plt
            # import seaborn as sns
            # sns.set(style="whitegrid")
            plot_range = 1000
            plt.figure(figsize=(12, 8))
            plt.subplot(2, 2, 1)
            plt.scatter(gt_d[:plot_range].cpu().detach().numpy(), pred_d[:plot_range].cpu().detach().numpy(), c='blue', label='Predicted')
            plt.scatter(gt_d[:plot_range].cpu().detach().numpy(), gt_d[:plot_range].cpu().detach().numpy(), c='yellow', label='Ground Truth')
            plt.xlabel('Ground Truth Distance')
            plt.ylabel('Predicted Distance')
            plt.title('Predicted vs Ground Truth Distance')
            plt.legend()
            plt.subplot(2, 2, 2)
            plt.scatter(gt_d[:plot_range].cpu().detach().numpy(), pred_grad_norm[:plot_range].cpu().detach().numpy(), c='red', label='Predicted Gradient')
            plt.xlabel('Ground Truth Distance')
            plt.ylabel('Predicted Gradient')
            plt.title('Predicted Gradient')
            plt.legend()
            plt.subplot(2, 2, 3)
            plt.scatter(gt_d[:plot_range].cpu().detach().numpy(), d_error[:plot_range].cpu().detach().numpy(), c='green', label='Ground Truth vs Error')
            plt.xlabel('Ground Truth Distance')
            plt.ylabel('Distance Error')
            plt.title('Ground Truth vs Distance Error')
            plt.legend()
            plt.subplot(2, 2, 4)
            # ground truth distance 的直方图
            plt.hist(gt_d[:].squeeze(-1).cpu().detach().numpy(), bins=50, alpha=0.5, label='Ground Truth Distance', color='blue')
            # predicted distance 的直方图
            plt.hist(pred_d[:].squeeze(-1).cpu().detach().numpy(), bins=50, alpha=0.5, label='Predicted Distance', color='red')
            plt.xlabel('Distance')
            plt.ylabel('Frequency')
            plt.title('Distance Histogram')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(CUR_PATH,f'my_eval_{cube_points}.png'))
            plt.show()
            # 计算MAE和RMSE
            pred_d = pred_d.squeeze(-1).reshape(-1).cpu().detach().numpy()
            gt_d = gt_d.reshape(-1).cpu().detach().numpy()
            pred_grad = pred_grad.reshape(-1,DoF).cpu().detach().numpy()
            MAE_d = np.mean(np.abs(pred_d - gt_d))
            RMSE_d = np.sqrt(np.mean((pred_d - gt_d)**2))
            print(f'MAE_d: {MAE_d}, RMSE_d: {RMSE_d}')
    def my_eval_2(self,model):
        # model
        # input: [x,q] (B,3+DoF)
        # 在data中取出8个构成cube的点
        from parallel_data_generator import DataGenerator
        data_generator = DataGenerator(self.device,self.robot,self.paths,serial_idx=self.serial_idx,with_base=self.use_base)
        data_generator.serial_idx = self.serial_idx
        n_cube = 8
        idx = torch.randint(0,len(self.data['x']),(n_cube,))
        cube_poses = self.data['x'][idx] 
        cube_ground_truth_q = self.data['q'][idx]
        # cube_poses=[[10,10,0]]
        for cube_points, gt_q in zip(cube_poses,cube_ground_truth_q):
        
            # cube_edge = 0 # cube的边长
            # cube_points  = torch.stack([self.data['x'][20*20*x+20*y+z] \
            #                                             for x in [cube_pos[0],cube_pos[0]+cube_edge] \
            #                                             for y in [cube_pos[1],cube_pos[1]+cube_edge]\
            #                                             for z in [cube_pos[2],cube_pos[2]+cube_edge]]).float().to(device)
            # cube_ground_truth_q  = torch.stack([self.data['q'][20*20*x+20*y+z] \
            #                                             for x in [cube_pos[0],cube_pos[0]+cube_edge] \
            #                                             for y in [cube_pos[1],cube_pos[1]+cube_edge]\
            #                                             for z in [cube_pos[2],cube_pos[2]+cube_edge]]).float().to(device)
            # cube_points:(N,3) (N=sample_num)
            # cube_ground_truth_q:(N,100,DoF,DoF)
            cube_points = cube_points.float().to(device).unsqueeze(0)
            serial = self.robot.serials[self.serial_idx]
            DoF = serial.dof
            bp_sdf = self.bp_sdf
            bdf_model = torch.load(self.bp_sdf_model_path)
            q_max = serial.theta_max[self.used_joints]
            q_min = serial.theta_min[self.used_joints]
            # 在DoF维度上采样test_sample_num个点
            test_sample_num = 100
            if self.use_base:
                base_min = serial.theta_min_base
                base_max = serial.theta_max_base
                q_max = torch.cat([q_max,base_max],dim=0)
                q_min = torch.cat([q_min,base_min],dim=0)
                DoF +=6
            q_sampled = torch.rand(DoF).to(self.device).unsqueeze(0).expand(test_sample_num,-1) * (q_max-q_min) + q_min
            if self.robot.robot == 'panda':
                eval_joint_idx = torch.randint(5,DoF,(1,)).to(self.device) 
            else:
                eval_joint_idx = torch.randint(0,DoF,(1,)).to(self.device)
            # 在5-6之间采样一个整数作为评估的关节（因为前面几个关节对末端影响较大，后面几个关节可能过于平滑）
            print(f'eval_joint_idx:{eval_joint_idx}')
            q_sampled[:,eval_joint_idx] = torch.linspace(\
            q_min[eval_joint_idx].item(), q_max[eval_joint_idx].item(), test_sample_num).to(self.device).unsqueeze(-1)
            q_sampled.requires_grad = True
            
            # 获得模型预测的距离和梯度my_eval_2
            pred_d, pred_grad = self.inference_d_wrt_q(cube_points,q_sampled,model,return_grad = True)
            
            print(f'pred_d_min:{pred_d.min()}, pred_d_max:{pred_d.max()}, pred_d_mean:{pred_d.mean()}')
            # 计算ground truth的距离和梯度
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            gt_d = data_generator.distance_q(cube_points,q_sampled)
            print(f'gt_d.min:{gt_d.min()}, gt_d.max:{gt_d.max()},gt_d.mean:{gt_d.mean()}')
            # print(f'pred_d:{pred_d.shape}, gt_d:{gt_d.shape}')
            d_error = pred_d - gt_d
            # 把q_sampled, d_error, pred_grad, pred_d, gt_d reshape成(batch_size, DoF), 
            # 叠在一起，然后按照d_error的大小排序
            q_sampled = q_sampled.reshape(-1,DoF)
            d_error = d_error.reshape(-1,1)
            # print(f'pred_grad:{pred_grad.shape}, pred_d:{pred_d.shape}, gt_d:{gt_d.shape}')
            pred_grad = pred_grad.reshape(-1,DoF)
            pred_d = pred_d.reshape(-1,1)
            gt_d = gt_d.reshape(-1,1)
            
            sorted_idx = torch.linspace(0,gt_d.shape[0]-1,gt_d.shape[0],dtype=torch.long).to(self.device)
            q_sampled = q_sampled[sorted_idx]
            d_error = d_error[sorted_idx]
            pred_grad = pred_grad[sorted_idx]
            pred_grad_norm = torch.norm(pred_grad,dim=-1,keepdim=True)
            pred_d = pred_d[sorted_idx]
            gt_d = gt_d[sorted_idx]
            configuration = (sorted_idx * q_min[eval_joint_idx] + (1 - sorted_idx) * q_max[eval_joint_idx])/len(sorted_idx)
            configuration = configuration.cpu().detach().numpy()
            import matplotlib.pyplot as plt
            # import seaborn as sns
            # sns.set(style="whitegrid")
            plot_range = 1000
            plt.figure(figsize=(12, 8))
            plt.subplot(2, 2, 1)
            plt.scatter(configuration, pred_d[:plot_range].cpu().detach().numpy(), c='blue', label='Predicted')
            plt.scatter(configuration, gt_d[:plot_range].cpu().detach().numpy(), c='yellow', label='Ground Truth')
            plt.xlabel('configuration')
            plt.ylabel('Predicted Distance')
            # y轴范围从0开始
            plt.ylim(bottom=0)
            plt.title('Predicted vs Ground Truth Distance 1')
            plt.legend()
            plt.subplot(2, 2, 2)
            plt.scatter(configuration, pred_d[:plot_range].cpu().detach().numpy(), c='blue', label='Predicted')
            plt.scatter(configuration, gt_d[:plot_range].cpu().detach().numpy(), c='yellow', label='Ground Truth')
            plt.xlabel('configuration')
            plt.ylabel('Predicted Distance')
            plt.title('Predicted vs Ground Truth Distance 2')
            plt.legend()
            plt.subplot(2, 2, 3)
            plt.scatter(configuration, d_error[:plot_range].cpu().detach().numpy(), c='green', label='Ground Truth vs Error')
            plt.xlabel('configuration')
            plt.ylabel('Distance Error')
            plt.title('Ground Truth vs Distance Error')
            plt.legend()
            plt.subplot(2, 2, 4)
            # gradient 的散点图
            plt.scatter(configuration, pred_grad_norm[:plot_range].cpu().detach().numpy(), c='red', label='Predicted Gradient')
            plt.xlabel('configuration')
            plt.ylabel('Predicted Gradient')
            plt.title('Predicted Gradient')
            plt.legend()
            plt.tight_layout()
            # 在图上添加文字
            text = f'Joint {eval_joint_idx.item()} Evaluation'+\
                f'\nCube Position: {cube_points}' + \
                    f"q Sampled: {q_sampled[0].cpu().detach().numpy()}"
            plt.text(0.4, 0.95, text, horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes, fontsize=10)
            save_path = os.path.join(CUR_PATH,f"{eval_joint_idx.item()}_{self.paths['model_dict'].split('.')[0].split('/')[-1]}")
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            plt.savefig(os.path.join(save_path,f"slice_{cube_points.cpu().detach().numpy()}.png"))
            # plt.show()
            # 计算MAE和RMSE
            pred_d = pred_d.squeeze(-1).reshape(-1).cpu().detach().numpy()
            gt_d = gt_d.reshape(-1).cpu().detach().numpy()
            pred_grad = pred_grad.reshape(-1,DoF).cpu().detach().numpy()
            MAE_d = np.mean(np.abs(pred_d - gt_d))
            RMSE_d = np.sqrt(np.mean((pred_d - gt_d)**2))
            print(f'MAE_d: {MAE_d}, RMSE_d: {RMSE_d}')   

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CDF Model Training with Base DoF Support')
    parser.add_argument('--data_path', type=str, default='data_with_base_dof_0.npy', 
                       help='Path to the raw data file')
    parser.add_argument('--with_writer', action='store_true', 
                       help='Whether to use TensorBoard writer')
    parser.add_argument('--eval', action='store_true', 
                       help='Whether to evaluate the model')
    parser.add_argument('--train', action='store_true', 
                       help='Whether to train the model')
    parser.add_argument('--epoches', type=int, default=50000, 
                       help='Number of training epochs')
    parser.add_argument('--batch_x', type=int, default=10, 
                       help='Batch size for x')
    parser.add_argument('--batch_q', type=int, default=100, 
                       help='Batch size for q')
    parser.add_argument('--signed_distance', action='store_true', 
                       help='Whether to use signed distance')
    parser.add_argument('--max_q_per_link', type=int, default=100, 
                       help='Maximum number of q samples per link')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                       help='Device to use for training/evaluation')
    parser.add_argument('--model_dict', type=str, default='leaphand_base_mlp.pt', 
                       help='Path to save/load the model dictionary')
    parser.add_argument('--robot', type=str, default='leaphand', 
                       help='Robot type', choices=['panda', 'dexhand', 'leaphand'])
    parser.add_argument('--serial_idx', type=int, default=0, 
                       help='Serial index for different fingers')
    parser.add_argument('--network_type', type=str, default='mlp', 
                       choices=['mlp', 'siren'], help='Type of neural network to use')
    parser.add_argument('--use_base', action='store_true', 
                       help='Whether to include 6DoF base in configuration space')
    
    args = parser.parse_args()
    print(f'args:{args}')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    hprams = {
        'batch_x': args.batch_x,
        'batch_q': args.batch_q,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'signed_distance': args.signed_distance,
        'max_q_per_link': args.max_q_per_link,
        'epoches': args.epoches,
        'optimizer': 'Adam',
        'learning_rate': 0.001,
        'scheduler': 'ReduceLROnPlateau',
        'd_loss': 5.0,
        'eikonal_loss': 0.01,
        'tension_loss': 0.01,
        'gradient_loss': 0.1,
        'network_type': args.network_type,
        'use_base': args.use_base
    }
    
    CUR_DIR = os.path.dirname(os.path.abspath(__file__))
    paths = {
        'urdf': os.path.join(CUR_DIR, f'../../RDF/descriptions/{args.robot}/*.urdf'),
        'meshes': os.path.join(CUR_DIR, f'../../RDF/descriptions/{args.robot}/meshes/*.stl'),
        'points': os.path.join(CUR_DIR, f'../../RDF/data/{args.robot}/sdf_points/'),
        'model': os.path.join(CUR_DIR, f'../../RDF/models/{args.robot}/BP_8.pt'),
        'raw_data': os.path.join(CUR_DIR, f'data/{args.robot}/{args.data_path}'),
        'model_dict': os.path.join(CUR_DIR, f'model_dict/{args.robot}/{args.model_dict}'),
    }
    
    if args.with_writer:
        from torch.utils.tensorboard import SummaryWriter
        i = 0
        run_name = f'{args.robot}_cdf_v2_{args.network_type}_base{args.use_base}_{i}'
        while os.path.exists(os.path.join(CUR_PATH, f'runs/{run_name}')):
            i += 1
            run_name = f'{args.robot}_cdf_v2_{args.network_type}_base{args.use_base}_{i}'
        
        writer = SummaryWriter(os.path.join(CUR_PATH, f'runs/{run_name}'))
        print(f'writer path: {os.path.join(CUR_PATH, f"runs/{run_name}")}')
        
        writer.add_text('info', f'CDF V2 training with {args.network_type.upper()} network')
        writer.add_hparams(hparam_dict=hprams, metric_dict={})
        cdf = CDF_V2(device, paths=paths, robot=args.robot, network_type=args.network_type,
                     writer=writer, signed_distance=args.signed_distance, 
                     serial_idx=args.serial_idx, use_base=args.use_base)
    else:
        cdf = CDF_V2(device, paths=paths, robot=args.robot, network_type=args.network_type,
                     writer=None, signed_distance=args.signed_distance, 
                     serial_idx=args.serial_idx, use_base=args.use_base)
    
    print(f"Serial chain links: {cdf.robot.serials[cdf.serial_idx].all_links}")
    print(f"Network type: {args.network_type}")
    print(f"Use base: {args.use_base}")
    
    if args.train:
        cdf.batch_x = args.batch_x
        cdf.batch_q = args.batch_q
        cdf.max_q_per_link = args.max_q_per_link
        cdf.train_nn(epoches=hprams['epoches'])
    
    if args.eval:
        DoF = cdf.robot.serials[cdf.serial_idx].dof
        input_dims = 3 + DoF + (6 if args.use_base else 0)
        model = cdf.create_network(input_dims)
        key = max(torch.load(paths['model_dict']).keys())
        model.load_state_dict(torch.load(paths['model_dict'])[key])
        model.to(device)
        cdf.my_eval_2(model)
