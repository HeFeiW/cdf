# -----------------------------------------------------------------------------
# DataLoader for CVAE training on contact configuration distribution
# -----------------------------------------------------------------------------

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

class CDFDataset(Dataset):
    """
    Dataset for learning p(Δq_contact | q_0, x_idx)
    
    Data format:
        - tensor_data: shape (nbData, nbData, max_q_per_x, num_joints)
        - Each grid point x_idx has a set of contact configurations Q_idx
    
    Sampling strategy A-q: 距离放缩采样 (Distance-scaled sampling)
        给定 q_0 和候选集合 Q_idx:
        1. q_min = argmin_{q_i in Q_idx} ||q_i - q_0||
        2. q_contact ~ {q_i | ||q_i - q_min|| <= (1+gamma)||q_min - q_0||}
    Sampling strategy A-d: 距离放缩采样 (Distance-scaled sampling)
        给定 q_0 和候选集合 Q_idx:
        1. q_min = argmin_{q_i in Q_idx} ||q_i - q_0||
        2. q_contact ~ {q_i | ||q_i - q_min|| <= (1+gamma)||q_min - q_0||}
        3. d = ||q_contact - q_0||
    """
    
    def __init__(self, data_path, num_joints=2, gamma=0.5, samples_per_grid=10, 
                 q_min=-np.pi, q_max=np.pi, task_space=[[-4.0,-4.0],[4.0,4.0]],sampling_strategy='A-q'):
        """
        Args:
            data_path: path to .pt file containing tensor_data
            num_joints: dimension of configuration space
            gamma: scaling factor for distance threshold (default: 0.5)
            samples_per_grid: number of (q_0, q_contact) samples per grid point
            q_min, q_max: configuration space bounds
            task_space: workspace bounds for x_idx
            sampling_strategy: 'A-q' or 'A-d' (currently only 'A-q' implemented)
        """
        super().__init__()
        
        # Load data and move to CPU (device handling done in training loop)
        self.tensor_data = torch.load(data_path).cpu()  # (nbData, nbData, max_q_per_x, num_joints)
        self.num_joints = num_joints
        self.gamma = gamma
        self.samples_per_grid = samples_per_grid
        self.q_min = q_min
        self.q_max = q_max
        self.task_space = task_space
        self.sampling_strategy = sampling_strategy
        
        # Data dimensions
        self.nbData_x, self.nbData_y, self.max_q_per_x, _ = self.tensor_data.shape
        
        # Generate grid coordinates for x_idx
        x = torch.linspace(self.task_space[0][0], self.task_space[1][0], self.nbData_x)
        y = torch.linspace(self.task_space[0][1], self.task_space[1][1], self.nbData_y)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        self.grid_coords = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)  # (nbData_x*nbData_y, 2)
        
        # Pre-compute valid samples
        print("Pre-computing valid training samples...")
        self.samples = []
        self._precompute_samples()
        print(f"Total valid samples: {len(self.samples)}")
        
    def _precompute_samples(self):
        """
        Pre-compute all valid (x_idx, q_0, q_contact, Δq) tuples
        """
        for i in range(self.nbData_x):
            for j in range(self.nbData_y):
                # Get all valid contact configs for this grid point
                Q_idx = self.tensor_data[i, j]  # (max_q_per_x, num_joints)
                valid_mask = Q_idx[:, 0] != torch.inf
                Q_idx_valid = Q_idx[valid_mask]  # (M, num_joints)
                if len(Q_idx_valid) == 0:
                    continue
                
                # Get grid coordinate
                grid_idx = i * self.nbData_y + j
                x_idx = self.grid_coords[grid_idx]  # (2,)
                
                # Sample multiple q_0 for this grid point
                for _ in range(self.samples_per_grid):
                    # Randomly sample q_0 from configuration space (on CPU)
                    q_0 = torch.rand(self.num_joints) * (self.q_max - self.q_min) + self.q_min
                    
                    # Apply distance-scaled sampling
                    q_contact = self._distance_scaled_sampling(q_0, Q_idx_valid)
                    
                    if q_contact is not None:
                        # Compute Δq
                        delta_q = q_contact - q_0
                        # if self.sampling_strategy == 'A-d':
                        #     # Append distance as additional feature
                        #     d = torch.norm(delta_q).unsqueeze(0)  # (1,)
                        #     delta_q = torch.cat([delta_q, d], dim=0)  # (num_joints + 1,)
                        # Store sample
                        self.samples.append({
                            'x_idx': x_idx,
                            'q_0': q_0,
                            'q_contact': q_contact,
                            'delta_q': delta_q
                        })

    
    def _distance_scaled_sampling(self, q_0, Q_idx):
        """
        距离放缩采样策略
        
        Args:
            q_0: query configuration (num_joints,)
            Q_idx: candidate contact configurations (M, num_joints)
        
        Returns:
            q_contact: sampled contact configuration or None if no valid samples
        """
        # Compute distances from q_0 to all candidates
        distances = torch.norm(Q_idx - q_0.unsqueeze(0), dim=-1)  # (M,)
        
        # Find nearest configuration
        min_dist, min_idx = torch.min(distances, dim=0)
        q_min = Q_idx[min_idx]
        
        # Define threshold
        threshold = (1 + self.gamma) * min_dist
        
        # Filter candidates within threshold
        valid_mask = distances <= threshold
        valid_candidates = Q_idx[valid_mask]
        
        if len(valid_candidates) == 0:
            return None
        
        # Randomly sample one from valid candidates
        sample_idx = torch.randint(0, len(valid_candidates), (1,)).item()
        q_contact = valid_candidates[sample_idx]
        
        return q_contact
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Returns:
            condition: [q_0, x_idx] concatenated (num_joints + 2,)
            delta_q: target output (num_joints,)
        """
        sample = self.samples[idx]
        
        # Construct condition c = [q_0, x_idx] (ensure CPU and detached)
        condition = torch.cat([sample['q_0'].cpu(), sample['x_idx'].cpu()], dim=0)
        delta_q = sample['delta_q'].cpu()
        
        return condition, delta_q


def get_dataloader(data_path, batch_size=64, num_joints=2, gamma=0.5, 
                   samples_per_grid=10, shuffle=True, num_workers=0,
                     sampling_strategy='A-q'):
    """
    Create DataLoader for CVAE training
    
    Args:
        data_path: path to .pt file
        batch_size: batch size for training
        num_joints: dimension of configuration space
        gamma: scaling factor for distance-scaled sampling
        samples_per_grid: number of samples per grid point
        shuffle: whether to shuffle data
        num_workers: number of workers for data loading
    
    Returns:
        DataLoader instance
    """
    dataset = CDFDataset(
        data_path=data_path,
        num_joints=num_joints,
        gamma=gamma,
        samples_per_grid=samples_per_grid,
        sampling_strategy=sampling_strategy
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader


if __name__ == "__main__":
    # Test the dataloader
    data_path = "../2Dexamples/data22.pt"
    
    if os.path.exists(data_path):
        print(f"Loading data from {data_path}")
        dataloader = get_dataloader(data_path, batch_size=32, samples_per_grid=5)
        
        print(f"\nDataset size: {len(dataloader.dataset)}")
        print(f"Number of batches: {len(dataloader)}")
        
        # Test one batch
        for condition, delta_q in dataloader:
            print(f"\nBatch shape:")
            print(f"  Condition (q_0 + x_idx): {condition.shape}")
            print(f"  Delta q: {delta_q.shape}")
            print(f"\nSample values:")
            print(f"  Condition[0]: {condition[0]}")
            print(f"  Delta_q[0]: {delta_q[0]}")
            break
    else:
        print(f"Data file not found: {data_path}")
        print("Please run generate_data() in cdf.py first")