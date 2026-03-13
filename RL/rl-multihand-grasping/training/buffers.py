"""
buffers.py
经验回放缓冲区
"""

import numpy as np
import torch
from typing import Dict, Tuple, Generator, Optional


class RolloutBuffer:
    """
    Rollout 缓冲区
    
    存储训练数据用于 PPO 更新
    """
    
    def __init__(
        self,
        buffer_size: int,
        num_envs: int,
        pointcloud_shape: Tuple[int, int],
        q_dim: int,
        action_dim: int,
        device: str = "cuda",
        gamma: float = 0.99,
        gae_lambda: float = 0.95
    ):
        """
        初始化缓冲区
        
        Args:
            buffer_size: 缓冲区大小（每个环境的步数）
            num_envs: 并行环境数
            pointcloud_shape: 点云形状 (N, 4)
            q_dim: 关节维度
            action_dim: 动作维度
            device: 计算设备
            gamma: 折扣因子
            gae_lambda: GAE lambda
        """
        self.buffer_size = buffer_size
        self.num_envs = num_envs
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        # 分配存储空间
        self.pointclouds = np.zeros(
            (buffer_size, num_envs, *pointcloud_shape), dtype=np.float32
        )
        self.qs = np.zeros(
            (buffer_size, num_envs, q_dim), dtype=np.float32
        )
        self.actions = np.zeros(
            (buffer_size, num_envs, action_dim), dtype=np.float32
        )
        self.rewards = np.zeros(
            (buffer_size, num_envs), dtype=np.float32
        )
        self.dones = np.zeros(
            (buffer_size, num_envs), dtype=np.float32
        )
        self.values = np.zeros(
            (buffer_size, num_envs), dtype=np.float32
        )
        self.log_probs = np.zeros(
            (buffer_size, num_envs), dtype=np.float32
        )
        
        # GAE 计算结果
        self.advantages = np.zeros(
            (buffer_size, num_envs), dtype=np.float32
        )
        self.returns = np.zeros(
            (buffer_size, num_envs), dtype=np.float32
        )
        
        self.ptr = 0
        self.full = False
    
    def add(
        self,
        pointcloud: np.ndarray,
        q: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        value: np.ndarray,
        log_prob: np.ndarray
    ) -> None:
        """
        添加一步数据
        
        Args:
            pointcloud: (num_envs, N, 4)
            q: (num_envs, q_dim)
            action: (num_envs, action_dim)
            reward: (num_envs,)
            done: (num_envs,)
            value: (num_envs,)
            log_prob: (num_envs,)
        """
        self.pointclouds[self.ptr] = pointcloud
        self.qs[self.ptr] = q
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        
        self.ptr += 1
        if self.ptr >= self.buffer_size:
            self.full = True
            self.ptr = 0
    
    def compute_returns_and_advantages(
        self,
        last_values: np.ndarray,
        last_dones: np.ndarray
    ) -> None:
        """
        计算 GAE 优势和回报
        
        Args:
            last_values: 最后一步的 value 估计
            last_dones: 最后一步的 done 标志
        """
        last_gae = 0
        
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - last_dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.dones[step + 1]
                next_values = self.values[step + 1]
            
            delta = (
                self.rewards[step] + 
                self.gamma * next_values * next_non_terminal - 
                self.values[step]
            )
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            self.advantages[step] = last_gae
        
        self.returns = self.advantages + self.values
    
    def get(
        self,
        batch_size: int
    ) -> Generator[Dict[str, torch.Tensor], None, None]:
        """
        获取 mini-batch 数据
        
        Args:
            batch_size: batch 大小
            
        Yields:
            batch: 数据 batch
        """
        total_size = self.buffer_size * self.num_envs
        indices = np.random.permutation(total_size)
        
        # 展平数据
        pointclouds = self.pointclouds.reshape(-1, *self.pointclouds.shape[2:])
        qs = self.qs.reshape(-1, self.qs.shape[-1])
        actions = self.actions.reshape(-1, self.actions.shape[-1])
        values = self.values.reshape(-1)
        log_probs = self.log_probs.reshape(-1)
        advantages = self.advantages.reshape(-1)
        returns = self.returns.reshape(-1)
        
        for start in range(0, total_size, batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]
            
            yield {
                "pointcloud": torch.from_numpy(pointclouds[batch_indices]).to(self.device),
                "q": torch.from_numpy(qs[batch_indices]).to(self.device),
                "actions": torch.from_numpy(actions[batch_indices]).to(self.device),
                "old_values": torch.from_numpy(values[batch_indices]).to(self.device),
                "old_log_probs": torch.from_numpy(log_probs[batch_indices]).to(self.device),
                "advantages": torch.from_numpy(advantages[batch_indices]).to(self.device),
                "returns": torch.from_numpy(returns[batch_indices]).to(self.device),
            }
    
    def reset(self) -> None:
        """重置缓冲区"""
        self.ptr = 0
        self.full = False


class ReplayBuffer:
    """
    经验回放缓冲区（可选，用于 off-policy 方法）
    """
    
    def __init__(
        self,
        capacity: int,
        pointcloud_shape: Tuple[int, int],
        q_dim: int,
        action_dim: int,
        device: str = "cuda"
    ):
        """
        初始化回放缓冲区
        
        Args:
            capacity: 缓冲区容量
            pointcloud_shape: 点云形状
            q_dim: 关节维度
            action_dim: 动作维度
            device: 计算设备
        """
        self.capacity = capacity
        self.device = device
        
        self.pointclouds = np.zeros((capacity, *pointcloud_shape), dtype=np.float32)
        self.qs = np.zeros((capacity, q_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_pointclouds = np.zeros((capacity, *pointcloud_shape), dtype=np.float32)
        self.next_qs = np.zeros((capacity, q_dim), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        
        self.ptr = 0
        self.size = 0
    
    def add(
        self,
        pointcloud: np.ndarray,
        q: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_pointcloud: np.ndarray,
        next_q: np.ndarray,
        done: bool
    ) -> None:
        """添加经验"""
        self.pointclouds[self.ptr] = pointcloud
        self.qs[self.ptr] = q
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_pointclouds[self.ptr] = next_pointcloud
        self.next_qs[self.ptr] = next_q
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """随机采样"""
        indices = np.random.choice(self.size, batch_size, replace=False)
        
        return {
            "pointcloud": torch.from_numpy(self.pointclouds[indices]).to(self.device),
            "q": torch.from_numpy(self.qs[indices]).to(self.device),
            "actions": torch.from_numpy(self.actions[indices]).to(self.device),
            "rewards": torch.from_numpy(self.rewards[indices]).to(self.device),
            "next_pointcloud": torch.from_numpy(self.next_pointclouds[indices]).to(self.device),
            "next_q": torch.from_numpy(self.next_qs[indices]).to(self.device),
            "dones": torch.from_numpy(self.dones[indices]).to(self.device),
        }
    
    def __len__(self) -> int:
        return self.size
