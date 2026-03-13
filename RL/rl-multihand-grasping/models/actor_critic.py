"""
actor_critic.py
Actor-Critic 网络
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Dict, Tuple, Optional
import numpy as np

from .pointnet_encoder import PointNetEncoder
from .joint_encoder import JointEncoder


class ActorCritic(nn.Module):
    """
    Actor-Critic 网络
    
    组合 PointNet 点云编码器 和 关节编码器
    
    输出:
    - Actor: mean + log_std (22-dim)
    - Critic: V(s)
    """
    
    def __init__(self, config: dict):
        """
        初始化 Actor-Critic 网络
        
        Args:
            config: 网络配置
        """
        super().__init__()
        
        network_config = config["network"]
        
        # PointNet 编码器
        pointnet_config = network_config["pointnet"]
        self.pointnet = PointNetEncoder(
            input_dim=pointnet_config["input_dim"],
            hidden_dims=pointnet_config["hidden_dims"],
            output_dim=pointnet_config["output_dim"],
            use_batch_norm=pointnet_config["use_batch_norm"]
        )
        
        # 关节编码器
        joint_config = network_config["joint_encoder"]
        self.joint_encoder = JointEncoder(
            input_dim=joint_config["input_dim"],
            hidden_dims=joint_config["hidden_dims"],
            output_dim=joint_config["output_dim"]
        )
        
        # 特征融合维度
        feature_dim = pointnet_config["output_dim"] + joint_config["output_dim"]
        
        # Actor 网络
        actor_config = network_config["actor"]
        self.actor = self._build_mlp(
            feature_dim,
            actor_config["hidden_dims"],
            actor_config["output_dim"]
        )
        
        # Actor 的 log_std（可学习）
        self.log_std = nn.Parameter(
            torch.zeros(actor_config["output_dim"])
        )
        self.log_std_min = actor_config["log_std_min"]
        self.log_std_max = actor_config["log_std_max"]
        
        # Critic 网络
        critic_config = network_config["critic"]
        self.critic = self._build_mlp(
            feature_dim,
            critic_config["hidden_dims"],
            critic_config["output_dim"]
        )
        
        # 输出维度
        self.action_dim = actor_config["output_dim"]
        
        # 初始化权重
        self._init_weights()
    
    def _build_mlp(
        self,
        input_dim: int,
        hidden_dims: list,
        output_dim: int
    ) -> nn.Sequential:
        """构建 MLP"""
        dims = [input_dim] + hidden_dims + [output_dim]
        layers = []
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.LayerNorm(dims[i+1]))
                layers.append(nn.ReLU())
        
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Actor 最后一层使用小的初始化
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        
        # Critic 最后一层
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)
    
    def forward_features(
        self,
        pointcloud: torch.Tensor,
        q: torch.Tensor
    ) -> torch.Tensor:
        """
        提取观测特征
        
        Args:
            pointcloud: (B, N, 4) 点云
            q: (B, 22) 关节位置
            
        Returns:
            features: (B, feature_dim) 融合特征
        """
        # 编码点云
        pointcloud_feat = self.pointnet(pointcloud)
        
        # 编码关节
        joint_feat = self.joint_encoder(q)
        
        # 特征融合
        features = torch.cat([pointcloud_feat, joint_feat], dim=1)
        
        return features
    
    def forward(
        self,
        obs: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            obs: 观测字典 {"pointcloud": ..., "q": ...}
            
        Returns:
            action: 采样的动作
            log_prob: 动作的对数概率
            value: 状态价值
        """
        pointcloud = obs["pointcloud"]
        q = obs["q"]
        
        # 提取特征
        features = self.forward_features(pointcloud, q)
        
        # Actor: 计算动作分布
        action_mean = self.actor(features)
        log_std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
        action_std = torch.exp(log_std)
        
        # 采样动作
        dist = Normal(action_mean, action_std)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        # Critic: 计算价值
        value = self.critic(features)
        
        return action, log_prob, value.squeeze(-1)
    
    def act(
        self,
        obs: Dict[str, torch.Tensor],
        deterministic: bool = False
    ) -> torch.Tensor:
        """
        选择动作
        
        Args:
            obs: 观测字典
            deterministic: 是否确定性策略
            
        Returns:
            action: 动作
        """
        pointcloud = obs["pointcloud"]
        q = obs["q"]
        
        # 提取特征
        features = self.forward_features(pointcloud, q)
        
        # 计算动作
        action_mean = self.actor(features)
        
        if deterministic:
            return action_mean
        
        log_std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
        action_std = torch.exp(log_std)
        
        dist = Normal(action_mean, action_std)
        action = dist.sample()
        
        return action
    
    def value(
        self,
        obs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        计算状态价值 V(s)
        
        Args:
            obs: 观测字典
            
        Returns:
            value: 状态价值
        """
        pointcloud = obs["pointcloud"]
        q = obs["q"]
        
        # 提取特征
        features = self.forward_features(pointcloud, q)
        
        # Critic
        value = self.critic(features)
        
        return value.squeeze(-1)
    
    def evaluate_actions(
        self,
        obs: Dict[str, torch.Tensor],
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        评估给定动作
        
        Args:
            obs: 观测字典
            actions: 动作
            
        Returns:
            log_prob: 动作对数概率
            entropy: 熵
            value: 状态价值
        """
        pointcloud = obs["pointcloud"]
        q = obs["q"]
        
        # 提取特征
        features = self.forward_features(pointcloud, q)
        
        # Actor
        action_mean = self.actor(features)
        log_std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
        action_std = torch.exp(log_std)
        
        dist = Normal(action_mean, action_std)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        # Critic
        value = self.critic(features)
        
        return log_prob, entropy, value.squeeze(-1)
    
    def get_action_distribution(
        self,
        obs: Dict[str, torch.Tensor]
    ) -> Normal:
        """
        获取动作分布
        
        Args:
            obs: 观测字典
            
        Returns:
            dist: 动作分布
        """
        pointcloud = obs["pointcloud"]
        q = obs["q"]
        
        features = self.forward_features(pointcloud, q)
        
        action_mean = self.actor(features)
        log_std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
        action_std = torch.exp(log_std)
        
        return Normal(action_mean, action_std)
