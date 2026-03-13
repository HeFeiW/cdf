"""
pointnet_encoder.py
PointNet 点云编码器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class TNet(nn.Module):
    """
    Transformation Network
    用于学习点云的空间变换
    """
    
    def __init__(self, k: int = 3):
        """
        初始化 T-Net
        
        Args:
            k: 输入/输出维度
        """
        super().__init__()
        self.k = k
        
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: (B, k, N) 输入点云
            
        Returns:
            transform: (B, k, k) 变换矩阵
        """
        batch_size = x.shape[0]
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        x = torch.max(x, 2)[0]  # 全局最大池化
        
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        
        # 添加单位矩阵
        identity = torch.eye(self.k, device=x.device).view(1, self.k * self.k).repeat(batch_size, 1)
        x = x + identity
        x = x.view(-1, self.k, self.k)
        
        return x


class PointNetEncoder(nn.Module):
    """
    PointNet 点云编码器
    
    输入: (N, 4) 点云 (x, y, z, label)
    输出: (256,) 特征向量
    """
    
    def __init__(
        self,
        input_dim: int = 4,
        hidden_dims: List[int] = [64, 128, 256],
        output_dim: int = 256,
        use_batch_norm: bool = True,
        use_tnet: bool = False
    ):
        """
        初始化 PointNet 编码器
        
        Args:
            input_dim: 输入点维度 (默认 4: x, y, z, label)
            hidden_dims: 隐藏层维度
            output_dim: 输出特征维度
            use_batch_norm: 是否使用 BatchNorm
            use_tnet: 是否使用 T-Net
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_tnet = use_tnet
        
        # 可选的 T-Net
        if use_tnet:
            self.tnet = TNet(k=input_dim)
        
        # 共享 MLP 层
        dims = [input_dim] + hidden_dims
        layers = []
        
        for i in range(len(dims) - 1):
            layers.append(nn.Conv1d(dims[i], dims[i+1], 1))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(dims[i+1]))
            layers.append(nn.ReLU())
        
        self.shared_mlp = nn.Sequential(*layers)
        
        # 最终全连接层
        self.fc = nn.Sequential(
            nn.Linear(hidden_dims[-1], output_dim),
            nn.BatchNorm1d(output_dim) if use_batch_norm else nn.Identity(),
            nn.ReLU()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: (B, N, input_dim) 点云
            
        Returns:
            features: (B, output_dim) 全局特征
        """
        # 转换为 (B, input_dim, N)
        x = x.transpose(1, 2)
        
        # 可选的空间变换
        if self.use_tnet:
            transform = self.tnet(x)
            x = x.transpose(1, 2)
            x = torch.bmm(x, transform)
            x = x.transpose(1, 2)
        
        # 共享 MLP
        x = self.shared_mlp(x)
        
        # 全局最大池化
        x = torch.max(x, 2)[0]
        
        # 最终映射
        features = self.fc(x)
        
        return features
    
    def get_output_dim(self) -> int:
        """获取输出维度"""
        return self.output_dim


class PointNetPlusPlus(nn.Module):
    """
    简化版 PointNet++ (可选替换)
    
    使用 Set Abstraction 层进行层次化特征提取
    """
    
    def __init__(
        self,
        input_dim: int = 4,
        output_dim: int = 256,
        use_batch_norm: bool = True
    ):
        """
        初始化 PointNet++
        
        Args:
            input_dim: 输入维度
            output_dim: 输出维度
            use_batch_norm: 是否使用 BatchNorm
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 简化版：直接使用多层 PointNet
        self.sa1 = self._make_sa_layer(input_dim, 64, use_batch_norm)
        self.sa2 = self._make_sa_layer(64, 128, use_batch_norm)
        self.sa3 = self._make_sa_layer(128, 256, use_batch_norm)
        
        self.fc = nn.Sequential(
            nn.Linear(256, output_dim),
            nn.BatchNorm1d(output_dim) if use_batch_norm else nn.Identity(),
            nn.ReLU()
        )
    
    def _make_sa_layer(
        self,
        in_dim: int,
        out_dim: int,
        use_bn: bool
    ) -> nn.Sequential:
        """创建 Set Abstraction 层"""
        layers = [
            nn.Conv1d(in_dim, out_dim, 1),
        ]
        if use_bn:
            layers.append(nn.BatchNorm1d(out_dim))
        layers.append(nn.ReLU())
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: (B, N, input_dim) 点云
            
        Returns:
            features: (B, output_dim) 全局特征
        """
        x = x.transpose(1, 2)  # (B, C, N)
        
        x = self.sa1(x)
        x = self.sa2(x)
        x = self.sa3(x)
        
        # 全局池化
        x = torch.max(x, 2)[0]
        
        # 最终映射
        features = self.fc(x)
        
        return features
    
    def get_output_dim(self) -> int:
        """获取输出维度"""
        return self.output_dim
