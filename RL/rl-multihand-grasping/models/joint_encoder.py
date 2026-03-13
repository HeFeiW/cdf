"""
joint_encoder.py
关节状态编码器
"""

import torch
import torch.nn as nn
from typing import List


class JointEncoder(nn.Module):
    """
    关节状态编码器
    
    输入: (22,) 关节位置
    输出: (64,) 特征向量
    """
    
    def __init__(
        self,
        input_dim: int = 22,
        hidden_dims: List[int] = [64, 64],
        output_dim: int = 64,
        use_layer_norm: bool = True,
        activation: str = "relu"
    ):
        """
        初始化关节编码器
        
        Args:
            input_dim: 输入维度 (22 DoF)
            hidden_dims: 隐藏层维度
            output_dim: 输出维度
            use_layer_norm: 是否使用 LayerNorm
            activation: 激活函数类型
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 构建 MLP
        dims = [input_dim] + hidden_dims + [output_dim]
        layers = []
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            
            if i < len(dims) - 2:  # 除了最后一层
                if use_layer_norm:
                    layers.append(nn.LayerNorm(dims[i+1]))
                
                if activation == "relu":
                    layers.append(nn.ReLU())
                elif activation == "tanh":
                    layers.append(nn.Tanh())
                elif activation == "elu":
                    layers.append(nn.ELU())
                elif activation == "leaky_relu":
                    layers.append(nn.LeakyReLU(0.1))
        
        # 最后一层激活
        if use_layer_norm:
            layers.append(nn.LayerNorm(output_dim))
        layers.append(nn.ReLU())
        
        self.mlp = nn.Sequential(*layers)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: (B, input_dim) 关节位置
            
        Returns:
            features: (B, output_dim) 特征向量
        """
        return self.mlp(x)
    
    def get_output_dim(self) -> int:
        """获取输出维度"""
        return self.output_dim


class JointEncoderWithAttention(nn.Module):
    """
    带注意力机制的关节编码器
    
    将关节分组处理（base + fingers），然后用注意力融合
    """
    
    def __init__(
        self,
        base_dim: int = 6,
        finger_dim: int = 16,
        hidden_dim: int = 64,
        output_dim: int = 64,
        num_fingers: int = 4
    ):
        """
        初始化关节编码器
        
        Args:
            base_dim: base 自由度
            finger_dim: 手指自由度
            hidden_dim: 隐藏层维度
            output_dim: 输出维度
            num_fingers: 手指数量
        """
        super().__init__()
        
        self.base_dim = base_dim
        self.finger_dim = finger_dim
        self.num_fingers = num_fingers
        self.joints_per_finger = finger_dim // num_fingers
        self.output_dim = output_dim
        
        # Base 编码器
        self.base_encoder = nn.Sequential(
            nn.Linear(base_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # 手指编码器（共享权重）
        self.finger_encoder = nn.Sequential(
            nn.Linear(self.joints_per_finger, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # 注意力融合
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True
        )
        
        # 最终映射
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * (1 + num_fingers), output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: (B, 22) 关节位置
            
        Returns:
            features: (B, output_dim) 特征向量
        """
        batch_size = x.shape[0]
        
        # 分离 base 和 finger
        base = x[:, :self.base_dim]
        fingers = x[:, self.base_dim:]
        
        # 编码 base
        base_feat = self.base_encoder(base)  # (B, hidden_dim)
        
        # 编码每个手指
        finger_feats = []
        for i in range(self.num_fingers):
            start = i * self.joints_per_finger
            end = (i + 1) * self.joints_per_finger
            finger = fingers[:, start:end]
            finger_feat = self.finger_encoder(finger)
            finger_feats.append(finger_feat)
        
        # 堆叠手指特征
        finger_feats = torch.stack(finger_feats, dim=1)  # (B, num_fingers, hidden_dim)
        
        # 添加 base 特征作为 query
        base_feat_expanded = base_feat.unsqueeze(1)  # (B, 1, hidden_dim)
        
        # 自注意力
        all_feats = torch.cat([base_feat_expanded, finger_feats], dim=1)
        attn_out, _ = self.attention(all_feats, all_feats, all_feats)
        
        # 展平并映射
        attn_out = attn_out.view(batch_size, -1)
        output = self.fc(attn_out)
        
        return output
    
    def get_output_dim(self) -> int:
        """获取输出维度"""
        return self.output_dim
