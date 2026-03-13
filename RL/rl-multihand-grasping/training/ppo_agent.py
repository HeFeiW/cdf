"""
ppo_agent.py
PPO 智能体
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple, Optional
import os

from models.actor_critic import ActorCritic
from .buffers import RolloutBuffer


class PPOAgent:
    """
    PPO 智能体
    
    实现 Proximal Policy Optimization 算法
    """
    
    def __init__(
        self,
        config: dict,
        device: str = "cuda"
    ):
        """
        初始化 PPO 智能体
        
        Args:
            config: 配置字典
            device: 计算设备
        """
        self.config = config
        self.device = device
        
        # PPO 参数
        ppo_config = config["ppo"]
        self.gamma = ppo_config["gamma"]
        self.gae_lambda = ppo_config["gae_lambda"]
        self.clip_ratio = ppo_config["clip_ratio"]
        self.value_loss_coef = ppo_config["value_loss_coef"]
        self.entropy_coef = ppo_config["entropy_coef"]
        self.max_grad_norm = ppo_config["max_grad_norm"]
        
        # 训练参数
        training_config = config["training"]
        self.num_epochs = training_config["num_epochs"]
        self.batch_size = training_config["batch_size"]
        self.normalize_advantage = training_config["normalize_advantage"]
        
        # 创建模型
        self.model = ActorCritic(config).to(device)
        
        # 优化器
        opt_config = config["optimizer"]
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=opt_config["learning_rate"],
            eps=opt_config["adam_eps"],
            weight_decay=opt_config["weight_decay"]
        )
        
        # 学习率调度器
        self.scheduler = None
    
    @torch.no_grad()
    def select_action(
        self,
        obs: Dict[str, np.ndarray],
        deterministic: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        选择动作
        
        Args:
            obs: 观测字典
            deterministic: 是否确定性策略
            
        Returns:
            action: 动作
            value: 状态价值
            log_prob: 动作对数概率
        """
        # 转换为 tensor
        obs_tensor = {
            "pointcloud": torch.from_numpy(obs["pointcloud"]).float().to(self.device),
            "q": torch.from_numpy(obs["q"]).float().to(self.device)
        }
        
        # 添加 batch 维度
        if obs_tensor["pointcloud"].dim() == 2:
            obs_tensor["pointcloud"] = obs_tensor["pointcloud"].unsqueeze(0)
            obs_tensor["q"] = obs_tensor["q"].unsqueeze(0)
        
        if deterministic:
            action = self.model.act(obs_tensor, deterministic=True)
            value = self.model.value(obs_tensor)
            log_prob = torch.zeros_like(value)
        else:
            action, log_prob, value = self.model(obs_tensor)
        
        return (
            action.cpu().numpy().squeeze(),
            value.cpu().numpy().squeeze(),
            log_prob.cpu().numpy().squeeze()
        )
    
    def update(
        self,
        buffer: RolloutBuffer
    ) -> Dict[str, float]:
        """
        更新策略
        
        Args:
            buffer: 经验缓冲区
            
        Returns:
            info: 训练信息
        """
        # 存储训练信息
        policy_losses = []
        value_losses = []
        entropy_losses = []
        approx_kls = []
        clip_fractions = []
        
        for epoch in range(self.num_epochs):
            for batch in buffer.get(self.batch_size):
                # 获取 batch 数据
                obs = {
                    "pointcloud": batch["pointcloud"],
                    "q": batch["q"]
                }
                actions = batch["actions"]
                old_log_probs = batch["old_log_probs"]
                advantages = batch["advantages"]
                returns = batch["returns"]
                old_values = batch["old_values"]
                
                # 归一化优势
                if self.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                # 评估动作
                log_probs, entropy, values = self.model.evaluate_actions(obs, actions)
                
                # 计算比率
                ratio = torch.exp(log_probs - old_log_probs)
                
                # 策略损失
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 价值损失（带 clip）
                values_clipped = old_values + torch.clamp(
                    values - old_values, -self.clip_ratio, self.clip_ratio
                )
                value_loss_1 = (values - returns) ** 2
                value_loss_2 = (values_clipped - returns) ** 2
                value_loss = 0.5 * torch.max(value_loss_1, value_loss_2).mean()
                
                # 熵损失
                entropy_loss = -entropy.mean()
                
                # 总损失
                loss = (
                    policy_loss +
                    self.value_loss_coef * value_loss +
                    self.entropy_coef * entropy_loss
                )
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # 记录信息
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
                
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - torch.log(ratio)).mean()
                    approx_kls.append(approx_kl.item())
                    clip_fraction = torch.mean((torch.abs(ratio - 1) > self.clip_ratio).float())
                    clip_fractions.append(clip_fraction.item())
        
        return {
            "policy_loss": np.mean(policy_losses),
            "value_loss": np.mean(value_losses),
            "entropy_loss": np.mean(entropy_losses),
            "approx_kl": np.mean(approx_kls),
            "clip_fraction": np.mean(clip_fractions),
        }
    
    def save(self, path: str) -> None:
        """保存模型"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
        }, path)
    
    def load(self, path: str) -> None:
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    def get_value(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """获取状态价值"""
        with torch.no_grad():
            obs_tensor = {
                "pointcloud": torch.from_numpy(obs["pointcloud"]).float().to(self.device),
                "q": torch.from_numpy(obs["q"]).float().to(self.device)
            }
            
            if obs_tensor["pointcloud"].dim() == 2:
                obs_tensor["pointcloud"] = obs_tensor["pointcloud"].unsqueeze(0)
                obs_tensor["q"] = obs_tensor["q"].unsqueeze(0)
            
            value = self.model.value(obs_tensor)
        
        return value.cpu().numpy().squeeze()
