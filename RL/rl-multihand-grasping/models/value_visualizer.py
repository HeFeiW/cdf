"""
value_visualizer.py
V(s) 可视化工具
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Optional, Tuple
import os


class ValueVisualizer:
    """
    Value 可视化工具
    
    功能:
    - 给定 state → V(s)
    - 点扰动 → ΔV
    - 点云着色图
    """
    
    def __init__(self, model, device: str = "cuda"):
        """
        初始化可视化工具
        
        Args:
            model: ActorCritic 模型
            device: 计算设备
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    @torch.no_grad()
    def compute_value(
        self,
        obs: Dict[str, np.ndarray]
    ) -> float:
        """
        计算状态价值
        
        Args:
            obs: 观测字典
            
        Returns:
            value: V(s)
        """
        # 转换为 tensor
        pointcloud = torch.from_numpy(obs["pointcloud"]).float().unsqueeze(0).to(self.device)
        q = torch.from_numpy(obs["q"]).float().unsqueeze(0).to(self.device)
        
        obs_tensor = {"pointcloud": pointcloud, "q": q}
        
        value = self.model.value(obs_tensor)
        
        return value.item()
    
    @torch.no_grad()
    def compute_value_gradient(
        self,
        obs: Dict[str, np.ndarray],
        wrt: str = "pointcloud"
    ) -> np.ndarray:
        """
        计算 value 对输入的梯度
        
        Args:
            obs: 观测字典
            wrt: 对哪个变量求梯度 ("pointcloud" 或 "q")
            
        Returns:
            gradient: 梯度
        """
        # 转换为 tensor
        pointcloud = torch.from_numpy(obs["pointcloud"]).float().unsqueeze(0).to(self.device)
        q = torch.from_numpy(obs["q"]).float().unsqueeze(0).to(self.device)
        
        if wrt == "pointcloud":
            pointcloud.requires_grad_(True)
        else:
            q.requires_grad_(True)
        
        obs_tensor = {"pointcloud": pointcloud, "q": q}
        
        value = self.model.value(obs_tensor)
        value.backward()
        
        if wrt == "pointcloud":
            return pointcloud.grad.squeeze(0).cpu().numpy()
        else:
            return q.grad.squeeze(0).cpu().numpy()
    
    @torch.no_grad()
    def compute_point_contributions(
        self,
        obs: Dict[str, np.ndarray],
        epsilon: float = 0.01
    ) -> np.ndarray:
        """
        计算每个点对 value 的贡献
        
        使用扰动法估计
        
        Args:
            obs: 观测字典
            epsilon: 扰动大小
            
        Returns:
            contributions: (N,) 每个点的贡献
        """
        pointcloud = obs["pointcloud"]  # (N, 4)
        n_points = len(pointcloud)
        
        # 计算基准 value
        base_value = self.compute_value(obs)
        
        contributions = np.zeros(n_points)
        
        for i in range(n_points):
            # 创建扰动后的点云
            perturbed = pointcloud.copy()
            perturbed[i, :3] += epsilon * np.random.randn(3)
            
            perturbed_obs = {"pointcloud": perturbed, "q": obs["q"]}
            perturbed_value = self.compute_value(perturbed_obs)
            
            contributions[i] = abs(perturbed_value - base_value)
        
        return contributions
    
    def visualize_value_on_pointcloud(
        self,
        obs: Dict[str, np.ndarray],
        save_path: Optional[str] = None,
        method: str = "gradient"
    ) -> None:
        """
        在点云上可视化 value
        
        Args:
            obs: 观测字典
            save_path: 保存路径
            method: 可视化方法 ("gradient" 或 "perturbation")
        """
        pointcloud = obs["pointcloud"]
        
        if method == "gradient":
            gradient = self.compute_value_gradient(obs, wrt="pointcloud")
            values = np.linalg.norm(gradient[:, :3], axis=1)
        else:
            values = self.compute_point_contributions(obs)
        
        # 归一化
        values_norm = (values - values.min()) / (values.max() - values.min() + 1e-8)
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(
            pointcloud[:, 0],
            pointcloud[:, 1],
            pointcloud[:, 2],
            c=values_norm,
            cmap='hot',
            s=10,
            alpha=0.8
        )
        
        plt.colorbar(scatter, ax=ax, label='Value Contribution')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Value Visualization (V={self.compute_value(obs):.3f})')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def visualize_value_landscape(
        self,
        base_obs: Dict[str, np.ndarray],
        joint_indices: List[int] = [0, 1],
        n_steps: int = 20,
        delta: float = 0.1,
        save_path: Optional[str] = None
    ) -> None:
        """
        可视化 value 在关节空间的 landscape
        
        Args:
            base_obs: 基准观测
            joint_indices: 要变化的关节索引
            n_steps: 采样步数
            delta: 变化范围
            save_path: 保存路径
        """
        assert len(joint_indices) == 2, "需要恰好两个关节索引"
        
        i, j = joint_indices
        base_q = base_obs["q"].copy()
        
        # 创建网格
        q_i_range = np.linspace(base_q[i] - delta, base_q[i] + delta, n_steps)
        q_j_range = np.linspace(base_q[j] - delta, base_q[j] + delta, n_steps)
        
        Q_i, Q_j = np.meshgrid(q_i_range, q_j_range)
        values = np.zeros_like(Q_i)
        
        for m in range(n_steps):
            for n in range(n_steps):
                q = base_q.copy()
                q[i] = Q_i[m, n]
                q[j] = Q_j[m, n]
                
                obs = {"pointcloud": base_obs["pointcloud"], "q": q}
                values[m, n] = self.compute_value(obs)
        
        # 绘制
        fig = plt.figure(figsize=(12, 5))
        
        # 3D 曲面
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.plot_surface(Q_i, Q_j, values, cmap='viridis', alpha=0.8)
        ax1.set_xlabel(f'Joint {i}')
        ax1.set_ylabel(f'Joint {j}')
        ax1.set_zlabel('Value')
        ax1.set_title('Value Landscape (3D)')
        
        # 2D 等高线
        ax2 = fig.add_subplot(122)
        contour = ax2.contourf(Q_i, Q_j, values, levels=20, cmap='viridis')
        plt.colorbar(contour, ax=ax2, label='Value')
        ax2.scatter(base_q[i], base_q[j], c='red', s=100, marker='x', label='Current')
        ax2.set_xlabel(f'Joint {i}')
        ax2.set_ylabel(f'Joint {j}')
        ax2.set_title('Value Landscape (2D)')
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def compare_states(
        self,
        obs_list: List[Dict[str, np.ndarray]],
        labels: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ) -> None:
        """
        比较多个状态的 value
        
        Args:
            obs_list: 观测列表
            labels: 标签列表
            save_path: 保存路径
        """
        n_states = len(obs_list)
        
        if labels is None:
            labels = [f"State {i}" for i in range(n_states)]
        
        values = [self.compute_value(obs) for obs in obs_list]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar(range(n_states), values, color='steelblue', alpha=0.8)
        ax.set_xticks(range(n_states))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel('Value V(s)')
        ax.set_title('Value Comparison')
        
        # 添加数值标签
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
