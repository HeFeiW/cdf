"""
visualizer.py
调试可视化工具
"""

import pybullet as p
import numpy as np
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Visualizer:
    """
    可视化工具
    
    可选调试功能:
    - 渲染场景
    - 显示点云
    - 显示抓取轨迹
    
    注意: 所有 PyBullet API 调用都使用 physicsClientId 参数
    """
    
    def __init__(self, sim=None):
        """
        初始化可视化工具
        
        Args:
            sim: PyBullet 仿真实例
        """
        self.sim = sim
        self.physics_client = sim.physics_client if sim else 0
        self.debug_lines = []
        self.debug_points = []
    
    def draw_point(
        self,
        position: np.ndarray,
        color: List[float] = [1, 0, 0],
        size: float = 0.01,
        lifetime: float = 0
    ) -> int:
        """
        在场景中绘制点
        
        Args:
            position: 点位置
            color: 颜色 RGB
            size: 点大小
            lifetime: 生存时间 (0 表示永久)
            
        Returns:
            point_id: 点 ID
        """
        # 使用小球表示点
        visual_shape = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=size,
            rgbaColor=color + [1],
            physicsClientId=self.physics_client
        )
        
        point_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visual_shape,
            basePosition=position,
            physicsClientId=self.physics_client
        )
        
        self.debug_points.append(point_id)
        return point_id
    
    def draw_line(
        self,
        start: np.ndarray,
        end: np.ndarray,
        color: List[float] = [1, 0, 0],
        width: float = 1,
        lifetime: float = 0
    ) -> int:
        """
        在场景中绘制线段
        
        Args:
            start: 起点
            end: 终点
            color: 颜色 RGB
            width: 线宽
            lifetime: 生存时间
            
        Returns:
            line_id: 线段 ID
        """
        line_id = p.addUserDebugLine(
            start, end, color, width, lifetime,
            physicsClientId=self.physics_client
        )
        self.debug_lines.append(line_id)
        return line_id
    
    def draw_coordinate_frame(
        self,
        position: np.ndarray,
        orientation: np.ndarray,
        length: float = 0.1
    ) -> List[int]:
        """
        绘制坐标系
        
        Args:
            position: 原点位置
            orientation: 姿态四元数
            length: 轴长度
            
        Returns:
            line_ids: 线段 ID 列表
        """
        rotation_matrix = np.array(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)  # 纯数学运算，不需要 physicsClientId
        
        x_axis = position + rotation_matrix[:, 0] * length
        y_axis = position + rotation_matrix[:, 1] * length
        z_axis = position + rotation_matrix[:, 2] * length
        
        ids = []
        ids.append(self.draw_line(position, x_axis, [1, 0, 0]))  # X: 红
        ids.append(self.draw_line(position, y_axis, [0, 1, 0]))  # Y: 绿
        ids.append(self.draw_line(position, z_axis, [0, 0, 1]))  # Z: 蓝
        
        return ids
    
    def draw_bounding_box(
        self,
        center: np.ndarray,
        size: np.ndarray,
        color: List[float] = [0, 1, 0]
    ) -> List[int]:
        """
        绘制边界框
        
        Args:
            center: 中心位置
            size: 尺寸 [x, y, z]
            color: 颜色
            
        Returns:
            line_ids: 线段 ID 列表
        """
        half_size = np.array(size) / 2
        
        # 8 个角点
        corners = []
        for dx in [-1, 1]:
            for dy in [-1, 1]:
                for dz in [-1, 1]:
                    corners.append(center + np.array([
                        dx * half_size[0],
                        dy * half_size[1],
                        dz * half_size[2]
                    ]))
        
        # 12 条边
        edges = [
            (0, 1), (2, 3), (4, 5), (6, 7),  # x 方向
            (0, 2), (1, 3), (4, 6), (5, 7),  # y 方向
            (0, 4), (1, 5), (2, 6), (3, 7)   # z 方向
        ]
        
        ids = []
        for i, j in edges:
            ids.append(self.draw_line(corners[i], corners[j], color))
        
        return ids
    
    def clear_debug_items(self) -> None:
        """清除所有调试绘制"""
        for line_id in self.debug_lines:
            p.removeUserDebugItem(line_id, physicsClientId=self.physics_client)
        self.debug_lines = []
        
        for point_id in self.debug_points:
            p.removeBody(point_id, physicsClientId=self.physics_client)
        self.debug_points = []
    
    def visualize_pointcloud(
        self,
        points: np.ndarray,
        labels: Optional[np.ndarray] = None,
        save_path: Optional[str] = None
    ) -> None:
        """
        可视化点云（使用 matplotlib）
        
        Args:
            points: (N, 3) 点云坐标
            labels: (N,) 标签（用于着色）
            save_path: 保存路径
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        if labels is not None:
            colors = np.where(labels > 0.5, 'red', 'gray')
            ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                      c=colors, s=1, alpha=0.5)
        else:
            ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                      s=1, alpha=0.5)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Point Cloud Visualization')
        
        # 设置相同的轴比例
        max_range = np.max(np.abs(points)) * 1.1
        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
        ax.set_zlim(-max_range, max_range)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def visualize_trajectory(
        self,
        positions: List[np.ndarray],
        save_path: Optional[str] = None
    ) -> None:
        """
        可视化轨迹
        
        Args:
            positions: 位置列表
            save_path: 保存路径
        """
        positions = np.array(positions)
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制轨迹线
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
               'b-', linewidth=2, label='Trajectory')
        
        # 标记起点和终点
        ax.scatter(*positions[0], c='green', s=100, marker='o', label='Start')
        ax.scatter(*positions[-1], c='red', s=100, marker='x', label='End')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Trajectory Visualization')
        ax.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def visualize_value_on_pointcloud(
        self,
        points: np.ndarray,
        values: np.ndarray,
        save_path: Optional[str] = None
    ) -> None:
        """
        在点云上可视化 value
        
        Args:
            points: (N, 3) 点云
            values: (N,) value 值
            save_path: 保存路径
        """
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 归一化 value 用于着色
        values_norm = (values - values.min()) / (values.max() - values.min() + 1e-8)
        
        scatter = ax.scatter(
            points[:, 0], points[:, 1], points[:, 2],
            c=values_norm, cmap='viridis', s=5
        )
        
        plt.colorbar(scatter, ax=ax, label='Value')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Value Visualization on Point Cloud')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
