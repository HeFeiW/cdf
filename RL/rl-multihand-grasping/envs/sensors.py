"""
sensors.py
点云传感器模块
"""

import numpy as np
from typing import List, Tuple, Optional


class PointCloudSensor:
    """
    点云传感器
    
    负责:
    - 从 PyBullet depth camera 采样点云
    - 转换到手心坐标系
    - 固定数量采样
    - 为点云添加标签
    """
    
    def __init__(self, sim, config: dict):
        """
        初始化点云传感器
        
        Args:
            sim: PyBullet 仿真实例
            config: 环境配置
        """
        self.sim = sim
        self.config = config
        
        # 点云参数
        self.num_points = config["pointcloud"]["num_points"]
        self.camera_distance = config["pointcloud"]["camera_distance"]
        self.camera_yaw = config["pointcloud"]["camera_yaw"]
        self.camera_pitch = config["pointcloud"]["camera_pitch"]
        self.fov = config["pointcloud"]["fov"]
        self.near = config["pointcloud"]["near"]
        self.far = config["pointcloud"]["far"]
        self.image_width = config["pointcloud"]["image_width"]
        self.image_height = config["pointcloud"]["image_height"]
        
    def get_pointcloud(
        self,
        hand_pose: Tuple[np.ndarray, np.ndarray],
        object_ids: List[int],
        target_id: int
    ) -> np.ndarray:
        """
        获取点云
        
        Args:
            hand_pose: (position, orientation) 手部位姿
            object_ids: 所有物体 ID
            target_id: 目标物体 ID
            
        Returns:
            pointcloud: (N, 4) 点云 (x, y, z, label)
        """
        hand_position, hand_orientation = hand_pose
        
        # 获取深度图像
        depth_image, segmentation_mask = self._get_depth_image(hand_position)
        
        # 从深度图像提取点云
        points_world, labels = self._depth_to_pointcloud(
            depth_image, segmentation_mask, hand_position, target_id
        )
        
        # 转换到手心坐标系
        points_local = self._transform_to_hand_frame(
            points_world, hand_position, hand_orientation
        )
        
        # 固定数量采样
        pointcloud = self._sample_points(points_local, labels, self.num_points)
        
        return pointcloud
    
    def _get_depth_image(
        self,
        target_position: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取深度图像和分割掩码
        
        Args:
            target_position: 相机朝向的目标位置
            
        Returns:
            depth_image: 深度图像
            segmentation_mask: 分割掩码
        """
        # 计算相机位置
        view_matrix = self.sim.compute_view_matrix(
            target_position=target_position,
            distance=self.camera_distance,
            yaw=self.camera_yaw,
            pitch=self.camera_pitch,
            roll=0,
            up_axis_index=2
        )
        
        projection_matrix = self.sim.compute_projection_matrix(
            fov=self.fov,
            aspect=self.image_width / self.image_height,
            near=self.near,
            far=self.far
        )
        
        # 获取图像
        _, _, _, depth_image, segmentation_mask = self.sim.get_camera_image(
            width=self.image_width,
            height=self.image_height,
            view_matrix=view_matrix,
            projection_matrix=projection_matrix
        )
        
        return depth_image, segmentation_mask
    
    def _depth_to_pointcloud(
        self,
        depth_image: np.ndarray,
        segmentation_mask: np.ndarray,
        camera_target: np.ndarray,
        target_id: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        将深度图像转换为点云
        
        Args:
            depth_image: 深度图像
            segmentation_mask: 分割掩码
            camera_target: 相机目标位置
            target_id: 目标物体 ID
            
        Returns:
            points: (M, 3) 点云坐标
            labels: (M,) 标签 (1=目标, 0=其他)
        """
        # 创建像素网格
        u = np.arange(self.image_width)
        v = np.arange(self.image_height)
        u, v = np.meshgrid(u, v)
        
        # 转换深度值
        depth = self.far * self.near / (
            self.far - (self.far - self.near) * depth_image
        )
        
        # 过滤无效深度
        valid_mask = (depth > self.near) & (depth < self.far * 0.99)
        
        # 计算 3D 坐标
        fx = self.image_width / (2 * np.tan(np.radians(self.fov / 2)))
        fy = fx
        cx = self.image_width / 2
        cy = self.image_height / 2
        
        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        z = depth
        
        # 提取有效点
        points = np.stack([x[valid_mask], y[valid_mask], z[valid_mask]], axis=1)
        
        # 生成标签
        seg_values = segmentation_mask[valid_mask]
        labels = (seg_values == target_id).astype(np.float32)
        
        return points, labels
    
    def _transform_to_hand_frame(
        self,
        points: np.ndarray,
        hand_position: np.ndarray,
        hand_orientation: np.ndarray
    ) -> np.ndarray:
        """
        将点云转换到手心坐标系
        
        Args:
            points: (M, 3) 世界坐标系点云
            hand_position: (3,) 手心位置
            hand_orientation: (4,) 手心姿态四元数
            
        Returns:
            points_local: (M, 3) 手心坐标系点云
        """
        # 获取旋转矩阵
        rotation_matrix = self.sim.quaternion_to_matrix(hand_orientation)
        
        # 平移
        points_centered = points - hand_position
        
        # 旋转到局部坐标系
        points_local = points_centered @ rotation_matrix
        
        return points_local
    
    def _sample_points(
        self,
        points: np.ndarray,
        labels: np.ndarray,
        num_points: int
    ) -> np.ndarray:
        """
        采样固定数量的点
        
        Args:
            points: (M, 3) 点云
            labels: (M,) 标签
            num_points: 目标点数
            
        Returns:
            sampled: (N, 4) 采样后的点云
        """
        n = len(points)
        
        if n == 0:
            # 没有点，返回零填充
            return np.zeros((num_points, 4), dtype=np.float32)
        
        if n >= num_points:
            # 随机采样
            indices = np.random.choice(n, num_points, replace=False)
        else:
            # 重复采样
            indices = np.random.choice(n, num_points, replace=True)
        
        sampled_points = points[indices]
        sampled_labels = labels[indices].reshape(-1, 1)
        
        return np.concatenate([sampled_points, sampled_labels], axis=1)
