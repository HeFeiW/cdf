"""
object_manager.py
管理物体加载与初始化
"""

import numpy as np
from typing import List, Tuple, Optional
import os


class ObjectManager:
    """
    物体管理器
    
    负责:
    - 加载目标物体和障碍物
    - 随机生成物体位置和姿态
    - 管理物体 ID
    """
    
    def __init__(self, sim, config: dict):
        """
        初始化物体管理器
        
        Args:
            sim: PyBullet 仿真实例
            config: 环境配置
        """
        self.sim = sim
        self.config = config
        
        # 物体配置
        self.num_obstacles_min = config["objects"]["num_obstacles_min"]
        self.num_obstacles_max = config["objects"]["num_obstacles_max"]
        self.spawn_area = config["objects"]["spawn_area"]
        self.table_position = config["scene"]["table_position"]
        self.table_size = config["scene"]["table_size"]
        
        # 物体 ID
        self.target_id = None
        self.obstacle_ids = []
        
        # 物体资源路径
        self.objects_path = "objects/"
        
    def spawn_objects(self, np_random: np.random.Generator) -> None:
        """
        生成物体
        
        Args:
            np_random: numpy 随机数生成器
        """
        # 清除之前的物体
        self._clear_objects()
        
        # 计算生成区域
        table_center = np.array(self.table_position[:2])
        spawn_half = np.array(self.spawn_area) / 2
        table_height = self.table_position[2] + self.table_size[2] / 2
        
        # 生成目标物体
        target_pos = self._sample_position(
            np_random, table_center, spawn_half, table_height
        )
        target_orn = self._sample_orientation(np_random)
        self.target_id = self._load_target_object(target_pos, target_orn)
        
        # 生成障碍物
        num_obstacles = np_random.integers(
            self.num_obstacles_min, self.num_obstacles_max + 1
        )
        
        occupied_positions = [target_pos[:2]]
        
        for _ in range(num_obstacles):
            # 采样不重叠的位置
            for attempt in range(20):
                pos = self._sample_position(
                    np_random, table_center, spawn_half, table_height
                )
                if self._is_valid_position(pos[:2], occupied_positions, min_dist=0.08):
                    break
            else:
                continue
            
            orn = self._sample_orientation(np_random)
            obstacle_id = self._load_obstacle_object(pos, orn)
            self.obstacle_ids.append(obstacle_id)
            occupied_positions.append(pos[:2])
    
    def _sample_position(
        self,
        np_random: np.random.Generator,
        center: np.ndarray,
        half_size: np.ndarray,
        height: float
    ) -> np.ndarray:
        """采样位置"""
        x = np_random.uniform(center[0] - half_size[0], center[0] + half_size[0])
        y = np_random.uniform(center[1] - half_size[1], center[1] + half_size[1])
        z = height + 0.02  # 稍微抬高避免穿透
        return np.array([x, y, z])
    
    def _sample_orientation(
        self,
        np_random: np.random.Generator
    ) -> np.ndarray:
        """采样姿态 (四元数)"""
        # 简单起见，只绕 z 轴旋转
        yaw = np_random.uniform(-np.pi, np.pi)
        return self.sim.euler_to_quaternion([0, 0, yaw])
    
    def _is_valid_position(
        self,
        pos: np.ndarray,
        occupied: List[np.ndarray],
        min_dist: float
    ) -> bool:
        """检查位置是否有效（不重叠）"""
        for occ in occupied:
            if np.linalg.norm(pos - occ) < min_dist:
                return False
        return True
    
    def _load_target_object(
        self,
        position: np.ndarray,
        orientation: np.ndarray
    ) -> int:
        """加载目标物体"""
        # 使用简单的几何体作为目标物体
        # 实际使用时可以替换为具体的 URDF
        target_id = self.sim.load_object(
            urdf_path=None,  # 使用原始形状
            position=position,
            orientation=orientation,
            shape_type="box",
            size=[0.04, 0.04, 0.04],
            color=[1, 0, 0, 1],  # 红色
            mass=0.1
        )
        return target_id
    
    def _load_obstacle_object(
        self,
        position: np.ndarray,
        orientation: np.ndarray
    ) -> int:
        """加载障碍物"""
        obstacle_id = self.sim.load_object(
            urdf_path=None,
            position=position,
            orientation=orientation,
            shape_type="box",
            size=[0.03, 0.03, 0.05],
            color=[0.5, 0.5, 0.5, 1],  # 灰色
            mass=0.2
        )
        return obstacle_id
    
    def _clear_objects(self) -> None:
        """清除所有物体"""
        if self.target_id is not None:
            self.sim.remove_body(self.target_id)
            self.target_id = None
        
        for obs_id in self.obstacle_ids:
            self.sim.remove_body(obs_id)
        self.obstacle_ids = []
    
    def get_target_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取目标物体位姿
        
        Returns:
            position: (3,) 位置
            orientation: (4,) 四元数
        """
        return self.sim.get_body_pose(self.target_id)
    
    def get_target_id(self) -> int:
        """获取目标物体 ID"""
        return self.target_id
    
    def get_obstacle_ids(self) -> List[int]:
        """获取所有障碍物 ID"""
        return self.obstacle_ids
    
    def get_all_object_ids(self) -> List[int]:
        """获取所有物体 ID"""
        return [self.target_id] + self.obstacle_ids
    
    def get_num_obstacles(self) -> int:
        """获取障碍物数量"""
        return len(self.obstacle_ids)
