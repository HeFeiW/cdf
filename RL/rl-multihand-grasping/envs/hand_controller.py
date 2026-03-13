"""
hand_controller.py
管理 LeapHand 机械手控制
"""

import os
import numpy as np
from typing import List, Tuple, Optional


class HandController:
    """
    手部控制器
    
    负责:
    - 映射 Δq → new q
    - clip 到 joint limits
    - base movement（6-DoF）
    - 获取手部状态
    """
    
    def __init__(self, sim, config: dict):
        """
        初始化手部控制器
        
        Args:
            sim: PyBullet 仿真实例
            config: 手部配置
        """
        self.sim = sim
        self.config = config
        
        # 手部 ID
        self.hand_id = None
        
        # 关节配置
        self.base_dof = config["joints"]["base_dof"]
        self.finger_dof = config["joints"]["finger_dof"]
        self.total_dof = self.base_dof + self.finger_dof
        
        # 关节限制
        base_limits = config["joints"]["base_limits"]["position"]
        finger_limits = config["joints"]["finger_limits"]["position"]
        
        # 构建完整的关节限制
        self.joint_lower = np.array(base_limits["lower"] + finger_limits["lower"] * 4)
        self.joint_upper = np.array(base_limits["upper"] + finger_limits["upper"] * 4)
        
        # 初始姿态
        self.initial_base = np.array(config["initial_pose"]["base"])
        self.initial_fingers = np.array(config["initial_pose"]["fingers"])
        self.initial_q = np.concatenate([self.initial_base, self.initial_fingers])
        
        # 当前关节位置
        self.current_q = self.initial_q.copy()
        
        # 控制参数
        self.position_gain = config["control"]["position_gain"]
        self.velocity_gain = config["control"]["velocity_gain"]
        self.max_force = config["control"]["max_force"]
        
        # 指尖 link 名称
        self.fingertip_names = config["fingertips"]["link_names"]
        self.fingertip_indices = []
        
    def reset(self) -> None:
        """重置手部到初始状态"""
        # 加载手部 URDF
        urdf_path = self.config["hand"]["urdf_path"]
        self.hand_id = self.sim.load_hand(
            urdf_path=urdf_path,
            base_position=self.initial_base[:3],
            base_orientation=self.sim.euler_to_quaternion(self.initial_base[3:])
        )
        
        # 获取指尖 link 索引
        self.fingertip_indices = self.sim.get_link_indices(
            self.hand_id, self.fingertip_names
        )
        
        # 设置初始关节位置
        self.current_q = self.initial_q.copy()
        self._set_joint_positions(self.initial_fingers)
    
    def apply_delta_q(self, delta_q: np.ndarray) -> None:
        """
        应用关节位置增量
        
        Args:
            delta_q: (22,) 关节位置增量
        """
        # 更新关节位置
        new_q = self.current_q + delta_q
        
        # clip 到关节限制
        new_q = np.clip(new_q, self.joint_lower, self.joint_upper)
        
        self.current_q = new_q
        
        # 分离 base 和 finger
        base_q = new_q[:self.base_dof]
        finger_q = new_q[self.base_dof:]
        
        # 设置 base 位姿
        self._set_base_pose(base_q)
        
        # 设置手指关节
        self._set_joint_positions(finger_q)
    
    def _set_base_pose(self, base_q: np.ndarray) -> None:
        """设置 base 位姿"""
        position = base_q[:3]
        euler = base_q[3:]
        orientation = self.sim.euler_to_quaternion(euler)
        self.sim.set_base_pose(self.hand_id, position, orientation)
    
    def _set_joint_positions(self, joint_positions: np.ndarray) -> None:
        """设置手指关节位置"""
        self.sim.set_joint_positions(
            self.hand_id,
            joint_positions,
            position_gain=self.position_gain,
            velocity_gain=self.velocity_gain,
            max_force=self.max_force
        )
    
    def get_joint_positions(self) -> np.ndarray:
        """
        获取当前关节位置
        
        Returns:
            q: (22,) 关节位置
        """
        return self.current_q.copy()
    
    def get_base_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取 base 位姿
        
        Returns:
            position: (3,) 位置
            orientation: (4,) 四元数
        """
        return self.sim.get_base_pose(self.hand_id)
    
    def get_hand_center(self) -> np.ndarray:
        """
        获取手心位置
        
        Returns:
            center: (3,) 手心位置
        """
        pos, _ = self.get_base_pose()
        return pos
    
    def get_fingertip_positions(self) -> np.ndarray:
        """
        获取所有指尖位置
        
        Returns:
            positions: (4, 3) 指尖位置
        """
        positions = []
        for idx in self.fingertip_indices:
            pos = self.sim.get_link_position(self.hand_id, idx)
            positions.append(pos)
        return np.array(positions)
    
    def get_hand_id(self) -> int:
        """获取手部 ID"""
        return self.hand_id
