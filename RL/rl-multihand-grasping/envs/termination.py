"""
termination.py
终止条件逻辑
"""

import numpy as np
from typing import Tuple, List, Dict, Any


class TerminationChecker:
    """
    终止条件检查器
    
    管理终止规则:
    - step 超限
    - 碰撞
    - 是否 truncated
    - 是否成功
    - 是否掉落
    """
    
    def __init__(self, config: dict):
        """
        初始化终止检查器
        
        Args:
            config: 环境配置
        """
        self.max_steps = config["task"]["max_steps"]
        self.success_height = config["task"]["success_height_threshold"]
        self.table_height = config["scene"]["table_position"][2]
        
        # 成功持续计数
        self.success_count = 0
        self.required_success_steps = 10  # 需要保持成功状态的步数
        
    def check(
        self,
        target_position: np.ndarray,
        target_initial_height: float,
        has_collision: bool,
        step_count: int,
        contacts: List[Dict],
        target_id: int
    ) -> Tuple[bool, bool, bool]:
        """
        检查终止条件
        
        Args:
            target_position: 目标物体位置
            target_initial_height: 目标物体初始高度
            has_collision: 是否与障碍物碰撞
            step_count: 当前步数
            contacts: 接触信息列表
            target_id: 目标物体 ID
            
        Returns:
            terminated: 是否终止（成功或失败）
            truncated: 是否截断（超时）
            success: 是否成功
        """
        terminated = False
        truncated = False
        success = False
        
        # 检查是否成功（提升到指定高度且稳定）
        height_gain = target_position[2] - target_initial_height
        is_grasping = self._check_grasping(contacts, target_id)
        
        if height_gain >= self.success_height and is_grasping:
            self.success_count += 1
            if self.success_count >= self.required_success_steps:
                terminated = True
                success = True
        else:
            self.success_count = 0
        
        # 检查是否掉落
        if self._check_dropped(target_position, target_initial_height):
            terminated = True
            success = False
        
        # 检查步数限制
        if step_count >= self.max_steps:
            truncated = True
        
        return terminated, truncated, success
    
    def _check_grasping(
        self,
        contacts: List[Dict],
        target_id: int
    ) -> bool:
        """
        检查是否正在抓取目标物体
        """
        contact_count = 0
        for contact in contacts:
            if contact.get("body_b") == target_id:
                contact_count += 1
        
        # 至少两个手指接触目标物体
        return contact_count >= 2
    
    def _check_dropped(
        self,
        target_position: np.ndarray,
        target_initial_height: float
    ) -> bool:
        """
        检查物体是否掉落
        """
        # 如果物体低于初始高度太多，认为掉落
        if target_position[2] < target_initial_height - 0.1:
            return True
        
        # 如果物体低于桌面，认为掉落
        if target_position[2] < self.table_height:
            return True
        
        return False
    
    def reset(self) -> None:
        """重置状态"""
        self.success_count = 0
