"""
reward.py
可组合的 dense reward 计算
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class RewardTerms:
    """奖励分解结构"""
    dist: float = 0.0          # 距离项
    finger: float = 0.0        # 指尖-物体距离项
    lift: float = 0.0          # 提升项
    stable: float = 0.0        # 稳定项
    collision: float = 0.0     # 碰撞项
    action: float = 0.0        # 动作惩罚
    success: float = 0.0       # 成功奖励
    total: float = 0.0         # 总奖励


class RewardCalculator:
    """
    奖励计算器
    
    计算各个奖励分量并组合
    """
    
    def __init__(self, config: dict):
        """
        初始化奖励计算器
        
        Args:
            config: 环境配置
        """
        self.weights = config["reward_weights"]
        self.success_height = config["task"]["success_height_threshold"]
        
        # 上一帧的状态（用于计算增量奖励）
        self.prev_target_height = None
        self.prev_hand_dist = None
        self.prev_finger_dist = None
        
    def compute(
        self,
        hand_center: np.ndarray,
        fingertip_positions: np.ndarray,
        target_position: np.ndarray,
        target_initial_height: float,
        action: np.ndarray,
        has_collision: bool,
        contacts: List[Dict],
        target_id: int
    ) -> RewardTerms:
        """
        计算奖励
        
        Args:
            hand_center: (3,) 手心位置
            fingertip_positions: (4, 3) 指尖位置
            target_position: (3,) 目标物体位置
            target_initial_height: 目标物体初始高度
            action: (22,) 动作
            has_collision: 是否与障碍物碰撞
            contacts: 接触信息列表
            target_id: 目标物体 ID
            
        Returns:
            reward_terms: 奖励分解
        """
        reward_terms = RewardTerms()
        
        # 1. 距离奖励（手心靠近目标）
        reward_terms.dist = self._compute_distance_reward(
            hand_center, target_position
        )
        
        # 2. 指尖奖励（指尖靠近目标）
        reward_terms.finger = self._compute_finger_reward(
            fingertip_positions, target_position
        )
        
        # 3. 提升奖励
        reward_terms.lift = self._compute_lift_reward(
            target_position, target_initial_height
        )
        
        # 4. 稳定奖励（抓取后保持稳定）
        is_grasping = self._check_grasping(contacts, target_id)
        reward_terms.stable = self._compute_stable_reward(
            target_position, is_grasping
        )
        
        # 5. 碰撞惩罚
        reward_terms.collision = self._compute_collision_penalty(has_collision)
        
        # 6. 动作惩罚（鼓励平滑动作）
        reward_terms.action = self._compute_action_penalty(action)
        
        # 计算总奖励
        reward_terms.total = (
            self.weights["distance"] * reward_terms.dist +
            self.weights["finger"] * reward_terms.finger +
            self.weights["lift"] * reward_terms.lift +
            self.weights["stable"] * reward_terms.stable +
            self.weights["collision"] * reward_terms.collision +
            self.weights["action"] * reward_terms.action
        )
        
        # 更新历史状态
        self._update_history(hand_center, fingertip_positions, target_position)
        
        return reward_terms
    
    def _compute_distance_reward(
        self,
        hand_center: np.ndarray,
        target_position: np.ndarray
    ) -> float:
        """
        计算距离奖励（越近越好）
        """
        dist = np.linalg.norm(hand_center - target_position)
        
        # 奖励距离减小
        if self.prev_hand_dist is not None:
            reward = self.prev_hand_dist - dist
        else:
            reward = 0.0
        
        # 添加距离的负指数奖励
        reward += np.exp(-5 * dist) * 0.1
        
        return reward
    
    def _compute_finger_reward(
        self,
        fingertip_positions: np.ndarray,
        target_position: np.ndarray
    ) -> float:
        """
        计算指尖奖励（指尖靠近目标）
        """
        # 计算每个指尖到目标的距离
        distances = np.linalg.norm(
            fingertip_positions - target_position, axis=1
        )
        mean_dist = np.mean(distances)
        
        # 奖励距离减小
        if self.prev_finger_dist is not None:
            reward = self.prev_finger_dist - mean_dist
        else:
            reward = 0.0
        
        # 额外奖励多个手指接近
        close_fingers = np.sum(distances < 0.05)
        reward += close_fingers * 0.05
        
        return reward
    
    def _compute_lift_reward(
        self,
        target_position: np.ndarray,
        target_initial_height: float
    ) -> float:
        """
        计算提升奖励
        """
        current_height = target_position[2]
        height_gain = current_height - target_initial_height
        
        # 奖励高度增加
        if self.prev_target_height is not None:
            reward = current_height - self.prev_target_height
        else:
            reward = 0.0
        
        # 额外奖励达到一定高度
        if height_gain > 0.05:
            reward += 0.1
        if height_gain > self.success_height:
            reward += 0.5
        
        return reward
    
    def _compute_stable_reward(
        self,
        target_position: np.ndarray,
        is_grasping: bool
    ) -> float:
        """
        计算稳定奖励（抓取后保持稳定）
        """
        if not is_grasping:
            return 0.0
        
        # 如果正在抓取，给予稳定奖励
        reward = 0.1
        
        # 如果物体位置变化不大，额外奖励
        if self.prev_target_height is not None:
            height_change = abs(target_position[2] - self.prev_target_height)
            if height_change < 0.01:
                reward += 0.05
        
        return reward
    
    def _compute_collision_penalty(self, has_collision: bool) -> float:
        """
        计算碰撞惩罚
        """
        return -1.0 if has_collision else 0.0
    
    def _compute_action_penalty(self, action: np.ndarray) -> float:
        """
        计算动作惩罚（鼓励平滑动作）
        """
        return -np.sum(action ** 2)
    
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
    
    def _update_history(
        self,
        hand_center: np.ndarray,
        fingertip_positions: np.ndarray,
        target_position: np.ndarray
    ) -> None:
        """更新历史状态"""
        self.prev_hand_dist = np.linalg.norm(hand_center - target_position)
        self.prev_finger_dist = np.mean(
            np.linalg.norm(fingertip_positions - target_position, axis=1)
        )
        self.prev_target_height = target_position[2]
    
    def reset(self) -> None:
        """重置历史状态"""
        self.prev_target_height = None
        self.prev_hand_dist = None
        self.prev_finger_dist = None
