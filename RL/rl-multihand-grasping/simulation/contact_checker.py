"""
contact_checker.py
碰撞和接触检查
"""

import pybullet as p
import numpy as np
from typing import List, Dict, Tuple


class ContactChecker:
    """
    接触检查器
    
    判断是否有 penetration > 0
    
    注意: 所有 PyBullet API 调用都使用 physicsClientId 参数
    """
    
    def __init__(self, sim):
        """
        初始化接触检查器
        
        Args:
            sim: PyBullet 仿真实例
        """
        self.sim = sim
        self.physics_client = sim.physics_client
    
    def check_penetration(
        self,
        hand_id: int,
        obstacle_ids: List[int]
    ) -> bool:
        """
        检查手部与障碍物是否有穿透
        
        Args:
            hand_id: 手部 ID
            obstacle_ids: 障碍物 ID 列表
            
        Returns:
            has_penetration: 是否有穿透
        """
        return self.sim.check_penetration(hand_id, obstacle_ids)
    
    def get_contact_points(
        self,
        body_a: int,
        body_b: int
    ) -> List[Dict]:
        """
        获取两个物体之间的接触点
        
        Args:
            body_a: 物体 A ID
            body_b: 物体 B ID
            
        Returns:
            contacts: 接触点列表
        """
        contact_points = p.getContactPoints(body_a, body_b, physicsClientId=self.physics_client)
        
        contacts = []
        for cp in contact_points:
            contacts.append({
                "position_on_a": np.array(cp[5]),
                "position_on_b": np.array(cp[6]),
                "normal": np.array(cp[7]),
                "distance": cp[8],
                "normal_force": cp[9],
                "link_a": cp[3],
                "link_b": cp[4]
            })
        
        return contacts
    
    def get_closest_points(
        self,
        body_a: int,
        body_b: int,
        distance_threshold: float = 0.1
    ) -> List[Dict]:
        """
        获取两个物体之间的最近点
        
        Args:
            body_a: 物体 A ID
            body_b: 物体 B ID
            distance_threshold: 距离阈值
            
        Returns:
            closest_points: 最近点列表
        """
        closest_points = p.getClosestPoints(body_a, body_b, distance_threshold, physicsClientId=self.physics_client)
        
        points = []
        for cp in closest_points:
            points.append({
                "position_on_a": np.array(cp[5]),
                "position_on_b": np.array(cp[6]),
                "normal": np.array(cp[7]),
                "distance": cp[8],
                "link_a": cp[3],
                "link_b": cp[4]
            })
        
        return points
    
    def get_all_contacts(self, body_id: int) -> List[Dict]:
        """
        获取物体的所有接触
        
        Args:
            body_id: 物体 ID
            
        Returns:
            contacts: 接触列表
        """
        contact_points = p.getContactPoints(bodyA=body_id, physicsClientId=self.physics_client)
        
        contacts = []
        for cp in contact_points:
            contacts.append({
                "body_a": cp[1],
                "body_b": cp[2],
                "link_a": cp[3],
                "link_b": cp[4],
                "position": np.array(cp[5]),
                "normal": np.array(cp[7]),
                "distance": cp[8],
                "normal_force": cp[9]
            })
        
        return contacts
    
    def is_in_contact(
        self,
        body_a: int,
        body_b: int
    ) -> bool:
        """
        检查两个物体是否接触
        
        Args:
            body_a: 物体 A ID
            body_b: 物体 B ID
            
        Returns:
            is_contact: 是否接触
        """
        contact_points = p.getContactPoints(body_a, body_b, physicsClientId=self.physics_client)
        return len(contact_points) > 0
    
    def count_contact_links(
        self,
        hand_id: int,
        target_id: int
    ) -> int:
        """
        统计与目标物体接触的 link 数量
        
        Args:
            hand_id: 手部 ID
            target_id: 目标物体 ID
            
        Returns:
            count: 接触的 link 数量
        """
        contact_points = p.getContactPoints(hand_id, target_id, physicsClientId=self.physics_client)
        contact_links = set()
        
        for cp in contact_points:
            contact_links.add(cp[3])  # link index on hand
        
        return len(contact_links)
    
    def get_penetration_depth(
        self,
        hand_id: int,
        obstacle_ids: List[int]
    ) -> float:
        """
        获取最大穿透深度
        
        Args:
            hand_id: 手部 ID
            obstacle_ids: 障碍物 ID 列表
            
        Returns:
            max_penetration: 最大穿透深度（负值表示穿透）
        """
        max_penetration = 0.0
        
        for obs_id in obstacle_ids:
            contact_points = p.getContactPoints(hand_id, obs_id, physicsClientId=self.physics_client)
            for cp in contact_points:
                if cp[8] < max_penetration:
                    max_penetration = cp[8]
        
        return max_penetration
