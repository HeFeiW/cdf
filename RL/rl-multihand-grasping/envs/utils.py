"""
utils.py
环境工具函数
"""

import yaml
import numpy as np
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """
    加载 YAML 配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        config: 配置字典
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def normalize_angle(angle: float) -> float:
    """
    将角度归一化到 [-π, π]
    
    Args:
        angle: 输入角度
        
    Returns:
        normalized: 归一化后的角度
    """
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    四元数乘法
    
    Args:
        q1: 第一个四元数 [x, y, z, w]
        q2: 第二个四元数 [x, y, z, w]
        
    Returns:
        q: 结果四元数
    """
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ])


def quaternion_inverse(q: np.ndarray) -> np.ndarray:
    """
    四元数逆
    
    Args:
        q: 输入四元数 [x, y, z, w]
        
    Returns:
        q_inv: 逆四元数
    """
    return np.array([-q[0], -q[1], -q[2], q[3]])


def transform_point(
    point: np.ndarray,
    position: np.ndarray,
    quaternion: np.ndarray
) -> np.ndarray:
    """
    变换点坐标
    
    Args:
        point: (3,) 点坐标
        position: (3,) 平移
        quaternion: (4,) 旋转四元数
        
    Returns:
        transformed: (3,) 变换后的坐标
    """
    # 将点表示为四元数
    p = np.array([point[0], point[1], point[2], 0])
    
    # 旋转：q * p * q^-1
    q_inv = quaternion_inverse(quaternion)
    rotated = quaternion_multiply(
        quaternion_multiply(quaternion, p), q_inv
    )
    
    # 平移
    return rotated[:3] + position


def fps_sampling(points: np.ndarray, num_samples: int) -> np.ndarray:
    """
    最远点采样 (Farthest Point Sampling)
    
    Args:
        points: (N, D) 点云
        num_samples: 采样数量
        
    Returns:
        indices: (M,) 采样点索引
    """
    n = len(points)
    if n <= num_samples:
        return np.arange(n)
    
    # 初始化
    indices = np.zeros(num_samples, dtype=np.int64)
    distances = np.full(n, np.inf)
    
    # 随机选择第一个点
    indices[0] = np.random.randint(n)
    
    for i in range(1, num_samples):
        # 更新距离
        last_point = points[indices[i-1]]
        new_distances = np.linalg.norm(points - last_point, axis=1)
        distances = np.minimum(distances, new_distances)
        
        # 选择最远点
        indices[i] = np.argmax(distances)
    
    return indices
