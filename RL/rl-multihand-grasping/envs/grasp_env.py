"""
grasp_env.py
主环境类，实现 Gymnasium API
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any
import yaml

from .object_manager import ObjectManager
from .hand_controller import HandController
from .sensors import PointCloudSensor
from .reward import RewardCalculator, RewardTerms
from .termination import TerminationChecker
from .utils import load_config
from simulation.pybullet_sim import PyBulletSim


@dataclass
class State:
    """环境状态结构"""
    pointcloud: np.ndarray  # (N, 4) - (x, y, z, label)
    q: np.ndarray  # (22,) - joint positions


class MultiHandGraspEnv(gym.Env):
    """
    多指手抓取环境
    
    观测空间:
        - pointcloud: (N, 4) 点云 (x, y, z, label)
        - q: (22,) 关节位置
    
    动作空间:
        - delta_q: (22,) 关节位置增量
    
    任务目标:
        在不碰撞障碍物的前提下，从 clutter 中抓取目标物体并稳定提升到桌面以上 0.1m
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(
        self,
        env_config_path: str = "configs/env_config.yaml",
        hand_config_path: str = "configs/hand_config.yaml",
        render_mode: Optional[str] = None,
    ):
        """
        初始化环境
        
        Args:
            env_config_path: 环境配置文件路径
            hand_config_path: 手部配置文件路径
            render_mode: 渲染模式
        """
        super().__init__()
        
        # 加载配置
        self.env_config = load_config(env_config_path)
        self.hand_config = load_config(hand_config_path)
        
        self.render_mode = render_mode
        
        # 环境参数
        self.num_points = self.env_config["pointcloud"]["num_points"]
        self.max_steps = self.env_config["task"]["max_steps"]
        self.delta_q_clip = self.env_config["action"]["delta_q_clip"]
        self.action_frequency = self.env_config["action"]["action_frequency"]
        self.simulation_frequency = self.env_config["action"]["simulation_frequency"]
        self.steps_per_action = self.simulation_frequency // self.action_frequency
        
        # 动作和观测维度
        self.action_dim = 22  # 6 base + 16 finger joints
        self.joint_dim = 22
        
        # 定义动作空间 (Δq)
        self.action_space = spaces.Box(
            low=-self.delta_q_clip,
            high=self.delta_q_clip,
            shape=(self.action_dim,),
            dtype=np.float32
        )
        
        # 定义观测空间
        self.observation_space = spaces.Dict({
            "pointcloud": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.num_points, 4),
                dtype=np.float32
            ),
            "q": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.joint_dim,),
                dtype=np.float32
            )
        })
        
        # 初始化仿真环境
        enable_gui = render_mode == "human" or self.env_config["render"]["enable_gui"]
        print("Initializing simulation with GUI =", enable_gui)
        self.sim = PyBulletSim(enable_gui=enable_gui)
        
        # 初始化各模块
        self.object_manager = ObjectManager(self.sim, self.env_config)
        self.hand_controller = HandController(self.sim, self.hand_config)
        self.sensor = PointCloudSensor(self.sim, self.env_config)
        self.reward_calculator = RewardCalculator(self.env_config)
        self.termination_checker = TerminationChecker(self.env_config)
        
        # 状态变量
        self.step_count = 0
        self.target_initial_height = 0.0
        
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        重置环境
        
        Args:
            seed: 随机种子
            options: 额外选项
            
        Returns:
            observation: 初始观测
            info: 额外信息
        """
        super().reset(seed=seed)
        
        # 重置仿真
        self.sim.reset()
        
        # 加载场景（桌面等）
        self.sim.load_scene(self.env_config)
        
        # 生成物体
        self.object_manager.spawn_objects(self.np_random)
        target_pos = self.object_manager.get_target_pose()[0]
        self.target_initial_height = target_pos[2]
        
        # 重置手部
        self.hand_controller.reset()
        
        # 运行几步稳定仿真
        for _ in range(50):
            self.sim.step_simulation()
        
        # 获取初始观测
        observation = self._get_observation()
        
        # 重置计数器
        self.step_count = 0
        
        info = {
            "target_position": target_pos,
            "num_obstacles": self.object_manager.get_num_obstacles()
        }
        
        return observation, info
    
    def step(
        self,
        action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        执行一步动作
        
        Args:
            action: Δq (22,)
            
        Returns:
            observation: 新的观测
            reward: 奖励
            terminated: 是否终止
            truncated: 是否截断
            info: 额外信息
        """
        # 裁剪动作
        action = np.clip(action, -self.delta_q_clip, self.delta_q_clip)
        
        # 应用动作
        self.hand_controller.apply_delta_q(action)
        
        # 运行仿真
        for _ in range(self.steps_per_action):
            self.sim.step_simulation()
        
        # 获取观测
        observation = self._get_observation()
        
        # 获取当前状态信息
        hand_center = self.hand_controller.get_hand_center()
        fingertip_positions = self.hand_controller.get_fingertip_positions()
        target_position, target_orientation = self.object_manager.get_target_pose()
        
        # 获取碰撞信息
        hand_id = self.hand_controller.get_hand_id()
        obstacle_ids = self.object_manager.get_obstacle_ids()
        target_id = self.object_manager.get_target_id()
        
        contacts = self.sim.get_contacts(hand_id, obstacle_ids + [target_id])
        has_collision = self.sim.check_penetration(hand_id, obstacle_ids)
        
        # 计算奖励
        reward_terms = self.reward_calculator.compute(
            hand_center=hand_center,
            fingertip_positions=fingertip_positions,
            target_position=target_position,
            target_initial_height=self.target_initial_height,
            action=action,
            has_collision=has_collision,
            contacts=contacts,
            target_id=target_id
        )
        
        # 检查终止条件
        self.step_count += 1
        terminated, truncated, success = self.termination_checker.check(
            target_position=target_position,
            target_initial_height=self.target_initial_height,
            has_collision=has_collision,
            step_count=self.step_count,
            contacts=contacts,
            target_id=target_id
        )
        
        # 添加成功奖励
        if success:
            reward_terms.success = self.env_config["reward_weights"]["success"]
            reward_terms.total += reward_terms.success
        
        info = {
            "reward_terms": reward_terms,
            "success": success,
            "step_count": self.step_count,
            "target_height": target_position[2],
            "has_collision": has_collision
        }
        
        return observation, reward_terms.total, terminated, truncated, info
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """获取当前观测"""
        # 获取手部姿态
        hand_pose = self.hand_controller.get_base_pose()
        
        # 获取物体 ID
        object_ids = self.object_manager.get_all_object_ids()
        target_id = self.object_manager.get_target_id()
        
        # 获取点云
        pointcloud = self.sensor.get_pointcloud(
            hand_pose=hand_pose,
            object_ids=object_ids,
            target_id=target_id
        )
        
        # 获取关节位置
        q = self.hand_controller.get_joint_positions()
        
        return {
            "pointcloud": pointcloud.astype(np.float32),
            "q": q.astype(np.float32)
        }
    
    def render(self) -> Optional[np.ndarray]:
        """渲染环境"""
        if self.render_mode == "rgb_array":
            return self.sim.render_camera()
        return None
    
    def close(self):
        """关闭环境"""
        self.sim.close()
    
    def get_value_state(self) -> State:
        """获取用于计算 V(s) 的状态"""
        obs = self._get_observation()
        return State(
            pointcloud=obs["pointcloud"],
            q=obs["q"]
        )
