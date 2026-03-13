"""
evaluator.py
策略评估器
"""

import numpy as np
import torch
from typing import Dict, List, Optional
import os


class Evaluator:
    """
    策略评估器
    
    评估训练后的策略性能
    """
    
    def __init__(
        self,
        env,
        agent,
        device: str = "cuda"
    ):
        """
        初始化评估器
        
        Args:
            env: 环境实例
            agent: 智能体
            device: 计算设备
        """
        self.env = env
        self.agent = agent
        self.device = device
    
    def evaluate(
        self,
        num_episodes: int = 10,
        deterministic: bool = True,
        render: bool = False,
        save_trajectories: bool = False
    ) -> Dict[str, float]:
        """
        评估策略
        
        Args:
            num_episodes: 评估 episode 数
            deterministic: 是否使用确定性策略
            render: 是否渲染
            save_trajectories: 是否保存轨迹
            
        Returns:
            info: 评估结果
        """
        episode_rewards = []
        episode_lengths = []
        episode_successes = []
        
        trajectories = [] if save_trajectories else None
        
        for ep in range(num_episodes):
            obs, _ = self.env.reset()
            done = False
            total_reward = 0.0
            length = 0
            
            if save_trajectories:
                traj = {"obs": [], "actions": [], "rewards": []}
            
            while not done:
                # 选择动作
                action, _, _ = self.agent.select_action(obs, deterministic=deterministic)
                
                if save_trajectories:
                    traj["obs"].append(obs.copy())
                    traj["actions"].append(action.copy())
                
                # 执行动作
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                
                if save_trajectories:
                    traj["rewards"].append(reward)
                
                if render:
                    self.env.render()
                
                total_reward += reward
                length += 1
                done = terminated or truncated
                obs = next_obs
            
            episode_rewards.append(total_reward)
            episode_lengths.append(length)
            episode_successes.append(info.get("success", False))
            
            if save_trajectories:
                trajectories.append(traj)
        
        result = {
            "mean_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "min_reward": np.min(episode_rewards),
            "max_reward": np.max(episode_rewards),
            "mean_length": np.mean(episode_lengths),
            "success_rate": np.mean(episode_successes),
            "num_episodes": num_episodes,
        }
        
        if save_trajectories:
            result["trajectories"] = trajectories
        
        return result
    
    def evaluate_single(
        self,
        deterministic: bool = True,
        verbose: bool = True
    ) -> Dict:
        """
        评估单个 episode
        
        Args:
            deterministic: 是否使用确定性策略
            verbose: 是否打印详情
            
        Returns:
            info: 评估结果
        """
        obs, reset_info = self.env.reset()
        done = False
        total_reward = 0.0
        length = 0
        
        reward_history = []
        value_history = []
        
        while not done:
            # 获取动作和价值
            action, value, _ = self.agent.select_action(obs, deterministic=deterministic)
            
            # 执行动作
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            
            reward_history.append(reward)
            value_history.append(value)
            
            if verbose:
                print(f"Step {length}: reward={reward:.4f}, value={value:.4f}")
            
            total_reward += reward
            length += 1
            done = terminated or truncated
            obs = next_obs
        
        result = {
            "total_reward": total_reward,
            "length": length,
            "success": info.get("success", False),
            "reward_history": reward_history,
            "value_history": value_history,
            "final_info": info,
        }
        
        if verbose:
            print(f"\nEpisode finished:")
            print(f"  Total reward: {total_reward:.4f}")
            print(f"  Length: {length}")
            print(f"  Success: {result['success']}")
        
        return result
    
    def compute_value_stats(
        self,
        num_samples: int = 100
    ) -> Dict[str, float]:
        """
        计算 value 统计信息
        
        Args:
            num_samples: 采样数
            
        Returns:
            stats: 统计信息
        """
        values = []
        
        for _ in range(num_samples):
            obs, _ = self.env.reset()
            value = self.agent.get_value(obs)
            values.append(value)
        
        return {
            "mean_value": np.mean(values),
            "std_value": np.std(values),
            "min_value": np.min(values),
            "max_value": np.max(values),
        }
    
    def save_results(
        self,
        results: Dict,
        save_dir: str,
        prefix: str = "eval"
    ) -> None:
        """
        保存评估结果
        
        Args:
            results: 评估结果
            save_dir: 保存目录
            prefix: 文件前缀
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存为 numpy 文件
        save_path = os.path.join(save_dir, f"{prefix}_results.npy")
        np.save(save_path, results)
        
        # 保存为文本摘要
        summary_path = os.path.join(save_dir, f"{prefix}_summary.txt")
        with open(summary_path, "w") as f:
            for key, value in results.items():
                if isinstance(value, (int, float)):
                    f.write(f"{key}: {value}\n")
