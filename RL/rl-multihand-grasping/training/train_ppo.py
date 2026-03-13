"""
train_ppo.py
PPO 训练入口
"""

import os
import sys
import yaml
import numpy as np
import torch
from datetime import datetime
from typing import Dict, Optional
import logging

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.grasp_env import MultiHandGraspEnv
from envs.utils import load_config
from training.ppo_agent import PPOAgent
from training.buffers import RolloutBuffer
from training.evaluator import Evaluator


def setup_logging(log_dir: str) -> logging.Logger:
    """设置日志"""
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger("train_ppo")
    logger.setLevel(logging.INFO)
    
    # 文件处理器
    fh = logging.FileHandler(os.path.join(log_dir, "train.log"))
    fh.setLevel(logging.INFO)
    
    # 控制台处理器
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # 格式化
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


def create_envs(
    env_config_path: str,
    hand_config_path: str,
    num_envs: int = 1
) -> list:
    """创建环境列表"""
    envs = []
    for _ in range(num_envs):
        env = MultiHandGraspEnv(
            env_config_path=env_config_path,
            hand_config_path=hand_config_path
        )
        envs.append(env)
    return envs


def collect_rollouts(
    envs: list,
    agent: PPOAgent,
    buffer: RolloutBuffer,
    n_steps: int
) -> Dict[str, float]:
    """
    收集 rollout 数据
    
    Args:
        envs: 环境列表
        agent: PPO 智能体
        buffer: 经验缓冲区
        n_steps: 收集步数
        
    Returns:
        info: 收集信息
    """
    num_envs = len(envs)
    
    # 初始化环境状态（如果需要）
    if not hasattr(collect_rollouts, "obs_list"):
        print("Resetting environments for rollout collection.")
        collect_rollouts.obs_list = [env.reset()[0] for env in envs]
    obs_list = collect_rollouts.obs_list
    
    episode_rewards = []
    episode_lengths = []
    episode_successes = []
    
    current_rewards = [0.0] * num_envs
    current_lengths = [0] * num_envs
    
    for step in range(n_steps):
        # 收集所有环境的动作
        pointclouds = np.stack([obs["pointcloud"] for obs in obs_list])
        qs = np.stack([obs["q"] for obs in obs_list])
        
        obs_batch = {"pointcloud": pointclouds, "q": qs}
        actions, values, log_probs = agent.select_action(obs_batch)
        # 执行动作
        next_obs_list = []
        rewards = []
        dones = []
        
        for i, env in enumerate(envs):
            action = actions[i] if num_envs > 1 else actions
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            next_obs_list.append(next_obs)
            rewards.append(reward)
            dones.append(terminated or truncated)
            
            current_rewards[i] += reward
            current_lengths[i] += 1
            
            if terminated or truncated:
                episode_rewards.append(current_rewards[i])
                episode_lengths.append(current_lengths[i])
                episode_successes.append(info.get("success", False))
                
                current_rewards[i] = 0.0
                current_lengths[i] = 0
                next_obs_list[i], _ = env.reset()
        
        # 添加到缓冲区
        buffer.add(
            pointcloud=pointclouds,
            q=qs,
            action=actions if num_envs > 1 else actions.reshape(1, -1),
            reward=np.array(rewards),
            done=np.array(dones),
            value=values if num_envs > 1 else np.array([values]),
            log_prob=log_probs if num_envs > 1 else np.array([log_probs])
        )
        
        obs_list = next_obs_list
    
    collect_rollouts.obs_list = obs_list
    
    # 计算最后的 value
    last_pointclouds = np.stack([obs["pointcloud"] for obs in obs_list])
    last_qs = np.stack([obs["q"] for obs in obs_list])
    last_obs = {"pointcloud": last_pointclouds, "q": last_qs}
    last_values = agent.get_value(last_obs)
    last_dones = np.zeros(num_envs)
    
    # 计算优势和回报
    buffer.compute_returns_and_advantages(last_values, last_dones)
    
    return {
        "mean_reward": np.mean(episode_rewards) if episode_rewards else 0.0,
        "mean_length": np.mean(episode_lengths) if episode_lengths else 0.0,
        "success_rate": np.mean(episode_successes) if episode_successes else 0.0,
        "num_episodes": len(episode_rewards),
    }


def train(
    env_config_path: str = "configs/env_config.yaml",
    hand_config_path: str = "configs/hand_config.yaml",
    ppo_config_path: str = "configs/ppo_config.yaml",
    log_dir: Optional[str] = None,
    resume_path: Optional[str] = None
) -> None:
    """
    训练入口
    
    Args:
        env_config_path: 环境配置路径
        hand_config_path: 手部配置路径
        ppo_config_path: PPO 配置路径
        log_dir: 日志目录
        resume_path: 恢复训练路径
    """
    # 加载配置
    ppo_config = load_config(ppo_config_path)
    env_config = load_config(env_config_path)
    
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 设置随机种子
    seed = ppo_config.get("seed", 42)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # 设置日志
    if log_dir is None:
        log_dir = os.path.join(
            "checkpoints",
            datetime.now().strftime("%Y%m%d_%H%M%S")
        )
    logger = setup_logging(log_dir)
    logger.info(f"Config: {ppo_config}")
    
    # 训练参数
    training_config = ppo_config["training"]
    total_timesteps = training_config["total_timesteps"]
    num_envs = training_config["num_envs"]
    steps_per_update = training_config["steps_per_update"]
    
    logging_config = ppo_config["logging"]
    log_interval = logging_config["log_interval"]
    save_interval = logging_config["save_interval"]
    eval_interval = logging_config["eval_interval"]
    
    # 创建环境
    logger.info("Creating environments...")
    envs = create_envs(env_config_path, hand_config_path, num_envs)
    
    # 创建智能体
    logger.info("Creating agent...")
    agent = PPOAgent(ppo_config, device=device)
    
    # 恢复训练
    if resume_path and os.path.exists(resume_path):
        logger.info(f"Resuming from {resume_path}")
        agent.load(resume_path)
    
    # 创建缓冲区
    pointcloud_shape = (env_config["pointcloud"]["num_points"], 4)
    q_dim = 22
    action_dim = 22
    
    buffer = RolloutBuffer(
        buffer_size=steps_per_update,
        num_envs=num_envs,
        pointcloud_shape=pointcloud_shape,
        q_dim=q_dim,
        action_dim=action_dim,
        device=device,
        gamma=ppo_config["ppo"]["gamma"],
        gae_lambda=ppo_config["ppo"]["gae_lambda"]
    )
    
    # 创建评估器
    eval_env = MultiHandGraspEnv(env_config_path, hand_config_path)
    evaluator = Evaluator(eval_env, agent, device=device)
    
    # 训练循环
    logger.info("Starting training...")
    num_updates = total_timesteps // (num_envs * steps_per_update)
    timesteps = 0
    
    for update in range(1, num_updates + 1):
        # 收集数据
        rollout_info = collect_rollouts(envs, agent, buffer, steps_per_update)
        timesteps += num_envs * steps_per_update
        print(f"Collected {timesteps} timesteps.")
        print(f"Rollout info: {rollout_info}")
        
        # 更新策略
        update_info = agent.update(buffer)
        buffer.reset()
        
        # 日志
        if update % log_interval == 0:
            logger.info(
                f"Update {update}/{num_updates} | "
                f"Timesteps {timesteps} | "
                f"Reward: {rollout_info['mean_reward']:.2f} | "
                f"Success: {rollout_info['success_rate']:.2%} | "
                f"Policy Loss: {update_info['policy_loss']:.4f} | "
                f"Value Loss: {update_info['value_loss']:.4f}"
            )
        
        # 保存
        if update % save_interval == 0:
            save_path = os.path.join(log_dir, f"checkpoint_{update}.pt")
            agent.save(save_path)
            logger.info(f"Saved checkpoint to {save_path}")
        
        # 评估
        if update % eval_interval == 0:
            eval_info = evaluator.evaluate(
                num_episodes=logging_config["num_eval_episodes"]
            )
            logger.info(
                f"Evaluation | "
                f"Mean Reward: {eval_info['mean_reward']:.2f} | "
                f"Success Rate: {eval_info['success_rate']:.2%}"
            )
    
    # 保存最终模型
    final_path = os.path.join(log_dir, "final_model.pt")
    agent.save(final_path)
    logger.info(f"Training complete. Saved final model to {final_path}")
    
    # 关闭环境
    for env in envs:
        env.close()
    eval_env.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train PPO agent for grasping")
    parser.add_argument("--env-config", type=str, default="configs/env_config.yaml")
    parser.add_argument("--hand-config", type=str, default="configs/hand_config.yaml")
    parser.add_argument("--ppo-config", type=str, default="configs/ppo_config.yaml")
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    
    args = parser.parse_args()
    
    train(
        env_config_path=args.env_config,
        hand_config_path=args.hand_config,
        ppo_config_path=args.ppo_config,
        log_dir=args.log_dir,
        resume_path=args.resume
    )
