"""
visualize_value.py
V(s) 可视化脚本
"""

import os
import sys
import argparse
import numpy as np
import torch

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.grasp_env import MultiHandGraspEnv
from envs.utils import load_config
from training.ppo_agent import PPOAgent
from models.value_visualizer import ValueVisualizer


def main():
    parser = argparse.ArgumentParser(description="Visualize value function")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint file")
    parser.add_argument("--env-config", type=str, default="configs/env_config.yaml",
                        help="Path to environment config")
    parser.add_argument("--hand-config", type=str, default="configs/hand_config.yaml",
                        help="Path to hand config")
    parser.add_argument("--ppo-config", type=str, default="configs/ppo_config.yaml",
                        help="Path to PPO config")
    parser.add_argument("--save-dir", type=str, default="value_visualization",
                        help="Directory to save visualizations")
    parser.add_argument("--num-samples", type=int, default=5,
                        help="Number of random states to visualize")
    parser.add_argument("--method", type=str, default="gradient",
                        choices=["gradient", "perturbation"],
                        help="Visualization method")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 创建环境
    env = MultiHandGraspEnv(
        env_config_path=args.env_config,
        hand_config_path=args.hand_config
    )
    
    # 加载配置和智能体
    ppo_config = load_config(args.ppo_config)
    agent = PPOAgent(ppo_config, device=device)
    agent.load(args.checkpoint)
    print(f"Loaded checkpoint from {args.checkpoint}")
    
    # 创建可视化工具
    visualizer = ValueVisualizer(agent.model, device=device)
    
    # 收集随机状态
    print(f"\nCollecting {args.num_samples} random states...")
    states = []
    for i in range(args.num_samples):
        obs, _ = env.reset()
        
        # 执行几步获得不同状态
        for _ in range(np.random.randint(1, 20)):
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()
                break
        
        states.append(obs)
    
    # 可视化每个状态
    print("\nVisualizing values...")
    for i, obs in enumerate(states):
        value = visualizer.compute_value(obs)
        print(f"State {i+1}: V(s) = {value:.4f}")
        
        # 点云可视化
        save_path = os.path.join(args.save_dir, f"state_{i+1}_pointcloud.png")
        visualizer.visualize_value_on_pointcloud(
            obs, save_path=save_path, method=args.method
        )
        print(f"  Saved to {save_path}")
    
    # 对第一个状态绘制 value landscape
    if len(states) > 0:
        print("\nVisualizing value landscape...")
        save_path = os.path.join(args.save_dir, "value_landscape.png")
        visualizer.visualize_value_landscape(
            states[0],
            joint_indices=[0, 1],
            save_path=save_path
        )
        print(f"Saved to {save_path}")
    
    # 比较所有状态
    print("\nComparing states...")
    save_path = os.path.join(args.save_dir, "value_comparison.png")
    labels = [f"State {i+1}" for i in range(len(states))]
    visualizer.compare_states(states, labels=labels, save_path=save_path)
    print(f"Saved to {save_path}")
    
    # 关闭环境
    env.close()
    print("\nVisualization complete!")


if __name__ == "__main__":
    main()
