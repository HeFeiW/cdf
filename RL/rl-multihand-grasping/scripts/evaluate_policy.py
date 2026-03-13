"""
evaluate_policy.py
策略评估脚本
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
from training.evaluator import Evaluator


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained policy")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint file")
    parser.add_argument("--env-config", type=str, default="configs/env_config.yaml",
                        help="Path to environment config")
    parser.add_argument("--hand-config", type=str, default="configs/hand_config.yaml",
                        help="Path to hand config")
    parser.add_argument("--ppo-config", type=str, default="configs/ppo_config.yaml",
                        help="Path to PPO config")
    parser.add_argument("--num-episodes", type=int, default=100,
                        help="Number of evaluation episodes")
    parser.add_argument("--deterministic", action="store_true",
                        help="Use deterministic policy")
    parser.add_argument("--render", action="store_true",
                        help="Render environment")
    parser.add_argument("--save-dir", type=str, default="eval_results",
                        help="Directory to save results")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 创建环境
    render_mode = "human" if args.render else None
    env = MultiHandGraspEnv(
        env_config_path=args.env_config,
        hand_config_path=args.hand_config,
        render_mode=render_mode
    )
    
    # 加载配置和智能体
    ppo_config = load_config(args.ppo_config)
    agent = PPOAgent(ppo_config, device=device)
    agent.load(args.checkpoint)
    print(f"Loaded checkpoint from {args.checkpoint}")
    
    # 创建评估器
    evaluator = Evaluator(env, agent, device=device)
    
    # 评估
    print(f"\nEvaluating for {args.num_episodes} episodes...")
    results = evaluator.evaluate(
        num_episodes=args.num_episodes,
        deterministic=args.deterministic,
        render=args.render
    )
    
    # 打印结果
    print("\n" + "="*50)
    print("Evaluation Results")
    print("="*50)
    print(f"Mean Reward:   {results['mean_reward']:.4f} ± {results['std_reward']:.4f}")
    print(f"Min Reward:    {results['min_reward']:.4f}")
    print(f"Max Reward:    {results['max_reward']:.4f}")
    print(f"Mean Length:   {results['mean_length']:.2f}")
    print(f"Success Rate:  {results['success_rate']:.2%}")
    print("="*50)
    
    # 保存结果
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, "eval_results.npy")
    np.save(save_path, results)
    print(f"\nResults saved to {save_path}")
    
    # 关闭环境
    env.close()


if __name__ == "__main__":
    main()
