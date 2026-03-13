#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Test script for QP motion planner in 2D
# usage: python test_qp_planner.py
# -----------------------------------------------------------------------------

import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import sys

CUR_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(CUR_PATH)

from robot2D_torch import Robot2D
from primitives2D_torch import Circle, Box
from mlp import MLPRegression
from qp_mp_tao_2d import build_planner_2d

PI = np.pi


def test_qp_planner_simple():
    """简单测试QP规划器的基本功能"""
    print("="*60)
    print("Test 1: Simple QP Planner Test")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 创建2连杆机器人
    link_lengths = torch.tensor([[2.0, 2.0]]).float()
    link_parent_map = {1: 0, 2: 1}
    init_states = torch.zeros((1, 2))
    robot = Robot2D(num_links=2, init_states=init_states, link_lengths=link_lengths,
                    link_parent_map=link_parent_map, device=device)
    print(f"Created robot with {robot.num_links} links")
    
    # 创建一个简单的MLP模型 (或加载训练好的模型)
    model_path = os.path.join(CUR_PATH, 'model/siren_lr1e4_eik1_ep5000_model22.pth')
    if os.path.exists(model_path):
        print(f"Loading trained model from {model_path}")
        cdf_model = torch.load(model_path).to(device)
    else:
        print("No trained model found, using random initialized model")
        cdf_model = MLPRegression(input_dims=4, output_dims=1, 
                                  mlp_layers=[128, 64, 32],
                                  skips=[], act_fn=torch.nn.ReLU, nerf=True).to(device)
    
    # 创建QP规划器
    planner = build_planner_2d(robot, cdf_model, device, dt=0.05, cons_u=1.5, 
                               solver='ipopt', safety_buffer=0.05)
    print("QP planner created successfully")
    
    # 创建障碍物和目标
    obs_objs = [Circle(center=torch.tensor([1.5, 1.0]).to(device), radius=0.3, 
                      attract=False, device=device)]
    targ_objs = [Circle(center=torch.tensor([2.5, 0.0]).to(device), radius=0.2, 
                       attract=True, device=device)]
    
    print(f"Created {len(obs_objs)} obstacles and {len(targ_objs)} targets")
    
    # 初始配置
    q_start = torch.tensor([0.0, 0.0]).to(device)
    print(f"Starting configuration: {q_start.cpu().numpy()}")
    
    # 执行几步规划
    q_trajectory = [q_start.cpu().numpy()]
    q_current = q_start.clone()
    
    max_steps = 50
    print(f"\nRunning {max_steps} planning steps...")
    
    for step in range(max_steps):
        try:
            q_next = planner.step(q_current, obs_objs, targ_objs)
            q_trajectory.append(q_next.cpu().numpy())
            
            delta = torch.norm(q_next - q_current)
            if step % 10 == 0:
                print(f"  Step {step}: q = {q_next.cpu().numpy()}, delta = {delta:.6f}")
            
            if delta < 1e-3:
                print(f"  Converged at step {step}")
                break
            
            q_current = q_next
        except Exception as e:
            print(f"  Error at step {step}: {e}")
            break
    
    q_trajectory = np.array(q_trajectory)
    print(f"\nCompleted {len(q_trajectory)} steps")
    print(f"Final configuration: {q_trajectory[-1]}")
    
    # 简单可视化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # C空间轨迹
    print(f'shape of q_trajectory: {q_trajectory.shape}')
    ax1.plot(q_trajectory[:, 0], q_trajectory[:, 1], 'b-o', markersize=3, linewidth=1.5)
    ax1.plot(q_trajectory[0, 0], q_trajectory[0, 1], 'go', markersize=10, label='Start')
    ax1.plot(q_trajectory[-1, 0], q_trajectory[-1, 1], 'r*', markersize=15, label='End')
    ax1.set_xlabel('Joint 1 [rad]')
    ax1.set_ylabel('Joint 2 [rad]')
    ax1.set_title('Configuration Space Trajectory')
    ax1.grid(True)
    ax1.legend()
    ax1.axis('equal')
    
    # 关节角度随时间变化
    time_steps = np.arange(len(q_trajectory))
    ax2.plot(time_steps, q_trajectory[:, 0], 'b-', label='Joint 1', linewidth=2)
    ax2.plot(time_steps, q_trajectory[:, 1], 'r-', label='Joint 2', linewidth=2)
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Joint Angle [rad]')
    ax2.set_title('Joint Angles vs Time')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    save_path = os.path.join(CUR_PATH, 'image/test_qp_simple.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization to {save_path}")
    
    print("\n" + "="*60)
    print("Test 1 completed successfully!")
    print("="*60 + "\n")
    
    return q_trajectory


def test_qp_planner_with_cdf():
    """使用实际的CDF类进行更完整的测试"""
    print("="*60)
    print("Test 2: QP Planner with CDF Integration")
    print("="*60)
    
    try:
        from cdf import CDF2D
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 创建CDF实例
        cdf = CDF2D(device)
        print(f"Created CDF2D with {cdf.num_joints} joints")
        
        # 创建场景
        obs_objs = [Circle(center=torch.tensor([1.5, 0.8]).to(device), radius=0.3, 
                          attract=False, device=device),
                   Box(center=torch.tensor([0.5, -1.2]).to(device), w=0.4, h=0.4, 
                       attract=False, device=device)]
        targ_objs = [Circle(center=torch.tensor([2.0, -1.5]).to(device), radius=0.2, 
                           attract=True, device=device)]
        
        # 加载或创建模型
        model_path = os.path.join(CUR_PATH, 'model/model.pth')
        if os.path.exists(model_path):
            print(f"Loading trained model from {model_path}")
            cdf_model = torch.load(model_path).to(device)
        else:
            print("No trained model found, using random initialized model")
            cdf_model = MLPRegression(input_dims=4, output_dims=1, 
                                      mlp_layers=[1024, 512, 256, 128, 128],
                                      skips=[], act_fn=torch.nn.ReLU, nerf=True).to(device)
        
        # 创建规划器
        planner = build_planner_2d(cdf.robot, cdf_model, device, dt=0.05, cons_u=1.0,
                                   solver='ipopt', safety_buffer=0.05)
        
        # 执行规划
        q_start = torch.tensor([0.0, 1.0]).to(device)
        q_trajectory = [q_start.cpu().numpy()]
        q_current = q_start.clone()
        
        max_steps = 200
        print(f"Running up to {max_steps} planning steps...")
        
        for step in range(max_steps):
            q_next = planner.step(q_current, obs_objs, targ_objs)
            q_trajectory.append(q_next.cpu().numpy())
            
            delta = torch.norm(q_next - q_current)
            if step % 30 == 0:
                print(f"  Step {step}: delta = {delta:.6f}")
            
            if delta < 1e-3:
                print(f"  Converged at step {step}")
                break
            
            q_current = q_next
        
        q_trajectory = np.array(q_trajectory)
        print(f"Completed {len(q_trajectory)} steps")
        
        # 完整可视化
        fig = plt.figure(figsize=(18, 6))
        
        # C空间 + CDF
        ax1 = plt.subplot(1, 3, 1)
        cdf.plot_cdf(ax=ax1, obj_lists=obs_objs + targ_objs)
        ax1.plot(q_trajectory[:, 0], q_trajectory[:, 1], 'r-', linewidth=2, alpha=0.8, label='QP Trajectory')
        ax1.plot(q_trajectory[0, 0], q_trajectory[0, 1], 'go', markersize=10, label='Start')
        ax1.plot(q_trajectory[-1, 0], q_trajectory[-1, 1], 'r*', markersize=15, label='End')
        ax1.set_title('QP Planning in C-Space with CDF')
        ax1.legend()
        
        # 任务空间
        ax2 = plt.subplot(1, 3, 2)
        cdf.plot_objects(ax2, obs_objs + targ_objs)
        q_traj_torch = torch.from_numpy(q_trajectory).float().to(device).unsqueeze(1)
        cdf.robot.plot_trajectory(ax=ax2, joint_trajectory=q_traj_torch)
        ax2.set_title('Task Space Trajectory')
        ax2.set_xlim(-4.0, 4.0)
        ax2.set_ylim(-4.0, 4.0)
        ax2.set_aspect('equal', 'box')
        
        # 关节角度
        ax3 = plt.subplot(1, 3, 3)
        time_steps = np.arange(len(q_trajectory))
        for i in range(q_trajectory.shape[1]):
            ax3.plot(time_steps, q_trajectory[:, i], label=f'Joint {i+1}', linewidth=2)
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('Joint Angle [rad]')
        ax3.set_title('Joint Angles vs Time')
        ax3.grid(True)
        ax3.legend()
        
        plt.tight_layout()
        save_path = os.path.join(CUR_PATH, 'image/test_qp_with_cdf.png')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
        
        print("\n" + "="*60)
        print("Test 2 completed successfully!")
        print("="*60 + "\n")
        
        return q_trajectory
        
    except ImportError as e:
        print(f"Cannot import CDF2D: {e}")
        print("Skipping Test 2")
        return None


if __name__ == '__main__':
    print("\n" + "="*60)
    print("QP Motion Planner 2D - Test Suite")
    print("="*60 + "\n")
    
    # Test 1: Simple test
    try:
        q_traj1 = test_qp_planner_simple()
    except Exception as e:
        print(f"Test 1 failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Integration with CDF
    try:
        q_traj2 = test_qp_planner_with_cdf()
    except Exception as e:
        print(f"Test 2 failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)
    
    # Show plots if available, set timeout to prevent hanging
    timeout = 30  # seconds
    try:
        plt.show(block=False)
        plt.pause(timeout)
        plt.close('all')
    except Exception as e:
        print(f"Error showing plots: {e}")
