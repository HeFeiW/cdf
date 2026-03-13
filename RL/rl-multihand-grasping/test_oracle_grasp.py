"""
test_oracle_grasp.py
简易测试程序：复刻正式实验代码逻辑
- 使用 PyBulletSim 封装
- 加载 LeapHand 手模型
- 使用 ObjectManager 管理物体
- 使用 HandController 控制手部
- Oracle 策略：手的中心靠近物体中心
"""

import os
import sys
import time
import numpy as np
import pybullet as p

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from simulation.pybullet_sim import PyBulletSim
from envs.object_manager import ObjectManager
from envs.hand_controller import HandController
from envs.utils import load_config


class OracleGraspTest:
    """
    Oracle 抓取测试
    
    复刻正式实验代码逻辑，使用模块化组件
    """
    
    def __init__(
        self,
        env_config_path: str = "configs/env_config.yaml",
        hand_config_path: str = "configs/hand_config.yaml"
    ):
        """
        初始化测试环境
        
        Args:
            env_config_path: 环境配置路径
            hand_config_path: 手部配置路径
        """
        # 加载配置
        self.env_config = load_config(env_config_path)
        self.hand_config = load_config(hand_config_path)
        
        # 修改配置以启用 GUI
        self.env_config["render"]["enable_gui"] = True
        
        # 初始化仿真环境（启用 GUI）
        print("Initializing PyBullet simulation with GUI...")
        self.sim = PyBulletSim(enable_gui=True)
        
        # 设置相机视角
        p.resetDebugVisualizerCamera(
            cameraDistance=0.8,
            cameraYaw=45,
            cameraPitch=-30,
            cameraTargetPosition=[0, 0, 0.45],
            physicsClientId=self.sim.physics_client
        )
        
        # 初始化物体管理器
        self.object_manager = ObjectManager(self.sim, self.env_config)
        
        # 初始化手部控制器
        self.hand_controller = HandController(self.sim, self.hand_config)
        
        # 随机数生成器
        self.np_random = np.random.default_rng(seed=42)
        
        # 状态
        self.target_initial_height = 0.0
        
    def reset(self):
        """重置环境"""
        print("\n" + "="*60)
        print("Resetting environment...")
        print("="*60)
        
        # 重置仿真
        self.sim.reset()
        
        # 加载场景
        print("Loading scene (plane + table)...")
        self.sim.load_scene(self.env_config)
        
        # 生成物体
        print("Spawning objects...")
        self.object_manager.spawn_objects(self.np_random)
        
        # 获取目标位置
        target_pos, _ = self.object_manager.get_target_pose()
        self.target_initial_height = target_pos[2]
        print(f"Target object at: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
        print(f"Number of obstacles: {self.object_manager.get_num_obstacles()}")
        
        # 重置手部
        print("Resetting hand controller...")
        self.hand_controller.reset()
        
        # 打印手部信息
        hand_id = self.hand_controller.get_hand_id()
        num_joints = p.getNumJoints(hand_id, physicsClientId=self.sim.physics_client)
        print(f"Hand loaded with {num_joints} joints")
        
        # 打印关节信息
        print("\nJoint information:")
        for i in range(min(num_joints, 10)):  # 只打印前10个
            joint_info = p.getJointInfo(hand_id, i, physicsClientId=self.sim.physics_client)
            joint_name = joint_info[1].decode('utf-8')
            joint_type = joint_info[2]
            link_name = joint_info[12].decode('utf-8')
            print(f"  Joint {i}: {joint_name} (type={joint_type}, link={link_name})")
        if num_joints > 10:
            print(f"  ... and {num_joints - 10} more joints")
        
        # 稳定仿真
        print("\nStabilizing simulation...")
        for _ in range(100):
            self.sim.step_simulation()
            time.sleep(1.0/480.0)
        
        print("Environment reset complete!")
        return target_pos
    
    def oracle_policy(self, target_pos: np.ndarray, phase: int) -> tuple:
        """
        Oracle 策略
        
        策略逻辑:
        - Phase 0: 手心水平移动到目标上方
        - Phase 1: 手心下降到抓取高度
        - Phase 2: 收拢手指抓取
        - Phase 3: 提升物体
        
        Args:
            target_pos: 目标位置 (3,)
            phase: 当前阶段
            
        Returns:
            delta_q: 关节位置增量 (22,)
            new_phase: 新的阶段
        """
        # 获取当前状态
        hand_center = self.hand_controller.get_hand_center()
        current_q = self.hand_controller.get_joint_positions()
        
        # 初始化增量
        delta_q = np.zeros(22)  # 6 base + 16 finger
        
        # 控制增益
        k_pos = 0.02  # 位置增益
        k_finger = 0.02  # 手指增益
        
        if phase == 0:
            # Phase 0: 水平接近目标上方
            target_above = target_pos.copy()
            target_above[2] = target_pos[2] + 0.15  # 目标上方 15cm
            
            diff = target_above - hand_center
            
            # 计算 base 位置增量 (前3维是 x, y, z)
            delta_q[0] = k_pos * diff[0]  # x
            delta_q[1] = k_pos * diff[1]  # y
            delta_q[2] = 0  # 保持高度
            
            # 检查是否到达
            if np.linalg.norm(diff[:2]) < 0.02:
                print("Phase 0 complete: Reached above target")
                return delta_q, 1
                
        elif phase == 1:
            # Phase 1: 下降到抓取高度
            target_grasp = target_pos.copy()
            target_grasp[2] = target_pos[2] + 0.08  # 目标上方 8cm
            
            diff = target_grasp - hand_center
            
            # 同时调整水平位置和下降
            delta_q[0] = k_pos * diff[0]
            delta_q[1] = k_pos * diff[1]
            delta_q[2] = k_pos * diff[2]
            
            # 检查是否到达
            if np.linalg.norm(diff) < 0.015:
                print("Phase 1 complete: Reached grasp height")
                return delta_q, 2
                
        elif phase == 2:
            # Phase 2: 收拢手指
            # 手指关节增量 (索引 6-21)
            for i in range(16):
                # 增加手指弯曲角度
                delta_q[6 + i] = k_finger
            
            # 经过一定时间后进入提升阶段
            # 这里通过外部计数器控制
            return delta_q, 2
            
        elif phase == 3:
            # Phase 3: 提升
            delta_q[2] = k_pos * 2  # 向上移动
            
        return delta_q, phase
    
    def check_grasp_success(self) -> bool:
        """检查是否成功抓取并提升"""
        target_pos, _ = self.object_manager.get_target_pose()
        height_gain = target_pos[2] - self.target_initial_height
        return height_gain > 0.05  # 提升超过 5cm
    
    def get_contact_info(self) -> dict:
        """获取接触信息"""
        hand_id = self.hand_controller.get_hand_id()
        target_id = self.object_manager.get_target_id()
        obstacle_ids = self.object_manager.get_obstacle_ids()
        
        # 与目标的接触
        target_contacts = self.sim.get_contacts(hand_id, [target_id])
        
        # 与障碍物的碰撞
        has_collision = self.sim.check_penetration(hand_id, obstacle_ids)
        
        return {
            "num_target_contacts": len(target_contacts),
            "has_obstacle_collision": has_collision
        }
    
    def visualize_debug_info(self, target_pos: np.ndarray, phase: int, step: int):
        """可视化调试信息"""
        hand_center = self.hand_controller.get_hand_center()
        
        # 画从手心到目标的线
        p.addUserDebugLine(
            hand_center.tolist(),
            target_pos.tolist(),
            lineColorRGB=[0, 1, 0],
            lineWidth=2,
            lifeTime=0.1,
            physicsClientId=self.sim.physics_client
        )
        
        # 显示状态文本
        if step % 50 == 0:
            dist = np.linalg.norm(hand_center - target_pos)
            contact_info = self.get_contact_info()
            
            print(f"Step {step:4d} | Phase {phase} | "
                  f"Dist: {dist:.3f} | "
                  f"Hand: [{hand_center[0]:.2f}, {hand_center[1]:.2f}, {hand_center[2]:.2f}] | "
                  f"Contacts: {contact_info['num_target_contacts']} | "
                  f"Collision: {contact_info['has_obstacle_collision']}")
    
    def run(self, max_steps: int = 2000):
        """
        运行测试
        
        Args:
            max_steps: 最大步数
        """
        # 重置环境
        target_pos = self.reset()
        
        print("\n" + "="*60)
        print("Starting Oracle Policy Grasp Test")
        print("="*60)
        print("\nPhase 0: Moving above target...")
        
        phase = 0
        phase_counter = 0
        success = False
        
        phase_names = [
            "Moving above target",
            "Descending to grasp",
            "Closing fingers",
            "Lifting object"
        ]
        
        for step in range(max_steps):
            # 获取当前目标位置（可能被移动）
            target_pos, _ = self.object_manager.get_target_pose()
            
            # Oracle 策略计算动作
            delta_q, new_phase = self.oracle_policy(target_pos, phase)
            
            # 检测阶段变化
            if new_phase != phase:
                phase = new_phase
                phase_counter = 0
                print(f"\nPhase {phase}: {phase_names[phase]}")
            
            phase_counter += 1
            
            # Phase 2（抓取）持续一段时间后进入 Phase 3
            if phase == 2 and phase_counter > 100:
                phase = 3
                phase_counter = 0
                print(f"\nPhase {phase}: {phase_names[phase]}")
            
            # 应用动作
            self.hand_controller.apply_delta_q(delta_q)
            
            # 仿真步进
            self.sim.step_simulation()
            time.sleep(1.0/480.0)
            
            # 可视化
            self.visualize_debug_info(target_pos, phase, step)
            
            # 检查成功
            if phase == 3 and self.check_grasp_success():
                success = True
                print("\n" + "="*60)
                print("SUCCESS! Object lifted above threshold!")
                print("="*60)
                break
        
        # 结果
        print("\n" + "="*60)
        print("Test Result:", "SUCCESS" if success else "FAILED")
        print("="*60)
        
        # 最终状态
        target_pos, _ = self.object_manager.get_target_pose()
        height_gain = target_pos[2] - self.target_initial_height
        print(f"Final target height gain: {height_gain:.3f} m")
        
        # 保持显示
        print("\nPress Enter to exit...")
        input()
        
        self.sim.close()


def main():
    """主函数"""
    print("="*60)
    print("Oracle Grasp Test - Using Formal Experiment Code Structure")
    print("="*60)
    print("\nThis test uses:")
    print("  - PyBulletSim: Simulation wrapper")
    print("  - ObjectManager: Object spawning")
    print("  - HandController: LeapHand control")
    print("  - Oracle Policy: Move hand center towards target")
    print()
    
    # 配置路径
    env_config = os.path.join(PROJECT_ROOT, "configs/env_config.yaml")
    hand_config = os.path.join(PROJECT_ROOT, "configs/hand_config.yaml")
    
    # 创建测试实例
    test = OracleGraspTest(
        env_config_path=env_config,
        hand_config_path=hand_config
    )
    
    # 运行测试
    test.run(max_steps=2000)


if __name__ == "__main__":
    main()
