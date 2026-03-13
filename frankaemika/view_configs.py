#!/usr/bin/env python3
"""
简化版可视化 - 不依赖ROS2
直接在终端中显示构型信息，并生成可视化脚本
"""

import json
import argparse
import os


def visualize_configs_simple(config_file):
    """
    简单的交互式可视化 - 在终端显示构型信息
    """
    
    print(f"\n{'='*60}")
    print(f"接触构型浏览器")
    print(f"{'='*60}")
    
    # 加载数据
    with open(config_file, 'r') as f:
        data = json.load(f)
    
    robot_name = data['robot_name']
    configurations = data['configurations']
    num_configs = len(configurations)
    joint_names = data['joint_names']
    
    print(f"机器人: {robot_name}")
    print(f"构型数量: {num_configs}")
    print(f"关节数: {len(joint_names)}")
    print(f"\n物体信息:")
    print(f"  路径: {data['object']['mesh_path']}")
    print(f"  位置: {data['object']['position']}")
    print(f"  姿态: {data['object']['orientation']}")
    print(f"  缩放: {data['object']['scale']}")
    
    print(f"\n{'='*60}")
    print(f"控制: n(下一个) p(上一个) q(退出) s(保存当前) d(详细信息)")
    print(f"{'='*60}\n")
    
    current_idx = 0
    
    def show_config(idx):
        """显示构型信息"""
        config = configurations[idx]
        print(f"\n{'='*60}")
        print(f"构型 {idx+1}/{num_configs}")
        print(f"{'='*60}")
        print(f"ID: {config['id']}")
        print(f"距离: {config['distance']:.6f}")
        print(f"\n关节角度:")
        for i, (name, angle) in enumerate(zip(joint_names, config['joint_angles'])):
            print(f"  {name}: {angle:.4f} rad ({angle*180/3.14159:.2f}°)")
        print(f"{'='*60}")
    
    def save_single_config(idx, output_file):
        """保存单个构型为独立文件"""
        config = configurations[idx]
        single_data = {
            'robot_name': robot_name,
            'joint_names': joint_names,
            'configuration': config,
            'object': data['object']
        }
        with open(output_file, 'w') as f:
            json.dump(single_data, f, indent=2)
        print(f"已保存到: {output_file}")
    
    # 显示第一个构型
    show_config(current_idx)
    
    # 交互循环
    while True:
        try:
            cmd = input("\n命令 (n/p/q/s/d): ").strip().lower()
            
            if cmd == 'n':
                current_idx = (current_idx + 1) % num_configs
                show_config(current_idx)
            
            elif cmd == 'p':
                current_idx = (current_idx - 1) % num_configs
                show_config(current_idx)
            
            elif cmd == 'q':
                print("退出...")
                break
            
            elif cmd == 's':
                output = f"config_{current_idx:03d}.json"
                save_single_config(current_idx, output)
            
            elif cmd == 'd':
                # 显示详细统计
                distances = [c['distance'] for c in configurations]
                print(f"\n{'='*60}")
                print(f"全局统计")
                print(f"{'='*60}")
                print(f"距离统计:")
                print(f"  均值: {sum(distances)/len(distances):.6f}")
                print(f"  最小: {min(distances):.6f}")
                print(f"  最大: {max(distances):.6f}")
                print(f"  中位数: {sorted(distances)[len(distances)//2]:.6f}")
                print(f"{'='*60}")
            
            elif cmd == '':
                # 空输入，默认下一个
                current_idx = (current_idx + 1) % num_configs
                show_config(current_idx)
            
            else:
                print(f"未知命令: {cmd}")
                print("可用命令: n(下一个) p(上一个) q(退出) s(保存) d(统计)")
        
        except KeyboardInterrupt:
            print("\n\n退出...")
            break
        except EOFError:
            print("\n\n退出...")
            break
    
    print("\n浏览结束\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='简单的构型浏览器')
    parser.add_argument('--config_file', type=str, default='contact_configs.json',
                       help='配置文件路径')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config_file):
        print(f"错误: 找不到文件 {args.config_file}")
        print("请先运行生成命令:")
        print("  python3 visualize_cdf.py --mode generate")
    else:
        visualize_configs_simple(args.config_file)
