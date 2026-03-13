# Author: HeFeiWang
# Date: 20260122
# Visualization functions related to cdf.

import os
import sys
import json
import argparse
import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation

# Add paths
# CUR_PATH使用父目录，因为这个文件放在debugging目录下，而相关的模型和数据都在上一级目录中
CUR_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(CUR_PATH)
sys.path.append(os.path.join(CUR_PATH, '../../RDF'))
sys.path.append(os.path.join(CUR_PATH, '../../RDF/panda_layers'))

# Import required modules
from para_nn_cdf_v2 import CDF_V2
from parallel_robot_layer import ParallelRobotLayer
from mlp import MLPRegression
from multi_finger_cdf import MultiFingerCDF

def generate_contact_configurations(
    robot='leaphand',
    obj_mesh_path=None,
    obj_position=None,
    obj_orientation=None,
    obj_scale=1.0,
    num_samples=1000,
    num_configs=50,
    num_iterations=100,
    lr=0.01,
    epoch=None,
    device='cuda',
    output_file='contact_configs.json'
):
    """
    在空间中放置物体，采样点，进行CDF query，投影机器人构型到零测集
    
    Args:
        robot: 机器人类型 ('leaphand', 'panda', 'dexhand')
        obj_mesh_path: 物体mesh路径
        obj_position: 物体位置 [x, y, z]
        obj_orientation: 物体姿态 [roll, pitch, yaw] (弧度)
        obj_scale: 物体缩放
        num_samples: 从物体上采样的点数
        num_configs: 要生成的机器人构型数量
        num_iterations: 投影迭代次数
        lr: 投影学习率
        epoch: 加载模型的epoch
        device: 'cuda' or 'cpu'
        output_file: 输出JSON文件路径
        
    Returns:
        result_data: 包含所有构型和物体信息的字典
    """
    
    print(f"\n{'='*60}")
    print(f"生成接触构型")
    print(f"{'='*60}")
    print(f"机器人: {robot}")
    print(f"物体: {obj_mesh_path}")
    print(f"构型数量: {num_configs}, 采样点数: {num_samples}")
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # 默认物体位置和姿态
    if obj_position is None:
        obj_position = [0.0, 0.0, 0.1]
    if obj_orientation is None:
        obj_orientation = [0.0, 0.0, 0.0]
    
    # 加载多手指CDF模型
    print(f"\n加载多手指CDF模型...")
    multi_cdf = MultiFingerCDF(robot=robot, device=device)
    model_dir = os.path.join(CUR_PATH, f'model_dict/{robot}')
    multi_cdf.load_models(model_dir, epoch=epoch)
    
    total_dof = multi_cdf.num_fingers * multi_cdf.config['dof_per_finger']
    print(f"机器人总自由度: {total_dof} ({multi_cdf.num_fingers} fingers × {multi_cdf.config['dof_per_finger']} dof)")
    
    # 加载物体mesh
    print(f"\n加载物体mesh: {obj_mesh_path}")
    obj_mesh = trimesh.load(obj_mesh_path)
    obj_mesh.apply_scale(obj_scale)
    
    # 应用物体变换
    obj_transform = np.eye(4)
    obj_transform[:3, :3] = Rotation.from_euler('xyz', obj_orientation).as_matrix()
    obj_transform[:3, 3] = obj_position
    obj_mesh.apply_transform(obj_transform)
    
    print(f"物体范围: {obj_mesh.bounds}")
    
    # 从物体表面采样点
    print(f"\n从物体采样 {num_samples} 个点...")
    sampled_points, face_indices = trimesh.sample.sample_surface(obj_mesh, num_samples)
    sampled_points = torch.tensor(sampled_points, dtype=torch.float32).to(device)
    
    print(f"采样点范围: [{sampled_points.min(dim=0)[0].cpu().numpy()}, "
          f"{sampled_points.max(dim=0)[0].cpu().numpy()}]")
    
    # 随机初始化机器人构型
    print(f"\n生成 {num_configs} 个随机初始构型...")
    
    # 初始化base transformation
    base_init = torch.eye(4).unsqueeze(0).repeat(num_configs, 1, 1).to(device)  # [num_configs, 4, 4]，使用repeat而不是expand
    
    # 随机位置 [-0.1, 0.1]
    base_init[:, :3, 3] = torch.rand(num_configs, 3).to(device) * 0.2 - 0.1
    
    # 随机姿态 [-pi/2, pi/2]
    base_ori_init = torch.rand(num_configs, 3).to(device) * np.pi - (np.pi / 2)
    rotation_matrices = torch.tensor(
        Rotation.from_euler('xyz', base_ori_init.cpu().numpy()).as_matrix(), 
        dtype=torch.float32
    ).to(device)
    base_init[:, :3, :3] = rotation_matrices
    
    # 随机初始化关节角度
    q_init = torch.rand(num_configs, total_dof).to(device) * 2 - 1  # [-1, 1]
    q_init = q_init * 1.5  # 扩大到 [-1.5, 1.5]
    
    # 扩展采样点为batch
    points_batch = sampled_points.unsqueeze(0).expand(num_configs, -1, -1)  # [num_configs, num_samples, 3]
    
    # 投影到零测集
    print(f"\n投影到CDF零测集...")
    q_final, base_final, cdf_final = multi_cdf.project_to_zero_level_set(
        q_init, 
        base_init,
        points_batch, 
        max_iters=num_iterations, 
        step_scale=lr,  # 使用step_scale代替lr
        threshold=1e-4
    )
    
    # 最终评估
    print(f"\n最终评估...")
    final_distances = cdf_final.abs().mean(dim=1).cpu().numpy()  # [num_configs]
    
    print(f"最终距离统计:")
    print(f"  均值: {final_distances.mean():.6f}")
    print(f"  标准差: {final_distances.std():.6f}")
    print(f"  最小值: {final_distances.min():.6f}")
    print(f"  最大值: {final_distances.max():.6f}")
    
    # 准备保存数据
    print(f"\n准备保存数据...")
    
    # 获取关节名称
    joint_names = []
    for finger_idx in range(multi_cdf.num_fingers):
        finger_name = multi_cdf.config['finger_names'][finger_idx]
        for dof_idx in range(multi_cdf.config['dof_per_finger']):
            joint_names.append(f'joint_{finger_idx*multi_cdf.config["dof_per_finger"] + dof_idx}')
    
    # 构建结果数据
    result_data = {
        'robot_name': robot,
        'num_fingers': multi_cdf.num_fingers,
        'dof': total_dof,
        'joint_names': joint_names,
        'object': {
            'mesh_path': obj_mesh_path,
            'position': obj_position,
            'orientation': obj_orientation,  # [roll, pitch, yaw]
            'scale': obj_scale,
            'transform': obj_transform.tolist()
        },
        'configurations': []
    }
    
    # 添加每个构型
    for i in range(num_configs):
        base_transform = base_final[i].detach().cpu().numpy()
        config = {
            'id': i,
            'joint_angles': q_final[i].detach().cpu().numpy().tolist(),
            'base_position': base_transform[:3, 3].tolist(),  # [x, y, z]
            'base_orientation': Rotation.from_matrix(base_transform[:3, :3]).as_quat().tolist(),  # [x, y, z, w]
            'base_transform': base_transform.tolist(),  # 完整的4x4变换矩阵
            'distance': float(final_distances[i])
        }
        result_data['configurations'].append(config)
    
    # 保存为JSON
    output_path = os.path.join(CUR_DIR, output_file)
    with open(output_path, 'w') as f:
        json.dump(result_data, f, indent=2)
    
    print(f"\n结果已保存到: {output_path}")
    print(f"{'='*60}\n")
    
    return result_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CDF接触构型可视化')
    parser.add_argument('--robot', type=str, default='leaphand',
                       choices=['leaphand', 'panda', 'dexhand'],
                       help='机器人类型')
    parser.add_argument('--obj_mesh', type=str, 
                       default='/workspace/cdf/models/ACE_Coffee_Mug_Kristen_16_oz_cup/meshes/model.obj',
                       help='物体mesh路径')
    parser.add_argument('--obj_position', type=float, nargs=3,
                       default=[0.0, 0.0, 0.0],
                       help='物体位置 [x y z]')
    parser.add_argument('--obj_orientation', type=float, nargs=3,
                       default=[0.0, 0.0, 0.0],
                       help='物体姿态 [roll pitch yaw] (弧度)')
    parser.add_argument('--obj_scale', type=float, default=1.0,
                       help='物体缩放因子')
    parser.add_argument('--num_samples', type=int, default=1000,
                       help='从物体采样的点数')
    parser.add_argument('--num_configs', type=int, default=50,
                       help='生成的构型数量')
    parser.add_argument('--num_iterations', type=int, default=100,
                       help='投影迭代次数')
    parser.add_argument('--lr', type=float, default=0.8,
                       help='投影步长缩放因子 (0-1)')
    parser.add_argument('--epoch', type=int, default=None,
                       help='加载模型的epoch')
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备 (cuda/cpu)')
    parser.add_argument('--output', type=str, default='contact_configs.json',
                       help='输出文件名')
    parser.add_argument('--config_file', type=str, default='contact_configs.json',
                       help='用于可视化的配置文件')
    
    args = parser.parse_args()
    
    generate_contact_configurations(
        robot=args.robot,
        obj_mesh_path=args.obj_mesh,
        obj_position=args.obj_position,
        obj_orientation=args.obj_orientation,
        obj_scale=args.obj_scale,
        num_samples=args.num_samples,
        num_configs=args.num_configs,
        num_iterations=args.num_iterations,
        lr=args.lr,
        epoch=args.epoch,
        device=args.device,
        output_file=args.output
    )