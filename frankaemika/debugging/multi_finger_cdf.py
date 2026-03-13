"""
Multi-Finger CDF Model Manager
处理多个手指的CDF模型加载和查询
"""

import os
import re
import torch
import numpy as np
from mlp import MLPRegression


def _remap_legacy_state_dict(state_dict, input_dims, nerf=True):
    """
    Remap old 'net.X.linear' key format to current 'layers.0.X.0' format,
    and infer the mlp_layers list from the checkpoint's weight shapes.

    Returns (new_state_dict, mlp_layers) where mlp_layers is None if remapping
    was not needed (i.e. the state_dict is already in the new format).
    """
    if not any(k.startswith('net.') for k in state_dict):
        return state_dict, None  # already in new format

    # Collect all layer indices present in the checkpoint
    indices = sorted({
        int(m.group(1))
        for k in state_dict
        for m in [re.match(r'net\.(\d+)\.', k)]
        if m
    })
    num_layers = len(indices)  # includes the output layer

    # Infer hidden sizes from weight shapes
    mlp_layers = []
    for i in indices[:-1]:
        key = f'net.{i}.linear.weight'
        mlp_layers.append(state_dict[key].shape[0])

    # Remap keys
    new_sd = {}
    for i in indices[:-1]:
        new_sd[f'layers.0.{i}.0.weight'] = state_dict[f'net.{i}.linear.weight']
        new_sd[f'layers.0.{i}.0.bias']   = state_dict[f'net.{i}.linear.bias']
    last = indices[-1]
    new_sd[f'layers.0.{last}.0.weight'] = state_dict[f'net.{last}.weight']
    new_sd[f'layers.0.{last}.0.bias']   = state_dict[f'net.{last}.bias']

    return new_sd, mlp_layers


class MultiFingerCDF:
    """管理多个手指的CDF模型"""
    
    def __init__(self, robot='leaphand', device='cuda'):
        """
        初始化多手指CDF模型
        
        Args:
            robot: 机器人类型 ('leaphand', 'allegro', etc.)
            device: 'cuda' or 'cpu'
        """
        self.robot = robot
        self.device = device
        self.cdf_models = []
        self.num_fingers = 0
        
        # 机器人配置
        self.robot_configs = {
            'leaphand': {
                'num_fingers': 4,
                'finger_names': ['finger0', 'finger1', 'finger2', 'thumb'],
                'dof_per_finger': 4,  # 每个手指的自由度
                'model_files': {
                    0: 'leaphand_finger0_mlp_base.pt',
                    1: 'leaphand_finger1_mlp_base.pt', 
                    2: 'leaphand_finger2_mlp_base.pt',
                    3: 'leaphand_finger3_mlp_base.pt'
                },
                'input_dims': 13,  # 4 (joint angles) + 6 (base transformation) + 3 (point position)
                'output_dims': 1
            },
            'allegro': {
                'num_fingers': 4,
                'finger_names': ['index', 'middle', 'ring', 'thumb'],
                'dof_per_finger': 4,
                'input_dims': 13,
                'output_dims': 1
            }
        }
        
        if robot not in self.robot_configs:
            raise ValueError(f"Unsupported robot: {robot}. Supported: {list(self.robot_configs.keys())}")
        
        self.config = self.robot_configs[robot]
        self.num_fingers = self.config['num_fingers']
        
    def load_models(self, model_dir, epoch=None):
        """
        加载所有手指的CDF模型
        
        Args:
            model_dir: 模型文件所在目录
            epoch: 要加载的epoch，如果为None则自动选择最后一个epoch
        """
        print(f"\n加载{self.robot}的CDF模型...")
        print(f"模型目录: {model_dir}")
        
        for finger_idx in range(self.num_fingers):
            finger_name = self.config['finger_names'][finger_idx]
            
            # 构建模型文件路径
            if 'model_files' in self.config:
                model_file = self.config['model_files'][finger_idx]
            else:
                model_file = f"{self.robot}_{finger_name}_mlp_base.pt"
            
            model_path = os.path.join(model_dir, model_file)
            
            if not os.path.exists(model_path):
                print(f"  警告: 模型文件不存在: {model_path}")
                # 尝试通用文件名
                alt_path = os.path.join(model_dir, f"finger{finger_idx}_mlp_base.pt")
                if os.path.exists(alt_path):
                    model_path = alt_path
                else:
                    raise FileNotFoundError(f"找不到手指{finger_idx}的模型文件")
            
            print(f"  加载 Finger {finger_idx} ({finger_name}): {os.path.basename(model_path)}")
            
            # 加载state_dict
            state_dict_all = torch.load(model_path, map_location='cpu', weights_only=False)
            
            # 处理不同的state_dict格式
            if isinstance(state_dict_all, dict):
                # 如果是字典，检查是否有epoch keys
                if epoch is not None:
                    if epoch in state_dict_all:
                        state_dict = state_dict_all[epoch]
                    else:
                        raise KeyError(f"Epoch {epoch} not found in model file. Available: {list(state_dict_all.keys())[-5:]}...")
                else:
                    # 自动选择最后一个epoch
                    epochs = [k for k in state_dict_all.keys() if isinstance(k, int)]
                    if epochs:
                        epoch = max(epochs)
                        state_dict = state_dict_all[epoch]
                        print(f"    使用 epoch {epoch}")
                    else:
                        # 假设整个字典就是state_dict
                        state_dict = state_dict_all
                        print(f"    直接使用state_dict")
            else:
                state_dict = state_dict_all
            
            # Remap legacy key format if needed, and infer architecture
            state_dict, legacy_mlp_layers = _remap_legacy_state_dict(
                state_dict, self.config['input_dims'], nerf=True
            )
            mlp_layers = legacy_mlp_layers if legacy_mlp_layers is not None else [1024, 512, 256, 128, 128]
            if legacy_mlp_layers is not None:
                print(f"    (legacy checkpoint, remapped keys, inferred mlp_layers={mlp_layers})")

            # 创建模型
            model = MLPRegression(
                input_dims=self.config['input_dims'],
                output_dims=self.config['output_dims'],
                mlp_layers=mlp_layers,
                skips=[],
                act_fn=torch.nn.ReLU,
                nerf=True
            )
            
            # 加载权重
            try:
                model.load_state_dict(state_dict)
                print(f"    ✓ 成功加载")
            except Exception as e:
                print(f"    ✗ 加载失败: {e}")
                raise
            
            model.to(self.device)
            model.eval()
            self.cdf_models.append(model)
        
        print(f"✓ 成功加载 {len(self.cdf_models)} 个手指的CDF模型\n")
    
    def query(self, q, base, points, finger_idx=None):
        """
        查询CDF值
        
        Args:
            q: 关节角度 [batch_size, num_fingers * dof_per_finger] 或 [batch_size, dof_per_finger] for single finger
            base: 机器人基座初始变换 [batch_size, 4, 4]
            points: 查询点 [batch_size, num_points, 3]
            finger_idx: 如果指定，只查询该手指；否则查询所有手指
            
        Returns:
            cdf_values: [batch_size, num_points] or [batch_size, num_fingers, num_points]
        """
        if len(self.cdf_models) == 0:
            raise RuntimeError("模型未加载，请先调用 load_models()")
        
        batch_size = q.shape[0]
        num_points = points.shape[1]
        
        # 不使用 torch.no_grad()，以保留梯度用于优化
        if finger_idx is not None:
            # 查询单个手指
            q_finger = q[:, finger_idx*self.config['dof_per_finger']:(finger_idx+1)*self.config['dof_per_finger']]
            # q_finger: [batch_size, dof_per_finger]
            # points: [batch_size, num_points, 3]
            
            # 扩展q到每个点
            q_expanded = q_finger.unsqueeze(1).expand(-1, num_points, -1)  # [batch_size, num_points, dof_per_finger]
            # base from [batch_size, 4,4] to [batch_size, num_points, 6] (position + flattened rotation)
            base_translation = base[:, :3, 3]  # [batch_size, 3]
            base_rotation = base[:, :3, :3]  # [batch_size, 3, 3]
            base_euler = roatation_matrix_to_euler(base_rotation)  # [batch_size, 3]
            
            base_expanded = torch.cat([base_translation, base_euler], dim=-1)  # [batch_size, 6]
            base_expanded = base_expanded.unsqueeze(1).expand(-1, num_points, -1)  # [batch_size, num_points, 6]
            # 拼接 [q, point]
            model_input = torch.cat([q_expanded, base_expanded, points], dim=-1)  # [batch_size, num_points, dof_per_finger + 6 + 3]
            
            # Reshape for MLP
            model_input = model_input.reshape(-1, self.config['input_dims'])  # [batch_size*num_points, 11]
            
            # 查询
            cdf_values = self.cdf_models[finger_idx](model_input)  # [batch_size*num_points, 1]
            cdf_values = cdf_values.reshape(batch_size, num_points)  # [batch_size, num_points]
            
            return cdf_values
        else:
            # 查询所有手指
            all_cdf_values = []
            for i in range(self.num_fingers):
                cdf_values = self.query(q, base, points, finger_idx=i)
                all_cdf_values.append(cdf_values)
            
            # Stack: [batch_size, num_fingers, num_points]
            return torch.stack(all_cdf_values, dim=1)
    
    def get_min_cdf(self, q, base, points):
        """
        获取所有手指中的最小CDF值（最接近的手指）
        
        Args:
            q: [batch_size, total_dof]
            base: [batch_size, 4, 4] 机器人基座初始变换
            points: [batch_size, num_points, 3]
            
        Returns:
            min_cdf: [batch_size, num_points] 最小CDF值
            min_finger_idx: [batch_size, num_points] 对应的手指索引
        """
        all_cdf = self.query(q, base, points)  # [batch_size, num_fingers, num_points]
        min_cdf, min_finger_idx = torch.min(all_cdf, dim=1)  # [batch_size, num_points]
        return min_cdf, min_finger_idx
    
    def get_gradient(self, q, base, points, finger_idx=None):
        """
        获取CDF对关节角度的梯度
        
        Args:
            q: [batch_size, total_dof] 需要requires_grad=True
            base: [batch_size, 4, 4] 机器人基座初始变换
            points: [batch_size, num_points, 3]
            finger_idx: 指定手指或None（所有手指）
            
        Returns:
            gradients: [batch_size, num_points, total_dof] or per finger
        """
        if not q.requires_grad:
            q.requires_grad_(True)
        
        cdf_values = self.query(q, base, points, finger_idx=finger_idx)
        
        if finger_idx is not None:
            # Single finger gradient
            cdf_sum = cdf_values.sum()
            grad = torch.autograd.grad(cdf_sum, q, create_graph=True)[0]
            return grad
        else:
            # All fingers
            gradients = []
            for i in range(self.num_fingers):
                cdf_i = self.query(q, base, points, finger_idx=i)
                cdf_sum = cdf_i.sum()
                grad_i = torch.autograd.grad(cdf_sum, q, retain_graph=True, create_graph=True)[0]
                gradients.append(grad_i)
            return torch.stack(gradients, dim=1)
    
    def project_to_zero_level_set(self, q_init, base_init, points, max_iters=100, step_scale=0.8, threshold=1e-3):
        """
        将关节配置投影到CDF零测集（接触表面）
        
        利用CDF的性质：
        1. CDF梯度的模长为1（已归一化）
        2. CDF的值就是到零测集的距离
        3. 投影公式：q_new = q - cdf * grad_q, b_new = b - cdf * grad_b
        
        Args:
            q_init: [batch_size, total_dof] 初始关节角度
            base_init: [batch_size, 4, 4] 机器人基座初始变换
            points: [batch_size, num_points, 3] 查询点（通常是物体表面点）
            max_iters: 最大迭代次数
            step_scale: 步长缩放因子（0-1），用于防止overshooting
            threshold: 收敛阈值
            
        Returns:
            q_final: [batch_size, total_dof] 投影后的关节角度
            b_final: [batch_size, 4, 4] 投影后的base变换
            cdf_values: [batch_size, num_points] 最终CDF值
        """
        q = q_init.clone()
        b = base_init.clone()
        
        for iter_idx in range(max_iters):
            q.requires_grad_(True)
            b.requires_grad_(True)
            
            # 获取最小CDF值
            min_cdf, _ = self.get_min_cdf(q, b, points)  # [batch_size, num_points]
            
            # 取每个配置的最小CDF（最接近的点）
            # 或者取前k小的平均（更鲁棒）
            # 这里使用最小值，因为我们只需要至少一个点接触
            cdf_per_config, _ = min_cdf.min(dim=1)  # [batch_size]
            # 如果要更鲁棒，可以用：
            # k = min(10, min_cdf.shape[1])  # 取前10小的点
            # cdf_per_config = torch.topk(min_cdf, k, dim=1, largest=False)[0].mean(dim=1)
            
            # 计算梯度
            # 对q的梯度
            grad_q = torch.autograd.grad(
                cdf_per_config.sum(), 
                q, 
                retain_graph=True,
                create_graph=False
            )[0]  # [batch_size, total_dof]
            
            # 对base的梯度（需要特殊处理4x4矩阵）
            grad_b = torch.autograd.grad(
                cdf_per_config.sum(), 
                b, 
                retain_graph=False,
                create_graph=False
            )[0]  # [batch_size, 4, 4]
            
            with torch.no_grad():
                # CDF投影公式：x_new = x - cdf * grad
                # 扩展cdf_per_config的维度以匹配q和b
                cdf_expanded_q = cdf_per_config.unsqueeze(1)  # [batch_size, 1]
                cdf_expanded_b = cdf_per_config.view(-1, 1, 1)  # [batch_size, 1, 1]
                
                # 投影更新（带步长缩放）
                q = q - step_scale * cdf_expanded_q * grad_q
                b = b - step_scale * cdf_expanded_b * grad_b
                
                # 重新计算CDF用于监控
                min_cdf_check, _ = self.get_min_cdf(q, b, points)
                mean_cdf = min_cdf_check.abs().mean().item()
                
                if iter_idx % 10 == 0 or iter_idx == max_iters - 1:
                    print(f"  Iter {iter_idx:3d}: Mean |CDF| = {mean_cdf:.6f}, "
                          f"Max |CDF| = {min_cdf_check.abs().max().item():.6f}")
                
                if mean_cdf < threshold:
                    print(f"  ✓ 收敛 at iteration {iter_idx}, Mean |CDF| = {mean_cdf:.6f}")
                    break
        
        # 最终评估
        with torch.no_grad():
            min_cdf_final, _ = self.get_min_cdf(q, b, points)
        
        return q.detach(), b.detach(), min_cdf_final.detach()

def roatation_matrix_to_euler(R):
    """将旋转矩阵转换为欧拉角（ZYX顺序）(批量版)"""
    # TODO: 是zyx顺序吗？
    sy = torch.sqrt(R[:,0,0] * R[:,0,0] +  R[:,1,0] * R[:,1,0])
    singular = sy < 1e-6
    x = torch.atan2(R[:,2,1] , R[:,2,2])
    y = torch.atan2(-R[:,2,0], sy)
    z = torch.atan2(R[:,1,0], R[:,0,0])
    euler = torch.zeros(R.shape[0], 3).to(R.device)
    euler[~singular, 0] = x[~singular]
    euler[~singular, 1] = y[~singular]
    euler[~singular, 2] = z[~singular]
    # 处理奇异情况
    euler[singular, 0] = torch.atan2(-R[singular,1,2], R[singular,1,1])
    euler[singular, 1] = torch.atan2(-R[singular,2,0], sy[singular])
    euler[singular, 2] = 0
    return euler
    
if __name__ == '__main__':
    # 测试代码
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 创建多手指CDF
    multi_cdf = MultiFingerCDF(robot='leaphand', device=device)
    
    # 加载模型
    model_dir = '/workspace/cdf/frankaemika/model_dict/leaphand'
    multi_cdf.load_models(model_dir, epoch=49900)
    
    # 测试查询
    batch_size = 2
    num_points = 10
    total_dof = 16  # leaphand: 4 fingers * 4 dof
    
    q = torch.randn(batch_size, total_dof).to(device)
    points = torch.randn(batch_size, num_points, 3).to(device)
    
    print("\n测试CDF查询...")
    base_init = torch.eye(4).to(device).unsqueeze(0).expand(batch_size, -1, -1)
    cdf_values = multi_cdf.query(q, base_init, points)
    print(f"CDF values shape: {cdf_values.shape}")  # [batch_size, num_fingers, num_points]
    
    min_cdf, min_finger = multi_cdf.get_min_cdf(q, base_init, points)
    print(f"Min CDF shape: {min_cdf.shape}")  # [batch_size, num_points]
    print(f"Min finger idx shape: {min_finger.shape}")
    
    print("\n测试投影到零测集...")
    base_init = torch.eye(4).to(device).unsqueeze(0).expand(batch_size, -1, -1)
    q_proj, cdf_final = multi_cdf.project_to_zero_level_set(
        q,
        base_init,
        points,
        max_iters=50,
        lr=0.01
    )
    print(f"Projected q shape: {q_proj.shape}")
    print(f"Final CDF mean: {cdf_final.abs().mean().item():.6f}")
    
    print("\n✓ 测试完成！")
