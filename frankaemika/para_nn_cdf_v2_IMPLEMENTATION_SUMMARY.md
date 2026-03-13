# Implementation Summary: para_nn_cdf_v2.py

## 概述 (Overview)

`para_nn_cdf_v2.py` 是 `para_nn_cdf.py` 的增强版本，主要增加了以下功能：

1. **支持Base自由度**: 每根手指可以有6DoF的base pose (3平移 + 3旋转)
2. **支持多种网络架构**: 可选择MLP或SIREN作为神经网络
3. **专为LeapHand设计**: 支持训练4根手指，每根手指10DoF (4关节 + 6 base)

## 主要改动 (Key Changes)

### 1. 初始化函数增强
```python
def __init__(self, ..., network_type='mlp', use_base=False):
    self.network_type = network_type  # 'mlp' or 'siren'
    self.use_base = use_base          # 是否使用base DoF
```

### 2. 配置空间维度处理

**不带base**: `DoF = 4` (仅关节角度)
**带base**: `DoF + 6 = 10` (关节角度 + base pose)

关键函数都做了相应修改：
- `sample_q()`: 采样配置时包含base参数
- `compute_sdf()`: 将base参数转换为pose矩阵
- `decode_distance()`: 处理扩展的梯度维度

### 3. 网络创建工厂方法

```python
def create_network(self, input_dims, output_dims=1):
    if self.network_type == 'mlp':
        # MLP with ReLU: [1024, 512, 256, 128, 128]
        model = MLPRegression(...)
    elif self.network_type == 'siren':
        # SIREN with sine: [256, 256, 256, 256]
        model = Siren(...)
    return model
```

### 4. SDF计算支持Base

```python
def compute_sdf(self, x, q, return_index=False):
    if self.use_base:
        DoF = self.robot.serials[self.serial_idx].dof
        q_joint = q[:, :DoF]      # 前DoF维是关节角度
        q_base = q[:, DoF:]       # 后6维是base pose
        # 转换base为4x4变换矩阵
        pose = utils.q_to_poseMatrix(None, q_base).to(self.device).float()
    else:
        q_joint = q
        pose = torch.eye(4).unsqueeze(0).expand(len(q), 4, 4).float()
    
    # 使用pose和q_joint计算SDF
    d, _ = self.bp_sdf.get_serial_sdf_batch(x, pose, q_joint, ...)
```

### 5. 配置采样支持Base

```python
def sample_q(self, batch_q=None):
    serial = self.robot.serials[self.serial_idx]
    DoF = serial.dof
    
    # 采样关节角度
    q_joint = serial.theta_min + torch.rand(batch_q, DoF) * (serial.theta_max - serial.theta_min)
    
    if self.use_base:
        # 采样base DoF
        base_min = serial.theta_min_base  # [-0.5, -0.5, -0.5, -π, -π, -π]
        base_max = serial.theta_max_base  # [0.5, 0.5, 0.5, π, π, π]
        q_base = base_min + torch.rand(batch_q, 6) * (base_max - base_min)
        q_sampled = torch.cat([q_joint, q_base], dim=-1)  # (batch_q, DoF+6)
    else:
        q_sampled = q_joint  # (batch_q, DoF)
    
    return q_sampled
```

### 6. 前向传播处理

```python
def inference(self, x, q, model):
    DoF = self.robot.serials[self.serial_idx].dof
    total_dof = DoF + 6 if self.use_base else DoF
    
    # 准备输入
    x_cat = x.unsqueeze(1).expand(-1, len(q), -1).reshape(-1, 3)
    q_cat = q.unsqueeze(0).expand(len(x), -1, -1).reshape(-1, total_dof)
    inputs = torch.cat([x_cat, q_cat], dim=-1)  # (B, 3+total_dof)
    
    # 根据网络类型调用
    if self.network_type == 'siren':
        cdf_pred, _ = model.forward(inputs)  # SIREN返回两个值
    else:
        cdf_pred = model.forward(inputs)     # MLP返回一个值
    
    return cdf_pred
```

## 数据格式 (Data Format)

### 输入数据 (.pt文件)
```python
{
    'x': Tensor(G, 3),              # G个网格点的笛卡尔坐标
    'q': Tensor(G, M, D, D),        # G个点，每个点M个样本，D维配置
    'k': Tensor(G),                 # 网格点索引
}
```

其中：
- `G`: 网格点数量 (例如 20^3 = 8000)
- `M`: 每个link的最大采样数 (max_q_per_link = 100)
- `D`: 配置空间维度
  - 不带base: `D = DoF` (例如4)
  - 带base: `D = DoF + 6` (例如10)

### 网络输入输出

**输入**: `[x, y, z, q1, ..., qn, tx, ty, tz, rx, ry, rz]`
- LeapHand finger不带base: 维度 = 3 + 4 = 7
- LeapHand finger带base: 维度 = 3 + 4 + 6 = 13

**输出**: 标量距离值

## Base Pose表示 (Base Pose Representation)

6DoF base pose: `[tx, ty, tz, rx, ry, rz]`
- `tx, ty, tz`: 平移 (translation)
- `rx, ry, rz`: 旋转欧拉角 (rotation in Euler angles)

转换为4x4变换矩阵：
```python
pose = torch.zeros(B, 4, 4)
pose[:, :3, 3] = q_base[:, :3]    # translation
pose[:, :3, :3] = euler_to_matrix(q_base[:, 3:6])  # rotation
pose[:, 3, 3] = 1.0
```

## LeapHand训练流程 (Training Workflow)

### 步骤1: 数据生成
```bash
python parallel_data_generator.py --robot leaphand
# 生成4个文件:
# - data_with_base_dof_0.npy (食指)
# - data_with_base_dof_1.npy (中指)
# - data_with_base_dof_2.npy (无名指)
# - data_with_base_dof_3.npy (拇指)
```

### 步骤2: 训练单个手指
```bash
# MLP网络
python para_nn_cdf_v2.py \
    --train \
    --use_base \
    --robot leaphand \
    --serial_idx 0 \
    --network_type mlp \
    --data_path data_with_base_dof_0.pt \
    --model_dict leaphand_finger0_mlp_base.pt

# SIREN网络
python para_nn_cdf_v2.py \
    --train \
    --use_base \
    --robot leaphand \
    --serial_idx 0 \
    --network_type siren \
    --data_path data_with_base_dof_0.pt \
    --model_dict leaphand_finger0_siren_base.pt
```

### 步骤3: 训练所有手指
```bash
# 使用提供的脚本
bash train_leaphand_all_fingers.sh

# 或使用SIREN
NETWORK_TYPE=siren bash train_leaphand_all_fingers.sh
```

## 关键技术点 (Key Technical Points)

### 1. 维度一致性
- 训练时: 输入维度 = 3 + DoF + (6 if use_base else 0)
- 推理时: 必须使用相同的维度设置
- 梯度: 对所有维度（关节+base）都计算梯度

### 2. Pose变换
- Base pose存储为6维向量: `[tx, ty, tz, rx, ry, rz]`
- 使用时转换为4x4矩阵
- 在SDF计算中用于变换查询点

### 3. 网络选择
**MLP优点**:
- 训练快
- 推理快
- 通用性好

**SIREN优点**:
- 更光滑的距离场
- 更好的梯度性质
- 适合隐式表示

### 4. 梯度计算
```python
# 对于带base的情况，梯度包含关节和base的偏导数
grad = ∇_q d(x, q) 
     = [∂d/∂q1, ..., ∂d/∂qn, ∂d/∂tx, ∂d/∂ty, ∂d/∂tz, ∂d/∂rx, ∂d/∂ry, ∂d/∂rz]
```

## 性能考虑 (Performance Considerations)

1. **内存使用**
   - 带base: 维度增加50% (4→10)
   - 梯度计算: 内存需求增加
   - 建议: 如GPU内存不足，减小batch_x和batch_q

2. **训练时间**
   - 带base: 约2倍训练时间
   - SIREN vs MLP: 相近
   - 推荐epoch: 30000-50000

3. **精度**
   - 使用混合精度训练 (AMP)
   - 梯度裁剪防止溢出
   - 学习率调度器

## 验证方法 (Validation)

```python
# 评估指标
MAE = |d_pred - d_true|.mean()           # 平均绝对误差
RMSE = sqrt((d_pred - d_true)^2.mean())  # 均方根误差
SR = (|d_pred - d_true| < 0.03).mean()   # 成功率 (3cm阈值)
```

## 与原版差异总结 (Differences Summary)

| 特性 | para_nn_cdf.py | para_nn_cdf_v2.py |
|-----|---------------|------------------|
| Base DoF | ❌ | ✅ |
| 网络类型 | 仅MLP | MLP + SIREN |
| 输入维度 | 3 + DoF | 3 + DoF + (6?) |
| Pose处理 | 固定单位矩阵 | 可变pose矩阵 |
| 多手指训练 | 需手动 | 提供脚本 |
| 文档 | 基础 | 完整 |

## 使用建议 (Recommendations)

1. **首次使用**: 先用小epoch数（如1000）测试
2. **网络选择**: 先尝试MLP，如果需要更光滑的场再用SIREN
3. **数据检查**: 确保data_with_base_dof_*.pt存在且正确
4. **监控训练**: 使用TensorBoard观察loss曲线
5. **GPU推荐**: 至少8GB显存

## 文件清单 (File List)

- `para_nn_cdf_v2.py`: 主程序
- `train_leaphand_all_fingers.sh`: 训练脚本
- `README_CDF_V2.md`: 详细文档
- `IMPLEMENTATION_SUMMARY.md`: 本文件
