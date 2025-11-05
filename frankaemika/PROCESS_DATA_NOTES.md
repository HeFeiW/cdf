# process_data 实现说明

## 概述

已经根据 `para_nn_cdf.py` 补全了 `para_nn_cdf_v2.py` 中的 `process_data` 方法，并增强了对 base DoF 的支持。

## 主要修改

### 1. process_data 方法完整实现

```python
def process_data(self, data):
    """
    处理原始数据，对每个关节的配置进行降采样
    
    参数:
        data: {key: {'x': (N,3), 'q': (N,DoF) or (N,DoF+6), 'idx': (N)}}
              idx: 表示哪个link是最后的碰撞link
    
    返回:
        final_data: {'x': (G,3), 'q': (G,max_q_per_link,D,D), 'k': (G)}
                    G: 网格点数量
                    D: DoF 或 DoF+6 (取决于 use_base)
    """
```

**关键特性:**

1. **自动检测配置维度**: 根据 `use_base` 参数自动处理 DoF 或 DoF+6 维度
2. **智能降采样**: 
   - 对于带 base 的数据，只使用关节角度进行 FPS (Farthest Point Sampling)
   - 降采样后保留完整配置（包括 base pose）
3. **数据验证**: 自动移除无效数据点（全 inf）

### 2. 数据结构变化

#### 不带 base (原版):
```python
q_lib: (max_q_per_link, DoF, DoF)
# 示例: LeapHand finger
# shape: (100, 4, 4)
```

#### 带 base (v2):
```python
q_lib: (max_q_per_link, DoF+6, DoF)
# 示例: LeapHand finger with base
# shape: (100, 10, 4)
# 其中 10 = 4(joints) + 6(base)
```

### 3. 自动处理逻辑

在 `__init__` 中添加了智能处理：

```python
# 自动检测并处理原始数据
if 'raw_data' in paths and os.path.exists(paths['raw_data']):
    if not os.path.exists(paths['data']):
        # 只在处理后的数据不存在时才处理
        print(f"Processing raw data from {paths['raw_data']}")
        self.raw_data = np.load(paths['raw_data'], allow_pickle=True).item()
        self.process_data(self.raw_data)
    else:
        print(f"Processed data already exists, skipping processing")
```

### 4. FPS 降采样改进

对于带 base 的数据，FPS 只考虑关节角度：

```python
if self.use_base:
    # 只使用关节角度进行 FPS
    q_joint_only = q[mask][:, :DoF]
    fps_q_joint = pytorch3d.ops.sample_farthest_points(
        q_joint_only.unsqueeze(0), K=self.max_q_per_link
    )[0].squeeze()
    
    # 找到 FPS 样本的索引
    distances = torch.cdist(fps_q_joint, q_joint_only)
    fps_indices = distances.argmin(dim=1)
    
    # 获取完整配置（包括 base）
    fps_q_full = q[mask][fps_indices]
    q_lib[:, :config_dim, i-1] = fps_q_full
```

**为什么这样做？**
- Base pose 的6个维度（3平移+3旋转）与关节角度的尺度不同
- FPS 在关节空间中更有意义
- 但最终保存完整的配置（关节+base）

## 使用示例

### 处理带 base 的数据

```bash
python para_nn_cdf_v2.py \
    --train \
    --use_base \
    --robot leaphand \
    --serial_idx 0 \
    --data_path data_with_base_dof_0.pt \
    --raw data_with_base_dof_0.npy
```

### 处理不带 base 的数据

```bash
python para_nn_cdf_v2.py \
    --train \
    --robot leaphand \
    --serial_idx 0 \
    --data_path data_finger_0.pt \
    --raw data_finger_0.npy
```

## 测试

使用提供的测试脚本验证实现：

```bash
python test_process_data.py
```

测试内容：
1. 检查原始数据是否存在
2. 处理带 base DoF 的数据
3. 验证处理后的数据维度
4. 检查配置空间维度是否正确

## 与原版的差异

| 特性 | para_nn_cdf.py | para_nn_cdf_v2.py |
|-----|---------------|------------------|
| 配置维度 | 固定 DoF | DoF 或 DoF+6 |
| FPS 策略 | 直接对 q 采样 | 只对关节角度采样 |
| 数据结构 | (M, DoF, DoF) | (M, D, DoF) |
| 自动处理 | 需手动注释 | 智能检测 |

其中 M = max_q_per_link, D = DoF 或 DoF+6

## 数据流程

```
原始数据 (.npy)
└─> {key: {'x': (N,3), 'q': (N,10), 'idx': (N)}}
    │
    ├─> process_data()
    │   ├─> 按 link 分组
    │   ├─> FPS 降采样（只用关节角度）
    │   └─> 保留完整配置
    │
    └─> 处理后数据 (.pt)
        └─> {'x': (G,3), 'q': (G,100,10,4), 'k': (G)}
```

## 注意事项

1. **数据生成**: 确保使用 `parallel_data_generator.py` 生成带 base 的数据
2. **维度一致**: 训练和推理时必须使用相同的 `use_base` 设置
3. **内存使用**: 带 base 的数据会增加约 2.5x 内存占用（4→10 维度）
4. **兼容性**: 支持向后兼容不带 base 的数据

## 调试技巧

### 检查数据维度
```python
import torch
data = torch.load('data_with_base_dof_0.pt')
print(f"x shape: {data['x'].shape}")      # (G, 3)
print(f"q shape: {data['q'].shape}")      # (G, 100, 10, 4)
print(f"k shape: {data['k'].shape}")      # (G,)
```

### 验证配置空间
```python
from para_nn_cdf_v2 import CDF_V2

cdf = CDF_V2(device, paths, robot='leaphand', use_base=True, serial_idx=0)
DoF = cdf.robot.serials[0].dof
config_dim = DoF + 6 if cdf.use_base else DoF
print(f"Expected config dim: {config_dim}")
print(f"Actual config dim: {cdf.data['q'].shape[2]}")
```

## 文件清单

- `para_nn_cdf_v2.py`: 主实现文件（已更新 process_data）
- `test_process_data.py`: 测试脚本
- `PROCESS_DATA_NOTES.md`: 本文件

## 下一步

1. 运行测试验证实现: `python test_process_data.py`
2. 如果数据未生成，先运行: `python parallel_data_generator.py --robot leaphand`
3. 开始训练: `bash train_leaphand_all_fingers.sh`

## 问题排查

### 问题 1: pytorch3d 未安装
```bash
pip install pytorch3d
# 或使用 conda
conda install -c fvcore -c iopath -c conda-forge pytorch3d
```

### 问题 2: 数据文件未找到
确保先生成数据：
```bash
cd /home/hefei/cdf/frankaemika
python parallel_data_generator.py --robot leaphand
```

### 问题 3: 维度不匹配
检查 use_base 设置：
- 数据生成时使用的 use_base
- 训练时使用的 use_base
- 两者必须一致

---

**完成日期**: 2025-11-03  
**版本**: 2.0  
**作者**: Based on para_nn_cdf.py implementation
