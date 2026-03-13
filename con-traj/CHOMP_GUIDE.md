# CHOMP轨迹优化器使用指南

## 概述

CHOMP（Covariant Hamiltonian Optimization for Motion Planning）是一种轨迹优化算法，通过最小化平滑性成本和障碍成本来生成无碰撞的平滑轨迹。本实现支持2D点机器人的轨迹规划，可以处理圆形、矩形和线段等多种形状的障碍物。

## 数学公式

### 目标函数

CHOMP的目标是最小化以下成本函数：

$$J(\xi) = c_s(\xi) + \lambda_{obs} \cdot c_{obs}(\xi)$$

其中：
- $\xi$ 是轨迹（一系列配置点）
- $c_s(\xi)$ 是平滑性成本
- $c_{obs}(\xi)$ 是障碍成本  
- $\lambda_{obs}$ 是障碍成本权重

### 平滑性成本

平滑性成本鼓励轨迹光滑，由速度和加速度成本组成：

$$c_s(\xi) = \frac{1}{2} \xi^T A \xi$$

其中 $A$ 是平滑性矩阵：

$$A = B^T B \cdot w_v + C^T C \cdot w_a + \epsilon I$$

- $B$ 是一阶差分矩阵（计算速度）
- $C$ 是二阶差分矩阵（计算加速度）
- $w_v, w_a$ 分别是速度和加速度权重
- $\epsilon$ 是阻尼系数（用于数值稳定性）

### 障碍成本

障碍成本基于有符号距离函数（SDF）：

$$c_{obs}(\xi) = \sum_{t=0}^{T-1} \phi(d_t) \cdot dt$$

其中 $\phi(d)$ 是势函数：

$$\phi(d) = \begin{cases}
0 & \text{if } d \geq c_{clearance} \\
\frac{1}{2c_{clearance}}(c_{clearance} - d)^2 & \text{if } 0 \leq d < c_{clearance} \\
\frac{1}{2}c_{clearance} - d & \text{if } d < 0
\end{cases}$$

- $d$ 是点到最近障碍物的有符号距离
- $c_{clearance}$ 是期望的安全间隙

### 优化算法

使用协变梯度下降法：

$$\xi^{(k+1)} = \xi^{(k)} - \alpha A^{-1} \nabla J(\xi^{(k)})$$

其中：
- $\alpha$ 是步长
- $\nabla J$ 是目标函数的梯度
- $A^{-1}$ 是平滑性矩阵的逆（协变度量）

## 实现说明

### 核心类

#### `Trajectory`
表示一条轨迹，包含：
- `xi`: (T, D) 形状的配置点数组
- `dt`: 时间步长
- `T`: 轨迹点数
- `D`: 配置空间维数

#### `SmoothnessCost`
计算轨迹的平滑性成本和梯度。

#### `ObstacleCost`
计算基于环境的障碍成本和梯度。使用有符号距离函数。

#### `Environment2D`
2D环境模型，包含：
- 障碍物集合（圆形、矩形、线段）
- SDF计算和梯度
- 势函数定义

#### `CHOMPOptimizer`
优化算法实现，进行协变梯度下降。

### 支持的障碍物类型

1. **Circle2D** - 圆形障碍物
   - 参数：中心 (center)，半径 (radius)
   
2. **Box2D** - 矩形障碍物
   - 参数：中心 (center)，宽度 (width)，高度 (height)
   
3. **Segment2D** - 线段障碍物（带可选胶囊体半径）
   - 参数：端点 (point_a, point_b)，可选半径 (radius)

## 配置文件说明

配置文件采用YAML格式，主要部分包括：

### 轨迹参数 (trajectory)
- `num_points`: 轨迹点数（通常1000）
- `dimensions`: 配置空间维数（2D点机器人为2）
- `time_step`: 时间步长（可选，默认为 1/(T-1)）

### 任务参数 (task)
- `start`: 起点坐标 [x, y]
- `goal`: 终点坐标 [x, y]

### 障碍物参数 (obstacles)
- `clearance`: 期望的安全间隙距离
- `shapes`: 障碍物列表，每个包含：
  - `type`: 障碍物类型 (circle/box/segment)
  - 相应的几何参数

### 平滑性参数 (smoothness)
- `velocity_weight`: 速度成本权重（默认1.0）
- `acceleration_weight`: 加速度成本权重（默认1.0）
- `damping`: 矩阵阻尼系数（默认1e-4）

### 障碍成本参数 (obstacle_cost)
- `weight`: 相对于平滑性成本的权重（默认0.1）

### 优化器参数 (optimizer)
- `step_size`: 梯度下降步长（默认0.001）
- `num_iterations`: 优化迭代数（默认100）

### 可视化参数 (visualization)
- `plot_xlim`, `plot_ylim`: 绘图范围
- `save_trajectory_plot`: 是否保存轨迹演化图
- `save_gradient_plot`: 是否保存梯度可视化图
- 等相关保存选项

## 使用方法

### 方式1：命令行运行

```bash
cd /home/hefei/cdf/con-traj
python CHOMP.py chomp_config.yaml
```

或使用默认配置：
```bash
python CHOMP.py
```

### 方式2：在Python代码中使用

#### 方法A：使用配置文件

```python
from CHOMP import run_chomp_from_config

# 从配置文件运行
env2d, final_traj, optimizer = run_chomp_from_config('chomp_config.yaml', verbose=True)
```

#### 方法B：直接使用类

```python
from CHOMP import *
import numpy as np

# 创建轨迹
T, D = 1000, 2
start = np.array([0.1, 0.1])
goal = np.array([0.6, 0.9])
alphas = np.linspace(0.0, 1.0, T).reshape(-1, 1)
waypoints = (1 - alphas) * start + alphas * goal
traj = Trajectory(waypoints=waypoints, dt=1.0/(T-1))

# 创建环境
shapes = [
    Circle2D(center=(0.2, 0.4), radius=0.1),
    Circle2D(center=(0.4, 0.6), radius=0.1),
]
env = Environment2D(shapes=shapes, clearance=0.04)

# 创建成本函数
A = build_smoothness_A(T, D, damping=1e-4)
A_inv = np.linalg.inv(A)
smooth_cost = SmoothnessCost(A)
obs_cost = ObstacleCost(env)

# 创建优化器
optimizer = CHOMPOptimizer(
    smooth_cost=smooth_cost,
    obstacle_cost=obs_cost,
    step_size=0.001,
    A_inv=A_inv,
    lambda_obs=0.1,
)

# 运行优化
for _ in range(100):
    traj = optimizer.step(traj)
```

## 配置示例

### 基础2D障碍物规避

```yaml
trajectory:
  num_points: 1000
  dimensions: 2

task:
  start: [0.1, 0.1]
  goal: [0.6, 0.9]

obstacles:
  clearance: 0.05
  shapes:
    - type: circle
      center: [0.2, 0.4]
      radius: 0.15
    - type: box
      center: [0.5, 0.7]
      width: 0.2
      height: 0.15

optimizer:
  step_size: 0.001
  num_iterations: 200
```

### 高精度规划

```yaml
trajectory:
  num_points: 2000
  dimensions: 2

smoothness:
  velocity_weight: 1.0
  acceleration_weight: 2.0
  damping: 1e-6

obstacle_cost:
  weight: 0.05

optimizer:
  step_size: 0.0005
  num_iterations: 500
```

## 性能调优建议

### 当轨迹不够光滑时：
- 增加 `acceleration_weight`（加速度权重）
- 降低 `num_points`（减少自由度）
- 增加 `num_iterations`（更多优化迭代）

### 当轨迹与障碍物碰撞时：
- 增加 `obstacle_cost.weight`（障碍权重）
- 减少 `step_size`（更小的步长）
- 增加 `clearance`（更大安全间隙）
- 增加 `num_iterations`

### 当优化速度太慢时：
- 减少 `num_points`
- 降低 `num_iterations`
- 增加 `step_size`（但要小心）

### 当结果不稳定时：
- 增加 `smoothness.damping`
- 减少 `step_size`
- 增加 `num_iterations`

## 依赖

- numpy
- matplotlib
- pyyaml

## 输出

程序生成两个可视化图：

1. **轨迹演化图** (`chomp_2d_result.png`)
   - 显示优化过程中轨迹的演变
   - 颜色从暗到亮表示迭代过程

2. **梯度可视化图** (`chomp_2d_final_gradients.png`)
   - 显示最终轨迹
   - 蓝色箭头：平滑性梯度
   - 红色箭头：障碍成本梯度

## 常见问题

**Q: 为什么轨迹仍然与障碍物碰撞？**
A: 增加障碍权重或安全间隙，或运行更多迭代。

**Q: 轨迹太曲折怎么办？**
A: 增加加速度权重，或在配置中增加平滑性成本权重。

**Q: 如何处理更复杂的机器人模型？**
A: 修改 `forward_kinematics()` 和 `sdf_grad()` 中的Jacobian计算。

**Q: 支持3D规划吗？**
A: 当前实现针对2D优化。支持3D需要修改SDF计算和FK。

## 参考文献

- Ratliff, N. D., Zucker, M., Barto, A. G., & Chestnutt, J. (2009). CHOMP: Gradient optimization techniques for efficient motion planning. 2009 IEEE International Conference on Robotics and Automation.
