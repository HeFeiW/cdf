# 2D机器人QP运动规划器 - 实现总结

## 概述

根据 `qp_mp_tao.py`（3D多指机器人的QP规划器），我创建了适用于2D机器人的QP运动规划模块，并集成到 `example.py` 中进行可视化测试。

## 创建的文件

### 1. qp_mp_tao_2d.py - 核心规划器模块

**主要类和函数：**

- **`QPPlanner2D`类**: 2D机器人的QP运动规划器
  - `__init__()`: 初始化规划器，设置机器人、CDF模型、参数等
  - `step()`: 执行一步规划，返回下一时刻的关节配置
  - `inference_cdf()`: 使用训练好的MLP模型推理CDF距离和梯度

- **`solve_optimization_problem()`**: 求解QP优化问题
  - 目标函数：最小化到目标的距离 + 控制成本
  - 约束：避障约束 + 控制输入界限约束

- **`build_planner_2d()`**: 工厂函数，用于创建规划器实例

- **`create_system_matrices()`**: 创建系统动力学矩阵 A 和 B

**关键特性：**
- 支持任意自由度的2D机器人
- 基于CDF神经网络模型进行距离场推理
- 使用CasADi求解QP问题（支持ipopt、osqp等多种求解器）
- 同时支持障碍物排斥和目标吸引

### 2. test_qp_planner.py - 测试脚本

包含两个测试函数：

- **`test_qp_planner_simple()`**: 基础功能测试
  - 创建简单的2连杆机器人
  - 运行50步规划
  - 可视化C空间轨迹和关节角度变化

- **`test_qp_planner_with_cdf()`**: 完整集成测试
  - 使用实际的CDF2D类
  - 创建包含多个障碍物和目标的场景
  - 运行完整的规划循环（最多200步）
  - 生成三种可视化：C空间+CDF、任务空间、关节角度

### 3. example.py - 集成到主示例文件

**新增函数：**

```python
def plot_qp_planning(obj_lists, filename, q_start=None, q_goal=None, max_steps=200):
```

- 加载或创建CDF模型
- 创建QP规划器
- 执行运动规划循环
- 生成三种可视化图像：
  1. C空间中的规划轨迹（叠加在CDF等高线上）
  2. 任务空间中的机器人运动轨迹
  3. 关节角度随时间的变化

**在主程序中添加了测试代码：**
- 创建测试场景（包含圆形和方形障碍物）
- 调用 `plot_qp_planning()` 进行规划和可视化
- 保存结果图像到 `image/test_qp_qp_planning.png`

### 4. README_QP_PLANNER.md - 详细文档

完整的英文文档，包含：
- 模块概述和功能特性
- 类和函数的详细说明
- 优化问题的数学公式
- 使用示例代码
- 参数调优指南
- 故障排查建议

## 技术实现细节

### QP优化问题公式

**目标函数：**
```
minimize: (1/2) * u^T * H * u + h^T * u
```

其中：
- `H = (B^T * grad_targ^T * grad_targ * B) * dt^2 + R`
- `h = 2 * B^T * grad_targ^T * dist_targ * dt`
- `R`: 控制代价矩阵（对角矩阵）

**约束条件：**
```
1. 避障约束: -grad_obs * B * u * dt <= dist_obs - safety_buffer
2. 控制界限: -cons_u <= u <= cons_u
```

**系统动力学：**
```
q_{k+1} = A * q_k + B * u_k
```

对于单积分系统：`A = I`, `B = dt * I`

### CDF推理流程

1. 从障碍物/目标对象表面采样点
2. 构造MLP输入：`[p_x, p_y, q_1, q_2, ..., q_n]`
3. 批量前向传播获得距离预测
4. 取所有采样点中的最小距离
5. 如果需要梯度，使用PyTorch的autograd计算

### 与3D版本的主要差异

| 方面 | 3D版本 (qp_mp_tao.py) | 2D版本 (qp_mp_tao_2d.py) |
|------|----------------------|-------------------------|
| 机器人表示 | ParallelRobotLayer（多指） | Robot2D（单链） |
| CDF接口 | CDF类 + robot_layer | 直接MLP推理 |
| 输入格式 | 点云 (N, 3) | 对象表面 (N, 2) |
| 模型数量 | 每根手指一个模型 | 单个MLP模型 |
| 规划范围 | 全机器人（多个串联） | 单个串联链 |

## 使用方法

### 基础用法

```python
from qp_mp_tao_2d import build_planner_2d
from cdf import CDF2D

# 创建CDF实例
cdf = CDF2D(device)

# 加载模型
cdf_model = torch.load('model/model.pth').to(device)

# 创建规划器
planner = build_planner_2d(cdf.robot, cdf_model, device, 
                           dt=0.05, cons_u=1.0, safety_buffer=0.05)

# 定义场景
obs_objs = [Circle(...), Box(...)]  # 障碍物
targ_objs = [Circle(...)]            # 目标

# 执行规划
q_current = torch.tensor([0.0, 0.0])
for step in range(max_steps):
    q_next = planner.step(q_current, obs_objs, targ_objs)
    # ... 检查收敛 ...
    q_current = q_next
```

### 运行测试

```bash
# 进入2D示例目录
cd /home/hefei/cdf/2Dexamples

# 运行独立测试
python test_qp_planner.py

# 运行集成示例（包含QP规划测试）
python example.py
```

### 可视化输出

测试会生成以下图像（保存在 `image/` 目录）：

1. **test_qp_simple.png**: 基础测试结果
   - C空间轨迹
   - 关节角度随时间变化

2. **test_qp_with_cdf.png**: 完整集成测试结果
   - C空间轨迹 + CDF等高线
   - 任务空间中的机器人运动
   - 关节角度时间序列

3. **test_qp_qp_planning.png**: example.py中的测试结果
   - 完整的三视图可视化

## 参数调优建议

### 关键参数

- **dt**: 时间步长
  - 更小 → 轨迹更平滑，需要更多步数
  - 建议: 0.01 ~ 0.1

- **cons_u**: 控制输入约束
  - 更大 → 运动更快，但可能不够平滑
  - 建议: 0.5 ~ 3.0

- **safety_buffer**: 安全缓冲距离
  - 更大 → 更保守，更安全
  - 建议: 0.01 ~ 0.2

- **solver**: QP求解器类型
  - 'ipopt': 最鲁棒（推荐）
  - 'osqp': 速度快
  - 'qpOASES', 'qrqp': 其他选项

### 预设配置

```python
# 保守配置（安全、慢速）
build_planner_2d(robot, model, device, 
                 dt=0.01, cons_u=0.5, safety_buffer=0.1)

# 平衡配置（推荐）
build_planner_2d(robot, model, device, 
                 dt=0.05, cons_u=1.5, safety_buffer=0.05)

# 激进配置（快速、风险较高）
build_planner_2d(robot, model, device, 
                 dt=0.1, cons_u=3.0, safety_buffer=0.01)
```

## 常见问题解决

### 1. 求解器收敛失败
- 尝试更换求解器（ipopt最鲁棒）
- 增大 `safety_buffer`
- 减小 `cons_u`

### 2. 机器人移动太慢
- 增大 `dt` 或 `cons_u`
- 检查目标梯度的大小

### 3. 出现震荡
- 增大控制代价（cost_mat_R的对角元素）
- 减小 `dt`

### 4. 与障碍物碰撞
- 增大 `safety_buffer`
- 检查CDF模型的准确性
- 验证障碍物定义是否正确

## 下一步改进方向

- [ ] 支持2D多链/多指机器人
- [ ] 根据速度自适应调整安全缓冲
- [ ] 支持动态障碍物
- [ ] 轨迹后处理和平滑
- [ ] 在线更新CDF模型
- [ ] MPC风格的滚动优化

## 测试状态

所有文件已创建并准备就绪，可以运行测试：

✓ qp_mp_tao_2d.py - 核心模块
✓ test_qp_planner.py - 测试脚本  
✓ example.py - 已集成QP规划
✓ README_QP_PLANNER.md - 详细文档

建议按以下顺序测试：
1. 先运行 `test_qp_planner.py` 验证基础功能
2. 再运行 `example.py` 查看完整集成效果
3. 根据需要调整参数并重新测试

## 依赖项

- PyTorch
- CasADi
- NumPy
- Matplotlib

项目内部依赖：
- robot2D_torch.py
- primitives2D_torch.py
- mlp.py
- cdf.py

## 总结

成功创建了一个完整的2D机器人QP运动规划模块，包括：
- 核心规划算法实现
- 测试和验证脚本
- 与现有CDF框架的集成
- 详细的文档说明

该模块可以直接用于2D机器人的实时运动规划，支持避障和目标到达任务。
