# Multi-Hand Grasping with RL

基于 PPO 的多指手抓取强化学习项目。

## 项目目标

在 PyBullet 仿真环境中训练 LeapHand 机械手完成：

> **在不碰撞障碍物的前提下，从 clutter 中抓取目标物体并稳定提升到桌面以上 0.1m。**

## 项目结构

```
rl-multihand-grasping/
├── configs/                 # 配置文件
│   ├── env_config.yaml     # 环境配置
│   ├── hand_config.yaml    # 手部配置
│   └── ppo_config.yaml     # PPO 训练配置
│
├── envs/                    # 环境模块
│   ├── grasp_env.py        # 主环境（Gymnasium API）
│   ├── object_manager.py   # 物体管理
│   ├── hand_controller.py  # 手部控制
│   ├── sensors.py          # 点云传感器
│   ├── reward.py           # 奖励计算
│   ├── termination.py      # 终止条件
│   └── utils.py            # 工具函数
│
├── simulation/              # 仿真模块
│   ├── pybullet_sim.py     # PyBullet 封装
│   ├── contact_checker.py  # 碰撞检测
│   └── visualizer.py       # 可视化工具
│
├── models/                  # 神经网络模块
│   ├── pointnet_encoder.py # PointNet 编码器
│   ├── joint_encoder.py    # 关节编码器
│   ├── actor_critic.py     # Actor-Critic 网络
│   └── value_visualizer.py # V(s) 可视化
│
├── training/                # 训练模块
│   ├── ppo_agent.py        # PPO 智能体
│   ├── buffers.py          # 经验缓冲区
│   ├── train_ppo.py        # 训练入口
│   └── evaluator.py        # 评估器
│
├── scripts/                 # 工具脚本
│   ├── evaluate_policy.py  # 策略评估
│   └── visualize_value.py  # 价值可视化
│
├── objects/                 # 物体模型
├── hand_models/             # 手部 URDF
└── README.md
```

## 安装依赖

```bash
pip install gymnasium numpy torch pybullet pyyaml matplotlib
```

## 快速开始

### 训练

```bash
python training/train_ppo.py \
    --env-config configs/env_config.yaml \
    --hand-config configs/hand_config.yaml \
    --ppo-config configs/ppo_config.yaml
```

### 评估

```bash
python scripts/evaluate_policy.py \
    --checkpoint checkpoints/final_model.pt \
    --num-episodes 100 \
    --deterministic
```

### 可视化 V(s)

```bash
python scripts/visualize_value.py \
    --checkpoint checkpoints/final_model.pt \
    --save-dir value_visualization
```

## 观测空间

```python
observation = {
    "pointcloud": np.ndarray(shape=(N, 4)),  # (x, y, z, label)
    "q": np.ndarray(shape=(22,))             # joint positions
}
```

## 动作空间

```python
action = np.ndarray(shape=(22,))  # Δq (joint position delta)
```

## 奖励设计

- **距离奖励**: 手心靠近目标物体
- **指尖奖励**: 指尖接近目标
- **提升奖励**: 目标物体高度增加
- **稳定奖励**: 抓取后保持稳定
- **碰撞惩罚**: 与障碍物碰撞
- **动作惩罚**: 鼓励平滑动作
- **成功奖励**: 达成任务目标

## 核心设计原则

1. **环境与策略学习完全解耦** - 支持接入任何 RL 框架
2. **仿真与几何处理解耦** - 模块化设计
3. **模型结构可替换** - 易于升级网络架构
4. **Strong typing** - 清晰的数据接口
5. **YAML 配置** - 避免硬编码，易于调参

## 配置说明

### env_config.yaml

- 场景设置（桌面大小、位置）
- 点云参数（数量、相机配置）
- 物体配置（障碍物数量范围）
- 任务参数（成功阈值、最大步数）
- 奖励权重

### hand_config.yaml

- 关节限制
- 控制参数
- 初始姿态

### ppo_config.yaml

- PPO 超参数（γ, λ, clip ratio 等）
- 网络结构
- 训练参数

## 扩展

项目设计支持未来扩展：

- 多手协作
- 多物体场景
- 多相机视角
- Domain randomization
- Sim2Real 迁移
