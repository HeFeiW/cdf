#  **一、项目整体目标（明确且无歧义）**

本项目的目标是训练一个机械手（LeapHand 模型）在 PyBullet 仿真环境中完成：

> **在不碰撞障碍物的前提下，从 clutter 中抓取目标物体并稳定提升到桌面以上 0.1m。**

训练方法：**PPO（actor-critic）**
观测：点云（带标签）+ 机械手当前 configuration
动作：Δjoint（22 维）
奖励：dense reward（靠近目标、接近接触、提升、稳定、动作惩罚、碰撞惩罚、成功奖励）。
另需支持：

* **计算 V(s)**（用于可视化）
* 可视化点云 + value contribution（可选）

---

#  **二、核心设计原则**

1. **环境（Env）与策略学习（RL）完全解耦**
   任何 RL 框架都能无缝接入（SB3 / RLlib / cleanRL / custom PPO）。

2. **仿真与几何处理解耦**
   点云处理、碰撞判定、reward 计算、渲染全部是模块化的。

3. **模型结构可替换**
   可以随时换成更强的 PointNet++、KPConv、Transformer、GNN。

4. **Strong typing**（结构化数据接口）
   定义清晰的 dataclass：State、Action、RewardTerms。

5. **面向未来扩展**
   支持多个场景：多手、多物体、多 camera、真实场景 domain randomization。

---

#  **三、工程目录结构（精简但模块清晰）**

```
rl-multihand-grasping/
│
├── configs/
│   ├── env_config.yaml
│   ├── ppo_config.yaml
│   └── hand_config.yaml
│
├── envs/
│   ├── __init__.py
│   ├── grasp_env.py         # 主环境（符合 Gymnasium API）
│   ├── object_manager.py    # 管理物体加载、随机生成
│   ├── hand_controller.py   # 控制 LeapHand（joint limits、Δq 应用）
│   ├── sensors.py           # 点云采样、转换坐标系
│   ├── reward.py            # dense reward 分解
│   ├── termination.py       # 终止条件逻辑
│   └── utils.py             # pybullet helper
│
├── simulation/
│   ├── pybullet_sim.py      # 仿真基础封装
│   ├── contact_checker.py   # penetration 检查
│   └── visualizer.py        # 调试渲染
│
├── models/
│   ├── pointnet_encoder.py
│   ├── joint_encoder.py
│   ├── actor_critic.py
│   └── value_visualizer.py  # V(s) 可视化工具
│
├── training/
│   ├── train_ppo.py         # 训练入口
│   ├── buffers.py
│   ├── ppo_agent.py
│   └── evaluator.py
│
├── objects/                  # 物体 OBJ/URDF
│
├── hand_models/              # LeapHand URDF, meshes
│
├── scripts/
│   ├── evaluate_policy.py
│   └── visualize_value.py
│
└── README.md
```

---

#  **四、模块说明**

---

# ① **configs/**

以 YAML 方式定义所有参数（易调参）

* `env_config.yaml`

  * 桌面大小
  * 点云数量
  * 障碍物数量范围
  * 成功阈值
  * 最大 step 数
  * reward 权重
  * 碰撞惩罚权重
  * joint Δ clip
  * action 频率与仿真频率

* `hand_config.yaml`

  * joint limits
  * joint 类型
  * base 的自由度与限制

* `ppo_config.yaml`

  * PPO 参数：γ、λ、lr、batch_size、epochs
  * 网络结构尺寸

---

# ② **envs/** — 环境与任务逻辑（核心）

### **grasp_env.py**

实现 Gymnasium Env：

```python
class MultiHandGraspEnv(gym.Env):
    def __init__(self, config):
        ...
    def reset(self):
        return observation
    def step(self, action: np.ndarray):
        # apply Δq
        # run physics
        # get obs
        # compute reward breakdown
        # check termination
        return observation, reward, terminated, truncated, info
```

输入输出严格定义（不依赖 RL 工具）。

---

### **object_manager.py**

负责所有物体加载与初始化：

* uniform 选择目标物体
* 随机生成障碍物数量
* 随机初始位置/姿态
* 按桌面尺寸 valid 采样

接口：

```python
class ObjectManager:
    def spawn_objects()
    def get_target_pose()
    def get_all_object_ids()
```

---

### **hand_controller.py**

管理 LeapHand：

* 映射 Δq → new q
* clip 到 joint limits
* base movement（6-DoF）
* get hand base pose
* get fingertip poses（用于 reward）

接口：

```python
class HandController:
    def apply_delta_q(delta_q)
    def get_joint_positions()
    def get_fingertip_positions()
    def get_hand_center()
```

---

### **sensors.py**

点云模块：

* 从 PyBullet depth camera 采样
* 转换到 **手心坐标系**
* 固定数量采样（FPS or random sampling）
* 为点云添加标签（目标=1，其他=0）

接口：

```python
class PointCloudSensor:
    def get_pointcloud(hand_pose, object_ids) -> np.ndarray # (N, 4)
```

---

### **reward.py**

实现 **可组合的 dense reward**：

包含：

* 距离项
* 指尖-物体距离项
* 提升项
* 稳定项
* 碰撞项
* 动作惩罚
* 成功奖励

返回一个结构化对象：

```python
@dataclass
class RewardTerms:
    dist: float
    finger: float
    lift: float
    stable: float
    collision: float
    action: float
    success: float
    total: float
```

---

### **termination.py**

管理终止规则：

* step 超限
* 碰撞
* 是否 truncated
* 是否成功
* 是否掉落

接口：

```python
def check_termination(state, contacts, step_count) -> (terminated, truncated)
```

---

# ③ **simulation/** — PyBullet 基础层

### pybullet_sim.py

封装：

* world reset
* 加载 plane/table
* load/unload URDF
* stepSimulation()
* contact info

使 Env 不直接操作 PyBullet 接口。

---

### contact_checker.py

判断是否有 **penetration > 0**：

```python
def check_penetration(hand_id, obstacle_ids) -> bool
```

---

### visualizer.py

可选调试：

* 渲染场景
* 显示点云
* 显示抓取轨迹

---

# ④ **models/** — 神经网络模块

### pointnet_encoder.py

输入 (N,4) → feature vector (256)

### joint_encoder.py

输入 (22,) → feature (64)

### actor_critic.py

组合 pointnet + joint encoder，输出：

* actor: mean + log_std (22-dim)
* critic: V(s)

明确:

```python
class ActorCritic(nn.Module):
    def forward_obs(state) -> features
    def act(state) -> Δq
    def value(state) -> V
```

---

### value_visualizer.py

工具：

* 给定 state → V(s)
* 点扰动 → ΔV
* 点云着色图

---

# ⑤ **training/** — PPO 训练框架

### ppo_agent.py

自定义 PPO（或对接 SB3 的 PPO）

### train_ppo.py

训练入口：

* 加载 configs
* 创建 Env
* 构建 ActorCritic
* 训练 loop
* 保存 checkpoint
* 保存 value 可视化脚本

---

# ⑥ **scripts/**

轻量工具脚本：

* `evaluate_policy.py`：跑一次 episode
* `visualize_value.py`：可视化 critic 对给定 state 的输出

---

# ⑦ **objects/** & **hand_models/**

资源文件。

---

# ⑧ **README.md**

描述如何：

* 安装依赖
* 训练
* 评估
* 可视化 V(s)

---

#  **五、交互接口细化（给 AI coding 工具的硬性规范）**

### Observation（state）

```python
state = {
    "pointcloud": np.ndarray(shape=(N,4), dtype=float),  # (x,y,z,label)
    "q": np.ndarray(shape=(22,), dtype=float),           # joint + base
}
```

### Action

```python
action = np.ndarray(shape=(22,), dtype=float) # Δq
```

### Env Step 输出

```python
obs, reward_total, terminated, truncated, info
info = {
    "reward_terms": RewardTerms,  
    "success": bool
}
```

---

#  **六、原则**

* 每个模块职责单一
* 每个模块接口清晰
* YAML 配置避免硬编码
* 训练脚本只管训练
* Env 可以接入任何 RL 框架
* 手、仿真、点云逻辑完全解耦
* reward 独立，可单独测试
* 未来支持：多手、多 camera、真实机器人 sim2real
