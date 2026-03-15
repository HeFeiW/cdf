# LeapHand 接触数据集说明文档

> 模块文件：`leaphand_contact_data_generator.py`
> 最后更新：2026-03-13

---

## 1. 背景与问题定义

### 1.1 多指手运动链结构

LeapHand 有 **4 根手指**，每根手指构成一条**串联运动链（serial kinematic chain）**：

```
palm_base (固定) → palm_lower_left (palm，所有手指共用)
    ↓
  [finger_0]  mcp_joint  → pip  → dip  → fingertip
  [finger_1]  mcp_joint_2 → pip_2 → dip_2 → fingertip_2
  [finger_2]  mcp_joint_3 → pip_3 → dip_3 → fingertip_3
  [finger_3]  thumb_left_temp_base → thumb_pip → thumb_dip → thumb_fingertip
```

- `palm_lower_left`：底座/手掌 link，通过**固定关节**与 `palm_base` 连接，为 4 根手指**共用**。
- 每根手指有 **4 个旋转关节**（DOF = 4），关节角范围由 URDF 定义。
- 数据生成时，每次只处理**一条运动链（一根手指）**，串联链的 DOF = 4。

### 1.2 数据生成目标

**不含底座变换（`with_base=False`）：** $\mathcal{Q} \subset \mathbb{R}^4$

$$
\mathcal{Q}(p) = \{ q \in \mathbb{R}^4 \mid \text{dist}(p,\ r(q)) \approx 0 \}
$$

**含6-DOF底座变换（`with_base=True`）：** $\mathcal{Q} \subset \mathbb{R}^{10}$

$$
\mathcal{Q}(p) = \{ [q_\text{joint},\ q_\text{base}] \in \mathbb{R}^{10} \mid
\text{dist}(p,\ r_\text{base}(q_\text{joint}, q_\text{base})) \approx 0 \}
$$

其中 $q_\text{base} = [t_x, t_y, t_z, r_x, r_y, r_z] \in \mathbb{R}^6$（平移 + ZYX 欧拉角，
平移范围 = workspace，旋转 ∈ $[-\pi, \pi]^3$）。

---

## 2. 两种数据生成方法

### 2.1 Method 1：IK-based（Task-space → Config-space）

**类**：`Method1IKDataGenerator`

#### 算法步骤

1. 将 workspace 离散化为 $N_t^3$ 个均匀网格格点 $\{p_j\}$。
2. 对每个 $p$，在关节软限位内随机初始化 `batchsize` 个构型 $\{q_1,\ldots,q_B\}$。
3. L-BFGS 优化最小化 $\sum_i \text{sdf}(p, q_i)^2$（批量并行，50步）。
4. 筛选满足条件的解：$|\text{sdf}(p, q^*)| < \varepsilon$ 且 $q^* \in$ soft limits。

**`with_base=True` 时：** 优化空间扩展为 $\mathbb{R}^{10}$（手指关节 + 底座），
随机初始化和 soft limits 均覆盖完整的10D空间。

#### 主要参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `n_task_discrete` | 20 | task space 每轴离散数 |
| `batchsize` | 20000 | 优化批大小（每点的随机初始化数） |
| `epsilon` | 1e-3 | SDF 接触判定阈值（米） |
| `with_base` | False | 是否包含底座6-DOF |

---

### 2.2 Method 2：FK-based（Config-space → Task-space）

**类**：`Method2FKDataGenerator`

#### 算法步骤（无底座，`with_base=False`）

1. 将 workspace 离散化为 $N_t^3$ 个 task space 格点 $\{p_j\}$。
2. 对每个手指 DOF 均匀离散 $N_c$ 点，生成 $N_c^4$ 个配置 $\{q_i\}$ 的网格。
3. 分批 FK：`serial.forward(I, q_batch)` 获取所有 link 顶点世界坐标。
4. 对每对 $(q_i, p_j)$ 计算最近顶点距离 $d_{ij}$；若 $d_{ij} <$ threshold，记为接触对。

#### 算法步骤（有底座，`with_base=True`）

步骤 2 替换为**混合采样策略**：
- 手指关节：$N_c^4$ 个均匀网格配置；
- 底座6-DOF：为每个手指配置独立随机采样 $N_\text{base}$ 个底座位姿；
- 总配置数：$N_c^4 \times N_\text{base}$。

步骤 3 的 FK 调用变为 `serial.forward(pose_matrix(q_base), q_joint)`，
其中 $q_\text{base}$ 经 `_q_to_pose_matrix` 转换为 $(B, 4, 4)$ SE(3) 矩阵。

#### 内存管理（三层分批）

```
config_batch_size=200  (外层：FK 并行效率)
  └─ dist_sub_batch_size=5  (中层：距离计算显存控制)
       └─ task_batch_size=100  (内层：task points 分批)
          → torch.cdist: (Bsub, T_b, Nv_link) 中间矩阵
          → 峰值显存 = Bsub × T_b × Nv_link × 4 bytes
                     = 5 × 100 × 43315 × 4 ≈ 86MB/link
```

若 GPU 显存充裕，可调大以加速：`dist_sub_batch_size=10, task_batch_size=200` → ~346MB，约快 4×。

#### 主要参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `n_task_discrete` | 20 | task space 每轴离散数 |
| `n_config_discrete` | 10 | 手指每 DOF 离散数 |
| `with_base` | False | 是否包含底座6-DOF |
| `n_base_random` | 50 | with_base=True 时底座随机采样数 |
| `threshold` | 5e-3 | 顶点距离接触判定阈值（米，5mm） |
| `config_batch_size` | 200 | FK 并行批大小 |
| `dist_sub_batch_size` | 5 | 距离计算 config 子批（显存控制） |
| `task_batch_size` | 100 | 距离计算 task point 批大小 |

---

## 3. 资源估算

> **基准来源（GPU: RTX 3090/A5000，leaphand finger_0，DOF=4）**
> - Method 2 (N_c=8, N_t=10 → 1000 pts)：**8.34s** 实测
> - Method 1 (batchsize=5000 抽样 100 pts)：**0.116 s/pt** 实测（外推 batchsize=20000 约 ×3）
> - Method 1 with_base 实测数据时长（Nov 2025 文件时间戳推算）：**~68 min/finger**
> - 实测文件大小：method1 no_base ~240MB/finger，method1 with_base ~3.7–4.4GB/finger

快速打印估算（无需加载模型）：
```bash
python leaphand_contact_data_generator.py --method estimate
```

### 3.1 Method 2 (FK-based) — 单根手指，N_t=20（8000 task pts）

| `N_c` | 总配置数 | 配置维度 | 估算耗时 | 估算存储 |
|-------|---------|---------|---------|---------|
| 8     | 4,096   | 4D      | ~14s    | ~7MB    |
| **10**    | **10,000**  | **4D**  | **~2.7min** | **~17MB** |
| 12    | 20,736  | 4D      | ~5.6min | ~35MB   |
| 15    | 50,625  | 4D      | ~14min  | ~85MB   |
| — with_base=True — |||||
| N_c=8,  N_base=20 | 81,920  | 10D | ~44min | ~600MB |
| N_c=8,  N_base=50 | 204,800 | 10D | ~1.8h  | ~1.5GB |
| **N_c=10, N_base=20** | **200,000** | **10D** | **~1.8h** | **~1.4GB** |
| N_c=10, N_base=50 | 500,000 | 10D | ~4.5h  | ~3.5GB |

> 粗体行为推荐默认值。4根手指全量生成 → 时间×4，存储×4。

### 3.2 Method 1 (IK-based) — 单根手指，N_t=20（8000 task pts）

| 模式 | 配置维度 | 估算耗时/点 | 总耗时 | 实测存储 |
|------|---------|-----------|-------|---------|
| no_base | 4D | ~0.35 s/pt | **~47 min** | **~240 MB** |
| with_base | 10D | ~0.50 s/pt | **~67 min** | **~3.7–4.4 GB** |

### 3.3 四根手指全量生成汇总

| 场景 | 总耗时 | 总存储 | 备注 |
|------|-------|-------|------|
| M2, N_c=10, no_base | ~11 min | ~70 MB | 最快，无需模型 |
| M2, N_c=12, no_base | ~22 min | ~140 MB | 更高覆盖率 |
| M2, N_c=8, N_base=50, with_base | ~7 h | ~6 GB | |
| M2, N_c=10, N_base=20, with_base | ~7 h | ~5.5 GB | |
| M1, no_base | ~3 h | ~960 MB | 高质量 |
| M1, with_base | **~4.5 h** | **~15 GB** | 需事后压缩为 .pt |

---

## 4. 数据集格式

### 4.1 `.npy` 原始数据文件

保存路径：`<save_dir>/<robot>/<method_tag>/finger_<i>/data.npy`

其中 `method_tag` 为：`method1_ik` / `method2_fk` / `method2_fk_base`（with_base 时）

```python
data = {
    point_idx: {            # int，task space 格点索引 (0 ~ N_t^3 - 1)
        'x':   np.ndarray,  # shape (3,)，task space 格点坐标（米）
        'q':   np.ndarray,  # shape (M, dof_total)
                            # no_base:   dof_total = 4   → [q0, q1, q2, q3]
                            # with_base: dof_total = 10  → [q0,q1,q2,q3, tx,ty,tz,rx,ry,rz]
        'idx': np.ndarray,  # shape (M,)，最近接触 link 索引（serial.all_links 中的位置）
    },
    ...
}
```

#### 加载示例

```python
import numpy as np

data = np.load('data.npy', allow_pickle=True).item()

pt = data[1500]
print(pt['x'])     # [0.01, -0.05, 0.02]（米）
print(pt['q'])     # shape (M, 4) 或 (M, 10)
print(pt['idx'])   # shape (M,)

# with_base 时分离 joint 和 base
q_joint = pt['q'][:, :4]   # (M, 4) 手指关节角（弧度）
q_base  = pt['q'][:, 4:]   # (M, 6) [tx, ty, tz, rx, ry, rz]
```

### 4.2 配置文件 `config.json`

```jsonc
{
  "method": "method2_fk_base",   // method1_ik / method2_fk / method2_fk_base
  "robot": "leaphand",
  "serial_idx": 0,
  "dof": 4,
  "with_base": true,
  "dof_total": 10,               // 4 (no_base) 或 10 (with_base)
  "timestamp": "2026-03-13T...",

  "workspace": {"min": [-0.2,-0.2,-0.1], "max": [0.06,0.02,0.1]},
  "joint_limits_soft": {"min": [...], "max": [...]},
  "base_limits": {               // 仅 with_base=true 时出现
    "min": [-0.2,-0.2,-0.1,-3.14,-3.14,-3.14],
    "max": [ 0.06, 0.02, 0.1, 3.14, 3.14, 3.14]
  },

  "n_task_discrete": 20,
  "n_config_discrete": 10,
  "n_base_random": 50,           // 仅 with_base=true 时出现
  "total_task_points": 8000,
  "total_config_points": 500000,
  "contact_threshold_m": 0.005,
  "config_batch_size": 200,
  "dist_sub_batch_size": 5,
  "task_batch_size": 100,
  "device": "cuda:0",
  "total_time_seconds": 9876.5,

  "stats": {
    "total_points": 8000, "nonempty_points": 5200,
    "coverage_ratio": 0.65,
    "mean_configs_per_point": 412.3,
    "max_configs_per_point": 3800,
    "total_contact_pairs": 3298400
  }
}
```

### 4.3 目录结构

```
<save_dir>/
└── leaphand/
    ├── method1_ik/
    │   └── finger_{0-3}/  {data.npy, config.json}
    ├── method2_fk/           ← with_base=False
    │   └── finger_{0-3}/  {data.npy, config.json}
    ├── method2_fk_base/      ← with_base=True
    │   └── finger_{0-3}/  {data.npy, config.json}
    └── benchmark/
        └── finger_{0-3}/  {benchmark.json}
```

---

## 5. 使用方法

### 5.1 命令行

```bash
cd /home/hefei/cdf/frankaemika

# 仅打印资源估算（无需加载模型，几乎即时）
python leaphand_contact_data_generator.py --method estimate

# Method 2, no_base，全部手指
python leaphand_contact_data_generator.py \
    --method method2 --n_task 20 --n_config 10

# Method 2, with_base，仅 finger_0
python leaphand_contact_data_generator.py \
    --method method2 --with_base --n_config 8 --n_base_random 50 --finger 0

# Method 1, with_base
python leaphand_contact_data_generator.py \
    --method method1 --with_base --n_task 20 --batchsize 20000 --finger 0

# 效率基准测试（快速，小规模）
python leaphand_contact_data_generator.py \
    --method benchmark --n_task 10 --n_config 8

# 效率基准测试（含 with_base）
python leaphand_contact_data_generator.py \
    --method benchmark --n_task 8 --n_config 6 --with_base --n_base_random 10
```

### 5.2 Python API

```python
import torch
from panda_layers.parallel_robot_layer import ParallelRobotLayer
from leaphand_contact_data_generator import (
    Method1IKDataGenerator, Method2FKDataGenerator,
    estimate_generation_resources, run_efficiency_benchmark,
)

device = torch.device("cuda:0")
paths = {"urdf": "../../RDF/descriptions/leaphand/*.urdf",
         "meshes": "../../RDF/descriptions/leaphand/meshes/*.stl",
         "model": "../../RDF/models/leaphand/BP_8.pt"}
robot = ParallelRobotLayer(device=device, robot="leaphand", paths=paths)

# 资源估算（无需模型）
estimate_generation_resources(dof=4, n_task=20)

# Method 2, no_base
gen2 = Method2FKDataGenerator(
    device, robot, paths, serial_idx=0,
    with_base=False, n_task_discrete=20, n_config_discrete=10)
data2, cfg2 = gen2.generate(save_dir="./data")

# Method 2, with_base
gen2b = Method2FKDataGenerator(
    device, robot, paths, serial_idx=0,
    with_base=True, n_task_discrete=20, n_config_discrete=8, n_base_random=50)
data2b, _ = gen2b.generate(save_dir="./data")

# Method 1, with_base（需要 BP-SDF 模型 + torchmin）
gen1 = Method1IKDataGenerator(
    device, robot, paths, serial_idx=0,
    with_base=True, n_task_discrete=20, batchsize=20000)
data1, _ = gen1.generate(save_dir="./data")
```

---

## 6. 方法对比

| 维度 | Method 1 (IK) | Method 2 no_base | Method 2 with_base |
|------|-------------|-----------------|-----------------|
| 算法 | 梯度优化 IK | FK + 顶点距离 | FK + 底座变换 + 顶点距离 |
| 需要SDF模型 | **是** | **否** | **否** |
| 配置空间维度 | 4D / 10D | 4D | 10D |
| 采样策略 | 随机初始化+优化 | 完整网格 | 手指网格 × 底座随机 |
| 接触集完整性 | 高 | 中 | 中高 |
| 单指耗时（20^3） | 47–67 min | 2–14 min | 44 min–4.5 h |
| 单指存储 | 240MB / 4GB | 7–85 MB | 600MB–3.5 GB |
| 建议场景 | 高质量训练集 | 快速验证/无模型 | with_base 高质量集 |

---

## 7. 常见问题

**Q: Method 2 with_base 为什么不用完整的 10D 网格？**

A: $N_c^{10}$ 即使 $N_c=3$ 也达 59k 配置，$N_c=4$ 超过 1M，且在 10D 空间中均匀网格
稀疏性极高。"手指网格 × 底座随机"策略在保证手指空间覆盖的同时，用随机采样填充底座空间，
总体效率更高。

**Q: threshold 如何选择？**

A: 5mm 是合理默认值（leaphand STL 网格顶点间距约 1–5mm）。减小 threshold 可提高精确度
但会降低覆盖率；增大 threshold 覆盖率高但可能把"离表面较远"的配置也纳入。

**Q: 显存不足怎么办？**

A: 减小 `dist_sub_batch_size`（2 或 1）和 `task_batch_size`（50），峰值显存
= `dist_sub_batch * task_batch * max_Nv * 4` bytes。

**Q: 如何将 `.npy` 数据降采样为 `.pt` 格式？**

A: 参考现有 `data_with_base_dof_*.pt` 的生成方式，每个 task space 点截取最多
100 个配置，填充为固定形状 `(N_t^3, 100, dof_total, 1)` 的 tensor，便于 DataLoader。

**Q: `idx` link 索引的含义？**

A: 在 `serial.all_links` 中的 0-based 位置，标识接触最近的 link，用于 CDF 距离函数
中确定有效 DOF 数（$q_{1:k}$，$k$ = link_idx）。
