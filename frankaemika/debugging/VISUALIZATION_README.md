# CDF可视化工具使用说明

这个工具用于可视化基于CDF模型生成的机器人接触构型。

## 功能

1. **生成接触构型**: 在空间中放置物体，从物体表面采样点，通过CDF query将随机初始化的机器人构型投影到零测集，生成与物体有接触的构型
2. **可视化构型**: 支持两种可视化方式
   - 简单模式：终端文本浏览
   - RViz模式：3D可视化（需要ROS2）

## 文件说明

- `visualize_cdf.py` - 主程序，包含生成和可视化功能
- `view_configs.py` - 简单的终端浏览器
- `visualize_cdf.sh` - 快捷脚本（可选）

## 使用方法

### 步骤1: 生成接触构型

```bash
python3 visualize_cdf.py --mode generate \
    --robot leaphand \
    --serial_idx 0 \
    --obj_mesh /workspace/cdf/models/ACE_Coffee_Mug_Kristen_16_oz_cup/meshes/model.obj \
    --obj_position 0.0 0.0 0.4 \
    --obj_orientation 0.0 0.0 0.0 \
    --obj_scale 1.0 \
    --num_samples 1000 \
    --num_configs 50 \
    --num_iterations 10 \
    --device cuda \
    --output contact_configs.json
```

**参数说明**:
- `--robot`: 机器人类型 (leaphand/panda/dexhand)
- `--serial_idx`: 串联链索引（用于多指机器人，如leaphand）
- `--obj_mesh`: 物体mesh文件路径
- `--obj_position`: 物体位置 [x, y, z]
- `--obj_orientation`: 物体姿态 [roll, pitch, yaw] 弧度
- `--obj_scale`: 物体缩放因子
- `--num_samples`: 从物体表面采样的点数
- `--num_configs`: 要生成的构型数量
- `--num_iterations`: 投影到零测集的迭代次数
- `--device`: 计算设备 (cuda/cpu)
- `--output`: 输出JSON文件名

**输出**: 生成 `contact_configs.json` 文件，包含所有构型和物体信息

### 步骤2: 查看构型（简单模式）

使用终端文本浏览器查看生成的构型：

```bash
python3 view_configs.py --config_file contact_configs.json
```

**交互命令**:
- `n` - 下一个构型
- `p` - 上一个构型  
- `q` - 退出
- `s` - 保存当前构型为独立文件
- `d` - 显示全局统计信息

### 步骤3: 3D可视化（需要ROS2）

使用RViz进行3D可视化：

```bash
# 方式1: 使用脚本
./visualize_cdf.sh visualize

# 方式2: 手动启动
python3 visualize_cdf.py --mode visualize --config_file contact_configs.json
```

**前置要求**:
- 已安装ROS2
- 已编译 `show_urdf` 包
- 已source ROS2环境

**交互控制**:
- `n` - 下一个构型
- `p` - 上一个构型
- `q` - 退出

## 示例

### 示例1: 为LeapHand生成抓取构型

```bash
# 生成构型
python3 visualize_cdf.py --mode generate \
    --robot leaphand \
    --serial_idx 0 \
    --obj_mesh /workspace/cdf/models/004/textured.obj \
    --num_configs 100 \
    --num_iterations 15 \
    --output leaphand_grasps.json

# 查看
python3 view_configs.py --config_file leaphand_grasps.json
```

### 示例2: 为Panda生成不同位置的物体构型

```bash
python3 visualize_cdf.py --mode generate \
    --robot panda \
    --obj_mesh /workspace/cdf/models/sphere.obj \
    --obj_position -0.1 0.0 0.5 \
    --num_configs 30 \
    --output panda_configs.json
```

## 输出文件格式

生成的JSON文件包含以下信息：

```json
{
  "robot_name": "leaphand",
  "serial_idx": 0,
  "dof": 4,
  "joint_names": ["joint_0", "joint_1", "joint_2", "joint_3"],
  "object": {
    "mesh_path": "/path/to/object.obj",
    "position": [x, y, z],
    "orientation": [roll, pitch, yaw],
    "scale": 1.0,
    "transform": [[...], [...], [...], [...]]
  },
  "configurations": [
    {
      "id": 0,
      "joint_angles": [q0, q1, q2, q3],
      "distance": 0.0012
    },
    ...
  ]
}
```

## 故障排除

### 问题1: CUDA out of memory

解决方案：
- 减少 `--num_samples` (从1000降到500)
- 减少 `--num_configs` (从50降到20)
- 使用CPU: `--device cpu`

### 问题2: 模型文件找不到

确保模型文件存在：
- LeapHand: `model_dict/leaphand/finger_no_base.pt`
- Panda: `model_dict/panda/panda_mlp_base.pt`

如果不存在，需要先训练CDF模型。

### 问题3: RViz无法启动

检查：
1. ROS2是否安装: `ros2 --version`
2. 是否source环境: `source /opt/ros/humble/setup.bash`
3. show_urdf包是否编译

如果RViz有问题，可以只使用简单模式浏览。

## 高级用法

### 自定义投影参数

修改 `visualize_cdf.py` 中的投影逻辑：

```python
# 在projection循环中添加自定义逻辑
for iter_idx in range(num_iterations):
    # ... 现有代码 ...
    
    # 添加额外的约束或优化
    # 例如：添加碰撞检测、限制工作空间等
```

### 批量生成多个物体

创建脚本批量处理：

```bash
#!/bin/bash
for obj in /workspace/cdf/models/*/textured.obj; do
    name=$(basename $(dirname $obj))
    python3 visualize_cdf.py --mode generate \
        --obj_mesh $obj \
        --output "configs_${name}.json"
done
```

## 参考

- CDF模型: `para_nn_cdf_v2.py`
- 机器人层: `../../RDF/panda_layers/parallel_robot_layer.py`
- 可视化包: `../../RDF_ori/show_urdf_ws/src/show_urdf/`
