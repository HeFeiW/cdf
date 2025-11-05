# CDF 项目复现和修改文档

## 目录结构

```
.
├── README.md              # 原有README
├── note.md                # 项目中文说明/笔记
├── requirements.txt       # Python依赖包列表
├── 2Dexamples/            # 2D示例与相关代码
├── frankaemika/           # Franka Emika Panda机器人相关代码与数据，增加了适配dexhand
                            ，leaphand以及可适配其他机器人的功能
```

### 主要文件和文件夹说明

- `README.md`  
英文项目说明，包含项目背景、依赖、用法等。

- `note.md  `
中文项目结构说明及笔记。

- `requirements.txt`
项目所需Python依赖包列表，安装方法：
```bash
pip install -r requirements.txt
```

#### `2Dexamples/`
- `cdf.py`：2D CDF主程序，实现CDF2D类，计算和可视化Robot2D的sdf,cdf并可视化。
  关键函数：  
```python
    def inference_sdf(self,q,obj_lists,return_grad = False): 
        # 返回每个输入的q到所有障碍物总体的最小signed distance
        # q: (B,Dof)
        # obj_lists: list of objects
        # return:
        # sdf: (B,)/(B,2) (取决于是否return_grad)
        # using predefined object 
        # 先用正运动学计算q参数下机器人形态,然后在上面采样很多点为kpts）
        # 计算每个物体到每个采样点的signed distance
        # 取每个采样点到所有物体的最小signed distance
        # 取同一q下所有采样点的最小signed distance作为q的signed distance，并返回

    def find_q(self,obj_lists,batchsize = None):
        # find q that makes d(x,q) = 0. x is the obstacle surface
        # using L-BFGS method
        # 即寻找障碍物表面的zero-level-set configuration space
        # 返回满足d(x,q)=0的q：（B,Dof）
        def cost_function(q):
            # 定义代价为所有配置点到障碍物表面的最短距离的平方和
            #  find q that d(x,q) = 0
            # q : B,2
        res = minimize( # 用L-BFGS方法寻找满足d(x,q)=0的q, 
            cost_function, 
            q, # 随机初始化q
            method='l-bfgs', 
            options=dict(line_search='strong-wolfe'),
            max_iter=50,# 限制了优化的最大迭代次数为 50
            disp=0
            )
```
- `robot2D_torch.py`：2D机器人运动学与采样相关函数。实现了Robot2D类，以及相应的forwardkinematics和surface points sampler。
- `primitives2D_torch.py`：2D几何体SDF实现与可视化工具。
- `robot_plot2D.py`：一些2D机器人绘图工具函数。
- `mlp.py`： 2D神经网络MLPRegression模型定义和前向推理实现。
- `model.py`：2D神经网络模型相关代码。
- `nn_cdf.py`：2D神经CDF模型训练与推理代码。
- `nn_cdf_casadi.py`：2D神经CDF模型训练与推理代码。
- `eval_img/`：2D实验结果图片。
- `data2D.pt`、`model.pth`：2D实验数据与模型文件。  

使用说明：  
在`cdf.py`中定义Robot2D和障碍物，然后运行
```bash
python 2Dexamples/cdf.py
```
可视化SDF与CDF效果，展示梯度投影等功能。`plot_projection(obj_lists):`函数展示了cdfsdf上沿梯度投影的C-Space路径和工作空间路径。`plot_sdf`和`plot_cdf`函数分别实现了sdf和cdf的等高线图绘制。`shooting`计算沿着测地线移动（即正交于梯度方向）的路径。


#### `frankaemika/`

- `data_generator.py`：生成Franka机器人SDF/CDF训练数据。
- `forward_data_generator.py`：前向数据生成相关代码。
- `mlp.py`：神经网络模型定义。
- `nn_cdf.py`：Franka机器人神经CDF模型训练与评估主程序，包含关键类CDF。
- `mp_ik.py`：运动规划与逆运动学实验主程序。
- `keep_in_contact.py`、`push_door.py`、`push_obj.py`：零水平集运动、推门/推物体实验代码。
- `pybullet_panda_sim.py`：PyBullet仿真环境接口。
- `ref_environment.py`：参考环境设置。
- `throw_wik.py`、`wik_eval.py`：守门员实验与逆运动学评估。
- `README.md`：Franka相关说明。
- `data/`：存放不同机器人的数据（目前只收集了Panda机器人数据，另两个DoF太高，难以用代码本身提供的方法生成数据）。
- `model_dict/`：存放训练好的模型字典文件（目前只训练了Panda机器人模型）。
- `eval_img/`：评估结果图片。
- `runs/`：TensorBoard训练日志。

使用说明：

数据生成
```bash
python frankaemika/data_generator.py
```
神经CDF模型训练与评估
```bash
python frankaemika/nn_cdf.py
# 参数选择：
# --robot: 选择机器人类型（panda, dexhand, leaphand）
# --train: 是否训练模型
# --eval: 是否评估模型
# --with_writer: 是否使用TensorBoard记录日志
# --signed_distance: 是否使用有符号距离
# --batch_x : x的批量大小
# --batch_q : q的批量大小
# --epoches : 训练轮数
# --max_q_per_link : 每个连杆的最大q采样数
# --device : 训练/评估使用的设备（cuda或cpu）
```
运动规划与逆运动学实验
```bash
python frankaemika/mp_ik.py
python frankaemika/wik_eval.py
python frankaemika/throw_wik.py
```
零水平集运动实验
```bash
python frankaemika/keep_in_contact.py
python frankaemika/push_door.py
python frankaemika/push_obj.py
```

零水平集运动关键函数和功能介绍
2Dexamples 关键函数
Robot2D.surface_points_sampler：采样机器人关节间表面点。
plot_sdf_contour：绘制SDF等高线与可视化。
frankaemika 关键类与函数
CDF：神经CDF模型主类，包含训练、评估、推理等方法。
my_eval_1/my_eval_2：模型评估与可视化，输出MAE/RMSE等指标。
main_loop：运动规划与逆运动学主循环。
main_loop：零水平集运动实验主循环。
analysis_data：数据分析与统计。
用法说明
安装依赖
2D示例运行
可视化SDF与CDF效果，展示梯度投影等功能。

Franka机器人实验
数据生成：
神经CDF模型训练与评估：
运动规划与逆运动学实验：
零水平集运动实验：
评估与可视化
评估结果图片存放于frankaemika/eval_signed_1/等文件夹。
TensorBoard日志可在frankaemika/runs/查看。
参考与扩展
更多用法和实验请参考 README.md。
支持自定义机器人，只需实现前向运动学层并生成SDF数据。
如需了解具体函数实现，可查阅对应源代码文件。例如：

Robot2D.surface_points_sampler
CDF

## 其他工具

### `show_urdf`：用于可视化URDF文件的工具。
[GitHub链接](https://github.com/HeFeiW/urdf_viewer.git)  
#### 目录结构
```
|-- workspace/
    |-- Dockerfile
    |-- .gitignore
    |-- README.md
    |-- src/
        |-- show_urdf/
        |   |-- __init__.py
        |   |-- meshes/
        |   |-- dexhand-meshes/
        |   |-- leaphand-meshes/
        |   |-- launch/
        |   |   |-- display_launch.py
        |   |-- broadcast_rot.py
        |   |-- panda.urdf
        |   |-- dexhand.urdf
        |   |-- leaphand.urdf
        |
        |-- package.xml
        |-- setup.py
```
#### 使用：  
环境配置见`Dockerfile`，也可以手动安装所需ROS2包(`robot_state_publisher_gui`)。
```
ros
```
```bash
cd workspace
colcon build
source install/setup.bash
ros2 launch show_urdf display_launch.py robot:=panda.urdf # choices: panda.urdf, dexhand.urdf, leaphand.urdf，指定使用的urdf文件相对于包目录的路径。
```
加载不同的URDF文件和对应的meshes以可视化对应机器人。提供GUI界面以交互式调整机器人关节角度。
使用时要在setup.py中指定正确的URDF文件路径，meshes路径等，才能在build后正确加载资源。例如：
```python
# setup.py
# ...
def get_data_files():
    data_files = [('share/' + package_name, ['package.xml'])]
    directories_to_install = [# 在此处添加需要安装的目录（在package中即src/show_urdf/的路径）
        'launch',
        'meshes',
        'leaphand-meshes',
        'dexhand-meshes',
    ]
# ...
```
另外，需要检查urdf文件中mesh路径是否正确。例如：
```xml
<mesh filename="package://show_urdf/meshes/panda_link1.stl"/>
```

### `run-docker.bash`：用于快速启动包含所需环境的Docker容器的脚本。
使用：
```bash
./run-docker.bash
# 参数说明：
# --image-name 要运行的Docker镜像名称
# --gpu-device 指定GPU设备
# --ip 显示IP地址
# --port 显示端口
# --no-display 不挂载显示相关配置
# --use-port-forward 使用端口转发
# --port_in_docker Docker容器内的端口
# --port_in_host 主机上的端口
# --volume 额外挂载的卷(格式为 <host_path>:<container_path>)
```

### tensorboard使用说明
运行：
```bash
tensorboard --logdir=/workspace/cdf/frankaemika/runs/ --port 6006 --bind_all
# 然后在浏览器中打开 http://localhost:6006 查看训练日志
```
注意：如果使用Docker容器运行代码，确保容器内的6006端口映射到主机的6006端口。（`. run-docker.bash --use-port-forward --port 6006 --port_in_docker 6006`）

在本地使用MobaXTerm通过SSH隧道转发端口连接到远程服务器的6006端口：
```bash
ssh -L 6006:localhost:6006 user@remote_server
```
