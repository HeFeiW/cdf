# 在pybullet中使用Franka Panda/leaphand机器人进行抓取任务
# 命令行参数：
# --step_size: 在zero-level-set上移动的步长，默认值为1
# --display_mode: 显示模式，GUI或DIRECT，默认值为GUI
# --robot: 机器人名称，默认值为leaphand
# --model_dict: 用于保存/加载模型字典的路径，默认值为finger_no_base.pt
# --device: 使用的设备，默认值为cuda
# --data_path: data/下用于保存结果的子目录，默认值为test
# --cdf: 使用CDF进行碰撞检测
# --qp: 对CDF使用QP投影
# --sdf_type: 使用的SDF模型类型，bp_sdf、siren_sdf或qsdf，默认值为bp_sdf

import pybullet as p
import pybullet_data as pd
import numpy as np
import sys
from task_utils import seperate_target_obstacle
sys.path.append("../../RDF")
sys.path.append("../../RDF/panda_layers")
from parallel_bf_sdf import ParallelBPSDF
from qsdf import QSDF
from parallel_robot_layer import ParallelRobotLayer
from para_nn_cdf import CDF
from pybullet_utils import load_sdf_model, sample_points_from_obj
from qp_mp_tao import QPPlanner
import time
from pybullet_robot_sim import PandaSim, SphereManager
import torch
import os
CUR_PATH = os.path.dirname(os.path.realpath(__file__))
from mlp import MLPRegression
import argparse
def plt_contact_map(contact_points, obj, robot,p):
    """Plot the contact points between the robot and the object."""
    import matplotlib.pyplot as plt
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D

    contact_positions = np.array([cp[5] for cp in contact_points])  # Extract contact positions
    if len(contact_positions) == 0:
        print("No contact points found.")
        return
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(contact_positions[:, 0], contact_positions[:, 1], contact_positions[:, 2], c='r', marker='o', label='Contact Points')
    # Plot the robot and object
    robot_pos, robot_orn = p.getBasePositionAndOrientation(robot.panda)
    obj_pos, obj_orn = p.getBasePositionAndOrientation(obj)
    robot_shape = p.getVisualShapeData(robot.panda)
    obj_shape = p.getVisualShapeData(obj)
    for shape in robot_shape:
        if shape[2] == p.GEOM_SPHERE:
            radius = shape[3][0]
            sphere = SphereManager(p, robot_pos, radius, color=[0, 1, 0, 0.5])
            sphere.draw()
    for shape in obj_shape:
        if shape[2] == p.GEOM_SPHERE:
            radius = shape[3][0]
            sphere = SphereManager(p, obj_pos, radius, color=[1, 0, 0, 0.5])
            sphere.draw()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Contact Points between Robot and Object')
    ax.legend()
    plt.show()
# def sample_points_from_box(obj_id, num_samples=100):
#     """Sample points on the surface of the object."""
#     position, orientation = p.getBasePositionAndOrientation(obj_id)
#     half_extents = np.array(p.getVisualShapeData(obj_id)[0][3])  # Get half extents from visual shape data
#     points = []
#     for _ in range(num_samples):
#         x = np.random.uniform(-half_extents[0], half_extents[0])
#         y = np.random.uniform(-half_extents[1], half_extents[1])
#         z = np.random.uniform(-half_extents[2], half_extents[2])
#     #    Transform the point to the object's local space and orient it
#         point = np.array([x, y, z])
#         point = np.dot(np.array(p.getMatrixFromQuaternion(orientation)).reshape(-1, 3), point) + np.array(position)
#         points.append(point)
#     # Convert to numpy array
#     return np.array(points)

def main_loop():
    # 处理命令行参数
    parser = argparse.ArgumentParser(description='Franka Panda CDF Example')
    parser.add_argument('--step_size', type=float, default=1, help='Step size for moving on the zero-level set')
    parser.add_argument('--display_mode', type=str, default='GUI', choices=['GUI', 'DIRECT'], help='Display mode: GUI or DIRECT')
    parser.add_argument('--robot', type=str, default='leaphand', help='Robot name (default: leaphand)')
    parser.add_argument('--model_dict', type=str, default='finger_no_base.pt', help='Path to save/load the model dictionary')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (default: cuda)')
    parser.add_argument('--data_path', type=str, default='data_finger_no_base.pt', help='Sub-directory in data/ to save the results (default: test)')
    parser.add_argument('--cdf', action='store_true', help='Use CDF for collision checking')
    parser.add_argument('--qp', action='store_true', help='Use QP projection for CDF')
    parser.add_argument('--sdf_type', type=str, default='qsdf', choices=['bp_sdf', 'siren_sdf','qsdf'], help='Type of SDF model to use (default: bp_sdf)')
    args = parser.parse_args()
    N_FUNC = 8
    DOMAIN_MIN = -1.0
    DOMAIN_MAX = 1.0
    # 在zero-level-set上移动的step_size
    step_size = args.step_size
    # 设置显示模式
    display_mode = args.display_mode
    if display_mode == 'GUI':
        p.connect(p.GUI, options='--background_color_red=0.5 --background_color_green=0.5' +
                                    ' --background_color_blue=0.5 --width=1600 --height=1000')
    else:
        p.connect(p.DIRECT)
    loop_num = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    # --------- load model and cdf -----------
    CUR_DIR = os.path.dirname(os.path.realpath(__file__))
    SDF_DIR = os.path.dirname(os.path.realpath(__file__))+"/../../RDF"
    paths = {
        'urdf': os.path.join(SDF_DIR,f'descriptions/{args.robot}/*.urdf'),
        'meshes': os.path.join(SDF_DIR,f'descriptions/{args.robot}/meshes/*.stl'),
        'points': os.path.join(SDF_DIR,f'data/{args.robot}/sdf_points/'),
        # 'model':os.path.join(SDF_DIR, f'models/{args.robot}/BP_{N_FUNC}.pt'), # bp_sdf model dict
        'model':'/workspace/RDF/siren_model.pth', # siren_sdf model dict
        'data': os.path.join(CUR_DIR,f'data/{args.robot}/{args.data_path}'),
        'model_dict': {0: os.path.join(CUR_DIR,f'model_dict/{args.robot}/finger_no_base.pt'),
                       1: os.path.join(CUR_DIR,f'model_dict/{args.robot}/finger_no_base.pt'),
                       2: os.path.join(CUR_DIR,f'model_dict/{args.robot}/finger_no_base.pt'),
                       3: os.path.join(CUR_DIR,f'model_dict/{args.robot}/thumb_good_fingertip.pt')
            }
        }
    # --- load the robot layer ---
    robot_layer = ParallelRobotLayer(device=args.device,paths=paths,robot=args.robot)
            
    # --- use sdf or cdf for collision checking ---
    use_cdf = args.cdf
    use_qp = args.qp
    if use_cdf:
        cdf = CDF(device=args.device, paths=paths,robot=args.robot,signed_distance=True,serial_idx=0)
        # --- TODO: temporarily load the cdf model like this for leaphand,
        # with input_dims=4+3=7 ---
        cdf_models = []
        for i in range(len(robot_layer.serials)):
            cdf_model = MLPRegression(input_dims=7, output_dims=1, mlp_layers=[1024, 512, 256, 128, 128],skips=[], act_fn=torch.nn.ReLU, nerf=True)
            cdf_model.load_state_dict(torch.load(paths['model_dict'][i])[49900])
            cdf_model.to(device)
            cdf_models.append(cdf_model)
        # --- use qp or projection for cdf ---
        if use_qp:
            # initialize the QP solver
            qp_solver = QPPlanner(robot_layer=robot_layer,
                                cdf=cdf,
                                cdf_models=cdf_models,
                                dt=0.01,
                                cons_u=2.7,
                                solver='ipopt',
                                safety_buffer=0.7,
                                device=args.device)
            # debug: temporarily set full target joint positions as max joint positions
            xf_full = robot_layer.theta_max.unsqueeze(0).cpu().numpy()
    else:
        if use_qp:
            print('Warning: --qp flag is set but --cdf flag is not set. Ignoring --qp flag.')
            use_qp = False
        # --- load the bp_sdf model ---
        bp_sdf = ParallelBPSDF(n_func=N_FUNC,
                            domain_min=DOMAIN_MIN,
                            domain_max=DOMAIN_MAX,
                            robot=robot_layer,
                            device=args.device,
                            paths=paths)
        
        # --- load the q_sdf model ---
        if args.sdf_type == 'qsdf':
            q_sdfs = []
            for serial in robot_layer.serials:
                used_links = serial.all_links.copy()
                if 'palm_lower_left' in used_links:
                    used_links.remove('palm_lower_left') 
                q_sdfs.append(QSDF(robot=serial,paths=paths,device=args.device,used_links=used_links))
        
        # --- load the bp_sdf model ---
        elif args.sdf_type == 'bp_sdf':
            bp_sdf_model = ParallelBPSDF(n_func=N_FUNC,
                                        domain_min=DOMAIN_MIN,
                                        domain_max=DOMAIN_MAX,
                                        robot=robot_layer,
                                        device=args.device,
                                        paths=paths)

    # --------- pybullet setup -----------
    p.setPhysicsEngineParameter(maxNumCmdPer1ms=1000)
    p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)
    #p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=110, cameraPitch=-10, cameraTargetPosition=[0, 0, 0.5])
    # p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=90, cameraPitch=0, cameraTargetPosition=[0, 0, 0.5])
    #p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=110, cameraPitch=-25, cameraTargetPosition=[0, 0, 0.5])
    p.setAdditionalSearchPath(pd.getDataPath())
    
    # NOTE: need high frequency
    hz = 1000
    delta_t = 1.0 / hz
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(delta_t)
    p.setRealTimeSimulation(0)
    
    # --------- load robot -----------
    base_pos = [0, 0, 0.5]
    base_rot = p.getQuaternionFromEuler([0, 0, 0])
    robot = PandaSim(p, base_pos, base_rot)
    q0 = robot.get_joint_positions()
    print('q0=',q0)
    q_init = torch.tensor([q0],requires_grad=True).to(device).float()
    q_init[0,13] = -0.3
    q_init[0,12] = -1.0
    robot.set_joint_positions(q_init[0].detach().cpu().numpy())
    p.stepSimulation()
    pandaNumDofs = robot.dof
    for joint_idx in range(pandaNumDofs):
        p.enableJointForceTorqueSensor(robot.panda, joint_idx, enableSensor=True)
    # 用idx_mask来将sdf的关节值映射到robot的关节值
    IDX_MASK = torch.zeros(pandaNumDofs).int()
    for joint in robot.Joint2Idx.keys():
        IDX_MASK[robot.Joint2Idx[joint][0]] = robot_layer.Joint2Idx[joint]
    # # # ----------- load plane -----------
    # plane_id = p.createCollisionShape(p.GEOM_PLANE)
    # p.createMultiBody(baseMass=0, baseCollisionShapeIndex=plane_id, basePosition=[0, 0, 0])
    # # ----------- load box -----------
    # box_size = np.array([0.6,0.01,0.3])
    # box_center = np.array([0.0,0.3,0.3])
    # box = p.createVisualShape(p.GEOM_BOX, halfExtents=box_size, rgbaColor=[0.8500, 0.3250, 0.0980, 1.0])
    # p.createMultiBody(baseVisualShapeIndex=box,
    #                                     basePosition=box_center)

    # ------------- load a object to grasp -----------
    obj_to_grasp = 'mug'  # 'sphere' or 'mug'
    # ----- sphere -----
    if obj_to_grasp == 'sphere':
        obj_size = np.array([0.02,0.02,0.02])
        obj_center = np.array([-0.10, -0.1, 0.42])
        obj_orientation = [0, 0, 0, 1]  # No rotation
        # sphere
        obj_visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.06, rgbaColor=[0.8500, 0.3250, 0.0980, 1.0])
        obj_collision = p.createCollisionShape(p.GEOM_SPHERE, radius=0.06)
        # set baseMass=0 to make the object static
        obj = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=obj_collision, baseVisualShapeIndex=obj_visual, basePosition=obj_center)
        p.changeDynamics(obj, -1, lateralFriction=0.5, spinningFriction=0.1, rollingFriction=0.1)
    # ----- mug -----
    elif obj_to_grasp == 'mug':
        model_path = "/workspace/cdf/models/"
        # 列出model_path下的所有文件夹
        mug_models = [f for f in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, f))]
        print(f"Available mug models: {mug_models}")
        idx = input(f"Select a mug model by index (0 to {len(mug_models)-1}): ")
        sdf_path = f"/workspace/cdf/models/{mug_models[int(idx)]}/model.sdf"
        base_position = [0, 0, 0.5]
        base_orientation = p.getQuaternionFromEuler([0, 0, 0])
        obj = load_sdf_model(sdf_path, base_position, base_orientation)
        print(f"Loaded object ID: {obj}")
        # 设置物体的位置和朝向
        if idx == '1':
            obj_center = np.array([-0.12, -0.1, 0.2])
            obj_orientation = (0.025025054425244452, 0.025025054425244556, -0.7066638144485776, 0.7066638144485773)
            p.resetBasePositionAndOrientation(obj, obj_center, obj_orientation)
            p.changeDynamics(obj, -1, lateralFriction=0.5, spinningFriction=0.1, rollingFriction=0.1)
        obj_center = np.array([-0.10, 0, 0.4])
        obj_orientation = p.getQuaternionFromEuler([np.pi/2, np.pi/2, 0])  # 无旋转
        # obj_center = np.array([-0.08, -0.04, 0.35])
        # obj_orientation = (-0.010324690776177509, 0.06031225002339917, -0.7045299372610894, 0.7070314001233444)
        p.resetBasePositionAndOrientation(obj, obj_center, obj_orientation)
        ok = False
        while not ok and display_mode == 'GUI':
            # 用上下左右，< > 键调整物体x, y, z位置变大或变小
            
            # 用1,2,3 键调整物体绕x,y,z轴旋转
            p.stepSimulation()
            keys = p.getKeyboardEvents()
            for k in keys:
                print(f'key pressed: {k}')
                if keys[k] & p.KEY_WAS_TRIGGERED:
                    if k == ord('i'):  # up
                        obj_center[1] += 0.01
                    elif k == ord('k'):  # down
                        obj_center[1] -= 0.01
                    elif k == ord('j'):  # left
                        obj_center[0] -= 0.01
                    elif k == ord('l'):  # right
                        obj_center[0] += 0.01
                    elif k == ord('u'):  # z+
                        obj_center[2] += 0.01
                    elif k == ord('o'):  # z-
                        obj_center[2] -= 0.01
                    elif k == ord('1'):  # rot x+
                        euler = list(p.getEulerFromQuaternion(obj_orientation))
                        euler[0] += 0.1
                        obj_orientation = p.getQuaternionFromEuler(euler)
                    elif k == ord('2'):  # rot y+
                        euler = list(p.getEulerFromQuaternion(obj_orientation))
                        euler[1] += 0.1
                        obj_orientation = p.getQuaternionFromEuler(euler)
                    elif k == ord('3'):  # rot z+
                        euler = list(p.getEulerFromQuaternion(obj_orientation))
                        euler[2] += 0.1
                        obj_orientation = p.getQuaternionFromEuler(euler)
                    elif k == ord('4'):  # rot x-
                        euler = list(p.getEulerFromQuaternion(obj_orientation))
                        euler[0] -= 0.1
                        obj_orientation = p.getQuaternionFromEuler(euler)
                    elif k == ord('5'):  # rot y-
                        euler = list(p.getEulerFromQuaternion(obj_orientation))
                        euler[1] -= 0.1
                        obj_orientation = p.getQuaternionFromEuler(euler)
                    elif k == ord('6'):  # rot z-
                        euler = list(p.getEulerFromQuaternion(obj_orientation))
                        euler[2] -= 0.1
                        obj_orientation = p.getQuaternionFromEuler(euler)
                    elif k == ord('q'):  # quit adjustment
                        ok = True
            p.resetBasePositionAndOrientation(obj, obj_center, obj_orientation)
        p.changeDynamics(obj, -1, lateralFriction=0.5, spinningFriction=0.1, rollingFriction=0.1)
        print(f'Final object center: {obj_center}, orientation: {obj_orientation}')
    # box
    # obj_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=obj_size, rgbaColor=[0.8500, 0.3250, 0.0980, 1.0])
    # obj_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=obj_size)
    
    # cylinder
    # obj_visual = p.createVisualShape(p.GEOM_CYLINDER, radius=0.02, length=0.02, rgbaColor=[0.8500, 0.3250, 0.0980, 1.0])
    # obj_collision = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.02, height=0.02)
    
    
    # --- initialize the task space  ---
    task_space = np.array([[-0.5, 0.5], # x-axis
                          [-0.5, 0.5], # y-axis
                          [ 0.0, 1.0]]) # z-axis
    base_pos = np.array([0.0, 0.0, 0.0])
        
    # ------------- initialize the camera -----------
    if display_mode == 'GUI':
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(lightPosition=[5, 5, 5])
        # p.resetDebugVisualizerCamera(cameraDistance=0.25, cameraYaw=180, cameraPitch=0, cameraTargetPosition=[0, 0, 0.45]) # 正侧面
        # p.resetDebugVisualizerCamera(cameraDistance=0.25, cameraYaw=180, cameraPitch=-30, cameraTargetPosition=[0, 0, 0.45]) # 从指尖方向看
        # p.resetDebugVisualizerCamera(cameraDistance=0.25, cameraYaw=90, cameraPitch=-60, cameraTargetPosition=[0, 0, 0.45]) # 从正前方
        p.resetDebugVisualizerCamera(cameraDistance=0.25, cameraYaw=60, cameraPitch=20, cameraTargetPosition=obj_center) # 从指尖方向看
        c=input('Press any key to continue')
    ############ main loop ############
    for _ in range(loop_num):
        for _ in range(300):
            robot.set_joint_positions(q0)
            p.stepSimulation()
        p.resetBasePositionAndOrientation(obj, obj_center, obj_orientation)
        p.resetBaseVelocity(obj, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0])

        q = q_init
        x = sample_points_from_obj(obj, num_samples=1000)
        x = torch.from_numpy(np.array(x)).to(device).float()
        targ_x, obs_x = seperate_target_obstacle(x)
        while(True):
            # print(f'joint positions:{robot.get_joint_positions()}')
            torques = p.getJointStates(robot.panda, range(pandaNumDofs))
            # print(f'torques: {torques}')
            print('')
            position, orientation = p.getBasePositionAndOrientation(obj)

            # 获得robot base link的位置
            robot_base_pos, robot_base_orn = p.getBasePositionAndOrientation(robot.panda)
            matrix = torch.tensor(p.getMatrixFromQuaternion(robot_base_orn)).reshape(3,3).to(device)
            # x_in_robot_frame = (torch.matmul(matrix.T, (x - torch.tensor(robot_base_pos).to(device).float()).T)).T
            targ_x_in_robot_frame = (torch.matmul(matrix.T, (targ_x - torch.tensor(robot_base_pos).to(device).float()).T)).T
            obs_x_in_robot_frame = (torch.matmul(matrix.T, (obs_x - torch.tensor(robot_base_pos).to(device).float()).T)).T
            # x_in_robot_frame: (N, 3)
            q = torch.tensor([robot.get_joint_positions()],requires_grad=True).to(device).float()
            print(f'q:', q)
            contact_points = p.getContactPoints(bodyA=robot.panda, bodyB=obj)
            # 获得contact的robot link index
            contact_links = set([cp[3] for cp in contact_points])
            print(f'contact links: {contact_links}')
            in_contact = bool(contact_points)
            # if not in_contact:
            if True:
                q_next = q.clone()
                # print(f'q_init: {q_init}')
                for i,serial in enumerate(robot_layer.serials):
                    print(f'serial {i}')
                    pose = torch.eye(4).unsqueeze(0).to(device).float()
                    pose[:, :3, 3] = torch.tensor(base_pos).to(device).float()
                    theta = torch.stack([q[:,robot_layer.Joint2Idx[joint]] for joint in serial.Joint2Idx.keys()],dim=-1)
                    if use_cdf:
                        if use_qp:
                            # use qp for cdf
                            # qp 只能处理B=1情况!!
                            full_next_q = qp_solver.step(q_full=q.squeeze(0), obs_pts=obs_x_in_robot_frame, targ_pts=targ_x_in_robot_frame)
                            theta = torch.stack([full_next_q[robot_layer.Joint2Idx[joint]] for joint in serial.Joint2Idx.keys()],dim=-1).unsqueeze(0)
                            print('diff in theta:', theta - torch.stack([q[:,robot_layer.Joint2Idx[joint]] for joint in serial.Joint2Idx.keys()],dim=-1))
                        else:# not using qp
                            # use projection for cdf
                            cdf_model = cdf_models[i]
                            cdf_min, cdf_grad = cdf.inference_d_wrt_q(x_in_robot_frame, theta,cdf_model)
                            q_proj = cdf.projection(theta, cdf_min, cdf_grad)
                            print('cdf_min:', cdf_min)
                            print('cdf_grad:', cdf_grad)
                            print('theta before:', theta)
                            theta = q_proj
                            print('theta after:', theta)
                        if in_contact:
                            print('--- in contact ---')
                            c = input('Press any key to continue')
                    else:
                        if args.sdf_type == 'bp_sdf':
                            sdf,grad = bp_sdf.get_serial_sdf_with_joints_grad_batch(x_in_robot_frame,pose,theta,bp_sdf_model,used_links = None, serial_idx = i)
                            sdf_min, min_idx = torch.min(sdf, dim=1)
                            grad = grad[:, min_idx, :].squeeze(1)
                            theta = theta - grad * step_size
                            print('sdf_min:', sdf_min)
                            print('grad:', grad)
                            print('theta:', theta)
                            print('q_next before:', q_next)
                        elif args.sdf_type == 'qsdf':
                            used_links = []
                            for link in q_sdfs[i].used_links:
                                if robot.Link2Idx[link] not in contact_links:
                                    used_links.append(link)
                                else:
                                    used_links.clear()
                                    
                            print(f'used_links: {used_links}')
                            if len(used_links) == 0:
                                print('no used links, skip this serial')
                                continue
                            sdf,grad = q_sdfs[i].get_sdf_with_joints_grad(x_in_robot_frame,pose,theta,used_links=used_links)
                            sdf_min, min_idx = torch.min(sdf, dim=1)
                            grad = grad[:, min_idx, :].squeeze(1)
                            theta = theta - grad * step_size
                            print('sdf_min:', sdf_min)
                            print('grad:', grad)
                            print('theta:', theta)
                            print('q_next before:', q_next)
                            if in_contact:
                                print('--- in contact ---')
                                c = input('Press any key to continue')
                        elif args.sdf_type == 'siren_sdf':
                            raise NotImplementedError('siren_sdf not implemented yet')
                    for j,joint in enumerate(serial.Joint2Idx.keys()):
                        q_next[:,robot_layer.Joint2Idx[joint]] = theta[:,j]
                print(f'q_next: {q_next}')
                print('simulating...')
                robot.set_joint_positions(q_next[0].data.cpu().numpy())
            else:
                
                print(f'in contact')
                contact_points = p.getContactPoints(bodyA=robot.panda, bodyB=obj)
                print(contact_points)
                continue
                c=input('press any key to continue')
                q = q_init
                d,grad = cdf.inference_d_wrt_q(x_in_robot_frame,q,sdf_model)
                # 找到与grad正交的方向
                n_q = len(grad)

                q_normal = torch.ones(n_q).to(device) - (torch.sum(grad, dim=-1) / (torch.norm(grad, dim=-1)*torch.norm(grad, dim=-1))) * grad
                q_normal = q_normal / torch.norm(q_normal, dim=-1, keepdim=True)
                q_next = q - step_size * q_normal
                q_init = q_next
                time.sleep(0.1)
                print(f'd: {d.cpu().detach().numpy()[0]}')
                # print(f'q_next: {q_next}, q_normal: {q_normal}')
                robot.set_joint_positions(q_next[0].data.cpu().numpy())
            p.stepSimulation()

    p.disconnect()


if __name__ == '__main__':
    main_loop()