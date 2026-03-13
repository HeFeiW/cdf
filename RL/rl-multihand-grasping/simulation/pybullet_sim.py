"""
pybullet_sim.py
PyBullet 仿真基础封装
"""

import pybullet as p
import pybullet_data
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import os


class PyBulletSim:
    """
    PyBullet 仿真封装类
    
    封装:
    - world reset
    - 加载 plane/table
    - load/unload URDF
    - stepSimulation()
    - contact info
    
    注意: 所有 PyBullet API 调用都使用 physicsClientId 参数，
    以支持多环境并行仿真。
    """
    
    def __init__(self, enable_gui: bool = False):
        """
        初始化仿真环境
        
        Args:
            enable_gui: 是否启用 GUI
        """
        self.enable_gui = enable_gui
        
        # 连接 PyBullet
        if enable_gui:
            self.physics_client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.physics_client)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1, physicsClientId=self.physics_client)
        else:
            self.physics_client = p.connect(p.DIRECT)
        
        # 设置搜索路径
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # 仿真参数
        self.time_step = 1.0 / 240.0
        p.setTimeStep(self.time_step, physicsClientId=self.physics_client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.physics_client)
        
        # 场景物体 ID
        self.plane_id = None
        self.table_id = None
        
    def reset(self) -> None:
        """重置仿真环境"""
        p.resetSimulation(physicsClientId=self.physics_client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.physics_client)
        p.setTimeStep(self.time_step, physicsClientId=self.physics_client)
        
        self.plane_id = None
        self.table_id = None
    
    def load_scene(self, config: dict) -> None:
        """
        加载场景（地面和桌子）
        
        Args:
            config: 环境配置
        """
        # 加载地面
        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self.physics_client)
        
        # 创建桌子
        table_size = config["scene"]["table_size"]
        table_position = config["scene"]["table_position"]
        
        # 创建桌面碰撞形状
        table_collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[s / 2 for s in table_size],
            physicsClientId=self.physics_client
        )
        
        # 创建桌面视觉形状
        table_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[s / 2 for s in table_size],
            rgbaColor=[0.6, 0.4, 0.2, 1],  # 木质颜色
            physicsClientId=self.physics_client
        )
        
        # 创建桌面刚体
        self.table_id = p.createMultiBody(
            baseMass=0,  # 静态物体
            baseCollisionShapeIndex=table_collision,
            baseVisualShapeIndex=table_visual,
            basePosition=table_position,
            physicsClientId=self.physics_client
        )
    
    def step_simulation(self) -> None:
        """执行一步仿真"""
        p.stepSimulation(physicsClientId=self.physics_client)
    
    def load_object(
        self,
        urdf_path: Optional[str],
        position: np.ndarray,
        orientation: np.ndarray,
        shape_type: str = "box",
        size: List[float] = None,
        color: List[float] = None,
        mass: float = 0.1
    ) -> int:
        """
        加载物体
        
        Args:
            urdf_path: URDF 文件路径（如果为 None 则使用原始形状）
            position: 位置
            orientation: 姿态四元数
            shape_type: 形状类型
            size: 尺寸
            color: 颜色
            mass: 质量
            
        Returns:
            object_id: 物体 ID
        """
        if urdf_path is not None and os.path.exists(urdf_path):
            object_id = p.loadURDF(
                urdf_path,
                basePosition=position,
                baseOrientation=orientation,
                physicsClientId=self.physics_client
            )
        else:
            # 使用原始形状
            if size is None:
                size = [0.05, 0.05, 0.05]
            if color is None:
                color = [0.5, 0.5, 0.5, 1]
            
            if shape_type == "box":
                collision_shape = p.createCollisionShape(
                    p.GEOM_BOX,
                    halfExtents=[s / 2 for s in size],
                    physicsClientId=self.physics_client
                )
                visual_shape = p.createVisualShape(
                    p.GEOM_BOX,
                    halfExtents=[s / 2 for s in size],
                    rgbaColor=color,
                    physicsClientId=self.physics_client
                )
            elif shape_type == "sphere":
                collision_shape = p.createCollisionShape(
                    p.GEOM_SPHERE,
                    radius=size[0] / 2,
                    physicsClientId=self.physics_client
                )
                visual_shape = p.createVisualShape(
                    p.GEOM_SPHERE,
                    radius=size[0] / 2,
                    rgbaColor=color,
                    physicsClientId=self.physics_client
                )
            elif shape_type == "cylinder":
                collision_shape = p.createCollisionShape(
                    p.GEOM_CYLINDER,
                    radius=size[0] / 2,
                    height=size[2],
                    physicsClientId=self.physics_client
                )
                visual_shape = p.createVisualShape(
                    p.GEOM_CYLINDER,
                    radius=size[0] / 2,
                    length=size[2],
                    rgbaColor=color,
                    physicsClientId=self.physics_client
                )
            else:
                raise ValueError(f"Unknown shape type: {shape_type}")
            
            object_id = p.createMultiBody(
                baseMass=mass,
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=position,
                baseOrientation=orientation,
                physicsClientId=self.physics_client
            )
        
        return object_id
    
    def load_hand(
        self,
        urdf_path: str,
        base_position: np.ndarray,
        base_orientation: np.ndarray
    ) -> int:
        """
        加载手部 URDF
        
        Args:
            urdf_path: URDF 路径
            base_position: 基座位置
            base_orientation: 基座姿态
            
        Returns:
            hand_id: 手部 ID
        """
        # 如果 URDF 不存在，创建一个简单的手部模型
        if not os.path.exists(urdf_path):
            print(f"URDF file not found: {urdf_path}, creating simple hand model.")
            return self._create_simple_hand(base_position, base_orientation)
        
        hand_id = p.loadURDF(
            urdf_path,
            basePosition=base_position,
            baseOrientation=base_orientation,
            useFixedBase=False,
            flags=p.URDF_USE_SELF_COLLISION,
            physicsClientId=self.physics_client
        )
        return hand_id
    
    def _create_simple_hand(
        self,
        base_position: np.ndarray,
        base_orientation: np.ndarray
    ) -> int:
        """
        创建简单的手部模型（用于测试）
        """
        # 创建手掌
        palm_collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[0.04, 0.04, 0.02],
            physicsClientId=self.physics_client
        )
        palm_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.04, 0.04, 0.02],
            rgbaColor=[0.9, 0.8, 0.7, 1],
            physicsClientId=self.physics_client
        )
        
        hand_id = p.createMultiBody(
            baseMass=0.5,
            baseCollisionShapeIndex=palm_collision,
            baseVisualShapeIndex=palm_visual,
            basePosition=base_position,
            baseOrientation=base_orientation,
            physicsClientId=self.physics_client
        )
        
        return hand_id
    
    def remove_body(self, body_id: int) -> None:
        """移除物体"""
        if body_id is not None:
            p.removeBody(body_id, physicsClientId=self.physics_client)
    
    def get_body_pose(self, body_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取物体位姿
        
        Returns:
            position: (3,) 位置
            orientation: (4,) 四元数
        """
        pos, orn = p.getBasePositionAndOrientation(body_id, physicsClientId=self.physics_client)
        return np.array(pos), np.array(orn)
    
    def set_base_pose(
        self,
        body_id: int,
        position: np.ndarray,
        orientation: np.ndarray
    ) -> None:
        """设置物体基座位姿"""
        p.resetBasePositionAndOrientation(body_id, position, orientation, physicsClientId=self.physics_client)
    
    def get_base_pose(self, body_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """获取物体基座位姿"""
        return self.get_body_pose(body_id)
    
    def set_joint_positions(
        self,
        body_id: int,
        joint_positions: np.ndarray,
        position_gain: float = 0.1,
        velocity_gain: float = 0.01,
        max_force: float = 50.0
    ) -> None:
        """
        设置关节位置
        
        Args:
            body_id: 物体 ID
            joint_positions: 关节位置
            position_gain: 位置增益
            velocity_gain: 速度增益
            max_force: 最大力矩
        """
        num_joints = p.getNumJoints(body_id, physicsClientId=self.physics_client)
        num_positions = len(joint_positions)
        
        joint_indices = list(range(min(num_joints, num_positions)))
        target_positions = joint_positions[:len(joint_indices)]
        
        p.setJointMotorControlArray(
            body_id,
            joint_indices,
            p.POSITION_CONTROL,
            targetPositions=target_positions,
            positionGains=[position_gain] * len(joint_indices),
            velocityGains=[velocity_gain] * len(joint_indices),
            forces=[max_force] * len(joint_indices),
            physicsClientId=self.physics_client
        )
    
    def get_joint_positions(self, body_id: int) -> np.ndarray:
        """获取关节位置"""
        num_joints = p.getNumJoints(body_id, physicsClientId=self.physics_client)
        positions = []
        for i in range(num_joints):
            state = p.getJointState(body_id, i, physicsClientId=self.physics_client)
            positions.append(state[0])
        return np.array(positions)
    
    def get_link_indices(
        self,
        body_id: int,
        link_names: List[str]
    ) -> List[int]:
        """
        获取 link 索引
        
        Args:
            body_id: 物体 ID
            link_names: link 名称列表
            
        Returns:
            indices: link 索引列表
        """
        num_joints = p.getNumJoints(body_id, physicsClientId=self.physics_client)
        name_to_index = {}
        
        for i in range(num_joints):
            joint_info = p.getJointInfo(body_id, i, physicsClientId=self.physics_client)
            link_name = joint_info[12].decode("utf-8")
            name_to_index[link_name] = i
        
        indices = []
        for name in link_names:
            if name in name_to_index:
                indices.append(name_to_index[name])
            else:
                indices.append(-1)  # base link
        
        return indices
    
    def get_link_position(self, body_id: int, link_index: int) -> np.ndarray:
        """获取 link 位置"""
        if link_index < 0:
            pos, _ = self.get_body_pose(body_id)
            return pos
        
        state = p.getLinkState(body_id, link_index, physicsClientId=self.physics_client)
        return np.array(state[0])
    
    def get_contacts(
        self,
        body_a: int,
        body_b_list: List[int]
    ) -> List[Dict]:
        """
        获取接触信息
        
        Args:
            body_a: 主物体 ID
            body_b_list: 被接触物体 ID 列表
            
        Returns:
            contacts: 接触信息列表
        """
        contacts = []
        
        for body_b in body_b_list:
            contact_points = p.getContactPoints(body_a, body_b, physicsClientId=self.physics_client)
            for cp in contact_points:
                contacts.append({
                    "body_a": body_a,
                    "body_b": body_b,
                    "link_a": cp[3],
                    "link_b": cp[4],
                    "position": np.array(cp[5]),
                    "normal": np.array(cp[7]),
                    "distance": cp[8],
                    "normal_force": cp[9]
                })
        
        return contacts
    
    def check_penetration(
        self,
        body_a: int,
        body_b_list: List[int]
    ) -> bool:
        """
        检查是否有穿透
        
        Args:
            body_a: 主物体 ID
            body_b_list: 被检测物体 ID 列表
            
        Returns:
            has_penetration: 是否有穿透
        """
        for body_b in body_b_list:
            contact_points = p.getContactPoints(body_a, body_b, physicsClientId=self.physics_client)
            for cp in contact_points:
                if cp[8] < -0.001:  # 穿透深度
                    return True
        
        return False
    
    def euler_to_quaternion(self, euler: np.ndarray) -> np.ndarray:
        """欧拉角转四元数"""
        return np.array(p.getQuaternionFromEuler(euler))
    
    def quaternion_to_euler(self, quaternion: np.ndarray) -> np.ndarray:
        """四元数转欧拉角"""
        return np.array(p.getEulerFromQuaternion(quaternion))
    
    def quaternion_to_matrix(self, quaternion: np.ndarray) -> np.ndarray:
        """四元数转旋转矩阵"""
        return np.array(p.getMatrixFromQuaternion(quaternion)).reshape(3, 3)
    
    def compute_view_matrix(
        self,
        target_position: np.ndarray,
        distance: float,
        yaw: float,
        pitch: float,
        roll: float,
        up_axis_index: int = 2
    ) -> np.ndarray:
        """计算视图矩阵"""
        return p.computeViewMatrixFromYawPitchRoll(
            target_position, distance, yaw, pitch, roll, up_axis_index,
            physicsClientId=self.physics_client
        )
    
    def compute_projection_matrix(
        self,
        fov: float,
        aspect: float,
        near: float,
        far: float
    ) -> np.ndarray:
        """计算投影矩阵"""
        return p.computeProjectionMatrixFOV(fov, aspect, near, far)
    
    def get_camera_image(
        self,
        width: int,
        height: int,
        view_matrix: np.ndarray,
        projection_matrix: np.ndarray
    ) -> Tuple:
        """获取相机图像"""
        return p.getCameraImage(
            width, height, view_matrix, projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL if self.enable_gui else p.ER_TINY_RENDERER,
            physicsClientId=self.physics_client
        )
    
    def render_camera(self, width: int = 640, height: int = 480) -> np.ndarray:
        """渲染相机图像"""
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            [0, 0, 0.5], 1.5, 45, -30, 0, 2,
            physicsClientId=self.physics_client
        )
        projection_matrix = p.computeProjectionMatrixFOV(60, width/height, 0.01, 10)
        
        _, _, rgb, _, _ = p.getCameraImage(
            width, height, view_matrix, projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL if self.enable_gui else p.ER_TINY_RENDERER,
            physicsClientId=self.physics_client
        )
        
        return np.array(rgb)[:, :, :3]
    
    def close(self) -> None:
        """关闭仿真"""
        p.disconnect(physicsClientId=self.physics_client)
