import pybullet as p
import os
import xml.etree.ElementTree as ET
import numpy as np

def load_sdf_model(sdf_path, base_position, base_orientation):
    """
    从 SDF 文件加载模型，并返回碰撞形状和视觉形状。
    """
    # 解析 SDF 文件
    tree = ET.parse(sdf_path)
    root = tree.getroot()

    # 提取模型信息
    model = root.find("model")
    if model is None:
        raise ValueError("SDF 文件中未找到 <model> 标签")

    # 初始化变量
    collision_shape = None
    visual_shape = None

    # 遍历模型中的链接
    for link in model.findall("link"):
        # 获取碰撞形状
        collision = link.find("collision")
        if collision is not None:
            geometry = collision.find("geometry")
            
            mesh = geometry.find("mesh")
            print(f'Found mesh in collision: {mesh}:{collision}')
            if mesh is not None:
                mesh_file = mesh.find("uri").text.strip()
                mesh_file = os.path.join(os.path.dirname(sdf_path), mesh_file.replace("model://", ""))
                scale = mesh.find("scale")
                mesh_scale = [1, 1, 1] if scale is None else list(map(float, scale.text.strip().split()))
                collision_shape = p.createCollisionShape(
                    shapeType=p.GEOM_MESH,
                    fileName=mesh_file,
                    meshScale=mesh_scale
                )
            else:
                # box, sphere, cylinder 等其他几何形状的处理
                # 可能有多个几何形状，都要处理
                collision_shapes = []
                box = geometry.find("box")
                if box is not None:
                    size = list(map(float, box.find("size").text.strip().split()))
                    collision_shape = p.createCollisionShape(
                        shapeType=p.GEOM_BOX,
                        halfExtents=[s / 2 for s in size]
                    )
                    collision_shapes.append(collision_shape)
                sphere = geometry.find("sphere")
                if sphere is not None:
                    radius = float(sphere.find("radius").text.strip())
                    collision_shape = p.createCollisionShape(
                        shapeType=p.GEOM_SPHERE,
                        radius=radius
                    )
                    collision_shapes.append(collision_shape)
                cylinder = geometry.find("cylinder")
                if cylinder is not None:
                    radius = float(cylinder.find("radius").text.strip())
                    length = float(cylinder.find("length").text.strip())
                    collision_shape = p.createCollisionShape(
                        shapeType=p.GEOM_CYLINDER,
                        radius=radius,
                        height=length
                    )
                    collision_shapes.append(collision_shape)
                if len(collision_shapes) > 1:
                    collision_shape = p.createCollisionShape(
                        shapeType=p.GEOM_COMPOUND,
                        collisionFramePositions=[[0,0,0]]*len(collision_shapes),
                        collisionFrameOrientations=[[0,0,0,1]]*len(collision_shapes),
                        childShapeIndices=collision_shapes
                    )
                

        # 获取视觉形状
        visual = link.find("visual")
        if visual is not None:
            geometry = visual.find("geometry")
            mesh = geometry.find("mesh")
            if mesh is not None:
                mesh_file = mesh.find("uri").text.strip()
                mesh_file = os.path.join(os.path.dirname(sdf_path), mesh_file.replace("model://", ""))
                scale = mesh.find("scale")
                mesh_scale = [1, 1, 1] if scale is None else list(map(float, scale.text.strip().split()))
                visual_shape = p.createVisualShape(
                    shapeType=p.GEOM_MESH,
                    fileName=mesh_file,
                    meshScale=mesh_scale
                )
    # 确保碰撞形状和视觉形状已创建
    if collision_shape is None:
        raise ValueError("未能从 SDF 文件中提取碰撞形状")
    if visual_shape is None:
        raise ValueError("未能从 SDF 文件中提取视觉形状")

    # 创建物体
    obj_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=base_position,
        baseOrientation=base_orientation
    )

    return obj_id
def sample_points_from_obj(obj_id, num_samples=100):
    """Sample points on the surface of the object."""
    # 获取物体的网格数据,如果物体的collision shape是mesh的话
    mesh_data = p.getMeshData(obj_id, -1, flags=p.MESH_DATA_SIMULATION_MESH)
    if len(mesh_data[1]) > 0:
        vertices = np.array(mesh_data[1])
        num_vertices = vertices.shape[0]
        sampled_indices = np.random.choice(num_vertices, num_samples, replace=True)
        sampled_points = vertices[sampled_indices]
    else:
        # 说明物体没有mesh数据，可能是box, sphere, cylinder 等简单形状，在这种情况下，在简单形状表面均匀采样点
        vertices = []
        collision_shapes = p.getCollisionShapeData(obj_id, -1)
        
        print(f'collision_shapes: {collision_shapes}')
        for i in range(collision_shapes+1):
            child_shape = p.getCollisionShapeData(obj_id, i)
            print(f'child_shape: {child_shape}')
            # 递归处理compound shape的子shape
            if child_shape[0][2] == p.GEOM_COMPOUND:
                sub_obj_id = p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=child_shape[0][4],
                    basePosition=[0,0,0],
                    baseOrientation=[0,0,0,1]
                )
            if child_shape[0][2] == p.GEOM_BOX:
                half_extents = child_shape[0][3]
                for i in range(num_samples):
                    face = np.random.randint(0, 6)
                    x = np.random.uniform(-half_extents[0], half_extents[0])
                y = np.random.uniform(-half_extents[1], half_extents[1])
                z = np.random.uniform(-half_extents[2], half_extents[2])
                if face == 0:
                    vertices.append([half_extents[0], y, z])
                elif face == 1:
                    vertices.append([-half_extents[0], y, z])
                elif face == 2:
                    vertices.append([x, half_extents[1], z])
                elif face == 3:
                    vertices.append([x, -half_extents[1], z])
                elif face == 4:
                    vertices.append([x, y, half_extents[2]])
                else:
                    vertices.append([x, y, -half_extents[2]])
            elif p.getCollisionShapeData(obj_id, -1)[0][2] == p.GEOM_SPHERE:
                radius = p.getCollisionShapeData(obj_id, -1)[0][3][0]
                phi = np.pi * (3. - np.sqrt(5.))  # 黄金角
                for i in range(num_samples):
                    y = 1 - (i / float(num_samples - 1)) * 2  # y 从 1 到 -1
                    radius_xy = np.sqrt(1 - y * y)  # 半径在 xy 平面上的投影
                    theta = phi * i  # 黄金角度
                    x = np.cos(theta) * radius_xy
                    z = np.sin(theta) * radius_xy
                    vertices.append([x * radius, y * radius, z * radius])
            elif p.getCollisionShapeData(obj_id, -1)[0][2] == p.GEOM_CYLINDER:
                radius = p.getCollisionShapeData(obj_id, -1)[0][3][0]
                height = p.getCollisionShapeData(obj_id, -1)[0][3][1]
                for i in range(num_samples):
                    # 在圆柱侧面和顶面均匀采样
                    if i % 2 == 0:
                        theta = np.random.uniform(0, 2 * np.pi)
                        z = np.random.uniform(-height / 2, height / 2)
                        x = radius * np.cos(theta)
                        y = radius * np.sin(theta)
                        vertices.append([x, y, z])
                    else:
                        theta = np.random.uniform(0, 2 * np.pi)
                        x = radius * np.cos(theta)
                        y = radius * np.sin(theta)
                        z = height / 2 if np.random.rand() > 0.5 else -height / 2
                        vertices.append([x, y, z])
        vertices = np.array(vertices)
        num_vertices = vertices.shape[0]
        sampled_indices = np.random.choice(num_vertices, num_samples, replace=True)
        sampled_points = vertices[sampled_indices]
    
    # 根据物体的当前位置和朝向变换采样点
    base_pos, base_orn = p.getBasePositionAndOrientation(obj_id)
    rot_matrix = np.array(p.getMatrixFromQuaternion(base_orn)).reshape(3, 3)
    sampled_points = np.dot(sampled_points, rot_matrix.T) + np.array(base_pos)
    return sampled_points
if __name__ == "__main__":
    # 初始化 PyBullet
    p.connect(p.GUI)
    sdf_path = "/workspace/cdf/Threshold_Porcelain_Coffee_Mug_All_Over_Bead_White/model.sdf"
    base_position = [0, 0, 0.5]
    base_orientation = p.getQuaternionFromEuler([0, 0, 0])
    obj_id = load_sdf_model(sdf_path, base_position, base_orientation)
    print(f"Loaded object ID: {obj_id}")
    # 设置物体的位置和朝向
    obj_center = np.array([-0.10, -0.1, 0.42])
    obj_orientation = p.getQuaternionFromEuler([0, 0, 0])  # 无旋转
    p.resetBasePositionAndOrientation(obj_id, obj_center, obj_orientation)
    p.changeDynamics(obj_id, -1, lateralFriction=0.5, spinningFriction=0.1, rollingFriction=0.1)
    # 可视化采样点
    sampled_points = sample_points_from_obj(obj_id, num_samples=100)
    print(f"Sampled points shape: {sampled_points.shape}")
    # 用matplotlib可视化采样点和物体的base位置
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(sampled_points[:, 0], sampled_points[:, 1], sampled_points[:, 2], c='r', marker='o')
        base_pos, _ = p.getBasePositionAndOrientation(obj_id)
        ax.scatter(base_pos[0], base_pos[1], base_pos[2], c='b', marker='^', s=100)
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.show()
    except ImportError:
        print("matplotlib 未安装，无法进行可视化。")
    # 运行模拟
    while True:
        p.stepSimulation()