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
    mesh_data = p.getMeshData(obj_id, -1, flags=p.MESH_DATA_SIMULATION_MESH)
    vertices = np.array(mesh_data[1])
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