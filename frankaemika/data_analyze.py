import sys
import os
import numpy as np
import trimesh
import copy
import torch
CUR_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(CUR_DIR+'/../../RDF')
from panda_layer.parallel_robot_layer import ParallelRobotLayer
DATA_PATH = CUR_DIR + '/data/leaphand/'+'data_thumb_no_base.npy'
data = np.load(DATA_PATH,allow_pickle=True).item()
data_with_base = np.load(CUR_DIR + '/data/leaphand/'+'/data_thumb.npy',allow_pickle=True).item()
# scene = trimesh.Scene()
workspace = [[np.inf, -np.inf],[np.inf, -np.inf],[np.inf, -np.inf]]
cnt=0
paths = {
    'urdf': f'../../RDF/descriptions/leaphand/*.urdf',
    'meshes': f'../../RDF/descriptions/leaphand/meshes/*.stl'
}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
robot = ParallelRobotLayer(device,paths=paths,robot='leaphand').to(device)
theta = torch.zeros(1,robot.dof).float().to(device)
pose = torch.from_numpy(np.identity(4)).to(device).reshape(-1, 4, 4).expand(len(theta),-1,-1).float()
trans = robot.get_link_mesh_transformations(pose, theta)
# mcp=trans['mcp_joint']
# mcp_2=trans['mcp_joint_2']
# mcp_3=trans['mcp_joint_3']
# print(mcp)
# print(mcp_2)
# print(mcp_3)
# # 计算mcp_joint到mcp_joint_2的变换矩阵
# mcp_to_mcp2 = torch.matmul(torch.inverse(mcp), mcp_2)[0]
# print('mcp_to_mcp2:',mcp_to_mcp2)
# # 计算mcp_joint到mcp_joint_3的变换矩阵
# mcp_to_mcp3 = torch.matmul(torch.inverse(mcp), mcp_3)[0]
# print('mcp_to_mcp3:',mcp_to_mcp3)
# data2 = copy.deepcopy(data)
# data3 = copy.deepcopy(data)
scene = trimesh.Scene()
for k,v in data.items():
    x=torch.from_numpy(v['x']).float().to(device).unsqueeze(0)
    x_with_base = torch.from_numpy(data_with_base[k]['x']).float().to(device).unsqueeze(0)
    for i in range(3):
        workspace[i][0] = min(workspace[i][0],x[0][i])
        workspace[i][1] = max(workspace[i][1],x[0][i])
    idx = torch.from_numpy(v['idx'])
    q = torch.from_numpy(v['q']).float()
    n = len(idx)
    n_with_base = len(data_with_base[k]['idx'])
    # x2 = torch.matmul(mcp_to_mcp2, torch.cat([x,torch.ones(1,1).to(device)],dim=1).transpose(0,1))[:3,:].transpose(0,1)
    # x3 = torch.matmul(mcp_to_mcp3, torch.cat([x,torch.ones(1,1).to(device)],dim=1).transpose(0,1))[:3,:].transpose(0,1)
    # data2[k]['x'] = x2[0].cpu().numpy()
    # data3[k]['x'] = x3[0].cpu().numpy()
    # print(f'{k}, number of points: {n}')
    # 在scene中添加点云，n==0的点不画,用不同颜色区分n的不同范围,n越大，颜色越红
    if n>0:
        # mask = (idx!=0)
        # q=q[mask]
        # idx=idx[mask]
        # data[k]={
        #     'x':x[0].cpu().numpy(),
        #     'q':q.cpu().numpy(),
        #     'idx':idx.cpu().numpy()
        # }
        
        idx_count = torch.bincount(idx)
        if idx_count[0]>0:
            print(idx_count)
        # if idx_count[0]>0:
        #     print(f'warning: idx 0 exists, count: {idx_count[0]}')
        # 画出点云
        # x = x.cpu().numpy()
        
        # x2 = x2.cpu().numpy()
        # x3 = x3.cpu().numpy()
        # print(x,x2,x3)
        # pcd1 = trimesh.points.PointCloud(x,colors=[255,0,0,100])
        
        # pcd2 = trimesh.points.PointCloud(x2,colors=[0,0,255,255])
        # pcd3 = trimesh.points.PointCloud(x3,colors=[255,0,0,255])
        # scene.add_geometry(pcd1)
        
        # scene.add_geometry(pcd2)
        # scene.add_geometry(pcd3)
    if n_with_base>0:
        idx_count_with_base = torch.bincount(torch.from_numpy(data_with_base[k]['idx']))
        if idx_count_with_base[0]>0:
            print(idx_count_with_base,'with base')
        # x_with_base = x_with_base.cpu().numpy()
        # pcd1_with_base = trimesh.points.PointCloud(x_with_base,colors=[0,0,255,100])
        # scene.add_geometry(pcd1_with_base)
# torch_path = CUR_DIR + '/data_thumb_no_base.npy'
# np.save(torch_path,data)
# np.save(CUR_DIR + '/data_finger_no_base_2.npy',data2)
# np.save(CUR_DIR + '/data_finger_no_base_3.npy',data3)
# print(f'save to {torch_path}')
# # 用十二条线段表示出workspace的边界,面透明度为0.1
# print(f'number of idx 0: {cnt}')
for i in range(3):
    workspace[i][0] = workspace[i][0].cpu().numpy()
    workspace[i][1] = workspace[i][1].cpu().numpy()
extents = [workspace[0][1]-workspace[0][0],workspace[1][1]-workspace[1][0],workspace[2][1]-workspace[2][0]]
transform = trimesh.transformations.translation_matrix([(workspace[0][1]+workspace[0][0])/2,(workspace[1][1]+workspace[1][0])/2,(workspace[2][1]+workspace[2][0])/2])
print(f'workspace: {workspace}')
print(f'extents: {extents}')

box = trimesh.creation.box(
    extents=extents,
    transform=transform,
    face_colors=[200,200,200,50]
    )
scene.add_geometry(box)
scene.show()  # display the scene in an interactive window