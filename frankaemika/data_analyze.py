import sys
import os
import numpy as np
import trimesh
CUR_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = CUR_DIR + '/data_again.npy'
data = np.load(DATA_PATH,allow_pickle=True).item()
scene = trimesh.Scene()
workspace = [[np.inf, -np.inf],[np.inf, -np.inf],[np.inf, -np.inf]]
for k,v in data.items():
    x=[v['x']]
    for i in range(3):
        workspace[i][0] = min(workspace[i][0],x[0][i])
        workspace[i][1] = max(workspace[i][1],x[0][i])
    q=v['q']
    n=len(q)
    print(f'{k}, number of points: {n}')
    # 在scene中添加点云，n==0的点不画,用不同颜色区分n的不同范围,n越大，颜色越红
    if n>0:
        color = np.array([[min(255,50+n*20),0,255-min(200,n*20),200]]*len(x))
        pcd = trimesh.points.PointCloud(x,colors=color)
        scene.add_geometry(pcd)
# 用十二条线段表示出workspace的边界,面透明度为0.1
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