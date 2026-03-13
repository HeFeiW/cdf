import sys
import os
import numpy as np
import copy
import torch
import matplotlib.pyplot as plt
CUR_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(CUR_DIR+'/../../../RDF')
from panda_layers.parallel_robot_layer import ParallelRobotLayer
DATA_PATH = CUR_DIR + '/leaphand/'+'data_with_base_dof_0.pt'
data = torch.load(DATA_PATH, map_location='cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 打印数据的结构
for key in data:
    print(f"{key}: {type(data[key])}, shape: {data[key].shape if isinstance(data[key], torch.Tensor) else 'N/A'}")
    # 检查q 的值范围
    if key == 'q':
        print(f"q min: {data[key].min().item()}, q max: {data[key].max().item()}")