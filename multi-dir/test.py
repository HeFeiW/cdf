# 加载数据集，检查数据集结构

import os
import torch
import numpy as np
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
# data_dir = os.path.join(CUR_DIR, '../2Dexamples/data22.pt')
# data = torch.load(data_dir)
# print("Data.shape:", data.shape)
# print("Data type:", type(data))
# xs = []
# qs = []
# for i in range(data.shape[0]):
#     for j in range(data.shape[1]):
#         pass
npy_data = np.load(os.path.join(CUR_DIR, '../2Dexamples/data22.npy'), allow_pickle=True).item()
for key in npy_data.keys():
    print(npy_data[key]['q'].shape, npy_data[key]['p'].shape)
final_data = {
            'x': torch.cat([npy_data[k]['p'].unsqueeze(0) for k in npy_data.keys()], dim=0),
            'q': torch.cat([npy_data[k]['q'].unsqueeze(0) for k in npy_data.keys()], dim=0),
            'k': torch.tensor([k for k in npy_data.keys()]).to(device='cuda:0')
        }
# 保存为pt文件
torch.save(final_data, os.path.join(CUR_DIR, '../2Dexamples/data22_converted.pt'))
    
