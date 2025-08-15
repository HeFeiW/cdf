import torch
print(torch.cuda.is_available())  # 应为 True
print(torch.rand(1).cuda())   