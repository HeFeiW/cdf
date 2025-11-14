import torch
import os

# 设置显存分配策略
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'

# 检查 GPU 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 打印显存状态
def print_memory_status(step):
    print(f"[{step}] Allocated memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"[{step}] Reserved memory: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    print(f"[{step}] Free memory: {(torch.cuda.memory_reserved() - torch.cuda.memory_allocated() )/ 1024**2:.2f} MB")
    print("-" * 50)

# 模拟输入数据
def generate_test_data(batch_q, DoF, device):
    q_min = torch.zeros(DoF).to(device)
    q_max = torch.ones(DoF).to(device)
    q_sampled = q_min + torch.rand(batch_q, DoF).to(device) * (q_max - q_min)
    q_sampled.requires_grad = True
    return q_sampled

# 测试函数
def test_inference_d_wrt_q():
    print_memory_status("Start")

    # 模拟输入数据
    batch_q = 1000  # 批量大小
    DoF = 10       # 自由度
    test_sample_num = 1000  # 测试样本数量

    # 生成测试数据
    print("Generating test data...")
    q_sampled = generate_test_data(test_sample_num, DoF, device)
    x = torch.rand(1, 3).to(device)  # 模拟输入点
    print_memory_status("After data generation")

    # 模拟模型
    class DummyModel(torch.nn.Module):
        def forward(self, inputs):
            return torch.norm(inputs, dim=-1, keepdim=True)

    model = DummyModel().to(device)

    # 模拟推理
    print("Running inference...")
    x_cat = x.unsqueeze(1).expand(-1, len(q_sampled), -1).reshape(-1, 3)
    q_cat = q_sampled.unsqueeze(0).expand(len(x), -1, -1).reshape(-1, DoF)
    inputs = torch.cat([x_cat, q_cat], dim=-1)

    # 前向传播
    cdf_pred = model(inputs)
    d = cdf_pred.abs().reshape(len(x), len(q_sampled)).min(dim=0)[0]
    print_memory_status("After forward pass")

    # 计算梯度
    print("Computing gradients...")
    grad = torch.autograd.grad(
        d, q_sampled, torch.ones_like(d),
        retain_graph=True, create_graph=True
    )[0]
    print_memory_status("After gradient computation")

    # 清理显存
    del x, q_sampled, x_cat, q_cat, inputs, cdf_pred, d, grad
    torch.cuda.empty_cache()
    print_memory_status("After cleanup")

# 运行测试
if __name__ == "__main__":
    print("Total GPU memory (MB):", torch.cuda.get_device_properties(0).total_memory / 1024**2)
    print("memory reserved (MB):", torch.cuda.memory_reserved(0) / 1024**2)
    print("memory allocated (MB):", torch.cuda.memory_allocated(0) / 1024**2)

    test_inference_d_wrt_q()