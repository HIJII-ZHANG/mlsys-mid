"""
任务1：对比3个模型的显存占用差异
"""
import torch
import torch.nn as nn
import torch.optim as optim
import time
import subprocess

from simple_cnn import get_simple_cnn
from resnet18 import get_resnet18
from mobilenet_v2 import get_mobilenet_v2


def count_parameters(model):
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_gpu_memory():
    """获取GPU显存使用情况"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
                              capture_output=True, text=True)
        return int(result.stdout.strip().split('\n')[0])
    except Exception:
        return 0


def train_one_epoch(model, device, batch_size=32):
    """训练一个epoch并记录时间"""
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 创建随机数据模拟训练
    num_batches = 100  # 模拟100个batch

    start_time = time.time()

    for _ in range(num_batches):
        # 生成随机输入数据 (batch_size, 3, 32, 32) - CIFAR-10 size
        inputs = torch.randn(batch_size, 3, 32, 32).to(device)
        labels = torch.randint(0, 10, (batch_size,)).to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    end_time = time.time()
    return end_time - start_time


def test_model(model_name, model, device, batch_size=32):
    """测试单个模型"""
    print(f"\n{'='*60}")
    print(f"测试模型: {model_name}")
    print(f"{'='*60}")

    model = model.to(device)

    # 1. 计算参数量
    params = count_parameters(model)
    print(f"理论参数量: {params:,} ({params/1e6:.2f}M)")

    # 2. 获取初始显存
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    initial_memory = get_gpu_memory()

    # 进行一次前向传播以分配显存
    dummy_input = torch.randn(batch_size, 3, 32, 32).to(device)
    _ = model(dummy_input)

    # 3. 记录显存占用
    memory_after_model = get_gpu_memory()
    memory_allocated = torch.cuda.memory_allocated(device) / 1024**2  # MB
    memory_reserved = torch.cuda.memory_reserved(device) / 1024**2  # MB

    print(f"实际显存占用 (nvidia-smi): {memory_after_model - initial_memory} MB")
    print(f"PyTorch分配的显存: {memory_allocated:.2f} MB")
    print(f"PyTorch保留的显存: {memory_reserved:.2f} MB")

    # 4. 训练一个epoch并记录时间
    print(f"\n开始训练一个epoch (batch_size={batch_size}, 100 batches)...")
    epoch_time = train_one_epoch(model, device, batch_size)
    print(f"训练一个epoch的时间: {epoch_time:.2f} 秒")

    # 记录训练后的峰值显存
    peak_memory = torch.cuda.max_memory_allocated(device) / 1024**2
    final_memory = get_gpu_memory()
    print(f"训练峰值显存 (PyTorch): {peak_memory:.2f} MB")
    print(f"训练后显存 (nvidia-smi): {final_memory - initial_memory} MB")

    return {
        'model_name': model_name,
        'parameters': params,
        'memory_mb': memory_after_model - initial_memory,
        'peak_memory_mb': peak_memory,
        'epoch_time': epoch_time
    }


def run_task1():
    """执行任务1：对比三个模型"""
    print("\n" + "="*80)
    print("任务1：对比3个模型的显存占用差异")
    print("="*80 + "\n")

    # 检查CUDA是否可用
    if not torch.cuda.is_available():
        print("警告: CUDA不可用，将使用CPU进行测试")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")
        print(f"使用设备: {device}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    batch_size = 32
    results = []

    # 空挂载防止缓存池误差
    print("\n预热GPU缓存池...")
    warmup_model = get_mobilenet_v2(num_classes=10).to(device)
    dummy_input = torch.randn(batch_size, 3, 32, 32).to(device)
    _ = warmup_model(dummy_input)
    del warmup_model, dummy_input
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    time.sleep(2)
    print("预热完成！\n")

    # 1. 测试 SimpleCNN
    print("\n开始测试 SimpleCNN...")
    simple_cnn = get_simple_cnn(num_classes=10)
    result = test_model("SimpleCNN", simple_cnn, device, batch_size)
    results.append(result)
    del simple_cnn
    torch.cuda.empty_cache()
    time.sleep(2)

    # 2. 测试 ResNet18
    print("\n开始测试 ResNet18...")
    resnet18 = get_resnet18(num_classes=10)
    result = test_model("ResNet18", resnet18, device, batch_size)
    results.append(result)
    del resnet18
    torch.cuda.empty_cache()
    time.sleep(2)

    # 3. 测试 MobileNetV2
    print("\n开始测试 MobileNetV2...")
    mobilenet = get_mobilenet_v2(num_classes=10)
    result = test_model("MobileNetV2", mobilenet, device, batch_size)
    results.append(result)
    del mobilenet
    torch.cuda.empty_cache()

    # 打印汇总对比
    print(f"\n{'='*80}")
    print("对比结果汇总")
    print(f"{'='*80}")
    print(f"{'模型':<15} {'参数量(M)':<15} {'显存(MB)':<15} {'峰值显存(MB)':<18} {'训练时间(s)':<15}")
    print(f"{'-'*80}")
    for r in results:
        print(f"{r['model_name']:<15} {r['parameters']/1e6:<15.2f} {r['memory_mb']:<15} "
              f"{r['peak_memory_mb']:<18.2f} {r['epoch_time']:<15.2f}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    run_task1()
