"""
任务4：用Profiler找瓶颈
使用PyTorch Profiler分析训练过程，找到最耗时的操作
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.profiler as profiler
from mobilenet_v2 import get_mobilenet_v2


class RandomDataset(Dataset):
    """生成随机数据的数据集"""
    def __init__(self, size=1000):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # 生成随机图像和标签
        image = torch.randn(3, 32, 32)
        label = torch.randint(0, 10, (1,)).item()
        return image, label


def train_with_profiler(model, device, batch_size=64, num_workers=4, profile_steps=20):
    """使用Profiler进行训练分析"""
    dataset = RandomDataset(size=1000)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True
    )

    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"开始使用Profiler分析训练过程...")
    print(f"配置: batch_size={batch_size}, num_workers={num_workers}")
    print(f"将分析前 {profile_steps} 个训练步骤\n")

    # 配置Profiler
    with profiler.profile(
        activities=[
            profiler.ProfilerActivity.CPU,
            profiler.ProfilerActivity.CUDA,
        ],
        schedule=profiler.schedule(
            wait=1,      # 预热1步
            warmup=2,    # 热身2步
            active=3,    # 活跃记录3步
            repeat=2     # 重复2次
        ),
        on_trace_ready=profiler.tensorboard_trace_handler('./profiler_logs'),
        record_shapes=True,      # 记录张量形状
        profile_memory=True,     # 记录内存使用
        with_stack=True          # 记录Python堆栈
    ) as prof:

        step = 0
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            prof.step()  # 通知profiler步骤完成
            step += 1

            if step % 5 == 0:
                print(f"  步骤 {step}/{profile_steps} 完成")

            if step >= profile_steps:
                break

    print(f"\n✅ Profiling完成！")
    return prof


def analyze_profiler_results(prof):
    """分析Profiler结果"""
    print("\n" + "="*80)
    print("Profiler分析结果")
    print("="*80 + "\n")

    # 1. 按CPU时间排序的前10个操作
    print("【Top 10 最耗CPU时间的操作】")
    print("-"*80)
    print(prof.key_averages().table(
        sort_by="cpu_time_total",
        row_limit=10,
        max_src_column_width=50
    ))

    # 2. 按CUDA时间排序的前10个操作
    print("\n【Top 10 最耗GPU时间的操作】")
    print("-"*80)
    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=10,
        max_src_column_width=50
    ))

    # 3. 按内存使用排序
    print("\n【Top 10 最耗显存的操作】")
    print("-"*80)
    print(prof.key_averages().table(
        sort_by="self_cuda_memory_usage",
        row_limit=10,
        max_src_column_width=50
    ))

    # 4. 导出Chrome trace文件
    trace_file = "profiler_trace.json"
    prof.export_chrome_trace(trace_file)
    print(f"\n📊 Chrome trace文件已导出: {trace_file}")
    print(f"   查看方法：")
    print(f"   1. 打开Chrome浏览器")
    print(f"   2. 访问 chrome://tracing")
    print(f"   3. 点击 'Load' 按钮加载 {trace_file}")

    # 5. 生成分析建议
    print("\n" + "="*80)
    print("分析建议")
    print("="*80)

    key_averages = prof.key_averages()

    # 找出最耗时的操作
    cpu_ops = sorted(key_averages, key=lambda x: x.cpu_time_total, reverse=True)[:3]
    cuda_ops = sorted(key_averages, key=lambda x: x.cuda_time_total, reverse=True)[:3]

    print("\n🔍 最耗时的操作：")
    print(f"\nCPU端：")
    for i, op in enumerate(cpu_ops, 1):
        print(f"  {i}. {op.key}: {op.cpu_time_total/1000:.2f}ms")

    print(f"\nGPU端：")
    for i, op in enumerate(cuda_ops, 1):
        print(f"  {i}. {op.key}: {op.cuda_time_total/1000:.2f}ms")

    print("\n💡 优化建议：")

    # 检查是否有数据加载瓶颈
    data_ops = [op for op in key_averages if 'DataLoader' in op.key or 'data' in op.key.lower()]
    if data_ops and sum(op.cpu_time_total for op in data_ops) > 1000000:  # >1s
        print("  ⚠️  数据加载可能是瓶颈，建议：")
        print("     - 增加 num_workers")
        print("     - 使用 pin_memory=True")
        print("     - 考虑数据预处理优化")

    # 检查是否有CPU到GPU数据传输瓶颈
    copy_ops = [op for op in key_averages if 'copy' in op.key.lower() or 'to' in op.key.lower()]
    if copy_ops and sum(op.cuda_time_total for op in copy_ops) > 500000:  # >0.5s
        print("  ⚠️  数据传输可能是瓶颈，建议：")
        print("     - 使用 pin_memory=True")
        print("     - 使用 non_blocking=True")
        print("     - 减少CPU-GPU数据传输频率")

    # 检查卷积操作
    conv_ops = [op for op in key_averages if 'conv' in op.key.lower()]
    if conv_ops:
        total_conv_time = sum(op.cuda_time_total for op in conv_ops)
        total_time = sum(op.cuda_time_total for op in key_averages)
        if total_time > 0:
            conv_ratio = total_conv_time / total_time * 100
            print(f"  ℹ️  卷积操作占GPU时间的 {conv_ratio:.1f}%")
            if conv_ratio > 60:
                print("     - 这是正常的，卷积是计算密集型操作")
                print("     - 可以考虑使用混合精度训练(AMP)加速")

    print("\n📈 在Chrome Tracing中查看：")
    print("  - 蓝色条：CPU操作")
    print("  - 绿色条：GPU操作（CUDA kernels）")
    print("  - 如果CPU和GPU时间条没有重叠，说明存在互相等待")
    print("  - 寻找最长的时间条，那就是主要瓶颈")

    print("\n" + "="*80 + "\n")


def export_tensorboard_logs():
    """提示如何查看TensorBoard日志"""
    print("\n📊 TensorBoard分析：")
    print("="*80)
    print("除了Chrome trace，还可以使用TensorBoard查看更详细的分析：")
    print("\n运行以下命令启动TensorBoard：")
    print("  tensorboard --logdir=./profiler_logs")
    print("\n然后在浏览器中打开：")
    print("  http://localhost:6006")
    print("\nTensorBoard提供：")
    print("  - 操作级别的性能分析")
    print("  - 内存使用时间线")
    print("  - GPU利用率统计")
    print("  - 算子级别的执行时间分布")
    print("="*80 + "\n")


def run_task4():
    """执行任务4：Profiler分析"""
    print("\n" + "="*80)
    print("任务4：用Profiler找瓶颈")
    print("="*80 + "\n")

    # 检查CUDA是否可用
    if not torch.cuda.is_available():
        print("警告: CUDA不可用，Profiler分析在CPU上进行")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")
        print(f"使用设备: {device}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print()

    # 创建模型
    model = get_mobilenet_v2(num_classes=10).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型: MobileNetV2")
    print(f"参数量: {params:,} ({params/1e6:.2f}M)\n")

    # 运行Profiler
    prof = train_with_profiler(
        model=model,
        device=device,
        batch_size=64,
        num_workers=4,
        profile_steps=20
    )

    # 分析结果
    analyze_profiler_results(prof)

    # TensorBoard提示
    export_tensorboard_logs()

    print("提示：")
    print("  1. 查看 profiler_trace.json 文件了解详细的执行时序")
    print("  2. 在Chrome Tracing中可以缩放和平移查看不同时间段")
    print("  3. 点击具体操作可以看到参数和堆栈信息")
    print("  4. 对比不同配置（batch_size, num_workers等）的trace文件\n")


if __name__ == "__main__":
    run_task4()
