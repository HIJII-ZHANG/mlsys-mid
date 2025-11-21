import torch
import torch.nn as nn
import torch.optim as optim
import subprocess
import matplotlib.pyplot as plt
import matplotlib
from mobilenet_v2 import get_mobilenet_v2

# 配置matplotlib支持中文显示
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def get_gpu_memory():
    """获取GPU显存使用情况（MB）"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
                              capture_output=True, text=True)
        return int(result.stdout.strip().split('\n')[0])
    except Exception:
        return 0


def test_batch_size(model, device, batch_size):
    """测试指定批大小的显存占用"""
    try:
        model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # 清空缓存
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        # 记录初始显存
        initial_memory = get_gpu_memory()

        # 生成随机输入数据 (batch_size, 3, 32, 32)
        inputs = torch.randn(batch_size, 3, 32, 32).to(device)
        labels = torch.randint(0, 10, (batch_size,)).to(device)

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 记录显存占用
        torch.cuda.synchronize()
        peak_memory_pytorch = torch.cuda.max_memory_allocated(device) / 1024**2  # MB
        final_memory = get_gpu_memory()
        memory_used = final_memory - initial_memory

        return {
            'success': True,
            'memory_mb': memory_used,
            'peak_memory_pytorch': peak_memory_pytorch
        }

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            return {
                'success': False,
                'error': 'OOM',
                'memory_mb': None
            }
        else:
            raise e


def main():
    # 检查CUDA是否可用
    if not torch.cuda.is_available():
        print("错误: 需要CUDA支持才能进行此测试")
        return

    device = torch.device("cuda:0")
    print(f"使用设备: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"总显存: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.0f} MB")
    print(f"{'='*70}\n")

    # 创建模型
    model = get_mobilenet_v2(num_classes=10).to(device)
    print(f"模型: MobileNetV2")
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"参数量: {params:,} ({params/1e6:.2f}M)\n")

    # 测试不同的批大小
    batch_sizes = []
    memory_usage = []
    oom_batch_size = None

    batch_size = 8
    max_batch_size = 2048  # 设置一个上限防止无限循环

    print("开始测试批大小...\n")
    print(f"{'批大小':<12} {'显存占用(MB)':<20} {'状态':<15}")
    print(f"{'-'*50}")

    while batch_size <= max_batch_size:
        result = test_batch_size(model, device, batch_size)

        if result['success']:
            batch_sizes.append(batch_size)
            memory_usage.append(result['memory_mb'])
            print(f"{batch_size:<12} {result['memory_mb']:<20} {'成功':<15}")

            # 清理显存
            torch.cuda.empty_cache()

            # 翻倍批大小
            batch_size *= 2
        else:
            oom_batch_size = batch_size
            print(f"{batch_size:<12} {'N/A':<20} {'OOM (显存不足)':<15}")
            break

    print(f"\n{'='*70}")
    print(f"测试完成！")
    print(f"最大可用批大小: {batch_sizes[-1] if batch_sizes else 'N/A'}")
    if oom_batch_size:
        print(f"OOM批大小: {oom_batch_size}")
    print(f"{'='*70}\n")

    # 绘制显存占用图
    if batch_sizes:
        plt.figure(figsize=(10, 6))
        plt.plot(batch_sizes, memory_usage, marker='o', linewidth=2, markersize=8, label='memory used')

        # 标注每个点的值
        for i, (bs, mem) in enumerate(zip(batch_sizes, memory_usage)):
            plt.annotate(f'{mem} MB',
                        xy=(bs, mem),
                        xytext=(5, 5),
                        textcoords='offset points',
                        fontsize=9)

        # 如果有OOM，标注OOM点
        if oom_batch_size:
            # 在最后一个成功点和OOM点之间画一个标记
            plt.axvline(x=oom_batch_size, color='r', linestyle='--', alpha=0.7, label='OOM point')
            plt.scatter([oom_batch_size], [memory_usage[-1]],
                       color='red', s=200, marker='x', linewidths=3,
                       label=f'OOM (batch_size={oom_batch_size})', zorder=5)
            plt.text(oom_batch_size, memory_usage[-1] * 0.95,
                    'OOM', fontsize=12, color='red', fontweight='bold',
                    ha='center')

        plt.xlabel('Batch Size', fontsize=12)
        plt.ylabel('Memory Used (MiB)', fontsize=12)
        plt.title('MobileNetV2 - Batch Size vs Memory Used', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)

        # 使用对数刻度显示x轴（因为批大小是指数增长）
        plt.xscale('log', base=2)

        # 设置x轴刻度
        if oom_batch_size:
            all_batch_sizes = batch_sizes + [oom_batch_size]
        else:
            all_batch_sizes = batch_sizes
        plt.xticks(all_batch_sizes, [str(bs) for bs in all_batch_sizes])

        plt.tight_layout()

        # 保存图片
        output_file = 'batch_size_memory_usage.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"图表已保存到: {output_file}")

        # 显示图表
        plt.show()


if __name__ == "__main__":
    main()
