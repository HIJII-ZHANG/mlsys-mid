import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
import subprocess
import threading
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


class GPUMonitor:
    """GPU利用率监控器"""
    def __init__(self):
        self.running = False
        self.gpu_utils = []
        self.timestamps = []
        self.start_time = None

    def get_gpu_utilization(self):
        """获取GPU利用率"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,nounits,noheader'],
                capture_output=True, text=True, timeout=1
            )
            return int(result.stdout.strip().split('\n')[0])
        except Exception:
            return 0

    def monitor(self):
        """监控线程"""
        self.start_time = time.time()
        while self.running:
            util = self.get_gpu_utilization()
            current_time = time.time() - self.start_time
            self.gpu_utils.append(util)
            self.timestamps.append(current_time)
            time.sleep(0.5)  # 每0.5秒采样一次

    def start(self):
        """开始监控"""
        self.running = True
        self.gpu_utils = []
        self.timestamps = []
        self.thread = threading.Thread(target=self.monitor)
        self.thread.start()

    def stop(self):
        """停止监控"""
        self.running = False
        self.thread.join()

    def get_stats(self):
        """获取统计信息"""
        if not self.gpu_utils:
            return {}

        import numpy as np
        utils = np.array(self.gpu_utils)

        # 计算空闲时间（GPU利用率低于10%的时间）
        idle_count = np.sum(utils < 10)
        idle_ratio = idle_count / len(utils) * 100

        return {
            'mean': np.mean(utils),
            'std': np.std(utils),
            'min': np.min(utils),
            'max': np.max(utils),
            'idle_ratio': idle_ratio,
            'samples': len(utils)
        }


def train_with_workers(model, device, batch_size, num_workers, num_batches=100):
    """使用指定worker数量训练模型"""
    dataset = RandomDataset(size=num_batches * batch_size)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True if num_workers > 0 else False
    )

    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 启动GPU监控
    monitor = GPUMonitor()
    monitor.start()

    start_time = time.time()

    batch_count = 0
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_count += 1
        if batch_count >= num_batches:
            break

    training_time = time.time() - start_time

    # 停止监控
    monitor.stop()

    stats = monitor.get_stats()

    return {
        'num_workers': num_workers,
        'training_time': training_time,
        'gpu_stats': stats,
        'timestamps': monitor.timestamps,
        'gpu_utils': monitor.gpu_utils
    }


def analyze_results(results):
    """分析不同worker数量的结果"""
    print(f"\n{'='*80}")
    print("GPU利用率分析结果")
    print(f"{'='*80}")
    print(f"{'Worker数':<12} {'训练时间(s)':<15} {'平均利用率(%)':<18} {'标准差(%)':<15} {'空闲率(%)':<15}")
    print(f"{'-'*80}")

    descriptions = []

    for r in results:
        stats = r['gpu_stats']
        print(f"{r['num_workers']:<12} {r['training_time']:<15.2f} "
              f"{stats['mean']:<18.2f} {stats['std']:<15.2f} {stats['idle_ratio']:<15.2f}")

        # 生成描述
        desc = generate_description(r)
        descriptions.append(desc)

    print(f"{'='*80}\n")

    # 打印详细描述
    print("详细观察描述：\n")
    for desc in descriptions:
        print(desc)
        print()

    return descriptions


def generate_description(result):
    """生成文字描述"""
    num_workers = result['num_workers']
    stats = result['gpu_stats']

    # 分析波动情况
    if stats['std'] > 20:
        stability = "波动剧烈"
    elif stats['std'] > 10:
        stability = "波动较大"
    else:
        stability = "相对平稳"

    # 分析空闲情况
    if stats['idle_ratio'] > 20:
        idle_desc = f"GPU经常空闲，约{stats['idle_ratio']:.1f}%的时间处于空闲状态"
    elif stats['idle_ratio'] > 5:
        idle_desc = f"GPU偶尔空闲，空闲率为{stats['idle_ratio']:.1f}%"
    else:
        idle_desc = "GPU保持高利用率，几乎无空闲"

    description = (
        f"【Worker = {num_workers}】\n"
        f"  - GPU平均利用率: {stats['mean']:.1f}%\n"
        f"  - 利用率波动: {stability}（标准差={stats['std']:.1f}%）\n"
        f"  - 空闲情况: {idle_desc}\n"
        f"  - 训练时间: {result['training_time']:.2f}秒"
    )

    # 添加具体观察
    if num_workers == 0:
        description += "\n  - 观察: 数据加载在主进程，训练时会有明显的数据等待间隔"
    elif num_workers >= 4:
        description += "\n  - 观察: 多worker并行加载数据，GPU利用更充分"

    return description


def plot_gpu_utilization(results):
    """绘制GPU利用率对比图"""
    import matplotlib.pyplot as plt
    import matplotlib

    # 配置matplotlib支持中文显示
    matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans', 'WenQuanYi Zen Hei']
    matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, result in enumerate(results):
        ax = axes[idx]

        timestamps = result['timestamps']
        gpu_utils = result['gpu_utils']
        num_workers = result['num_workers']
        stats = result['gpu_stats']

        # 绘制利用率曲线
        ax.plot(timestamps, gpu_utils, linewidth=1.5, alpha=0.7)
        ax.axhline(y=stats['mean'], color='r', linestyle='--',
                   label=f'Average: {stats["mean"]:.1f}%', linewidth=2)
        ax.axhline(y=10, color='orange', linestyle=':',
                   label='空闲阈值 (10%)', linewidth=1.5)

        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel('GPU Utilization (%)', fontsize=11)
        ax.set_title(f'Worker = {num_workers} | 标准差={stats["std"]:.1f}% | 空闲率={stats["idle_ratio"]:.1f}%',
                     fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        ax.set_ylim(-5, 105)

    plt.tight_layout()

    output_file = 'gpu_utilization_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"GPU利用率对比图已保存到: {output_file}")

    plt.show()


def main():
    # 检查CUDA是否可用
    if not torch.cuda.is_available():
        print("错误: 需要CUDA支持才能进行此测试")
        return

    device = torch.device("cuda:0")
    print(f"使用设备: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'='*80}\n")

    # 固定参数
    batch_size = 64
    num_batches = 100
    worker_list = [0, 2, 4, 8]

    print(f"测试配置:")
    print(f"  - 模型: MobileNetV2")
    print(f"  - 批大小: {batch_size}")
    print(f"  - 训练批次数: {num_batches}")
    print(f"  - 测试Worker数量: {worker_list}")
    print(f"\n开始测试...\n")

    results = []

    for num_workers in worker_list:
        print(f"{'='*80}")
        print(f"测试 num_workers = {num_workers}")
        print(f"{'='*80}")

        # 创建新模型实例
        model = get_mobilenet_v2(num_classes=10).to(device)

        # 训练并监控
        result = train_with_workers(model, device, batch_size, num_workers, num_batches)
        results.append(result)

        print(f"完成！训练时间: {result['training_time']:.2f}秒")
        print(f"GPU平均利用率: {result['gpu_stats']['mean']:.2f}%")
        print(f"利用率标准差: {result['gpu_stats']['std']:.2f}%\n")

        # 清理
        del model
        torch.cuda.empty_cache()
        time.sleep(2)

    # 分析结果
    analyze_results(results)

    # 绘制对比图
    plot_gpu_utilization(results)

    # 推荐最佳worker数
    print(f"\n{'='*80}")
    print("推荐建议:")
    print(f"{'='*80}")

    # 找到标准差最小且平均利用率最高的配置
    best_stability = min(results, key=lambda x: x['gpu_stats']['std'])
    best_utilization = max(results, key=lambda x: x['gpu_stats']['mean'])

    print(f"最稳定配置: num_workers = {best_stability['num_workers']} "
          f"(标准差={best_stability['gpu_stats']['std']:.2f}%)")
    print(f"最高利用率配置: num_workers = {best_utilization['num_workers']} "
          f"(平均利用率={best_utilization['gpu_stats']['mean']:.2f}%)")

    # 综合推荐
    recommended = min(results,
                     key=lambda x: x['gpu_stats']['std'] - x['gpu_stats']['mean'] * 0.1)
    print(f"\n综合推荐: num_workers = {recommended['num_workers']}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
