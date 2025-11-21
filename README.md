# 深度学习模型性能测试

## 项目简介

这是一个用于测试深度学习模型性能的工具集，包含四个主要测试任务：

1. **任务1：对比3个模型的显存占用差异**
2. **任务2：找出模型能用的最大批大小**
3. **任务3：找到让GPU使用率最平稳的worker数量**
4. **任务4：用Profiler找瓶颈（进阶）**

## 项目结构

```
.
├── main.py                    # 主入口文件，选择执行哪个任务
├── task1_compare_models.py    # 任务1：对比三个模型
├── task2_max_batch_size.py    # 任务2：测试最大批大小
├── task3_gpu_workers.py       # 任务3：测试最优worker数量
├── task4_profiler.py          # 任务4：Profiler性能分析
├── simple_cnn.py              # SimpleCNN模型定义
├── resnet18.py                # ResNet18模型定义
├── mobilenet_v2.py            # MobileNetV2模型定义
└── pyproject.toml             # 项目依赖配置
```

## 环境要求

- Python >= 3.12
- NVIDIA GPU with CUDA support
- nvidia-smi 命令可用

## 安装依赖

### 使用 uv (推荐)

```bash
uv sync
```

### 使用 pip

```bash
pip install torch torchvision matplotlib numpy
```

## 使用方法

### 运行主程序（交互式选择任务）

```bash
python main.py
```

运行后会显示菜单，选择要执行的任务编号即可。

### 直接运行单个任务

```bash
# 任务1：对比三个模型
python task1_compare_models.py

# 任务2：测试最大批大小
python task2_max_batch_size.py

# 任务3：测试最优worker数量
python task3_gpu_workers.py

# 任务4：Profiler性能分析
python task4_profiler.py
```

## 任务详情

### 任务1：对比3个模型的显存占用差异

**测试模型：**
- SimpleCNN
- ResNet18
- MobileNetV2

**测试参数：**
- 批大小：32
- 训练批次：100

**记录指标：**
- 理论参数量
- 实际显存占用（nvidia-smi）
- PyTorch分配的显存
- 训练一个epoch的时间
- 训练峰值显存

**输出：**
- 控制台打印对比表格

### 任务2：找出模型能用的最大批大小

**测试模型：** MobileNetV2

**测试方法：**
- 批大小从8开始
- 每次翻倍测试：8 → 16 → 32 → 64 → 128 → ...
- 直到出现 CUDA Out of Memory 错误

**输出：**
- 控制台打印每个批大小的显存占用
- 生成图表：`batch_size_memory_usage.png`
  - 横轴：批大小（对数刻度）
  - 纵轴：显存占用（MiB）
  - 红色标记：OOM点

### 任务3：找到让GPU使用率最平稳的worker数量

**测试模型：** MobileNetV2

**测试参数：**
- 批大小：64
- num_workers：0, 2, 4, 8

**测试方法：**
- 在后台线程实时监控GPU利用率
- 每0.5秒采样一次GPU-Util
- 训练100个批次

**记录指标：**
- GPU平均利用率
- 利用率标准差（波动幅度）
- 空闲率（利用率<10%的时间占比）
- 训练总时间

**输出：**
- 控制台打印统计表格和详细描述
- 生成图表：`gpu_utilization_comparison.png`
  - 4个子图分别显示不同worker数量下的GPU利用率曲线
  - 红色虚线：平均利用率
  - 橙色虚线：空闲阈值（10%）

**观察要点：**
- worker=0 时，GPU会有明显的空闲间隔（数据加载在主进程）
- worker数量增加，GPU利用率更平稳
- 过多worker可能导致CPU开销增加

### 任务4：用Profiler找瓶颈（进阶）

**测试模型：** MobileNetV2

**测试参数：**
- 批大小：64
- num_workers：4
- 分析步骤：20步

**测试方法：**
- 使用 PyTorch Profiler 包裹训练循环
- 记录CPU和CUDA操作的时间
- 记录内存使用情况
- 生成详细的trace文件

**输出：**
- 控制台打印Top 10操作（按CPU/GPU/内存排序）
- Chrome trace文件：`profiler_trace.json`
- TensorBoard日志：`./profiler_logs/`

**如何查看Chrome Trace：**
1. 打开Chrome浏览器
2. 访问 `chrome://tracing`
3. 点击"Load"按钮，选择 `profiler_trace.json`
4. 使用WASD键缩放和平移时间轴

**如何查看TensorBoard：**
```bash
tensorboard --logdir=./profiler_logs
```
然后访问 http://localhost:6006

**观察要点：**
- 蓝色条：CPU操作
- 绿色条：GPU Kernel（CUDA操作）
- 找时间条最长的操作（主要瓶颈）
- 检查CPU和GPU时间条是否重叠
  - 重叠：CPU和GPU并行工作，效率高
  - 不重叠：存在等待，效率低
- 查看数据加载、数据传输、计算的时间占比

## 额外说明

### 实时监控GPU（推荐配合任务3使用）

在另一个终端窗口运行：

```bash
watch -n 1 nvidia-smi
```

这样可以实时观察GPU利用率的变化情况。

### 注意事项

1. 运行前确保GPU没有被其他程序占用
2. 任务2可能会触发OOM错误，这是正常的测试行为
3. 任务3需要较长运行时间（约2-5分钟）
4. 任务4会生成较大的trace文件，建议只分析少量步骤
5. 生成的图表会自动保存在当前目录

## 技术栈

- **PyTorch**: 深度学习框架
- **torchvision**: 预训练模型
- **matplotlib**: 数据可视化
- **numpy**: 数值计算
- **tensorboard**: Profiler可视化
- **subprocess**: 调用nvidia-smi获取GPU信息
- **threading**: 后台监控GPU利用率

## 许可证

MIT License
