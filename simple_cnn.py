import torch.nn as nn


class SimpleCNN(nn.Module):
    """简单的卷积神经网络模型（优化版）"""
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        # 卷积层 - 减少通道数
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # 使用全局平均池化替代大的全连接层
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 小的全连接层
        self.fc = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 卷积 + 池化层
        x = self.pool(self.relu(self.conv1(x)))  # 32x32 -> 16x16
        x = self.pool(self.relu(self.conv2(x)))  # 16x16 -> 8x8
        x = self.pool(self.relu(self.conv3(x)))  # 8x8 -> 4x4

        # 全局平均池化
        x = self.global_pool(x)  # 4x4 -> 1x1
        x = x.view(x.size(0), -1)  # 展平为 (batch, 64)

        # 分类层
        x = self.fc(x)
        return x


def get_simple_cnn(num_classes=10):
    """获取SimpleCNN模型"""
    return SimpleCNN(num_classes=num_classes)
