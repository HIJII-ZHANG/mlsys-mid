"""
Heavy ResNet - 一个计算密集的ResNet变体
通过增加层数和通道数来制造更重的计算负担
"""
import torch
import torch.nn as nn
import torchvision.models as models


class HeavyResNet(nn.Module):
    """重量级ResNet50，增加计算复杂度"""
    def __init__(self, num_classes=10):
        super(HeavyResNet, self).__init__()
        # 使用ResNet50作为基础（比ResNet18更重）
        self.backbone = models.resnet50(weights=None)

        # 修改第一层，接受更大的输入
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # 修改最后的全连接层
        self.backbone.fc = nn.Linear(2048, num_classes)

        # 添加额外的计算层，增加延迟
        self.extra_layers = nn.Sequential(
            nn.Conv2d(2048, 2048, kernel_size=3, padding=1),
            nn.BatchNorm2d(2048),
            nn.ReLU(),
            nn.Conv2d(2048, 2048, kernel_size=3, padding=1),
            nn.BatchNorm2d(2048),
            nn.ReLU()
        )

    def forward(self, x):
        # 通过ResNet的前几层
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # 添加额外计算
        residual = x
        x = self.extra_layers(x)
        x = x + residual  # Skip connection

        # 全局平均池化和分类
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.backbone.fc(x)

        return x


def get_heavy_resnet(num_classes=10):
    """获取重量级ResNet模型"""
    return HeavyResNet(num_classes=num_classes)


if __name__ == "__main__":
    # 测试模型
    model = get_heavy_resnet(num_classes=10)

    # 计算参数量
    params = sum(p.numel() for p in model.parameters())
    print(f"HeavyResNet参数量: {params:,} ({params/1e6:.2f}M)")

    # 测试前向传播
    x = torch.randn(4, 3, 224, 224)
    y = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {y.shape}")
