from torchvision import models
import torch.nn as nn


def get_mobilenet_v2(num_classes=10):
    """获取MobileNetV2模型"""
    model = models.mobilenet_v2(weights=None)
    # 修改最后的分类器以适应类别数
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model
