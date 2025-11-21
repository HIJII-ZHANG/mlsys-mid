from torchvision import models


def get_resnet18(num_classes=10):
    """获取ResNet18模型"""
    model = models.resnet18(weights=None)
    # 修改最后的全连接层以适应类别数
    model.fc = models.resnet.nn.Linear(model.fc.in_features, num_classes)
    return model
