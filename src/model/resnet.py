import torch
from torch import nn
from torchvision.models import resnet18


class ResNet(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()  # type: ignore
        self.model: nn.Module = resnet18(pretrained=True)
        self.model.fc = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
