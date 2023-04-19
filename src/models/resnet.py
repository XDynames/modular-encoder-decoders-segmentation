import torch
from torch import nn
from torchvision import models

MODELS = {
    "resnet18": models.resnet18,
    "resnet32": models.resnet34,
    "resnet50": models.resnet50,
    "resnet101": models.resnet101,
    "resnet152": models.resnet152,
}


class SteeringAdjustment(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x[:, 2] = (x[:, 2] * 2) - 1
        return x


def get_modified_resent(name: str, sigmoid_output: bool) -> nn.Module:
    resnet = MODELS[name]()
    if sigmoid_output:
        resnet.fc = nn.sequential(
            [
                nn.Linear(in_features=512, out_features=3, bias=True),
                nn.Sigmoid(),
                SteeringAdjustment(),
            ]
        )
    else:
        resnet.fc = nn.Linear(in_features=512, out_features=3, bias=True)
    return resnet
