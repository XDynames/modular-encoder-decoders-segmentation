
from torch import nn
from torchvision import models

MODELS = {
    'resnet18': models.resnet18,
    'resnet32': models.resnet32,
    'resnet50': models.resnet50,
    'resnet101': models.resnet101,
    'resnet152': models.resnet152,
}


def get_modified_resent(name: str) -> nn.Module:
    resnet = MODELS[name]()
    resnet.fc = nn.Linear(in_features=512, out_features=3, bias=True)
    return resnet
