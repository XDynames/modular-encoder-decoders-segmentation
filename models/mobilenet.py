from typing import List, Tuple

import torch
import torchvision
from loguru import logger
from torch import nn
from torch.utils.checkpoint import checkpoint

"""
    MobileNetV2 wrapper of pytorch's implementation
    Enables gradient checkpointing and intermediate
    feature representations to be returned in the
    forward pass to multi-scale decoder networks
"""


class MobilenetV2Encoder(nn.Module):
    def __init__(self, model: nn.Module, output_stride: int = 32):
        super(MobilenetV2Encoder, self).__init__()
        self._output_stride = output_stride
        self._level1 = nn.Sequential(
            *list(model.features)[0:2], *list(model.features)[2:4]
        )
        self._level2 = nn.Sequential(*list(model.features)[4:7])
        self._level3_1 = nn.Sequential(*list(model.features)[7:11])
        self._level3 = nn.Sequential(*list(model.features)[11:14])
        self._level4_1 = nn.Sequential(*list(model.features)[14:17])
        self._level4 = nn.Sequential(model.features[17])

    def forward(
        self, x: torch.Tensor, gradient_chk: bool = False
    ) -> List[Tuple[str, torch.Tensor]]:
        if gradient_chk:
            l1 = checkpoint(self._level1, x)  # 24, 1 / 4
            l2 = checkpoint(self._level2, l1)  # 32, 1 / 8
            l3_1 = checkpoint(self._level3_1, l2)  # 64, 1 / 16
            l3 = checkpoint(self._level3, l3_1)  # 96, 1 / 16
            l4_1 = checkpoint(self._level4_1, l3)  # 160, 1 / 32
            l4 = checkpoint(self._level4, l4_1)  # 320, 1 / 32
        else:
            l1 = self._level1(x)  # 24, 1 / 4
            l2 = self._level2(l1)  # 32, 1 / 8
            l3_1 = self._level3_1(l2)  # 64, 1 / 16
            l3 = self._level3(l3_1)  # 96, 1 / 16
            l4_1 = self._level4_1(l3)  # 160, 1 / 32
            l4 = self._level4(l4_1)  # 320, 1 / 32

        return [
            ("level1", l1),
            ("level2", l2),
            ("level3_1", l3_1),
            ("level3", l3),
            ("level4_1", l4_1),
            ("level4", l4),
        ]


"""
    Returns MobileNetV2 that can be used as an encoder for LWRefinenet
      optionally loaded with Imagenet pretrained weights
"""


def build(variant: str = "v2", imagenet: bool = False) -> nn.Module:
    if imagenet:
        weights = "IMAGENET1K_V2"
        logger.info("Initialising with imagenet pretrained weights")
    model = torchvision.models.mobilenet_v2(weights=weights)
    return model
