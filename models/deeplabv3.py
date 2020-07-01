import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import *

class DeepLabV3plus(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        decoder: nn.Module,
        in_channels: int,
        out_channels: int,
        atrous_rates: List[int] = [6, 12, 18],
    ):
        super(DeepLabV3plus, self).__init__()

        self.encoder = backbone
        self.aspp = ASPP(in_channels, out_channels, atrous_rates)
        self.decoder = decoder

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        print('input.size()', input.size())
        backbone_logits = self.encoder(input)
        print('backbone_logits.size()', backbone_logits.size())
        aspp_features = self.aspp(input)
        print('aspp_features.size()', aspp_features.size())
        x = torch.cat([backbone_logits, aspp_features], axis=1)
        print('x.size()', x.size())
        out = self.decoder(x)
        print('out.size()', out.size())
        return out


class ASPP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, atrous_rates: List[int]):
        super(ASPP, self).__init__()

        # build aspp convs
        self.aspp_convs = self.build_aspp_convs(in_channels, out_channels, atrous_rates)

        self.project = nn.Sequential(
            nn.Conv2d(
                len(self.aspp_convs) * out_channels, out_channels, 1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

    def build_aspp_convs(self, in_channels, out_channels, atrous_rates):
        # 1x1 Conv
        convs = [nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )]
        
        # 3x3 Conv's at rates
        convs.extend([
            ASPPConv(in_channels, out_channels, rate)
            for rate in atrous_rates
        ])
        
        # image pooling
        convs.append(ASPPPooling(in_channels, out_channels))

        return convs

    def forward(self, x):
        res = []
        for conv in self.aspp_convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)
    
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)
    
model = DeepLabV3plus(
    backbone = nn.Sequential(
        nn.Conv2d(3, 256, 3, padding=1, bias=False),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True),
    ),
    decoder = nn.Sequential(
        nn.Conv2d(512, 256, 3, padding=1, bias=False),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 16, 1)
    ),
    in_channels = 3,
    out_channels = 256
    

)
_ = model(torch.zeros((2, 3, 256, 256)))