from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F

from encoder_decoder import EncoderDecoder
from .utils.layer_factory import conv1x1, convbnrelu

class DeepLabV3plus(EncoderDecoder):
    def __init__(
        self,
        backbone: nn.Module,
        n_classes: int,
        in_channels: int,
        out_channels: int,
        atrous_rates: List[int] = [6, 12, 18],
    ):
        super(DeepLabV3plus, self).__init__()

        self._encoder = backbone
        self._aspp = ASPP(in_channels, out_channels, atrous_rates)
        self._decoder = Decoder()
        self._classification = conv1x1(256, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print('input.size()', x.shape)
        _, l2, _, l4 = self._encoder(x)
        print('1/4_logits.size()', l2[1].shape)
        print('1/8 - 1/16 logits', l4[1].shape)
        aspp_features = self._aspp(l4)
        print('aspp_features.size()', aspp_features.shape)
        print('x.size()', x.shape)
        out = self._decoder(l2, aspp_features)
        print('out.size()', out.shape)
        out = self._classification(out)
        return F.interpolate(out, mode='bilinear', size=x.shape[2:])

    def build_dim_reducers(self):
        encoder_channel_sizes = self.get_representation_channels()
        for n_channels, level_name in encoder_channel_sizes:
            # Deeplabv3+ fixed decoder channel width to 48 and uses
            #   the 1/4 and 1/16
            if level_name.split('_')[0] in {'level1', 'level4'}:
                setattr(self, 'dimRed_' + n_channels[1], 
                        conv1x1(n_channels[0], 48, bias=False))

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self._conv_block = nn.Sequential([convbnrelu(304, 256, 3),
                                          convbnrelu(256, 256, 3)])

    def forward(self, low_level_features, aspp_features):
        low_level_features = F.interpolate(low_level_features,
                                            mode='bilinear',
                                            size=aspp_features.shape[2:])
        x = torch.cat([low_level_features, aspp_features], axis=1)
        return self._conv_block(x)

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