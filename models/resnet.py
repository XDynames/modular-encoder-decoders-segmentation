'''
    ResNet wrapper of pytorches implementation
        Enables gradient checkpointing and intermediate
        feature representations to be returned in the 
        forward pass to multi-scale decoder networks 
'''
from typing import *

import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .utils.layer_factory import conv3x3


'''
    Use imported version of Pytorch's ResNet to construct encoder
      decoder interfaceable version
'''
class ResnetEncoder(nn.Module):
    def __init__(self, model: nn.Module, output_stride: int):
        super(ResnetEncoder, self).__init__()
        self._output_stride = output_stride

        self.level1 = Ignore2ndArg(nn.Sequential(
                                   *list(model.children())[0:4],   
                                   *list(model.layer1.children())))
        self.level2 = Ignore2ndArg(nn.Sequential(
                                   *list(model.layer2.children())))
        self.level3 = Ignore2ndArg(nn.Sequential( 
                                   *list(model.layer3.children())))
        self.level4 = Ignore2ndArg(nn.Sequential( 
                                   *list(model.layer4.children())))
        # Dummy Tensor so that checkpoint can be used on first conv block
        self._dummy = torch.ones(1, requires_grad=True)
        self._deeplab_surgery()
    
    # Returns intermediate representations for use in decoder
    # representation spatial size depends on output stride [32,16,8]
    def forward(
        self,
        x: torch.Tensor,
        gradient_chk: bool=False
    ) -> List[Tuple[str, torch.Tensor]]:
        if gradient_chk:
            dummy = self._dummy
            l1 = checkpoint(self.level1, x,  dummy)	# 1/4
            l2 = checkpoint(self.level2, l1, dummy)	# 1/8
            l3 = checkpoint(self.level3, l2, dummy)	# 1/16 - 1/8
            l4 = checkpoint(self.level4, l3, dummy)	# 1/32 - 1/16 - 1/8
        else:
            l1 = self.level1(x)     # 1/4
            l2 = self.level2(l1)    # 1/8
            l3 = self.level3(l2)    # 1/16 - 1/8
            l4 = self.level4(l3)    # 1/32 - 1/16 - 1/8

        return [('level1', l1), ('level2', l2), ('level3', l3), ('level4', l4)]
        
    # Network surgery for use in deeplabv3+
    def _deeplab_surgery(self):
        if self._output_stride in {8, 16}:
            self.level4.module[0].downsample[0].stride = (1, 1)
            self.level4.module[0].conv2.stride = (1, 1)
            
        if self._output_stride == 8:
            self.level3.module[0].downsample[0].stride = (1, 1)
            self.level3.module[0].conv2.dilation = (2, 2)
            self.level3.module[0].conv2.padding = (2, 2)
            self.level3.module[0].conv2.stride = (1, 1)
            self.level4.module[0].conv2.dilation = (4, 4)
            self.level4.module[0].conv2.padding = (4, 4)
        
        if self._output_stride == 16:
            self.level4.module[0].conv2.dilation = (2, 2)
            self.level4.module[0].conv2.padding = (2, 2)

# Ignores dummy tensor in checkpointed modules
class Ignore2ndArg(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(
        self,
        x: torch.Tensor,
        dummy_arg: torch.Tensor=None
    ) -> torch.Tensor:
        return self.module(x)

'''
    Returns the specified variant of ResNet, optionaly loaded with Imagenet
        pretrained weights
'''
def build_resnet(
    variant: str='50',
    imagenet: bool=False,
    output_stride: int=32
) -> nn.Module:
    if variant in model_dict.keys():
        model = model_dict[variant](imagenet)
    else:
    	print("Invalid or unimplemented ResNet Variant")
    	print("Valid options are: '18', '32', '50', '101', '152'")
    # Convert to Encoder-Decoder integtable version
    return ResnetEncoder(model, output_stride)

model_dict = {'18'  : torchvision.models.resnet18,
              '34'  : torchvision.models.resnet34,
              '50'  : torchvision.models.resnet50,
              '101' : torchvision.models.resnet101,
              '152' : torchvision.models.resnet152,
            }