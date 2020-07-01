'''
    ResNet wrapper of pytorches implementation
    Enables gradient checkpointing and intermediate
    feature representations to be returned in the 
    forward pass to multi-scale decoder networks 
'''
import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .weight_standardised import models

# Ignores dummy tensor in checkpointed modules
class Ignore2ndArg(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self,x, dummy_arg=None):
        return self.module(x)

# Use imported version of Pytorch's ResNet to construct encoder
#  decoder interfaceable version
class RefinedResNet(nn.Module):

    def __init__(self, model):
        super(RefinedResNet, self).__init__()
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
        self.dummy = torch.ones(1, requires_grad=True)
        # Dropout
        #self.do = nn.Dropout(p=0.5)

    # Returns intermediate representations for use in RefineNet
    def forward(self, x, gradient_chk=False):
        if gradient_chk:
            dummy = self.dummy
            l1 = checkpoint(self.level1, x,  dummy)	# 1/4
            l2 = checkpoint(self.level2, l1, dummy)	# 1/8
            l3 = checkpoint(self.level3, l2, dummy)	# 1/16
            l4 = checkpoint(self.level4, l3, dummy)	# 1/32
        else:
            l1 = self.level1(x)     # 1/4
            l2 = self.level2(l1)    # 1/8
            l3 = self.level3(l2)    # 1/16
            l4 = self.level4(l3)    # 1/32
        # Dropout layers
        #l4 = self.do(l4)
        #l3 = self.do(l3)

        return [(l1, 'level1'), (l2, 'level2'), (l3, 'level3'), (l4, 'level4')]

# Returns the specified variant of ResNet, optionaly loaded with Imagenet
# pretrained weights or using group normilisation and weight standarsisation
def build_ResNet(variant = '50', imagenet=False, group_norm=False):
    model = None
    model_library = models if group_norm else torchvision.models
    if group_norm: print("Using Group Normalisation and Weight Standardisation")
    else: print("Using Batch Normalisation")
    if variant == '18': model = model_library.resnet18(pretrained = imagenet)
    if variant == '34': model = model_library.resnet34(pretrained = imagenet)
    if variant == '50': model = model_library.resnet50(pretrained = imagenet)
    if variant == '101': model = model_library.resnet101(pretrained = imagenet)
    if variant == '152': model = model_library.resnet152(pretrained = imagenet)
    if model == None:
    	print("Invalid or unimplemented ResNet Variant")
    	print("Valid options are: '18', '32', '50', '101', '152'")
    # Convert to Encoder-Decoder integtable version
    model = RefinedResNet(model)

    return model
