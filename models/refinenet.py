"""
    RefineNet-LightWeight

    RefineNet-LigthWeight PyTorch for non-commercial purposes

    Copyright (c) 2018, Vladimir Nekrasov (vladimir.nekrasov@adelaide.edu.au)
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice, this
    list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
    FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
    OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder_decoder import EncoderDecoder
from . import resnet, mobilenet
from .utils.helpers import maybe_download
from .utils.layer_factory import conv1x1, conv3x3

# Light Weight RefineNet
class RefineNetLW(EncoderDecoder):

    def __init__(self, encoder: nn.Module, num_classes: int, **kwargs):
        super(RefineNetLW, self).__init__()
        # Back bone feature extractor from model collection
        self._encoder = encoder

        # Light Weight RefineNet Decoder layers
        # 1x1 Dimensionality matching layers
        self._build_dim_reducers()
        # CRP Units
        self._crp_level1 = self._make_crp(256, 256, 4)
        self._crp_level2 = self._make_crp(256, 256, 4)
        self._crp_level3 = self._make_crp(256, 256, 4)

        # Inter-Depth Fusion Modules   
        self._fusion_level3_2 = Fusion(256, 256)      
        self._fusion_level2_1 = Fusion(256, 256)

        # Deep blocks specific to ResNet implementation
        if self._encoder.__class__.__name__ == 'RefinedResNet':
            self._crp_level4 = self._make_crp(512, 512, 4)
            self._fusion_level4_3 = Fusion(512, 256)
        else:
            self._crp_level4 = self._make_crp(256, 256, 4)
            self._fusion_level4_3 = Fusion(256, 256)

        # Classification Layer
        self._classification = conv3x3(256, num_classes, bias=True)

    # Wrapper to construct Chained Residual Pooling units
    def _make_crp(self, in_planes: int, out_planes: int, stages: int):
        layers = [ CRPBlock(in_planes, out_planes,stages) ]
        return nn.Sequential(*layers)

    # Creates incoming representation dimensionality matching convolutions
    def _build_dim_reducers(self):
        chnl_sizes = self._encoder_channels
        for i in range(len(chnl_sizes)):
            # ResNet has a larger terminal CRP of 512 channels
            if  (self._encoder.__class__.__name__ == 'RefinedResNet' and
                                chnl_sizes[i][0].split('_')[0] == 'level4'):
                setattr(self, '_dimRed' + chnl_sizes[i][0], 
                conv1x1(chnl_sizes[i][1], 512, bias=False))
            else:
                setattr(self, '_dimRed' + chnl_sizes[i][0], 
                conv1x1(chnl_sizes[i][1], 256, bias=False))
    
    '''
        Channel dimensionality matching for each intermediate representation
            passed from the encoder network to the decoder
    '''
    def _match_channel_dimensions(
        self,
        representations: List[Tuple[str, torch.Tensor]]
    ) -> List[torch.Tensor]:
        lastConvName, levelReps = ' _ _ ', []
        for representation in representations:
            # Retrieve and use the relevant convolotion
            currentConvName = '_dimRed_' + representation[0]
            currentRep = getattr(self, currentConvName)(representation[1])
            # Sum intermediates from the same level after matching
            if len(currentConvName.split('_')) > 3:
                same_level_rep = currentRep
            elif currentConvName.split('_')[2] == lastConvName.split('_')[2]:
                same_level_rep += currentRep
                levelReps.append(same_level_rep)
            else:
                # Add the final representation for the previous level
                levelReps.append(currentRep)

            lastConvName = currentConvName
        # List of channel matched representations
        return levelReps
    
    def forward(
        self,
        x: torch.Tensor,
        gradient_chk: bool=False,
        upsample: bool=True
    ) -> torch.Tensor:
        # Hacky - gradient checkpoint breaks without this... Seems to add memory
        x.requires_grad = True
    	# Encoder representations
        intermediates = self._encoder(x, gradient_chk)
        l1, l2, l3, l4 = self._match_channel_dimensions(intermediates)
        # Level 4: Deepest representation - 1/32
        l4 = nn.ReLU()(l4)
        l4 = self._crp_level4(l4)
        # Level 3: Intermediate representaton - 1/16 
        # Fusion Level 4 and 3
        l3 = self._fusion_level4_3(l4, l3, l3.size()[2:])
        l3 = self._crp_level3(l3)
        # Level 2: Intermediate representation - 1/8
        # Fusion Level 3 and 2
        l2 = self._fusion_level3_2(l3, l2, l2.size()[2:])
        l2 = self._crp_level2(l2)
        # Level 1: Shallowest representation - 1/4
        # Fusion Level 2 and 1
        l1 = self._fusion_level2_1(l2, l1, l1.size()[2:])
        l1 = self._crp_level1(l1)
        l1 = self._classification(l1)
        if upsample:
            l1 = F.interpolate(l1, mode='bicubic', size=x.shape[2:])
        return l1

    

# Builds the specified version of RefineNet
# num_classes - Number of predictable classes in dataset used
# model_variant - Type of Encoder to use
# pretrained - Initialise to a pretrained version of RefineNet 
def build(
    num_classes: int,
    encoder_variant: str,
    pretrained: bool=False
) -> nn.Module:
    # Extract architecture and type of encoder
    architecture = encoder_variant.split('_')[0]
    version = encoder_variant.split('_')[1]
    encoder = None
    
    if not pretrained: print("Loading ImageNet")
    else: print("Initialised Randomly")

    # Check if model is implemented
    if architecture == 'resnet':
        encoder = resnet.build_resnet(version, not pretrained)

    if architecture == 'mobilenet':
        encoder = mobilenet.build_mobilenet_v2(not pretrained)

    if architecture == 'xception':
        print("TODO: Build Xception call")

    if encoder == None:
        print("Invalid or unimplemented Encoder architecture")
        print("Valid options are: 'resnet', 'mobilenet', 'xception'")

    # Build RefineNet using the Encoder
    model = RefineNetLW(encoder, num_classes)

    # Currently implmeneted for ResNet variants only
    '''
    if pretrained:
        dataset = data_info.get(num_classes, None) # Might cause conflicts later
        bname = version + '_' + dataset.lower()
        if architecture == 'resnet':   key = 'rf_lw' + bname
        if architecture == 'mobilenet': key = 'mb' + bname
        # Get the URL from the hashmap
        url = models_urls[bname]
        # Download and intialise the model to the retrieved pretrain
        model.load_state_dict(maybe_download(key, url), strict=True)
        print('Loaded Pretrained Segmentation Network')
    '''

    return model

# Light Weight Fusion Module
class Fusion(nn.Module):
    # input(1/2)_chnls - number of channel in the tesnor
    # Input2 is the shallower respresentation from the Encoder
    def __init__(self, input1_chnls: torch.Tensor, input2_chnls: torch.Tensor):
        super(Fusion, self).__init__()
        # Dimesionality Reducing convolution layers
        self.input2_dimreduce = conv1x1(input2_chnls, input2_chnls, bias=False)
        self.input1_dimreduce = conv1x1(input1_chnls, input2_chnls, bias=False)

    # Fuses representations from different levels of the Encoder
    # upsample_size - size of the 
    def forward(
        self,
        input1: torch.Tensor,
        input2: torch.Tensor,
        upsample_size: int
    ) -> torch.Tensor:
        # Deeper representation pre-fuse
        input1 = self.input1_dimreduce(input1)
        input1 = nn.Upsample(size=upsample_size, mode='bilinear',
                                                align_corners=True)(input1)
        # Shallow representation pre-fuse
        input2 = self.input2_dimreduce(input2)
        
        # Fuse
        input1 = input1 + input2
        input1 = nn.ReLU()(input1)
        return input1

# Light Weight Chained Residual Pooling units
class CRPBlock(nn.Module):
    # in_planes - Number of Input channels
    # out_planes - Number of Output channels
    # n_stages -  Number of consecutive maxpool, conv1x1 applications 
    #             (2 in paper implimentation, 4 in code)
    def __init__(self, in_planes: int, out_planes: int, n_stages: int):
        super(CRPBlock, self).__init__()
        for i in range(n_stages):
            # Add a seperate 1x1 conv for each stage
            setattr(self, '{}_{}'.format(i + 1, 'outvar_dimred'),
                    conv1x1(in_planes if (i == 0) else out_planes,
                            out_planes, stride=1,
                            bias=False))
        self.stride = 1
        self.n_stages = n_stages
        self.maxpool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)

    # Applies consectuive maxpool and 1x1 conv, summing residuals
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        top = x
        for i in range(self.n_stages):
            top = self.maxpool(top)
            top = getattr(self, '{}_{}'.format(i + 1, 'outvar_dimred'))(top)
            x = top + x
        return x

data_info = {
    7 : 'Person',
    21: 'VOC',
    19: 'cityscapes',
    32: 'camvid',
    40: 'NYU',
    60: 'Context'
    }

models_urls = {
    '50_person'  : 'https://cloudstor.aarnet.edu.au/plus/s/DzfdzBuyOk1yYWk/download',
    '101_person' : 'https://cloudstor.aarnet.edu.au/plus/s/0XsqwZWOWUuE4Uo/download',
    '152_person' : 'https://cloudstor.aarnet.edu.au/plus/s/SohSadvEqXfSTvM/download',

    '50_voc'     : 'https://cloudstor.aarnet.edu.au/plus/s/Nst0gOuoCP08W3G/download',
    '101_voc'    : 'https://cloudstor.aarnet.edu.au/plus/s/A1qRIctPSwB0TnR/download',
    '152_voc'    : 'https://cloudstor.aarnet.edu.au/plus/s/qajvUxhl8mayMCM/download',

    '50_nyu'     : 'https://cloudstor.aarnet.edu.au/plus/s/ZAWooBxGPyPsVlN/download',
    '101_nyu'    : 'https://cloudstor.aarnet.edu.au/plus/s/bOkkdVY4pemBzBe/download',
    '152_nyu'    : 'https://cloudstor.aarnet.edu.au/plus/s/3nW50w7p7pgrtYh/download',

    '101_context': 'https://cloudstor.aarnet.edu.au/plus/s/iWjdRAamKlrOSnm/download',
    '152_context': 'https://cloudstor.aarnet.edu.au/plus/s/KOVS1HaEuuTIuI3/download',
    'v2_voc': 'https://cloudstor.aarnet.edu.au/plus/s/PsEL9uEuxOtIxJV/download', # Not Working
    }