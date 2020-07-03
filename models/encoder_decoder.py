from typing import *

import torch
import torch.nn as nn

class EncoderDecoder(nn.Module):
    def __init__(self):
        super(EncoderDecoder, self).__init__()

    '''
        Channel dimensionality matching for each intermediate representation
            passed from the encoder network to the decoder
    '''
    def match_channel_dimension(self, representations: List[torch.Tensor]):
        lastConvName, levelReps = ' _ ', []
        for representation in representations:
            # Retrieve and use the relevant convolotion
            currentConvName = 'dimRed_' + representation[1]
            currentRep = getattr(self, currentConvName)(representation[0])
            # Sum intermediates from the same level after matching
            if len(currentConvName.split('_')) > 2:
                same_level_rep = currentRep
            elif currentConvName.split('_')[1] == lastConvName.split('_')[1]:
                same_level_rep += currentRep
                levelReps.append(same_level_rep)
            else:
                # Add the final representation for the previous level
                levelReps.append(currentRep)

            lastConvName = currentConvName
        # List of channel matched representations
        return levelReps

    '''
        Retrieves intermediate representation's number of channels
    '''
    def get_representation_channels(self):
        currentLevel, chnl_sizes = 'level1', []
        # Get the paramters from the model and their names
        for name, param in self._encoder.named_parameters():
            tmpLevel = name.split('.')[0]
            # If it is the last layer of a level, add its name and size
            if not tmpLevel == currentLevel:
                chnl_sizes.append((currentLevel, tmpParam.size()[0]))
                currentLevel = tmpLevel
            tmpParam = param
        
        chnl_sizes.append((tmpLevel, tmpParam.size()[0]))
        return chnl_sizes 