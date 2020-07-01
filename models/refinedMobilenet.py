'''
	MobileNetV2 wrapper of pytorches implementation
	Enables gradient checkpointing and intermediate
	feature representations to be returned in the 
	forward pass to multi-scale decoder networks 
'''
import torch
import torchvision
from torch import nn
from torch.utils.checkpoint import checkpoint

# MobilenetV2 to be used with Refinenet
class RefinedMobilenetV2(nn.Module):

	def __init__(self, model):
		super(RefinedMobilenetV2, self).__init__()
		self.level1 = nn.Sequential(*list(model.features)[0:2],
									*list(model.features)[2:4])
		self.level2 = nn.Sequential(*list(model.features)[4:7])
		self.level3_1 = nn.Sequential(*list(model.features)[7:11])
		self.level3 = nn.Sequential(*list(model.features)[11:14])
		self.level4_1 = nn.Sequential(*list(model.features)[14:17])
		self.level4 = nn.Sequential(model.features[17])

	def forward(self, x, gradient_chk=False):
		if gradient_chk:
			l1 = checkpoint(self.level1, x)       # 24, 1 / 4
			l2 = checkpoint(self.level2, l1)      # 32, 1 / 8
			l3_1 = checkpoint(self.level3_1, l2)  # 64, 1 / 16
			l3 = checkpoint(self.level3, l3_1)    # 96, 1 / 16
			l4_1 = checkpoint(self.level4_1, l3)  # 160, 1 / 32
			l4 = checkpoint(self.level4, l4_1)    # 320, 1 / 32
		else:
			l1 = self.level1(x)       # 24, 1 / 4
			l2 = self.level2(l1)      # 32, 1 / 8
			l3_1 = self.level3_1(l2)  # 64, 1 / 16
			l3 = self.level3(l3_1)    # 96, 1 / 16
			l4_1 = self.level4_1(l3)  # 160, 1 / 32
			l4 = self.level4(l4_1)    # 320, 1 / 32

		return [ (l1, 'level1'), (l2, 'level2'), (l3_1, 'level3_1'),
				 (l3, 'level3'), (l4_1, 'level4_1'), (l4, 'level4') ]


'''
	Returns MobileNetV2 that can be used as an encoder for LWRefinenet
 	 optionaly loaded with Imagenet pretrained weights
'''
def build_MobilenetV2(imagenet=False):
	model = torchvision.models.mobilenet_v2(pretrained = imagenet)
	# Convert to encoder version
	model = RefinedMobilenetV2(model)
	return model
