import torch
import torchvision

from models.deeplabv3plus import DeepLabV3plus
from models.resnet import build_resnet

encoder = build_resnet(output_stride=8)
model = DeepLabV3plus(encoder, 12)

from datasets.segmentation import CamVid

dataset = CamVid("../seg_datasets/CamVid")
image1 = torchvision.transforms.functional.to_tensor(dataset[0][0])
image2 = torchvision.transforms.functional.to_tensor(dataset[1][0])

images = torch.stack([image1, image2], dim=0)

print('Without gradient checkpointing')
model(images, False)
print('\nWith gradient checkpointing')
model(images, True)