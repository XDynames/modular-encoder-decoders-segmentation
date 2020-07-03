import torch
import torchvision

from models import refinenet

print('Testing Mobilenet')
model = refinenet.build(12, 'mobilenet_v2')

from datasets.segmentation import CamVid

dataset = CamVid("../seg_datasets/CamVid")
image1 = torchvision.transforms.functional.to_tensor(dataset[0][0])
image2 = torchvision.transforms.functional.to_tensor(dataset[1][0])

images = torch.stack([image1, image2], dim=0)

print('Without gradient checkpointing')
model(images, False)
print('\nWith gradient checkpointing')
model(images, True)

print('Testing resnet')
model = refinenet.build(24, 'resnet_50')

print('Without gradient checkpointing')
model(images, False)
print('\nWith gradient checkpointing')
model(images, True)