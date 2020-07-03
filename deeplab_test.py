import torch

from models.deeplabv3plus import DeepLabV3plus
from models.resnet import build_resnet

encoder = build_resnet(output_stride=16)
model = DeepLabV3plus(encoder, 12)

from ..Bootstrap.datasets.segmentation import CamVid # Fix this

dataset = CamVid("../seg_datasets/CamVid")
x = torch.transform.functional.to_tensor(dataset[0][0])

model(x)