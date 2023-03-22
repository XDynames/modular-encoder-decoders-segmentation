import argparse

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from . import segmentation


class ReSizeNormalise:
    def __init__(self, args: argparse.ArgumentParser):
        db_info = segmentation.getInformation(args.dataset_name)
        pytorch_transforms = [
            transforms.Resize(
                tuple(db_info["size"]),
                InterpolationMode.NEAREST,
            ),
            transforms.ToTensor(),
        ]
        self._preprep = transforms.Compose(pytorch_transforms)
        if "normalisation" in db_info:
            self._normalise = transforms.Normalize(*db_info["normalisation"])
        else:
            self._normalise = None

    def __call__(self, image: Image, target: Image) -> torch.Tensor:
        image = self._preprep(image)
        if self._normalise is not None:
            image = self._normalise(image)
        target = self._preprep(target)
        return image, target
