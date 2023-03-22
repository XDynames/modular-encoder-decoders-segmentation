import argparse
import os

import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import Cityscapes, VOCSegmentation

from datasets.constants import (
    VOC_SEG_CLASSES,
    BINARY_CITYSCAPES_SEG_CLASSES,
    CITYSCAPES_SEG_CLASSES,
    MONZA_SEG_CLASSES,
)

# Stores information for implemented dataset, normalisation
# statistics calculated across training and validation sets
db_info = {
    "ade20k": {
        "n_classes": 151,
        "size": [512, 512],
        "ignore_index": -100,
        "class_labels": None,
        "normalisation": [[0.489, 0.465, 0.429], [0.256, 0.253, 0.272]],
    },
    "camvid": {
        "n_classes": 12,
        "size": [360, 480],
        "ignore_index": -100,
        "class_labels": None,
        "normalisation": [[0.391, 0.405, 0.414], [0.297, 0.305, 0.301]],
    },
    "cityscapes": {
        "n_classes": 19,
        "size": [768, 768],
        "ignore_index": -100,
        "class_labels": CITYSCAPES_SEG_CLASSES,
        "normalisation": [[0.288, 0.327, 0.286], [0.190, 0.190, 0.187]],
    },
    "binary_cityscapes": {
        "n_classes": 2,
        "size": [720, 1280],
        "ignore_index": -100,
        "class_labels": BINARY_CITYSCAPES_SEG_CLASSES,
        "normalisation": [[0.288, 0.327, 0.286], [0.190, 0.190, 0.187]],
    },
    "pascal_voc": {
        "n_classes": 21,
        "size": [512, 512],
        "class_labels": VOC_SEG_CLASSES,
        "normalisation": [[0.457, 0.441, 0.405], [0.267, 0.264, 0.281]],
    },
    "kitti": {
        "n_classes": 19,
        "size": [368, 1240],
        "ignore_index": -100,
        "class_labels": None,
        "normalisation": [[0.379, 0.398, 0.384], [0.308, 0.318, 0.326]],
    },
    "imagenet": {
        "normalisation": [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
    },
    "monza": {
        "n_classes": 3,
        "size": [720, 1280],
        "ignore_index": -100,
        "class_labels": MONZA_SEG_CLASSES,
    },
}


# Return the information for the relevant dataset
def getInformation(dataset_name):
    return db_info[dataset_name]


# Returns the dataset specified in the passed arguments
def build_dataset(
    args: argparse.Namespace,
    image_set: str = "train",
    transform: torchvision.transforms = None,
) -> Dataset:
    if args.dataset_name == "camvid":
        dataset = CamVid(
            args.dataset_dir,
            train_transform=transform,
            val_transform=transform,
        )
        dataset.setStage(image_set)
        return dataset

    if args.dataset_name == "pascal_voc":
        return CustomVOC(
            args.dataset_dir,
            image_set=image_set,
            transforms=transform,
        )
    if args.dataset_name == "cityscapes":
        return CustomCityscapes(
            args.dataset_dir,
            split=image_set,
            transforms=transform,
            target_type="semantic",
        )
    if args.dataset_name == "binary_cityscapes":
        return BinaryCityscapes(
            args.dataset_dir,
            split=image_set,
            transforms=transform,
            target_type="semantic",
        )
    if args.dataset_name == "monza":
        return MonzaRoad(
            args.dataset_dir,
            train_transform=transform,
            val_transform=transform,
        )


# Override of pytorch Cityscapes dataset to ensure appropriate label
# formating when loading ground truth segmentation maps
class CustomCityscapes(Cityscapes):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._label_mapping = torch.Tensor(
            [
                -100 if details[2] == 255 else details[2]
                for details in Cityscapes.classes
            ]
        )

    # Change all classes not evaluated to ignore label (-100)
    def pre_process_target(self, target):
        target *= 255
        target = target.long()
        target = self._label_mapping[target]
        return target.long()

    def __getitem__(self, index: int) -> torch.Tensor:
        image, target = super().__getitem__(index)
        target = self.pre_process_target(target)
        return image, target


class BinaryCityscapes(CustomCityscapes):
    def pre_process_target(self, target):
        target *= 255
        target = target.long()
        target = self._label_mapping[target]
        target[target > 0] = 1
        return target.long()


# Override of pytorch Pascal VOC segmentaton dataset to ensure
# appropriate label formating when loading ground truth segmentation maps
class CustomVOC(VOCSegmentation):
    def _encode_target(self, target: torch.Tensor) -> torch.LongTensor:
        # Re-scale to interger labels from [0,1]
        target = target * 255
        # Set any lables above 21 to the ignore label
        target[(target > 21)] = -100
        return target.long()

    def __getitem__(self, index: int) -> torch.Tensor:
        image, target = super().__getitem__(index)
        target = self._encode_target(target)
        return image, target


# Abstract Class for implementing custom databases with
class CustomDataset(Dataset):
    # Returns the number of the current stages file pairs
    def __len__(self):
        return len(self._sampleFiles[self._stage])

    # Identity function to be overridden in each dataset's implementation
    def _encode_target(self, target: torch.Tensor) -> torch.Tensor:
        return target

    # Returns the image and ground truth at the specified index
    # within the set files for the current stage
    def __getitem__(self, idx: int) -> torch.Tensor:
        # Create paths to the images and ground truth requested
        imagePath = os.path.join(
            self._root, self._sampleFiles[self._stage][idx][0]
        )
        targetPath = os.path.join(
            self._root, self._sampleFiles[self._stage][idx][1]
        )

        # Load the files as PIL images
        image = Image.open(imagePath)
        target = Image.open(targetPath)

        # Ensures black and white images can be batched with RGB
        if image.mode == "L":
            image = image.convert("RGB")

        # Apply transforms to the image
        if self._transforms[self._stage]:
            if self._transforms[self._stage + "GT"]:
                image = self._transforms[self._stage](image)
                target = self._transforms[self._stage + "GT"](target)
            else:
                image, target = self._transforms[self._stage](image, target)
            # Encode loaded map to integer labels
            target = self._encode_target(target)
        return image, target

    # Mutator to swap stages
    def setStage(self, stage: str):
        self._stage = stage


class MonzaRoad(CustomDataset):
    # Monza road and track limits dataset
    def __init__(
        self,
        root: str,
        train_transform: torchvision.transforms = None,
        train_transformGT: torchvision.transforms = None,
        val_transform: torchvision.transforms = None,
        val_transformGT: torchvision.transforms = None,
    ):
        # Store database root directory
        self._root = root
        # Store image transforms
        self._transforms = {
            "train": train_transform,
            "trainGT": train_transformGT,
            "val": val_transform,
            "valGT": train_transformGT,
        }
        # Flag for which filelist/transform to be used
        self._stage = "train"
        # Store list of image GT pairs for each section
        self._sampleFiles = {
            "train": self._getImageList("train"),
            "val": self._getImageList("train"),
        }

    def _getImageList(self, stage: str):
        path = os.path.join(self._root, stage)
        filenames = os.listdir(os.path.join(self._root, stage))
        samples = [file.split(".")[0].split("-")[0] for file in filenames]
        file_pairs = [
            (
                os.path.join(path, sample + ".jpeg"),
                os.path.join(path, sample + "-trainids.png"),
            )
            for sample in samples
        ]
        return file_pairs

    def _encode_target(self, target: torch.Tensor) -> torch.LongTensor:
        target = target * 255
        target[target > 2] = 1
        return target.long()


class CamVid(CustomDataset):
    # CamVid Cambridge-driving Labeled Video Database
    # http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/
    def __init__(
        self,
        root: str,
        train_transform: torchvision.transforms = None,
        train_transformGT: torchvision.transforms = None,
        val_transform: torchvision.transforms = None,
        val_transformGT: torchvision.transforms = None,
    ):
        # Store database root directory
        self._root = root
        # Store image transforms
        self._transforms = {
            "train": train_transform,
            "trainGT": train_transformGT,
            "val": val_transform,
            "valGT": train_transformGT,
        }
        # Flag for which filelist/transform to be used
        self._stage = "train"
        # Store list of image GT pairs for each section
        self._sampleFiles = {
            "train": self._getImageList("train"),
            "val": self._getImageList("val"),
        }

    # Infer set lists from root directory and strip
    # filenames for raw and annotated images
    def _getImageList(self, stage: str):
        imageFiles = os.listdir(os.path.join(self._root, stage))
        targetFiles = os.listdir(
            os.path.join(self._root, "-".join([stage, "mask"]))
        )
        file_pairs = [
            (
                os.path.join(stage, image),
                os.path.join("-".join([stage, "mask"]), gt),
            )
            for image, gt in zip(imageFiles, targetFiles)
        ]
        return file_pairs

    # Override _encode_target for CamVid
    def _encode_target(self, target: torch.Tensor) -> torch.LongTensor:
        target = target * 255
        target[target == 255] = -100
        return target.long()


class KITTI(CustomDataset):
    # KITTI Visison Benchmark Suite
    # http://www.cvlibs.net/datasets/kitti/

    def __init__(
        self,
        root: str,
        val_size: int,
        train_transform: torchvision.transforms = None,
        train_transformGT: torchvision.transforms = None,
        val_transform: torchvision.transforms = None,
        val_transformGT: torchvision.transforms = None,
    ):
        # Store database root directory
        self._root = root
        # Store image transforms
        self._transforms = {
            "train": train_transform,
            "trainGT": train_transformGT,
            "val": val_transform,
            "valGT": train_transformGT,
        }
        # Flag for which filelist/transform to be used
        self._stage = "train"

        # Build a list of files for images and ground truths
        # for the folder structure
        imageFiles = os.listdir(
            os.path.join(self._root, "training", "image_2")
        )
        targetFiles = os.listdir(
            os.path.join(self._root, "training", "semantic_trainID")
        )
        valImageFiles = os.listdir(os.path.join(self._root, "val", "image_2"))
        valTargetFiles = os.listdir(
            os.path.join(self._root, "val", "semantic_trainID")
        )
        # Make a list of image target pairs
        trainSamples = [
            (
                os.path.join("training", "image_2", image),
                os.path.join("training", "semantic_trainID", GT),
            )
            for image, GT in zip(imageFiles, targetFiles)
        ]
        valSamples = [
            (
                os.path.join("val", "image_2", image),
                os.path.join("val", "semantic_trainID", GT),
            )
            for image, GT in zip(valImageFiles, valTargetFiles)
        ]

        # Store the list of samples as training and validation
        self._sampleFiles = {"train": trainSamples, "val": valSamples}

    """ Classes that are not assessed
        void_classes = [ 0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15,
                                         16, 18, 29, 30, -1 ]
        # Classes that are assessed
        valid_classes = [ 7, 8, 11, 12, 13, 17, 19, 20, 21, 22,
                          23, 24, 25, 26, 27, 28, 31, 32, 33 ]
    """

    def _encode_target(self, target: torch.Tensor) -> torch.LongTensor:
        target = target * 255
        # Remap 255 to -100
        target[target == 255] = -100
        return target


class ADE20K(CustomDataset):
    # AE20K MIT Scene Parsing Challenge 2016
    # https://groups.csail.mit.edu/vision/datasets/ADE20K/
    def __init__(
        self,
        root: str,
        train_transform: torchvision.transforms = None,
        train_transformGT: torchvision.transforms = None,
        val_transform: torchvision.transforms = None,
        val_transformGT: torchvision.transforms = None,
    ):
        # Store database root directory
        self._root = root
        # Store image transforms
        self._transforms = {
            "train": train_transform,
            "trainGT": train_transformGT,
            "val": val_transform,
            "valGT": train_transformGT,
        }
        # Flag for which filelist/transform to be used
        self._stage = "train"
        # Retrieve filenames for training and validation sets
        trainImageFiles = os.listdir(
            os.path.join(self._root, "images", "training")
        )
        trainTargetFiles = os.listdir(
            os.path.join(self._root, "annotations", "training")
        )
        valImageFiles = os.listdir(
            os.path.join(self._root, "images", "validation")
        )
        valTargetFiles = os.listdir(
            os.path.join(self._root, "annotations", "validation")
        )
        # Convert to image, ground truth pairs
        trainSamples = [
            (
                os.path.join("images", "training", image),
                os.path.join("annotations", "training", GT),
            )
            for image, GT in zip(trainImageFiles, trainTargetFiles)
        ]
        valSamples = [
            (
                os.path.join("images", "validation", image),
                os.path.join("annotations", "validation", GT),
            )
            for image, GT in zip(valImageFiles, valTargetFiles)
        ]
        # Store sample pairs
        self._sampleFiles = {"train": trainSamples, "val": valSamples}

    # Override _encode_target for ADE20K
    def _encode_target(self, target: torch.Tensor) -> torch.LongTensor:
        target = target * 255 - 1
        target[target == -1] = -100
        return target
