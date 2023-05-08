from dataclasses import dataclass
from typing import Tuple

from torchvision.datasets import Cityscapes


@dataclass
class ClassInformation:
    name: str
    train_id: int
    colour: Tuple[int]


VOC_SEG_CLASSES = [
    ClassInformation("ignore", 0, (0, 0, 0)),
    ClassInformation("aeroplane", 1, (0, 0, 0)),
    ClassInformation("bicycle", 2, (0, 0, 0)),
    ClassInformation("bird", 3, (0, 0, 0)),
    ClassInformation("boat", 4, (0, 0, 0)),
    ClassInformation("bottle", 5, (0, 0, 0)),
    ClassInformation("bus", 6, (0, 0, 0)),
    ClassInformation("car", 7, (0, 0, 0)),
    ClassInformation("cat", 8, (0, 0, 0)),
    ClassInformation("chair", 9, (0, 0, 0)),
    ClassInformation("cow", 10, (0, 0, 0)),
    ClassInformation("dining_table", 11, (0, 0, 0)),
    ClassInformation("dog", 12, (0, 0, 0)),
    ClassInformation("horse", 13, (0, 0, 0)),
    ClassInformation("motorbike", 14, (0, 0, 0)),
    ClassInformation("people", 15, (0, 0, 0)),
    ClassInformation("potted_plant", 16, (0, 0, 0)),
    ClassInformation("sheep", 17, (0, 0, 0)),
    ClassInformation("sofa", 18, (0, 0, 0)),
    ClassInformation("train", 19, (0, 0, 0)),
    ClassInformation("tv_monitor", 20, (0, 0, 0)),
]

CITYSCAPES_SEG_CLASSES = [
    ClassInformation(
        class_details.name,
        class_details.train_id,
        class_details.color,
    )
    for class_details in Cityscapes.classes
    if not class_details.ignore_in_eval
]
CITYSCAPES_SEG_CLASSES.append(ClassInformation("ignored", 255, (0, 0, 0)))

BINARY_CITYSCAPES_SEG_CLASSES = [
    ClassInformation("road", 0, (128, 64, 128)),
    ClassInformation("not_road", 1, (0, 0, 0)),
    ClassInformation("ignored", 255, (40, 40, 40)),
]

MONZA_SEG_CLASSES = [
    ClassInformation("road", 0, (84, 84, 84)),
    ClassInformation("curb", 1, (255, 119, 51)),
    ClassInformation("track_limit", 2, (255, 255, 255)),
    ClassInformation("sand", 3, (255, 255, 0)),
    ClassInformation("grass", 4, (170, 255, 128)),
    ClassInformation("vehicle", 5, (255, 42, 0)),
    ClassInformation("structure", 6, (153, 153, 255)),
    ClassInformation("people", 7, (255, 179, 204)),
    ClassInformation("void", -1, (0, 0, 0)),
]
