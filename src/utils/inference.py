from pathlib import Path
from PIL import Image

import cv2
import numpy as np
import torch
from loguru import logger
from torchvision import transforms
from tqdm import tqdm

from src.datasets.data import ACDataModule

IMAGE_SIZE = (1280, 720)


WEIGHTS_PATH = Path(
    "/home/james/Documents/ac-imitation-learning/agents/pixels2actions/weights/resnet18-v1-lr.pt"
)
DATA_PATH = (
    "/mnt/data/aarc/recordings/monza/audi_r8_lms_2016/imitation-learning-fastlaps"
)


class InferenceVideo:
    """
    Resnet model mapping frames to actions
    """

    def __init__(self):
        self.agent = torch.load(WEIGHTS_PATH).to("cuda:0").eval()
        self._transform = transforms.ToTensor()
        self.dataloader = ACDataModule(DATA_PATH, 1, 1).val_dataloader()

    def infer(self, image: torch.Tensor) -> np.array:
        image = image.to("cuda:0")
        with torch.no_grad():
            action = self.agent(image)
        return action[0].to("cpu").numpy()

    def generate_frame(self, image: torch.Tensor, action: torch.Tensor):
        action_hat = self.infer(image)
        logger.info(f"Predicted {action_hat}")
        logger.info(f"GT: {action}")

    def generate_video(self):
        for sample in tqdm(self.dataloader):
            image, action = sample
            self.generate_frame(image, action)


def main():
    video_generator = InferenceVideo()
    video_generator.generate_video()


if __name__ == "__main__":
    main()
