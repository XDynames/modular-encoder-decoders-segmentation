import io
from pathlib import Path
from PIL import Image
from typing import Dict

import numpy as np
import torch
from loguru import logger
from turbojpeg import TJPF_BGRX, TurboJPEG
from torchvision import transforms

from src.interface import AssettoCorsaInterface

TURBO_JPEG = TurboJPEG()

WEIGHTS_PATH = Path(
    "/home/james/Documents/ac-imitation-learning/agents/pixels2actions/weights/resnet18-v1-lr.pt"
)


class BasicImitationAgent(AssettoCorsaInterface):
    """
    Resnet model mapping frames to actions
    """

    def setup(self):
        self.agent = torch.load(WEIGHTS_PATH).to("cuda:0").eval()
        self._transform = transforms.Compose(
            [
                transforms.Resize((1280, 720)),
                transforms.ToTensor(),
            ]
        )

    def behaviour(self, observation: Dict) -> np.array:
        image = observation["image"]
        image = TURBO_JPEG.encode(image, pixel_format=TJPF_BGRX)
        image = Image.open(io.BytesIO(image))
        image = self._transform(image)
        image = image.to("cuda:0").permute(0, 2, 1).unsqueeze(dim=0)
        with torch.no_grad():
            action = self.agent(image)
        logger.info(f"{action}")
        action = action[0].to("cpu").numpy()
        # Rescale throttle and brake to [0,1], and steering angle to be between [-1, 1]
        action[0:2] = np.clip(action[0:2], 0, 1)
        action[2] = np.clip(action[2], -1, 1)
        # action[0] += 0.5
        logger.info(
            f"Throttle: {action[0]:0.2f}, "
            + f"Brake: {action[1]:0.2f}, "
            + f"Steering: {action[2]:0.2f}"
        )
        # Action [steering_angle, brake, throttle]
        return action[::-1]


def main():
    agent = BasicImitationAgent()
    agent.run()


if __name__ == "__main__":
    main()
