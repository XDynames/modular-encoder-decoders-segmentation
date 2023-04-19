from pathlib import Path

import torch

from src.trainer import ImitiationDriver

WEIGHTS_PATH = Path("./agents/pixels2actions/weights/resnet18-v1-lr.ckpt")

model = ImitiationDriver.load_from_checkpoint(
    WEIGHTS_PATH,
    model_name="resnet18",
    loss_name="l1",
    sigmoid_output=False,
)
torch.save(model._model, WEIGHTS_PATH.with_suffix(".pt"))
