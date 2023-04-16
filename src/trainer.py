import torch
import lightning.pytorch as pl
from src.models.resnet import get_modified_resent


class ImitiationDriver(pl.LightningModule):
    def __init__(self, model_name: str):
        super().__init__()
        self._model = get_modified_resent(model_name)

    def training_step(self, batch: torch.tensor, batch_idx: int):
        images, targets = batch

    def validation_step(self, batch: torch.tensor, batch_idx: int):
        images, targets = batch
