import torch
import lightning.pytorch as pl
from src.models.resnet import get_modified_resent

LOSS_FUNCTIONS = {
    "l1": torch.nn.L1Loss,
    "l2": torch.nn.MSELoss,
    "smooth l1": torch.nn.SmoothL1Loss,
}
ACTION_NAMES = ["throttle", "brake", "steering"]


class ImitiationDriver(pl.LightningModule):
    def __init__(self, model_name: str, loss_name: str, sigmoid_output: bool):
        super().__init__()
        self._model = get_modified_resent(model_name, sigmoid_output)
        self._loss = LOSS_FUNCTIONS[loss_name](reduction="none")
        self._validation_outputs = []

    def training_step(self, batch: torch.tensor, batch_idx: int):
        images, targets = batch
        predicted = self._model(images)
        loss = self._loss(predicted, targets)
        self.log("loss/train", loss.mean(), sync_dist=True)
        self.log_errors(loss, "train")
        return loss.mean()

    def validation_step(self, batch: torch.tensor, batch_idx: int):
        images, targets = batch
        predicted = self._model(images)
        loss = self._loss(predicted, targets)
        self._validation_outputs.append({"val_loss": loss.mean()})
        self.log_errors(loss, "val")

    def log_errors(self, loss: torch.Tensor, stage: str):
        for error, name in zip(loss.mean(dim=1), ACTION_NAMES):
            self.log(f"error/{stage}-{name}", error, sync_dist=True)

    def on_validation_epoch_end(self):
        outputs = self._validation_outputs
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        self.log("loss/val", avg_loss, sync_dist=True)
        self._validation_outputs.clear()
