import argparse
from typing import Dict, List

import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
from pytorch_lightning.metrics import functional
from torch import nn

from datasets import segmentation
from models import deeplabv3plus, mobilenet, refinenet, resnet

decoder_types = {
    "deeplab_v3+": deeplabv3plus.DeepLabV3plus,
    "refinenet": refinenet.RefineNetLW,
}
encoder_types = {
    "resnet": resnet.ResnetEncoder,
    "mobilenet": mobilenet.MobilenetV2Encoder,
}
# Add your own encoders implementations here
encoder_variants = {
    "resnet_18": torchvision.models.resnet18,
    "resnet_34": torchvision.models.resnet34,
    "resnet_50": torchvision.models.resnet50,
    "resnet_101": torchvision.models.resnet101,
    "resnet_152": torchvision.models.resnet152,
    "mobilenet_v2": torchvision.models.mobilenet_v2,
}


class SegmentationTrainer(pl.LightningModule):
    def build_model(self) -> nn.Module:
        hparams = self.hparams
        db_info = segmentation.getInformation(hparams.dataset_name)
        n_classes = db_info["num_classes"]

        encoder_variant = hparams.encoder
        encoder_type = hparams.encoder.split("_")[0]

        backbone = encoder_variants[encoder_variant](hparams.imagenet)
        encoder = encoder_types[encoder_type](backbone, hparams.output_stride)
        decoder = decoder_types[hparams.decoder](
            encoder,
            n_classes,
            interpolation_mode="bilinear",
            classification_head=None,
            verbose_sizes=False,
        )
        return decoder

    def configure_optimizers(self) -> torch.optim.Optimizer:
        hparams = self.hparams

        optimiser = torch.optim.SGD(
            self._model.parameters(),
            lr=hparams.lr,
            weight_decay=hparams.decay,
            momentum=hparams.momentum,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimiser, 150)

        return [optimiser], [scheduler]

    def __init__(self, hparams: argparse.Namespace):
        super(SegmentationTrainer, self).__init__()
        # All provided arguments
        self.hparams = hparams
        # Key variables
        self._root_dir = hparams.dataset_dir
        self._decoder_name = hparams.decoder
        self._encoder_name = hparams.encoder
        # Network and Training
        self._model = self.build_model()
        self._loss = nn.CrossEntropyLoss()
        self._grad_ckpt = hparams.gradient_ckpt
        self._eval_metric = functional.classification.iou
        self._device = torch.device(self._get_device)

    def _get_device(self):
        return "cuda:0" if torch.cuda.is_available() else "cpu"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._model(x, self._grad_ckpt)

    def training_step(
        self, batch: torch.Tensor, batch_idx: int, optimizer_idx: int = 0
    ) -> Dict:
        images, targets = batch
        targets = targets.squeeze(1)

        predictions = self.forward(images)

        loss = self._loss(predictions, targets)
        batch_IoU = self._eval_metric(predictions, targets)
        self.log("train_loss", loss)
        self.log("train_IoU", batch_IoU)
        return {"loss": loss}

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> Dict:
        images, targets = batch
        predictions = self.forward(images)
        targets = targets.squeeze(1)
        return {
            "val_loss": self._loss(predictions, targets),
            "val_IoU": self._eval_metric(predictions, targets),
        }

    def validation_epoch_end(self, outputs: List[Dict]) -> Dict:
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_IoU = np.mean([x["val_IoU"].item() for x in outputs])
        self.log("val_loss", avg_loss)
        self.log("val_IoU", avg_IoU)

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> Dict:
        images, targets = batch
        predictions = self.forward(images)
        targets = targets.squeeze()
        return {"val_IoU": self._eval_metric(predictions, targets)}

    def test_epoch_end(self, outputs: List[Dict]) -> Dict:
        avg_IoU = np.mean([x["val_IoU"] for x in outputs])
        print("Finished testing with mIoU: {}%".format(avg_IoU * 100))
        return {"avg_IoU": avg_IoU}
