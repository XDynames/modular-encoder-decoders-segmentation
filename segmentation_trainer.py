import argparse
from typing import Dict, List

import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
from loguru import logger
from torch import nn
from torchmetrics.classification import MulticlassJaccardIndex

from datasets import segmentation
from models import deeplabv3plus, mobilenet, refinenet, resnet

decoder_types = {
    "deeplab": deeplabv3plus.DeepLabV3plus,
    "refinenet": refinenet.RefineNetLW,
}
encoder_types = {
    "resnet": resnet.ResnetEncoder,
    "mobilenet": mobilenet.MobilenetV2Encoder,
}
# Add your own encoders implementations here
encoder_variants = {
    "resnet": resnet.build,
    "mobilenet": torchvision.models.mobilenet_v2,
}


class SegmentationTrainer(pl.LightningModule):
    def build_model(self) -> nn.Module:
        hparams = self.hparams["hparams"]
        db_info = segmentation.getInformation(hparams.dataset_name)
        n_classes = db_info["n_classes"]

        encoder_type, encoder_variant = hparams.encoder.split("_")

        backbone = encoder_variants[encoder_type](
            encoder_variant, hparams.imagenet
        )
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
        hparams = self.hparams["hparams"]
        optimiser = torch.optim.SGD(
            self._model.parameters(),
            lr=hparams.lr,
            weight_decay=hparams.decay,
            momentum=hparams.momentum,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimiser,
            step_size=hparams.step_lr_every_n_steps,
            gamma=hparams.lr_step_factor,
        )
        return [optimiser], [scheduler]

    def __init__(self, hparams: argparse.Namespace):
        super().__init__()
        self.save_hyperparameters()
        # Network and Training
        self._model = self.build_model()
        self._loss = nn.CrossEntropyLoss()
        self._grad_ckpt = hparams.gradient_ckpt
        self._setup_evaluation_metric()
        self._setup_class_labels()

    def _setup_evaluation_metric(self):
        hparams = self.hparams["hparams"]
        db_info = segmentation.getInformation(hparams.dataset_name)
        self._eval_metric = MulticlassJaccardIndex(
            db_info["n_classes"],
            average=None,
            ignore_index=db_info["ignore_index"],
        )

    def _setup_class_labels(self):
        hparams = self.hparams["hparams"]
        db_info = segmentation.getInformation(hparams.dataset_name)
        if db_info["class_labels"] is None:
            class_labels = range(0, db_info["n_classes"])
        else:
            class_labels = self._get_class_names(db_info)
        self.class_labels = class_labels

    def _get_class_names(self, db_info: Dict) -> List[str]:
        class_labels = sorted(
            db_info["class_labels"],
            key=lambda x: x.train_id,
        )
        return [class_info.name for class_info in class_labels]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._model(x, self._grad_ckpt)

    def training_step(
        self,
        batch: torch.Tensor,
        batch_idx: int,
        optimizer_idx: int = 0,
    ) -> Dict:
        images, targets = batch
        targets = targets.squeeze(1)

        predictions = self.forward(images)

        loss = self._loss(predictions, targets)
        batch_ious = self._eval_metric(predictions, targets)
        self.log("loss/train", loss, sync_dist=True)
        self.log_iou("train", batch_ious)
        return {"loss": loss}

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> Dict:
        images, targets = batch
        predictions = self.forward(images)
        targets = targets.squeeze(1)
        return {
            "val_loss": self._loss(predictions, targets),
            "val_iou": self._eval_metric(predictions, targets),
        }

    def validation_epoch_end(self, outputs: List[Dict]) -> Dict:
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_ious = torch.stack([x["val_iou"] for x in outputs]).mean(dim=0)
        self.log("loss/val", avg_loss, sync_dist=True)
        self.log_iou("val", avg_ious)

    def log_iou(self, stage: str, label_ious: torch.Tensor):
        for iou, label in zip(label_ious, self.class_labels):
            self.log(f"iou/{stage}_{label}", iou, sync_dist=True)
        self.log(f"iou/{stage}_mIoU", label_ious.mean(), sync_dist=True)

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> Dict:
        images, targets = batch
        predictions = self.forward(images)
        targets = targets.squeeze()
        return {"test_iou": self._eval_metric(predictions, targets)}

    def test_epoch_end(self, outputs: List[Dict]) -> Dict:
        avg_IoU = np.mean([x["test_iou"] for x in outputs])
        logger.info("Finished testing with mIoU: {}%".format(avg_IoU * 100))
        return {"avg_IoU": avg_IoU}
