from collections import defaultdict
from typing import Dict, List

import pytorch_lightning as pl
import torch
import segmentation_models_pytorch as smp
from torch import nn
from torchmetrics.classification import MulticlassJaccardIndex

from datasets import segmentation

SEGMENTATION_MODELS = {
    "deeplab": smp.DeepLabV3Plus,
    "fpn": smp.FPN,
}


class Segmentor(pl.LightningModule):
    def build_model(self) -> nn.Module:
        decoder = SEGMENTATION_MODELS[self._decoder_name](
            encoder_name=self._encoder_name,
            encoder_weights="imagenet"
            if self._use_imagenet_pretrain
            else None,
            classes=self._n_classes,
        )
        return decoder

    def __init__(
        self,
        encoder_name: str,
        decoder_name: str,
        dataset_name: str,
    ):
        super().__init__()
        self._dataset_name = dataset_name
        self._extract_dataset_information()
        self._model = self.build_model()
        self._loss = nn.CrossEntropyLoss()
        self._setup_evaluation_metric()
        self._setup_class_labels()
        self.validation_outputs = defaultdict(list)
        self.test_outputs = defaultdict(list)

    def _extract_dataset_information(self):
        db_info = segmentation.getInformation(self._dataset_name)["n_classes"]
        self._n_classes = db_info["n_classes"]
        self._ignore_index = db_info["ignore_index"]
        self._class_labels = db_info["class_labels"]

    def _setup_evaluation_metric(self):
        self._eval_metric = MulticlassJaccardIndex(
            self._n_classes,
            average=None,
            ignore_index=self.ignore_index,
        )

    def _setup_class_labels(self):
        if self._class_labels is None:
            class_labels = range(0, self._n_classes)
        else:
            class_labels = self._get_class_names()
        self.class_labels = class_labels

    def _get_class_names(self) -> List[str]:
        class_labels = sorted(self._class_labels, key=lambda x: x.train_id)
        return [class_info.name for class_info in class_labels]

    def training_step(
        self,
        batch: torch.Tensor,
        batch_idx: int,
        optimizer_idx: int = 0,
    ) -> Dict:
        images, targets = batch
        predictions = self._model(images)
        targets = targets.squeeze(1)
        loss = self._loss(predictions, targets)
        batch_ious = self._eval_metric(predictions, targets)
        self.log("loss/train", loss, sync_dist=True)
        self.log_iou("train", batch_ious)
        return {"loss": loss}

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        images, targets = batch
        predictions = self.forward(images)
        targets = targets.squeeze(1)
        loss = self._loss(predictions, targets)
        iou = self._eval_metric(predictions, targets)
        self.validation_outputs["loss"].append(loss)
        self.validation_outputs["iou"].append(iou)

    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.validation_outputs["loss"]).mean()
        avg_ious = torch.stack(self.validation_outputs["iou"]).mean(dim=0)
        self.log("loss/val", avg_loss, sync_dist=True)
        self.log_iou("val", avg_ious)
        self.validation_outputs.clear()

    def log_iou(self, stage: str, label_ious: torch.Tensor):
        for iou, label in zip(label_ious, self.class_labels):
            self.log(f"iou/{stage}_{label}", iou, sync_dist=True)
        self.log(f"iou/{stage}_mIoU", label_ious.mean(), sync_dist=True)

    def test_step(self, batch: torch.Tensor, batch_idx: int):
        images, targets = batch
        predictions = self.forward(images)
        targets = targets.squeeze()
        iou = self._eval_metric(predictions, targets)
        self.test_outputs["iou"].append(iou)

    def on_test_epoch_end(self):
        avg_ious = torch.stack(self.test_outputs["iou"]).mean(dim=0)
        self.log_iou("test", avg_ious)
