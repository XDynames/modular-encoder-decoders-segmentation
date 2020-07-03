import os
import argparse
from typing import *

from torch.utils.data import DataLoader
from pytorch_lightning import metrics
import pytorch_lightning as pl
from torch import nn
import torchvision
import numpy as np
import torch

from datasets.augment import ReSize_Normalise
from models import refinenet, deeplabv3plus
from models import resnet, mobilenet
from datasets import segmentation

decoder_types = {
    'deeplab_v3+': deeplabv3plus.DeepLabV3plus,
    'refinenet': refinenet.RefineNetLW,
}
encoder_types = {
    'resnet' : resnet.ResnetEncoder,
    'mobilenet' : mobilenet.MobilenetV2Encoder,
}
# Add your own encoders implmentations here
encoder_variants = {
    'resnet_18': torchvision.models.resnet18,
    'resnet_34'  : torchvision.models.resnet34,
    'resnet_50'  : torchvision.models.resnet50,
    'resnet_101' : torchvision.models.resnet101,
    'resnet_152' : torchvision.models.resnet152,
    'mobilenet_v2' : torchvision.models.mobilenet_v2,
}

class SegmentationTrainer(pl.LightningModule):
    # TODO: Abstract this into a decoder factory class
    def build_model(self) -> nn.Module:
        hparams = self.hparams
        db_info =  segmentation.getInformation(hparams.dataset_name)
        n_classes =  db_info['num_classes']
        
        encoder_variant = hparams.encoder
        encoder_type  = hparams.encoder.split('_')[0]

        backbone = encoder_variants[encoder_variant](hparams.imagenet)
        encoder = encoder_types[encoder_type](backbone, hparams.output_stride)
        decoder = decoder_types[hparams.decoder](
            encoder, n_classes,
            atrous_rates=hparams.atrous_rates,
            interpolation_mode='bilinear',
            classification_head=None,
            verbose_sizes=False
        )
        return decoder

    # TODO: Add options for different optimisers and schedulers
    def configure_optimizers(self) -> torch.optim.Optimizer:
        hparams = self.hparams

        optimiser = torch.optim.SGD(
            self._model.parameters(),
            lr=hparams.lr,
            weight_decay=hparams.decay,
            momentum=hparams.momentum
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
        self._eval_metric = metrics.classification.IoU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._model(x, self._grad_ckpt)

    def training_step(
        self,
        batch: torch.Tensor,
        batch_idx: int,
        optimizer_idx: int=0
    ) -> Dict:
        images, targets = batch
        targets = targets.squeeze(1)

        predictions = self.forward(images)

        loss = self._loss(predictions, targets)
        batch_IoU = self._eval_metric(predictions, targets)
        tensorboard_logs = {'train_loss': loss,
                            'train_IoU':batch_IoU }
        return { 'loss': loss, 'log' : tensorboard_logs }

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> Dict:
        images, targets = batch
        predictions = self.forward(images)
        targets = targets.squeeze(1)

        return {'val_loss': self._loss(predictions, targets), 
                'val_IoU': self._eval_metric(predictions, targets) }

    def validation_epoch_end(self, outputs: List[Dict]) -> Dict:
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_IoU = np.mean([x['val_IoU'] for x in outputs])

        tensorboard_logs = {'val_loss': avg_loss, 'val_IoU': avg_IoU}

        return {'avg_val_loss': avg_loss,
                'avg_IoU': avg_IoU,
                'log': tensorboard_logs}

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> Dict:
        images, targets = batch
        predictions = self.forward(images)
        targets = targets.squeeze()
        return { 'val_IoU' : self._eval_metric(predictions, targets) }

    def test_epoch_end(self, outputs: List[Dict]) -> Dict:
        avg_IoU = np.mean([x['val_IoU'] for x in outputs])
        print('Finished testing with mIoU: {}%'.format(avg_IoU * 100))
        return { 'avg_IoU' : avg_IoU }
    
    def train_dataloader(self) -> DataLoader:
        hparams = self.hparams
        train_dataset = segmentation.build_dataset(hparams, transform=ReSize_Normalise(hparams))
        return DataLoader(train_dataset, batch_size=hparams.batch_size, num_workers=4,
                                                        shuffle=True, drop_last=True)

    def val_dataloader(self) -> DataLoader:
        hparams = self.hparams
        val_dataset = segmentation.build_dataset(hparams, 'val', ReSize_Normalise(hparams))        
        return DataLoader(val_dataset, batch_size=hparams.val_batch_size, num_workers=4)

    def test_dataloader(self) -> DataLoader:
        hparams = self.hparams
        val_dataset = segmentation.build_dataset(hparams, 'val', ReSize_Normalise(hparams))        
        return DataLoader(val_dataset, batch_size=hparams.val_batch_size, num_workers=4)
