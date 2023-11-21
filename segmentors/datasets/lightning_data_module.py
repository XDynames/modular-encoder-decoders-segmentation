import argparse

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from datasets import segmentation
from datasets.augment import ReSizeNormalise


class SegmentationDataModule(pl.LightningDataModule):
    def __init__(self, args: argparse.ArgumentParser):
        super().__init__()
        self.save_hyperparameters()

    def train_dataloader(self) -> DataLoader:
        hparams = self.hparams["args"]
        train_dataset = segmentation.build_dataset(
            hparams, transform=ReSizeNormalise(hparams)
        )
        return DataLoader(
            train_dataset,
            batch_size=hparams.batch_size,
            num_workers=hparams.n_workers,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        hparams = self.hparams["args"]
        val_dataset = segmentation.build_dataset(
            hparams, "val", ReSizeNormalise(hparams)
        )
        return DataLoader(
            val_dataset,
            batch_size=hparams.val_batch_size,
            num_workers=hparams.n_workers,
        )

    def test_dataloader(self) -> DataLoader:
        hparams = self.hparams["args"]
        val_dataset = segmentation.build_dataset(
            hparams, "val", ReSizeNormalise(hparams)
        )
        return DataLoader(
            val_dataset,
            batch_size=hparams.val_batch_size,
            num_workers=hparams.n_workers,
        )
