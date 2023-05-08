import argparse

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from datasets import segmentation
from datasets.augment import ReSizeNormalise


class SegmentationDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        dataset_path: str,
        dataset_name: str,
        n_workers: int,
    ):
        super().__init__()
        self._dataset_path = dataset_path
        self._dataset_name = dataset_name
        self._n_workers = n_workers
        self._batch_size = batch_size

    def train_dataloader(self) -> DataLoader:
        train_dataset = segmentation.build_dataset(
            self._dataset_path,
            self._dataset_name,
            "train",
            transform=ReSizeNormalise(self._dataset_name),
        )
        return DataLoader(
            train_dataset,
            batch_size=self._batch_size,
            num_workers=self._n_workers,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        val_dataset = segmentation.build_dataset(
            self._dataset_path,
            self._dataset_name,
            "val",
            ReSizeNormalise(self._dataset_name),
        )
        return DataLoader(
            val_dataset,
            batch_size=self._batch_size,
            num_workers=self._n_workers,
        )

    def test_dataloader(self) -> DataLoader:
        val_dataset = segmentation.build_dataset(
            self._dataset_path,
            self._dataset_name,
            "val",
            ReSizeNormalise(self._dataset_name),
        )
        return DataLoader(
            val_dataset,
            batch_size=self._batch_size,
            num_workers=self._n_workers,
        )
