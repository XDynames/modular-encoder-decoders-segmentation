import pytorch_lightning as pl
from torch.utils.data import DataLoader

from datasets import segmentation
from datasets.augment import ReSize_Normalise


class SegmentationDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.hparams = args

    def train_dataloader(self) -> DataLoader:
        hparams = self.hparams
        train_dataset = segmentation.build_dataset(
            hparams, transform=ReSize_Normalise(hparams)
        )
        return DataLoader(
            train_dataset,
            batch_size=hparams.batch_size,
            num_workers=hparams.n_workers,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        hparams = self.hparams
        val_dataset = segmentation.build_dataset(
            hparams, "val", ReSize_Normalise(hparams)
        )
        return DataLoader(
            val_dataset,
            batch_size=hparams.val_batch_size,
            num_workers=hparams.n_workers,
        )

    def test_dataloader(self) -> DataLoader:
        hparams = self.hparams
        val_dataset = segmentation.build_dataset(
            hparams, "val", ReSize_Normalise(hparams)
        )
        return DataLoader(
            val_dataset,
            batch_size=hparams.val_batch_size,
            num_workers=hparams.n_workers,
        )
