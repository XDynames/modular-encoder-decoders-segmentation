from pathlib import Path
from PIL import Image

import torch
import lightning.pytorch as pl
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from src.datasets.load import load_game_state


class ACDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_directory: str,
        batch_size: int,
        n_workers: int,
    ):
        super().__init__()
        self._data_directory = data_directory
        self._batch_size = batch_size
        self._n_workers = n_workers

    def train_dataloader(self):
        training_set = ACCaptureDataset(self._data_directory, "train")
        return self._get_data_loader(training_set)

    def val_dataloader(self):
        val_set = ACCaptureDataset(self._data_directory, "val")
        return self._get_data_loader(val_set)

    def _get_data_loader(self, dataset: Dataset) -> DataLoader:
        data_loader = DataLoader(
            dataset,
            batch_size=self._batch_size,
            shuffle=True,
            n_workers=self._n_workers,
            pin_memory=True,
        )
        return data_loader


class ACCaptureDataset(Dataset):
    def __init__(self, data_directory: str, split: str):
        super().__init__()
        self._data_directory = Path(data_directory).joinpath(split)
        self._setup_sample_paths()
        self._setup_transforms()

    def _setup_sample_paths(self):
        sample_filepaths = self._data_directory.glob("*/*.bin")
        self.sample_paths = [filepath for filepath in sample_filepaths]

    def _setup_transforms(self):
        self._transform = transforms.Compose(
            [
                transforms.Resize(1280, 720),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.sample_paths)

    def __getitem__(self, idx: int):
        sample_path = self.sample_paths[idx]
        state = self._load_state(sample_path)
        image = self._load_frame(sample_path)
        return image, state

    def _load_state(self, sample_path: Path) -> torch.Tensor:
        state = load_game_state(sample_path)
        action = [state["throttle"], state["brake"], state["steering"]]
        return torch.Tensor(action)

    def _load_frame(self, state_path: Path) -> torch.Tensor:
        image = Image.open(state_path.with_suffix(".jpeg"))
        return self._transform(image)
