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
        n_stacked_frames: int,
        n_skip_frames: int,
        batch_size: int,
        n_workers: int,
    ):
        super().__init__()
        self._data_directory = data_directory
        self._batch_size = batch_size
        self._n_workers = n_workers
        self._n_stacked_frames = n_stacked_frames
        self._n_skip_frames = n_skip_frames

    def train_dataloader(self, shuffle: bool = True):
        training_set = ACCaptureDataset(
            self._data_directory,
            "train",
            self._n_stacked_frames,
            self._n_skip_frames,
        )
        return self._get_data_loader(training_set, shuffle)

    def val_dataloader(self):
        val_set = ACCaptureDataset(
            self._data_directory,
            "val",
            self._n_stacked_frames,
            self._n_skip_frames,
        )
        return self._get_data_loader(val_set, False)

    def _get_data_loader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        data_loader = DataLoader(
            dataset,
            batch_size=self._batch_size,
            shuffle=shuffle,
            num_workers=self._n_workers,
            pin_memory=True,
        )
        return data_loader


class ACCaptureDataset(Dataset):
    def __init__(
        self,
        data_directory: str,
        split: str,
        n_stacked_frames: int,
        n_skip_frames: int,
    ):
        super().__init__()
        self._data_directory = Path(data_directory).joinpath(split)
        self._setup_sample_paths()
        self._setup_transforms()
        self._n_stacked_frames = n_stacked_frames
        self._n_skip_frames = n_skip_frames
        self._window_size = n_stacked_frames * n_skip_frames

    def _setup_sample_paths(self):
        sample_filepaths = self._data_directory.glob("*.bin")
        self.sample_paths = sorted(sample_filepaths, key=lambda x: int(x.stem))

    def _setup_transforms(self):
        self._transform = transforms.Compose(
            [
                transforms.Resize((1280, 720)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        first_sample_idx = self._n_stacked_frames * self._n_skip_frames
        return len(self.sample_paths) - first_sample_idx + 1

    def __getitem__(self, idx: int):
        images = []
        idx += self._window_size - 1
        state = self._load_state(self.sample_paths[idx])
        for i in range(0, self._window_size, self._n_skip_frames + 1):
            sample_path = self.sample_paths[idx - i]
            images.append(self._load_frame(sample_path))
        return torch.cat(images), state

    def _load_state(self, sample_path: Path) -> torch.Tensor:
        state = load_game_state(sample_path)
        action = [state["throttle"], state["brake"], state["steering_angle"]]
        return torch.Tensor(action)

    def _load_frame(self, state_path: Path) -> torch.Tensor:
        image = Image.open(state_path.with_suffix(".jpeg"))
        return self._transform(image)
