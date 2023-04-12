import pytorch_lightning as pl
from torch.utils.data import Dataset

class ACDataModule(pl.LightningDataModule):
    def setup(self):
        pass

class ACCaptureDataset(Dataset):
    def __init__(self):
        super().__init__()
        