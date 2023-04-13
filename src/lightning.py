
import pytorch_lightning as pl
from src.models.resnet import get_modified_resent

class ImitiationDriver(pl.LightningModules):
    def __init__(self):
        super().__init__()
    
    def training_step(self, batch: torch.tensor, batch_idx: int):
        pass

    def validation_step(self, batch: torch.tensor, batch_idx: int):
        pass

    def configure_optimizers(self):
        pass