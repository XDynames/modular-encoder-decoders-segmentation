import torch
from lightning.pytorch.cli import LightningCLI, ReduceLROnPlateau

from src.trainer import ImitiationDriver
from src.datasets.data import ACDataModule


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_optimizer_args(torch.optim.Adam)
        parser.add_lr_scheduler_args(ReduceLROnPlateau)


def cli_main():
    cli = MyLightningCLI(ImitiationDriver, ACDataModule)


if __name__ == "__main__":
    cli_main()
