import torch
from lightning.pytorch.cli import LightningCLI, ReduceLROnPlateau

from src.trainer import ImitiationDriver
from src.datasets.data import ACDataModule

torch.set_float32_matmul_precision("medium")


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_optimizer_args(torch.optim.Adam)
        parser.add_lr_scheduler_args(ReduceLROnPlateau)


def cli_main():
    cli = MyLightningCLI(
        model_class=ImitiationDriver,
        datamodule_class=ACDataModule,
        save_config_kwargs={"overwrite": True},
        seed_everything_default=1337,
    )


if __name__ == "__main__":
    cli_main()
