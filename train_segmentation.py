import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy

from datasets.lightning_data_module import SegmentationDataModule
from segmentation_arguments import get_training_arguments
from segmentation_trainer import SegmentationTrainer

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("medium")


def main():
    args = get_training_arguments()

    wandb_logger = WandbLogger(
        project=args.project_name,
        name=args.run_name,
        log_model=True,
    )
    data_module = SegmentationDataModule(args)
    trainer = Trainer(
        accumulate_grad_batches=args.accumulate_grad_batches,
        check_val_every_n_epoch=args.val_interval,
        max_epochs=args.n_epochs,
        strategy=DDPStrategy(find_unused_parameters=False),
        precision=args.precision,
        devices=args.gpus,
        log_every_n_steps=args.log_every_n_steps,
        logger=wandb_logger,
        accelerator="gpu",
    )
    model = SegmentationTrainer(args)
    trainer.fit(model, data_module)


if __name__ == "__main__":
    main()
