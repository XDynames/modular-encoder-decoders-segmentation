import argparse


def get_training_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # Dataset and loading settings
    parser.add_argument(
        "--dataset-name",
        type=str,
        help="Dataset name, see top of datasets.py for valid options",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        help="Root folder where images are stored",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size of samples durring training",
    )
    parser.add_argument(
        "--val-batch-size",
        type=int,
        help="Batch size of samples durring validation",
    )
    parser.add_argument(
        "--val-interval",
        type=int,
        help="Run Validation loop every val_interval training epochs",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=4,
        help="Number of worker threads used in dataloader",
    )
    # Model and training settings
    parser.add_argument(
        "--encoder",
        type=str,
        help="Backbone encoder network architecture to use",
    )
    parser.add_argument(
        "--decoder",
        type=str,
        help="Decoder network architecture to use",
    )
    parser.add_argument(
        "--output-stride",
        type=int,
        default=32,
        help="Output stride for encoder backbone in deeplabv3+",
    )
    parser.add_argument(
        "--imagenet",
        action="store_true",
        help="Initialise model weights with imagenet pretrain",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.0,
        help="Momentum parameter for optimiser",
    )
    parser.add_argument(
        "--decay",
        type=float,
        default=0.0,
        help="Weight decay parameter",
    )
    parser.add_argument(
        "--lr",
        type=float,
        help="Initial Learning rate",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        help="Number of epochs to train for",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=32,
        help="Floating point precision",
    )
    parser.add_argument(
        "--accumulate-grad-batches",
        type=int,
        default=1,
        help="Number of batches to accumulate gradients over",
    )
    parser.add_argument(
        "--step-lr-every-n-steps",
        type=int,
        default=150,
        help="Step the learning rate schedular every n steps",
    )
    parser.add_argument(
        "--lr-step-factor",
        type=float,
        default=1,
        help="Number to multiple the learning rate by when stepped",
    )
    # Resourcing
    parser.add_argument(
        "--gpus",
        type=str,
        default="",
        help="Which GPUs to use",
    )
    parser.add_argument(
        "--gradient-ckpt",
        action="store_true",
        help="Gradient checkpoints encoder modules",
    )
    # Checkpoint settings
    parser.add_argument(
        "--resume-from-ckpt-path",
        type=str,
        default="None",
        help="Path to model weights file",
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default="None",
        help="Path to save model checkpoints to",
    )
    # W&B settings
    parser.add_argument(
        "--entity",
        type=str,
        help="Entity the project to log runs under",
    )
    parser.add_argument("--project-name", type=str, help="Project to log run under")
    parser.add_argument("--run-name", type=str, help="Name used to log the run")
    # Logging
    parser.add_argument(
        "--log-every-n-steps",
        type=int,
        default=10,
        help="Logging interval",
    )
    return parser.parse_args()


def get_testing_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # Dataset and loading settings
    parser.add_argument(
        "--dataset-name",
        type=str,
        help="Dataset name, see top of datasets.py for valid options",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        help="Root folder where images are stored",
    )
    parser.add_argument(
        "--val-batch-size",
        type=int,
        help="Batch size of samples durring validation",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=4,
        help="Number of worker threads used in dataloader",
    )
    # Model and training settings
    parser.add_argument("--model", type=str, help="Network architecture to train")
    parser.add_argument(
        "--precision",
        type=int,
        default=32,
        help="Floating point precision",
    )
    # Resourcing
    parser.add_argument(
        "--gpus",
        type=str,
        default="",
        help="GPUs to use",
    )
    # Resume from checkpoint
    parser.add_argument(
        "--ckpt-filename",
        type=str,
        default="None",
        help="Path to model weights file",
    )
    return parser.parse_args()
