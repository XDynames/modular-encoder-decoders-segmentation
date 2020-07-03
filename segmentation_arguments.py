''' 
	Argument Parsers for training segmentation models with pytorch lightning  
	- get_training_argument
	- get_testing_arguments
'''
from typing import *
import argparse

def get_training_arguments() -> argparse.Namespace:
	parser = argparse.ArgumentParser()
	# Dataset and loading settings
	parser.add_argument('--dataset_name', type=str,
							help='Datatset name, see top of datasets.py for valid options')
	parser.add_argument('--dataset_dir', type=str,
							help='Root folder where images are stored')
	parser.add_argument('--batch_size', type=int,
							help='Batch size of samples durring training')
	parser.add_argument('--val_batch_size', type=int,
							help='Batch size of samples durring validation')
	parser.add_argument('--val_interval', type=int,
							help='Run Validation loop every val_interval training epochs')
	# Model and training settings
	parser.add_argument('--encoder', type=str,
							help='Backbone encoder network architecture to use')
	parser.add_argument('--decoder', type=str,
							help='Decoder network architecture to use')
	parser.add_argument('--output_stride', type=int, default=32,
							help="Output stride for encoder backbone in deeplabv3+")
	parser.add_argument('--atrous_rates', type=lambda x:[int(y) for y in x.split(',')],
							default='6,12,18',
							help='Dilation rates used in ASPP layer for deeplabv3+')
	parser.add_argument('--imagenet', type=str, default='False',
							help="Intiialises model wieghts with imagenet pretrain")
	parser.add_argument('--momentum', type=float, default=0.0,
							help='Momentum parameter for optimiser')
	parser.add_argument('--decay' , type=float, default=0.0,
							help='Weight decay parameter')
	parser.add_argument('--lr', type=float,
							help='Initial Learning rate')
	parser.add_argument('--num_epochs', type=int,
							help='Number of epochs to train for')
	parser.add_argument('--amp_level', type=str, default='O0',
							help='Automatic mix precision level')
	parser.add_argument("--accumulate_grad_batches", type=int, default=1,
							help='Number of batches to accumulate gradients over')	
	# Resourcing
	parser.add_argument('--gpus', type=str, default="",
							help='Which GPUs to use')
	parser.add_argument('--gradient_ckpt', type=str, default='False',
							help="Gradient checkpoints encoder modules")
	# Resume from checkpoint
	parser.add_argument('--ckpt_filename', type=str, default="None",
							help='Path to model wieghts file')
	return parser.parse_args()

def get_testing_arguments() -> argparse.Namespace:
	parser = argparse.ArgumentParser()
	# Dataset and loading settings
	parser.add_argument('--dataset_name', type=str,
							help='Datatset name, see top of datasets.py for valid options')
	parser.add_argument('--dataset_dir', type=str,
							help='Root folder where images are stored')
	parser.add_argument('--val_batch_size', type=int,
							help='Batch size of samples durring validation')
	# Model and training settings
	parser.add_argument('--model', type=str,
							help='Network architecture to train')
	parser.add_argument('--amp_level', type=str, default='O0',
							help='Automatic mix precision level')
	# Resourcing
	parser.add_argument('--gpus', type=str, default="",
							help='GPUs to use')
	# Resume from checkpoint
	parser.add_argument('--ckpt_filename', type=str, default="None",
							help='Path to model wieghts file')
	return parser.parse_args()