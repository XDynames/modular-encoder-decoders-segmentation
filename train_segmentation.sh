#!bin/bash

# Script for running network training
python train_segmentation.py \
	--dataset_name camvid \
	--dataset_dir ../../datasets/camvid/ \
	--decoder refinenet \
	--encoder resnet_18 \
	--imagenet \
	--batch_size 64 \
	--val_batch_size 32 \
	--val_interval 10 \
	--lr 0.03 \
	--decay 1e-5 \
	--momentum 0.9 \
	--num_epochs 300 \
	--amp_level O2 \
	--gradient_ckpt \
	--gpus "0"
