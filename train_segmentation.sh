#!bin/bash

# Script for running network training
CUDA_VISIBLE_DEVICES="1" python train_segmentation.py \
	--dataset_name camvid \
	--dataset_dir ../seg_datasets/CamVid/ \
	--decoder refinenet \
	--encoder resnet_18 \
	--imagenet \
	--batch_size 24 \
	--val_batch_size  32 \
	--val_interval 10 \
	--lr 0.003 \
	--decay 1e-5 \
	--momentum 0.9 \
	--num_epochs 300 \
	--amp_level O2 \
	--gradient_ckpt False \
	--gpus "0"

CUDA_VISIBLE_DEVICES="1" python train_segmentation.py \
	--dataset_name camvid \
	--dataset_dir ../seg_datasets/CamVid/ \
	--decoder deeplab_v3+ \
	--atrous_rates 6,12,18 \
	--output_stride 16 \
	--encoder resnet_18 \
	--imagenet \
	--batch_size 24 \
	--val_batch_size  32 \
	--val_interval 10 \
	--lr 0.003 \
	--decay 1e-5 \
	--momentum 0.9 \
	--num_epochs 300 \
	--amp_level O2 \
	--gradient_ckpt False \
	--gpus "0"