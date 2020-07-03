#!bin/bash

# Script for running network training
CUDA_VISIBLE_DEVICES="1" python train_segmentation.py \
	--dataset_name camvid \
	--dataset_dir ../seg_datasets/CamVid/ \
	--decoder deeplab_v3+ \
	--atrous_rates 6,12,18 \
	--output_stride 16 \
	--encoder resnet_18 \
	--imagenet True \
	--batch_size 24 \
	--val_batch_size  32 \
	--lr 0.003 \
	--decay 1e-5 \
	--momentum 0.9 \
	--num_epochs 300 \
	--amp_level O2 \
	--gpus "0"