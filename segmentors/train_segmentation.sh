export "WANDB_API_KEY"="e8d24680a3708a131030ff9268996add924a526d"
python /workspace/segmentors/train_segmentation.py \
	--project-name monza-track \
	--run-name fpn-resent-18-v3 \
	--dataset-name monza \
	--dataset-dir /mnt/data/segmentation/monza/v2 \
	--decoder fpn \
	--encoder resnet_18 \
	--imagenet \
	--batch-size 64 \
	--val-batch-size 32 \
	--val-interval 1 \
	--lr 0.03 \
	--step-lr-every-n-steps 100 \
	--lr-step-factor 0.1 \
	--decay 1e-5 \
	--momentum 0.9 \
	--n-epochs 300 \
	--precision 16 \
	--gpus "-1"
