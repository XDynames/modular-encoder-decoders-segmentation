docker run \
    --rm \
    --shm-size 80G \
    --gpus "all" \
    -v /mnt/data/:/data/ \
    -v /home/james/Documents/lightning-segmentation/configs/:/configs/ \
    lightning-segmentation \
    python ./src/main.py fit --config /configs/train.yaml
